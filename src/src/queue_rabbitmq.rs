//! RabbitMQ async job consumer (feature `rabbitmq`, spec §8 "Scenario S3"
//! extended).
//!
//! Manual-ack AMQP consumer: each `QueueJobMessage` is driven through the
//! shared `S3ZipWorker` (same `ZipProcessor`, audit log and webhook as the
//! S3 path). Success → ack. Failure → the message is republished to a **retry
//! queue** with a per-message TTL (exponential backoff, capped) and the
//! original delivery is acked; when the TTL expires the retry queue
//! dead-letters the message back onto the work queue. After
//! `RABBITMQ_MAX_RETRIES` attempts the message is published to the **DLQ**
//! and the failure logged — poison messages never loop forever.
//!
//! Topology (declared at startup, idempotent):
//! - work queue  `anonimizzazione-jobs` (durable)
//! - retry queue `anonimizzazione-jobs-retry` (durable) → dead-letters back
//!   to the work queue via the default exchange
//! - DLQ         `anonimizzazione-jobs-dlq` (durable)
//!
//! Job concurrency is bounded by `RABBITMQ_PREFETCH` (max unacked deliveries)
//! and inside each job by the shared S3 job semaphore (`S3_MAX_CONCURRENT_JOBS`)
//! and the per-image semaphore.

use std::sync::Arc;

use anyhow::{Context, Result};
use futures_util::StreamExt;
use lapin::options::{
    BasicAckOptions, BasicConsumeOptions, BasicPublishOptions, BasicQosOptions,
    QueueDeclareOptions,
};
use lapin::types::{AMQPValue, FieldTable, LongString, ShortString};
use lapin::{BasicProperties, Channel, Connection, ConnectionProperties};

use crate::config::RabbitMqSettings;
use crate::queue_metrics::QueueMetrics;
use crate::zip_worker_s3::{QueueJobMessage, S3ZipWorker};

/// Runs the consumer forever. Declares the queue topology, then consumes with
/// manual acks. Returns only on an irrecoverable connection/channel error.
pub async fn run_consumer(
    worker: S3ZipWorker,
    settings: Arc<RabbitMqSettings>,
    metrics: Arc<QueueMetrics>,
) -> Result<()> {
    let conn = Connection::connect(
        &settings.url,
        ConnectionProperties::default().with_connection_name("anonimizzazione-consumer".into()),
    )
    .await
    .with_context(|| format!("connect to RabbitMQ at {}", settings.url))?;
    let channel = conn.create_channel().await.context("create AMQP channel")?;
    channel
        .basic_qos(settings.prefetch, BasicQosOptions::default())
        .await
        .context("set prefetch")?;

    let retry_queue = format!("{}-retry", settings.queue);
    declare_topology(&channel, &settings, &retry_queue).await?;

    tracing::info!(
        "RabbitMQ consumer ready: url={} queue={} retry={} dlq={} (prefetch {}, max {} retries)",
        settings.url,
        settings.queue,
        retry_queue,
        settings.dlq,
        settings.prefetch,
        settings.max_retries,
    );

    let mut consumer = channel
        .basic_consume(
            settings.queue.clone().into(),
            "anonimizzazione-consumer".into(),
            BasicConsumeOptions::default(),
            FieldTable::default(),
        )
        .await
        .context("basic_consume")?;

    while let Some(delivery) = consumer.next().await {
        let delivery = match delivery {
            Ok(d) => d,
            Err(e) => {
                tracing::warn!("RabbitMQ delivery error: {e}; continuing");
                continue;
            }
        };
        // Each delivery runs in its own task; prefetch bounds how many are
        // in flight at once, the shared job semaphore bounds real work.
        metrics
            .received
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let worker = worker.clone();
        let channel = channel.clone();
        let settings = settings.clone();
        let retry_queue = retry_queue.clone();
        let metrics = metrics.clone();
        tokio::spawn(async move {
            if let Err(e) = handle_delivery(
                &worker,
                &channel,
                &settings,
                &retry_queue,
                &metrics,
                delivery,
            )
            .await
            {
                tracing::warn!("RabbitMQ message handler failed: {e:#}");
            }
        });
    }
    Ok(())
}

/// Declares work queue, retry queue (dead-letters back to the work queue) and
/// the DLQ. Idempotent — re-running with the same names is a no-op.
async fn declare_topology(
    channel: &Channel,
    settings: &RabbitMqSettings,
    retry_queue: &str,
) -> Result<()> {
    // Work queue: plain durable queue.
    channel
        .queue_declare(
            settings.queue.clone().into(),
            QueueDeclareOptions::durable(),
            FieldTable::default(),
        )
        .await
        .with_context(|| format!("declare work queue {}", settings.queue))?;

    // Retry queue: after its per-message TTL expires, RabbitMQ dead-letters
    // the message to the default exchange with routing key = work queue name,
    // so it lands back on the work queue with an incremented x-death count.
    let mut retry_args = FieldTable::default();
    retry_args.insert(
        ShortString::from("x-dead-letter-exchange"),
        AMQPValue::LongString(LongString::from("")), // default exchange
    );
    retry_args.insert(
        ShortString::from("x-dead-letter-routing-key"),
        AMQPValue::LongString(LongString::from(settings.queue.clone())),
    );
    channel
        .queue_declare(
            retry_queue.into(),
            QueueDeclareOptions::durable(),
            retry_args,
        )
        .await
        .with_context(|| format!("declare retry queue {retry_queue}"))?;

    channel
        .queue_declare(
            settings.dlq.clone().into(),
            QueueDeclareOptions::durable(),
            FieldTable::default(),
        )
        .await
        .with_context(|| format!("declare DLQ {}", settings.dlq))?;
    Ok(())
}

/// One delivery: parse → run → ack / retry-queue / DLQ.
async fn handle_delivery(
    worker: &S3ZipWorker,
    channel: &Channel,
    settings: &RabbitMqSettings,
    retry_queue: &str,
    metrics: &Arc<QueueMetrics>,
    delivery: lapin::message::Delivery,
) -> Result<()> {
    let job_id = match delivery.properties.message_id() {
        Some(id) => id.to_string(),
        None => uuid::Uuid::new_v4().to_string(),
    };

    let payload: QueueJobMessage = match serde_json::from_slice(&delivery.data) {
        Ok(p) => p,
        Err(e) => {
            // Poison message: straight to the DLQ, then ack the original.
            tracing::error!("RabbitMQ message {job_id}: unparseable body ({e}), sending to DLQ");
            metrics.dlq.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            publish(
                channel,
                &settings.dlq,
                &delivery.data,
                delivery.properties.clone(),
            )
            .await?;
            delivery
                .ack(BasicAckOptions::default())
                .await
                .context("ack poison message")?;
            return Ok(());
        }
    };

    let attempts = x_death_attempts(&delivery.properties);
    match worker.run_queue_job(&payload, &job_id).await {
        Ok(()) => {
            tracing::info!("RabbitMQ job {job_id} completed, acking");
            metrics
                .completed
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            delivery
                .ack(BasicAckOptions::default())
                .await
                .context("ack completed job")?;
        }
        Err(e) => {
            metrics.failed.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if attempts + 1 >= settings.max_retries {
                tracing::error!(
                    "RabbitMQ job {job_id} failed after {} attempts, sending to DLQ: {e:#}",
                    attempts + 1
                );
                metrics.dlq.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                publish(
                    channel,
                    &settings.dlq,
                    &delivery.data,
                    delivery.properties.clone(),
                )
                .await?;
            } else {
                // Retry with exponential backoff: republish to the retry queue
                // with a per-message TTL, then ack the original delivery.
                let backoff_ms = backoff_ms(settings, attempts);
                tracing::warn!(
                    "RabbitMQ job {job_id} failed (attempt {}/{}), retrying in {backoff_ms} ms: {e:#}",
                    attempts + 1,
                    settings.max_retries,
                );
                let props = delivery
                    .properties
                    .clone()
                    .with_expiration(ShortString::from(backoff_ms.to_string()));
                publish(channel, retry_queue, &delivery.data, props).await?;
            }
            delivery
                .ack(BasicAckOptions::default())
                .await
                .context("ack after republish")?;
        }
    }
    Ok(())
}

/// Exponential backoff: `base * 2^attempts`, capped at `max_backoff_secs`.
fn backoff_ms(settings: &RabbitMqSettings, attempts: u32) -> u64 {
    let base = settings.retry_backoff_secs.max(1);
    let cap = settings.max_backoff_secs.max(1);
    let exp = base.saturating_mul(1u64 << attempts.min(20));
    (exp.min(cap)) * 1000
}

/// Sum of the `count` fields across the message's `x-death` headers — the
/// broker increments this every time the message is dead-lettered (each retry
/// cycle = one trip through the retry queue).
fn x_death_attempts(props: &BasicProperties) -> u32 {
    let Some(headers) = props.headers() else {
        return 0;
    };
    let Some(value) = headers.inner().get(&ShortString::from("x-death")) else {
        return 0;
    };
    let Some(entries) = value.as_array() else {
        return 0;
    };
    let mut total: u64 = 0;
    for entry in entries.as_slice() {
        let Some(table) = entry.as_field_table() else {
            continue;
        };
        let Some(count) = table.inner().get(&ShortString::from("count")) else {
            continue;
        };
        if let Some(n) = count.as_long_long_int() {
            total += n as u64;
        } else if let Some(n) = count.as_long_int() {
            total += n as u64;
        } else if let Some(n) = count.as_short_int() {
            total += n as u64;
        } else if let Some(n) = count.as_short_short_int() {
            total += n as u64;
        }
    }
    total as u32
}

/// Publishes `body` to the default exchange with `routing_key` = queue name.
async fn publish(channel: &Channel, queue: &str, body: &[u8], props: BasicProperties) -> Result<()> {
    channel
        .basic_publish(
            "".into(),
            queue.into(),
            BasicPublishOptions::default(),
            body,
            props,
        )
        .await
        .context("basic_publish")?
        .await
        .context("publisher confirm")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn settings() -> RabbitMqSettings {
        RabbitMqSettings {
            url: "amqp://127.0.0.1:5672".into(),
            queue: "jobs".into(),
            dlq: "jobs-dlq".into(),
            prefetch: 4,
            max_retries: 3,
            retry_backoff_secs: 5,
            max_backoff_secs: 300,
        }
    }

    #[test]
    fn backoff_is_exponential_and_capped() {
        let s = settings();
        assert_eq!(backoff_ms(&s, 0), 5_000); // base
        assert_eq!(backoff_ms(&s, 1), 10_000); // ×2
        assert_eq!(backoff_ms(&s, 2), 20_000); // ×4
        // Attempt 6: 5 * 64 = 320 s → capped at max_backoff_secs (300 s).
        assert_eq!(backoff_ms(&s, 6), 300_000);
    }

    #[test]
    fn x_death_attempts_counts_only_numeric_counts() {
        // No headers → 0 attempts.
        assert_eq!(x_death_attempts(&BasicProperties::default()), 0);

        // One dead-letter entry with count=2 (broker increments per cycle).
        let mut entry = FieldTable::default();
        entry.insert(
            ShortString::from("count"),
            AMQPValue::LongLongInt(2),
        );
        entry.insert(
            ShortString::from("reason"),
            AMQPValue::LongString(LongString::from("expired")),
        );
        let mut headers = FieldTable::default();
        headers.insert(
            ShortString::from("x-death"),
            AMQPValue::FieldArray(vec![AMQPValue::FieldTable(entry)].into()),
        );
        let props = BasicProperties::default().with_headers(headers);
        assert_eq!(x_death_attempts(&props), 2);

        // A header that is not x-death is ignored.
        let mut headers = FieldTable::default();
        headers.insert(
            ShortString::from("x-custom"),
            AMQPValue::LongLongInt(99),
        );
        let props = BasicProperties::default().with_headers(headers);
        assert_eq!(x_death_attempts(&props), 0);
    }
}