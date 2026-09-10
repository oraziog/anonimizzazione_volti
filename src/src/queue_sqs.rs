//! SQS async job consumer (feature `queue`, spec §8 "Scenario S3" extended).
//!
//! Long-polls a standard SQS queue for `QueueJobMessage` payloads and drives
//! each through the shared `S3ZipWorker` — the *same* `ZipProcessor`, audit
//! log and completion webhook as `/anonymize/s3`. A successfully processed
//! message is deleted (ack); a failed one is left in place and redelivered
//! after the visibility timeout (SQS retry).
//!
//! **Resilience.** The poll loop never exits: transient SDK/network errors
//! (throttling, 5xx, connection resets) are retried with exponential backoff
//! (`SQS_POLL_INTERVAL_SECS * 2^n`, capped at `SQS_MAX_BACKOFF_SECS`). The
//! client is rebuilt after a long outage (auto-reconnect).
//!
//! **DLQ.** At startup the consumer ensures the queue has a redrive policy
//! pointing at a `{queue}-dlq` dead-letter queue (`maxReceiveCount =
//! SQS_MAX_RECEIVE_ATTEMPTS`), creating the DLQ if missing — so poison
//! messages never loop forever. Counter metrics are published to
//! `GET /operator/queues`.
//!
//! Credentials come from the standard `AWS_*` chain (`SQS_REGION` overrides
//! `AWS_REGION`). Job concurrency is bounded by `S3_MAX_CONCURRENT_JOBS`
//! (the shared job semaphore inside `S3ZipWorker`).

use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use aws_sdk_sqs::types::{MessageSystemAttributeName, QueueAttributeName};
use aws_sdk_sqs::Client as SqsClient;

use crate::config::SqsSettings;
use crate::queue_metrics::QueueMetrics;
use crate::zip_worker_s3::{QueueJobMessage, S3ZipWorker};

/// Infinite SQS consumer loop. Never returns: transient errors are retried
/// with exponential backoff, the client is rebuilt after a prolonged outage.
pub async fn run_consumer(
    worker: S3ZipWorker,
    settings: Arc<SqsSettings>,
    metrics: Arc<QueueMetrics>,
) -> Result<()> {
    let mut backoff_secs = settings.poll_interval_secs.max(1);
    let mut client = build_client(&settings).await?;

    // Best-effort DLQ setup: create the dead-letter queue + redrive policy.
    // A failure here is logged but does not stop the consumer (SQS would
    // still redeliver messages; only the durable dead-lettering is lost).
    let dlq_url = match ensure_redrive_policy(&client, &settings).await {
        Ok(url) => Some(url),
        Err(e) => {
            tracing::warn!(
                "SQS redrive policy setup failed (messages will redeliver without a DLQ): {e:#}"
            );
            None
        }
    };

    tracing::info!(
        "SQS consumer ready: queue={} region={:?} (long-poll {}s, visibility {}s, {} msg/poll, max {} attempts)",
        settings.queue_url,
        settings.region,
        settings.wait_seconds,
        settings.visibility_timeout_seconds,
        settings.max_messages,
        settings.max_receive_attempts,
    );

    // Refresh the DLQ depth roughly every 30 s for `/operator/queues`.
    let mut depth_tick: u64 = 0;
    loop {
        match poll_once(&client, &worker, &settings, &metrics).await {
            Ok(n) => {
                // Success resets the backoff to the configured base.
                backoff_secs = settings.poll_interval_secs.max(1);
                if n == 0 {
                    tokio::time::sleep(Duration::from_secs(backoff_secs)).await;
                }
            }
            Err(e) => {
                tracing::warn!("SQS poll failed (retrying in {backoff_secs}s): {e:#}");
                tokio::time::sleep(Duration::from_secs(backoff_secs)).await;
                backoff_secs = (backoff_secs * 2).min(settings.max_backoff_secs.max(1));
                if backoff_secs >= settings.max_backoff_secs.max(1) {
                    // Rebuild the client: the old one may hold a dead
                    // connection / expired credentials after a long outage.
                    match build_client(&settings).await {
                        Ok(c) => client = c,
                        Err(e) => tracing::warn!("SQS client rebuild failed: {e:#}"),
                    }
                }
            }
        }

        depth_tick += 1;
        if depth_tick.is_multiple_of(10) {
            if let Some(url) = &dlq_url {
                refresh_dlq_depth(&client, url, &metrics).await;
            }
        }
    }
}

/// Best-effort `ApproximateNumberOfMessages` of the DLQ, exposed via
/// `/operator/queues`. Failures are silent (metrics just stay stale).
async fn refresh_dlq_depth(client: &SqsClient, dlq_url: &str, metrics: &Arc<QueueMetrics>) {
    let Ok(resp) = client
        .get_queue_attributes()
        .queue_url(dlq_url)
        .attribute_names(QueueAttributeName::ApproximateNumberOfMessages)
        .send()
        .await
    else {
        return;
    };
    if let Some(depth) = resp
        .attributes
        .and_then(|m| m.get(&QueueAttributeName::ApproximateNumberOfMessages).cloned())
        .and_then(|v| v.parse::<u64>().ok())
    {
        metrics.dlq_depth.store(depth, std::sync::atomic::Ordering::Relaxed);
    }
}

async fn build_client(settings: &SqsSettings) -> Result<SqsClient> {
    let sdk_config = aws_config::load_from_env().await;
    let mut builder = aws_sdk_sqs::config::Builder::from(&sdk_config);
    if let Some(region) = &settings.region {
        builder = builder.region(aws_sdk_sqs::config::Region::new(region.clone()));
    }
    Ok(SqsClient::from_conf(builder.build()))
}

/// Ensures the work queue has a redrive policy pointing at a dead-letter
/// queue, creating the DLQ if it does not exist. The DLQ name is the work
/// queue name + `-dlq`; `maxReceiveCount` comes from `SQS_MAX_RECEIVE_ATTEMPTS`.
/// Returns the DLQ URL on success (used for the depth refresh).
async fn ensure_redrive_policy(client: &SqsClient, settings: &SqsSettings) -> Result<String> {
    let queue_name = queue_name_from_url(&settings.queue_url)
        .context("cannot derive queue name from SQS_QUEUE_URL")?;
    let dlq_name = format!("{queue_name}-dlq");

    // Work queue ARN (verified to exist; only the DLQ arn is used below).
    let _work_arn = client
        .get_queue_attributes()
        .queue_url(&settings.queue_url)
        .attribute_names(QueueAttributeName::QueueArn)
        .send()
        .await
        .context("get work queue ARN")?
        .attributes
        .and_then(|m| m.get(&QueueAttributeName::QueueArn).cloned())
        .context("work queue has no QueueArn attribute")?;

    // DLQ URL: look it up, create it if missing (create_queue is idempotent).
    let dlq_url = match client.get_queue_url().queue_name(&dlq_name).send().await {
        Ok(r) => r.queue_url.context("get_queue_url returned no URL")?,
        Err(_) => client
            .create_queue()
            .queue_name(&dlq_name)
            .send()
            .await
            .context("create DLQ")?
            .queue_url
            .context("create_queue returned no URL")?,
    };
    let dlq_arn = client
        .get_queue_attributes()
        .queue_url(&dlq_url)
        .attribute_names(QueueAttributeName::QueueArn)
        .send()
        .await
        .context("get DLQ ARN")?
        .attributes
        .and_then(|m| m.get(&QueueAttributeName::QueueArn).cloned())
        .context("DLQ has no QueueArn attribute")?;

    // Set the redrive policy on the work queue.
    let policy = serde_json::json!({
        "deadLetterTargetArn": dlq_arn,
        "maxReceiveCount": settings.max_receive_attempts,
    })
    .to_string();
    client
        .set_queue_attributes()
        .queue_url(&settings.queue_url)
        .attributes(QueueAttributeName::RedrivePolicy, policy)
        .send()
        .await
        .context("set redrive policy")?;
    tracing::info!(
        "SQS DLQ ready: {dlq_name} (redrive maxReceiveCount={}, arn={dlq_arn})",
        settings.max_receive_attempts
    );
    Ok(dlq_url)
}

/// Last path segment of the queue URL = queue name (SQS URL format).
fn queue_name_from_url(url: &str) -> Option<String> {
    url.rsplit('/').next().filter(|s| !s.is_empty()).map(String::from)
}

/// One receive + dispatch round. Each message is handled in its own task; the
/// shared job semaphore inside the worker bounds real parallelism.
async fn poll_once(
    client: &SqsClient,
    worker: &S3ZipWorker,
    settings: &SqsSettings,
    metrics: &Arc<QueueMetrics>,
) -> Result<usize> {
    let resp = client
        .receive_message()
        .queue_url(&settings.queue_url)
        .max_number_of_messages(settings.max_messages)
        .wait_time_seconds(settings.wait_seconds)
        .visibility_timeout(settings.visibility_timeout_seconds)
        .message_system_attribute_names(MessageSystemAttributeName::ApproximateReceiveCount)
        .send()
        .await
        .context("SQS receive_message")?;

    let messages = resp.messages.unwrap_or_default();
    let count = messages.len();
    metrics.received.fetch_add(count as u64, std::sync::atomic::Ordering::Relaxed);
    for msg in messages {
        let client = client.clone();
        let worker = worker.clone();
        let settings = settings.clone();
        let metrics = metrics.clone();
        tokio::spawn(async move {
            if let Err(e) = handle_message(&client, &worker, &settings, &metrics, &msg).await {
                tracing::warn!("SQS message handler failed: {e:#}");
            }
        });
    }
    Ok(count)
}

/// Processes one SQS message. Success → delete (ack) + `completed`. Failure →
/// leave unacked (redelivery) + `failed`; after `max_receive_attempts`
/// receives the message is deleted + `dlq`. A malformed payload is a poison
/// message and is dropped immediately (+`dlq`).
async fn handle_message(
    client: &SqsClient,
    worker: &S3ZipWorker,
    settings: &SqsSettings,
    metrics: &Arc<QueueMetrics>,
    msg: &aws_sdk_sqs::types::Message,
) -> Result<()> {
    let Some(receipt) = msg.receipt_handle.clone() else {
        return Ok(()); // nothing to ack (should not happen)
    };
    let job_id = msg
        .message_id()
        .map(|s| s.to_string())
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

    let payload: QueueJobMessage = match msg.body().map(serde_json::from_str) {
        Some(Ok(p)) => p,
        _ => {
            tracing::error!("SQS message {job_id}: unparseable body, deleting (poison)");
            metrics.dlq.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            delete_message(client, &settings.queue_url, &receipt).await;
            return Ok(());
        }
    };

    match worker.run_queue_job(&payload, &job_id).await {
        Ok(()) => {
            tracing::info!("SQS job {job_id} completed, acking");
            metrics.completed.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            delete_message(client, &settings.queue_url, &receipt).await;
        }
        Err(e) => {
            let attempts = receive_count(msg);
            metrics.failed.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            tracing::warn!(
                "SQS job {job_id} failed (attempt {attempts}/{}): {e:#}",
                settings.max_receive_attempts
            );
            if attempts >= settings.max_receive_attempts {
                // Give up: delete so the queue does not redeliver forever.
                tracing::error!("SQS job {job_id} exceeded max attempts, dropping message");
                metrics.dlq.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                delete_message(client, &settings.queue_url, &receipt).await;
            }
            // Otherwise leave unacked → visible again after the timeout.
        }
    }
    Ok(())
}

/// `ApproximateReceiveCount` of the message, defaulting to 1 (this delivery).
fn receive_count(msg: &aws_sdk_sqs::types::Message) -> i32 {
    msg.attributes
        .as_ref()
        .and_then(|m| m.get(&MessageSystemAttributeName::ApproximateReceiveCount))
        .and_then(|v| v.parse::<i32>().ok())
        .unwrap_or(1)
}

async fn delete_message(client: &SqsClient, queue_url: &str, receipt: &str) {
    if let Err(e) = client
        .delete_message()
        .queue_url(queue_url)
        .receipt_handle(receipt)
        .send()
        .await
    {
        tracing::warn!("SQS delete_message failed: {e}");
    }
}