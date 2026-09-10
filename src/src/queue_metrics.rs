//! Live metrics of the async queue consumers (`GET /operator/queues`).
//!
//! Each consumer (SQS feature `queue`, RabbitMQ feature `rabbitmq`) registers
//! a `QueueMetrics` entry and bumps counters as it works; the operator
//! endpoint reads the registry. Counters are atomics so the hot paths never
//! take a lock; the registry itself is a cheap `Mutex<HashMap>` (written once
//! at startup, read on demand).

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// Per-consumer counters. Updated by the consumer tasks, read by
/// `/operator/queues`.
#[derive(Debug, Default)]
pub struct QueueMetrics {
    /// Messages pulled off the queue (successfully or not).
    pub received: AtomicU64,
    /// Messages processed and acked / deleted.
    pub completed: AtomicU64,
    /// Messages that failed and were left for redelivery / retry.
    pub failed: AtomicU64,
    /// Messages that exceeded the retry budget and went to the DLQ (or were
    /// dropped as poison).
    pub dlq: AtomicU64,
    /// Count of messages currently in the dead-letter queue (best-effort,
    /// refreshed by the consumer when the broker supports it).
    pub dlq_depth: AtomicU64,
}

impl QueueMetrics {
    pub fn snapshot(&self) -> QueueMetricsSnapshot {
        QueueMetricsSnapshot {
            received: self.received.load(Ordering::Relaxed),
            completed: self.completed.load(Ordering::Relaxed),
            failed: self.failed.load(Ordering::Relaxed),
            dlq: self.dlq.load(Ordering::Relaxed),
            dlq_depth: self.dlq_depth.load(Ordering::Relaxed),
        }
    }
}

/// Plain copy of the counters for serialization.
#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct QueueMetricsSnapshot {
    pub received: u64,
    pub completed: u64,
    pub failed: u64,
    pub dlq: u64,
    pub dlq_depth: u64,
}

/// Registry shared between the consumers and the operator endpoint.
#[derive(Debug, Clone, Default)]
pub struct QueueMetricsRegistry {
    inner: Arc<Mutex<HashMap<String, Arc<QueueMetrics>>>>,
}

impl QueueMetricsRegistry {
    /// Registers (or returns the existing) metrics for `name`.
    pub fn entry(&self, name: &str) -> Arc<QueueMetrics> {
        let mut map = self.inner.lock().unwrap();
        map.entry(name.to_string())
            .or_insert_with(|| Arc::new(QueueMetrics::default()))
            .clone()
    }

    /// List of `(name, snapshot)` pairs for the operator endpoint.
    pub fn snapshot_all(&self) -> Vec<(String, QueueMetricsSnapshot)> {
        let map = self.inner.lock().unwrap();
        let mut out: Vec<(String, QueueMetricsSnapshot)> = map
            .iter()
            .map(|(k, v)| (k.clone(), v.snapshot()))
            .collect();
        out.sort_by(|a, b| a.0.cmp(&b.0));
        out
    }
}