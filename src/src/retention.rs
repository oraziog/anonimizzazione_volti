//! DATA_DIR housekeeping: retention of STORE outputs and the operator jobs
//! ledger.
//!
//! # Retention (env `RETENTION_*`)
//! The anonymized outputs (`<input>_elaborato.zip`, the spec §8 "scrittura
//! output STORE") accumulate in `DATA_DIR` forever unless something removes
//! them. A background loop runs every `RETENTION_INTERVAL_SECS` and applies
//! two rules, both optional:
//!
//! - **age**: delete outputs older than `RETENTION_MAX_DAYS` days;
//! - **size**: when the total size of remaining outputs exceeds
//!   `RETENTION_MAX_GB`, delete the oldest eligible ones first until the
//!   total is back under the threshold.
//!
//! Outputs modified within the last `RETENTION_MIN_AGE_SECS` seconds are
//! never touched (they may be mid-write or currently streaming back to a
//! client). The whole policy is disabled by `RETENTION_ENABLED=false` or by
//! setting both thresholds to 0.
//!
//! # Jobs ledger
//! Every finished job appends one JSON line to `DATA_DIR/jobs.jsonl`
//! (bounded to [`JOBS_LEDGER_MAX_LINES`]); `GET /operator/jobs` reads it
//! back (gated by `X-Operator-Key`) and cross-checks the outputs still on
//! disk, reporting any orphan `*_elaborato.zip` files.

use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::config::{Config, RuntimeConfig};
use crate::zip_worker::{ArchiveEntryError, ZipJobOutcome};

/// Newline-delimited JSON ledger of finished jobs (in `DATA_DIR`).
pub const JOBS_LEDGER_FILENAME: &str = "jobs.jsonl";
/// Cap on ledger lines: appends beyond this rotate the file (keep the newest
/// half), so the file and `GET /operator/jobs` stay cheap forever.
pub const JOBS_LEDGER_MAX_LINES: usize = 1000;

/// One finished job, as recorded in the ledger and served by `/operator/jobs`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobLedgerEntry {
    /// Uploaded archive name (or the joined list for /anonymize/batch).
    pub input: String,
    /// Output ZIP file name under DATA_DIR (`<input>_elaborato.zip`).
    pub output: String,
    pub processed_images: usize,
    pub error_count: usize,
    pub output_size: u64,
    pub started_at: String,
    pub finished_at: String,
    pub duration_ms: u64,
    /// First few per-entry failures (same text as `X-Processing-Errors-Detail`).
    #[serde(default)]
    pub errors: Vec<JobLedgerError>,
    /// Set by the operator endpoint: whether the output file still exists.
    #[serde(default)]
    pub on_disk: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobLedgerError {
    pub entry: String,
    pub error: String,
}

/// Result of one retention pass (logged and returned by the tests).
#[derive(Debug, Clone)]
pub struct RetentionReport {
    pub deleted: usize,
    pub freed_bytes: u64,
    pub remaining: usize,
    pub remaining_bytes: u64,
}

/// Periodic STORE-output cleanup. Runs one pass immediately, then sleeps
/// `RETENTION_INTERVAL_SECS` between passes (forever).
pub async fn retention_loop(runtime: RuntimeConfig) {
    let snapshot = runtime.snapshot();
    tracing::info!(
        "STORE retention policy active: max_days={} max_gb={} interval={}s (min_age={}s)",
        snapshot.retention_max_days,
        snapshot.retention_max_gb,
        snapshot.retention_interval_secs,
        snapshot.retention_min_age_secs
    );
    loop {
        let snapshot = runtime.snapshot();
        let interval_secs = snapshot.retention_interval_secs;
        let inner = snapshot;
        match tokio::task::spawn_blocking(move || run_retention(&inner)).await {
            Ok(Ok(report)) => tracing::info!(
                "STORE retention pass: deleted {} files ({} bytes freed); {} output(s) remain ({} bytes)",
                report.deleted,
                report.freed_bytes,
                report.remaining,
                report.remaining_bytes
            ),
            Ok(Err(e)) => tracing::warn!("STORE retention pass failed: {e:#}"),
            Err(e) => tracing::error!("STORE retention worker panicked: {e}"),
        }
        tokio::time::sleep(Duration::from_secs(interval_secs)).await;
    }
}

/// One retention pass over `DATA_DIR`: applies the age rule, then the size
/// rule (oldest-eligible-first) to whatever remains. Only `*_elaborato.zip`
/// files are considered; everything else in DATA_DIR (sqlite, model cache,
/// spools, scratch dirs, the ledger) is never touched.
pub fn run_retention(cfg: &Config) -> Result<RetentionReport> {
    // Disable semantics are enforced here too, not just at loop startup: with
    // RETENTION_ENABLED=false (or both thresholds 0) nothing is ever deleted.
    if !cfg.retention_active() {
        return Ok(RetentionReport {
            deleted: 0,
            freed_bytes: 0,
            remaining: 0,
            remaining_bytes: 0,
        });
    }
    let now = std::time::SystemTime::now();

    let mut files: Vec<(PathBuf, std::time::SystemTime, u64)> = Vec::new();
    for entry in std::fs::read_dir(&cfg.data_dir)
        .with_context(|| format!("read DATA_DIR {}", cfg.data_dir.display()))?
    {
        let path = entry?.path();
        if !path.is_file() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if !name.ends_with("_elaborato.zip") {
            continue;
        }
        let meta = path.metadata()?;
        files.push((path, meta.modified().unwrap_or(now), meta.len()));
    }

    let mut deleted = 0usize;
    let mut freed = 0u64;

    // Rule 1 — age: anything older than RETENTION_MAX_DAYS is removed.
    if cfg.retention_max_days > 0 {
        let cutoff = now - Duration::from_secs(cfg.retention_max_days as u64 * 86_400);
        files.retain(|(path, mtime, len)| {
            if *mtime < cutoff {
                match std::fs::remove_file(path) {
                    Ok(()) => {
                        deleted += 1;
                        freed += *len;
                        false
                    }
                    Err(e) => {
                        tracing::warn!("STORE retention: cannot remove {}: {e}", path.display());
                        true
                    }
                }
            } else {
                true
            }
        });
    }

    // Rule 2 — size: while the total exceeds the threshold, evict the oldest
    // file that is old enough (never a fresh output).
    if cfg.retention_max_gb > 0.0 {
        let max_bytes = (cfg.retention_max_gb * 1024.0 * 1024.0 * 1024.0) as u64;
        let mut total: u64 = files.iter().map(|(_, _, len)| *len).sum();
        if total > max_bytes {
            // Oldest first (SystemTime is Ord).
            files.sort_by_key(|(_, mtime, _)| *mtime);
            let min_age = now - Duration::from_secs(cfg.retention_min_age_secs);
            for (path, mtime, len) in files.iter_mut() {
                if total <= max_bytes {
                    break;
                }
                if *mtime > min_age {
                    continue; // in-flight output: never touch
                }
                match std::fs::remove_file(path.as_path()) {
                    Ok(()) => {
                        total = total.saturating_sub(*len);
                        deleted += 1;
                        freed += *len;
                    }
                    Err(e) => {
                        tracing::warn!("STORE retention: cannot remove {}: {e}", path.display());
                    }
                }
            }
        }
    }

    // The size pass deletes on disk while iterating; drop the removed entries
    // so the report counts only what is actually still there.
    files.retain(|(path, _, _)| path.exists());
    let remaining_bytes: u64 = files.iter().map(|(_, _, len)| *len).sum();
    Ok(RetentionReport {
        deleted,
        freed_bytes: freed,
        remaining: files.len(),
        remaining_bytes,
    })
}

// ─── Jobs ledger ─────────────────────────────────────────────────────────────

/// Appends one finished job to `DATA_DIR/jobs.jsonl`, rotating to the newest
/// [`JOBS_LEDGER_MAX_LINES`] lines when the cap is exceeded. Called from the
/// `/anonymize` and `/anonymize/batch` handlers (the single-job lock
/// guarantees a single writer).
pub fn record_job(
    data_dir: &Path,
    input: &str,
    outcome: &ZipJobOutcome,
    started_at: DateTime<Utc>,
) -> Result<()> {
    let finished_at = Utc::now();
    let entry = JobLedgerEntry {
        input: input.to_string(),
        output: outcome.output_name.clone(),
        processed_images: outcome.processed_count,
        error_count: outcome.error_count,
        output_size: outcome.output_size,
        started_at: started_at.to_rfc3339(),
        finished_at: finished_at.to_rfc3339(),
        duration_ms: (finished_at - started_at).num_milliseconds().max(0) as u64,
        errors: outcome
            .errors
            .iter()
            .take(10)
            .map(|e: &ArchiveEntryError| JobLedgerError {
                entry: e.entry.clone(),
                error: e.message.clone(),
            })
            .collect(),
        on_disk: false,
    };

    let path = data_dir.join(JOBS_LEDGER_FILENAME);
    let mut lines: Vec<String> = match std::fs::read_to_string(&path) {
        Ok(text) => text
            .lines()
            .map(str::to_string)
            .filter(|l| !l.trim().is_empty())
            .collect(),
        Err(_) => Vec::new(),
    };
    lines.push(serde_json::to_string(&entry)?);
    if lines.len() > JOBS_LEDGER_MAX_LINES {
        lines.drain(..lines.len() - JOBS_LEDGER_MAX_LINES / 2);
    }
    std::fs::write(&path, lines.join("\n") + "\n")
        .with_context(|| format!("write jobs ledger {}", path.display()))
}

/// Reads the ledger in file order (oldest first). Invalid / torn lines are
/// skipped, so a concurrent write can never break the endpoint.
pub fn read_jobs_ledger(data_dir: &Path) -> Result<Vec<JobLedgerEntry>> {
    let path = data_dir.join(JOBS_LEDGER_FILENAME);
    let text = std::fs::read_to_string(&path)?;
    let mut out = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Ok(entry) = serde_json::from_str::<JobLedgerEntry>(line) {
            out.push(entry);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static DIR_COUNTER: AtomicUsize = AtomicUsize::new(0);

    fn temp_dir(tag: &str) -> PathBuf {
        let n = DIR_COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "av_retention_test_{tag}_{}_{n}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_aged(path: &Path, data: &[u8], age_secs: u64) {
        std::fs::write(path, data).unwrap();
        let t = std::time::SystemTime::now() - Duration::from_secs(age_secs);
        // set_modified needs a write handle (Windows rejects read-only ones).
        std::fs::OpenOptions::new()
            .write(true)
            .open(path)
            .unwrap()
            .set_modified(t)
            .unwrap();
    }

    #[test]
    fn age_rule_removes_only_old_outputs() {
        let dir = temp_dir("age");
        let old = dir.join("old_elaborato.zip");
        let fresh = dir.join("fresh_elaborato.zip");
        let unrelated = dir.join("keepme.db");
        write_aged(&old, b"old", 40 * 86_400);
        write_aged(&fresh, b"fresh", 60);
        std::fs::write(&unrelated, b"db").unwrap();

        let mut cfg = Config::test_default();
        cfg.data_dir = dir.clone();
        cfg.retention_enabled = true;
        cfg.retention_max_days = 30;
        cfg.retention_max_gb = 0.0; // size rule off

        let report = run_retention(&cfg).unwrap();
        assert_eq!(report.deleted, 1);
        assert_eq!(report.freed_bytes, 3);
        assert!(!old.exists());
        assert!(fresh.exists());
        assert!(unrelated.exists(), "non-output files must never be touched");
    }

    #[test]
    fn size_rule_evicts_oldest_first_and_spares_fresh() {
        let dir = temp_dir("size");
        // Three 1 KB outputs aged 2 h / 1 h / 2 min. A ~2 KB threshold forces
        // exactly one eviction; the oldest (2 h) must go first and the young
        // one must never be evicted even when over the threshold.
        for (name, age) in [
            ("a_elaborato.zip", 7200u64),
            ("b_elaborato.zip", 3600),
            ("c_elaborato.zip", 120),
        ] {
            write_aged(&dir.join(name), &vec![0u8; 1024], age);
        }
        let mut cfg = Config::test_default();
        cfg.data_dir = dir.clone();
        cfg.retention_enabled = true;
        cfg.retention_max_days = 0;
        // ~2.68 KB threshold: total is 3 KB → exactly one 1 KB eviction.
        cfg.retention_max_gb = 0.0000025;
        cfg.retention_min_age_secs = 60;

        let report = run_retention(&cfg).unwrap();
        assert_eq!(report.deleted, 1);
        assert_eq!(report.remaining, 2);
        assert!(
            !dir.join("a_elaborato.zip").exists(),
            "oldest must go first"
        );
        assert!(dir.join("b_elaborato.zip").exists());
        assert!(dir.join("c_elaborato.zip").exists());

        // A fresh output (younger than min_age) is never evicted even when
        // the total is over the threshold: b (1 h old) goes, d (10 s) stays.
        let cfg2 = cfg.clone();
        write_aged(&dir.join("d_elaborato.zip"), &vec![0u8; 1024], 10);
        let report = run_retention(&cfg2).unwrap();
        assert_eq!(
            report.deleted, 1,
            "only the oldest eligible file is removed"
        );
        assert!(
            !dir.join("b_elaborato.zip").exists(),
            "oldest eligible goes"
        );
        assert!(dir.join("d_elaborato.zip").exists(), "fresh output spared");
    }

    #[test]
    fn disabled_policy_does_nothing() {
        let dir = temp_dir("off");
        let p = dir.join("x_elaborato.zip");
        write_aged(&p, b"data", 400 * 86_400);
        let mut cfg = Config::test_default();
        cfg.data_dir = dir.clone();
        cfg.retention_enabled = false;
        cfg.retention_max_days = 1;
        cfg.retention_max_gb = 0.0;
        let report = run_retention(&cfg).unwrap();
        assert_eq!(report.deleted, 0);
        assert!(p.exists());
        // Both thresholds 0 also disable the rules (nothing to delete).
        cfg.retention_enabled = true;
        cfg.retention_max_days = 0;
        cfg.retention_max_gb = 0.0;
        let report = run_retention(&cfg).unwrap();
        assert_eq!(report.deleted, 0);
    }

    #[test]
    fn ledger_roundtrip_and_rotation_cap() {
        let dir = temp_dir("ledger");
        let outcome = ZipJobOutcome {
            output_path: dir.join("z_elaborato.zip"),
            output_size: 42,
            output_name: "z_elaborato.zip".into(),
            error_count: 2,
            processed_count: 10,
            camera_summaries: Vec::new(),
            errors: vec![ArchiveEntryError {
                entry: "readme.txt".into(),
                message: "unsupported entry".into(),
            }],
        };
        let started = Utc::now() - chrono::Duration::seconds(5);
        record_job(&dir, "z.zip", &outcome, started).unwrap();

        let entries = read_jobs_ledger(&dir).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].input, "z.zip");
        assert_eq!(entries[0].output, "z_elaborato.zip");
        assert_eq!(entries[0].processed_images, 10);
        assert_eq!(entries[0].error_count, 2);
        assert_eq!(entries[0].output_size, 42);
        assert_eq!(entries[0].errors[0].entry, "readme.txt");
        assert_eq!(entries[0].duration_ms, 5000);

        // Push past the cap: the ledger rotates and keeps the newest half.
        let mut outcome2 = outcome;
        for i in 0..JOBS_LEDGER_MAX_LINES + 20 {
            outcome2.output_name = format!("z{i}_elaborato.zip");
            outcome2.output_size = i as u64;
            record_job(&dir, "z.zip", &outcome2, started).unwrap();
        }
        let entries = read_jobs_ledger(&dir).unwrap();
        assert!(
            entries.len() <= JOBS_LEDGER_MAX_LINES,
            "ledger must never exceed its cap (got {})",
            entries.len()
        );
        // The newest record survived the rotation.
        let last = entries.last().unwrap();
        assert_eq!(
            last.output,
            format!("z{}_elaborato.zip", JOBS_LEDGER_MAX_LINES + 19)
        );
    }
}
