//! S3 job worker (feature `s3`, spec §8 "Scenario S3").
//!
//! Bridges the existing `ZipProcessor` pipeline to bucket storage: a job
//! downloads its input archive (streaming, to a scratch file under DATA_DIR),
//! runs the *same* per-image anonymization pipeline, uploads the resulting ZIP
//! back to the output bucket, writes a JSON audit log to the logs bucket and —
//! when allowed — calls a completion webhook with a presigned download URL.
//!
//! Job concurrency is bounded by `S3_MAX_CONCURRENT_JOBS` (a semaphore);
//! inside each job the per-image semaphore (`effective_concurrency`) still
//! applies, so several S3 tasks never oversubscribe the GPU/CPU pool.
//!
//! Job state lives in an in-memory tracker (`GET /status/:job_id`); the
//! durable record is the audit log object in the logs bucket.

use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use serde::Serialize;
use tokio::sync::{Mutex, Semaphore};

use crate::config::S3Settings;
use crate::s3_client::S3Client;
use crate::zip_worker::ZipProcessor;

/// Lifecycle of one S3 job (in-memory; `GET /status/:job_id`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum S3JobState {
    Queued,
    Running,
    Done,
    Failed,
}

/// Snapshot of one job, returned by the status endpoint.
#[derive(Debug, Clone, Serialize)]
pub struct S3JobStatus {
    pub job_id: String,
    pub input_key: String,
    pub output_key: String,
    pub state: S3JobState,
    pub processed_images: usize,
    pub error_count: usize,
    pub output_size_bytes: u64,
    pub etag: Option<String>,
    pub started_at: Option<String>,
    pub finished_at: Option<String>,
    pub error: Option<String>,
}

/// In-memory job registry. Not durable across restarts: the audit log object
/// (logs bucket) is the source of truth for completed jobs.
#[derive(Clone, Default)]
pub struct S3JobTracker {
    jobs: Arc<Mutex<std::collections::HashMap<String, S3JobStatus>>>,
}

impl S3JobTracker {
    pub fn new() -> Self {
        Self::default()
    }

    pub async fn submit(&self, job_id: String, input_key: String, output_key: String) {
        let status = S3JobStatus {
            job_id: job_id.clone(),
            input_key,
            output_key,
            state: S3JobState::Queued,
            processed_images: 0,
            error_count: 0,
            output_size_bytes: 0,
            etag: None,
            started_at: None,
            finished_at: None,
            error: None,
        };
        self.jobs.lock().await.insert(job_id, status);
    }

    pub async fn get(&self, job_id: &str) -> Option<S3JobStatus> {
        self.jobs.lock().await.get(job_id).cloned()
    }

    /// Applies `patch` under the lock (single-shot handler for Running/Done/
    /// Failed transitions).
    pub async fn patch(&self, job_id: &str, patch: impl FnOnce(&mut S3JobStatus)) {
        if let Some(job) = self.jobs.lock().await.get_mut(job_id) {
            patch(job);
        }
    }
}

/// Worker that drives one S3 job through the existing `ZipProcessor`.
#[derive(Clone)]
pub struct S3ZipWorker {
    pub(crate) s3: S3Client,
    pub(crate) processor: ZipProcessor,
    pub(crate) settings: Arc<S3Settings>,
    pub(crate) tracker: S3JobTracker,
    jobs_semaphore: Arc<Semaphore>,
}

/// Policy differences between a single `/anonymize/s3` job and a batch sweep.
#[derive(Debug, Clone, Copy)]
pub struct S3JobPolicy {
    /// Delete the input object from the input bucket after success.
    pub delete_input_on_success: bool,
}

impl S3ZipWorker {
    pub fn new(processor: ZipProcessor, s3: S3Client, settings: Arc<S3Settings>) -> Self {
        let tracker = S3JobTracker::new();
        Self {
            s3,
            processor,
            jobs_semaphore: Arc::new(Semaphore::new(settings.max_concurrent_jobs)),
            tracker,
            settings,
        }
    }

    /// Drives a full job: download → anonymize → upload → audit → (webhook).
    /// `move_failed_input` copies the input under `errori/<key>` and deletes
    /// the original when the job fails (single-job default), mirroring the
    /// local pipeline's error surfacing.
    pub async fn run_job(
        &self,
        job_id: &str,
        input_key: &str,
        output_key: &str,
        callback_url: Option<&str>,
        policy: S3JobPolicy,
    ) -> Result<()> {
        // Job-level concurrency cap: several queued jobs may wait here; the
        // per-image semaphore inside `processor` still bounds actual compute.
        let permit = self
            .jobs_semaphore
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| anyhow::anyhow!("S3 job semaphore closed"))?;

        let started = Instant::now();
        self.tracker
            .patch(job_id, |j| {
                j.state = S3JobState::Running;
                j.started_at = Some(chrono::Utc::now().to_rfc3339());
            })
            .await;

        let scratch = self.processor.data_dir().to_path_buf();
        let temp_in = scratch.join(format!("s3_{}.in", job_id));
        let outcome = self.process_one(job_id, input_key, output_key, &temp_in).await;

        let job_result = match outcome {
            Ok((outcome, etag)) => {
                if policy.delete_input_on_success {
                    if let Err(e) = self.s3.delete_object(&self.s3.bucket_input, input_key).await {
                        tracing::warn!("failed to delete consumed input {input_key}: {e}");
                    }
                }
                self.tracker
                    .patch(job_id, |j| {
                        j.state = S3JobState::Done;
                        j.processed_images = outcome.processed_count;
                        j.error_count = outcome.error_count;
                        j.output_size_bytes = outcome.output_size;
                        j.etag = Some(etag.clone());
                        j.finished_at = Some(chrono::Utc::now().to_rfc3339());
                    })
                    .await;
                tracing::info!(
                    "S3 job {job_id} done: {} → {} ({} images, {} errors, {} bytes, {} ms)",
                    input_key,
                    output_key,
                    outcome.processed_count,
                    outcome.error_count,
                    outcome.output_size,
                    started.elapsed().as_millis()
                );
                self.notify_completion(callback_url, job_id, input_key, output_key, &outcome, &etag).await;
                Ok(())
            }
            Err(e) => {
                // Move the input to an error folder so a manual/retry sweep can
                // pick it up without re-uploading, then delete the original.
                let error_key = format!("errori/{}", input_key);
                if let Err(copy_err) = self
                    .s3
                    .copy_object(
                        &self.s3.bucket_input,
                        input_key,
                        &self.s3.bucket_input,
                        &error_key,
                    )
                    .await
                {
                    tracing::warn!("could not park failed input as {error_key}: {copy_err}");
                } else if let Err(del_err) = self
                    .s3
                    .delete_object(&self.s3.bucket_input, input_key)
                    .await
                {
                    tracing::warn!("could not delete input {input_key} after parking: {del_err}");
                }
                self.tracker
                    .patch(job_id, |j| {
                        j.state = S3JobState::Failed;
                        j.error = Some(format!("{e:#}"));
                        j.finished_at = Some(chrono::Utc::now().to_rfc3339());
                    })
                    .await;
                self.write_audit_error(job_id, input_key, output_key, &format!("{e:#}"))
                    .await;
                tracing::error!("S3 job {job_id} failed for {input_key}: {e:#}");
                Err(e)
            }
        };

        // Scratch input + the processed local output ZIP are temporary by
        // design (already uploaded); never leave them behind for retention.
        let _ = tokio::fs::remove_file(&temp_in).await;
        drop(permit);
        job_result
    }

    /// Download → anonymize → upload. Returns the processed outcome + the S3
    /// ETag of the uploaded result.
    async fn process_one(
        &self,
        job_id: &str,
        input_key: &str,
        output_key: &str,
        temp_in: &std::path::Path,
    ) -> Result<(crate::zip_worker::ZipJobOutcome, String)> {
        tracing::info!("S3 job {job_id}: downloading s3://{}/{}", self.s3.bucket_input, input_key);
        let downloaded = self
            .s3
            .download_streaming(&self.s3.bucket_input, input_key, temp_in)
            .await
            .context("download input archive from S3")?;

        let outcome = self
            .processor
            .process_archive_file(input_key, temp_in)
            .await
            .with_context(|| format!("anonymization failed for {input_key} ({downloaded} bytes)"))?;
        tracing::info!(
            "S3 job {job_id}: {} → {} processed, {} errors, local {} ({} bytes)",
            input_key,
            outcome.processed_count,
            outcome.error_count,
            outcome.output_path.display(),
            outcome.output_size
        );

        let etag = self
            .s3
            .upload_streaming(&self.s3.bucket_output, output_key, &outcome.output_path)
            .await
            .with_context(|| format!("upload result s3://{}/{output_key}", self.s3.bucket_output))?;
        let uploaded_bytes = std::fs::metadata(&outcome.output_path)
            .map(|m| m.len())
            .unwrap_or(outcome.output_size);

        // The local anonymized ZIP under DATA_DIR was only a staging area;
        // purge it so the STORE retention loop treats it as nonexistent.
        let _ = tokio::fs::remove_file(&outcome.output_path).await;

        self.write_audit(
            job_id,
            input_key,
            output_key,
            &outcome,
            &etag,
            uploaded_bytes,
        )
        .await?;
        Ok((outcome, etag))
    }

    /// JSON audit record per job, uploaded to the logs bucket. This is the
    /// *durable* trace (the in-memory tracker dies with the process).
    async fn write_audit(
        &self,
        job_id: &str,
        input_key: &str,
        output_key: &str,
        outcome: &crate::zip_worker::ZipJobOutcome,
        etag: &str,
        input_bytes: u64,
    ) -> Result<()> {
        let entry = serde_json::json!({
            "job_id": job_id,
            "event": "processing_completed",
            "timestamp": chrono::Utc::now().to_rfc3339(),
            "input": input_key,
            "input_bytes": input_bytes,
            "output": output_key,
            "output_bytes": outcome.output_size,
            "etag": etag,
            "processed_images": outcome.processed_count,
            "error_count": outcome.error_count,
            "camera_summaries": outcome.camera_summaries.iter().map(|c| serde_json::json!({
                "camera": c.camera_id,
                "branch": c.branch.as_str(),
                "images": c.images,
                "detections_stored": c.detections_stored,
                "fp_crops_saved": c.fp_crops_saved,
            })).collect::<Vec<_>>(),
            "errors": outcome.errors.iter().map(|e| serde_json::json!({
                "entry": e.entry,
                "message": e.message,
            })).collect::<Vec<_>>(),
        });
        let key = format!("logs/{job_id}.json");
        self.s3
            .upload_bytes(&self.s3.bucket_logs, &key, serde_json::to_vec_pretty(&entry)?)
            .await
            .with_context(|| format!("write audit log s3://{}/{key}", self.s3.bucket_logs))
    }

    async fn write_audit_error(
        &self,
        job_id: &str,
        input_key: &str,
        output_key: &str,
        error: &str,
    ) {
        let entry = serde_json::json!({
            "job_id": job_id,
            "event": "processing_failed",
            "timestamp": chrono::Utc::now().to_rfc3339(),
            "input": input_key,
            "output": output_key,
            "error": error,
        });
        let key = format!("logs/{job_id}.json");
        if let Err(e) = self
            .s3
            .upload_bytes(
                &self.s3.bucket_logs,
                &key,
                serde_json::to_vec_pretty(&entry).unwrap_or_default(),
            )
            .await
        {
            tracing::warn!("could not write audit error {key}: {e}");
        }
    }

    /// Completion webhook. Only fires when the host is on the
    /// `S3_WEBHOOK_ALLOWED_HOSTS` allowlist — an empty allowlist disables
    /// webhooks entirely (anti-SSRF default). Failures are logged, never
    /// fail the job (the audit log already records completion).
    async fn notify_completion(
        &self,
        callback_url: Option<&str>,
        job_id: &str,
        input_key: &str,
        output_key: &str,
        outcome: &crate::zip_worker::ZipJobOutcome,
        etag: &str,
    ) {
        let Some(url) = callback_url else {
            return;
        };
        if !webhook_host_allowed(&self.settings, url) {
            tracing::warn!(
                "S3 job {job_id}: webhook {url} not on the S3_WEBHOOK_ALLOWED_HOSTS allowlist, skipped"
            );
            return;
        }
        let download_url = match self
            .s3
            .generate_presigned_url(&self.s3.bucket_output, output_key, 3600)
            .await
        {
            Ok(u) => u,
            Err(e) => {
                tracing::warn!("S3 job {job_id}: could not presign download URL for webhook: {e}");
                return;
            }
        };
        let payload = serde_json::json!({
            "event": "processing_completed",
            "job_id": job_id,
            "input": input_key,
            "output": output_key,
            "etag": etag,
            "processed_images": outcome.processed_count,
            "error_count": outcome.error_count,
            "output_size": outcome.output_size,
            "download_url": download_url,
        });
        if let Err(e) = reqwest::Client::new()
            .post(url)
            .json(&payload)
            .timeout(std::time::Duration::from_secs(10))
            .send()
            .await
        {
            tracing::warn!("S3 job {job_id}: webhook POST to {url} failed: {e}");
        }
    }
}

/// Allowlist check for the completion webhook URL: the `host[:port]` of the
/// parsed URL must be one of `S3_WEBHOOK_ALLOWED_HOSTS`. The port-only match
/// is exact; a bare host entry matches any port of that host.
pub fn webhook_host_allowed(settings: &S3Settings, url_str: &str) -> bool {
    if settings.webhook_allowed_hosts.is_empty() {
        return false;
    }
    let Ok(url) = reqwest::Url::parse(url_str) else {
        return false;
    };
    let Some(host) = url.host_str() else {
        return false;
    };
    let host = host.to_ascii_lowercase();
    settings.webhook_allowed_hosts.iter().any(|allowed| {
        if let Some((h, p)) = allowed.rsplit_once(':') {
            h == host && p == url.port().map(|p| p.to_string()).as_deref().unwrap_or("80")
        } else {
            allowed == &host
        }
    })
}

/// Default destination key for a submitted input: `elaborati/<stem>_elaborato.zip`
/// mirroring the local output-name rule (`output_stem` in `zip_worker`).
pub fn default_output_key(input_key: &str) -> String {
    let base = input_key.rsplit('/').next().unwrap_or(input_key);
    match base.rsplit_once('.') {
        Some((stem, _)) if !stem.is_empty() => format!("elaborati/{stem}_elaborato.zip"),
        _ => format!("elaborati/{base}_elaborato.zip"),
    }
}

/// One submitted job of a batch sweep (returned to the operator); carries the
/// listing metadata of the input object so a wide sweep stays auditable.
#[derive(Debug, Clone, Serialize)]
pub struct S3BatchSubmission {
    pub input_key: String,
    pub input_size: u64,
    pub input_etag: String,
    pub input_last_modified: String,
    pub input_storage_class: String,
    pub output_key: String,
    pub job_id: String,
}

/// Keys a batch sweep must skip: empty, already parked under `errori/`, or
/// not a supported archive (`.zip`/`.7z`/`.rar`).
fn is_ignored_key(key: &str) -> bool {
    if key.is_empty() || key.starts_with('/') {
        return true;
    }
    if key.split('/').any(|seg| seg == "errori") {
        return true;
    }
    let lower = key.to_ascii_lowercase();
    !(lower.ends_with(".zip") || lower.ends_with(".7z") || lower.ends_with(".rar"))
}

impl S3ZipWorker {
    /// Operator batch sweep: lists the input bucket under `prefix`, submits
    /// every eligible archive as a tracked background job and returns the
    /// accepted set. Real concurrency stays bounded by the job semaphore.
    pub async fn submit_batch(
        &self,
        prefix: &str,
        max_files: usize,
        delete_input_on_success: bool,
    ) -> Result<Vec<S3BatchSubmission>> {
        let objects = self
            .s3
            .list_objects(&self.s3.bucket_input, Some(prefix), Some(max_files as i32))
            .await
            .context("list input bucket for batch sweep")?;

        let mut submissions = Vec::new();
        for obj in objects {
            if is_ignored_key(&obj.key) {
                continue;
            }
            let input_key = obj.key;
            let output_key = default_output_key(&input_key);
            let job_id = uuid::Uuid::new_v4().to_string();
            self.tracker
                .submit(job_id.clone(), input_key.clone(), output_key.clone())
                .await;

            let worker = self.clone();
            let cb: Option<String> = None;
            let (jid, ik, ok) = (job_id.clone(), input_key.clone(), output_key.clone());
            tokio::spawn(async move {
                let _ = worker
                    .run_job(
                        &jid,
                        &ik,
                        &ok,
                        cb.as_deref(),
                        S3JobPolicy {
                            delete_input_on_success,
                        },
                    )
                    .await;
            });
            submissions.push(S3BatchSubmission {
                input_key,
                input_size: obj.size,
                input_etag: obj.etag,
                input_last_modified: obj.last_modified.to_rfc3339(),
                input_storage_class: obj.storage_class,
                output_key,
                job_id,
            });
        }
        Ok(submissions)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::S3Settings;

    fn settings_with_webhooks(hosts: Vec<String>) -> S3Settings {
        S3Settings {
            endpoint: Some("http://localhost:9000".into()),
            force_path_style: true,
            bucket_input: "in".into(),
            bucket_output: "out".into(),
            bucket_logs: "logs".into(),
            max_concurrent_jobs: 2,
            webhook_allowed_hosts: hosts,
        }
    }

    #[test]
    fn default_output_key_mirrors_local_rule() {
        assert_eq!(
            default_output_key("cam/abc.zip"),
            "elaborati/abc_elaborato.zip"
        );
        assert_eq!(
            default_output_key("archive.7z"),
            "elaborati/archive_elaborato.zip"
        );
        assert_eq!(
            default_output_key("noext"),
            "elaborati/noext_elaborato.zip"
        );
    }

    #[test]
    fn webhook_allowlist() {
        // Empty allowlist = webhooks disabled entirely.
        assert!(!webhook_host_allowed(&settings_with_webhooks(vec![]), "http://a/b"));

        let s = settings_with_webhooks(vec!["notifiche.internal".into(), "hook.it:8443".into()]);
        assert!(webhook_host_allowed(
            &s,
            "http://notifiche.internal/webhook"
        ));
        assert!(webhook_host_allowed(&s, "https://hook.it:8443/end"));
        // Port mismatch → rejected.
        assert!(!webhook_host_allowed(&s, "https://hook.it:9000/end"));
        // Different host → rejected.
        assert!(!webhook_host_allowed(&s, "http://evil.example/webhook"));
        // Non-URL → rejected.
        assert!(!webhook_host_allowed(&s, "not a url"));
        // Host case-insensitivity.
        assert!(webhook_host_allowed(
            &s,
            "http://NOTIFICHE.INTERNAL:80/x"
        ));
    }

    #[test]
    fn batch_ignored_keys() {
        assert!(is_ignored_key(""));
        assert!(is_ignored_key("/etc/passwd")); // absolute-ish key
        assert!(is_ignored_key("errori/abc.zip")); // already parked
        assert!(is_ignored_key("sub/errori/abc.zip"));
        assert!(is_ignored_key("readme.txt")); // not an archive
        assert!(is_ignored_key("cam.png"));
        assert!(!is_ignored_key("cam_001.zip"));
        assert!(!is_ignored_key("cam.7z"));
        assert!(!is_ignored_key("CAM_001.RAR"));
        assert!(!is_ignored_key("deep/path/cam.zip"));
    }
}