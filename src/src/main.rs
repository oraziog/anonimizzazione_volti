//! Freebuff Desktop — batch face anonymization service (Rust + ONNX).
//!
//! Spec §8 `main.rs`: runtime ONNX model resolution (download + verify +
//! fail-fast), SQLite pool, nightly ROI-finalization + retraining scheduler,
//! and the Axum HTTP server exposing:
//!   - `POST /anonymize`        single-job archive ingestion (§2), output
//!     streamed from disk
//!   - `POST /anonymize/batch`  multiple archives / chunked upload, no body
//!     limit (spooled to disk, processed in sequence, merged ZIP response)
//!   - `POST /anonymize/s3`      async S3 ingestion (feature `s3`): accepts an
//!     `input_key`, downloads from the input bucket, processes, uploads to the
//!     output bucket, writes an audit log, optionally calls a webhook
//!   - `GET  /status/:job_id`     async S3 job status (feature `s3`)
//!   - `GET  /health`
//!   - `GET  /operator/cameras` operator listing / FSM reset (§4), gated by
//!     the `X-Operator-Key` header
//!   - `GET  /operator/gpu`       inference execution-provider status + measured
//!     per-stage inference statistics, gated the same way
//!   - `GET  /operator/jobs`      last processed jobs + orphaned STORE outputs
//!     (reads `DATA_DIR/jobs.jsonl` + the on-disk outputs), gated the same way
//!
//! A background loop enforces the STORE-output retention policy
//! (`RETENTION_*` env vars) so the anonymized ZIPs accumulating in DATA_DIR
//! are pruned by age and/or total size (see `retention.rs`).

mod config;
mod db;
mod eval_fddb;
mod eval_wider;
mod model_loader;
mod models;
mod pipeline;
mod retention;
mod roi;
#[cfg(any(feature = "queue", feature = "rabbitmq"))]
mod queue_metrics;
#[cfg(feature = "s3")]
mod s3_client;
#[cfg(feature = "queue")]
mod queue_sqs;
#[cfg(feature = "rabbitmq")]
mod queue_rabbitmq;
#[cfg(feature = "retraining")]
mod training;
mod zip_worker;
#[cfg(feature = "s3")]
mod zip_worker_s3;

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use axum::body::Body;
use axum::extract::multipart::Field;
use axum::extract::{DefaultBodyLimit, Multipart, Path, Query, State};
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use chrono::{Duration as ChronoDuration, Utc};
use serde::Serialize;
use tokio::io::AsyncWriteExt;
use tokio::sync::Mutex;
use tokio_util::io::ReaderStream;

use crate::config::{Config, MaskSegmenter, RuntimeConfig};
use crate::db::{Camera, Db};
use crate::model_loader::{ensure_model, ExecutionSettings};
use crate::models::{ModelStore, SessionPool};
use crate::retention::{read_jobs_ledger, record_job, JobLedgerEntry};
use crate::roi::{extract_roi, RoiOutcome, RoiPolygon};
use crate::zip_worker::{sanitize_filename, ArchiveEntryError, ZipJobOutcome, ZipProcessor};

const OPERATOR_KEY_HEADER: &str = "x-operator-key";
const DB_FILENAME: &str = "anonimizzazione_volti.sqlite3";

#[derive(Clone)]
struct AppState {
    cfg: Arc<Config>,
    /// Hot-reloadable runtime settings (operator settings UI): the same
    /// handle shared by the worker and both background loops, so a patch
    /// applied here is picked up by the next snapshot.
    settings: RuntimeConfig,
    db: Db,
    worker: ZipProcessor,
    /// Global single-job lock: concurrent uploads get HTTP 429 (§2).
    job_lock: Arc<Mutex<()>>,
    /// S3 worker, present only when `S3_ENABLED=true` **and** built with
    /// `--features s3`; the `/anonymize/s3` + `/status/:job_id` routes are
    /// registered exactly then.
    #[cfg(feature = "s3")]
    s3: Option<S3ZipWorker>,
    /// Live counters of the async queue consumers (`/operator/queues`).
    #[cfg(any(feature = "queue", feature = "rabbitmq"))]
    queue_metrics: queue_metrics::QueueMetricsRegistry,
}

#[cfg(feature = "s3")]
use crate::s3_client::S3Client;
#[cfg(feature = "s3")]
use crate::zip_worker_s3::{default_output_key, webhook_host_allowed, S3ZipWorker};
#[cfg(feature = "s3")]
use serde::Deserialize;

// ─── Entry point ────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    if let Err(e) = run().await {
        // Fail-fast per spec §3: the service must not run with missing models
        // or an invalid configuration.
        tracing::error!("fatal startup error: {e:#}");
        std::process::exit(1);
    }
}

async fn run() -> Result<()> {
    // CLI subcommand: offline detector evaluation against WIDER FACE or
    // FDDB. No server, no config, no downloads.
    let cli_args: Vec<String> = std::env::args().skip(1).collect();
    match cli_args.first().map(|s| s.as_str()) {
        Some("eval-wider") => match eval_wider::run(&cli_args[1..]) {
            Ok(()) => std::process::exit(0),
            Err(e) => {
                tracing::error!("eval-wider failed: {e:#}");
                std::process::exit(1);
            }
        },
        Some("eval-fddb") => match eval_fddb::run(&cli_args[1..]) {
            Ok(()) => std::process::exit(0),
            Err(e) => {
                tracing::error!("eval-fddb failed: {e:#}");
                std::process::exit(1);
            }
        },
        _ => {}
    }

    let runtime = RuntimeConfig::from_env().context("invalid configuration")?;
    let cfg = runtime.snapshot();
    tracing::info!(
        "effective image concurrency: {} ({} core(s), env override {:?})",
        cfg.effective_concurrency(),
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1),
        cfg.max_concurrent_images
    );

    // Execution provider for every ONNX session (env `ORT_EXECUTION_PROVIDER`;
    // fail-fast: a configured GPU that cannot be used aborts startup instead of
    // silently running on CPU).
    crate::model_loader::configure_execution(ExecutionSettings {
        provider: cfg.execution_provider,
        device_id: cfg.gpu_device_id,
        gpu_memory_limit_bytes: cfg.gpu_memory_limit_bytes,
        enable_tf32: cfg.enable_tf32,
        enable_fp16: cfg.enable_fp16,
    })
    .context("configure execution provider")?;
    tracing::info!(
        "inference execution provider: {} (device {}, gpu mem limit {:?}, tf32 {}, fp16 {})",
        cfg.execution_provider.as_str(),
        cfg.gpu_device_id,
        cfg.gpu_memory_limit_bytes,
        cfg.enable_tf32,
        cfg.enable_fp16
    );

    // Storage layout.
    for dir in [
        &cfg.data_dir,
        &cfg.dataset_fp_dir,
        &cfg.dataset_seed_real_faces_dir,
        &cfg.models_backup_dir,
        &cfg.model_cache_dir,
    ] {
        std::fs::create_dir_all(dir)
            .with_context(|| format!("create data dir {}", dir.display()))?;
    }

    // Models: download at runtime with SHA-256 verification, fail-fast (§3).
    let http = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(30))
        .user_agent("anonimizzazione-volti/0.1")
        .build()
        .context("build HTTP client")?;

    let detector =
        ensure_model(&http, &cfg.yolo_model_url, cfg.yolo_sha256.as_deref(), &cfg.model_cache_dir)
            .await
            .with_context(|| "face detector model resolution failed")?;
    // Validate that the file is a loadable ONNX before serving (§3).
    model_loader::load_session(&detector.path)
        .with_context(|| format!("detector model not loadable: {}", detector.path.display()))?;
    tracing::info!(
        "YOLOv8-Face model ready at {} (downloaded: {})",
        detector.path.display(),
        detector.downloaded
    );

    let classifier_pool = match cfg.classifier_model_url.as_deref() {
        Some(url) => {
            let resolved = ensure_model(
                &http,
                url,
                cfg.classifier_sha256.as_deref(),
                &cfg.model_cache_dir,
            )
            .await
            .with_context(|| "classifier model resolution failed")?;
            model_loader::load_session(&resolved.path).with_context(|| {
                format!("classifier model not loadable: {}", resolved.path.display())
            })?;
            tracing::info!(
                "classifier ready at {} (downloaded: {})",
                resolved.path.display(),
                resolved.downloaded
            );
            // Persist the initial-deployment state so /operator/classifier
            // shows accuracy even before (or without) any nightly retrain
            // swap. Only written when no state exists yet or the model file
            // changed: an accuracy validated by the retraining flow must not
            // be overwritten with a placeholder value.
            #[cfg(feature = "retraining")]
            {
                let needs_write = match crate::training::read_classifier_state(&cfg) {
                    Some(st) => st.active_onnx != resolved.path.to_string_lossy(),
                    None => true,
                };
                if needs_write {
                    // Accuracy unknown at deploy time (no ground truth here):
                    // record 0.0 so the operator can tell it apart from a
                    // retrain-validated value, until the first nightly run
                    // validates and overwrites it.
                    if let Err(e) = crate::training::write_classifier_state(
                        &cfg,
                        &resolved.path,
                        0.0,
                    ) {
                        tracing::warn!("cannot persist initial classifier state: {e:#}");
                    }
                }
            }
            Some(SessionPool::new(resolved.path, cfg.effective_concurrency()))
        }
        None => {
            tracing::warn!(
                "no initial classifier configured — ACTIVE cameras will blur every \
                 in-ROI detection until the first successful nightly retraining"
            );
            None
        }
    };

    let db = Db::open(&cfg.data_dir.join(DB_FILENAME))
        .await
        .context("open SQLite database")?;

    // Optional ACTIVE face-mask segmenter (MediaPipe Selfie, env
    // `MASK_SEGMENTER=mediapipe`): downloaded/verified like the other models.
    let segmenter_pool = match cfg.mask_segmenter {
        MaskSegmenter::Off => None,
        MaskSegmenter::Mediapipe => {
            let resolved = ensure_model(
                &http,
                &cfg.selfie_segmenter_url,
                cfg.selfie_segmenter_sha256.as_deref(),
                &cfg.model_cache_dir,
            )
            .await
            .with_context(|| "face-mask segmenter model resolution failed")?;
            model_loader::load_session(&resolved.path).with_context(|| {
                format!(
                    "face-mask segmenter not loadable: {}",
                    resolved.path.display()
                )
            })?;
            tracing::info!(
                "face-mask segmenter ready at {} (downloaded: {})",
                resolved.path.display(),
                resolved.downloaded
            );
            Some(SessionPool::new(resolved.path, cfg.effective_concurrency()))
        }
    };

    // Optional COCO person detector for the head-fallback path
    // (`HEAD_FALLBACK_ENABLED`): blurs the upper fraction of person boxes
    // when the face detector finds no faces in an ACTIVE frame.
    let coco_pool = if cfg.head_fallback_enabled {
        let resolved = ensure_model(
            &http,
            &cfg.coco_model_url,
            cfg.coco_sha256.as_deref(),
            &cfg.model_cache_dir,
        )
        .await
        .with_context(|| "COCO person detector model resolution failed")?;
        model_loader::load_session(&resolved.path).with_context(|| {
            format!("COCO person detector not loadable: {}", resolved.path.display())
        })?;
        tracing::info!(
            "COCO person detector ready at {} (downloaded: {})",
            resolved.path.display(),
            resolved.downloaded
        );
        Some(SessionPool::new(resolved.path, cfg.effective_concurrency()))
    } else {
        tracing::info!("head-fallback disabled (HEAD_FALLBACK_ENABLED not set)");
        None
    };

    let store = ModelStore::with_coco(
        SessionPool::new(detector.path.clone(), cfg.effective_concurrency()),
        classifier_pool,
        segmenter_pool,
        coco_pool,
    );

// Nightly background job: ROI finalization for expired LEARNING cameras,
    // then (optionally) classifier retraining (§5, §6).
    {
        let sched = runtime.clone();
        let db = db.clone();
        let store = store.clone();
        tokio::spawn(async move {
            background_loop(sched, db, store).await;
        });
    }

    // STORE-output retention: periodic cleanup of the anonymized ZIPs that
    // accumulate in DATA_DIR (age and/or total-size rules, env `RETENTION_*`).
    if cfg.retention_active() {
        let runtime = runtime.clone();
        tokio::spawn(async move {
            retention::retention_loop(runtime).await;
        });
    }

    let state = AppState {
        cfg: cfg.clone(),
        settings: runtime.clone(),
        db: db.clone(),
        worker: ZipProcessor::new(runtime.clone(), db, store),
        job_lock: Arc::new(Mutex::new(())),
        #[cfg(feature = "s3")]
        s3: None,
        #[cfg(any(feature = "queue", feature = "rabbitmq"))]
        queue_metrics: queue_metrics::QueueMetricsRegistry::default(),
    };

    // S3 backend (feature `s3`, env `S3_ENABLED=true`): instantiate the
    // client, fail-fast on the three buckets (credentials + endpoint), and
    // share the SAME ZipProcessor the HTTP flow uses — the per-image
    // semaphore and session pool are not duplicated.
    #[cfg(feature = "s3")]
    let state = {
        let mut state = state;
        if let Some(settings) = state.cfg.s3.as_ref() {
            let s3_client = S3Client::from_settings(settings)
                .await
                .context("S3 client initialization")?;
            s3_client
                .check_bucket(&settings.bucket_input)
                .await
                .context("S3 input bucket check")?;
            s3_client
                .check_bucket(&settings.bucket_output)
                .await
                .context("S3 output bucket check")?;
            s3_client
                .check_bucket(&settings.bucket_logs)
                .await
                .context("S3 logs bucket check")?;
            let webhook_state = if settings.webhook_allowed_hosts.is_empty() {
                "disabled".to_string()
            } else {
                settings.webhook_allowed_hosts.join(",")
            };
            tracing::info!(
                "S3 backend ready: input={} output={} logs={} (jobs ≤ {}, webhooks {})",
                settings.bucket_input,
                settings.bucket_output,
                settings.bucket_logs,
                settings.max_concurrent_jobs,
                webhook_state
            );
            state.s3 = Some(S3ZipWorker::new(
                state.worker.clone(),
                s3_client,
                std::sync::Arc::new(settings.clone()),
                state.db.clone(),
            ));
        }
        state
    };

    // Async queue consumers (features `queue` = SQS, `rabbitmq` = RabbitMQ):
    // they share the SAME S3 worker (buckets, semaphores, pipeline), so job
    // concurrency stays bounded across all ingress paths.
    #[cfg(feature = "queue")]
    if let Some(sqs) = state.cfg.sqs.clone() {
        if let Some(worker) = state.s3.clone() {
            let settings = std::sync::Arc::new(sqs);
            let metrics = state.queue_metrics.entry("sqs");
            tokio::spawn(async move {
                if let Err(e) = queue_sqs::run_consumer(worker, settings, metrics).await {
                    tracing::error!("SQS consumer exited: {e:#}");
                }
            });
        } else {
            tracing::warn!(
                "SQS_ENABLED=true but the S3 backend is disabled — SQS consumer not started"
            );
        }
    }

    #[cfg(feature = "rabbitmq")]
    if let Some(rabbitmq) = state.cfg.rabbitmq.clone() {
        if let Some(worker) = state.s3.clone() {
            let settings = std::sync::Arc::new(rabbitmq);
            let metrics = state.queue_metrics.entry("rabbitmq");
            tokio::spawn(async move {
                if let Err(e) = queue_rabbitmq::run_consumer(worker, settings, metrics).await {
                    tracing::error!("RabbitMQ consumer exited: {e:#}");
                }
            });
        } else {
            tracing::warn!(
                "RABBITMQ_ENABLED=true but the S3 backend is disabled — RabbitMQ consumer not started"
            );
        }
    }

    let body_limit = state.cfg.body_limit_bytes;
    let bind_addr = state.cfg.bind_addr.clone();
    let router = Router::new()
        .route("/health", get(health))
        // /anonymize: single archive, bounded by BODY_LIMIT_BYTES.
        .route(
            "/anonymize",
            post(anonymize).layer(DefaultBodyLimit::max(body_limit)),
        )
        // /anonymize/batch: one or more archives in the same multipart form
        // (or a chunked upload); each archive is spooled to disk and processed
        // in sequence, so the total volume is not limited by BODY_LIMIT_BYTES.
        .route(
            "/anonymize/batch",
            post(anonymize_batch).layer(DefaultBodyLimit::disable()),
        )
        .route("/operator/cameras", get(op_cameras))
        .route("/operator/cameras/:id", get(op_camera))
        .route("/operator/cameras/:id/reset", post(op_reset))
        .route("/operator/cameras/:id/roi", post(op_set_roi))
        .route("/operator/retrain-audit", get(op_retrain_audit))
        .route("/operator/classifier", get(op_classifier))
        .route("/operator/gpu", get(op_gpu))
        .route("/operator/jobs", get(op_jobs))
        .route("/operator/settings", get(op_settings_page))
        .route(
            "/operator/settings.json",
            get(op_settings_json).post(op_settings_post),
        );

    #[cfg(any(feature = "queue", feature = "rabbitmq"))]
    let router = router.route("/operator/queues", get(op_queues));

    #[cfg(feature = "s3")]
    let router = if state.s3.is_some() {
        router
            .route("/anonymize/s3", post(anonymize_s3))
            .route("/status/:job_id", get(status_s3))
            .route("/operator/s3/sweep", post(op_s3_sweep))
    } else {
        router
    };

    let app = router.with_state(state);

    let listener = tokio::net::TcpListener::bind(&bind_addr)
        .await
        .with_context(|| format!("bind {bind_addr}"))?;
    tracing::info!("listening on http://{bind_addr}");
    axum::serve(listener, app)
        .await
        .context("HTTP server error")
}

// ─── Handlers ───────────────────────────────────────────────────────────────

async fn health() -> impl IntoResponse {
    Json(serde_json::json!({ "status": "ok" }))
}

// ─── S3 async ingestion (feature `s3`, spec §8 "Scenario S3") ────────────────

/// `POST /anonymize/s3` — validates the request, checks the input object
/// exists, and queues an asynchronous job (the response only accepts it; the
/// completion signal is `/status/:job_id` polling or the optional webhook).
#[cfg(feature = "s3")]
#[derive(Debug, Deserialize)]
struct S3AnonymizeRequest {
    /// Object key of the archive in the input bucket (`.zip`/`.7z`/`.rar`).
    input_key: String,
    /// Optional destination key in the output bucket; defaults to
    /// `elaborati/<input>_elaborato.zip`.
    output_key: Option<String>,
    /// Optional completion webhook; the host must be on
    /// `S3_WEBHOOK_ALLOWED_HOSTS` (anti-SSRF guard).
    callback_url: Option<String>,
}

#[cfg(feature = "s3")]
async fn anonymize_s3(
    State(state): State<AppState>,
    Json(payload): Json<S3AnonymizeRequest>,
) -> Response {
    let Some(worker) = state.s3.as_ref() else {
        // Route is registered only when S3_ENABLED=true; defense in depth.
        return json_err(
            StatusCode::NOT_FOUND,
            "S3 backend is disabled (S3_ENABLED=false or build without --features s3)",
        );
    };

    let input_key = payload.input_key.trim().to_string();
    if input_key.is_empty() || input_key.starts_with('/') {
        return json_err(StatusCode::BAD_REQUEST, "input_key must be a non-empty object key");
    }
    let output_key = match payload.output_key.as_deref() {
        Some(k) => {
            let k = k.trim().to_string();
            if k.is_empty() || k.starts_with('/') {
                return json_err(
                    StatusCode::BAD_REQUEST,
                    "output_key must be a non-empty object key",
                );
            }
            k
        }
        None => default_output_key(&input_key),
    };

    // Webhook guard: allowlist (host[:port]) or 400 at submit time — never
    // contact an arbitrary endpoint from the job task.
    if let Some(url) = payload.callback_url.as_deref() {
        if !webhook_host_allowed(&worker.settings, url) {
            return json_err(
                StatusCode::BAD_REQUEST,
                format!(
                    "callback_url {url} is not on the S3_WEBHOOK_ALLOWED_HOSTS allowlist \
                     (set the env var to enable webhooks)"
                ),
            );
        }
    }

    // Fail-fast existence check on the input object.
    match worker.s3.object_exists(&worker.s3.bucket_input, &input_key).await {
        Ok(true) => {}
        Ok(false) => {
            return json_err(
                StatusCode::NOT_FOUND,
                format!("s3://{}/{} not found", worker.s3.bucket_input, input_key),
            );
        }
        Err(e) => return json_err(StatusCode::INTERNAL_SERVER_ERROR, format!("S3 check failed: {e}")),
    }

    let job_id = uuid::Uuid::new_v4().to_string();
    worker
        .tracker
        .submit(job_id.clone(), input_key.clone(), output_key.clone())
        .await;

    let input = format!("s3://{}/{}", worker.s3.bucket_input, input_key);
    let output = format!("s3://{}/{}", worker.s3.bucket_output, output_key);

    let job_worker = worker.clone();
    let callback_url = payload.callback_url.clone();
    let job = (job_id.clone(), input_key.clone(), output_key.clone());
    tokio::spawn(async move {
        let (jid, ik, ok) = job;
        let _ = job_worker
            .run_job(
                &jid,
                &ik,
                &ok,
                callback_url.as_deref(),
                crate::zip_worker_s3::S3JobPolicy {
                    delete_input_on_success: false,
                },
            )
            .await;
    });

    Json(serde_json::json!({
        "status": "accepted",
        "job_id": job_id,
        "input": input,
        "output": output,
        "check_status_at": format!("/status/{job_id}"),
    }))
    .into_response()
}

/// `GET /status/:job_id` — in-memory async S3 job status.
#[cfg(feature = "s3")]
async fn status_s3(State(state): State<AppState>, Path(job_id): Path<String>) -> Response {
    let Some(worker) = state.s3.as_ref() else {
        return json_err(StatusCode::NOT_FOUND, "S3 backend is disabled");
    };
    match worker.tracker.get(&job_id).await {
        Some(status) => Json(status).into_response(),
        None => json_err(StatusCode::NOT_FOUND, format!("job {job_id} not found")),
    }
}

/// `POST /operator/s3/sweep` — operator batch sweep (feature `s3`): lists the
/// input bucket under an optional prefix and submits every eligible archive
/// (`.zip`/`.7z`/`.rar`, not already under `errori/`) as a tracked background
/// job. Jobs run through the shared S3 semaphore, so concurrency stays bounded
/// even for a large sweep.
#[cfg(feature = "s3")]
#[derive(Debug, Deserialize)]
struct S3SweepRequest {
    #[serde(default)]
    prefix: String,
    /// Max objects to list (default 50, clamped 1..=500).
    #[serde(default = "default_sweep_max_files")]
    max_files: usize,
    /// Delete the input object after a successful processing (default true).
    #[serde(default = "default_sweep_delete")]
    delete_input_on_success: bool,
}

#[cfg(feature = "s3")]
fn default_sweep_max_files() -> usize {
    50
}

#[cfg(feature = "s3")]
fn default_sweep_delete() -> bool {
    true
}

#[cfg(feature = "s3")]
async fn op_s3_sweep(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(payload): Json<S3SweepRequest>,
) -> Response {
    if let Err(resp) = operator_authorized(&state, &headers) {
        return resp;
    }
    let Some(worker) = state.s3.as_ref() else {
        return json_err(StatusCode::NOT_FOUND, "S3 backend is disabled");
    };
    let prefix = payload.prefix.trim().trim_start_matches('/').to_string();
    let max_files = payload.max_files.clamp(1, 500);
    match worker
        .submit_batch(&prefix, max_files, payload.delete_input_on_success)
        .await
    {
        Ok(jobs) => Json(serde_json::json!({
            "status": "accepted",
            "submitted": jobs.len(),
            "jobs": jobs,
        }))
        .into_response(),
        Err(e) => json_err(StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}")),
    }
}

/// `POST /anonymize` — single-job archive ingestion (§2: .zip/.7z/.rar).
/// The archive field is spooled to disk (never held in RAM), processed, and
/// the on-disk output ZIP is streamed back with
/// `X-Processing-Errors: <count>` and, when entries failed, a
/// `X-Processing-Errors-Detail` header (percent-encoded JSON list).
async fn anonymize(State(state): State<AppState>, mut multipart: Multipart) -> Response {
    // Single-job lock: reject concurrent uploads with 429 (§2).
    let Ok(_guard) = state.job_lock.try_lock() else {
        return json_err(
            StatusCode::TOO_MANY_REQUESTS,
            "another anonymization job is already running — retry later",
        );
    };

    let started_at = chrono::Utc::now();
    let Some((input_name, spool)) =
        spool_next_archive_field(&mut multipart, &state.cfg.data_dir, "upload").await
    else {
        return json_err(
            StatusCode::BAD_REQUEST,
            "no archive file found in multipart form (expected field 'file' with a .zip/.7z/.rar)",
        );
    };

    let outcome = match state.worker.process_archive_file(&input_name, &spool).await {
        Ok(out) => out,
        Err(e) => {
            tracing::error!("job '{input_name}' rejected: {e:#}");
            let _ = tokio::fs::remove_file(&spool).await;
            return json_err(StatusCode::BAD_REQUEST, format!("invalid upload: {e}"));
        }
    };
    // The spooled upload is transient; the anonymized output already lives
    // under DATA_DIR (spec §8 "scrittura output STORE").
    let _ = tokio::fs::remove_file(&spool).await;

    for summary in &outcome.camera_summaries {
        tracing::info!(
            "job '{input_name}': camera {} ({:?} branch) — {} image(s), {} detection(s) \
             stored, {} FP crop(s)",
            summary.camera_id,
            summary.branch,
            summary.images,
            summary.detections_stored,
            summary.fp_crops_saved
        );
    }
    tracing::info!(
        "job '{input_name}' finished: {} images, {} errors, archive {} ({} bytes)",
        outcome.processed_count,
        outcome.error_count,
        outcome.output_path.display(),
        outcome.output_size
    );
    // Operator ledger (DATA_DIR/jobs.jsonl) → GET /operator/jobs.
    if let Err(e) = record_job(&state.cfg.data_dir, &input_name, &outcome, started_at) {
        tracing::warn!("cannot record job '{input_name}' in the ledger: {e}");
    }
    respond_with_archive(&outcome).await
}

/// `POST /anonymize/batch` — multiple archives in one request (multipart with
/// repeated `file` fields, or a chunked upload). Each archive is spooled to
/// disk and processed **in sequence** under the single-job lock, so the total
/// volume is not limited by `BODY_LIMIT_BYTES` (the body limit is disabled on
/// this route; memory stays bounded because neither inputs nor outputs are
/// ever buffered in RAM). With a single archive the response is identical to
/// `/anonymize`; with several, the per-archive output ZIPs are merged into one
/// combined ZIP (`batch_elaborato.zip`), streaming entry-by-entry.
async fn anonymize_batch(State(state): State<AppState>, mut multipart: Multipart) -> Response {
    let Ok(_guard) = state.job_lock.try_lock() else {
        return json_err(
            StatusCode::TOO_MANY_REQUESTS,
            "another anonymization job is already running — retry later",
        );
    };

    let started_at = chrono::Utc::now();
    let mut outcomes: Vec<ZipJobOutcome> = Vec::new();
    // Archives that failed before producing any output (invalid format…):
    // reported in the headers and written to `batch_error.txt` in the merge.
    let mut hard_errors: Vec<ArchiveEntryError> = Vec::new();
    let mut archive_count = 0usize;
    let mut archive_names: Vec<String> = Vec::new();
    let mut spools: Vec<PathBuf> = Vec::new();

    loop {
        let Some((name, spool)) =
            spool_next_archive_field(&mut multipart, &state.cfg.data_dir, "batch").await
        else {
            break;
        };
        archive_count += 1;
        archive_names.push(name.clone());
        spools.push(spool.clone());
        match state.worker.process_archive_file(&name, &spool).await {
            Ok(outcome) => outcomes.push(outcome),
            Err(e) => {
                tracing::error!("batch job '{name}' rejected: {e:#}");
                hard_errors.push(ArchiveEntryError {
                    entry: name,
                    message: format!("invalid upload: {e}"),
                });
            }
        }
    }
    for s in &spools {
        let _ = tokio::fs::remove_file(s).await;
    }

    if archive_count == 0 {
        return json_err(
            StatusCode::BAD_REQUEST,
            "no archive file found in multipart form (expected at least one field 'file' with a .zip/.7z/.rar)",
        );
    }
    if outcomes.is_empty() {
        return json_err(
            StatusCode::BAD_REQUEST,
            format!(
                "none of the {} uploaded archive(s) could be processed ({} rejected)",
                archive_count,
                hard_errors.len()
            ),
        );
    }

    let response_outcome = if outcomes.len() == 1 && hard_errors.is_empty() {
        // Single healthy archive: behave exactly like /anonymize.
        outcomes.pop().expect("len == 1")
    } else {
        // Merge the per-archive outputs into one combined ZIP (streaming).
        let combined_name = "batch_elaborato.zip".to_string();
        let paths: Vec<PathBuf> = outcomes.iter().map(|o| o.output_path.clone()).collect();
        let (merged_path, merged_size) = match state
            .worker
            .merge_zips(&combined_name, &paths, &hard_errors)
            .await
        {
            Ok(v) => v,
            Err(e) => {
                tracing::error!("batch merge failed: {e:#}");
                return json_err(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("cannot merge batch outputs: {e}"),
                );
            }
        };
        // The merged archive is the deliverable; drop the per-archive STORE
        // copies to avoid duplicating gigabytes on disk.
        for o in &outcomes {
            let _ = tokio::fs::remove_file(&o.output_path).await;
        }
        let mut errors: Vec<ArchiveEntryError> = Vec::new();
        for o in &outcomes {
            errors.extend(o.errors.iter().cloned());
        }
        errors.extend(hard_errors.iter().cloned());
        ZipJobOutcome {
            output_path: merged_path,
            output_size: merged_size,
            output_name: combined_name,
            error_count: outcomes.iter().map(|o| o.error_count).sum::<usize>() + hard_errors.len(),
            processed_count: outcomes.iter().map(|o| o.processed_count).sum(),
            camera_summaries: outcomes
                .iter()
                .flat_map(|o| o.camera_summaries.clone())
                .collect(),
            errors,
        }
    };

    for summary in &response_outcome.camera_summaries {
        tracing::info!(
            "batch: camera {} ({:?} branch) — {} image(s), {} detection(s) stored, \
             {} FP crop(s)",
            summary.camera_id,
            summary.branch,
            summary.images,
            summary.detections_stored,
            summary.fp_crops_saved
        );
    }
    tracing::info!(
        "batch finished: {archive_count} archive(s), {} images, {} errors, archive {} ({} bytes)",
        response_outcome.processed_count,
        response_outcome.error_count,
        response_outcome.output_path.display(),
        response_outcome.output_size
    );
    if let Err(e) = record_job(
        &state.cfg.data_dir,
        &archive_names.join(", "),
        &response_outcome,
        started_at,
    ) {
        tracing::warn!("cannot record batch job in the ledger: {e}");
    }
    respond_with_archive(&response_outcome).await
}

/// Streams an on-disk output ZIP back to the client with the standard headers
/// (content-type, content-disposition, `X-Processing-Errors`,
/// `X-Processing-Errors-Detail`, content-length). The body is a `ReaderStream`
/// over the file, so memory stays bounded regardless of output size.
async fn respond_with_archive(outcome: &ZipJobOutcome) -> Response {
    let mut headers = outcome_headers(outcome);
    match tokio::fs::File::open(&outcome.output_path).await {
        Ok(file) => {
            headers.insert(
                header::CONTENT_LENGTH,
                outcome
                    .output_size
                    .to_string()
                    .parse()
                    .expect("digits are a valid header"),
            );
            let body = Body::from_stream(ReaderStream::new(file));
            (StatusCode::OK, headers, body).into_response()
        }
        Err(e) => {
            tracing::error!(
                "cannot open output archive {}: {e}",
                outcome.output_path.display()
            );
            json_err(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("output archive unavailable: {e}"),
            )
        }
    }
}

/// The shared response headers for an archive outcome.
fn outcome_headers(outcome: &ZipJobOutcome) -> HeaderMap {
    let disposition = format!(
        "attachment; filename=\"{}\"",
        outcome.output_name.replace('"', "")
    );
    let mut headers = HeaderMap::new();
    headers.insert(
        header::CONTENT_TYPE,
        "application/zip".parse().expect("static header"),
    );
    headers.insert(
        header::CONTENT_DISPOSITION,
        disposition.parse().expect("static header"),
    );
    headers.insert(
        header::HeaderName::from_static("x-processing-errors"),
        outcome
            .error_count
            .to_string()
            .parse()
            .expect("digits are a valid header"),
    );
    // Per-entry failures as a percent-encoded JSON array (first 50 entries,
    // so the header stays small): `[{"entry":"...","error":"..."},…]`.
    if !outcome.errors.is_empty() {
        let detail = serde_json::json!(outcome
            .errors
            .iter()
            .take(50)
            .map(|e| serde_json::json!({ "entry": e.entry, "error": e.message }))
            .collect::<Vec<_>>());
        headers.insert(
            header::HeaderName::from_static("x-processing-errors-detail"),
            percent_encode(&detail.to_string())
                .parse()
                .expect("percent-encoded header"),
        );
    }
    headers
}

/// Reads the *next* archive field of a multipart form (any field named `file`
/// or whose file name ends in .zip/.7z/.rar), spooling it to a temp file
/// under `DATA_DIR` so uploads never accumulate in RAM. Returns
/// `(upload name, spool path)` or `None` when the form is exhausted.
async fn spool_next_archive_field(
    multipart: &mut Multipart,
    data_dir: &std::path::Path,
    tag: &str,
) -> Option<(String, PathBuf)> {
    while let Ok(Some(field)) = multipart.next_field().await {
        if !is_archive_field(&field) {
            continue;
        }
        let name = field
            .file_name()
            .map(|s| s.to_string())
            .unwrap_or_else(|| "upload.zip".to_string());
        let stamp = chrono::Utc::now().format("%Y%m%d_%H%M%S%3f");
        let spool = data_dir.join(sanitize_filename(&format!("{tag}_{stamp}_{name}")));
        match spool_field(field, &spool).await {
            Ok(()) => return Some((name, spool)),
            Err(e) => {
                tracing::error!("cannot spool upload field: {e}");
                let _ = tokio::fs::remove_file(&spool).await;
            }
        }
    }
    None
}

/// True for the archive-bearing multipart fields (spec §2: field `file`, or
/// any field whose file name ends in .zip/.7z/.rar).
fn is_archive_field(field: &Field<'_>) -> bool {
    let name = field.file_name().map(|s| s.to_string());
    matches!(
        &name,
        Some(n) if matches!(
            n.to_ascii_lowercase().as_str(),
            s if s.ends_with(".zip") || s.ends_with(".7z") || s.ends_with(".rar")
        )
    ) || field.name() == Some("file")
}

/// Streams one multipart field's content into `path` (disk spool).
async fn spool_field(mut field: Field<'_>, path: &std::path::Path) -> Result<()> {
    let mut f = tokio::fs::File::create(path)
        .await
        .with_context(|| format!("create spool file {}", path.display()))?;
    while let Ok(Some(chunk)) = field.chunk().await {
        f.write_all(&chunk).await?;
    }
    f.flush().await?;
    Ok(())
}

/// Percent-encodes a header value so non-ASCII / quote bytes never break the
/// HTTP header line. Clients decode with `decodeURIComponent`.
fn percent_encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'~') {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

async fn op_cameras(State(state): State<AppState>, headers: HeaderMap) -> Response {
    if let Err(resp) = operator_authorized(&state, &headers) {
        return resp;
    }
    match state.db.all_cameras().await {
        Ok(cams) => {
            Json(cams.into_iter().map(CameraView::from).collect::<Vec<_>>()).into_response()
        }
        Err(e) => json_err(StatusCode::INTERNAL_SERVER_ERROR, format!("db error: {e}")),
    }
}

async fn op_camera(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> Response {
    if let Err(resp) = operator_authorized(&state, &headers) {
        return resp;
    }
    match state.db.camera_by_id(&id).await {
        Ok(Some(cam)) => Json(CameraView::from(cam)).into_response(),
        Ok(None) => json_err(StatusCode::NOT_FOUND, format!("camera {id} not found")),
        Err(e) => json_err(StatusCode::INTERNAL_SERVER_ERROR, format!("db error: {e}")),
    }
}

/// `GET /operator/retrain-audit` — last nightly retraining audit record
/// (status, accuracy, samples, swap/reject outcome), written by the
/// retraining job to `DATA_DIR/retrain_audit.json`.
async fn op_retrain_audit(State(state): State<AppState>, headers: HeaderMap) -> Response {
    if let Err(resp) = operator_authorized(&state, &headers) {
        return resp;
    }
    let path = state
        .cfg
        .data_dir
        .join(crate::config::RETRAIN_AUDIT_FILENAME);
    match tokio::fs::read_to_string(&path).await {
        Ok(text) => (
            StatusCode::OK,
            [(header::CONTENT_TYPE, "application/json")],
            text,
        )
            .into_response(),
        Err(_) => json_err(
            StatusCode::NOT_FOUND,
            "no retraining audit recorded yet (the nightly retraining has not run)",
        ),
    }
}

/// `GET /operator/classifier` — current state of the binary face classifier
/// used as the ACTIVE-branch second check (spec §4): whether it is loaded
/// (i.e. whether the second gate is actually running), the configured source
/// URL, the persisted accuracy + last swap from `classifier_state.json` and
/// the live inference counters. `loaded=false` means ACTIVE blurs every
/// in-ROI detection (fail-safe, no classifier filtering).
async fn op_classifier(State(state): State<AppState>, headers: HeaderMap) -> Response {
    if let Err(resp) = operator_authorized(&state, &headers) {
        return resp;
    }
    let pool = state.worker.store().classifier_pool();
    #[allow(unused_mut)]
    let mut persisted: Option<serde_json::Value> = None;
    #[cfg(feature = "retraining")]
    if let Some(st) = crate::training::read_classifier_state(&state.cfg) {
        persisted = Some(serde_json::json!({
            "active_onnx": st.active_onnx,
            "accuracy": st.accuracy,
            "updated_at": st.updated_at,
        }));
    }
    let (mut inference_count, mut inference_avg_ms) = (0u64, 0.0f64);
    for (stage, count, avg_ms) in crate::models::inference_stats() {
        if stage == "classifier" {
            inference_count = count;
            inference_avg_ms = avg_ms;
        }
    }
    // A classifier counts as loaded only when it is usable right now; a
    // persisted state alone means it was swapped in a previous process run.
    let active_path = pool.as_ref().map(|p| p.path().to_string_lossy().to_string());
    Json(serde_json::json!({
        "loaded": pool.is_some(),
        "active_check_enabled": pool.is_some(),
        "configured_url": state.cfg.classifier_model_url,
        "active_onnx": active_path,
        "persisted": persisted,
        "inference": {
            "count": inference_count,
            "avg_ms": inference_avg_ms,
        },
        "min_accuracy_for_swap": state.cfg.retrain_min_accuracy,
    }))
    .into_response()
}

/// `GET /operator/gpu` — current inference execution-provider configuration
/// plus measured per-stage inference statistics (`inference.*.count` /
/// `avg_ms`, process-lifetime over every processed image). `device_name` is a
/// best-effort `nvidia-smi` query (cached 60 s); it is `null` when the query
/// fails (no NVIDIA driver / CPU-only host).
async fn op_gpu(State(state): State<AppState>, headers: HeaderMap) -> Response {
    if let Err(resp) = operator_authorized(&state, &headers) {
        return resp;
    }
    let mut inference = serde_json::Map::new();
    for (stage, count, avg_ms) in crate::models::inference_stats() {
        inference.insert(
            stage.to_string(),
            serde_json::json!({ "count": count, "avg_ms": avg_ms }),
        );
    }
    Json(serde_json::json!({
        "execution_provider": state.cfg.execution_provider.as_str(),
        "device_id": state.cfg.gpu_device_id,
        "gpu_memory_limit_bytes": state.cfg.gpu_memory_limit_bytes,
        "enable_tf32": state.cfg.enable_tf32,
        "enable_fp16": state.cfg.enable_fp16,
        "device_name": gpu_device_name(),
        "inference": inference,
    }))
    .into_response()
}

/// Best-effort GPU device name via `nvidia-smi`, cached for 60 seconds.
/// Returns `None` (no UTF-8 output, no driver, missing binary) and never
/// blocks the request for long — this is purely informational.
fn gpu_device_name() -> Option<String> {
    use std::sync::Mutex;
    static CACHE: Mutex<Option<(std::time::Instant, String)>> = Mutex::new(None);
    if let Ok(mut cache) = CACHE.lock() {
        if let Some((at, name)) = cache.as_ref() {
            if at.elapsed() < Duration::from_secs(60) {
                return Some(name.clone());
            }
        }
        let name = std::process::Command::new("nvidia-smi")
            .args(["--query-gpu=name", "--format=csv,noheader,noenv"])
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
            .filter(|s| !s.is_empty());
        if let Some(name) = name.clone() {
            *cache = Some((std::time::Instant::now(), name));
        }
        return name;
    }
    None
}

/// `GET /operator/jobs` — the most recent processed jobs, read from the
/// `DATA_DIR/jobs.jsonl` ledger written by the `/anonymize` handlers, plus
/// any orphan `*_elaborato.zip` outputs still on disk. Query param `limit`
/// (default 20, max 500). Each job reports the archive, processed images,
/// errors (count + first per-entry failures), output size and timestamps.
async fn op_jobs(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(params): Query<JobsQuery>,
) -> Response {
    if let Err(resp) = operator_authorized(&state, &headers) {
        return resp;
    }
    let limit = params.limit.unwrap_or(20).clamp(1, 500);

    // Ledger (file order = chronological) → newest first, capped by limit.
    let mut jobs: Vec<JobLedgerEntry> = Vec::new();
    let mut ledger_outputs: std::collections::HashSet<String> = std::collections::HashSet::new();
    match read_jobs_ledger(&state.cfg.data_dir) {
        Ok(entries) => {
            for mut entry in entries {
                ledger_outputs.insert(entry.output.clone());
                entry.on_disk = state.cfg.data_dir.join(&entry.output).exists();
                jobs.push(entry);
            }
        }
        Err(e) => {
            // Missing ledger is fine (no jobs yet); anything else is logged.
            tracing::warn!("cannot read jobs ledger: {e}");
        }
    }
    let total_jobs = jobs.len();
    jobs.reverse();
    jobs.truncate(limit);

    // Orphan outputs: on-disk STORE ZIPs with no ledger entry (e.g. created
    // before the ledger existed, or by a crashed job).
    let mut orphans: Vec<OrphanOutput> = Vec::new();
    if let Ok(mut rd) = tokio::fs::read_dir(&state.cfg.data_dir).await {
        while let Ok(Some(entry)) = rd.next_entry().await {
            let name = entry.file_name().to_string_lossy().into_owned();
            if !name.ends_with("_elaborato.zip") || ledger_outputs.contains(&name) {
                continue;
            }
            if let Ok(meta) = entry.metadata().await {
                orphans.push(OrphanOutput {
                    file: name,
                    size_bytes: meta.len(),
                    modified_at: chrono::DateTime::<Utc>::from(
                        meta.modified().unwrap_or(std::time::SystemTime::UNIX_EPOCH),
                    )
                    .to_rfc3339(),
                });
            }
        }
    }
    orphans.sort_by(|a, b| b.modified_at.cmp(&a.modified_at));

    Json(serde_json::json!({
        "jobs": jobs,
        "orphan_outputs": orphans,
        "total_jobs": total_jobs,
    }))
    .into_response()
}

/// `GET /operator/queues` — live counters of the async queue consumers
/// (features `queue` / `rabbitmq`): received, completed, failed, sent to DLQ,
/// and the best-effort DLQ depth. Gated by `X-Operator-Key` like the other
/// operator routes.
#[cfg(any(feature = "queue", feature = "rabbitmq"))]
async fn op_queues(State(state): State<AppState>, headers: HeaderMap) -> Response {
    if let Err(resp) = operator_authorized(&state, &headers) {
        return resp;
    }
    let queues: Vec<serde_json::Value> = state
        .queue_metrics
        .snapshot_all()
        .into_iter()
        .map(|(name, snap)| {
            serde_json::json!({
                "queue": name,
                "received": snap.received,
                "completed": snap.completed,
                "failed": snap.failed,
                "dlq": snap.dlq,
                "dlq_depth": snap.dlq_depth,
            })
        })
        .collect();
    Json(serde_json::json!({ "queues": queues })).into_response()
}

/// `GET /operator/settings` — the operator settings UI. The page itself is an
/// inert static template (it contains no data), but the JSON endpoints it
/// talks to are gated by `X-Operator-Key` like every other operator route.
async fn op_settings_page() -> impl IntoResponse {
    axum::response::Html(SETTINGS_UI_HTML)
}

const SETTINGS_UI_HTML: &str = include_str!("settings_ui.html");

/// `GET /operator/settings.json` — schema + safe-effective/default values of
/// every hot-tunable knob, so the browser renders the form from data (no
/// server-side HTML template to keep in sync).
async fn op_settings_json(State(state): State<AppState>, headers: HeaderMap) -> Response {
    if let Err(resp) = operator_authorized(&state, &headers) {
        return resp;
    }
    Json(serde_json::json!({
        "fields": state.settings.field_specs(),
        "overrides_file": crate::config::RUNTIME_CONFIG_FILENAME,
    }))
    .into_response()
}

/// `POST /operator/settings.json` — body `{"set": {key: value, ...}}` applies
/// a patch (clamped + validated), `{"reset_all": true}` restores the plain
/// `.env` values. Each write is persisted to `DATA_DIR/runtime_config.json`.
#[derive(Debug, serde::Deserialize)]
struct SettingsPatch {
    set: Option<serde_json::Map<String, serde_json::Value>>,
    reset_all: Option<bool>,
}

async fn op_settings_post(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<SettingsPatch>,
) -> Response {
    if let Err(resp) = operator_authorized(&state, &headers) {
        return resp;
    }
    let result = if body.reset_all == Some(true) {
        state.settings.reset_to_env("", true)
    } else if let Some(patch) = body.set {
        state.settings.apply_patch(&patch)
    } else {
        return json_err(
            StatusCode::BAD_REQUEST,
            "body must contain 'set' or 'reset_all': true",
        );
    };
    match result {
        Ok(()) => Json(serde_json::json!({
            "fields": state.settings.field_specs(),
            "overrides_file": crate::config::RUNTIME_CONFIG_FILENAME,
        }))
        .into_response(),
        Err(e) => json_err(StatusCode::BAD_REQUEST, format!("{e:#}")),
    }
}

#[derive(Debug, serde::Deserialize)]
struct JobsQuery {
    limit: Option<usize>,
}

#[derive(Debug, serde::Serialize)]
struct OrphanOutput {
    file: String,
    size_bytes: u64,
    modified_at: String,
}

/// `POST /operator/cameras/:id/reset` — forced FSM regression (§4).
/// Body: `{"target_state": "INITIAL" | "LEARNING"}`.
async fn op_reset(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
    Json(body): Json<ResetRequest>,
) -> Response {
    if let Err(resp) = operator_authorized(&state, &headers) {
        return resp;
    }
    let updated = match body.target_state.as_str() {
        "INITIAL" => state.db.reset_to_initial(&id).await,
        "LEARNING" => state.db.reset_to_learning(&id).await,
        other => {
            return json_err(
                StatusCode::BAD_REQUEST,
                format!("target_state must be INITIAL or LEARNING, got '{other}'"),
            )
        }
    };
    match updated {
        Ok(true) => {
            tracing::info!("operator reset camera {id} to {}", body.target_state);
            json_err(
                StatusCode::OK,
                format!("camera {id} reset to {}", body.target_state),
            )
        }
        Ok(false) => json_err(StatusCode::NOT_FOUND, format!("camera {id} not found")),
        Err(e) => json_err(StatusCode::INTERNAL_SERVER_ERROR, format!("db error: {e}")),
    }
}

// ─── Operator auth + views ──────────────────────────────────────────────────

#[allow(clippy::result_large_err)] // axum Response on the error path
fn operator_authorized(state: &AppState, headers: &HeaderMap) -> Result<(), Response> {
    let Some(expected) = state.cfg.operator_api_key.as_deref() else {
        return Err(json_err(
            StatusCode::FORBIDDEN,
            "operator endpoints disabled (OPERATOR_API_KEY not configured)",
        ));
    };
    let provided = headers
        .get(OPERATOR_KEY_HEADER)
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default();
    if provided == expected {
        Ok(())
    } else {
        Err(json_err(StatusCode::FORBIDDEN, "invalid operator API key"))
    }
}

#[derive(Debug, Serialize)]
struct CameraView {
    id: String,
    state: String,
    learning_started_at: Option<String>,
    roi: Option<serde_json::Value>,
    frame_width: Option<u32>,
    frame_height: Option<u32>,
}

impl From<Camera> for CameraView {
    fn from(c: Camera) -> Self {
        CameraView {
            id: c.id,
            state: c.state.as_str().to_string(),
            learning_started_at: c.learning_started_at.map(|t| t.to_rfc3339()),
            roi: c.roi_json.and_then(|s| serde_json::from_str(&s).ok()),
            frame_width: c.frame_width,
            frame_height: c.frame_height,
        }
    }
}

#[derive(Debug, serde::Deserialize)]
struct ResetRequest {
    target_state: String,
}

/// Body of `POST /operator/cameras/:id/roi`: `{"type": "full"}` sets the
/// whole frame as the anonymization zone and activates the camera (bypasses
/// the extractor's area cap — for wide-angle / narrow-framed scenes where the
/// street fills the whole frame).
#[derive(Debug, serde::Deserialize)]
struct RoiRequest {
    r#type: String,
}

/// `POST /operator/cameras/:id/roi` — set a camera ROI without re-learning.
async fn op_set_roi(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
    Json(body): Json<RoiRequest>,
) -> Response {
    if let Err(resp) = operator_authorized(&state, &headers) {
        return resp;
    }
    let cam = match state.db.camera_by_id(&id).await {
        Ok(Some(c)) => c,
        Ok(None) => return json_err(StatusCode::NOT_FOUND, format!("camera {id} not found")),
        Err(e) => return json_err(StatusCode::INTERNAL_SERVER_ERROR, format!("db error: {e}")),
    };
    match body.r#type.as_str() {
        "full" => {
            let (Some(w), Some(h)) = (cam.frame_width, cam.frame_height) else {
                return json_err(
                    StatusCode::CONFLICT,
                    "camera has no frame geometry yet — upload a frame first",
                );
            };
            if w == 0 || h == 0 {
                return json_err(
                    StatusCode::CONFLICT,
                    "camera has no frame geometry yet — upload a frame first",
                );
            }
            let roi = crate::roi::RoiPolygon {
                polygon: vec![
                    [0.0, 0.0],
                    [w as f64 - 1.0, 0.0],
                    [w as f64 - 1.0, h as f64 - 1.0],
                    [0.0, h as f64 - 1.0],
                ],
                image_width: w,
                image_height: h,
                area_ratio: 1.0,
            };
            if let Err(e) = state.db.set_roi_and_activate(&id, &roi.to_json()).await {
                return json_err(StatusCode::INTERNAL_SERVER_ERROR, format!("db error: {e}"));
            }
            tracing::info!("operator set full-frame ROI for camera {id}");
            json_err(
                StatusCode::OK,
                format!("camera {id} ROI set to full frame and camera ACTIVE"),
            )
        }
        other => json_err(
            StatusCode::BAD_REQUEST,
            format!("roi type must be 'full', got '{other}'"),
        ),
    }
}

// ─── Helpers ────────────────────────────────────────────────────────────────

fn json_err(status: StatusCode, message: impl Into<String>) -> Response {
    (status, Json(serde_json::json!({ "error": message.into() }))).into_response()
}

/// Nightly scheduler (§5 ROI finalization + dynamic ROI, §6 retraining). Also
/// runs a catch-up pass at startup so cameras whose LEARNING window expired
/// while the service was down still transition promptly.
async fn background_loop(runtime: RuntimeConfig, db: Db, store: ModelStore) {
    tracing::info!("background scheduler started");
    let cfg = runtime.snapshot();
    finalize_expired_rois(&cfg, &db).await;
    reextract_active_rois(&cfg, &db).await;

    loop {
        let cfg = runtime.snapshot();
        let wait = cfg
            .retrain_schedule
            .next_occurrence_from(chrono::Local::now().time());
        tracing::info!("next nightly run in {}s", wait.as_secs());
        tokio::time::sleep(wait).await;
        tracing::info!("nightly run starting");
        let cfg = runtime.snapshot();
        finalize_expired_rois(&cfg, &db).await;
        reextract_active_rois(&cfg, &db).await;
        prune_old_detections(&cfg, &db).await;
        #[cfg(feature = "retraining")]
        {
            if let Err(e) = training::nightly_retrain(&cfg, &store).await {
                tracing::warn!("nightly retraining failed (keeping current classifier): {e}");
            }
        }
        #[cfg(not(feature = "retraining"))]
        {
            let _ = &store;
            tracing::warn!("retraining disabled (crate built without the 'retraining' feature)");
        }
    }
}

/// Spec §5.6: every LEARNING camera whose window has elapsed gets its ROI
/// extracted from the accumulated detection coordinates; success moves the
/// camera to ACTIVE, failure keeps it in LEARNING (anomaly logged for manual
/// review).
async fn finalize_expired_rois(cfg: &Config, db: &Db) {
    let Ok(cameras) = db.cameras_in_learning().await else {
        tracing::warn!("ROI finalization: cannot read cameras");
        return;
    };
    for cam in cameras {
        let Some(started) = cam.learning_started_at else {
            continue;
        };
        let elapsed = Utc::now() - started;
        let window = ChronoDuration::days(cfg.learning_days as i64);
        if elapsed < window {
            continue;
        }
        finalize_one_roi(cfg, db, &cam).await;
    }
}

async fn finalize_one_roi(cfg: &Config, db: &Db, cam: &Camera) {
    let cam_id = &cam.id;
    let Some(w) = cam.frame_width else {
        tracing::warn!(
            "camera {cam_id}: LEARNING window elapsed but frame geometry unknown; \
             staying in LEARNING (no frames seen?)"
        );
        return;
    };
    let Some(h) = cam.frame_height else {
        tracing::warn!("camera {cam_id}: missing frame height; staying in LEARNING");
        return;
    };

    let points = match db.detections_for_camera(cam_id).await {
        Ok(p) => p
            .into_iter()
            .map(|(x, y)| (x as f64, y as f64))
            .collect::<Vec<_>>(),
        Err(e) => {
            tracing::error!("camera {cam_id}: cannot read detections: {e}");
            return;
        }
    };

    let outcome = extract_roi(
        &points,
        w,
        h,
        cfg.roi_eps_px,
        cfg.roi_min_samples,
        cfg.roi_rdp_epsilon,
        cfg.roi_area_min,
        cfg.roi_area_max,
        cfg.roi_margin_pct,
    );

    match outcome {
        RoiOutcome::Polygon(roi) => {
            tracing::info!(
                "camera {cam_id}: ROI extracted ({:.1}% of frame, {} vertices) → ACTIVE",
                roi.area_ratio * 100.0,
                roi.polygon.len()
            );
            if let Err(e) = db.set_roi_and_activate(cam_id, &roi.to_json()).await {
                tracing::error!("camera {cam_id}: cannot persist ROI: {e}");
            }
        }
        RoiOutcome::Invalid(reason) => {
            // Spec §5.6: out-of-range ROI → camera stays in LEARNING longer,
            // anomaly logged for manual review.
            tracing::warn!(
                "camera {cam_id}: ROI validation failed ({reason}) — keeping camera in \
                 LEARNING for review"
            );
        }
        RoiOutcome::InsufficientData => {
            tracing::warn!(
                "camera {cam_id}: not enough detection data to extract a ROI — \
                 keeping camera in LEARNING"
            );
        }
    }
}

/// Dynamic ROI (ACTIVE cameras): re-extract the polygon every nightly pass
/// from the anonymized detections of the last `ROI_REEXTRACT_WINDOW_DAYS`,
/// adopting the candidate only when the stability guard approves
/// (`ROI_REEXTRACT_MIN_IOU` or a frame-geometry change). ACTIVE proto-ROIs
/// that cannot be re-extracted (no data / invalid) keep their deployed one.
async fn reextract_active_rois(cfg: &Config, db: &Db) {
    if !cfg.roi_reextract_enabled {
        return;
    }
    let Ok(cameras) = db.all_cameras().await else {
        tracing::warn!("dynamic ROI: cannot read cameras");
        return;
    };
    for cam in cameras {
        if cam.state != crate::db::CameraState::Active {
            continue;
        }
        reextract_one_roi(cfg, db, &cam).await;
    }
}

async fn reextract_one_roi(cfg: &Config, db: &Db, cam: &Camera) {
    let cam_id = &cam.id;
    let (Some(w), Some(h)) = (cam.frame_width, cam.frame_height) else {
        return;
    };
    let since = Utc::now() - ChronoDuration::days(cfg.roi_reextract_window_days as i64);
    let points = match db.detections_for_camera_since(cam_id, &since).await {
        Ok(p) => p
            .into_iter()
            .map(|(x, y)| (x as f64, y as f64))
            .collect::<Vec<_>>(),
        Err(e) => {
            tracing::error!("camera {cam_id}: dynamic ROI — cannot read detections: {e}");
            return;
        }
    };

    let candidate = match extract_roi(
        &points,
        w,
        h,
        cfg.roi_eps_px,
        cfg.roi_min_samples,
        cfg.roi_rdp_epsilon,
        cfg.roi_area_min,
        cfg.roi_area_max,
        cfg.roi_margin_pct,
    ) {
        RoiOutcome::Polygon(roi) => roi,
        RoiOutcome::Invalid(reason) => {
            tracing::debug!("camera {cam_id}: dynamic ROI candidate invalid ({reason}) — keep deployed");
            return;
        }
        RoiOutcome::InsufficientData => {
            tracing::debug!("camera {cam_id}: dynamic ROI — not enough fresh detections — keep deployed");
            return;
        }
    };

    let current = cam.roi_json.as_deref().and_then(RoiPolygon::from_json);
    match crate::roi::decide_roi(current.as_ref(), &candidate, cfg.roi_reextract_min_iou) {
        crate::roi::RoiDecision::KeepStable { iou } => {
            tracing::info!(
                "camera {cam_id}: dynamic ROI kept (IoU {iou:.3} ≥ {})",
                cfg.roi_reextract_min_iou
            );
        }
        crate::roi::RoiDecision::Replace => {
            tracing::info!(
                "camera {cam_id}: dynamic ROI replaced ({:.1}% of frame, {} vertices)",
                candidate.area_ratio * 100.0,
                candidate.polygon.len()
            );
            if let Err(e) = db.set_roi_and_activate(cam_id, &candidate.to_json()).await {
                tracing::error!("camera {cam_id}: dynamic ROI — cannot persist: {e}");
            }
        }
        crate::roi::RoiDecision::GeometryChanged => {
            let (ow, oh) = current
                .map(|c| (c.image_width, c.image_height))
                .unwrap_or((0, 0));
            tracing::info!(
                "camera {cam_id}: dynamic ROI re-adopted after frame geometry change \
                 ({ow}×{oh} → {}×{})",
                candidate.image_width,
                candidate.image_height
            );
            if let Err(e) = db.set_roi_and_activate(cam_id, &candidate.to_json()).await {
                tracing::error!("camera {cam_id}: dynamic ROI — cannot persist: {e}");
            }
        }
    }
}

/// Bounds the detections table: keeps the horizon needed by both the LEARNING
/// finalization (full `LEARNING_DAYS` history) and the dynamic-ROI window.
async fn prune_old_detections(cfg: &Config, db: &Db) {
    let horizon_days = cfg.learning_days.max(cfg.roi_reextract_window_days) as i64 + 1;
    let cutoff = Utc::now() - ChronoDuration::days(horizon_days);
    match db.prune_detections_older_than(&cutoff).await {
        Ok(removed) => {
            if removed > 0 {
                tracing::info!("pruned {removed} stale detections (horizon {horizon_days}d)");
            }
        }
        Err(e) => tracing::warn!("cannot prune stale detections: {e}"),
    }
}
