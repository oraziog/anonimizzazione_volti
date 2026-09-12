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
mod exif_camera;
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
#[cfg(test)]
mod testutil;
#[cfg(test)]
mod samples;
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
use crate::zip_worker::{
    sanitize_filename, ArchiveEntryError, CleanupPath, ZipJobOutcome, ZipProcessor,
};

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
    // The client bounds redirects; `ensure_model` additionally enforces the
    // https / SHA policy from the configuration.
    let model_policy = model_loader::ModelPolicy {
        allow_http: cfg.model_allow_http,
        sha_required: cfg.model_sha_required,
        max_redirects: cfg.model_max_redirects,
    };
    let http = model_loader::build_client(&model_policy).context("build HTTP client")?;

    let detector = ensure_model(
        &http,
        &cfg.yolo_model_url,
        cfg.yolo_sha256.as_deref(),
        &cfg.model_cache_dir,
        &model_policy,
    )
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
                &model_policy,
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
            // No CLASSIFIER_MODEL_URL configured: restore the last classifier
            // swapped by the nightly retraining from `classifier_state.json`.
            // The swap path validates the ONNX with `ort` before persisting,
            // so the restored file is known-loadable (re-checked defensively).
            #[cfg(feature = "retraining")]
            let restored = crate::training::read_classifier_state(&cfg).and_then(|st| {
                let p = PathBuf::from(&st.active_onnx);
                if !p.exists() {
                    tracing::warn!(
                        "persisted classifier {} no longer exists; ignoring",
                        p.display()
                    );
                    return None;
                }
                match model_loader::load_session(&p) {
                    Ok(_) => Some(p),
                    Err(e) => {
                        tracing::warn!(
                            "persisted classifier {} not loadable: {e:#}; ignoring",
                            p.display()
                        );
                        None
                    }
                }
            });
            #[cfg(not(feature = "retraining"))]
            let restored: Option<PathBuf> = None;
            match restored {
                Some(p) => {
                    tracing::info!(
                        "classifier restored from persisted state: {} (no CLASSIFIER_MODEL_URL configured)",
                        p.display()
                    );
                    Some(SessionPool::new(p, cfg.effective_concurrency()))
                }
                None => {
                    tracing::warn!(
                        "no initial classifier configured — ACTIVE cameras will blur every \
                         in-ROI detection until the first successful nightly retraining"
                    );
                    None
                }
            }
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
                &model_policy,
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
            &model_policy,
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

    // Last layer on purpose: hardening wraps every route (security headers).
    // The request deadline is NOT applied here: it belongs to the upload phase
    // only (see `IngestDeadline`), so a long job is never interrupted mid-way.
    let app = router
        .layer(axum::middleware::from_fn(harden))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(&bind_addr)
        .await
        .with_context(|| format!("bind {bind_addr}"))?;
    tracing::info!("listening on http://{bind_addr}");
    axum::serve(listener, app)
        .await
        .context("HTTP server error")
}

/// Per-request hardening applied to every route: the response headers a
/// browser-facing service should always send, plus a CSP tuned for the
/// self-contained operator settings page (inline script/style, same-origin
/// fetch only — no framing, no external origins).
///
/// The `REQUEST_TIMEOUT_SECS` deadline is deliberately **not** applied here:
/// wrapping the whole handler also caps processing, which would abort a
/// legitimate hours-long anonymization job mid-way. It is enforced on the
/// upload phase only by [`IngestDeadline`], where the slowloris risk actually
/// is (the global single-job lock is taken *before* the body is spooled).
///
/// Implemented with plain axum middleware so no new dependency is needed.
async fn harden(req: axum::extract::Request, next: axum::middleware::Next) -> Response {
    let mut resp = next.run(req).await;
    const HEADERS: [(&str, &str); 5] = [
        ("x-content-type-options", "nosniff"),
        ("x-frame-options", "DENY"),
        ("referrer-policy", "no-referrer"),
        ("cache-control", "no-store"),
        (
            "content-security-policy",
            "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; \
             connect-src 'self'; img-src 'self' data:; base-uri 'none'; \
             frame-ancestors 'none'; form-action 'none'",
        ),
    ];
    for (name, value) in HEADERS {
        if let (Ok(n), Ok(v)) = (
            header::HeaderName::from_bytes(name.as_bytes()),
            header::HeaderValue::from_str(value),
        ) {
            resp.headers_mut().insert(n, v);
        }
    }
    resp
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
    headers: HeaderMap,
    Json(payload): Json<S3AnonymizeRequest>,
) -> Response {
    let Some(worker) = state.s3.as_ref() else {
        // Route is registered only when S3_ENABLED=true; defense in depth.
        return json_err(
            StatusCode::NOT_FOUND,
            "S3 backend is disabled (S3_ENABLED=false or build without --features s3)",
        );
    };
    // S3 ingestion is unauthenticated by default in the spec, but anyone who
    // can reach the port could then process/overwrite arbitrary bucket keys.
    // With S3_INGEST_AUTH_REQUIRED=true (default) the same operator key gates
    // it; fail-closed when no key is configured.
    if state.cfg.s3_ingest_auth_required {
        if let Err(resp) = operator_authorized(&state, &headers) {
            return resp;
        }
    }

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
async fn status_s3(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(job_id): Path<String>,
) -> Response {
    let Some(worker) = state.s3.as_ref() else {
        return json_err(StatusCode::NOT_FOUND, "S3 backend is disabled");
    };
    // Job ids are UUIDv4 (not enumerable), but the status body leaks input and
    // output bucket keys: gate it like the rest of the S3 ingestion surface.
    if state.cfg.s3_ingest_auth_required {
        if let Err(resp) = operator_authorized(&state, &headers) {
            return resp;
        }
    }
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
    let cfg = state.settings.snapshot();
    // Upload deadline for this request (captured once, not hot-tunable).
    let ingest = IngestDeadline::from_secs(cfg.request_timeout_secs);
    // The per-archive cap is checked while spooling (before processing), so an
    // oversized upload cannot hold the global job lock while it fills the disk.
    let (input_name, spool) = match spool_next_archive_field(
        &mut multipart,
        &cfg.data_dir,
        "upload",
        cfg.max_archive_bytes,
        &ingest,
    )
    .await
    {
            Ok(Some(pair)) => pair,
            Ok(None) => {
                return json_err(
                    StatusCode::BAD_REQUEST,
                    "no archive file found in multipart form (expected field 'file' with a .zip/.7z/.rar)",
                )
            }
            Err(e) => return spool_error_response(&e),
        };
    // The spooled upload is transient; the anonymized output already lives
    // under DATA_DIR (spec §8 "scrittura output STORE"). The guard removes it
    // on every exit path — error, disconnect or panic — so nothing accumulates.
    let _spool_guard = CleanupPath::file(spool.clone());

    let outcome = match state.worker.process_archive_file(&input_name, &spool).await {
        Ok(out) => out,
        Err(e) => {
            tracing::error!("job '{}' rejected: {e:#}", log_safe(&input_name));
            return json_err(StatusCode::BAD_REQUEST, format!("invalid upload: {e}"));
        }
    };

    let safe_name = log_safe(&input_name);
    for summary in &outcome.camera_summaries {
        tracing::info!(
            "job '{safe_name}': camera {} ({:?} branch) — {} image(s), {} detection(s) \
             stored, {} FP crop(s)",
            summary.camera_id,
            summary.branch,
            summary.images,
            summary.detections_stored,
            summary.fp_crops_saved
        );
    }
    tracing::info!(
        "job '{safe_name}' finished: {} images, {} errors, archive {} ({} bytes)",
        outcome.processed_count,
        outcome.error_count,
        outcome.output_path.display(),
        outcome.output_size
    );
    // Operator ledger (DATA_DIR/jobs.jsonl) → GET /operator/jobs.
    if let Err(e) = record_job(&state.cfg.data_dir, &input_name, &outcome, started_at) {
        tracing::warn!("cannot record job '{safe_name}' in the ledger: {e}");
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
    // One guard per spooled archive: they are dropped (and the files removed)
    // when this handler returns, however it returns.
    let mut spool_guards: Vec<CleanupPath> = Vec::new();
    let cfg = state.settings.snapshot();
    // One deadline for the whole request: the budget is not multiplied by the
    // number of archives (a batch of 64 files over a slow link is bounded by
    // the knob, not by 64 × the knob).
    let ingest = IngestDeadline::from_secs(cfg.request_timeout_secs);
    let mut batch_bytes: u64 = 0;

    loop {
        // Bound the request both per archive and in total (0 = unlimited).
        let remaining = if cfg.max_batch_total_bytes == 0 {
            0
        } else {
            cfg.max_batch_total_bytes.saturating_sub(batch_bytes)
        };
        let cap = min_cap(cfg.max_archive_bytes, remaining);
        let (name, spool) = match spool_next_archive_field(
            &mut multipart,
            &cfg.data_dir,
            "batch",
            cap,
            &ingest,
        )
        .await
        {
            Ok(Some(pair)) => pair,
            Ok(None) => break,
            Err(e) => {
                // Cap hit (or upload deadline) while spooling this archive:
                // abort the whole batch.
                cleanup_batch_outputs(&outcomes);
                return spool_error_response(&e);
            }
        };
        // Guard first: the archive that trips the count cap below must be
        // removed on this early return too.
        spool_guards.push(CleanupPath::file(spool.clone()));
        archive_count += 1;
        // Checked on the *spooled* archive rather than at the top of the loop:
        // a form with exactly `max` archives must be accepted.
        if cfg.max_archives_per_batch > 0 && archive_count > cfg.max_archives_per_batch {
            cleanup_batch_outputs(&outcomes);
            return json_err(
                StatusCode::PAYLOAD_TOO_LARGE,
                format!(
                    "batch rejected: at most {} archive(s) per request (MAX_ARCHIVES_PER_BATCH)",
                    cfg.max_archives_per_batch
                ),
            );
        }
        archive_names.push(name.clone());
        batch_bytes += std::fs::metadata(&spool).map(|m| m.len()).unwrap_or(0);
        match state.worker.process_archive_file(&name, &spool).await {
            Ok(outcome) => outcomes.push(outcome),
            Err(e) => {
                tracing::error!("batch job '{}' rejected: {e:#}", log_safe(&name));
                hard_errors.push(ArchiveEntryError {
                    entry: name,
                    message: format!("invalid upload: {e}"),
                });
            }
        }
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
/// (the guard is documented where it is built)
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
            .map(|e| serde_json::json!({ "entry": e.entry, "error": redact_paths(&e.message) }))
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

/// Deadline for the **upload phase** of a request (`REQUEST_TIMEOUT_SECS`,
/// default 3600, 0 = off).
///
/// The deadline covers only how long a client may take to deliver its body:
/// the global single-job lock is acquired *before* the body is spooled, so a
/// client that dribbles (or never sends) its upload holds the service hostage
/// for everyone (slowloris DoS). Processing is intentionally unbounded — a
/// legit multi-hour anonymization job must not be interrupted — so the timer
/// is applied around the body reads instead of around the whole handler.
///
/// One instance is created per request, so a batch shares a single budget
/// across all of its archives rather than multiplying the deadline by the
/// number of files.
#[derive(Debug, Clone, Copy)]
struct IngestDeadline {
    at: Option<tokio::time::Instant>,
    secs: u64,
}

impl IngestDeadline {
    fn from_secs(secs: u64) -> Self {
        Self {
            at: (secs > 0).then(|| tokio::time::Instant::now() + Duration::from_secs(secs)),
            secs,
        }
    }

    /// Runs one body-ingest future under the deadline. On expiry the future is
    /// dropped (a half-written spool is removed by the caller) and an
    /// [`UploadTimedOut`] error is returned, which the handlers answer 408.
    async fn run<F, T>(&self, what: &str, fut: F) -> Result<T>
    where
        F: std::future::Future<Output = Result<T>>,
    {
        let Some(at) = self.at else {
            return fut.await;
        };
        match tokio::time::timeout_at(at, fut).await {
            Ok(result) => result,
            Err(_) => Err(anyhow::Error::new(UploadTimedOut { secs: self.secs }).context(format!(
                "{what} did not complete within REQUEST_TIMEOUT_SECS={}s",
                self.secs
            ))),
        }
    }
}

/// Reads the *next* archive field of a multipart form (any field named `file`
/// or whose file name ends in .zip/.7z/.rar), spooling it to a temp file
/// under `DATA_DIR` so uploads never accumulate in RAM. The header read and the
/// body transfer both count against `ingest` (the upload deadline). Returns
/// `(upload name, spool path)` or `None` when the form is exhausted.
async fn spool_next_archive_field(
    multipart: &mut Multipart,
    data_dir: &std::path::Path,
    tag: &str,
    max_bytes: u64,
    ingest: &IngestDeadline,
) -> Result<Option<(String, PathBuf)>> {
    loop {
        // A client that opens the connection and sends nothing stalls here: the
        // deadline has to cover this await too, not just the body chunks.
        let next = ingest
            .run("receiving the upload", async {
                multipart
                    .next_field()
                    .await
                    .map_err(|e| anyhow::Error::new(e).context("multipart form error"))
            })
            .await
            .map_err(|e| e.context("cannot read the multipart form"))?;
        let Some(field) = next else {
            return Ok(None);
        };
        if !is_archive_field(&field) {
            continue;
        }
        let name = field
            .file_name()
            .map(|s| s.to_string())
            .unwrap_or_else(|| "upload.zip".to_string());
        let stamp = chrono::Utc::now().format("%Y%m%d_%H%M%S%3f");
        let spool = data_dir.join(sanitize_filename(&format!("{tag}_{stamp}_{name}")));
        match ingest
            .run(
                &format!("upload '{name}'"),
                spool_field(field, &spool, max_bytes),
            )
            .await
        {
            Ok(()) => return Ok(Some((name, spool))),
            Err(e) => {
                // A size-cap violation, a timeout (or a disk error) must fail
                // the request: silently skipping to the next field would hide
                // it — and a half-received spool must not stay on disk.
                let _ = tokio::fs::remove_file(&spool).await;
                return Err(e.context("cannot spool upload field"));
            }
        }
    }
}

/// Removes the STORE outputs a batch had already produced when the request is
/// aborted mid-way (archive-count/volume cap). The spooled *inputs* are removed
/// by their `CleanupPath` guards; without this the per-archive outputs would sit
/// in DATA_DIR until the retention pass.
fn cleanup_batch_outputs(outcomes: &[ZipJobOutcome]) {
    for outcome in outcomes {
        if let Err(e) = std::fs::remove_file(&outcome.output_path) {
            if e.kind() != std::io::ErrorKind::NotFound {
                tracing::warn!(
                    "cannot remove aborted batch output {}: {e}",
                    outcome.output_path.display()
                );
            }
        }
    }
}

/// Maps a spool failure to the right HTTP status: `413` for a configured
/// volume cap (`MAX_ARCHIVE_BYTES` / `MAX_BATCH_TOTAL_BYTES`), `408` when the
/// client did not finish delivering the body within `REQUEST_TIMEOUT_SECS`,
/// `400` for anything else. Absolute paths are redacted from the text.
fn spool_error_response(e: &anyhow::Error) -> Response {
    let over_cap = e
        .chain()
        .find_map(|c| c.downcast_ref::<UploadTooLarge>().map(|u| u.limit));
    if let Some(limit) = over_cap {
        return json_err(
            StatusCode::PAYLOAD_TOO_LARGE,
            format!("upload rejected: exceeds the {limit}-byte limit"),
        );
    }
    let timed_out = e
        .chain()
        .find_map(|c| c.downcast_ref::<UploadTimedOut>().map(|u| u.secs));
    if let Some(secs) = timed_out {
        tracing::warn!(
            "upload aborted: the body was not received within REQUEST_TIMEOUT_SECS={secs}s \
             (partial spool removed, global job lock released)"
        );
        return json_err(
            StatusCode::REQUEST_TIMEOUT,
            format!(
                "upload rejected: the body was not received within REQUEST_TIMEOUT_SECS={secs}s \
                 (raise it, or set 0 to disable — processing time is not counted)"
            ),
        );
    }
    json_err(
        StatusCode::BAD_REQUEST,
        format!("invalid upload: {}", redact_paths(&format!("{e:#}"))),
    )
}

/// Marker for "the upload was not delivered within `REQUEST_TIMEOUT_SECS`", so
/// the handler can answer 408 (and the log can name the knob).
#[derive(Debug)]
struct UploadTimedOut {
    secs: u64,
}

impl std::fmt::Display for UploadTimedOut {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "upload did not complete within REQUEST_TIMEOUT_SECS={}s",
            self.secs
        )
    }
}

impl std::error::Error for UploadTimedOut {}

/// Marker for "the upload is larger than the configured cap", so the handler
/// can answer 413 instead of a generic 400.
#[derive(Debug)]
struct UploadTooLarge {
    limit: u64,
}

impl std::fmt::Display for UploadTooLarge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "upload exceeds the {}-byte limit", self.limit)
    }
}

impl std::error::Error for UploadTooLarge {}

/// Smaller of two caps where `0` means "unlimited".
fn min_cap(a: u64, b: u64) -> u64 {
    match (a, b) {
        (0, x) => x,
        (x, 0) => x,
        (x, y) => x.min(y),
    }
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
async fn spool_field(mut field: Field<'_>, path: &std::path::Path, max_bytes: u64) -> Result<()> {
    let mut f = tokio::fs::File::create(path)
        .await
        .with_context(|| format!("create spool file {}", path.display()))?;
    let mut written: u64 = 0;
    while let Ok(Some(chunk)) = field.chunk().await {
        written += chunk.len() as u64;
        // Checked per chunk, before the bytes are written: the cap bounds both
        // the disk usage and how long a client can hold the global job lock.
        if max_bytes > 0 && written > max_bytes {
            return Err(anyhow::Error::new(UploadTooLarge { limit: max_bytes })
                .context(format!("upload exceeds the {max_bytes}-byte limit")));
        }
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

/// Live state of the ACTIVE-branch classifier gate, as reported by
/// `GET /operator/classifier`.
///
/// A usable classifier and `CLASSIFIER_ENFORCE` are independent: the model can
/// be loaded for retraining / A-B validation while enforcement is off, in which
/// case the second check never *removes* a blur. `active_check_enabled` is true
/// only when the gate is genuinely filtering (`loaded && classifier_enforce`),
/// so the endpoint can never claim a check that is not running.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ClassifierGate {
    /// A usable classifier session pool exists right now.
    loaded: bool,
    /// `CLASSIFIER_ENFORCE` (live): may the classifier *remove* a blur?
    classifier_enforce: bool,
    /// `loaded && classifier_enforce` — the second check is really filtering.
    active_check_enabled: bool,
    /// Human-readable reason, so a `loaded=true` with
    /// `active_check_enabled=false` is self-explanatory.
    note: &'static str,
}

impl ClassifierGate {
    fn new(loaded: bool, classifier_enforce: bool) -> Self {
        let active_check_enabled = loaded && classifier_enforce;
        let note = if !loaded {
            "classifier not loaded — ACTIVE blurs every in-ROI detection"
        } else if !classifier_enforce {
            "CLASSIFIER_ENFORCE=false — model loaded for retraining/A-B, \
             ACTIVE blurs every in-ROI detection"
        } else {
            "classifier filtering active — in-ROI detections below the confirm \
             threshold are left unblurred"
        };
        Self {
            loaded,
            classifier_enforce,
            active_check_enabled,
            note,
        }
    }
}

/// `GET /operator/classifier` — current state of the binary face classifier
/// used as the ACTIVE-branch second check (spec §4): whether a model is loaded,
/// whether the check is actually *filtering* (`active_check_enabled`),
/// the live enforce flag + confirm threshold, the configured source URL, the
/// persisted accuracy + last swap from `classifier_state.json` and the live
/// inference counters. `active_check_enabled=false` means ACTIVE blurs every
/// in-ROI detection (fail-safe, no classifier filtering).
async fn op_classifier(State(state): State<AppState>, headers: HeaderMap) -> Response {
    if let Err(resp) = operator_authorized(&state, &headers) {
        return resp;
    }
    // Live snapshot: `classifier_enforce` / `classifier_confirm_threshold` are
    // hot-tunable, so the endpoint must report the current values, not the
    // startup `.env` ones.
    let cfg = state.settings.snapshot();
    let pool = state.worker.store().classifier_pool();
    #[allow(unused_mut)]
    let mut persisted: Option<serde_json::Value> = None;
    #[cfg(feature = "retraining")]
    if let Some(st) = crate::training::read_classifier_state(&cfg) {
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
    let gate = ClassifierGate::new(pool.is_some(), cfg.classifier_enforce);
    let active_path = pool.as_ref().map(|p| p.path().to_string_lossy().to_string());
    Json(serde_json::json!({
        "loaded": gate.loaded,
        "active_check_enabled": gate.active_check_enabled,
        "classifier_enforce": gate.classifier_enforce,
        "active_check_note": gate.note,
        "confirm_threshold": cfg.classifier_confirm_threshold,
        "configured_url": cfg.classifier_model_url,
        "active_onnx": active_path,
        "persisted": persisted,
        "inference": {
            "count": inference_count,
            "avg_ms": inference_avg_ms,
        },
        "min_accuracy_for_swap": cfg.retrain_min_accuracy,
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
    // `nvidia-smi` is blocking I/O (a process spawn, possibly a hung driver):
    // it must never run on a runtime worker, or one slow query costs a whole
    // worker — and the endpoint is documented as purely informational.
    let device_name = tokio::task::spawn_blocking(gpu_device_name)
        .await
        .ok()
        .flatten();
    Json(serde_json::json!({
        "execution_provider": state.cfg.execution_provider.as_str(),
        "device_id": state.cfg.gpu_device_id,
        "gpu_memory_limit_bytes": state.cfg.gpu_memory_limit_bytes,
        "enable_tf32": state.cfg.enable_tf32,
        "enable_fp16": state.cfg.enable_fp16,
        "device_name": device_name,
        "inference": inference,
    }))
    .into_response()
}

/// Best-effort GPU device name via `nvidia-smi`, cached for 60 seconds.
///
/// `None` (no driver, no UTF-8 output, missing binary) is cached too: on a
/// CPU-only host that is the common case, and *not* caching it meant spawning
/// a process on every request. The mutex guard never covers the subprocess, so
/// concurrent callers cannot serialize behind a slow query; the call itself
/// belongs in a blocking context (`spawn_blocking`, see `op_gpu`).
fn gpu_device_name() -> Option<String> {
    use std::sync::Mutex;
    static CACHE: Mutex<Option<(std::time::Instant, Option<String>)>> = Mutex::new(None);
    if let Ok(cache) = CACHE.lock() {
        if let Some((at, name)) = cache.as_ref() {
            if at.elapsed() < Duration::from_secs(60) {
                return name.clone();
            }
        }
    }
    let name = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=name", "--format=csv,noheader,noenv"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|s| !s.is_empty());
    if let Ok(mut cache) = CACHE.lock() {
        *cache = Some((std::time::Instant::now(), name.clone()));
    }
    name
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

/// Constant-time secret comparison. Both sides are hashed to a fixed 32-byte
/// digest first, so neither the *length* nor the *content* of the supplied key
/// leaks through timing: a plain `==` on `str` short-circuits on the length and
/// on the first differing byte, which is a remote oracle for recovering
/// `OPERATOR_API_KEY` one byte at a time (there is no rate limiting in front).
fn secret_eq(provided: &str, expected: &str) -> bool {
    use sha2::{Digest, Sha256};
    let a = Sha256::digest(provided.as_bytes());
    let b = Sha256::digest(expected.as_bytes());
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

/// Single denial message for every failure mode: the previous pair of distinct
/// bodies ("disabled (not configured)" vs "invalid key") told an unauthenticated
/// caller whether a key was configured at all. Failures are logged server-side
/// so brute-force attempts are visible.
const OPERATOR_DENIED: &str = "operator access denied";

#[allow(clippy::result_large_err)] // axum Response on the error path
fn operator_authorized(state: &AppState, headers: &HeaderMap) -> Result<(), Response> {
    let Some(expected) = state.cfg.operator_api_key.as_deref() else {
        tracing::warn!("operator request rejected: OPERATOR_API_KEY is not configured");
        return Err(json_err(StatusCode::FORBIDDEN, OPERATOR_DENIED));
    };
    let provided = headers
        .get(OPERATOR_KEY_HEADER)
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default();
    if secret_eq(provided, expected) {
        Ok(())
    } else {
        tracing::warn!(
            "operator access denied: missing or invalid `{OPERATOR_KEY_HEADER}` header"
        );
        Err(json_err(StatusCode::FORBIDDEN, OPERATOR_DENIED))
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

/// Renders an upload-supplied name for a log line: CR/LF (and other control
/// characters) in a multipart file name would let a caller forge whole log
/// entries. The name itself is left untouched everywhere else (it is the
/// source of the output archive name, and JSON-encoded in the ledger).
fn log_safe(name: &str) -> String {
    name.chars()
        .map(|c| if c.is_control() { '?' } else { c })
        .collect()
}

fn json_err(status: StatusCode, message: impl Into<String>) -> Response {
    // Internal failures (`anyhow` chains, AWS-SDK/SQLite messages) embed
    // absolute DATA_DIR/scratch paths and filesystem detail: never echo those
    // to the caller. The full chain is already in the server log.
    let message = redact_paths(&message.into());
    (status, Json(serde_json::json!({ "error": message }))).into_response()
}

/// Replaces filesystem paths with `<path>` in a string that is about to be sent
/// to an HTTP client (error bodies, `X-Processing-Errors-Detail`,
/// `<input>_error.txt`). Entry/object names are left alone — only tokens that
/// are absolute paths (POSIX, Windows drive, UNC) are touched.
pub(crate) fn redact_paths(msg: &str) -> String {
    let mut out = String::with_capacity(msg.len());
    for (i, token) in msg.split(' ').enumerate() {
        if i > 0 {
            out.push(' ');
        }
        let (start, end) = token_core(token);
        let core = &token[start..end];
        let is_url = core.starts_with("http://") || core.starts_with("https://");
        if !is_url && looks_like_path(core) {
            out.push_str(&token[..start]);
            out.push_str("<path>");
            out.push_str(&token[end..]);
        } else {
            out.push_str(token);
        }
    }
    out
}

/// Byte range of the token minus surrounding punctuation (`"`/`(`/`.`/`:`
///, …), so `'/data/x.zip':` is still recognised as a path.
fn token_core(token: &str) -> (usize, usize) {
    const PUNCT: &[u8] = b"\'\"()[]{}<>,;:.!?";
    let bytes = token.as_bytes();
    let mut start = 0usize;
    let mut end = bytes.len();
    while start < end && PUNCT.contains(&bytes[start]) {
        start += 1;
    }
    while end > start && PUNCT.contains(&bytes[end - 1]) {
        end -= 1;
    }
    (start, end)
}

/// Absolute filesystem path (POSIX, Windows drive or UNC). Relative names such
/// as archive entries (`CAM_001/foto.jpg`) are intentionally not matched.
fn looks_like_path(s: &str) -> bool {
    let b = s.as_bytes();
    if b.len() < 2 {
        return false;
    }
    if b[0] == b'/' {
        return true;
    }
    if b[0] == b'\\' && b[1] == b'\\' {
        return true;
    }
    b.len() >= 3 && b[0].is_ascii_alphabetic() && b[1] == b':' && (b[2] == b'\\' || b[2] == b'/')
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

#[cfg(test)]
mod classifier_gate_tests {
    use super::ClassifierGate;

    /// No usable model ⇒ no filtering, whatever the enforce knob says.
    #[test]
    fn not_loaded_is_never_active() {
        let g = ClassifierGate::new(false, true);
        assert!(!g.loaded);
        assert!(g.classifier_enforce);
        assert!(!g.active_check_enabled);
        assert!(g.note.contains("not loaded"), "note: {}", g.note);
    }

    /// The regression this endpoint had: a model loaded for retraining/A-B with
    /// `CLASSIFIER_ENFORCE=false` is NOT a running second check.
    #[test]
    fn loaded_without_enforce_is_not_filtering() {
        let g = ClassifierGate::new(true, false);
        assert!(g.loaded);
        assert!(!g.active_check_enabled);
        assert!(g.note.contains("CLASSIFIER_ENFORCE=false"), "note: {}", g.note);
    }

    #[test]
    fn loaded_and_enforced_is_filtering() {
        let g = ClassifierGate::new(true, true);
        assert!(g.loaded);
        assert!(g.classifier_enforce);
        assert!(g.active_check_enabled);
        assert!(g.note.contains("filtering active"), "note: {}", g.note);
    }
}

#[cfg(test)]
mod hardening_tests {
    use super::{
        log_safe, looks_like_path, redact_paths, secret_eq, spool_error_response, IngestDeadline,
        UploadTimedOut,
    };
    use std::time::Duration;

    #[tokio::test]
    async fn ingest_deadline_bounds_the_upload_only() {
        // Smallest deadline the knob can express (one second), so the test
        // costs ~1 s of real time and needs no paused clock.
        let on = IngestDeadline::from_secs(1);
        assert!(on.at.is_some());
        // A body delivered within the deadline goes through untouched.
        let v = on
            .run("receiving the upload", async { Ok::<u8, anyhow::Error>(7) })
            .await
            .unwrap();
        assert_eq!(v, 7);
        // A stalled body trips the deadline with the marker error...
        let err = on
            .run("receiving the upload", async {
                tokio::time::sleep(Duration::from_millis(1200)).await;
                Ok::<(), anyhow::Error>(())
            })
            .await
            .unwrap_err();
        assert!(err
            .chain()
            .any(|c| c.downcast_ref::<UploadTimedOut>().is_some()));
        // ...and the 408 response names the knob instead of leaking internals.
        let resp = spool_error_response(&err);
        assert_eq!(resp.status(), axum::http::StatusCode::REQUEST_TIMEOUT);

        // 0 = disabled: the same slow future is allowed to finish.
        let off = IngestDeadline::from_secs(0);
        assert!(off.at.is_none());
        off.run("receiving the upload", async {
            tokio::time::sleep(Duration::from_millis(50)).await;
            Ok::<(), anyhow::Error>(())
        })
        .await
        .unwrap();
    }

    #[test]
    fn cap_violations_still_map_to_413() {
        let err = anyhow::Error::new(super::UploadTooLarge { limit: 1024 })
            .context("upload exceeds the 1024-byte limit");
        let resp = spool_error_response(&err);
        assert_eq!(resp.status(), axum::http::StatusCode::PAYLOAD_TOO_LARGE);
    }

    #[test]
    fn secret_eq_matches_only_identical_secrets() {
        assert!(secret_eq("s3cret", "s3cret"));
        assert!(!secret_eq("s3cret", "s3cres"));
        // Length differences must not matter to the comparison itself.
        assert!(!secret_eq("short", "a-much-longer-secret"));
        assert!(!secret_eq("", "x"));
        assert!(!secret_eq("x", ""));
        assert!(secret_eq("", ""));
    }

    #[test]
    fn absolute_paths_are_redacted_from_client_facing_messages() {
        assert_eq!(
            redact_paths("cannot open /data/x.zip: no such file"),
            "cannot open <path>: no such file"
        );
        assert_eq!(
            redact_paths("failed at C:\\app\\data\\x.zip"),
            "failed at <path>"
        );
        assert_eq!(redact_paths("path '/data/a.zip' missing"), "path '<path>' missing");
        assert_eq!(redact_paths("\\\\srv\\share\\x.zip"), "<path>");
        assert_eq!(redact_paths("cannot create /data/out: permission denied"),
                   "cannot create <path>: permission denied");

        // URLs and relative archive entries must survive untouched.
        assert_eq!(
            redact_paths("callback http://hook.it:8443/x failed"),
            "callback http://hook.it:8443/x failed"
        );
        assert_eq!(
            redact_paths("entry CAM_001/foto.jpg undecodable"),
            "entry CAM_001/foto.jpg undecodable"
        );
        assert!(!looks_like_path("CAM_001/foto.jpg"));
        assert!(looks_like_path("/etc/passwd"));
    }

    #[test]
    fn log_lines_cannot_be_forged_by_upload_names() {
        assert_eq!(log_safe("ok.zip"), "ok.zip");
        assert_eq!(
            log_safe("x.zip\n2026-01-01 INFO fake entry"),
            "x.zip?2026-01-01 INFO fake entry"
        );
        assert_eq!(log_safe("a\r\nb"), "a??b");
    }
}
