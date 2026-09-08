//! Freebuff Desktop — batch face anonymization service (Rust + ONNX).
//!
//! Spec §8 `main.rs`: runtime ONNX model resolution (download + verify +
//! fail-fast), SQLite pool, nightly ROI-finalization + retraining scheduler,
//! and the Axum HTTP server exposing:
//!   - `POST /anonymize`        single-job archive ingestion (§2), output
//!     streamed from disk
//!   - `POST /anonymize/batch`  multiple archives / chunked upload, no body
//!     limit (spooled to disk, processed in sequence, merged ZIP response)
//!   - `GET  /health`
//!   - `GET  /operator/cameras` operator listing / FSM reset (§4), gated by
//!     the `X-Operator-Key` header
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
#[cfg(feature = "retraining")]
mod training;
mod zip_worker;

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

use crate::config::Config;
use crate::db::{Camera, Db};
use crate::model_loader::ensure_model;
use crate::models::{ModelStore, SessionPool};
use crate::retention::{read_jobs_ledger, record_job, JobLedgerEntry};
use crate::roi::{extract_roi, RoiOutcome};
use crate::zip_worker::{sanitize_filename, ArchiveEntryError, ZipJobOutcome, ZipProcessor};

const OPERATOR_KEY_HEADER: &str = "x-operator-key";
const DB_FILENAME: &str = "anonimizzazione_volti.sqlite3";

#[derive(Clone)]
struct AppState {
    cfg: Arc<Config>,
    db: Db,
    worker: ZipProcessor,
    /// Global single-job lock: concurrent uploads get HTTP 429 (§2).
    job_lock: Arc<Mutex<()>>,
}

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

    let cfg = Arc::new(Config::from_env().context("invalid configuration")?);
    tracing::info!(
        "effective image concurrency: {} ({} core(s), env override {:?})",
        cfg.effective_concurrency(),
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1),
        cfg.max_concurrent_images
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

    let yolo = ensure_model(
        &http,
        &cfg.yolo_model_url,
        cfg.yolo_sha256.as_deref(),
        &cfg.model_cache_dir,
    )
    .await
    .with_context(|| "YOLOv8-Face model resolution failed")?;
    // Validate that the file is a loadable ONNX before serving (§3).
    model_loader::load_session(&yolo.path)
        .with_context(|| format!("YOLO model not loadable: {}", yolo.path.display()))?;
    tracing::info!(
        "YOLO model ready at {} (downloaded: {})",
        yolo.path.display(),
        yolo.downloaded
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

    let store = ModelStore::new(
        SessionPool::new(yolo.path.clone(), cfg.effective_concurrency()),
        classifier_pool,
    );

    // Nightly background job: ROI finalization for expired LEARNING cameras,
    // then (optionally) classifier retraining (§5, §6).
    {
        let cfg = cfg.clone();
        let db = db.clone();
        let store = store.clone();
        tokio::spawn(async move {
            background_loop(cfg, db, store).await;
        });
    }

    // STORE-output retention: periodic cleanup of the anonymized ZIPs that
    // accumulate in DATA_DIR (age and/or total-size rules, env `RETENTION_*`).
    if cfg.retention_active() {
        let cfg = cfg.clone();
        tokio::spawn(async move {
            retention::retention_loop(cfg).await;
        });
    }

    let state = AppState {
        cfg: cfg.clone(),
        db: db.clone(),
        worker: ZipProcessor::new(cfg.clone(), db, store),
        job_lock: Arc::new(Mutex::new(())),
    };

    let body_limit = state.cfg.body_limit_bytes;
    let bind_addr = state.cfg.bind_addr.clone();
    let app = Router::new()
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
        .route("/operator/retrain-audit", get(op_retrain_audit))
        .route("/operator/jobs", get(op_jobs))
        .with_state(state);

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

// ─── Helpers ────────────────────────────────────────────────────────────────

fn json_err(status: StatusCode, message: impl Into<String>) -> Response {
    (status, Json(serde_json::json!({ "error": message.into() }))).into_response()
}

/// Nightly scheduler (§5 ROI finalization, §6 retraining). Also runs a
/// catch-up ROI pass at startup so cameras whose LEARNING window expired while
/// the service was down still transition promptly.
async fn background_loop(cfg: Arc<Config>, db: Db, store: ModelStore) {
    tracing::info!("background scheduler started");
    finalize_expired_rois(&cfg, &db).await;

    loop {
        let wait = cfg
            .retrain_schedule
            .next_occurrence_from(chrono::Local::now().time());
        tracing::info!("next nightly run in {}s", wait.as_secs());
        tokio::time::sleep(wait).await;
        tracing::info!("nightly run starting");
        finalize_expired_rois(&cfg, &db).await;
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
