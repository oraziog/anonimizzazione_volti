//! Centralized configuration from environment variables (spec §3, §9 + operational glue).
//!
//! Every knob of the service lives here. Values are parsed once at startup;
//! invalid values abort startup (fail-fast, consistent with the model policy).

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};

use crate::model_loader::ExecutionProvider;

/// Filename (under `DATA_DIR`) of the nightly-retraining audit JSON, served
/// by `GET /operator/retrain-audit`.
pub const RETRAIN_AUDIT_FILENAME: &str = "retrain_audit.json";
/// Filename (under `DATA_DIR`) of the persisted runtime-tunable overrides,
/// edited through the operator settings UI (`GET/POST /operator/settings.json`).
pub const RUNTIME_CONFIG_FILENAME: &str = "runtime_config.json";
/// Running-classifier state (active ONNX + validated accuracy), so the
/// operator can see what model is live and the A/B retraining gate can compare
/// a candidate against the deployed accuracy.
#[allow(dead_code)] // only referenced by the optional retraining feature
pub const CLASSIFIER_STATE_FILENAME: &str = "classifier_state.json";

/// `OPERATOR_API_KEY` wrapper whose `Debug` is redacted: a stray `{:?}` on the
/// whole `Config` (or on this field) can never print the secret. Note the
/// `PartialEq` derive is for tests only — authentication never compares the
/// strings directly (the gate hashes both sides and compares in constant time,
/// see `secret_eq` in `main.rs`).
#[derive(Clone, PartialEq, Eq)]
pub struct OperatorKey(Option<String>);

impl OperatorKey {
    pub fn new(raw: Option<String>) -> Self {
        Self(raw)
    }

    /// The configured key, or `None` when operator endpoints are disabled.
    pub fn as_deref(&self) -> Option<&str> {
        self.0.as_deref()
    }
}

impl std::fmt::Debug for OperatorKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.0 {
            Some(_) => f.write_str("OperatorKey(<redacted>)"),
            None => f.write_str("OperatorKey(none)"),
        }
    }
}

/// Fully-parsed service configuration.
#[derive(Debug, Clone)]
pub struct Config {
    pub bind_addr: String,
    pub body_limit_bytes: usize,
    /// Deadline for the **upload phase** of a request in seconds (env
    /// `REQUEST_TIMEOUT_SECS`, default 3600, 0 = disabled).
    ///
    /// It bounds only how long a client may take to deliver its body (the
    /// multipart header read and the archive transfer; for `/anonymize/batch`
    /// one shared budget covers all the archives of the request).
    /// **Processing time is not counted**: a job that takes hours still
    /// completes, because the timer is applied around the body reads rather
    /// than around the whole handler.
    ///
    /// The reason it exists: the global single-job lock is taken *before* the
    /// upload is spooled, so without a deadline one client that dribbles its
    /// body keeps every other client out (slowloris DoS).
    pub request_timeout_secs: u64,
    pub max_concurrent_images: Option<usize>,
    pub jpeg_quality: u8,

    pub yolo_model_url: String,
    pub classifier_model_url: Option<String>,
    pub model_cache_dir: PathBuf,
    pub yolo_sha256: Option<String>,
    pub classifier_sha256: Option<String>,
    /// Head-fallback: when the face detector finds no faces, run a COCO
    /// person detector and blur the upper fraction of each person box
    /// (env `HEAD_FALLBACK_ENABLED`, default off).
    pub head_fallback_enabled: bool,
    /// COCO person detector model URL (env `COCO_MODEL_URL`; e.g. the
    /// Ultralytics `yolov8n.onnx` with 80 COCO classes).
    pub coco_model_url: String,
    pub coco_sha256: Option<String>,
    /// Fraction of the person box height treated as the head region
    /// (env `HEAD_FALLBACK_FRACTION`, default 0.30).
    pub head_fallback_fraction: f32,
    /// Face-mask segmenter for the ACTIVE blur (env `MASK_SEGMENTER`;
    /// `off` = geometric hull/ellipse, `mediapipe` = MediaPipe Selfie).
    pub mask_segmenter: MaskSegmenter,
    /// Model URL for `MASK_SEGMENTER=mediapipe` (cached under `model_cache_dir`).
    pub selfie_segmenter_url: String,
    pub selfie_segmenter_sha256: Option<String>,
    /// Faces whose detection box is narrower than this many px are masked with
    /// the geometric hull/ellipse instead of the per-face selfie inference
    /// (env `SEGMENTER_MIN_BOX`, default 0 = disable the shortcut). At tiny
    /// sizes the 256² upscale degrades the silhouette and each ONNX run costs
    /// ~26 ms per face — skipping them preserves quality where the model adds
    /// nothing while cutting most of the ACTIVE budget on crowded scenes.
    pub segmenter_min_box_px: f32,
    /// Output format of the STORE archives (env `OUTPUT_FORMAT`; `keep`
    /// preserves each input's format, `jpeg`/`png` force it and rename the
    /// output entries accordingly).
    pub output_format: OutputFormat,
    /// Downscale the anonymized output so its larger side is ≤ this many px
    /// (env `OUTPUT_MAX_SIDE`, default 0 = off). Applied after anonymization
    /// — camera geometry and masks are left untouched.
    pub output_max_side_px: u32,
    /// Letterbox side in px for the YOLO detector input (env `YOLO_INPUT_SIZE`,
    /// default 640). Lower values (e.g. 512) speed up detection on UHD/4K
    /// sources at a small cost in small-face recall.
    pub yolo_input_size: u32,

    pub data_dir: PathBuf,
    pub dataset_fp_dir: PathBuf,
    pub dataset_seed_real_faces_dir: PathBuf,
    pub models_backup_dir: PathBuf,

    pub yolo_conf_threshold: f32,
    /// Detector confidence for ACTIVE cameras: usually *lower* than the
    /// LEARNING threshold — a false negative (unblurred face) is worse for
    /// GDPR than an extra blur, so ACTIVE leans sensitive (env
    /// `YOLO_CONF_THRESHOLD_ACTIVE`; falls back to `YOLO_CONF_THRESHOLD`).
    pub yolo_conf_threshold_active: f32,
    pub yolo_nms_iou: f32,
    pub fp_crop_conf_max: f32,
    pub initial_blur_sigma: f32,
    pub blur_hull_margin_pct: f32,
    /// Ellipse (and head-fallback ellipse) margin for the ACTIVE mask
    /// (env `BLUR_ELLIPSE_MARGIN`, default 0.05). Bigger margins widen the
    /// masked cover around the detection box — directly counters the
    /// "semi-transparent" look where only the skin core is blurred.
    pub blur_ellipse_margin: f32,
    /// Softening sigma for the ACTIVE mask edge (env `MASK_FEATHER_SIGMA`,
    /// default 0 = auto: reuse the blur sigma). A *small* fixed value (e.g.
    /// 8) keeps face silhouettes crisp instead of a wide translucent band on
    /// large close-ups (where the blur sigma reaches 50 px).
    pub mask_feather_sigma: f32,
    /// Classifier confirm gate for ACTIVE (env `CLASSIFIER_CONFIRM_THRESHOLD`,
    /// default 0.5): detection is blurred only if `p_face >= threshold`.
    /// Lower values raise recall at the cost of re-blurring dubious regions;
    /// set the model URL empty / `CLASSIFIER_MODEL_URL=` to blur everything
    /// the face detector finds (GDPR fail-safe: never fewer blurs than YOLO).
    pub classifier_confirm_threshold: f32,
    /// Whether the classifier confirmation may *remove* a blur (env
    /// `CLASSIFIER_ENFORCE`, default true). `false` loads the model for
    /// retraining/A-B but the ACTIVE branch blurs every YOLO detection
    /// (classification can never leave a detected face unblurred).
    pub classifier_enforce: bool,
    pub anon_mode: AnonMode,
    pub pixelate_cell_px: u32,

    pub learning_days: u32,
    pub roi_eps_px: f32,
    pub roi_min_samples: u32,
    pub roi_rdp_epsilon: f32,
    pub roi_area_min: f64,
    pub roi_area_max: f64,
    pub roi_margin_pct: f64,

    /// Dynamic ROI for ACTIVE cameras (env `ROI_REEXTRACT_ENABLED`, default
    /// true): every nightly pass re-extracts the polygon from the ACTIVE
    /// detections of the last `ROI_REEXTRACT_WINDOW_DAYS`, and swaps it only
    /// if it differs from the deployed one beyond `ROI_REEXTRACT_MIN_IOU`
    /// (stability guard against PTZ/scene flap).
    pub roi_reextract_enabled: bool,
    pub roi_reextract_window_days: u32,
    pub roi_reextract_min_iou: f64,

    pub retrain_schedule: ScheduleSpec,
    #[allow(dead_code)] // consumed by the optional retraining feature
    pub retrain_min_accuracy: f32,
    /// A/B regression tolerance: a retrained candidate is swapped only if its
    /// Rust-validated accuracy is ≥ `current − RETRAIN_REGRESSION_EPS` and still
    /// ≥ `RETRAIN_MIN_ACCURACY` (env `RETRAIN_REGRESSION_EPS`, default 0: the
    /// candidate must at least match the deployed model on the same holdout).
    #[allow(dead_code)] // consumed by the optional retraining feature
    pub retrain_regression_eps: f32,
    #[allow(dead_code)] // consumed by the optional retraining feature
    pub retrain_holdout_fraction: f32,
    #[allow(dead_code)] // consumed by the optional retraining feature
    pub retrain_epochs: u32,
    #[allow(dead_code)] // consumed by the optional retraining feature
    pub retrain_batch_size: u32,

    pub operator_api_key: OperatorKey,

    // STORE-output retention (DATA_DIR `*_elaborato.zip` cleanup).
    pub retention_enabled: bool,
    pub retention_max_days: u32,
    pub retention_max_gb: f64,
    pub retention_interval_secs: u64,
    /// Outputs modified more recently than this are never deleted (in-flight
    /// jobs / responses streaming right now).
    pub retention_min_age_secs: u64,

    // ── Hardening: archive / ingest limits (zip-bomb & DoS guards) ────────
    // Every value is a *cap*: 0 disables that specific check. The defaults
    // are generous for real image batches and still stop an amplification or
    // exhaustion attempt.
    /// Max entries accepted in a single archive (0 = unlimited).
    pub max_entries_per_archive: usize,
    /// Max bytes of a single decompressed entry (0 = unlimited).
    pub max_entry_bytes: u64,
    /// Max total decompressed bytes per archive (0 = unlimited).
    pub max_total_uncompressed_bytes: u64,
    /// Max uncompressed/compressed ratio per entry (0 = unlimited). A *spike*
    /// detector, not the main bound: `MAX_ENTRY_BYTES`/`MAX_TOTAL_*` already cap
    /// how much a hostile archive can make us allocate. The default (500) sits
    /// above even a heavily redundant scanned/bilevel image entry, so it only
    /// trips on deliberate amplification. Measured on the repo's own mixed
    /// archive: a 405 KB text entry deflates 1875x, which is why this knob must
    /// not be set at "a few hundred" without checking the workload.
    pub max_compression_ratio: u64,
    /// Image decode bounds handed to `image::Limits` (0 = unlimited).
    pub max_image_width: u32,
    pub max_image_height: u32,
    pub max_image_alloc_bytes: u64,
    /// Batch/upload volume caps (0 = unlimited).
    pub max_archives_per_batch: usize,
    pub max_archive_bytes: u64,
    pub max_batch_total_bytes: u64,

    // ── Hardening: model download policy ─────────────────────────────────
    /// Allow plain `http://` model URLs (default false: `https://` only, so a
    /// MITM cannot swap the ONNX the runtime parses).
    pub model_allow_http: bool,
    /// Refuse to download/start when no `MODEL_*_SHA256` is configured.
    pub model_sha_required: bool,
    /// Max redirects followed while downloading a model (0 = none).
    pub model_max_redirects: usize,

    // ── Hardening: S3 ingest ─────────────────────────────────────────────
    /// Require the operator key on `/anonymize/s3` and `/status/:job_id`
    /// (env `S3_INGEST_AUTH_REQUIRED`, default true). Only consulted by the
    /// `s3` feature routes.
    #[cfg_attr(not(feature = "s3"), allow(dead_code))]
    pub s3_ingest_auth_required: bool,

    // Inference execution provider (env `ORT_EXECUTION_PROVIDER`, default
    // `cpu`): which ONNX Runtime EP every session is built with. The GPU EPs
    // additionally require the crate to be compiled with the matching cargo
    // feature (see `model_loader` docs and the `Dockerfile.gpu`).
    pub execution_provider: ExecutionProvider,
    pub gpu_device_id: i32,
    /// Device-memory arena / TensorRT workspace cap (bytes, env
    /// `ORT_CUDA_MEMORY_LIMIT_BYTES`). `None` lets ONNX Runtime pick.
    pub gpu_memory_limit_bytes: Option<u64>,
    /// TensorFloat-32 for the CUDA EP on Ampere+ (env `ORT_ENABLE_TF32`).
    pub enable_tf32: bool,
    /// Reduced-precision fp16 for the TensorRT EP (env `ORT_ENABLE_FP16`).
    pub enable_fp16: bool,

    /// S3 storage backend (env `S3_ENABLED`). `None` when disabled — the
    /// HTTP-only disk-backed flow is then the only one compiled in, and the
    /// `s3` cargo feature is not even required.
    #[cfg(feature = "s3")]
    pub s3: Option<S3Settings>,

    /// SQS async job intake (feature `queue`, env `SQS_ENABLED`). `None` when
    /// disabled — no SQS code runs and no queue consumer is spawned.
    #[cfg(feature = "queue")]
    pub sqs: Option<SqsSettings>,

    /// RabbitMQ async job intake (feature `rabbitmq`, env `RABBITMQ_ENABLED`).
    #[cfg(feature = "rabbitmq")]
    pub rabbitmq: Option<RabbitMqSettings>,
}

/// S3 ingestion/storage configuration (feature `s3`, spec §8 "Scenario S3").
#[cfg(feature = "s3")]
#[derive(Debug, Clone)]
pub struct S3Settings {
    /// Endpoint URL for S3-compatible stores (MinIO / local). When set, the
    /// client uses path-style addressing (env `S3_ENDPOINT`; leave empty for
    /// real AWS, which uses virtual-hosted style).
    pub endpoint: Option<String>,
    /// Force path-style bucket addressing (env `S3_FORCE_PATH_STYLE`);
    /// defaults to `true` when `S3_ENDPOINT` is set, `false` otherwise.
    pub force_path_style: bool,
    /// Input bucket where cameras upload their archives (env `S3_BUCKET_INPUT`).
    pub bucket_input: String,
    /// Output bucket where the anonymized ZIPs land (env `S3_BUCKET_OUTPUT`).
    pub bucket_output: String,
    /// Per-job JSON audit logs (env `S3_BUCKET_LOGS`).
    pub bucket_logs: String,
    /// How many S3 jobs may run concurrently (env `S3_MAX_CONCURRENT_JOBS`,
    /// default 2). The per-image semaphore (`effective_concurrency`) still
    /// bounds GPU/CPU load inside each job.
    pub max_concurrent_jobs: usize,
    /// Optional allowlist of `host[:port]` that a completion webhook may call
    /// (env `S3_WEBHOOK_ALLOWED_HOSTS`, comma-separated). Empty = webhooks
    /// rejected at submit time (anti-{SSRF,abuse} default).
    pub webhook_allowed_hosts: Vec<String>,
    /// Honour `delete_input_on_success` carried by a queue message (env
    /// `S3_TRUST_MESSAGE_DELETE`, default false). A message is attacker-
    /// controlled data, so it must not be able to delete bucket objects unless
    /// the operator explicitly opts in. Same env/default as
    /// `Config::s3_trust_message_delete` (kept here because the worker only
    /// sees `S3Settings`).
    pub trust_message_delete: bool,
}

/// SQS consumer settings (feature `queue`, spec §8 "Scenario S3" extended).
#[cfg(feature = "queue")]
#[derive(Debug, Clone)]
pub struct SqsSettings {
    /// Queue URL to poll for job messages (`SQS_QUEUE_URL`).
    pub queue_url: String,
    /// Optional AWS region override (`SQS_REGION`); defaults to the standard
    /// `AWS_REGION` chain used by the S3 client.
    pub region: Option<String>,
    /// Max messages to fetch per poll (`SQS_MAX_MESSAGES`, default 10).
    pub max_messages: i32,
    /// Long-poll wait seconds (`SQS_WAIT_SECONDS`, default 20).
    pub wait_seconds: i32,
    /// Visibility timeout applied to in-flight messages (`SQS_VISIBILITY_TIMEOUT_SECONDS`,
    /// default 900). A job that crashes mid-run becomes visible again after
    /// this window and is redelivered.
    pub visibility_timeout_seconds: i32,
    /// Base backoff after a failed poll (exponential: `base * 2^n`, capped at
    /// `SQS_MAX_BACKOFF_SECS`). Keeps the consumer alive through transient
    /// throttling/5xx errors instead of hot-looping.
    pub poll_interval_secs: u64,
    /// Cap of the exponential poll backoff (`SQS_MAX_BACKOFF_SECS`, default 60).
    pub max_backoff_secs: u64,
    /// Max receive attempts before a message is dropped (poison) — also used
    /// as the `maxReceiveCount` of the auto-created redrive policy
    /// (`SQS_MAX_RECEIVE_ATTEMPTS`, default 5).
    pub max_receive_attempts: i32,
}

/// RabbitMQ consumer settings (feature `rabbitmq`, spec §8 "Scenario S3"
/// extended). The consumer declares the work queue + a dead-letter queue.
#[cfg(feature = "rabbitmq")]
#[derive(Debug, Clone)]
pub struct RabbitMqSettings {
    /// AMQP URL (`RABBITMQ_URL`, default `amqp://127.0.0.1:5672`).
    pub url: String,
    /// Work queue name (`RABBITMQ_QUEUE`, default `anonimizzazione-jobs`).
    pub queue: String,
    /// Dead-letter queue name (`RABBITMQ_DLQ`, default `anonimizzazione-jobs-dlq`).
    pub dlq: String,
    /// Prefetch count (`RABBITMQ_PREFETCH`, default 4): max unacked messages
    /// this consumer holds at once (each one is a concurrent job).
    pub prefetch: u16,
    /// Max delivery attempts before a message is sent to the DLQ
    /// (`RABBITMQ_MAX_RETRIES`, default 3).
    pub max_retries: u32,
    /// Base backoff in seconds between retries (exponential: `base * 2^n`,
    /// capped at `RABBITMQ_MAX_BACKOFF_SECS`).
    pub retry_backoff_secs: u64,
    /// Cap of the exponential backoff (`RABBITMQ_MAX_BACKOFF_SECS`, default 300).
    pub max_backoff_secs: u64,
}

/// Reads `SQS_*` env vars into an optional `SqsSettings` (feature `queue` only).
#[cfg(feature = "queue")]
fn sqs_settings_from_env() -> Result<Option<SqsSettings>> {
    let enabled = env_parse::<bool>("SQS_ENABLED", false)?;
    if !enabled {
        return Ok(None);
    }
    let queue_url = env_str("SQS_QUEUE_URL", "");
    if queue_url.is_empty() {
        anyhow::bail!("SQS_QUEUE_URL must be set when SQS is enabled");
    }
    Ok(Some(SqsSettings {
        queue_url,
        region: env_opt("SQS_REGION"),
        max_messages: env_parse::<i32>("SQS_MAX_MESSAGES", 10)?.clamp(1, 10),
        wait_seconds: env_parse::<i32>("SQS_WAIT_SECONDS", 20)?.clamp(0, 20),
        visibility_timeout_seconds: env_parse::<i32>("SQS_VISIBILITY_TIMEOUT_SECONDS", 900)?
            .max(0),
        poll_interval_secs: env_parse::<u64>("SQS_POLL_INTERVAL_SECS", 5)?,
        max_backoff_secs: env_parse::<u64>("SQS_MAX_BACKOFF_SECS", 60)?.max(1),
        max_receive_attempts: env_parse::<i32>("SQS_MAX_RECEIVE_ATTEMPTS", 5)?.max(1),
    }))
}

/// Reads `RABBITMQ_*` env vars into an optional `RabbitMqSettings` (feature
/// `rabbitmq` only).
#[cfg(feature = "rabbitmq")]
fn rabbitmq_settings_from_env() -> Result<Option<RabbitMqSettings>> {
    let enabled = env_parse::<bool>("RABBITMQ_ENABLED", false)?;
    if !enabled {
        return Ok(None);
    }
    Ok(Some(RabbitMqSettings {
        url: env_str("RABBITMQ_URL", "amqp://127.0.0.1:5672"),
        queue: env_str("RABBITMQ_QUEUE", "anonimizzazione-jobs"),
        dlq: env_str("RABBITMQ_DLQ", "anonimizzazione-jobs-dlq"),
        prefetch: env_parse::<u16>("RABBITMQ_PREFETCH", 4)?,
        max_retries: env_parse::<u32>("RABBITMQ_MAX_RETRIES", 3)?.max(1),
        retry_backoff_secs: env_parse::<u64>("RABBITMQ_RETRY_BACKOFF_SECS", 5)?,
        max_backoff_secs: env_parse::<u64>("RABBITMQ_MAX_BACKOFF_SECS", 300)?
            .max(1),
    }))
}

/// Reads `S3_*` env vars into an optional `S3Settings` (feature `s3` only).
#[cfg(feature = "s3")]
fn s3_settings_from_env() -> Result<Option<S3Settings>> {
    let enabled = env_parse::<bool>("S3_ENABLED", false)?;
    if !enabled {
        return Ok(None);
    }
    let endpoint = env_opt("S3_ENDPOINT")
        .filter(|v| !v.is_empty())
        .map(|v| v.trim_end_matches('/').to_string());
    let force_path_style = match std::env::var("S3_FORCE_PATH_STYLE") {
        Ok(v) if !v.trim().is_empty() => v
            .trim()
            .parse::<bool>()
            .context("S3_FORCE_PATH_STYLE must be true or false")?,
        _ => endpoint.is_some(),
    };
    let max_concurrent_jobs = env_parse::<usize>("S3_MAX_CONCURRENT_JOBS", 2)?.max(1);
    let webhook_allowed_hosts = std::env::var("S3_WEBHOOK_ALLOWED_HOSTS")
        .map(|v| {
            v.split(',')
                .map(|s| s.trim().to_ascii_lowercase())
                .filter(|s| !s.is_empty())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    let settings = S3Settings {
        endpoint,
        force_path_style,
        bucket_input: env_str("S3_BUCKET_INPUT", "anonimizzazione-input"),
        bucket_output: env_str("S3_BUCKET_OUTPUT", "anonimizzazione-output"),
        bucket_logs: env_str("S3_BUCKET_LOGS", "anonimizzazione-logs"),
        max_concurrent_jobs,
        webhook_allowed_hosts,
        trust_message_delete: env_parse("S3_TRUST_MESSAGE_DELETE", false)?,
    };

    for (name, v) in [
        ("S3_BUCKET_INPUT", &settings.bucket_input),
        ("S3_BUCKET_OUTPUT", &settings.bucket_output),
        ("S3_BUCKET_LOGS", &settings.bucket_logs),
    ] {
        if v.is_empty() {
            anyhow::bail!("{name} must not be empty when S3 is enabled");
        }
    }
    Ok(Some(settings))
}

/// Anonymization operation applied to detected faces (env `ANON_MODE`):
/// Gaussian-style box blur (default) or the mosaic/pixelation used by Google
/// Street View (`PIXELATE_CELL_PX` controls the block size).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnonMode {
    Blur,
    Pixelate,
}


/// Optional face-mask segmenter for the ACTIVE blur (env `MASK_SEGMENTER`):
/// `mediapipe` runs the MediaPipe Selfie Segmentation model per face and
/// blurs the selfie silhouette (dilated 25% of the minor side) in **union**
/// with the box ellipse; `off` (default) keeps the geometric
/// keypoint-hull/ellipse masks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaskSegmenter {
    Off,
    Mediapipe,
}

impl MaskSegmenter {
    fn parse(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "" | "off" | "none" | "disabled" => Ok(MaskSegmenter::Off),
            "mediapipe" | "selfie" | "selfie_segmentation" | "on" => Ok(MaskSegmenter::Mediapipe),
            other => anyhow::bail!("MASK_SEGMENTER '{other}' must be 'off' or 'mediapipe'"),
        }
    }
}

impl AnonMode {
    fn parse(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "blur" | "gaussian" => Ok(AnonMode::Blur),
            "pixelate" | "pixel" | "mosaic" | "pixellize" => Ok(AnonMode::Pixelate),
            other => anyhow::bail!("ANON_MODE '{other}' must be 'blur' or 'pixelate'"),
        }
    }
}

/// Output normalization for the STORE archives (env `OUTPUT_FORMAT`):
/// `keep` preserves each input's format (JPEG→JPEG, PNG→PNG), `jpeg` forces
/// every frame to re-encoded JPEG, `png` forces lossless PNG. The output name
/// follows: `.jpg` / `.png` regardless of the input extension.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    Keep,
    Jpeg,
    Png,
}

impl OutputFormat {
    fn parse(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "" | "keep" | "original" | "auto" => Ok(OutputFormat::Keep),
            "jpeg" | "jpg" => Ok(OutputFormat::Jpeg),
            "png" => Ok(OutputFormat::Png),
            other => anyhow::bail!("OUTPUT_FORMAT '{other}' must be 'keep', 'jpeg' or 'png'"),
        }
    }
}

/// Parses `OUTPUT_MAX_SIDE`: 0 (off) or a larger side ≥ 128 px (downscaling an
/// output below ~64 px corrupts the anonymized face region meaninglessly).
fn parse_output_max_side() -> Result<u32> {
    let v = env_parse::<u32>("OUTPUT_MAX_SIDE", 0)?;
    if v != 0 && v < 128 {
        anyhow::bail!("OUTPUT_MAX_SIDE must be 0 (off) or ≥ 128 px, got {v}");
    }
    Ok(v)
}

/// Parses `ORT_CUDA_DEVICE_ID`: the index of the GPU to use, clamped ≥ 0
/// (CUDA device handles are unsigned).
fn parse_gpu_device_id() -> Result<i32> {
    env_parse::<i32>("ORT_CUDA_DEVICE_ID", 0).map(|v| v.max(0))
}

/// Parses `ORT_CUDA_MEMORY_LIMIT_BYTES`: optional device-memory arena /
/// TensorRT workspace cap. `Some(0)` is rejected — a zero cap is a
/// configuration error, not "unlimited".
fn parse_gpu_memory_limit() -> Result<Option<u64>> {
    match env_opt("ORT_CUDA_MEMORY_LIMIT_BYTES") {
        Some(v) => {
            let bytes: u64 = v
                .parse()
                .context("ORT_CUDA_MEMORY_LIMIT_BYTES must be a byte count")?;
            if bytes == 0 {
                anyhow::bail!("ORT_CUDA_MEMORY_LIMIT_BYTES must be > 0");
            }
            Ok(Some(bytes))
        }
        None => Ok(None),
    }
}

/// HH:MM daily schedule (spec: `CRON_RETRAIN_SCHEDULE`, default `03:00`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScheduleSpec {
    pub hour: u32,
    pub minute: u32,
}

impl ScheduleSpec {
    fn parse(s: &str) -> Result<Self> {
        let (h, m) = s
            .split_once(':')
            .with_context(|| format!("CRON_RETRAIN_SCHEDULE '{s}' must be HH:MM"))?;
        let hour: u32 = h.parse().context("schedule hour must be 0-23")?;
        let minute: u32 = m.parse().context("schedule minute must be 0-59")?;
        if hour > 23 || minute > 59 {
            anyhow::bail!("schedule time out of range: {s}");
        }
        Ok(Self { hour, minute })
    }

    /// Duration from "now" until the next occurrence of this time of day.
    pub fn next_occurrence_from(&self, now: chrono::NaiveTime) -> Duration {
        let target =
            chrono::NaiveTime::from_hms_opt(self.hour, self.minute, 0).expect("valid h/m/s");
        let today_target = now;
        let secs_until_today = target.signed_duration_since(today_target).num_seconds();
        let secs = if secs_until_today > 0 {
            secs_until_today
        } else {
            secs_until_today + 24 * 3600
        };
        Duration::from_secs(secs as u64)
    }
}

fn env_str(name: &str, default: &str) -> String {
    std::env::var(name)
        .ok()
        .filter(|v| !v.trim().is_empty())
        .unwrap_or_else(|| default.to_string())
}

fn env_opt(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
}

fn env_parse<T: std::str::FromStr>(name: &str, default: T) -> Result<T> {
    match std::env::var(name) {
        Ok(v) if !v.trim().is_empty() => v
            .trim()
            .parse::<T>()
            .map_err(|_| anyhow::anyhow!("invalid value for {name}: '{v}'")),
        _ => Ok(default),
    }
}

fn env_path(name: &str, default: &str) -> PathBuf {
    PathBuf::from(env_str(name, default))
}

/// Parses a detector input side in px: multiple of 32 within 256..=1280
/// (ONNX exports are trained at 640 and accept dynamic sizes; 32-alignment
/// keeps the FPN grids integer-exact, and the range keeps memory bounded).
fn parse_input_size(name: &str) -> Result<u32> {
    validate_input_size(env_parse(name, 640u32)?, name)
}

/// Validates a detector input side: multiple of 32 within 256..=1280.
fn validate_input_size(v: u32, name: &str) -> Result<u32> {
    if !(256..=1280).contains(&v) {
        anyhow::bail!("{name} must be in 256..=1280, got {v}");
    }
    if !v.is_multiple_of(32) {
        anyhow::bail!("{name} must be a multiple of 32, got {v}");
    }
    Ok(v)
}

/// Auto-detected per-image concurrency: on machines with ≤ 4 physical cores
/// use **all** of them (the −2 headroom only hurts when cores are scarce — on
/// a 2-core box it would drop the service to a single worker); above 4 cores
/// keep the spec §9 `cores − 2` margin. Always ≥ 1.
pub fn default_concurrency() -> usize {
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(8);
    if cores <= 4 {
        cores.max(1)
    } else {
        cores - 2
    }
}

impl Config {
    /// Reads and validates all environment variables. Called once from `main`.
    pub fn from_env() -> Result<Self> {
        let max_concurrent_images = match std::env::var("MAX_CONCURRENT_IMAGES") {
            Ok(v) if !v.trim().is_empty() => Some(
                v.trim()
                    .parse::<usize>()
                    .context("MAX_CONCURRENT_IMAGES must be a positive integer")?,
            ),
            _ => None,
        };

        let detector_conf = env_parse::<f32>("YOLO_CONF_THRESHOLD", 0.20)?.clamp(0.01, 0.99);
        let detector_conf_active =
            env_parse::<f32>("YOLO_CONF_THRESHOLD_ACTIVE", detector_conf)?.clamp(0.01, 0.99);

        let cfg = Self {
            bind_addr: env_str("BIND_ADDR", "0.0.0.0:8080"),
            body_limit_bytes: env_parse("BODY_LIMIT_BYTES", 3_758_096_384usize)?,
            request_timeout_secs: env_parse("REQUEST_TIMEOUT_SECS", 3600u64)?,
            max_concurrent_images,
            jpeg_quality: env_parse::<u8>("JPEG_QUALITY", 95)?.clamp(1, 100),

            yolo_model_url: env_str(
                "YOLO_MODEL_URL",
                "https://github.com/yakhyo/yolov8-face-onnx-inference/releases/download/weights/yolov8n-face.onnx",
            ),
            classifier_model_url: env_opt("CLASSIFIER_MODEL_URL"),
            model_cache_dir: env_path("MODEL_CACHE_DIR", "/app/models/cache/"),
            yolo_sha256: env_opt("MODEL_YOLO_SHA256"),
            classifier_sha256: env_opt("MODEL_CLASSIFIER_SHA256"),
            head_fallback_enabled: env_parse::<bool>("HEAD_FALLBACK_ENABLED", false)?,
            coco_model_url: env_str(
                "COCO_MODEL_URL",
                "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.onnx",
            ),
            coco_sha256: env_opt("MODEL_COCO_SHA256"),
            head_fallback_fraction: env_parse::<f32>("HEAD_FALLBACK_FRACTION", 0.30)?
                .clamp(0.05, 0.9),
            yolo_input_size: parse_input_size("YOLO_INPUT_SIZE")?,
            mask_segmenter: MaskSegmenter::parse(&env_str("MASK_SEGMENTER", "off"))?,
            selfie_segmenter_url: env_str(
                "SELFIE_SEGMENTER_MODEL_URL",
                "https://huggingface.co/onnx-community/mediapipe_selfie_segmentation/resolve/main/onnx/model_quantized.onnx",
            ),
            selfie_segmenter_sha256: env_opt("MODEL_SELFIE_SHA256"),
            segmenter_min_box_px: env_parse::<f32>("SEGMENTER_MIN_BOX", 0.0)?.max(0.0),
            output_format: OutputFormat::parse(&env_str("OUTPUT_FORMAT", "keep"))?,
            output_max_side_px: parse_output_max_side()?,

            data_dir: env_path("DATA_DIR", "/app/data/"),
            dataset_fp_dir: env_path("DATASET_FP_DIR", "/app/dataset_falsi_positivi/"),
            dataset_seed_real_faces_dir: env_path(
                "DATASET_SEED_REAL_FACES_DIR",
                "/app/dataset_seed/real_faces/",
            ),
            models_backup_dir: env_path("MODELS_BACKUP_DIR", "/app/models/backup/"),

            yolo_conf_threshold: detector_conf,
            yolo_conf_threshold_active: detector_conf_active,
            yolo_nms_iou: env_parse::<f32>("YOLO_NMS_IOU", 0.45)?.clamp(0.05, 0.95),
            fp_crop_conf_max: env_parse::<f32>("FP_CROP_CONF_MAX", 0.50)?,
            initial_blur_sigma: env_parse::<f32>("INITIAL_BLUR_SIGMA", 20.0)?.max(1.0),
            blur_hull_margin_pct: env_parse::<f32>("BLUR_HULL_MARGIN_PCT", 0.10)?.clamp(0.0, 0.5),
            blur_ellipse_margin: env_parse::<f32>("BLUR_ELLIPSE_MARGIN", 0.05)?.clamp(0.0, 0.5),
            mask_feather_sigma: env_parse::<f32>("MASK_FEATHER_SIGMA", 0.0)?.clamp(0.0, 64.0),
            classifier_confirm_threshold: env_parse::<f32>("CLASSIFIER_CONFIRM_THRESHOLD", 0.5)?
                .clamp(0.01, 1.0),
            classifier_enforce: env_parse::<bool>("CLASSIFIER_ENFORCE", true)?,
            anon_mode: AnonMode::parse(&env_str("ANON_MODE", "blur"))?,
            pixelate_cell_px: env_parse::<u32>("PIXELATE_CELL_PX", 12)?.clamp(2, 128),

            learning_days: env_parse::<u32>("LEARNING_DAYS", 30)?.max(1),
            roi_eps_px: env_parse::<f32>("ROI_EPS_PX", 50.0)?.max(1.0),
            roi_min_samples: env_parse::<u32>("ROI_MIN_SAMPLES", 15)?.max(1),
            roi_rdp_epsilon: env_parse::<f32>("ROI_RDP_EPSILON", 5.0)?.max(0.1),
            roi_area_min: env_parse::<f64>("ROI_AREA_MIN", 0.10)?,
            roi_area_max: env_parse::<f64>("ROI_AREA_MAX", 0.90)?,
            roi_margin_pct: env_parse::<f64>("ROI_MARGIN_PCT", 0.05)?.clamp(0.0, 0.5),
            roi_reextract_enabled: env_parse::<bool>("ROI_REEXTRACT_ENABLED", true)?,
            roi_reextract_window_days: env_parse::<u32>("ROI_REEXTRACT_WINDOW_DAYS", 7)?
                .clamp(1, 365),
            roi_reextract_min_iou: env_parse::<f64>("ROI_REEXTRACT_MIN_IOU", 0.6)?
                .clamp(0.0, 0.95),

            retrain_schedule: ScheduleSpec::parse(&env_str("CRON_RETRAIN_SCHEDULE", "03:00"))?,
            retrain_min_accuracy: env_parse::<f32>("RETRAIN_MIN_ACCURACY", 0.85)?
                .clamp(0.5, 1.0),
            retrain_regression_eps: env_parse::<f32>("RETRAIN_REGRESSION_EPS", 0.0)?
                .clamp(0.0, 0.5),
            retrain_holdout_fraction: env_parse::<f32>("RETRAIN_HOLDOUT_FRACTION", 0.10)?
                .clamp(0.05, 0.5),
            retrain_epochs: env_parse::<u32>("RETRAIN_EPOCHS", 5)?.max(1),
            retrain_batch_size: env_parse::<u32>("RETRAIN_BATCH_SIZE", 32)?.max(1),

            operator_api_key: OperatorKey::new(env_opt("OPERATOR_API_KEY")),

            retention_enabled: env_parse::<bool>("RETENTION_ENABLED", true)?,
            retention_max_days: env_parse::<u32>("RETENTION_MAX_DAYS", 30)?,
            retention_max_gb: env_parse::<f64>("RETENTION_MAX_GB", 20.0)?,
            retention_interval_secs: env_parse::<u64>("RETENTION_INTERVAL_SECS", 3600)?.max(60),
            retention_min_age_secs: env_parse::<u64>("RETENTION_MIN_AGE_SECS", 1800)?,

            max_entries_per_archive: env_parse("MAX_ENTRIES_PER_ARCHIVE", 100_000usize)?,
            max_entry_bytes: env_parse("MAX_ENTRY_BYTES", 209_715_200u64)?,
            max_total_uncompressed_bytes: env_parse(
                "MAX_TOTAL_UNCOMPRESSED_BYTES",
                8_589_934_592u64,
            )?,
            max_compression_ratio: env_parse("MAX_COMPRESSION_RATIO", 500u64)?,
            max_image_width: env_parse("MAX_IMAGE_WIDTH", 20_000u32)?,
            max_image_height: env_parse("MAX_IMAGE_HEIGHT", 20_000u32)?,
            max_image_alloc_bytes: env_parse("MAX_IMAGE_ALLOC_BYTES", 536_870_912u64)?,
            max_archives_per_batch: env_parse("MAX_ARCHIVES_PER_BATCH", 64usize)?,
            max_archive_bytes: env_parse("MAX_ARCHIVE_BYTES", 3_758_096_384u64)?,
            max_batch_total_bytes: env_parse("MAX_BATCH_TOTAL_BYTES", 10_737_418_240u64)?,

            model_allow_http: env_parse("MODEL_ALLOW_HTTP", false)?,
            model_sha_required: env_parse("MODEL_SHA_REQUIRED", false)?,
            model_max_redirects: env_parse("MODEL_MAX_REDIRECTS", 2usize)?,

            s3_ingest_auth_required: env_parse("S3_INGEST_AUTH_REQUIRED", true)?,

            execution_provider: ExecutionProvider::parse(&env_str(
                "ORT_EXECUTION_PROVIDER",
                "cpu",
            ))?,
            gpu_device_id: parse_gpu_device_id()?,
            gpu_memory_limit_bytes: parse_gpu_memory_limit()?,
            enable_tf32: env_parse::<bool>("ORT_ENABLE_TF32", false)?,
            enable_fp16: env_parse::<bool>("ORT_ENABLE_FP16", false)?,

            #[cfg(feature = "s3")]
            s3: s3_settings_from_env()?,
            #[cfg(feature = "queue")]
            sqs: sqs_settings_from_env()?,
            #[cfg(feature = "rabbitmq")]
            rabbitmq: rabbitmq_settings_from_env()?,
        };

        if cfg.roi_area_min >= cfg.roi_area_max {
            anyhow::bail!("ROI_AREA_MIN must be < ROI_AREA_MAX");
        }
        if cfg.fp_crop_conf_max <= cfg.yolo_conf_threshold {
            anyhow::bail!("FP_CROP_CONF_MAX must be > YOLO_CONF_THRESHOLD");
        }
        cfg.validate_finite()?;
        Ok(cfg)
    }

    /// Rejects non-finite float knobs (`NaN`, `inf`). `f32::from_str`/
    /// `f64::from_str` accept the literals "NaN"/"inf", and neither `clamp`
    /// nor `max` drops NaN — it would silently poison blur sigma, thresholds
    /// and ratio checks. The env is operator-controlled, so this turns a typo
    /// into a startup error instead of a degraded (or DoS-prone) pipeline.
    fn validate_finite(&self) -> Result<()> {
        let checks: [(&str, f64); 21] = [
            ("HEAD_FALLBACK_FRACTION", self.head_fallback_fraction as f64),
            ("SEGMENTER_MIN_BOX", self.segmenter_min_box_px as f64),
            ("YOLO_CONF_THRESHOLD", self.yolo_conf_threshold as f64),
            (
                "YOLO_CONF_THRESHOLD_ACTIVE",
                self.yolo_conf_threshold_active as f64,
            ),
            ("YOLO_NMS_IOU", self.yolo_nms_iou as f64),
            ("FP_CROP_CONF_MAX", self.fp_crop_conf_max as f64),
            ("INITIAL_BLUR_SIGMA", self.initial_blur_sigma as f64),
            ("BLUR_HULL_MARGIN_PCT", self.blur_hull_margin_pct as f64),
            ("BLUR_ELLIPSE_MARGIN", self.blur_ellipse_margin as f64),
            ("MASK_FEATHER_SIGMA", self.mask_feather_sigma as f64),
            (
                "CLASSIFIER_CONFIRM_THRESHOLD",
                self.classifier_confirm_threshold as f64,
            ),
            ("ROI_EPS_PX", self.roi_eps_px as f64),
            ("ROI_RDP_EPSILON", self.roi_rdp_epsilon as f64),
            ("ROI_AREA_MIN", self.roi_area_min),
            ("ROI_AREA_MAX", self.roi_area_max),
            ("ROI_MARGIN_PCT", self.roi_margin_pct),
            ("ROI_REEXTRACT_MIN_IOU", self.roi_reextract_min_iou),
            ("RETRAIN_MIN_ACCURACY", self.retrain_min_accuracy as f64),
            ("RETRAIN_REGRESSION_EPS", self.retrain_regression_eps as f64),
            ("RETRAIN_HOLDOUT_FRACTION", self.retrain_holdout_fraction as f64),
            ("RETENTION_MAX_GB", self.retention_max_gb),
        ];
        for (name, value) in checks {
            if !value.is_finite() {
                anyhow::bail!("{name} must be a finite number, got {value}");
            }
        }
        Ok(())
    }

    /// Effective per-image concurrency: env override, else auto-detected
    /// (cores ≤ 4 → all cores; cores > 4 → cores − 2).
    pub fn effective_concurrency(&self) -> usize {
        self.max_concurrent_images
            .unwrap_or_else(default_concurrency)
            .max(1)
    }

    /// Whether the STORE-output retention policy is active: enabled **and** at
    /// least one rule (age or size) is on. Setting `RETENTION_ENABLED=false`
    /// or both thresholds to 0 disables the periodic cleanup entirely.
    pub fn retention_active(&self) -> bool {
        self.retention_enabled && (self.retention_max_days > 0 || self.retention_max_gb > 0.0)
    }

    /// Feather sigma for the ACTIVE mask edge: a configured
    /// `MASK_FEATHER_SIGMA > 0` wins over the (possibly huge) per-face blur
    /// sigma; `0` keeps the historical behavior (feather == blur sigma).
    pub fn mask_feather(&self, blur_sigma: f32) -> f32 {
        if self.mask_feather_sigma > 0.0 {
            self.mask_feather_sigma.min(blur_sigma)
        } else {
            blur_sigma
        }
    }
}

// ─── Runtime-tunable configuration (operator settings UI) ───────────────────

/// Input kind of a single runtime knob, used by the operator UI to pick the
/// right widget (number slider/input, checkbox or `<select>`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum FieldKind {
    Float,
    Int,
    Bool,
    Enum,
}

/// Schema + current/default value of one hot-tunable knob, serialized as-is to
/// `GET /operator/settings.json` so the browser renders the form from data
/// (no matching server-side HTML template to keep).
#[derive(Debug, Clone, serde::Serialize)]
pub struct FieldSpec {
    pub key: &'static str,
    /// Backing `.env` variable (the startup default this knob overrides).
    pub env: &'static str,
    pub label: &'static str,
    pub hint: &'static str,
    pub kind: FieldKind,
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub step: Option<f64>,
    pub options: Vec<&'static str>,
    pub value: serde_json::Value,
    pub default: serde_json::Value,
}

/// Hot-reloadable service configuration. Holds an `ArcSwap<Config>` (the
/// *effective* snapshots every reader loads per job via `snapshot()`) plus an
/// immutable record of the env-only defaults (for "reset to .env" and the UI's
/// `default` column). Runtime overrides are persisted to `DATA_DIR/runtime_config.json`
/// so a restart keeps the tuned values on top of the same env.
pub struct RuntimeConfig {
    /// Shared, hot-swappable effective config. Every clone points at the same
    /// `ArcSwap`, so an operator patch applied on *any* handle (the settings
    /// UI's) is immediately visible to every consumer (the ZipProcessor, the
    /// retention/background loops) — `ArcSwap` itself is not `Clone`, sharing it
    /// behind an `Arc` makes `RuntimeConfig::clone` an O(1) refcount bump
    /// instead of a deep `Config` copy.
    inner: Arc<arc_swap::ArcSwap<Config>>,
    base: Arc<Config>,
    overrides_file: Option<PathBuf>,
}

impl Clone for RuntimeConfig {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            base: self.base.clone(),
            overrides_file: self.overrides_file.clone(),
        }
    }
}

impl RuntimeConfig {
    /// Builds the effective config from the env, then applies
    /// `DATA_DIR/runtime_config.json` (if present) on top. A corrupt/unknown
    /// override aborts startup — the file is machine-written, a typo there is a
    /// configuration error, not a runtime accident.
    pub fn from_env() -> Result<Self> {
        let base = Arc::new(Config::from_env()?);
        let overrides_file = Some(base.data_dir.join(RUNTIME_CONFIG_FILENAME));
        let mut effective = base.as_ref().clone();
        if let Ok(text) = std::fs::read_to_string(base.data_dir.join(RUNTIME_CONFIG_FILENAME)) {
            let map: serde_json::Map<String, serde_json::Value> =
                serde_json::from_str(&text).context("invalid runtime_config.json")?;
            effective.apply_runtime_patch(&map)?;
        }
        Ok(Self {
            inner: Arc::new(arc_swap::ArcSwap::from_pointee(effective)),
            base,
            overrides_file,
        })
    }

    /// A non-persisting handle for unit tests / fixed configurations.
    #[allow(dead_code)] // only used from #[cfg(test)] modules and edge cases
    pub fn fixed(cfg: Config) -> Self {
        Self {
            inner: Arc::new(arc_swap::ArcSwap::from_pointee(cfg.clone())),
            base: Arc::new(cfg),
            overrides_file: None,
        }
    }

    /// Cheap snapshot of the effective config; callers pass `&cfg` to the
    /// detections/pipeline and must not hold the guard across awaits.
    pub fn snapshot(&self) -> Arc<Config> {
        self.inner.load_full()
    }

    /// The env-only defaults (what the service would run with no overrides).
    #[allow(dead_code)] // currently used internally via `field_specs`/reset paths
    pub fn base(&self) -> Arc<Config> {
        self.base.clone()
    }

    /// Applies a patch map (key → value) on top of the current effective
    /// config, atomically swapping it in and persisting it. Only the keys
    /// present in `patch` are changed; unknown keys are an error (fail-fast).
    pub fn apply_patch(&self, patch: &serde_json::Map<String, serde_json::Value>) -> Result<()> {
        let mut next = self.inner.load_full().as_ref().clone();
        next.apply_runtime_patch(patch)?;
        let next = Arc::new(next);
        self.inner.store(next.clone());
        if let Some(path) = &self.overrides_file {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            // Atomic write-rename so a crash mid-write never leaves a corrupt
            // overrides file that would fail the next startup.
            let tmp = path.with_extension("json.tmp");
            std::fs::write(&tmp, serde_json::to_vec_pretty(&next.runtime_values())?)?;
            std::fs::rename(&tmp, path)?;
        }
        Ok(())
    }

    /// Resets a single knob back to its env default (or removes the whole
    /// overrides file when `clear_all`).
    pub fn reset_to_env(&self, key: &str, clear_all: bool) -> Result<()> {
        let mut patch = serde_json::Map::new();
        if clear_all {
            if let Some(path) = &self.overrides_file {
                if path.exists() {
                    std::fs::remove_file(path)?;
                }
            }
            self.inner.store(self.base.clone());
            return Ok(());
        }
        let value = self
            .config_value(&self.base, key)
            .ok_or_else(|| anyhow::anyhow!("unknown runtime setting '{key}'"))?;
        patch.insert(key.to_string(), value);
        self.apply_patch(&patch)
    }

    /// Reads one knob (optionally in the given origin) as a JSON value.
    pub fn config_value(&self, cfg: &Config, key: &str) -> Option<serde_json::Value> {
        Config::runtime_field_specs(cfg, cfg)
            .into_iter()
            .find(|f| f.key == key)
            .map(|f| f.value)
    }

    /// Schema + effective/default values for the settings UI.
    pub fn field_specs(&self) -> Vec<FieldSpec> {
        Config::runtime_field_specs(&self.snapshot(), &self.base)
    }
}

/// Standard knobs exposed by the operator settings UI.
impl Config {
    /// List of hot-tunable field specs for the settings UI (`current` builds
    /// the effective values, `defaults` the "reset" baseline).
    fn runtime_field_specs(current: &Config, defaults: &Config) -> Vec<FieldSpec> {
        let f = |key, label, hint, kind, min, max, step, options: Vec<&'static str>| field_spec(
            current,
            defaults,
            key,
            label,
            hint,
            kind,
            min,
            max,
            step,
            options,
        );
        vec![
            f("yolo_conf_threshold", "Soglia detector (LEARNING)",
              "YOLO_CONF_THRESHOLD — confidenza minima dei volti accettati durante la fase di apprendimento.", FieldKind::Float, Some(0.01), Some(0.99), Some(0.01), vec![]),
            f("yolo_conf_threshold_active", "Soglia detector (ACTIVE)",
              "YOLO_CONF_THRESHOLD_ACTIVE — soglia dei volti da anonimizzare in tempo reale. Più bassa = più volti rilevati.", FieldKind::Float, Some(0.01), Some(0.99), Some(0.01), vec![]),
            f("yolo_nms_iou", "NMS IoU",
              "YOLO_NMS_IOU — soppressione dei riquadri sovrapposti.", FieldKind::Float, Some(0.05), Some(0.95), Some(0.05), vec![]),
            f("fp_crop_conf_max", "FP_CROP_CONF_MAX",
              "Confidenza massima sotto cui un riquadro viene considerato falso positivo e salvato nel dataset.", FieldKind::Float, Some(0.01), Some(0.99), Some(0.01), vec![]),
            f("classifier_confirm_threshold", "Soglia conferma viso",
              "CLASSIFIER_CONFIRM_THRESHOLD — probabilità minima del classificatore per confermare un viso.", FieldKind::Float, Some(0.01), Some(1.0), Some(0.01), vec![]),
            f("classifier_enforce", "Classificatore obbligatorio",
              "CLASSIFIER_ENFORCE — se ON, i visi sotto soglia non vengono anonimizzati.", FieldKind::Bool, None, None, None, vec![]),
            f("anon_mode", "Modalità anonimizzazione",
              "ANON_MODE — blur (sfocatura gaussiana) o pixelate (mosaico stile Street View).", FieldKind::Enum, None, None, None, vec!["blur", "pixelate"]),
            f("pixelate_cell_px", "Dimensione cella mosaico",
              "PIXELATE_CELL_PX — lato del blocco del pixelate in pixel.", FieldKind::Int, Some(2.0), Some(128.0), Some(1.0), vec![]),
            f("initial_blur_sigma", "Sigma sfocatura",
              "INITIAL_BLUR_SIGMA — intensità della sfocatura gaussiana di base.", FieldKind::Float, Some(1.0), Some(128.0), Some(1.0), vec![]),
            f("blur_hull_margin_pct", "Margine hull (%)",
              "BLUR_HULL_MARGIN_PCT — estensione percentuale della maschera intorno al contorno del viso.", FieldKind::Float, Some(0.0), Some(0.5), Some(0.01), vec![]),
            f("blur_ellipse_margin", "Margine ellisse",
              "BLUR_ELLIPSE_MARGIN — ingrandimento dell'ellisse del viso.", FieldKind::Float, Some(0.0), Some(0.5), Some(0.01), vec![]),
            f("mask_feather_sigma", "Feather maschera",
              "MASK_FEATHER_SIGMA — sfumatura del bordo della maschera (0 = come sigma sfocatura).", FieldKind::Float, Some(0.0), Some(64.0), Some(1.0), vec![]),
            f("segmenter_min_box_px", "Min box segmenter",
              "SEGMENTER_MIN_BOX — box sotto questa larghezza usa hull/ellisse geometrica invece del segmenter.", FieldKind::Float, Some(0.0), Some(512.0), Some(1.0), vec![]),
            f("head_fallback_fraction", "Frazione testa (fallback)",
              "HEAD_FALLBACK_FRACTION — frazione alta del box persona usata come regione testa.", FieldKind::Float, Some(0.05), Some(0.9), Some(0.05), vec![]),
            f("jpeg_quality", "Qualità JPEG",
              "JPEG_QUALITY — qualità di compressione degli output.", FieldKind::Int, Some(1.0), Some(100.0), Some(1.0), vec![]),
            f("output_max_side_px", "Lato max output",
              "OUTPUT_MAX_SIDE — ridimensiona il lato lungo dell'output (0 = off, ≥ 128).", FieldKind::Int, Some(0.0), Some(8192.0), Some(64.0), vec![]),
        ]
    }

    /// Applies a validated set of runtime overrides to this config. Keys not in
    /// the `runtime_field_specs` allowlist are an error (typo guard), values are
    /// clamped the same way `from_env` clamps them, and cross-field invariants
    /// (FP_CROP_CONF_MAX > YOLO_CONF_THRESHOLD) are re-checked afterwards.
    pub fn apply_runtime_patch(&mut self, patch: &serde_json::Map<String, serde_json::Value>) -> Result<()> {
        for (key, value) in patch {
            match key.as_str() {
                "yolo_conf_threshold" => self.yolo_conf_threshold = parse_float(key, value)?.clamp(0.01, 0.99),
                "yolo_conf_threshold_active" => self.yolo_conf_threshold_active = parse_float(key, value)?.clamp(0.01, 0.99),
                "yolo_nms_iou" => self.yolo_nms_iou = parse_float(key, value)?.clamp(0.05, 0.95),
                "fp_crop_conf_max" => self.fp_crop_conf_max = parse_float(key, value)?.clamp(0.01, 0.99),
                "classifier_confirm_threshold" => self.classifier_confirm_threshold = parse_float(key, value)?.clamp(0.01, 1.0),
                "classifier_enforce" => self.classifier_enforce = parse_bool(key, value)?,
                "anon_mode" => self.anon_mode = AnonMode::parse(&parse_string(key, value)?)?,
                "pixelate_cell_px" => self.pixelate_cell_px = parse_int(key, value)?.clamp(2, 128),
                "initial_blur_sigma" => self.initial_blur_sigma = parse_float(key, value)?.max(1.0),
                "blur_hull_margin_pct" => self.blur_hull_margin_pct = parse_float(key, value)?.clamp(0.0, 0.5),
                "blur_ellipse_margin" => self.blur_ellipse_margin = parse_float(key, value)?.clamp(0.0, 0.5),
                "mask_feather_sigma" => self.mask_feather_sigma = parse_float(key, value)?.clamp(0.0, 64.0),
                "segmenter_min_box_px" => self.segmenter_min_box_px = parse_float(key, value)?.max(0.0),
                "head_fallback_fraction" => self.head_fallback_fraction = parse_float(key, value)?.clamp(0.05, 0.9),
                "jpeg_quality" => self.jpeg_quality = parse_int(key, value)?.clamp(1, 100) as u8,
                "output_max_side_px" => {
                    let v = parse_int(key, value)?;
                    if v != 0 && v < 128 {
                        anyhow::bail!("output_max_side_px must be 0 or >= 128, got {v}");
                    }
                    self.output_max_side_px = v;
                }
                other => anyhow::bail!("unknown runtime setting '{other}'"),
            }
        }
        if self.fp_crop_conf_max <= self.yolo_conf_threshold {
            anyhow::bail!(
                "FP_CROP_CONF_MAX ({}) must be > YOLO_CONF_THRESHOLD ({})",
                self.fp_crop_conf_max,
                self.yolo_conf_threshold
            );
        }
        Ok(())
    }

    /// Serializes the current values of every runtime knob (the persisted
    /// `runtime_config.json` shape — restores exactly this state on reboots).
    pub fn runtime_values(&self) -> serde_json::Map<String, serde_json::Value> {
        let mut out = serde_json::Map::new();
        for field in Self::runtime_field_specs(self, self) {
            out.insert(field.key.to_string(), field.value);
        }
        out
    }
}

/// `.env` variable backing each hot-tunable knob. Kept explicit because the key
/// is a UI identifier while the env name is what operators actually set (they
/// are not always derivable: `segmenter_min_box_px` → `SEGMENTER_MIN_BOX`).
/// Must stay in sync with `runtime_field_specs`; the docs guard test fails
/// otherwise.
fn runtime_env_name(key: &str) -> &'static str {
    match key {
        "yolo_conf_threshold" => "YOLO_CONF_THRESHOLD",
        "yolo_conf_threshold_active" => "YOLO_CONF_THRESHOLD_ACTIVE",
        "yolo_nms_iou" => "YOLO_NMS_IOU",
        "fp_crop_conf_max" => "FP_CROP_CONF_MAX",
        "classifier_confirm_threshold" => "CLASSIFIER_CONFIRM_THRESHOLD",
        "classifier_enforce" => "CLASSIFIER_ENFORCE",
        "anon_mode" => "ANON_MODE",
        "pixelate_cell_px" => "PIXELATE_CELL_PX",
        "initial_blur_sigma" => "INITIAL_BLUR_SIGMA",
        "blur_hull_margin_pct" => "BLUR_HULL_MARGIN_PCT",
        "blur_ellipse_margin" => "BLUR_ELLIPSE_MARGIN",
        "mask_feather_sigma" => "MASK_FEATHER_SIGMA",
        "segmenter_min_box_px" => "SEGMENTER_MIN_BOX",
        "head_fallback_fraction" => "HEAD_FALLBACK_FRACTION",
        "jpeg_quality" => "JPEG_QUALITY",
        "output_max_side_px" => "OUTPUT_MAX_SIDE",
        _ => "",
    }
}

#[allow(clippy::too_many_arguments)] // internal builder; the `f` closure keeps the 16 call sites compact
fn field_spec(
    current: &Config,
    defaults: &Config,
    key: &'static str,
    label: &'static str,
    hint: &'static str,
    kind: FieldKind,
    min: Option<f64>,
    max: Option<f64>,
    step: Option<f64>,
    options: Vec<&'static str>,
) -> FieldSpec {
    FieldSpec {
        key,
        env: runtime_env_name(key),
        label,
        hint,
        kind,
        min,
        max,
        step,
        options,
        value: current_field_value(current, key),
        default: current_field_value(defaults, key),
    }
}

/// Extracts the current value of a runtime knob. Keys must stay in sync with
/// `runtime_field_specs`.
fn current_field_value(cfg: &Config, key: &str) -> serde_json::Value {
    match key {
        "yolo_conf_threshold" => serde_json::json!(cfg.yolo_conf_threshold),
        "yolo_conf_threshold_active" => serde_json::json!(cfg.yolo_conf_threshold_active),
        "yolo_nms_iou" => serde_json::json!(cfg.yolo_nms_iou),
        "fp_crop_conf_max" => serde_json::json!(cfg.fp_crop_conf_max),
        "classifier_confirm_threshold" => serde_json::json!(cfg.classifier_confirm_threshold),
        "classifier_enforce" => serde_json::json!(cfg.classifier_enforce),
        "anon_mode" => serde_json::json!(cfg.anon_mode.as_str()),
        "pixelate_cell_px" => serde_json::json!(cfg.pixelate_cell_px),
        "initial_blur_sigma" => serde_json::json!(cfg.initial_blur_sigma),
        "blur_hull_margin_pct" => serde_json::json!(cfg.blur_hull_margin_pct),
        "blur_ellipse_margin" => serde_json::json!(cfg.blur_ellipse_margin),
        "mask_feather_sigma" => serde_json::json!(cfg.mask_feather_sigma),
        "segmenter_min_box_px" => serde_json::json!(cfg.segmenter_min_box_px),
        "head_fallback_fraction" => serde_json::json!(cfg.head_fallback_fraction),
        "jpeg_quality" => serde_json::json!(cfg.jpeg_quality),
        "output_max_side_px" => serde_json::json!(cfg.output_max_side_px),
        _ => serde_json::Value::Null,
    }
}

impl AnonMode {
    pub fn as_str(self) -> &'static str {
        match self {
            AnonMode::Blur => "blur",
            AnonMode::Pixelate => "pixelate",
        }
    }
}

fn parse_float(key: &str, value: &serde_json::Value) -> Result<f32> {
    let v = value
        .as_f64()
        .ok_or_else(|| anyhow::anyhow!("{key}: expected a number"))?;
    if !v.is_finite() {
        anyhow::bail!("{key}: expected a finite number");
    }
    Ok(v as f32)
}
fn parse_int(key: &str, value: &serde_json::Value) -> Result<u32> {
    value
        .as_u64()
        .and_then(|v| u32::try_from(v).ok())
        .ok_or_else(|| anyhow::anyhow!("{key}: expected a positive integer"))
}
fn parse_bool(key: &str, value: &serde_json::Value) -> Result<bool> {
    value
        .as_bool()
        .ok_or_else(|| anyhow::anyhow!("{key}: expected true/false"))
}
fn parse_string(key: &str, value: &serde_json::Value) -> Result<String> {
    value
        .as_str()
        .map(str::to_string)
        .ok_or_else(|| anyhow::anyhow!("{key}: expected a string"))
}

#[cfg(test)]
impl Config {
    /// Deterministic env-independent configuration for unit tests.
    /// Mirrors the `from_env` defaults (paths are inert for tests).
    pub(crate) fn test_default() -> Self {
        Config {
            bind_addr: "127.0.0.1:0".into(),
            body_limit_bytes: 3_758_096_384,
            request_timeout_secs: 0,
            max_concurrent_images: Some(2),
            jpeg_quality: 95,
            yolo_model_url: String::new(),
            classifier_model_url: None,
            model_cache_dir: PathBuf::from("/tmp/av-models-cache"),
            yolo_sha256: None,
            classifier_sha256: None,
            head_fallback_enabled: false,
            coco_model_url: String::new(),
            coco_sha256: None,
            head_fallback_fraction: 0.30,
            yolo_input_size: 640,
            mask_segmenter: MaskSegmenter::Off,
            selfie_segmenter_url: String::new(),
            selfie_segmenter_sha256: None,
            segmenter_min_box_px: 0.0,
            output_format: OutputFormat::Keep,
            output_max_side_px: 0,
            data_dir: PathBuf::from("/tmp/av-data"),
            dataset_fp_dir: PathBuf::from("/tmp/av-fp"),
            dataset_seed_real_faces_dir: PathBuf::from("/tmp/av-seed"),
            models_backup_dir: PathBuf::from("/tmp/av-backup"),
            yolo_conf_threshold: 0.20,
            yolo_conf_threshold_active: 0.20,
            yolo_nms_iou: 0.45,
            fp_crop_conf_max: 0.50,
            initial_blur_sigma: 20.0,
            blur_hull_margin_pct: 0.10,
            blur_ellipse_margin: 0.05,
            mask_feather_sigma: 0.0,
            classifier_confirm_threshold: 0.5,
            classifier_enforce: true,
            anon_mode: AnonMode::Blur,
            pixelate_cell_px: 12,
            learning_days: 30,
            roi_eps_px: 50.0,
            roi_min_samples: 15,
            roi_rdp_epsilon: 5.0,
            roi_area_min: 0.10,
            roi_area_max: 0.90,
            roi_margin_pct: 0.05,
            roi_reextract_enabled: true,
            roi_reextract_window_days: 7,
            roi_reextract_min_iou: 0.6,
            retrain_schedule: ScheduleSpec { hour: 3, minute: 0 },
            retrain_min_accuracy: 0.85,
            retrain_regression_eps: 0.0,
            retrain_holdout_fraction: 0.10,
            retrain_epochs: 1,
            retrain_batch_size: 8,
            operator_api_key: OperatorKey::new(None),
            retention_enabled: false, // tests create their own temp data dirs
            retention_max_days: 30,
            retention_max_gb: 20.0,
            retention_interval_secs: 3600,
            retention_min_age_secs: 1800,
            max_entries_per_archive: 100_000,
            max_entry_bytes: 209_715_200,
            max_total_uncompressed_bytes: 8_589_934_592,
            max_compression_ratio: 500,
            max_image_width: 20_000,
            max_image_height: 20_000,
            max_image_alloc_bytes: 536_870_912,
            max_archives_per_batch: 64,
            max_archive_bytes: 3_758_096_384,
            max_batch_total_bytes: 10_737_418_240,
            model_allow_http: false,
            model_sha_required: false,
            model_max_redirects: 2,
            s3_ingest_auth_required: true,
            execution_provider: ExecutionProvider::Cpu,
            gpu_device_id: 0,
            gpu_memory_limit_bytes: None,
            enable_tf32: false,
            enable_fp16: false,
            #[cfg(feature = "s3")]
            s3: None,
            #[cfg(feature = "queue")]
            sqs: None,
            #[cfg(feature = "rabbitmq")]
            rabbitmq: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedule_parses_and_rolls_to_next_day() {
        let s = ScheduleSpec::parse("03:00").unwrap();
        assert_eq!(s, ScheduleSpec { hour: 3, minute: 0 });
        // At 04:00 the next occurrence is tomorrow 03:00 → 23h.
        let now = chrono::NaiveTime::from_hms_opt(4, 0, 0).unwrap();
        assert_eq!(s.next_occurrence_from(now).as_secs(), 23 * 3600);
        // At 02:00 the next occurrence is today 03:00 → 1h.
        let now = chrono::NaiveTime::from_hms_opt(2, 0, 0).unwrap();
        assert_eq!(s.next_occurrence_from(now).as_secs(), 3600);
        assert!(ScheduleSpec::parse("25:00").is_err());
        assert!(ScheduleSpec::parse("3pm").is_err());
    }

    #[test]
    fn default_concurrency_is_positive() {
        assert!(default_concurrency() >= 1);
    }

    #[test]
    fn input_size_validation() {
        assert_eq!(validate_input_size(640, "t").unwrap(), 640);
        assert_eq!(validate_input_size(512, "t").unwrap(), 512);
        assert!(validate_input_size(620, "t").is_err()); // non-multiple of 32
        assert!(validate_input_size(1280, "t").is_ok());
        assert!(validate_input_size(1216, "t").is_ok());
        assert!(validate_input_size(1281, "t").is_err()); // > 1280
        assert!(validate_input_size(224, "t").is_err()); // < 256
    }

    #[test]
    fn mask_segmenter_parsing() {
        assert_eq!(MaskSegmenter::parse("off").unwrap(), MaskSegmenter::Off);
        assert_eq!(MaskSegmenter::parse("").unwrap(), MaskSegmenter::Off);
        assert_eq!(
            MaskSegmenter::parse("mediapipe").unwrap(),
            MaskSegmenter::Mediapipe
        );
        assert_eq!(
            MaskSegmenter::parse("SELFIE").unwrap(),
            MaskSegmenter::Mediapipe
        );
        assert!(MaskSegmenter::parse("portrait").is_err());
    }

    #[test]
    fn blur_and_classifier_env_parsing() {
        // Defaults (unset).
        std::env::remove_var("BLUR_ELLIPSE_MARGIN");
        std::env::remove_var("MASK_FEATHER_SIGMA");
        std::env::remove_var("CLASSIFIER_CONFIRM_THRESHOLD");
        assert_eq!(env_parse::<f32>("BLUR_ELLIPSE_MARGIN", 0.05).unwrap(), 0.05);
        assert_eq!(env_parse::<f32>("MASK_FEATHER_SIGMA", 0.0).unwrap(), 0.0);
        assert_eq!(
            env_parse::<f32>("CLASSIFIER_CONFIRM_THRESHOLD", 0.5).unwrap(),
            0.5
        );
        // Parsed from env.
        std::env::set_var("BLUR_ELLIPSE_MARGIN", "0.15");
        std::env::set_var("MASK_FEATHER_SIGMA", "8");
        std::env::set_var("CLASSIFIER_CONFIRM_THRESHOLD", "0.2");
        assert_eq!(env_parse::<f32>("BLUR_ELLIPSE_MARGIN", 0.05).unwrap(), 0.15);
        assert_eq!(env_parse::<f32>("MASK_FEATHER_SIGMA", 0.0).unwrap(), 8.0);
        assert_eq!(
            env_parse::<f32>("CLASSIFIER_CONFIRM_THRESHOLD", 0.5).unwrap(),
            0.2
        );
        // CLASSIFIER_ENFORCE default true, switchable off.
        assert!(env_parse::<bool>("CLASSIFIER_ENFORCE", true).unwrap());
        std::env::set_var("CLASSIFIER_ENFORCE", "false");
        assert!(!env_parse::<bool>("CLASSIFIER_ENFORCE", true).unwrap());
        std::env::remove_var("CLASSIFIER_ENFORCE");
        std::env::remove_var("BLUR_ELLIPSE_MARGIN");
        std::env::remove_var("MASK_FEATHER_SIGMA");
        std::env::remove_var("CLASSIFIER_CONFIRM_THRESHOLD");
    }

    /// `f32::from_str` accepts "NaN"/"inf" and neither `clamp` nor `max`
    /// drops NaN — a typo in the env must fail loudly instead of poisoning
    /// the pipeline.
    #[test]
    fn non_finite_floats_are_rejected() {
        let cfg = Config::test_default();
        cfg.validate_finite().unwrap();

        let mut nan_sigma = Config::test_default();
        nan_sigma.initial_blur_sigma = f32::NAN;
        assert!(nan_sigma.validate_finite().is_err());

        let mut inf_gb = Config::test_default();
        inf_gb.retention_max_gb = f64::INFINITY;
        assert!(inf_gb.validate_finite().is_err());

        let mut nan_iou = Config::test_default();
        nan_iou.roi_reextract_min_iou = f64::NAN;
        assert!(nan_iou.validate_finite().is_err());
    }

    /// A runtime patch never accepts a non-numeric value; JSON itself cannot
    /// carry NaN/inf, so the finite guard in `parse_float` is belt-and-braces.
    #[test]
    fn runtime_patch_rejects_non_numeric() {
        let mut cfg = Config::test_default();
        let mut patch = serde_json::Map::new();
        patch.insert("initial_blur_sigma".to_string(), serde_json::Value::Null);
        assert!(cfg.apply_runtime_patch(&patch).is_err());
        patch.insert("unknown_knob".to_string(), serde_json::json!(1.0));
        assert!(cfg.apply_runtime_patch(&patch).is_err());
    }

    /// The operator key must never reach a log line, not even through a `{:?}`
    /// on the whole `Config`.
    #[test]
    fn operator_key_debug_is_redacted() {
        let key = OperatorKey::new(Some("super-secret".to_string()));
        let dbg = format!("{key:?}");
        assert!(!dbg.contains("super-secret"), "key leaked in {dbg}");
        assert!(dbg.contains("redacted"), "unexpected Debug: {dbg}");
        assert_eq!(key.as_deref(), Some("super-secret"));

        let mut cfg = Config::test_default();
        cfg.operator_api_key = OperatorKey::new(Some("super-secret".to_string()));
        assert!(!format!("{cfg:?}").contains("super-secret"));
        assert!(format!("{:?}", Config::test_default()).contains("none"));
    }

    #[test]
    fn mask_feather_mode() {
        let cfg = Config::test_default();
        // Auto mode: feather follows the blur sigma.
        assert_eq!(cfg.mask_feather(37.5), 37.5);
        // Configured cap wins and never exceeds the blur sigma.
        let mut capped = Config::test_default();
        capped.mask_feather_sigma = 8.0;
        assert_eq!(capped.mask_feather(37.5), 8.0);
        assert_eq!(capped.mask_feather(5.0), 5.0); // small faces: sigma wins
    }

    /// Guard against documentation drift: every hot-tunable knob must have a
    /// backing `.env` variable *and* be documented in `.env.example`, so a new
    /// runtime knob cannot ship undocumented. The operator UI serves the same
    /// names via `GET /operator/settings.json`.
    #[test]
    fn runtime_knobs_are_documented_in_env_example() {
        let cfg = Config::test_default();
        let specs = Config::runtime_field_specs(&cfg, &cfg);
        assert!(
            specs.len() >= 16,
            "expected the runtime knob set, got {}",
            specs.len()
        );
        let doc = std::fs::read_to_string(".env.example")
            .expect(".env.example must exist at the crate root");
        for f in &specs {
            assert!(
                !f.env.is_empty(),
                "runtime knob '{}' has no .env mapping: add it to runtime_env_name",
                f.key
            );
            assert!(
                doc.contains(f.env),
                "runtime knob '{}' is not documented in .env.example as {}",
                f.key,
                f.env
            );
        }
    }

    #[test]
    fn output_format_parsing() {
        assert_eq!(OutputFormat::parse("").unwrap(), OutputFormat::Keep);
        assert_eq!(OutputFormat::parse("keep").unwrap(), OutputFormat::Keep);
        assert_eq!(OutputFormat::parse("auto").unwrap(), OutputFormat::Keep);
        assert_eq!(OutputFormat::parse("jpeg").unwrap(), OutputFormat::Jpeg);
        assert_eq!(OutputFormat::parse("JPG").unwrap(), OutputFormat::Jpeg);
        assert_eq!(OutputFormat::parse("png").unwrap(), OutputFormat::Png);
        assert!(OutputFormat::parse("webp").is_err());
        assert!(OutputFormat::parse("bmp").is_err());
    }

    #[test]
    fn output_max_side_validation() {
        std::env::remove_var("OUTPUT_MAX_SIDE");
        assert_eq!(parse_output_max_side().unwrap(), 0); // unset → off
        std::env::set_var("OUTPUT_MAX_SIDE", "1920");
        assert_eq!(parse_output_max_side().unwrap(), 1920);
        std::env::set_var("OUTPUT_MAX_SIDE", "0");
        assert_eq!(parse_output_max_side().unwrap(), 0);
        std::env::set_var("OUTPUT_MAX_SIDE", "64");
        assert!(parse_output_max_side().is_err());
        std::env::remove_var("OUTPUT_MAX_SIDE");
    }

    #[test]
    fn gpu_settings_parsing() {
        std::env::remove_var("ORT_CUDA_DEVICE_ID");
        std::env::remove_var("ORT_CUDA_MEMORY_LIMIT_BYTES");
        assert_eq!(parse_gpu_device_id().unwrap(), 0); // unset → device 0
        assert_eq!(parse_gpu_memory_limit().unwrap(), None);
        std::env::set_var("ORT_CUDA_DEVICE_ID", "1");
        assert_eq!(parse_gpu_device_id().unwrap(), 1);
        std::env::set_var("ORT_CUDA_DEVICE_ID", "-1"); // CUDA indices are unsigned
        assert_eq!(parse_gpu_device_id().unwrap(), 0);
        std::env::set_var("ORT_CUDA_MEMORY_LIMIT_BYTES", "8589934592"); // 8 GiB
        assert_eq!(parse_gpu_memory_limit().unwrap(), Some(8 << 30));
        std::env::set_var("ORT_CUDA_MEMORY_LIMIT_BYTES", "0"); // a zero cap is a config error
        assert!(parse_gpu_memory_limit().is_err());
        std::env::set_var("ORT_CUDA_MEMORY_LIMIT_BYTES", "not-a-number");
        assert!(parse_gpu_memory_limit().is_err());
        std::env::remove_var("ORT_CUDA_DEVICE_ID");
        std::env::remove_var("ORT_CUDA_MEMORY_LIMIT_BYTES");
    }

    #[cfg(feature = "s3")]
    #[test]
    fn s3_settings_disabled_by_default() {
        std::env::set_var("S3_ENABLED", "false");
        std::env::remove_var("S3_ENDPOINT");
        assert!(s3_settings_from_env().unwrap().is_none());
        std::env::remove_var("S3_ENABLED");
    }

    #[cfg(feature = "s3")]
    #[test]
    fn s3_settings_parsing() {
        std::env::set_var("S3_ENABLED", "true");
        std::env::set_var("S3_ENDPOINT", "http://localhost:9000/");
        std::env::remove_var("S3_FORCE_PATH_STYLE");
        std::env::set_var("S3_BUCKET_INPUT", "ztl-input");
        std::env::set_var("S3_WEBHOOK_ALLOWED_HOSTS", "notifiche.internal:8443, altro.it");
        std::env::set_var("S3_MAX_CONCURRENT_JOBS", "4");

        let s = s3_settings_from_env().unwrap().expect("enabled");
        // trailing slash trimmed; endpoint set ⇒ path-style by default
        assert_eq!(s.endpoint.as_deref(), Some("http://localhost:9000"));
        assert!(s.force_path_style);
        assert_eq!(s.bucket_input, "ztl-input");
        assert_eq!(s.bucket_output, "anonimizzazione-output"); // default
        assert_eq!(s.max_concurrent_jobs, 4);
        assert_eq!(
            s.webhook_allowed_hosts,
            vec!["notifiche.internal:8443".to_string(), "altro.it".to_string()]
        );

        // Explicit path-style toggle overrides the endpoint default.
        std::env::set_var("S3_FORCE_PATH_STYLE", "false");
        let s = s3_settings_from_env().unwrap().unwrap();
        assert!(!s.force_path_style);

        // Hard error on invalid values at startup: `S3_MAX_CONCURRENT_JOBS` clamped
        // to min 1, but a non-numeric value is a config error.
        std::env::set_var("S3_MAX_CONCURRENT_JOBS", "many");
        assert!(s3_settings_from_env().is_err());
        std::env::set_var("S3_MAX_CONCURRENT_JOBS", "4");
        assert_eq!(s3_settings_from_env().unwrap().unwrap().max_concurrent_jobs, 4);
        std::env::remove_var("S3_ENABLED");
        std::env::remove_var("S3_ENDPOINT");
        std::env::remove_var("S3_FORCE_PATH_STYLE");
        std::env::remove_var("S3_BUCKET_INPUT");
        std::env::remove_var("S3_WEBHOOK_ALLOWED_HOSTS");
        std::env::remove_var("S3_MAX_CONCURRENT_JOBS");
    }
}
