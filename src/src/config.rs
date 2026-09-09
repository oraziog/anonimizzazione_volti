//! Centralized configuration from environment variables (spec §3, §9 + operational glue).
//!
//! Every knob of the service lives here. Values are parsed once at startup;
//! invalid values abort startup (fail-fast, consistent with the model policy).

use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Context, Result};

use crate::model_loader::ExecutionProvider;

/// Filename (under `DATA_DIR`) of the nightly-retraining audit JSON, served
/// by `GET /operator/retrain-audit`.
pub const RETRAIN_AUDIT_FILENAME: &str = "retrain_audit.json";
/// Running-classifier state (active ONNX + validated accuracy), so the
/// operator can see what model is live and the A/B retraining gate can compare
/// a candidate against the deployed accuracy.
#[allow(dead_code)] // only referenced by the optional retraining feature
pub const CLASSIFIER_STATE_FILENAME: &str = "classifier_state.json";

/// Fully-parsed service configuration.
#[derive(Debug, Clone)]
pub struct Config {
    pub bind_addr: String,
    pub body_limit_bytes: usize,
    pub max_concurrent_images: Option<usize>,
    pub jpeg_quality: u8,

    pub yolo_model_url: String,
    pub retinaface_model_url: String,
    pub classifier_model_url: Option<String>,
    pub model_cache_dir: PathBuf,
    pub yolo_sha256: Option<String>,
    pub retinaface_sha256: Option<String>,
    pub classifier_sha256: Option<String>,
    pub detector_mode: DetectorMode,
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
    /// Stretch-resize side in px for the RetinaFace input (env
    /// `RETINAFACE_INPUT_SIZE`, default 640).
    pub retinaface_input_size: u32,

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

    pub operator_api_key: Option<String>,

    // STORE-output retention (DATA_DIR `*_elaborato.zip` cleanup).
    pub retention_enabled: bool,
    pub retention_max_days: u32,
    pub retention_max_gb: f64,
    pub retention_interval_secs: u64,
    /// Outputs modified more recently than this are never deleted (in-flight
    /// jobs / responses streaming right now).
    pub retention_min_age_secs: u64,

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

/// Face detector backend (env `DETECTOR_MODE`): YOLOv8-Face (default) or the
/// lightweight MobileNetV1-0.25 RetinaFace export used for benchmarks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DetectorMode {
    Yolo,
    RetinaFace,
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

impl DetectorMode {
    fn parse(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "yolo" | "yolov8" | "yolov8-face" => Ok(DetectorMode::Yolo),
            "retinaface" | "retina" => Ok(DetectorMode::RetinaFace),
            other => anyhow::bail!("DETECTOR_MODE '{other}' must be 'yolo' or 'retinaface'"),
        }
    }
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
            max_concurrent_images,
            jpeg_quality: env_parse::<u8>("JPEG_QUALITY", 95)?.clamp(1, 100),

            yolo_model_url: env_str(
                "YOLO_MODEL_URL",
                "https://github.com/yakhyo/yolov8-face-onnx-inference/releases/download/weights/yolov8n-face.onnx",
            ),
            retinaface_model_url: env_str(
                "RETINAFACE_MODEL_URL",
                "https://github.com/yakhyo/retinaface-pytorch/releases/download/v0.0.1/retinaface_mv1_0.25.onnx",
            ),
            classifier_model_url: env_opt("CLASSIFIER_MODEL_URL"),
            model_cache_dir: env_path("MODEL_CACHE_DIR", "/app/models/cache/"),
            yolo_sha256: env_opt("MODEL_YOLO_SHA256"),
            retinaface_sha256: env_opt("MODEL_RETINAFACE_SHA256"),
            classifier_sha256: env_opt("MODEL_CLASSIFIER_SHA256"),
            detector_mode: DetectorMode::parse(&env_str("DETECTOR_MODE", "yolo"))?,
            yolo_input_size: parse_input_size("YOLO_INPUT_SIZE")?,
            retinaface_input_size: parse_input_size("RETINAFACE_INPUT_SIZE")?,
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

            operator_api_key: env_opt("OPERATOR_API_KEY"),

            retention_enabled: env_parse::<bool>("RETENTION_ENABLED", true)?,
            retention_max_days: env_parse::<u32>("RETENTION_MAX_DAYS", 30)?,
            retention_max_gb: env_parse::<f64>("RETENTION_MAX_GB", 20.0)?,
            retention_interval_secs: env_parse::<u64>("RETENTION_INTERVAL_SECS", 3600)?.max(60),
            retention_min_age_secs: env_parse::<u64>("RETENTION_MIN_AGE_SECS", 1800)?,

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
        };

        if cfg.roi_area_min >= cfg.roi_area_max {
            anyhow::bail!("ROI_AREA_MIN must be < ROI_AREA_MAX");
        }
        if cfg.fp_crop_conf_max <= cfg.yolo_conf_threshold {
            anyhow::bail!("FP_CROP_CONF_MAX must be > YOLO_CONF_THRESHOLD");
        }
        Ok(cfg)
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
}

#[cfg(test)]
impl Config {
    /// Deterministic env-independent configuration for unit tests.
    /// Mirrors the `from_env` defaults (paths are inert for tests).
    pub(crate) fn test_default() -> Self {
        Config {
            bind_addr: "127.0.0.1:0".into(),
            body_limit_bytes: 3_758_096_384,
            max_concurrent_images: Some(2),
            jpeg_quality: 95,
            yolo_model_url: String::new(),
            retinaface_model_url: String::new(),
            classifier_model_url: None,
            model_cache_dir: PathBuf::from("/tmp/av-models-cache"),
            yolo_sha256: None,
            retinaface_sha256: None,
            classifier_sha256: None,
            detector_mode: DetectorMode::Yolo,
            yolo_input_size: 640,
            retinaface_input_size: 640,
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
            operator_api_key: None,
            retention_enabled: false, // tests create their own temp data dirs
            retention_max_days: 30,
            retention_max_gb: 20.0,
            retention_interval_secs: 3600,
            retention_min_age_secs: 1800,
            execution_provider: ExecutionProvider::Cpu,
            gpu_device_id: 0,
            gpu_memory_limit_bytes: None,
            enable_tf32: false,
            enable_fp16: false,
            #[cfg(feature = "s3")]
            s3: None,
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
