//! Centralized configuration from environment variables (spec §3, §9 + operational glue).
//!
//! Every knob of the service lives here. Values are parsed once at startup;
//! invalid values abort startup (fail-fast, consistent with the model policy).

use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Context, Result};

/// Filename (under `DATA_DIR`) of the nightly-retraining audit JSON, served
/// by `GET /operator/retrain-audit`.
pub const RETRAIN_AUDIT_FILENAME: &str = "retrain_audit.json";

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

    pub retrain_schedule: ScheduleSpec,
    #[allow(dead_code)] // consumed by the optional retraining feature
    pub retrain_min_accuracy: f32,
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

impl DetectorMode {
    fn parse(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "yolo" | "yolov8" | "yolov8-face" => Ok(DetectorMode::Yolo),
            "retinaface" | "retina" => Ok(DetectorMode::RetinaFace),
            other => anyhow::bail!("DETECTOR_MODE '{other}' must be 'yolo' or 'retinaface'"),
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

            retrain_schedule: ScheduleSpec::parse(&env_str("CRON_RETRAIN_SCHEDULE", "03:00"))?,
            retrain_min_accuracy: env_parse::<f32>("RETRAIN_MIN_ACCURACY", 0.85)?
                .clamp(0.5, 1.0),
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
            retrain_schedule: ScheduleSpec { hour: 3, minute: 0 },
            retrain_min_accuracy: 0.85,
            retrain_holdout_fraction: 0.10,
            retrain_epochs: 1,
            retrain_batch_size: 8,
            operator_api_key: None,
            retention_enabled: false, // tests create their own temp data dirs
            retention_max_days: 30,
            retention_max_gb: 20.0,
            retention_interval_secs: 3600,
            retention_min_age_secs: 1800,
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
}
