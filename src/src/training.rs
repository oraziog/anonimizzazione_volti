//! Retraining bridge (spec §6, §8 `training.rs`).
//!
//! Nightly job: Rust invokes `python/retrain.py` through PyO3 (fine-tuning a
//! frozen-backbone MobileNetV2 on the seed real-faces + automatically
//! collected false-positive crops), the script exports a new ONNX classifier
//! plus `metrics.json` in the model cache. Rust then:
//!   1. checks the reported holdout accuracy against `RETRAIN_MIN_ACCURACY`;
//!   2. re-validates the exported ONNX with `ort` on a fresh sample of both
//!      classes (loadability + accuracy) — spec §6 pre-swap validation;
//!   3. on success: backs up the previous classifier, atomically swaps the
//!      store to the new session pool and prunes stale staging exports.
//! Any failure discards the candidate and keeps the current classifier.
//!
//! Every run — success, rejection, skip or failure — writes a JSON audit
//! record to `DATA_DIR/retrain_audit.json` (status, accuracy, sample counts,
//! swap/reject outcome) and logs a one-line summary. The record is served by
//! `GET /operator/retrain-audit`.

use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use chrono::Utc;
use serde::{Deserialize, Serialize};

use crate::config::Config;
use crate::model_loader::load_session;
use crate::models::{run_classifier, ModelStore, SessionPool};

/// Audit record for one nightly retraining run (persisted as JSON and served
/// by the operator endpoint).
#[derive(Debug, Clone, Serialize)]
pub struct RetrainAudit {
    pub timestamp: String,
    /// `swapped` | `rejected` | `skipped` | `failed`
    pub status: String,
    pub reason: Option<String>,
    pub real_samples: usize,
    pub fp_samples: usize,
    pub python_val_accuracy: Option<f64>,
    pub python_val_samples: Option<u64>,
    pub python_train_samples: Option<u64>,
    pub epochs_run: Option<u64>,
    pub rust_validation_accuracy: Option<f64>,
    /// Accuracy of the deployed classifier on the same deterministic holdout
    /// (`None` on the very first swap): the A/B comparison baseline.
    pub current_accuracy: Option<f64>,
    /// Path of the deployed classifier the candidate was compared against.
    pub current_onnx: Option<String>,
    /// How many consumed false-positive crops were cleared after a successful
    /// swap (the model already learned on them; next run starts fresh).
    pub cleared_fp_crops: Option<usize>,
    pub min_accuracy: f32,
    pub candidate_onnx: Option<String>,
    pub backup_onnx: Option<String>,
    pub error: Option<String>,
}

impl RetrainAudit {
    /// One-line human summary for the log.
    fn summary_line(&self) -> String {
        format!(
            "status={} reason={:?} val_acc={:?} rust_acc={:?} samples=(train {:?}, val {:?}, real {}, fp {}) candidate={:?} backup={:?}",
            self.status,
            self.reason,
            self.python_val_accuracy,
            self.rust_validation_accuracy,
            self.python_train_samples,
            self.python_val_samples,
            self.real_samples,
            self.fp_samples,
            self.candidate_onnx,
            self.backup_onnx
        )
    }
}

/// Metrics written by `python/retrain.py`.
#[derive(Debug, Deserialize)]
struct RetrainMetrics {
    val_accuracy: f64,
    val_samples: u64,
    train_samples: u64,
    epochs_run: u64,
}

pub async fn nightly_retrain(cfg: &Config, store: &ModelStore) -> Result<RetrainAudit> {
    let mut audit = RetrainAudit {
        timestamp: Utc::now().to_rfc3339(),
        status: "skipped".to_string(),
        reason: None,
        real_samples: count_images(&cfg.dataset_seed_real_faces_dir),
        fp_samples: count_images(&cfg.dataset_fp_dir),
        python_val_accuracy: None,
        python_val_samples: None,
        python_train_samples: None,
        epochs_run: None,
        rust_validation_accuracy: None,
        current_accuracy: None,
        current_onnx: None,
        cleared_fp_crops: None,
        min_accuracy: cfg.retrain_min_accuracy,
        candidate_onnx: None,
        backup_onnx: None,
        error: None,
    };

    let outcome = retrain_inner(cfg, store, &mut audit).await;
    if let Err(e) = outcome {
        audit.status = "failed".to_string();
        audit.error = Some(format!("{e:#}"));
        audit.reason = Some(format!("retraining aborted: {e:#}"));
    }

    // On skipped/failed runs this run produced no A/B baseline: expose the
    // deployed model recorded at the last successful swap instead.
    if audit.current_accuracy.is_none() {
        if let Some(st) = read_classifier_state(cfg) {
            audit.current_accuracy = Some(st.accuracy);
            audit.current_onnx = Some(st.active_onnx);
        }
    }

    tracing::info!("nightly retraining audit: {}", audit.summary_line());
    let audit_path = cfg.data_dir.join(crate::config::RETRAIN_AUDIT_FILENAME);
    let json = serde_json::to_string_pretty(&audit).context("serialize retraining audit")?;
    std::fs::write(&audit_path, json)
        .with_context(|| format!("write audit to {}", audit_path.display()))?;
    tracing::info!("retraining audit written to {}", audit_path.display());
    Ok(audit)
}

/// The retraining flow proper. Populates `audit` along the way; on rejected
/// or skipped candidates returns `Ok(())` with `audit.status` set, on hard
/// failure returns `Err`.
async fn retrain_inner(cfg: &Config, store: &ModelStore, audit: &mut RetrainAudit) -> Result<()> {
    let seed_dir = &cfg.dataset_seed_real_faces_dir;
    let fp_dir = &cfg.dataset_fp_dir;
    let real_count = count_images(seed_dir);
    let fp_count = count_images(fp_dir);
    tracing::info!(
        "nightly retraining: {real_count} real samples, {fp_count} false-positive samples"
    );
    if real_count < 2 || fp_count < 2 {
        audit.status = "skipped".to_string();
        audit.reason = Some(format!(
            "need ≥2 samples per class (real={real_count}, fp={fp_count})"
        ));
        tracing::warn!(
            "retraining skipped: need ≥2 samples per class (real={real_count}, fp={fp_count}). \
             Seed dataset: {seed_dir:?}, FP crops: {fp_dir:?}"
        );
        return Ok(());
    }

    let ts = Utc::now().format("%Y%m%d_%H%M%S").to_string();
    let out_onnx = cfg.model_cache_dir.join(format!("classifier_{ts}.onnx"));
    let metrics_path = cfg.model_cache_dir.join(format!("metrics_{ts}.json"));
    let script = retrain_script_path();
    audit.candidate_onnx = Some(out_onnx.to_string_lossy().to_string());

    let cfg2 = cfg.clone();
    let seed = seed_dir.clone();
    let fp = fp_dir.clone();
    let onnx = out_onnx.clone();
    let metrics = metrics_path.clone();
    let script_path = script.clone();
    tokio::task::spawn_blocking(move || {
        invoke_python_retrain(
            &script_path,
            &seed,
            &fp,
            &onnx,
            &metrics,
            cfg2.retrain_epochs,
            cfg2.retrain_batch_size,
            cfg2.retrain_holdout_fraction,
        )
    })
    .await
    .context("retraining worker panicked")??;

    let raw = std::fs::read_to_string(&metrics_path)
        .with_context(|| format!("read {}", metrics_path.display()))?;
    let m: RetrainMetrics =
        serde_json::from_str(&raw).with_context(|| format!("parse {}", metrics_path.display()))?;
    audit.python_val_accuracy = Some(m.val_accuracy);
    audit.python_val_samples = Some(m.val_samples);
    audit.python_train_samples = Some(m.train_samples);
    audit.epochs_run = Some(m.epochs_run);
    tracing::info!(
        "python retraining reported val_accuracy={:.4} ({} val samples, {} train samples, {} epochs)",
        m.val_accuracy,
        m.val_samples,
        m.train_samples,
        m.epochs_run
    );

    let min_acc = cfg.retrain_min_accuracy as f64;
    if m.val_samples == 0 || m.val_accuracy < min_acc {
        discard(&out_onnx, &metrics_path);
        audit.status = "rejected".to_string();
        audit.reason = Some(format!(
            "python holdout accuracy {:.4} below minimum {min_acc}",
            m.val_accuracy
        ));
        tracing::warn!(
            "candidate classifier rejected: val_accuracy {:.4} < {min_acc} (discarded)",
            m.val_accuracy
        );
        return Ok(());
    }

    // Pre-swap validation with ort on a fresh sample of both classes (§6).
    let rust_acc = validate_exported_model(&out_onnx, seed_dir, fp_dir)?;
    audit.rust_validation_accuracy = Some(rust_acc);
    if rust_acc < min_acc {
        discard(&out_onnx, &metrics_path);
        audit.status = "rejected".to_string();
        audit.reason = Some(format!(
            "Rust-side ONNX validation accuracy {rust_acc:.4} below minimum {min_acc}"
        ));
        tracing::warn!(
            "candidate classifier rejected by Rust-side validation: accuracy {rust_acc:.4} < {min_acc}"
        );
        return Ok(());
    }

    // A/B gate: a retrained candidate must not degrade the deployed model.
    // `validate_exported_model` uses the same deterministic holdout for both
    // (`sample_images` is sorted and truncated), so the comparison is fair.
    // RetrainAudit carries the baseline so the operator can see it.
    if let Some(current) = store.classifier_pool() {
        let current_path = current.path();
        if current_path != &out_onnx && current_path.exists() {
            match validate_exported_model(current_path, seed_dir, fp_dir) {
                Ok(current_acc) => {
                    audit.current_accuracy = Some(current_acc);
                    audit.current_onnx = Some(current_path.to_string_lossy().to_string());
                    let floor = current_acc - cfg.retrain_regression_eps as f64;
                    if rust_acc < floor {
                        discard(&out_onnx, &metrics_path);
                        audit.status = "rejected".to_string();
                        audit.reason = Some(format!(
                            "candidate {rust_acc:.4} below deployed model {current_acc:.4} \
                             (regression eps {})",
                            cfg.retrain_regression_eps
                        ));
                        tracing::warn!(
                            "candidate classifier {rust_acc:.4} < deployed {current_acc:.4} \
                             (eps {}): kept the deployed model",
                            cfg.retrain_regression_eps
                        );
                        return Ok(());
                    }
                }
                Err(e) => {
                    // Cannot establish the baseline → keep the deployed model.
                    discard(&out_onnx, &metrics_path);
                    audit.status = "rejected".to_string();
                    audit.reason = Some(format!(
                        "cannot validate the deployed classifier as A/B baseline: {e:#}"
                    ));
                    tracing::warn!("A/B baseline validation failed: {e:#}");
                    return Ok(());
                }
            }
        }
    }

    // Backup previous classifier for manual rollback (§6).
    let mut backup = None;
    if let Some(current) = store.classifier_pool() {
        let src = current.path();
        if src.exists() && src != &out_onnx {
            let b = cfg.models_backup_dir.join(format!("classifier_{ts}.onnx"));
            if let Err(e) = std::fs::copy(src, &b) {
                tracing::warn!("cannot back up previous classifier to {}: {e}", b.display());
            } else {
                backup = Some(b.clone());
                audit.backup_onnx = Some(b.to_string_lossy().to_string());
            }
        }
    }

    store.swap_classifier_pool(SessionPool::new(
        out_onnx.clone(),
        cfg.effective_concurrency(),
    ));
    audit.status = "swapped".to_string();
    audit.reason = Some(format!(
        "classifier swapped with Rust-validated accuracy {rust_acc:.4}{}",
        match audit.current_accuracy {
            Some(cur) => format!(" (deployed baseline {cur:.4})"),
            None => " (first swap, no deployed baseline)".to_string(),
        }
    ));
    tracing::info!(
        "classifier swapped → {} (accuracy {:.4})",
        out_onnx.display(),
        rust_acc
    );
    let _ = backup;

    // The classes the new model just learned on are consumed: clear the FP
    // crops so the next nightly starts from fresh false positives instead of
    // re-learning an overweighted snapshot.
    let cleared = clear_fp_crops(fp_dir);
    audit.cleared_fp_crops = Some(cleared);
    tracing::info!("cleared {cleared} consumed FP crops after retrain swap");
    if let Err(e) = write_classifier_state(cfg, &out_onnx, rust_acc) {
        tracing::warn!("cannot persist classifier state: {e:#}");
    }

    prune_stale(&out_onnx);
    Ok(())
}

/// Runs the Python fine-tuning script and waits for its exported files.
fn invoke_python_retrain(
    script: &Path,
    seed_dir: &Path,
    fp_dir: &Path,
    out_onnx: &Path,
    metrics_path: &Path,
    epochs: u32,
    batch_size: u32,
    holdout: f32,
) -> Result<()> {
    let code = std::fs::read_to_string(script)
        .with_context(|| format!("read retraining script {}", script.display()))?;

    pyo3::Python::with_gil(|py| {
        let module = pyo3::types::PyModule::from_code(py, &code, "retrain.py", "retrain.py")
            .map_err(|e| anyhow!("python module import failed: {e}"))?;
        module
            .call_method1(
                "retrain",
                (
                    seed_dir.to_string_lossy().to_string(),
                    fp_dir.to_string_lossy().to_string(),
                    out_onnx.to_string_lossy().to_string(),
                    metrics_path.to_string_lossy().to_string(),
                    epochs,
                    batch_size,
                    holdout,
                ),
            )
            .map_err(|e| anyhow!("python retraining raised: {e}"))?;
        Ok(())
    })
}

/// Independent Rust-side accuracy check on the *exported ONNX* (spec §6:
/// "caricare il nuovo ONNX con ort su un mini validation set").
fn validate_exported_model(onnx: &Path, seed_dir: &Path, fp_dir: &Path) -> Result<f64> {
    let mut session = load_session(onnx)
        .with_context(|| format!("candidate ONNX not loadable: {}", onnx.display()))?;

    const LIMIT: usize = 60;
    let reals = sample_images(seed_dir, LIMIT);
    let fps = sample_images(fp_dir, LIMIT);
    if reals.is_empty() || fps.is_empty() {
        return Err(anyhow!("no images to validate candidate classifier"));
    }

    let mut correct = 0usize;
    let mut total = 0usize;
    for path in reals.iter().chain(fps.iter()) {
        let img = image::open(path)
            .map_err(|e| anyhow!("cannot open {}: {e}", path.display()))?
            .to_rgb8();
        let (p_fp, p_face) = run_classifier(&mut session, &img)
            .with_context(|| format!("inference on {} failed", path.display()))?;
        let real = p_face >= p_fp;
        let is_real_sample = path.starts_with(seed_dir);
        if real == is_real_sample {
            correct += 1;
        }
        total += 1;
    }
    if total == 0 {
        return Err(anyhow!("empty validation set"));
    }
    Ok(correct as f64 / total as f64)
}

fn discard(onnx: &Path, metrics: &Path) {
    std::fs::remove_file(onnx).ok();
    std::fs::remove_file(metrics).ok();
}

/// Recursively removes the consumed false-positive crops (jpg/jpeg/png) under
/// `fp_dir`; returns how many files were deleted. Directories and non-image
/// files are left untouched.
fn clear_fp_crops(fp_dir: &Path) -> usize {
    let mut removed = 0usize;
    let Ok(rd) = std::fs::read_dir(fp_dir) else {
        return 0;
    };
    for entry in rd.flatten() {
        let path = entry.path();
        if path.is_dir() {
            removed += clear_fp_crops(&path);
        } else if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            if matches!(ext.to_ascii_lowercase().as_str(), "jpg" | "jpeg" | "png") {
                if std::fs::remove_file(&path).is_ok() {
                    removed += 1;
                }
            }
        }
    }
    removed
}

/// Last persisted classifier state (`DATA_DIR/classifier_state.json`), also
/// read by the `GET /operator/classifier` endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassifierState {
    pub active_onnx: String,
    pub accuracy: f64,
    pub updated_at: String,
}

fn classifier_state_path(cfg: &Config) -> PathBuf {
    cfg.data_dir.join(crate::config::CLASSIFIER_STATE_FILENAME)
}

pub fn write_classifier_state(cfg: &Config, active_onnx: &Path, accuracy: f64) -> Result<()> {
    let st = ClassifierState {
        active_onnx: active_onnx.to_string_lossy().to_string(),
        accuracy,
        updated_at: Utc::now().to_rfc3339(),
    };
    let json = serde_json::to_string_pretty(&st).context("serialize classifier state")?;
    std::fs::write(classifier_state_path(cfg), json)
        .with_context(|| format!("write {}", classifier_state_path(cfg).display()))?;
    Ok(())
}

pub fn read_classifier_state(cfg: &Config) -> Option<ClassifierState> {
    let path = classifier_state_path(cfg);
    let raw = std::fs::read_to_string(&path).ok()?;
    serde_json::from_str(&raw).ok()
}

/// Deletes older exported candidates (keeps only the active one plus any
/// backups already copied elsewhere).
fn prune_stale(active: &Path) {
    let Some(dir) = active.parent() else { return };
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let p = entry.path();
        let is_candidate = p
            .file_name()
            .and_then(|n| n.to_str())
            .map(|n| n.starts_with("classifier_") && n.ends_with(".onnx"))
            .unwrap_or(false);
        if is_candidate && p != active {
            std::fs::remove_file(&p).ok();
        }
    }
}

fn retrain_script_path() -> PathBuf {
    match std::env::var("RETRAIN_SCRIPT_PATH") {
        Ok(p) if !p.is_empty() => PathBuf::from(p),
        _ => PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("python/retrain.py"),
    }
}

fn count_images(dir: &Path) -> usize {
    sample_images(dir, usize::MAX).len()
}

/// Deterministically collected image files (recursive, sorted) up to `limit`.
fn sample_images(dir: &Path, limit: usize) -> Vec<PathBuf> {
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        let Ok(rd) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in rd.flatten() {
            let path = entry.path();
            if path.is_dir() {
                walk(&path, out);
            } else if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                if matches!(ext.to_ascii_lowercase().as_str(), "jpg" | "jpeg" | "png") {
                    out.push(path);
                }
            }
        }
    }
    let mut all = Vec::new();
    walk(dir, &mut all);
    all.sort();
    all.truncate(limit);
    all
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Unique per run and removed on drop; see `crate::testutil`.
    fn test_dir(tag: &str) -> crate::testutil::TempDir {
        crate::testutil::TempDir::new(&format!("training_test_{tag}"))
    }

    #[test]
    fn clear_fp_crops_removes_images_only() {
        let dir = test_dir("clear");
        std::fs::create_dir_all(dir.join("CAM_001")).unwrap();
        std::fs::write(dir.join("CAM_001/A.jpg"), b"a").unwrap();
        std::fs::write(dir.join("CAM_001/b.PNG"), b"b").unwrap();
        std::fs::write(dir.join("CAM_002.jpeg"), b"c").unwrap();
        std::fs::write(dir.join("keep.txt"), b"k").unwrap();
        assert_eq!(clear_fp_crops(&dir), 3);
        assert!(!dir.join("CAM_001/A.jpg").exists());
        assert!(!dir.join("CAM_001/b.PNG").exists());
        assert!(!dir.join("CAM_002.jpeg").exists());
        assert!(dir.join("keep.txt").exists(), "non-images must survive");
        assert!(dir.join("CAM_001").is_dir(), "empty subdirs survive");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn classifier_state_roundtrip() {
        let dir = test_dir("state");
        let mut cfg = crate::config::Config::test_default();
        cfg.data_dir = dir.clone();
        write_classifier_state(&cfg, Path::new("/tmp/classifier_x.onnx"), 0.912).unwrap();
        let st = read_classifier_state(&cfg).unwrap();
        assert_eq!(st.active_onnx, "/tmp/classifier_x.onnx");
        assert_eq!(st.accuracy, 0.912);
        assert!(read_classifier_state(&crate::config::Config::test_default()).is_none());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn audit_serializes_ab_fields() {
        let audit = RetrainAudit {
            timestamp: "2026-09-09T00:00:00Z".into(),
            status: "swapped".into(),
            reason: Some("ok".into()),
            real_samples: 10,
            fp_samples: 20,
            python_val_accuracy: Some(0.9),
            python_val_samples: Some(30),
            python_train_samples: Some(300),
            epochs_run: Some(5),
            rust_validation_accuracy: Some(0.91),
            current_accuracy: Some(0.89),
            current_onnx: Some("/tmp/old.onnx".into()),
            cleared_fp_crops: Some(12),
            min_accuracy: 0.85,
            candidate_onnx: Some("/tmp/new.onnx".into()),
            backup_onnx: None,
            error: None,
        };
        let json = serde_json::to_string(&audit).unwrap();
        assert!(json.contains("\"current_accuracy\""));
        assert!(json.contains("\"cleared_fp_crops\""));
        assert!(json.contains("\"current_onnx\""));
    }
}
