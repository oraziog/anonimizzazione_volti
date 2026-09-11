//! Manual sample-output generator (`#[ignore]`d): runs the real pipeline
//! (ACTIVE branch, real ONNX face detector from `models_cache/`) on the photos
//! in `test-wider-sample/` and writes anonymized + before/after PNGs under
//! `docs/images/sample_outputs/` — handy for reviewing the anonymization
//! quality before a release/commit.
//!
//! Run with (from `src/`):
//!   cargo test --release -- --ignored samples_anon

use std::path::PathBuf;

use image::{DynamicImage, RgbaImage};

use crate::config::Config;
use crate::db::CameraState;
use crate::models::{ModelStore, SessionPool};
use crate::pipeline::process_image;

const MODEL: &str = "models_cache/yolov8m-face.onnx";
const FALLBACK_MODEL: &str = "models_cache/yolov8n-face.onnx";
const INPUT_DIR: &str = "test-wider-sample";
const OUT_DIR: &str = "docs/images/sample_outputs";

fn pick_model() -> Option<PathBuf> {
    [MODEL, FALLBACK_MODEL]
        .iter()
        .map(PathBuf::from)
        .find(|p| p.exists())
}

#[test]
#[ignore = "generates demo outputs with real models; run on demand"]
fn samples_anon() {
    let Some(model) = pick_model() else {
        eprintln!("samples_anon: no face model in models_cache, skipping");
        return;
    };
    let entries: Vec<PathBuf> = std::fs::read_dir(INPUT_DIR)
        .map(|rd| {
            rd.filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| {
                    matches!(
                        p.extension().and_then(|s| s.to_str()),
                        Some("jpg" | "jpeg" | "png")
                    )
                })
                .collect()
        })
        .unwrap_or_default();
    if entries.is_empty() {
        eprintln!("samples_anon: no input images in {INPUT_DIR}, skipping");
        return;
    }
    std::fs::create_dir_all(OUT_DIR).expect("create sample output dir");

    let store = ModelStore::new(SessionPool::new(model, 1), None, None);
    let mut cfg = Config::test_default();
    cfg.yolo_conf_threshold_active = 0.15;
    cfg.classifier_enforce = false;

    for path in &entries {
        let img = match image::open(path) {
            Ok(i) => i.to_rgb8(),
            Err(e) => {
                eprintln!("samples_anon: cannot open {}: {e}", path.display());
                continue;
            }
        };
        let (w, h) = img.dimensions();
        let out = match process_image(&cfg, &store, CameraState::Active, None, &img) {
            Ok(o) => o,
            Err(e) => {
                eprintln!("samples_anon: process {} failed: {e:#}", path.display());
                continue;
            }
        };
        let stem = path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "sample".to_owned());
        let anon = PathBuf::from(OUT_DIR).join(format!("{stem}_anonimizzato.png"));
        out.processed.save(&anon).expect("save anonymized sample");

        // Before/after side by side: original | anonymized.
        let orig = DynamicImage::ImageRgb8(img).to_rgba8();
        let mut side = RgbaImage::new(w * 2, h);
        image::imageops::replace(&mut side, &orig, 0, 0);
        image::imageops::replace(&mut side, &out.processed, w as i64, 0);
        let cmp = PathBuf::from(OUT_DIR).join(format!("{stem}_prima_dopo.png"));
        side.save(&cmp).expect("save before/after sample");

        let abs = std::fs::canonicalize(OUT_DIR).expect("resolve sample output dir");
        println!(
            "samples_anon: wrote anonymized={} ({} face(s), branch {:?}) and comparison={}",
            anon.display(),
            out.detections.len(),
            out.branch,
            cmp.display()
        );
        println!("samples_anon: output dir: {}", abs.display());
    }
}