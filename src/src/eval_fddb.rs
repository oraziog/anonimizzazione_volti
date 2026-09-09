//! Offline evaluation of the YOLO detector against FDDB (Face Detection Data
//! Set and Benchmark): ellipse annotations + fold list files, official metric
//! is recall vs. false positives per image (FPPI) on a log scale.
//!
//! Run as a CLI subcommand (no server, no config):
//! ```text
//! anonimizzazione_volti eval-fddb \
//!     --model <yolov8n-face.onnx> \
//!     --images <fddb_images/> \
//!     --folds <FDDB-fold-01.txt,FDDB-fold-02.txt,...> \
//!     [--det-conf 0.001] [--nms-iou 0.45] [--match-iou 0.5] [--max-images N]
//! ```
//!
//! FDDB layout (from vis-www.cs.umass.edu/fddb):
//! - `FDDB-folds.tgz` → fold text files; each image block is
//!   ```text
//!   <relative image path without extension>
//!   <number of faces>
//!   <major_axis_radius> <minor_axis_radius> <angle_rad> <center_x> <center_y>
//!   ```
//! - images live under `<images>/<path>.jpg`.
//!
//! Similarity: the official evaluator compares *ellipses*; for a
//! self-contained harness we approximate each ellipse with its axis-aligned
//! bounding box and use box IoU (community-standard simplification, default
//! `--match-iou 0.5`). Matching is greedy per image, identical in spirit to
//! the WIDER harness. The ROC curve is built by sorting all detections by
//! descending score and accumulating TP/FP; results are reported at the usual
//! operating points (FPPI = 0.05 … 1.0).

use std::path::PathBuf;

use anyhow::{bail, Context, Result};

use crate::model_loader::load_session;
use crate::models::run_yolo;

const FPPI_POINTS: [f64; 5] = [0.05, 0.10, 0.25, 0.50, 1.00];

struct Face {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
}

struct FddbImage {
    /// Relative path without extension, e.g. `2002/07/19/big/img_130`.
    name: String,
    faces: Vec<Face>,
}

/// Axis-aligned bounding box of a rotated ellipse (FDDB stores it as
/// major/minor radius + angle + center).
fn ellipse_bbox(ra: f32, rb: f32, theta: f32, cx: f32, cy: f32) -> Face {
    let (sin_t, cos_t) = theta.sin_cos();
    let half_w = (ra * cos_t).powi(2) + (rb * sin_t).powi(2);
    let half_h = (ra * sin_t).powi(2) + (rb * cos_t).powi(2);
    let (half_w, half_h) = (half_w.sqrt(), half_h.sqrt());
    Face {
        x1: cx - half_w,
        y1: cy - half_h,
        x2: cx + half_w,
        y2: cy + half_h,
    }
}

/// Parses one or more FDDB fold files (comma-separated paths or a directory
/// containing `*.txt` folds, sorted).
fn parse_folds(folds_arg: &str) -> Result<Vec<FddbImage>> {
    let mut fold_files: Vec<PathBuf> = Vec::new();
    let p = PathBuf::from(folds_arg);
    if p.is_dir() {
        let mut entries: Vec<_> = std::fs::read_dir(&p)
            .with_context(|| format!("read folds dir {}", p.display()))?
            .flatten()
            .map(|e| e.path())
            .filter(|f| f.extension().and_then(|e| e.to_str()) == Some("txt"))
            .collect();
        entries.sort();
        fold_files.extend(entries);
    } else {
        fold_files.extend(
            folds_arg
                .split(',')
                .map(PathBuf::from)
                .filter(|f| !f.as_os_str().is_empty()),
        );
    }
    if fold_files.is_empty() {
        bail!("no FDDB fold files found for --folds {folds_arg:?}");
    }

    let mut images = Vec::new();
    for fold in &fold_files {
        let text = std::fs::read_to_string(fold)
            .with_context(|| format!("read fold file {}", fold.display()))?;
        let mut lines = text
            .lines()
            .map(|l| l.trim())
            .filter(|l| !l.is_empty())
            .peekable();
        while let Some(name) = lines.next() {
            let count_line = lines.next().with_context(|| {
                format!("{}: missing face count after {name:?}", fold.display())
            })?;
            let n: usize = count_line.parse().with_context(|| {
                format!("{}: invalid face count {count_line:?}", fold.display())
            })?;
            let mut faces = Vec::with_capacity(n);
            for _ in 0..n {
                let line = lines.next().with_context(|| {
                    format!("{}: fold truncated inside {name:?}", fold.display())
                })?;
                let f: Vec<f32> = line
                    .split_whitespace()
                    .map(|t| t.parse::<f32>())
                    .collect::<std::result::Result<_, _>>()
                    .with_context(|| {
                        format!("{}: invalid ellipse line {line:?}", fold.display())
                    })?;
                if f.len() < 5 {
                    bail!("{}: ellipse line has < 5 fields: {line:?}", fold.display());
                }
                faces.push(ellipse_bbox(f[0], f[1], f[2], f[3], f[4]));
            }
            images.push(FddbImage {
                name: name.to_string(),
                faces,
            });
        }
    }
    Ok(images)
}

fn iou(a: &Face, x1: f32, y1: f32, x2: f32, y2: f32) -> f32 {
    let inter_w = (a.x2.min(x2) - a.x1.max(x1)).max(0.0);
    let inter_h = (a.y2.min(y2) - a.y1.max(y1)).max(0.0);
    let inter = inter_w * inter_h;
    if inter <= 0.0 {
        return 0.0;
    }
    let area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    let area_b = (x2 - x1) * (y2 - y1);
    inter / (area_a + area_b - inter)
}

/// Greedy per-image assignment: returns `(score, is_tp)` in descending-score
/// order (ties broken by original order).
fn match_image(
    faces: &[Face],
    dets: &[(f32, f32, f32, f32, f32)],
    match_iou: f32,
) -> Vec<(f32, bool)> {
    let mut consumed = vec![false; faces.len()];
    let mut out = Vec::with_capacity(dets.len());
    for (x1, y1, x2, y2, score) in dets {
        let mut best_iou = 0.0f32;
        let mut best_idx = usize::MAX;
        for (i, f) in faces.iter().enumerate() {
            let o = iou(f, *x1, *y1, *x2, *y2);
            if o > best_iou {
                best_iou = o;
                best_idx = i;
            }
        }
        let tp = best_iou >= match_iou && best_idx != usize::MAX && !consumed[best_idx];
        if best_iou >= match_iou && best_idx != usize::MAX {
            consumed[best_idx] = true;
        }
        out.push((*score, tp));
    }
    out
}

pub fn run(args: &[String]) -> Result<()> {
    let mut model = None;
    let mut images = None;
    let mut folds = None;
    let mut det_conf = 0.001f32;
    let mut nms_iou = 0.45f32;
    let mut match_iou = 0.5f32;
    let mut max_images = 0usize;

    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut next = || {
            it.next()
                .with_context(|| format!("missing value for {a}"))
                .map(|s| s.to_string())
        };
        match a.as_str() {
            "--model" => model = Some(next()?),
            "--images" => images = Some(next()?),
            "--folds" => folds = Some(next()?),
            "--det-conf" => det_conf = next()?.parse().context("--det-conf must be a float")?,
            "--nms-iou" => nms_iou = next()?.parse().context("--nms-iou must be a float")?,
            "--match-iou" => match_iou = next()?.parse().context("--match-iou must be a float")?,
            "--max-images" => {
                max_images = next()?.parse().context("--max-images must be an integer")?
            }
            other => bail!("unknown eval-fddb argument: {other}"),
        }
    }
    let model = model.context("missing --model")?;
    let images = images.context("missing --images")?;
    let folds = folds.context("missing --folds")?;

    let gts = parse_folds(&folds)?;
    if gts.is_empty() {
        bail!("no images parsed from folds {folds:?}");
    }
    tracing::info!(
        "eval-fddb: {} images from folds, det_conf={det_conf}, nms_iou={nms_iou}, match_iou={match_iou}, model={model}",
        gts.len()
    );

    let mut session = load_session(&PathBuf::from(&model))?;

    let total_faces: usize = gts.iter().map(|g| g.faces.len()).sum();
    let mut all: Vec<(f32, bool, usize)> = Vec::new(); // (score, tp, image_idx)
    let mut processed = 0usize;
    let mut missing = 0usize;
    for (img_idx, gt_img) in gts.iter().enumerate() {
        if max_images > 0 && processed >= max_images {
            break;
        }
        let img_path = PathBuf::from(&images).join(format!("{}.jpg", gt_img.name));
        let img = match image::open(&img_path) {
            Ok(i) => i.to_rgb8(),
            Err(e) => {
                tracing::warn!("cannot open {}: {e}", img_path.display());
                missing += 1;
                continue;
            }
        };
        let dets = run_yolo(&mut session, &img, det_conf, nms_iou, 640)
            .with_context(|| format!("inference on {} failed", img_path.display()))?;
        let mut dets: Vec<(f32, f32, f32, f32, f32)> = dets
            .into_iter()
            .map(|d| (d.bbox.x0, d.bbox.y0, d.bbox.x1, d.bbox.y1, d.confidence))
            .collect();
        dets.sort_by(|a, b| b.4.total_cmp(&a.4));
        for (score, tp) in match_image(&gt_img.faces, &dets, match_iou) {
            all.push((score, tp, img_idx));
        }
        processed += 1;
        if processed.is_multiple_of(50) {
            tracing::info!("eval-fddb: {processed} images processed");
        }
    }

    if processed == 0 {
        bail!("no images could be processed (check --images layout; {missing} unreadable)");
    }
    tracing::info!("eval-fddb: done ({processed} processed, {missing} unreadable)");

    // Global ROC: sort all detections by descending score (stable, so ties
    // keep image order), accumulate TP/FP.
    all.sort_by(|a, b| b.0.total_cmp(&a.0));
    let mut curve: Vec<(f64, f64)> = Vec::with_capacity(all.len());
    let mut tp = 0usize;
    let mut fp = 0usize;
    for (_, is_tp, _) in &all {
        if *is_tp {
            tp += 1;
        } else {
            fp += 1;
        }
        curve.push((
            fp as f64 / processed as f64,
            tp as f64 / total_faces.max(1) as f64,
        ));
    }

    println!();
    println!("FDDB validation — {model}");
    println!(
        "det_conf={det_conf}  nms_iou={nms_iou}  match_iou={match_iou}  images={processed}  missing={missing}"
    );
    println!(
        "total faces={total_faces}  detections after gate={}",
        all.len()
    );
    println!();
    println!("{:<12} {:>12}", "FPPI (FP/img)", "recall");
    println!("{:<12} {:>12}", "--------------", "------");
    for target in FPPI_POINTS {
        // Recall at FPPI = target: the last curve point with fppi <= target.
        let mut recall = 0.0f64;
        for (fppi, r) in &curve {
            if *fppi <= target + 1e-9 {
                recall = *r;
            }
        }
        println!("{:<12.2} {:>12.4}", target, recall);
    }
    let (max_fppi, max_recall) = curve.last().copied().unwrap_or((0.0, 0.0));
    println!();
    println!("end of curve: recall={max_recall:.4} at FPPI={max_fppi:.4}");
    Ok(())
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn face(x1: f32, y1: f32, x2: f32, y2: f32) -> Face {
        Face { x1, y1, x2, y2 }
    }

    #[test]
    fn ellipse_bbox_axis_aligned() {
        // theta = 0 → axis-aligned ellipse, bbox = 2ra x 2rb.
        let b = ellipse_bbox(30.0, 20.0, 0.0, 100.0, 100.0);
        assert!((b.x1 - 70.0).abs() < 1e-4);
        assert!((b.y1 - 80.0).abs() < 1e-4);
        assert!((b.x2 - 130.0).abs() < 1e-4);
        assert!((b.y2 - 120.0).abs() < 1e-4);
        // Rotated 90° swaps the radii.
        let b = ellipse_bbox(30.0, 20.0, std::f32::consts::FRAC_PI_2, 100.0, 100.0);
        assert!((b.x1 - 80.0).abs() < 1e-3);
        assert!((b.y1 - 70.0).abs() < 1e-3);
        assert!((b.x2 - 120.0).abs() < 1e-3);
        assert!((b.y2 - 130.0).abs() < 1e-3);
    }

    #[test]
    fn fold_parser_roundtrip() {
        let dir = std::env::temp_dir().join("av-fddb-fold-test");
        std::fs::create_dir_all(&dir).unwrap();
        let fold = dir.join("FDDB-fold-01.txt");
        std::fs::write(
            &fold,
            "2002/07/19/big/img_130\n2\n30.0 20.0 0.0 100.0 100.0\n10.0 10.0 0.0 200.0 200.0\n\
             2002/07/19/big/img_131\n0\n",
        )
        .unwrap();
        let images = parse_folds(&fold.to_string_lossy()).unwrap();
        assert_eq!(images.len(), 2);
        assert_eq!(images[0].name, "2002/07/19/big/img_130");
        assert_eq!(images[0].faces.len(), 2);
        assert!((images[0].faces[0].x1 - 70.0).abs() < 1e-4);
        assert!(images[1].faces.is_empty());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn matching_counts_tp_once() {
        let faces = vec![face(0.0, 0.0, 100.0, 100.0)];
        // Two overlapping detections: the second must not double-count.
        let dets = vec![
            (0.0, 0.0, 100.0, 100.0, 0.9),
            (10.0, 10.0, 90.0, 90.0, 0.8),
            (300.0, 300.0, 400.0, 400.0, 0.7),
        ];
        let out = match_image(&faces, &dets, 0.5);
        assert_eq!(out, vec![(0.9, true), (0.8, false), (0.7, false)]);
    }
}
