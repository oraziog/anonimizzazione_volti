//! Offline evaluation of the YOLO detector against the WIDER FACE validation
//! ground truth (spec open point: benchmark the shipped model, tune
//! confidence/NMS thresholds).
//!
//! Run as a CLI subcommand (no server, no config):
//! ```text
//! anonimizzazione_volti eval-wider \
//!     --model <yolov8n-face.onnx> \
//!     --images <WIDER_val/> \
//!     --gt <wider_face_split/wider_face_val_bbx_gt.txt> \
//!     [--det-conf 0.001] [--nms-iou 0.45] [--iou 0.5] [--max-images N]
//! ```
//!
//! `--det-conf` and `--nms-iou` are the *detector* knobs (confidence gate +
//! NMS IoU, mirroring `YOLO_CONF_THRESHOLD` / `YOLO_NMS_IOU` from the
//! service config); `--iou` is the *evaluation* matching threshold (0.5,
//! following the reference evaluator). They are deliberately separate so the
//! effect of each knob on the metric can be measured independently.
//!
//! GT format (official `wider_face_val_bbx_gt.txt`), per image:
//! ```text
//! <name>.jpg
//! <count>
//! x1 y1 w h blur expression illumination occlusion pose ignore
//! ```
//! Difficulty is assigned **per face** (the official easy/medium/hard `.mat`
//! index lists are not shipped with the plain txt; this rule is the standard
//! attribute-based approximation of them):
//! - `easy`   — `min(w,h) >= 60` and blur/illumination/occlusion/pose all 0
//! - `hard`   — `min(w,h) < 30` or `blur == 2` or `occlusion == 2`
//! - `medium` — otherwise
//!
//! Faces marked `ignore == 1` (too small / dense regions) never count toward
//! recall but still consume a matching detection, exactly like the reference
//! evaluator (wondervictor/WiderFace-Evaluation).
//!
//! The matching + AP algorithm follows that reference implementation
//! (wondervictor/WiderFace-Evaluation `evaluation.py`), including its exact
//! counting semantics: greedy per-detection assignment to the best-overlapping
//! GT in score-descending order; a detection matched to a face outside the
//! current difficulty (or `ignore == 1`) is *excluded* from both TP and FP;
//! a duplicate match of an already-consumed in-setting face counts as a plain
//! FP; anything else unmatched is an FP. The 1000-point sweep walks each
//! image's sorted detections in descending score and sums per-threshold
//! prefix TP/proposals across images (no re-inference at each threshold),
//! then VOC-style AP with precision envelope per difficulty split.

use std::collections::BTreeMap;
use std::path::PathBuf;

use anyhow::{bail, Context, Result};

use crate::model_loader::load_session;
use crate::models::run_yolo;

const SETTING_NAMES: [&str; 3] = ["easy", "medium", "hard"];
const SWEEP_STEPS: usize = 1000;
const OPERATING_POINTS: [f32; 5] = [0.5, 0.25, 0.1, 0.05, 0.02];

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
enum Difficulty {
    Easy = 0,
    Medium = 1,
    Hard = 2,
}

#[derive(Clone, Copy, Debug)]
struct GtBox {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    diff: Difficulty,
    /// `pose` attribute from the GT (0 = typical frontal, 1 = atypical,
    /// e.g. profile). Used for the profile-recall breakdown.
    pose: f32,
    /// `ignore == 1` in the GT file: excluded from recall, still consumes
    /// matching detections.
    ignored: bool,
}

struct GtImage {
    name: String,
    boxes: Vec<GtBox>,
}

/// Per-detection label, assigned greedily in score-descending order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DetLabel {
    /// Matched an unconsumed in-setting face.
    Tp,
    /// Unmatched, or a duplicate of an already-consumed in-setting face.
    Fp,
    /// Matched an out-of-setting or `ignore == 1` face: counts as neither
    /// TP nor FP (mirrors `proposal_list = -1` in the reference).
    Excluded,
}

// ─── GT parsing ─────────────────────────────────────────────────────────────

fn parse_gt(path: &PathBuf) -> Result<Vec<GtImage>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("read GT file {}", path.display()))?;
    let lines = text.lines().map(|l| l.trim()).filter(|l| !l.is_empty());

    let mut images = Vec::new();
    let mut cur_name: Option<String> = None;
    let mut cur_boxes: Vec<GtBox> = Vec::new();
    let mut expected: usize = 0;

    let mut flush = |name: &Option<String>, boxes: &mut Vec<GtBox>| {
        if let Some(n) = name {
            images.push(GtImage {
                name: n.clone(),
                boxes: std::mem::take(boxes),
            });
        }
    };

    // State machine: name line → count line → count box lines.
    let mut state = 0u8; // 0 = expect name, 1 = expect count, 2 = expect boxes
    let mut parsed: usize = 0;
    for line in lines {
        match state {
            0 => {
                flush(&cur_name, &mut cur_boxes);
                cur_name = Some(line.to_string());
                state = 1;
            }
            1 => {
                expected = line
                    .parse::<usize>()
                    .with_context(|| format!("invalid face count line: {line:?}"))?;
                parsed = 0;
                state = 2;
            }
            _ => {
                let f: Vec<f32> = line
                    .split_whitespace()
                    .map(|t| t.parse::<f32>())
                    .collect::<std::result::Result<_, _>>()
                    .with_context(|| format!("invalid GT box line: {line:?}"))?;
                if f.len() < 4 {
                    bail!("GT box line has < 4 fields: {line:?}");
                }
                let (x1, y1, w, h) = (f[0], f[1], f[2], f[3]);
                let blur = f.get(4).copied().unwrap_or(0.0);
                let illum = f.get(6).copied().unwrap_or(0.0);
                let occ = f.get(7).copied().unwrap_or(0.0);
                let pose = f.get(8).copied().unwrap_or(0.0);
                let ignored = f.get(9).copied().unwrap_or(0.0) == 1.0;
                cur_boxes.push(GtBox {
                    x1,
                    y1,
                    x2: x1 + w,
                    y2: y1 + h,
                    diff: difficulty(w, h, blur, illum, occ, pose),
                    pose,
                    ignored,
                });
                parsed += 1;
                if parsed >= expected {
                    state = 0;
                }
            }
        }
    }
    flush(&cur_name, &mut cur_boxes);
    Ok(images)
}

/// Labels detections against GT faces carrying a specific `pose` value
/// (pose==1 = atypical profile, the case that motivated this benchmark).
/// Greedy matching against ALL boxes (so a detection matching a frontal face
/// consumes it and is not double-counted); TP only for non-ignored boxes with
/// the target pose, everything else Excluded/FP — same semantics as the
/// per-setting `label_detections`.
fn label_pose(
    gt: &[GtBox],
    pose_target: f32,
    dets: &[(f32, f32, f32, f32, f32)], // (x1, y1, x2, y2, score)
    iou_thresh: f32,
) -> Vec<DetLabel> {
    let mut consumed = vec![false; gt.len()];
    let mut labels = Vec::with_capacity(dets.len());
    for (x1, y1, x2, y2, _score) in dets {
        let mut best_iou = 0.0f32;
        let mut best_idx = usize::MAX;
        for (i, b) in gt.iter().enumerate() {
            let o = iou(b, *x1, *y1, *x2, *y2);
            if o > best_iou {
                best_iou = o;
                best_idx = i;
            }
        }
        if best_iou >= iou_thresh {
            let b = &gt[best_idx];
            if b.ignored || b.pose != pose_target {
                labels.push(DetLabel::Excluded);
            } else if !consumed[best_idx] {
                consumed[best_idx] = true;
                labels.push(DetLabel::Tp);
            } else {
                labels.push(DetLabel::Fp);
            }
        } else {
            labels.push(DetLabel::Fp);
        }
    }
    labels
}

fn difficulty(w: f32, h: f32, blur: f32, illum: f32, occ: f32, pose: f32) -> Difficulty {
    let min_side = w.min(h);
    if min_side >= 60.0 && blur == 0.0 && illum == 0.0 && occ == 0.0 && pose == 0.0 {
        Difficulty::Easy
    } else if min_side < 30.0 || blur == 2.0 || occ == 2.0 {
        Difficulty::Hard
    } else {
        Difficulty::Medium
    }
}

// ─── Matching (reference algorithm) ─────────────────────────────────────────

fn iou(a: &GtBox, x1: f32, y1: f32, x2: f32, y2: f32) -> f32 {
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

/// Greedy assignment for one setting on one image (detections sorted by
/// descending score), labelling each detection TP/FP/Excluded exactly like
/// the reference evaluator.
fn label_detections(
    gt: &[GtBox],
    setting: Difficulty,
    dets: &[(f32, f32, f32, f32, f32)], // (x1, y1, x2, y2, score)
    iou_thresh: f32,
) -> Vec<DetLabel> {
    let mut consumed = vec![false; gt.len()];
    let mut labels = Vec::with_capacity(dets.len());
    for (x1, y1, x2, y2, _score) in dets {
        // Best-overlap GT (greedy, per the reference implementation).
        let mut best_iou = 0.0f32;
        let mut best_idx = usize::MAX;
        for (i, b) in gt.iter().enumerate() {
            let o = iou(b, *x1, *y1, *x2, *y2);
            if o > best_iou {
                best_iou = o;
                best_idx = i;
            }
        }
        if best_iou >= iou_thresh {
            let b = &gt[best_idx];
            if b.ignored || b.diff != setting {
                // Matched an out-of-setting / ignored face: excluded (neither
                // TP nor FP), does NOT consume the GT (reference semantics).
                labels.push(DetLabel::Excluded);
            } else if !consumed[best_idx] {
                consumed[best_idx] = true;
                labels.push(DetLabel::Tp);
            } else {
                // Duplicate match of an already-consumed in-setting face.
                labels.push(DetLabel::Fp);
            }
        } else {
            labels.push(DetLabel::Fp);
        }
    }
    labels
}

// ─── AP (VOC-style, reference algorithm) ────────────────────────────────────

fn voc_ap(recall: &[f64], precision: &[f64]) -> f64 {
    let n = recall.len();
    let mut mrec = vec![0.0f64; n + 2];
    let mut mpre = vec![0.0f64; n + 2];
    mrec[1..=n].copy_from_slice(recall);
    mrec[n + 1] = 1.0;
    mpre[1..=n].copy_from_slice(precision);
    // Precision envelope (monotone non-increasing recall → max precision).
    for i in (0..n + 1).rev() {
        mpre[i] = mpre[i].max(mpre[i + 1]);
    }
    let mut ap = 0.0;
    for i in 0..n + 1 {
        if mrec[i + 1] != mrec[i] {
            ap += (mrec[i + 1] - mrec[i]) * mpre[i + 1];
        }
    }
    ap
}

struct SettingResult {
    count_faces: usize,
    sweep: Vec<(f64, f64)>, // (recall, precision) per sweep step
    ap: f64,
}

/// Builds the per-difficulty PR curve from the labelled detections of every
/// image. Scores are already descending within each image, so for sweep step
/// `k` (threshold `1-(k+1)/N`, decreasing with `k`) the per-image prefix is
/// `0..di` and is reached by advancing one pointer per step — exactly the
/// reference `img_pr_info` but summed across images instead of re-scanning
/// detections at each of the 1000 thresholds.
fn evaluate_setting(all_images: &[(Vec<f32>, Vec<DetLabel>)], count_faces: usize) -> SettingResult {
    let mut curve = vec![(0.0f64, 0.0f64); SWEEP_STEPS];
    if count_faces == 0 {
        return SettingResult {
            count_faces: 0,
            sweep: curve,
            ap: 0.0,
        };
    }
    let mut tp_at = vec![0usize; SWEEP_STEPS];
    let mut prop_at = vec![0usize; SWEEP_STEPS];
    for (scores, labels) in all_images {
        debug_assert_eq!(scores.len(), labels.len());
        // Per-detection cumulative (tp, proposals) inside the prefix.
        let mut tp = 0usize;
        let mut prop = 0usize;
        let mut pref_tp = Vec::with_capacity(labels.len());
        let mut pref_prop = Vec::with_capacity(labels.len());
        for lbl in labels {
            match lbl {
                DetLabel::Tp => {
                    tp += 1;
                    prop += 1;
                }
                DetLabel::Fp => prop += 1,
                DetLabel::Excluded => {}
            }
            pref_tp.push(tp);
            pref_prop.push(prop);
        }
        // Thresholds go from high (k=0, ~0.999) to low (k=N-1, ~0.001), so
        // the prefix only grows: one forward pointer per image, exact per
        // step.
        let mut di = 0usize;
        for k in 0..SWEEP_STEPS {
            let thresh = 1.0 - (k + 1) as f64 / SWEEP_STEPS as f64;
            while di < scores.len() && scores[di] as f64 >= thresh {
                di += 1;
            }
            if di > 0 {
                tp_at[k] += pref_tp[di - 1];
                prop_at[k] += pref_prop[di - 1];
            }
        }
    }
    for k in 0..SWEEP_STEPS {
        let recall = tp_at[k] as f64 / count_faces as f64;
        let precision = if prop_at[k] > 0 {
            tp_at[k] as f64 / prop_at[k] as f64
        } else {
            0.0
        };
        curve[k] = (recall, precision);
    }
    let recall: Vec<f64> = curve.iter().map(|(r, _)| *r).collect();
    let precision: Vec<f64> = curve.iter().map(|(_, p)| *p).collect();
    let ap = voc_ap(&recall, &precision);
    SettingResult {
        count_faces,
        sweep: curve,
        ap,
    }
}

// ─── CLI entry ──────────────────────────────────────────────────────────────

pub fn run(args: &[String]) -> Result<()> {
    let mut model = None;
    let mut images = None;
    let mut gt = None;
    let mut det_conf = 0.001f32;
    let mut nms_iou = 0.45f32;
    let mut iou = 0.5f32;
    let mut max_images = 0usize;
    let mut stride = 1usize;

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
            "--gt" => gt = Some(next()?),
            "--det-conf" => det_conf = next()?.parse().context("--det-conf must be a float")?,
            "--nms-iou" => nms_iou = next()?.parse().context("--nms-iou must be a float")?,
            "--iou" => iou = next()?.parse().context("--iou must be a float")?,
            "--max-images" => {
                max_images = next()?.parse().context("--max-images must be an integer")?
            }
            "--stride" => {
                stride = next()?
                    .parse::<usize>()
                    .context("--stride must be an integer")?
                    .max(1);
                tracing::info!("eval-wider: sampling every {stride}th GT image");
            }
            other => bail!("unknown eval-wider argument: {other}"),
        }
    }
    let model = model.context("missing --model")?;
    let images = images.context("missing --images")?;
    let gt = gt.context("missing --gt")?;

    let gts = parse_gt(&PathBuf::from(&gt))?;
    if gts.is_empty() {
        bail!("no images parsed from {}", PathBuf::from(&gt).display());
    }
    tracing::info!(
        "eval-wider: {} GT images, det_conf={det_conf}, nms_iou={nms_iou}, match_iou={iou}, model={model}",
        gts.len()
    );

    let mut session = load_session(&PathBuf::from(&model))?;

    // Per-setting accumulation: faces count + (scores, labels) per image.
    let mut acc = BTreeMap::<Difficulty, (usize, Vec<(Vec<f32>, Vec<DetLabel>)>)>::new();
    // Pose breakdown: (count_faces, labels per image) for pose==0 and pose==1.
    type PoseAcc = (usize, Vec<(Vec<f32>, Vec<DetLabel>)>);
    let mut acc_pose: [PoseAcc; 2] = std::array::from_fn(|_| (0usize, Vec::new()));
    let mut processed = 0usize;
    let mut missing = 0usize;
    for (idx, gt_img) in gts.iter().enumerate() {
        if stride > 1 && idx % stride != 0 {
            continue;
        }
        if max_images > 0 && processed >= max_images {
            break;
        }
        let img_path = PathBuf::from(&images).join(&gt_img.name);
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
        tracing::debug!("eval-wider: {} → {} detections", gt_img.name, dets.len());
        for (i, d) in dets.iter().take(6).enumerate() {
            tracing::debug!(
                "  top det {i}: ({:.0},{:.0},{:.0},{:.0}) conf {:.3}",
                d.bbox.x0,
                d.bbox.y0,
                d.bbox.x1,
                d.bbox.y1,
                d.confidence
            );
        }
        // Sort detections by descending score (sweep contract).
        let mut dets: Vec<(f32, f32, f32, f32, f32)> = dets
            .into_iter()
            .map(|d| (d.bbox.x0, d.bbox.y0, d.bbox.x1, d.bbox.y1, d.confidence))
            .collect();
        dets.sort_by(|a, b| b.4.total_cmp(&a.4));

        for setting in [Difficulty::Easy, Difficulty::Medium, Difficulty::Hard] {
            let entry = acc.entry(setting).or_insert_with(|| (0, Vec::new()));
            entry.0 += gt_img
                .boxes
                .iter()
                .filter(|b| !b.ignored && b.diff == setting)
                .count();
            let scores: Vec<f32> = dets.iter().map(|d| d.4).collect();
            entry
                .1
                .push((scores, label_detections(&gt_img.boxes, setting, &dets, iou)));
        }
        for pose_val in [0usize, 1usize] {
            acc_pose[pose_val].0 += gt_img
                .boxes
                .iter()
                .filter(|b| !b.ignored && b.pose == pose_val as f32)
                .count();
            let scores: Vec<f32> = dets.iter().map(|d| d.4).collect();
            acc_pose[pose_val]
                .1
                .push((scores, label_pose(&gt_img.boxes, pose_val as f32, &dets, iou)));
        }
        processed += 1;
        if processed.is_multiple_of(50) {
            tracing::info!("eval-wider: {processed} images processed");
        }
    }

    if processed == 0 {
        bail!("no images could be processed (check --images layout; {missing} unreadable)");
    }
    tracing::info!("eval-wider: done ({processed} processed, {missing} unreadable)");

    // Per-setting results.
    let mut per_setting = Vec::new();
    for setting in [Difficulty::Easy, Difficulty::Medium, Difficulty::Hard] {
        let (count_faces, images) = acc.get(&setting).cloned().unwrap_or((0, Vec::new()));
        let res = evaluate_setting(&images, count_faces);
        per_setting.push((setting, res));
    }

    // Pose breakdown results (recall of profile faces is the headline metric
    // for this benchmark).
    let mut pose_rows = Vec::new();
    for pose_val in [0usize, 1usize] {
        let (count, images) = &acc_pose[pose_val];
        let res = evaluate_setting(images, *count);
        pose_rows.push((pose_val, count, res));
    }

    println!();
    println!("WIDER FACE validation — {model}");
    println!(
        "det_conf={det_conf}  nms_iou={nms_iou}  match_iou={iou}  images={processed}  missing={missing}"
    );
    let header: Vec<String> = OPERATING_POINTS
        .iter()
        .flat_map(|c| vec![format!("R@{c}"), format!("P@{c}")])
        .collect();
    let dash: Vec<String> = OPERATING_POINTS
        .iter()
        .flat_map(|_| vec!["-----".into(), "-----".into()])
        .collect();
    println!();
    println!(
        "{:<8} {:>10} {:>8} {}",
        "setting",
        "faces",
        "AP",
        header.join(" ")
    );
    println!("{:<8} {:>10} {:>8} {}", "-------", "-----", "----", dash.join(" "));
    for (setting, res) in &per_setting {
        let name = SETTING_NAMES[*setting as usize];
        let mut cols = String::new();
        for c in OPERATING_POINTS {
            let (r, p) = point_at(&res.sweep, c);
            cols.push_str(&format!(" {r:>7.3} {p:>7.3}"));
        }
        println!(
            "{:<8} {:>10} {:>8.3}{}",
            name, res.count_faces, res.ap, cols
        );
    }
    println!();
    println!("Pose breakdown (pose=0 frontal, pose=1 atypical/profile):");
    let r_header: Vec<String> = OPERATING_POINTS.iter().map(|c| format!("R@{c}")).collect();
    let r_dash: Vec<&str> = OPERATING_POINTS.iter().map(|_| "-----").collect();
    println!("{:<8} {:>10} {:>8} {}", "pose", "faces", "AP", r_header.join(" "));
    println!("{:<8} {:>10} {:>8} {}", "----", "-----", "----", r_dash.join(" "));
    for (pose_val, count, res) in &pose_rows {
        let mut cols = String::new();
        for c in OPERATING_POINTS {
            let (r, _p) = point_at(&res.sweep, c);
            cols.push_str(&format!(" {r:>7.3}"));
        }
        println!(
            "{:<8} {:>10} {:>8.3}{}",
            pose_val, count, res.ap, cols
        );
    }
    println!();
    Ok(())
}

/// Recall/precision at the operating point defined by the last sweep
/// detection with score >= `conf`. Returns (recall, precision).
fn point_at(sweep: &[(f64, f64)], conf: f32) -> (f64, f64) {
    // The sweep is indexed by threshold 1-(k+1)/N; find the highest k whose
    // threshold is >= conf (i.e. the last detection above conf).
    let mut best = (0.0f64, 0.0f64);
    for (k, (recall, precision)) in sweep.iter().enumerate() {
        let thresh = 1.0 - (k + 1) as f64 / SWEEP_STEPS as f64;
        if thresh + 1e-9 >= conf as f64 {
            best = (*recall, *precision);
        }
    }
    best
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn gt_box(x1: f32, y1: f32, w: f32, h: f32, d: Difficulty, ignored: bool) -> GtBox {
        GtBox {
            x1,
            y1,
            x2: x1 + w,
            y2: y1 + h,
            diff: d,
            pose: 0.0,
            ignored,
        }
    }

    #[test]
    fn difficulty_rule() {
        assert_eq!(difficulty(80.0, 80.0, 0.0, 0.0, 0.0, 0.0), Difficulty::Easy);
        assert_eq!(
            difficulty(80.0, 80.0, 1.0, 0.0, 0.0, 0.0),
            Difficulty::Medium
        );
        assert_eq!(
            difficulty(80.0, 80.0, 0.0, 0.0, 1.0, 0.0),
            Difficulty::Medium
        );
        assert_eq!(difficulty(20.0, 80.0, 0.0, 0.0, 0.0, 0.0), Difficulty::Hard);
        assert_eq!(difficulty(80.0, 80.0, 2.0, 0.0, 0.0, 0.0), Difficulty::Hard);
        assert_eq!(
            difficulty(40.0, 40.0, 0.0, 0.0, 0.0, 0.0),
            Difficulty::Medium
        );
    }

    #[test]
    fn gt_parser_roundtrip() {
        let dir = std::env::temp_dir().join("av-eval-gt-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("gt.txt");
        std::fs::write(
            &path,
            "0--Parade/0_Parade_1.jpg\n2\n10 10 80 80 0 0 0 0 0 0\n5 5 10 10 1 0 0 1 0 0\n\
             1--Photo/1_Photo_2.jpg\n0\n",
        )
        .unwrap();
        let gts = parse_gt(&path).unwrap();
        assert_eq!(gts.len(), 2);
        assert_eq!(gts[0].name, "0--Parade/0_Parade_1.jpg");
        assert_eq!(gts[0].boxes.len(), 2);
        assert_eq!(gts[0].boxes[0].diff, Difficulty::Easy);
        assert!(!gts[0].boxes[0].ignored);
        assert_eq!(gts[0].boxes[1].diff, Difficulty::Hard);
        assert!(!gts[0].boxes[1].ignored);
        assert!(gts[1].boxes.is_empty());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn matching_and_ap() {
        // One image: one easy GT face at (0,0,100,100), detection overlapping
        // it with score 0.9 → AP must be 1.0 for easy, 0 for the others.
        let gt = vec![gt_box(0.0, 0.0, 100.0, 100.0, Difficulty::Easy, false)];
        for (setting, expected) in [
            (Difficulty::Easy, 1.0),
            (Difficulty::Medium, 0.0),
            (Difficulty::Hard, 0.0),
        ] {
            let dets = vec![(0.0, 0.0, 100.0, 100.0, 0.9)];
            let labels = label_detections(&gt, setting, &dets, 0.5);
            let images = vec![(vec![0.9], labels)];
            let res = evaluate_setting(&images, 1);
            assert!((res.ap - expected).abs() < 1e-6, "{setting:?}: {}", res.ap);
        }

        // Detection overlapping an ignored easy face: excluded → no TP, AP 0.
        let gt = vec![gt_box(0.0, 0.0, 100.0, 100.0, Difficulty::Easy, true)];
        let dets = vec![(0.0, 0.0, 100.0, 100.0, 0.9)];
        let labels = label_detections(&gt, Difficulty::Easy, &dets, 0.5);
        assert_eq!(labels, vec![DetLabel::Excluded]);
        let images = vec![(vec![0.9], labels)];
        let res = evaluate_setting(&images, 0);
        assert_eq!(res.ap, 0.0);
    }

    #[test]
    fn sweep_prefix_and_operating_points() {
        // One easy GT face. Detections (desc by score): a TP at 0.9, an FP at
        // 0.7, and a duplicate TP attempt at 0.4 (counts as FP, reference
        // semantics).
        let gt = vec![gt_box(0.0, 0.0, 100.0, 100.0, Difficulty::Easy, false)];
        let dets = vec![
            (0.0, 0.0, 100.0, 100.0, 0.9),
            (300.0, 300.0, 400.0, 400.0, 0.7),
            (10.0, 10.0, 90.0, 90.0, 0.4),
        ];
        let labels = label_detections(&gt, Difficulty::Easy, &dets, 0.5);
        assert_eq!(labels, vec![DetLabel::Tp, DetLabel::Fp, DetLabel::Fp]);
        let images = vec![(dets.iter().map(|d| d.4).collect(), labels)];
        let res = evaluate_setting(&images, 1);
        // P@0.5: prefix = {0.9 TP, 0.7 FP} → recall 1.0, precision 0.5.
        let (r50, p50) = point_at(&res.sweep, 0.5);
        assert!((r50 - 1.0).abs() < 1e-6, "r@0.5 = {r50}");
        assert!((p50 - 0.5).abs() < 1e-6, "p@0.5 = {p50}");
        // P@0.2: all three in the prefix → precision 1/3.
        let (_, p20) = point_at(&res.sweep, 0.2);
        assert!((p20 - 1.0 / 3.0).abs() < 1e-6, "p@0.2 = {p20}");
        // Perfect ordering (TP first, then the two FPs below it) gives the
        // AP of the full precision-1.0 recall-1.0 plateau: > 0.9.
        assert!(res.ap > 0.9, "ap = {}", res.ap);
    }

    #[test]
    fn excluded_detections_do_not_hurt_precision() {
        // A detection overlapping a hard face (out of the easy setting) is
        // excluded from easy precision — reference `proposal_list = -1`.
        let gt = vec![gt_box(0.0, 0.0, 100.0, 100.0, Difficulty::Hard, false)];
        let dets = vec![(0.0, 0.0, 100.0, 100.0, 0.9)];
        let labels = label_detections(&gt, Difficulty::Easy, &dets, 0.5);
        assert_eq!(labels, vec![DetLabel::Excluded]);
        // Add an easy face matched by a first TP detection so the setting has
        // one face; the detection over the hard face stays excluded and must
        // not dilute the easy precision.
        let gt = vec![
            gt_box(0.0, 0.0, 100.0, 100.0, Difficulty::Easy, false),
            gt_box(500.0, 0.0, 100.0, 100.0, Difficulty::Hard, false),
        ];
        let dets = vec![
            (0.0, 0.0, 100.0, 100.0, 0.9),
            (500.0, 0.0, 600.0, 100.0, 0.8),
        ];
        let labels = label_detections(&gt, Difficulty::Easy, &dets, 0.5);
        assert_eq!(labels, vec![DetLabel::Tp, DetLabel::Excluded]);
        let images = vec![(dets.iter().map(|d| d.4).collect(), labels)];
        let res = evaluate_setting(&images, 1);
        let (_, p50) = point_at(&res.sweep, 0.5);
        assert_eq!(p50, 1.0);
    }

    #[test]
    fn point_at_operating() {
        let sweep = vec![(0.0, 1.0); SWEEP_STEPS];
        let (_, p) = point_at(&sweep, 0.5);
        assert_eq!(p, 1.0);
    }
}
