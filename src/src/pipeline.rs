//! Processing pipeline (spec §8 `pipeline.rs`): FSM-conditional blur
//! application, mask compositing, false-positive crop extraction.
//!
//! INITIAL → cautious full-frame blur; LEARNING → YOLO box + 15% margin blur
//! with sigma = box_width / 8 clamped [5, 50], detection persistence, dubious
//! crop extraction; ACTIVE → ROI gate + optional classifier + polygon/ellipse
//! masked blur.

use anyhow::{anyhow, Result};
use image::{DynamicImage, Luma, Rgba};

use crate::config::{AnonMode, Config, MaskSegmenter};
use crate::db::CameraState;
use crate::models::{
    run_classifier, run_detector, run_selfie_segmenter, FaceDetection, Keypoints, ModelStore, Rect,
};
use crate::roi::RoiPolygon;

/// Which FSM branch processed the image (used for logging/metrics).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Branch {
    Initial,
    Learning,
    Active,
}

impl Branch {
    #[cfg(feature = "s3")]
    pub fn as_str(&self) -> &'static str {
        match self {
            Branch::Initial => "initial",
            Branch::Learning => "learning",
            Branch::Active => "active",
        }
    }
}

/// The anonymization operation configured for this service (env `ANON_MODE`):
/// Gaussian-style blur or Street-View-style pixelation (mosaic). Carries its
/// own parameters so every FSM branch dispatches on the same knob.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AnonOp {
    /// Gaussian-ish box blur with the given sigma.
    Blur(f32),
    /// Mosaic: average each `cell × cell` block.
    Pixelate(u32),
}

impl AnonOp {
    /// The service-wide operation, from config.
    pub fn from_cfg(cfg: &Config) -> Self {
        match cfg.anon_mode {
            AnonMode::Blur => AnonOp::Blur(cfg.initial_blur_sigma),
            AnonMode::Pixelate => AnonOp::Pixelate(cfg.pixelate_cell_px),
        }
    }

    /// Applies the operation to a rectangular region (LEARNING boxes).
    fn apply_region(&self, img: &mut image::RgbaImage, region: Rect) {
        match *self {
            AnonOp::Blur(s) => blur_region_rgba(img, region, s),
            AnonOp::Pixelate(c) => pixelate_region_rgba(img, region, c),
        }
    }

    /// Applies the operation only where `mask` is set (ACTIVE), softening the
    /// mask edge with `mask_sigma` first.
    fn apply_masked(&self, img: &mut image::RgbaImage, mask: &image::GrayImage, mask_sigma: f32) {
        match *self {
            AnonOp::Blur(s) => apply_masked_blur(img, mask, s),
            AnonOp::Pixelate(c) => apply_masked_pixelate(img, mask, mask_sigma, c),
        }
    }

    /// Applies the operation to the whole frame (INITIAL / failsafe).
    pub fn apply_full_frame(&self, img: &mut image::RgbaImage) {
        match *self {
            AnonOp::Blur(s) => blur_full_frame(img, s),
            AnonOp::Pixelate(c) => pixelate_full_frame(img, c),
        }
    }
}

/// Outcome of processing one image.
#[derive(Debug)]
pub struct ProcessOutcome {
    pub branch: Branch,
    pub detections: Vec<FaceDetection>,
    /// Detection centers to persist during LEARNING (background collection)
    /// and ACTIVE (fresh data for the scheduled ROI re-extraction): for ACTIVE
    /// only the actually-anonymized detections are recorded.
    pub record_detections: bool,
    /// Crops to save as false-positive candidates (LEARNING).
    pub fp_crops: Vec<(u32, u32, DynamicImage)>,
    /// The processed (blurred) RGBA frame, ready for encoding.
    pub processed: image::RgbaImage,
}

/// Gaussian sigma from box width: `sigma = box_width / 8` clamped to [5, 50]
/// (spec §4, used in both LEARNING and ACTIVE).
pub fn sigma_for_box(box_width: f32) -> f32 {
    (box_width / 8.0).clamp(5.0, 50.0)
}

/// Blur an image region in place with a box filter approximation of the
/// requested Gaussian sigma (integer radius, alpha-weighted edge handling).
///
/// A true Gaussian via `imageproc::filter::gaussian_blur_f32` allocates a full
/// RGBA f32 copy per call (~33MB at 1080p ×4 channels); for 10k images under
/// concurrency this box-blur approximation (2 passes ≈ Gaussian by CLT) keeps
/// peak memory bounded while remaining visually equivalent for anonymization.
fn blur_region_rgba(img: &mut image::RgbaImage, region: Rect, sigma: f32) {
    let radius = ((sigma * 1.5).round() as i32).max(2) as i64;
    box_blur_region(img, region, radius);
    box_blur_region(img, region, radius);
}

fn box_blur_region(img: &mut image::RgbaImage, region: Rect, radius: i64) {
    let (w, h) = (img.width() as i64, img.height() as i64);
    if w <= 0 || h <= 0 {
        return;
    }
    let x0 = (region.x0.floor() as i64).clamp(0, w - 1);
    let y0 = (region.y0.floor() as i64).clamp(0, h - 1);
    let x1 = (region.x1.ceil() as i64).clamp(0, w - 1);
    let y1 = (region.y1.ceil() as i64).clamp(0, h - 1);
    if x1 <= x0 || y1 <= y0 {
        return;
    }

    // Horizontal pass into a temp buffer, then vertical pass back into img.
    let rw = (x1 - x0 + 1) as usize;
    let rh = (y1 - y0 + 1) as usize;
    let mut tmp = vec![[0u32; 4]; rw * rh];

    for y in y0..=y1 {
        for x in x0..=x1 {
            let mut acc = [0u32; 4];
            let mut count = 0u32;
            for dx in (x - radius)..=(x + radius) {
                if dx < 0 || dx >= w {
                    continue;
                }
                let p = img.get_pixel(dx as u32, y as u32);
                acc[0] += p.0[0] as u32;
                acc[1] += p.0[1] as u32;
                acc[2] += p.0[2] as u32;
                acc[3] += p.0[3] as u32;
                count += 1;
            }
            let idx = ((y - y0) as usize) * rw + (x - x0) as usize;
            tmp[idx] = [
                acc[0] / count,
                acc[1] / count,
                acc[2] / count,
                acc[3] / count,
            ];
        }
    }
    for y in y0..=y1 {
        for x in x0..=x1 {
            let mut acc = [0u32; 4];
            let mut count = 0u32;
            for dy in (y - radius)..=(y + radius) {
                if dy < y0 || dy > y1 {
                    continue;
                }
                let idx = ((dy - y0) as usize) * rw + (x - x0) as usize;
                acc[0] += tmp[idx][0];
                acc[1] += tmp[idx][1];
                acc[2] += tmp[idx][2];
                acc[3] += tmp[idx][3];
                count += 1;
            }
            let p = img.get_pixel_mut(x as u32, y as u32);
            p.0[0] = (acc[0] / count) as u8;
            p.0[1] = (acc[1] / count) as u8;
            p.0[2] = (acc[2] / count) as u8;
            p.0[3] = (acc[3] / count) as u8;
        }
    }
}

/// Bounding box of the non-zero mask pixels. Returns `(x, y, w, h)` or
/// `(0, 0, 0, 0)` when the mask is empty (nothing to do).
fn mask_bbox(mask: &image::GrayImage) -> (u32, u32, u32, u32) {
    let (w, h) = mask.dimensions();
    let mut min = (w, h);
    let mut max = (0u32, 0u32);
    for (x, y, m) in mask.enumerate_pixels() {
        if m.0[0] > 0 {
            min = (min.0.min(x), min.1.min(y));
            max = (max.0.max(x + 1), max.1.max(y + 1));
        }
    }
    if max.0 <= min.0 || max.1 <= min.1 {
        return (0, 0, 0, 0);
    }
    (min.0, min.1, max.0 - min.0, max.1 - min.1)
}

/// Applies an operation only where the mask is set (mask: 255 = op, 0 = keep).
/// The mask is blurred slightly first to avoid hard aliasing at edges.
///
/// Feather, blur and compositing all run on the mask's **bounding box** only —
/// per-face cost is O(mask bbox) instead of O(full frame), which matters when
/// an ACTIVE frame holds many faces (one full-frame copy was made per face).
/// `op` receives the scratch copy plus its origin `(ox, oy)` in frame
/// coordinates (used by the mosaic to keep blocks aligned to the frame grid).
fn apply_masked(
    img: &mut image::RgbaImage,
    mask: &image::GrayImage,
    sigma: f32,
    op: impl Fn(&mut image::RgbaImage, i64, i64),
) {
    let (w, h) = img.dimensions();
    debug_assert_eq!((w, h), mask.dimensions());
    let (ox, oy, bw, bh) = mask_bbox(mask);
    if bw == 0 || bh == 0 {
        return;
    }
    let soft: image::GrayImage = {
        let f32_mask: image::ImageBuffer<image::Luma<f32>, Vec<f32>> =
            image::ImageBuffer::from_fn(bw, bh, |x, y| {
                image::Luma([mask.get_pixel(ox + x, oy + y).0[0] as f32])
            });
        let blurred = imageproc::filter::gaussian_blur_f32(&f32_mask, sigma);
        image::GrayImage::from_fn(bw, bh, |x, y| {
            let v = blurred.get_pixel(x, y).0[0];
            image::Luma([v.round().clamp(0.0, 255.0) as u8])
        })
    };

    let mut op_img = image::imageops::crop_imm(img, ox, oy, bw, bh).to_image();
    op(&mut op_img, ox as i64, oy as i64);

    for (x, y, m) in soft.enumerate_pixels() {
        let a = m.0[0] as u32;
        if a == 0 {
            continue;
        }
        let orig = img.get_pixel(x + ox, y + oy);
        let processed = op_img.get_pixel(x, y);
        let mixed = [
            ((processed.0[0] as u32 * a + orig.0[0] as u32 * (255 - a)) / 255) as u8,
            ((processed.0[1] as u32 * a + orig.0[1] as u32 * (255 - a)) / 255) as u8,
            ((processed.0[2] as u32 * a + orig.0[2] as u32 * (255 - a)) / 255) as u8,
            orig.0[3],
        ];
        *img.get_pixel_mut(x + ox, y + oy) = Rgba(mixed);
    }
}

/// Blur only where the mask is set.
fn apply_masked_blur(img: &mut image::RgbaImage, mask: &image::GrayImage, sigma: f32) {
    apply_masked(img, mask, sigma, |scratch, _ox, _oy| {
        let (w, h) = scratch.dimensions();
        let full = Rect {
            x0: 0.0,
            y0: 0.0,
            x1: w as f32,
            y1: h as f32,
        };
        blur_region_rgba(scratch, full, sigma)
    });
}

/// Mosaic/pixelate (Google Street View style) only where the mask is set.
fn apply_masked_pixelate(
    img: &mut image::RgbaImage,
    mask: &image::GrayImage,
    mask_sigma: f32,
    cell: u32,
) {
    apply_masked(img, mask, mask_sigma, |scratch, ox, oy| {
        // Operator runs on the mask's bbox crop; the region start(-ox)+local
        // is a uniform sub-cell shift of the frame-anchored grid (the mosaic
        // stays block-uniform; the shift is capped at `cell-1` px).
        let (w, h) = scratch.dimensions();
        pixelate_region_rgba(
            scratch,
            Rect {
                x0: -(ox as f32),
                y0: -(oy as f32),
                x1: w as f32,
                y1: h as f32,
            },
            cell,
        )
    });
}

/// Mosaic: every `cell × cell` block inside the region becomes its average
/// color (the classic Street View / TV-news face anonymization).
fn pixelate_region_rgba(img: &mut image::RgbaImage, region: Rect, cell: u32) {
    let cell = (cell as i64).max(2);
    let (w, h) = (img.width() as i64, img.height() as i64);
    if w <= 0 || h <= 0 {
        return;
    }
    let x0 = (region.x0.floor() as i64).clamp(0, w - 1);
    let y0 = (region.y0.floor() as i64).clamp(0, h - 1);
    let x1 = (region.x1.ceil() as i64).clamp(0, w);
    let y1 = (region.y1.ceil() as i64).clamp(0, h);
    if x1 <= x0 || y1 <= y0 {
        return;
    }
    let bx = ((x1 - x0 + cell - 1) / cell) as usize;
    let by = ((y1 - y0 + cell - 1) / cell) as usize;
    let mut avg = vec![[0u32; 4]; bx * by];
    let mut cnt = vec![0u32; bx * by];
    for y in y0..y1 {
        for x in x0..x1 {
            let gx = ((x - x0) / cell) as usize;
            let gy = ((y - y0) / cell) as usize;
            let p = img.get_pixel(x as u32, y as u32).0;
            let a = &mut avg[gy * bx + gx];
            a[0] += p[0] as u32;
            a[1] += p[1] as u32;
            a[2] += p[2] as u32;
            a[3] += p[3] as u32;
            cnt[gy * bx + gx] += 1;
        }
    }
    for y in y0..y1 {
        for x in x0..x1 {
            let gx = ((x - x0) / cell) as usize;
            let gy = ((y - y0) / cell) as usize;
            let c = cnt[gy * bx + gx];
            let a = avg[gy * bx + gx];
            let p = img.get_pixel_mut(x as u32, y as u32);
            p.0[0] = (a[0] / c) as u8;
            p.0[1] = (a[1] / c) as u8;
            p.0[2] = (a[2] / c) as u8;
            p.0[3] = (a[3] / c) as u8;
        }
    }
}

/// Full-frame mosaic (INITIAL / failsafe pixelate path).
pub fn pixelate_full_frame(img: &mut image::RgbaImage, cell: u32) {
    let (w, h) = img.dimensions();
    pixelate_region_rgba(
        img,
        Rect {
            x0: 0.0,
            y0: 0.0,
            x1: w as f32,
            y1: h as f32,
        },
        cell,
    );
}

/// Signed cross product of edges `(a→b)` and `(a→p)`; sign tells which side
/// of the oriented edge `p` lies on (used by the convex fill below).
fn cross2(a: (f32, f32), b: (f32, f32), p: (f32, f32)) -> f32 {
    (b.0 - a.0) * (p.1 - a.1) - (b.1 - a.1) * (p.0 - a.0)
}

/// Builds a mask from a convex polygon ring (hull of keypoints, spec §4
/// ACTIVE). Self-contained convex fill (half-plane test per pixel); the ring
/// is expanded about its centroid by `margin_pct` first.
fn mask_from_polygon(w: u32, h: u32, polygon: &[(f32, f32)], margin_pct: f32) -> image::GrayImage {
    let mut mask = image::GrayImage::from_pixel(w, h, image::Luma([0u8]));
    if polygon.len() < 3 {
        return mask;
    }
    let n = polygon.len();
    let cx: f32 = polygon.iter().map(|p| p.0).sum::<f32>() / n as f32;
    let cy: f32 = polygon.iter().map(|p| p.1).sum::<f32>() / n as f32;
    let ring: Vec<(f32, f32)> = polygon
        .iter()
        .map(|(x, y)| {
            (
                cx + (x - cx) * (1.0 + margin_pct),
                cy + (y - cy) * (1.0 + margin_pct),
            )
        })
        .collect();

    // Orientation (signed area ×2); degenerate rings produce no mask.
    let mut area2 = 0.0f32;
    for i in 0..n {
        let j = (i + 1) % n;
        area2 += ring[i].0 * ring[j].1 - ring[j].0 * ring[i].1;
    }
    let orient = if area2 > 0.0 { 1.0 } else { -1.0 };
    if area2 == 0.0 {
        return mask;
    }

    let x0 = ring
        .iter()
        .map(|p| p.0.floor())
        .fold(f32::MAX, f32::min)
        .max(0.0) as u32;
    let x1 = ring
        .iter()
        .map(|p| p.0.ceil())
        .fold(f32::MIN, f32::max)
        .min(w as f32) as u32;
    let y0 = ring
        .iter()
        .map(|p| p.1.floor())
        .fold(f32::MAX, f32::min)
        .max(0.0) as u32;
    let y1 = ring
        .iter()
        .map(|p| p.1.ceil())
        .fold(f32::MIN, f32::max)
        .min(h as f32) as u32;

    for y in y0..y1 {
        for x in x0..x1 {
            let p = (x as f32 + 0.5, y as f32 + 0.5);
            let mut inside = true;
            for i in 0..n {
                let j = (i + 1) % n;
                if cross2(ring[i], ring[j], p) * orient < 0.0 {
                    inside = false;
                    break;
                }
            }
            if inside {
                mask.put_pixel(x, y, image::Luma([255u8]));
            }
        }
    }
    mask
}

/// Builds a mask from an axis-aligned rect (used by tests / full cover).
#[allow(dead_code)]
fn mask_from_rect(w: u32, h: u32, r: Rect) -> image::GrayImage {
    let mut mask = image::GrayImage::from_pixel(w, h, image::Luma([0u8]));
    let x0 = r.x0.round().max(0.0) as i64;
    let y0 = r.y0.round().max(0.0) as i64;
    let x1 = (r.x1.round() as i64).min(w as i64 - 1);
    let y1 = (r.y1.round() as i64).min(h as i64 - 1);
    if x1 > x0 && y1 > y0 {
        for y in y0..=y1 {
            for x in x0..=x1 {
                mask.put_pixel(x as u32, y as u32, image::Luma([255u8]));
            }
        }
    }
    mask
}

/// Builds a mask from an ellipse inscribed in the bounding box (spec §4
/// fallback when keypoints are unavailable).
fn mask_from_ellipse(w: u32, h: u32, r: Rect, margin_pct: f32) -> image::GrayImage {
    let mut mask = image::GrayImage::from_pixel(w, h, image::Luma([0u8]));
    let cx = (r.x0 + r.x1) / 2.0;
    let cy = (r.y0 + r.y1) / 2.0;
    let rx = ((r.x1 - r.x0) / 2.0 * (1.0 + margin_pct)).max(1.0);
    let ry = ((r.y1 - r.y0) / 2.0 * (1.0 + margin_pct)).max(1.0);
    let x0 = (cx - rx).floor().max(0.0) as u32;
    let x1 = (cx + rx).ceil().min(w as f32 - 1.0) as u32;
    let y0 = (cy - ry).floor().max(0.0) as u32;
    let y1 = (cy + ry).ceil().min(h as f32 - 1.0) as u32;
    for y in y0..=y1 {
        for x in x0..=x1 {
            let dx = (x as f32 - cx) / rx;
            let dy = (y as f32 - cy) / ry;
            if dx * dx + dy * dy <= 1.0 {
                mask.put_pixel(x, y, image::Luma([255u8]));
            }
        }
    }
    mask
}

/// Top-left origin of the [`crop_clamped`] crop for `r`, in image coordinates
/// (same clamping the crop uses, so the segmenter silhouette overlays back on
/// the frame at the exact pixels it segmented).
fn clamped_crop_origin(w: u32, h: u32, r: Rect) -> (u32, u32) {
    (
        (r.x0.floor() as i64).clamp(0, w as i64 - 1) as u32,
        (r.y0.floor() as i64).clamp(0, h as i64 - 1) as u32,
    )
}

/// Whether the per-face selfie segmenter should run for `det` (ACTIVE).
/// `cfg.segmenter_min_box_px <= 0` keeps it on every face; otherwise faces
/// narrower than the threshold are masked geometrically (the 256² ONNX run
/// buys nothing at tiny sizes and costs ~26 ms per face).
fn segmenter_applies(cfg: &Config, det: &FaceDetection) -> bool {
    cfg.segmenter_min_box_px <= 0.0 || det.bbox.width() >= cfg.segmenter_min_box_px
}

/// ACTIVE mask via the MediaPipe Selfie segmenter (env `MASK_SEGMENTER`):
/// per-face silhouette dilated 25% of the minor box side (square kernel) in
/// **union** with the box ellipse (5% margin), so hair/skin never bleeds
/// beyond the face box. `img` is the original RGB frame; the crop fed to the
/// model is the same [`crop_clamped`] the classifier uses.
fn segmenter_mask(
    w: u32,
    h: u32,
    store: &ModelStore,
    img: &image::ImageBuffer<image::Rgb<u8>, Vec<u8>>,
    det: &FaceDetection,
) -> Result<image::GrayImage> {
    let pool = store
        .segmenter_pool()
        .ok_or_else(|| anyhow!("face-mask segmenter not loaded"))?;
    let crop = crop_clamped(img, det.bbox);
    let mut sess = pool.acquire()?;
    let raw = run_selfie_segmenter(&mut sess, &crop)?;
    drop(sess);

    // Dilation radius: 25% of the minor box side (approved A/B parameter);
    // float→u8 saturates, bounding the square kernel at 255px.
    let rad = ((0.25 * det.bbox.width().min(det.bbox.height())).round() as u8).max(2);
    let dilated =
        imageproc::morphology::dilate(&raw, imageproc::distance_transform::Norm::LInf, rad);

    // Union with the box ellipse over the full frame.
    let mut mask = mask_from_ellipse(w, h, det.bbox, 0.05);
    let (ox, oy) = clamped_crop_origin(w, h, det.bbox);
    for (x, y, p) in dilated.enumerate_pixels() {
        if p.0[0] > 0 {
            mask.put_pixel(ox + x, oy + y, Luma([255u8]));
        }
    }
    Ok(mask)
}

/// Convex hull of the 5 keypoints (spec §4 ACTIVE polygonal blur).
pub fn hull_of_keypoints(kps: &Keypoints) -> Vec<(f32, f32)> {
    let pts: Vec<(f64, f64)> = kps.iter().map(|(x, y)| (*x as f64, *y as f64)).collect();
    crate::roi::convex_hull(&pts)
        .into_iter()
        .map(|(x, y)| (x as f32, y as f32))
        .collect()
}

/// Full-frame cautious blur (INITIAL fallback, spec §4).
pub fn blur_full_frame(img: &mut image::RgbaImage, sigma: f32) {
    let (w, h) = img.dimensions();
    blur_region_rgba(
        img,
        Rect {
            x0: 0.0,
            y0: 0.0,
            x1: w as f32,
            y1: h as f32,
        },
        sigma,
    );
}

/// Crops a rect clamped to image bounds, min size 8×8.
pub fn crop_clamped(
    img: &image::ImageBuffer<image::Rgb<u8>, Vec<u8>>,
    r: Rect,
) -> image::ImageBuffer<image::Rgb<u8>, Vec<u8>> {
    let (w, h) = img.dimensions();
    let x0 = (r.x0.floor() as i64).clamp(0, w as i64 - 1) as u32;
    let y0 = (r.y0.floor() as i64).clamp(0, h as i64 - 1) as u32;
    let x1 = (r.x1.ceil() as i64).clamp(1, w as i64) as u32;
    let y1 = (r.y1.ceil() as i64).clamp(1, h as i64) as u32;
    let (cw, ch) = ((x1 - x0).max(8).min(w - x0), (y1 - y0).max(8).min(h - y0));
    image::imageops::crop_imm(img, x0, y0, cw, ch).to_image()
}

/// Processes one RGB8 image according to the camera's FSM state.
///
/// Returns the processed RGBA image plus metadata for persistence. This is
/// CPU-bound (session acquisition + inference + blur) — callers must run it
/// inside `spawn_blocking`/a worker thread.
pub fn process_image(
    cfg: &Config,
    store: &ModelStore,
    state: CameraState,
    roi_json: Option<&str>,
    img: &image::ImageBuffer<image::Rgb<u8>, Vec<u8>>,
) -> Result<ProcessOutcome> {
    let (w, h) = img.dimensions();
    let op = AnonOp::from_cfg(cfg);

    match state {
        CameraState::Initial => {
            // Cautious anonymization of the entire image (spec §4 INITIAL).
            let mut rgba = DynamicImage::ImageRgb8(img.clone()).to_rgba8();
            op.apply_full_frame(&mut rgba);
            Ok(ProcessOutcome {
                branch: Branch::Initial,
                detections: Vec::new(),
                record_detections: false,
                fp_crops: Vec::new(),
                processed: rgba,
            })
        }
        CameraState::Learning => {
            let detections = run_detector(cfg, store, img, cfg.yolo_conf_threshold)?;

            let mut rgba = DynamicImage::ImageRgb8(img.clone()).to_rgba8();
            for det in &detections {
                // Bounding box + fixed 15% margin per side (spec §4 LEARNING).
                let margin = 0.15 * det.bbox.width().max(det.bbox.height());
                let region = det.bbox.expand_clamped(margin, w, h);
                op.apply_region(&mut rgba, region);
            }

            // Background data collection (spec §4 LEARNING): dubious
            // detections become false-positive training candidates.
            let fp_crops: Vec<(u32, u32, DynamicImage)> = detections
                .iter()
                .filter(|d| d.confidence < cfg.fp_crop_conf_max)
                .map(|d| {
                    let (cx, cy) = d.bbox.center();
                    let crop = crop_clamped(img, d.bbox);
                    (cx as u32, cy as u32, DynamicImage::ImageRgb8(crop))
                })
                .collect();

            Ok(ProcessOutcome {
                branch: Branch::Learning,
                detections,
                record_detections: true,
                fp_crops,
                processed: rgba,
            })
        }
        CameraState::Active => {
            // ROI gate: only detections whose center falls inside the ROI are
            // considered (spec §4 ACTIVE).
            let roi = roi_json.and_then(RoiPolygon::from_json);

            let detections =
                run_detector(cfg, store, img, cfg.yolo_conf_threshold_active)?;

            let mut rgba = DynamicImage::ImageRgb8(img.clone()).to_rgba8();
            let mut kept: Vec<FaceDetection> = Vec::new();

            for det in &detections {
                let (cx, cy) = det.bbox.center();
                if let Some(roi) = &roi {
                    if !roi.contains(cx as f64, cy as f64) {
                        continue; // outside ROI → no blur
                    }
                }

                // Classifier second check (spec §4 ACTIVE). Fail-safe: if the
                // classifier is unavailable or errors, blur anyway (never
                // fewer blurs than pure YOLO would produce).
                let mut confirmed = true;
                if let Some(pool) = store.classifier_pool() {
                    let mut cls = pool.acquire()?;
                    let crop = crop_clamped(img, det.bbox);
                    match run_classifier(&mut cls, &crop) {
                        Ok((_p_fp, p_face)) => confirmed = p_face >= 0.5,
                        Err(e) => {
                            tracing::warn!("classifier failed, blurring anyway: {e}");
                        }
                    }
                    drop(cls);
                }
                if !confirmed {
                    continue; // confirmed false positive → no blur
                }

                // Anonymization geometry: with `MASK_SEGMENTER=mediapipe` a per-face
                // selfie silhouette (dilated ∪ box ellipse); any segmenter
                // failure falls back to the geometric hull/ellipse so a face
                // is never left unblurred by a model hiccup.
                let sigma = sigma_for_box(det.bbox.width());
                let geometry_mask = || match &det.keypoints {
                    Some(kps) => {
                        let hull = hull_of_keypoints(kps);
                        mask_from_polygon(w, h, &hull, cfg.blur_hull_margin_pct)
                    }
                    None => mask_from_ellipse(w, h, det.bbox, 0.05),
                };
                let mask = match cfg.mask_segmenter {
                    MaskSegmenter::Off => geometry_mask(),
                    MaskSegmenter::Mediapipe => {
                        // Small faces (below `SEGMENTER_MIN_BOX` px on the
                        // frame) skip the per-face selfie inference and fall
                        // back to the geometric mask: at tiny sizes the 256²
                        // upscale degrades the silhouette and the ~26 ms ONNX
                        // run per face dominates the ACTIVE budget.
                        if !segmenter_applies(cfg, det) {
                            geometry_mask()
                        } else {
                            match segmenter_mask(w, h, store, img, det) {
                                Ok(m) => m,
                                Err(e) => {
                                    tracing::warn!(
                                        "segmenter failed for {:?} — geometric mask fallback: {e}",
                                        det.bbox
                                    );
                                    geometry_mask()
                                }
                            }
                        }
                    }
                };
                op.apply_masked(&mut rgba, &mask, sigma);
                kept.push(det.clone());
            }

            Ok(ProcessOutcome {
                branch: Branch::Active,
                detections: kept,
                // The anonymized centers feed the scheduled dynamic-ROI
                // re-extraction; the ROI gate already filtered them in.
                record_detections: true,
                fp_crops: Vec::new(),
                processed: rgba,
            })
        }
    }
}

/// A store without any real model behind it — enough for tests that only
/// exercise branches which never touch a session (INITIAL).
#[cfg(test)]
fn store_without_models() -> ModelStore {
    let pool = crate::models::SessionPool::new(
        std::path::PathBuf::from("/nonexistent/model-for-tests.onnx"),
        0,
    );
    ModelStore::new(pool, None, None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    fn solid_img(w: u32, h: u32, v: u8) -> image::ImageBuffer<image::Rgb<u8>, Vec<u8>> {
        image::ImageBuffer::from_pixel(w, h, image::Rgb([v, v, v]))
    }

    #[test]
    fn sigma_formula_and_clamp() {
        assert!((sigma_for_box(80.0) - 10.0).abs() < 1e-5);
        assert_eq!(sigma_for_box(8.0), 5.0); // clamped low
        assert_eq!(sigma_for_box(1000.0), 50.0); // clamped high
    }

    #[test]
    fn full_frame_blur_flattens_region() {
        let mut img = DynamicImage::ImageRgb8(solid_img(64, 64, 0)).to_rgba8();
        // Put a bright square in the middle.
        for y in 20..44 {
            for x in 20..44 {
                img.put_pixel(x, y, Rgba([255, 255, 255, 255]));
            }
        }
        blur_full_frame(&mut img, 6.0);
        // Center pixel must now be mixed (no longer pure white).
        let c = img.get_pixel(32, 32).0;
        assert!(c[0] < 250, "center should be blurred, got {c:?}");
        // Corner stays (mostly) black after full-frame blur.
        let corner = img.get_pixel(1, 1).0;
        assert!(
            corner[0] < 60,
            "corner should be near-black, got {corner:?}"
        );
    }

    #[test]
    fn pixelate_flattens_blocks_and_keeps_outside() {
        let mut img = DynamicImage::ImageRgb8(solid_img(64, 64, 0)).to_rgba8();
        // Noise region 20..44: every pixel distinct.
        let mut seed = 7u32;
        for y in 20..44 {
            for x in 20..44 {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                let v = (seed >> 24) as u8;
                img.put_pixel(x, y, Rgba([v, v, v, 255]));
            }
        }
        pixelate_region_rgba(
            &mut img,
            Rect {
                x0: 20.0,
                y0: 20.0,
                x1: 44.0,
                y1: 44.0,
            },
            8,
        );
        // Inside the mosaic every pixel of a full block is uniform.
        let a = img.get_pixel(22, 22).0;
        let b = img.get_pixel(25, 27).0; // same 8×8 block (x,y ∈ 20..28)
        assert_eq!(a, b, "mosaic block must be uniform");
        // Outside untouched.
        assert_eq!(img.get_pixel(5, 5).0, [0, 0, 0, 255]);
        // AnonOp dispatch reaches the mosaic.
        let mut img2 = DynamicImage::ImageRgb8(solid_img(16, 16, 0)).to_rgba8();
        for y in 4..12 {
            for x in 4..12 {
                img2.put_pixel(x, y, Rgba([99, 99, 99, 255]));
            }
        }
        AnonOp::Pixelate(4).apply_region(
            &mut img2,
            Rect {
                x0: 4.0,
                y0: 4.0,
                x1: 12.0,
                y1: 12.0,
            },
        );
        assert_eq!(img2.get_pixel(5, 5).0, [99, 99, 99, 255]);
        assert_eq!(img2.get_pixel(0, 0).0, [0, 0, 0, 255]);
    }

    #[test]
    fn masked_blur_only_affects_mask() {
        let mut img = DynamicImage::ImageRgb8(solid_img(64, 64, 0)).to_rgba8();
        for y in 20..44 {
            for x in 20..44 {
                img.put_pixel(x, y, Rgba([255, 255, 255, 255]));
            }
        }
        let mask = mask_from_rect(
            64,
            64,
            Rect {
                x0: 20.0,
                y0: 20.0,
                x1: 44.0,
                y1: 44.0,
            },
        );
        apply_masked_blur(&mut img, &mask, 6.0);
        let inside = img.get_pixel(32, 32).0;
        let outside = img.get_pixel(5, 5).0;
        assert!(inside[0] < 250, "inside mask must be blurred");
        assert_eq!(outside, [0, 0, 0, 255], "outside mask must be untouched");
    }

    #[test]
    fn masked_mosaic_flattens_blocks_keeps_outside() {
        let mut img = DynamicImage::ImageRgb8(solid_img(64, 64, 0)).to_rgba8();
        let mut seed = 7u32;
        for y in 20..44 {
            for x in 20..44 {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                let v = (seed >> 24) as u8;
                img.put_pixel(x, y, Rgba([v, v, v, 255]));
            }
        }
        let mask = mask_from_rect(
            64,
            64,
            Rect {
                x0: 20.0,
                y0: 20.0,
                x1: 44.0,
                y1: 44.0,
            },
        );
        apply_masked_pixelate(&mut img, &mask, 1.0, 8);
        // Inside the mask the mosaic flattens the noise: two mid-block pixels
        // (same average, near-equal feather weight) must be near-identical
        // (the mask feather is ~98% opaque, so a small bleed of the original
        // remains — tolerance 8 covers it).
        let a = img.get_pixel(25, 25).0;
        let b = img.get_pixel(26, 26).0;
        assert!(
            a[0].abs_diff(b[0]) <= 8 && a[0] < 200,
            "masked mosaic flattened block, got {a:?} vs {b:?}"
        );
        // Outside the mask is untouched.
        assert_eq!(img.get_pixel(5, 5).0, [0, 0, 0, 255]);
        // Inside the mask the mosaic is visible (non-black block average).
        assert_ne!(a, [0, 0, 0, 255]);
    }

    #[test]
    fn ellipse_mask_covers_center_not_corners() {
        let m = mask_from_ellipse(
            100,
            100,
            Rect {
                x0: 25.0,
                y0: 25.0,
                x1: 75.0,
                y1: 75.0,
            },
            0.0,
        );
        assert_eq!(m.get_pixel(50, 50).0[0], 255);
        assert_eq!(m.get_pixel(26, 26).0[0], 0); // corner of bbox outside ellipse
    }

    #[test]
    fn polygon_mask_covers_hull() {
        let poly = vec![(10.0, 10.0), (90.0, 10.0), (50.0, 90.0)];
        let m = mask_from_polygon(100, 100, &poly, 0.0);
        assert_eq!(m.get_pixel(50, 30).0[0], 255);
        assert_eq!(m.get_pixel(5, 90).0[0], 0);
    }

    #[test]
    fn hull_of_keypoints_is_convex() {
        let kps: Keypoints = [
            (0.0, 0.0),
            (10.0, 0.0),
            (5.0, 5.0),
            (0.0, 10.0),
            (10.0, 10.0),
        ];
        let hull = hull_of_keypoints(&kps);
        assert_eq!(hull.len(), 4); // (5,5) is interior
    }

    #[test]
    fn initial_state_blurs_everything() {
        let cfg = Config::test_default();
        let store = store_without_models();
        let img = solid_img(320, 240, 128);
        let out = process_image(&cfg, &store, CameraState::Initial, None, &img).unwrap();
        assert_eq!(out.branch, Branch::Initial);
        assert!(!out.record_detections);
        let rgba = out.processed;
        // Fully blurred: variance collapses; center == edge value.
        let c = rgba.get_pixel(160, 120).0;
        let e = rgba.get_pixel(2, 2).0;
        assert!((c[0] as i32 - e[0] as i32).abs() < 40);
    }

    #[test]
    fn crop_clamped_never_panics_at_bounds() {
        let img = solid_img(50, 50, 10);
        let far = Rect {
            x0: -100.0,
            y0: -100.0,
            x1: 1000.0,
            y1: 1000.0,
        };
        let crop = crop_clamped(&img, far);
        assert_eq!(crop.dimensions(), (50, 50));
    }

    #[test]
    fn small_faces_skip_the_segmenter() {
        use crate::models::FaceDetection;
        let cfg = Config::test_default();
        let small = FaceDetection {
            bbox: Rect {
                x0: 10.0,
                y0: 10.0,
                x1: 40.0,
                y1: 50.0,
            },
            confidence: 0.9,
            keypoints: None,
        };
        let big = FaceDetection {
            bbox: Rect {
                x0: 10.0,
                y0: 10.0,
                x1: 110.0,
                y1: 130.0,
            },
            confidence: 0.9,
            keypoints: None,
        };
        // Threshold unset (0) → segmenter always applies.
        assert!(segmenter_applies(&cfg, &small));
        assert!(segmenter_applies(&cfg, &big));
        // Threshold 64 px → the 30 px box is masked geometrically...
        let mut cfg2 = cfg.clone();
        cfg2.segmenter_min_box_px = 64.0;
        assert!(!segmenter_applies(&cfg2, &small));
        // ...while the 100 px box still runs the segmenter.
        assert!(segmenter_applies(&cfg2, &big));
    }

    #[test]
    #[ignore = "requires the selfie segmentation model + parade photo on disk"]
    fn segmenter_mask_unions_ellipse_and_silhouette() {
        use crate::models::{ModelStore, SessionPool, FaceDetection};
        use std::path::PathBuf;
        let model = std::path::PathBuf::from("models_cache/model_quantized.onnx");
        let photo = std::path::PathBuf::from("testassets/0_Parade_marchingband_1_1004.jpg");
        if !model.exists() || !photo.exists() {
            eprintln!("segmenter-mask assets missing; skipping");
            return;
        }
        let img = image::open(&photo).unwrap().to_rgb8();
        let (w, h) = img.dimensions();
        // The reference top face box (parade photo), slightly dilated.
        let det = FaceDetection {
            bbox: Rect {
                x0: 620.0,
                y0: 170.0,
                x1: 710.0,
                y1: 260.0,
            },
            confidence: 0.99,
            keypoints: None,
        };
        let pool = SessionPool::new(model, 2);
        let store = ModelStore::new(
            SessionPool::new(PathBuf::from("/nonexistent/yolo.onnx"), 1),
            None,
            Some(pool),
        );
        let mask = segmenter_mask(w, h, &store, &img, &det).unwrap();
        assert_eq!(mask.dimensions(), (w, h));
        // Silhouette covers the face center.
        assert_eq!(mask.get_pixel(665, 215).0[0], 255);
        // The union is a superset of the plain box ellipse.
        let ell = mask_from_ellipse(w, h, det.bbox, 0.05);
        let covered = mask.iter().filter(|&&v| v > 0).count();
        let ell_covered = ell.iter().filter(|&&v| v > 0).count();
        assert!(covered >= ell_covered, "{covered} < {ell_covered}");
        // Far from the face the frame stays uncovered.
        assert_eq!(mask.get_pixel(100, 5).0[0], 0);
    }
}
