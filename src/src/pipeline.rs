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
    run_classifier, run_coco_persons, run_detector, run_selfie_segmenter, FaceDetection, Keypoints,
    ModelStore, Rect,
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

    /// Applies the operation only where the region's mask is set (ACTIVE),
    /// softening the mask edge with `mask_sigma` first.
    fn apply_masked(&self, img: &mut image::RgbaImage, region: &MaskRegion, mask_sigma: f32) {
        match *self {
            AnonOp::Blur(s) => apply_masked_blur(img, region, s, mask_sigma),
            AnonOp::Pixelate(c) => apply_masked_pixelate(img, region, mask_sigma, c),
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

/// Gaussian sigma from box width: `sigma = box_width / 4` clamped to [5, 50]
/// (spec §4, used in both LEARNING and ACTIVE).
pub fn sigma_for_box(box_width: f32) -> f32 {
    (box_width / 4.0).clamp(5.0, 50.0)
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
    //
    // Both passes slide the window instead of re-reading the whole ±radius
    // neighbourhood per pixel: the accumulator is seeded once and then updated
    // by one entering/leaving sample per step, so a pixel costs O(1) instead of
    // O(2·radius+1) (up to 151 samples at sigma 50) — same window, same integer
    // average, byte-identical output.
    let rw = (x1 - x0 + 1) as usize;
    let rh = (y1 - y0 + 1) as usize;
    let mut tmp = vec![[0u32; 4]; rw * rh];

    for y in y0..=y1 {
        // Window for x = x0: [x0-radius, x0+radius] ∩ [0, w-1].
        let mut acc = [0u32; 4];
        let mut count = 0u32;
        for dx in (x0 - radius).max(0)..=(x0 + radius).min(w - 1) {
            let p = img.get_pixel(dx as u32, y as u32);
            for (a, v) in acc.iter_mut().zip(p.0) {
                *a += v as u32;
            }
            count += 1;
        }
        let row = ((y - y0) as usize) * rw;
        for x in x0..=x1 {
            tmp[row + (x - x0) as usize] = [
                acc[0] / count,
                acc[1] / count,
                acc[2] / count,
                acc[3] / count,
            ];
            // Slide to x+1: the column right of the window enters, the leftmost
            // one leaves (it is always part of the current sum, so no wrap).
            let enter = x + radius + 1;
            if enter < w {
                let p = img.get_pixel(enter as u32, y as u32);
                for (a, v) in acc.iter_mut().zip(p.0) {
                    *a += v as u32;
                }
                count += 1;
            }
            let leave = x - radius;
            if leave >= 0 {
                let p = img.get_pixel(leave as u32, y as u32);
                for (a, v) in acc.iter_mut().zip(p.0) {
                    *a -= v as u32;
                }
                count -= 1;
            }
        }
    }

    // Vertical pass, same sliding window down each column; the window is
    // clamped to the region rows ([y0, y1]) as before.
    for x in x0..=x1 {
        let mut acc = [0u32; 4];
        let mut count = 0u32;
        for dy in y0..=(y0 + radius).min(y1) {
            let p = tmp[(dy - y0) as usize * rw + (x - x0) as usize];
            for (a, v) in acc.iter_mut().zip(p) {
                *a += v;
            }
            count += 1;
        }
        for y in y0..=y1 {
            let p = img.get_pixel_mut(x as u32, y as u32);
            p.0[0] = (acc[0] / count) as u8;
            p.0[1] = (acc[1] / count) as u8;
            p.0[2] = (acc[2] / count) as u8;
            p.0[3] = (acc[3] / count) as u8;
            let enter = y + radius + 1;
            if enter <= y1 {
                let p = tmp[(enter - y0) as usize * rw + (x - x0) as usize];
                for (a, v) in acc.iter_mut().zip(p) {
                    *a += v;
                }
                count += 1;
            }
            let leave = y - radius;
            if leave >= y0 {
                let p = tmp[(leave - y0) as usize * rw + (x - x0) as usize];
                for (a, v) in acc.iter_mut().zip(p) {
                    *a -= v;
                }
                count -= 1;
            }
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

/// A mask cropped to its non-zero bounding box plus its top-left corner in
/// frame coordinates. ACTIVE per-face work then stays O(face): the `mask_from_*`
/// constructors allocate a face-sized buffer instead of a full-frame one and
/// no full-frame scan is needed to locate the mask before blurring
/// (previously one `w×h` `GrayImage` + one full-frame `mask_bbox` scan were
/// spent per face on a busy frame).
struct MaskRegion {
    mask: image::GrayImage,
    origin: (u32, u32),
}

impl MaskRegion {
    /// Frame-coordinate mask sample — 0 outside the trimmed region.
    #[cfg(test)]
    fn frame_pixel(&self, x: u32, y: u32) -> image::Luma<u8> {
        let (ox, oy) = self.origin;
        if x < ox || y < oy {
            return image::Luma([0u8]);
        }
        let (dx, dy) = (x - ox, y - oy);
        let (w, h) = self.mask.dimensions();
        if dx >= w || dy >= h {
            return image::Luma([0u8]);
        }
        *self.mask.get_pixel(dx, dy)
    }
}

/// Trims a bbox-local mask to its non-zero bounding box, translating the frame
/// origin accordingly. An all-zero mask becomes empty (0×0) so `apply_masked`
/// short-circuits.
fn mask_trim(mask: image::GrayImage, origin: (u32, u32)) -> MaskRegion {
    let (bx, by, bw, bh) = mask_bbox(&mask);
    if bw == 0 || bh == 0 {
        return MaskRegion {
            mask: image::GrayImage::new(0, 0),
            origin,
        };
    }
    let cropped = image::imageops::crop_imm(&mask, bx, by, bw, bh).to_image();
    MaskRegion {
        mask: cropped,
        origin: (origin.0 + bx, origin.1 + by),
    }
}

/// Applies an operation only where the region's mask is set (255 = op, 0 = keep).
/// The mask is blurred slightly first to avoid hard aliasing at edges.
///
/// Feather, blur and compositing all run on the mask's **bounding box** only —
/// per-face cost is O(mask bbox) instead of O(full frame), which matters when
/// an ACTIVE frame holds many faces. `region` is already bbox-local (see
/// [`MaskRegion`]/`mask_trim`), so no full-frame scan is needed to find it.
/// `op` receives the scratch copy plus its origin `(ox, oy)` in frame
/// coordinates (used by the mosaic to keep blocks aligned to the frame grid).
fn apply_masked(
    img: &mut image::RgbaImage,
    region: &MaskRegion,
    feather_sigma: f32,
    op: impl Fn(&mut image::RgbaImage, i64, i64),
) {
    let (ox, oy) = region.origin;
    let (bw, bh) = region.mask.dimensions();
    if bw == 0 || bh == 0 {
        return;
    }
    let soft: image::GrayImage = {
        let f32_mask: image::ImageBuffer<image::Luma<f32>, Vec<f32>> =
            image::ImageBuffer::from_fn(bw, bh, |x, y| {
                image::Luma([region.mask.get_pixel(x, y).0[0] as f32])
            });
        let blurred = imageproc::filter::gaussian_blur_f32(&f32_mask, feather_sigma);
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

/// Blur only where the mask is set. `sigma` drives the Gaussian strength,
/// `feather_sigma` the mask-edge softness (kept small on large close-ups so the
/// silhouette stays crisp instead of a wide translucent band).
fn apply_masked_blur(
    img: &mut image::RgbaImage,
    region: &MaskRegion,
    sigma: f32,
    feather_sigma: f32,
) {
    apply_masked(img, region, feather_sigma, |scratch, _ox, _oy| {
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
    region: &MaskRegion,
    feather_sigma: f32,
    cell: u32,
) {
    apply_masked(img, region, feather_sigma, |scratch, ox, oy| {
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

/// Builds a bbox-local mask from a convex polygon ring (hull of keypoints, spec
/// §4 ACTIVE). Self-contained convex fill (half-plane test per pixel); the ring
/// is expanded about its centroid by `margin_pct` first. Only the ring's
/// bounding box is allocated and filled — no full-frame buffer or scan.
fn mask_from_polygon(w: u32, h: u32, polygon: &[(f32, f32)], margin_pct: f32) -> MaskRegion {
    if polygon.len() < 3 {
        return MaskRegion {
            mask: image::GrayImage::new(0, 0),
            origin: (0, 0),
        };
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
        return MaskRegion {
            mask: image::GrayImage::new(0, 0),
            origin: (0, 0),
        };
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
    if x1 <= x0 || y1 <= y0 {
        return MaskRegion {
            mask: image::GrayImage::new(0, 0),
            origin: (0, 0),
        };
    }

    let mut mask = image::GrayImage::from_pixel(x1 - x0, y1 - y0, image::Luma([0u8]));
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
                mask.put_pixel(x - x0, y - y0, image::Luma([255u8]));
            }
        }
    }
    mask_trim(mask, (x0, y0))
}

/// Builds a bbox-local mask from an axis-aligned rect (used by tests / full
/// cover).
#[allow(dead_code)]
fn mask_from_rect(w: u32, h: u32, r: Rect) -> MaskRegion {
    let x0 = r.x0.round().max(0.0) as i64;
    let y0 = r.y0.round().max(0.0) as i64;
    let x1 = (r.x1.round() as i64).min(w as i64 - 1);
    let y1 = (r.y1.round() as i64).min(h as i64 - 1);
    if !(x1 > x0 && y1 > y0) {
        return MaskRegion {
            mask: image::GrayImage::new(0, 0),
            origin: (0, 0),
        };
    }
    let mut mask = image::GrayImage::from_pixel(
        (x1 - x0 + 1) as u32,
        (y1 - y0 + 1) as u32,
        image::Luma([0u8]),
    );
    for y in y0..=y1 {
        for x in x0..=x1 {
            mask.put_pixel((x - x0) as u32, (y - y0) as u32, image::Luma([255u8]));
        }
    }
    mask_trim(mask, (x0 as u32, y0 as u32))
}

/// Frame-coordinate extent of an ellipse inscribed in `r` with `margin_pct`
/// added to both radii, clamped to the frame (inclusive range, or `None` when
/// a degenerate box leaves nothing to fill).
fn ellipse_extent(w: u32, h: u32, r: Rect, margin_pct: f32) -> Option<(u32, u32, u32, u32)> {
    let cx = (r.x0 + r.x1) / 2.0;
    let cy = (r.y0 + r.y1) / 2.0;
    let rx = ((r.x1 - r.x0) / 2.0 * (1.0 + margin_pct)).max(1.0);
    let ry = ((r.y1 - r.y0) / 2.0 * (1.0 + margin_pct)).max(1.0);
    let x0 = (cx - rx).floor().max(0.0) as u32;
    let x1 = (cx + rx).ceil().min(w as f32 - 1.0) as u32;
    let y0 = (cy - ry).floor().max(0.0) as u32;
    let y1 = (cy + ry).ceil().min(h as f32 - 1.0) as u32;
    if x1 < x0 || y1 < y0 {
        None
    } else {
        Some((x0, x1, y0, y1))
    }
}

/// Untrimmed bbox-local ellipse mask (extent-sized) plus its frame origin.
/// Note the local buffer can be a few rows/columns larger than the ellipse's
/// actual non-zero bbox (extreme rows/columns only touch the ellipse edge or
/// miss it entirely) — `mask_trim` tightens it at the call sites.
fn ellipse_local(
    w: u32,
    h: u32,
    r: Rect,
    margin_pct: f32,
) -> Option<(image::GrayImage, (u32, u32))> {
    let (x0, x1, y0, y1) = ellipse_extent(w, h, r, margin_pct)?;
    let cx = (r.x0 + r.x1) / 2.0;
    let cy = (r.y0 + r.y1) / 2.0;
    let rx = ((r.x1 - r.x0) / 2.0 * (1.0 + margin_pct)).max(1.0);
    let ry = ((r.y1 - r.y0) / 2.0 * (1.0 + margin_pct)).max(1.0);
    let mut mask = image::GrayImage::from_pixel(x1 - x0 + 1, y1 - y0 + 1, image::Luma([0u8]));
    for y in y0..=y1 {
        for x in x0..=x1 {
            let dx = (x as f32 - cx) / rx;
            let dy = (y as f32 - cy) / ry;
            if dx * dx + dy * dy <= 1.0 {
                mask.put_pixel(x - x0, y - y0, image::Luma([255u8]));
            }
        }
    }
    Some((mask, (x0, y0)))
}

/// Builds a mask from an ellipse inscribed in the bounding box (spec §4
/// fallback when keypoints are unavailable).
fn mask_from_ellipse(w: u32, h: u32, r: Rect, margin_pct: f32) -> MaskRegion {
    match ellipse_local(w, h, r, margin_pct) {
        Some((mask, origin)) => mask_trim(mask, origin),
        None => MaskRegion {
            mask: image::GrayImage::new(0, 0),
            origin: (0, 0),
        },
    }
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
/// model is the same [`crop_clamped`] the classifier uses. The union is
/// assembled over one small bbox-local buffer — no full-frame allocation.
fn segmenter_mask(
    w: u32,
    h: u32,
    store: &ModelStore,
    img: &image::ImageBuffer<image::Rgb<u8>, Vec<u8>>,
    det: &FaceDetection,
    ellipse_margin: f32,
) -> Result<MaskRegion> {
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

    // Union extent (frame coords) of the dilated silhouette region and the box
    // ellipse, recomputed within this per-face bbox.
    let (sx, sy) = clamped_crop_origin(w, h, det.bbox);
    let (sw, sh) = dilated.dimensions();
    let ell_opt = ellipse_local(w, h, det.bbox, ellipse_margin);
    let (mut ux0, mut uy0) = (sx, sy);
    let (mut ux1, mut uy1) = (sx + sw.saturating_sub(1), sy + sh.saturating_sub(1));
    if let Some((ell, (ex0, ey0))) = &ell_opt {
        let (ew, eh) = ell.dimensions();
        ux0 = ux0.min(*ex0);
        uy0 = uy0.min(*ey0);
        ux1 = ux1.max(ex0 + ew.saturating_sub(1));
        uy1 = uy1.max(ey0 + eh.saturating_sub(1));
    }
    let mut mask = image::GrayImage::from_pixel(ux1 - ux0 + 1, uy1 - uy0 + 1, image::Luma([0u8]));
    if let Some((ell, (ex0, ey0))) = &ell_opt {
        let (ew, eh) = ell.dimensions();
        for y in 0..eh {
            for x in 0..ew {
                if ell.get_pixel(x, y).0[0] > 0 {
                    mask.put_pixel(ex0 + x - ux0, ey0 + y - uy0, Luma([255u8]));
                }
            }
        }
    }
    for y in 0..sh {
        for x in 0..sw {
            if dilated.get_pixel(x, y).0[0] > 0 {
                mask.put_pixel(x + sx - ux0, y + sy - uy0, Luma([255u8]));
            }
        }
    }
    Ok(mask_trim(mask, (ux0, uy0)))
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

/// Converts an RGB8 frame to RGBA (alpha 255) in one pass. The previous
/// `DynamicImage::ImageRgb8(img.clone()).to_rgba8()` duplicated the RGB buffer
/// (3 bytes/px) *and* allocated the RGBA one (4 bytes/px); this allocates only
/// the RGBA buffer and fills it directly, removing a full-frame copy from the
/// per-image hot path.
fn rgb_to_rgba(img: &image::ImageBuffer<image::Rgb<u8>, Vec<u8>>) -> image::RgbaImage {
    let (w, h) = img.dimensions();
    let rgb = img.as_raw();
    let mut rgba = Vec::with_capacity((w as usize * h as usize) * 4);
    for px in rgb.as_chunks::<3>().0 {
        rgba.extend_from_slice(&[px[0], px[1], px[2], 255]);
    }
    image::RgbaImage::from_raw(w, h, rgba).expect("RGB→RGBA buffer is exactly w*h*4 bytes")
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
            let mut rgba = rgb_to_rgba(img);
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

            let mut rgba = rgb_to_rgba(img);
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

            let mut rgba = rgb_to_rgba(img);
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
                if cfg.classifier_enforce {
                    if let Some(pool) = store.classifier_pool() {
                        let mut cls = pool.acquire()?;
                        let crop = crop_clamped(img, det.bbox);
                        match run_classifier(&mut cls, &crop) {
                            Ok((_p_fp, p_face)) => {
                                confirmed = p_face >= cfg.classifier_confirm_threshold
                            }
                            Err(e) => {
                                tracing::warn!("classifier failed, blurring anyway: {e}");
                            }
                        }
                        drop(cls);
                    }
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
                    None => mask_from_ellipse(w, h, det.bbox, cfg.blur_ellipse_margin),
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
                            match segmenter_mask(w, h, store, img, det, cfg.blur_ellipse_margin) {
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
                op.apply_masked(&mut rgba, &mask, cfg.mask_feather(sigma));
                kept.push(det.clone());
            }

            // Head fallback: when the face detector found no faces (typical
            // for pure-profile or heavily occluded shots), fall back to a
            // COCO person detector and blur the upper fraction of each person
            // box (the head region). This is a privacy-preserving last resort
            // — blurring a head silhouette is safer than leaving a face
            // unblurred.
            if kept.is_empty() {
                if let Some(pool) = store.coco_pool() {
                    let mut coco = pool.acquire()?;
                    let persons = run_coco_persons(
                        &mut coco,
                        img,
                        cfg.yolo_conf_threshold_active,
                        cfg.yolo_nms_iou,
                        cfg.yolo_input_size,
                    )
                    .unwrap_or_else(|e| {
                        tracing::warn!("head-fallback person detection failed: {e}");
                        Vec::new()
                    });
                    drop(coco);
                    for p in &persons {
                        let (cx, cy) = p.bbox.center();
                        if let Some(roi) = &roi {
                            if !roi.contains(cx as f64, cy as f64) {
                                continue; // person outside ROI → skip
                            }
                        }
                        // Upper `head_fallback_fraction` of the person box ≈
                        // head region (conservative: includes shoulders).
                        let head_h =
                            (p.bbox.height() * cfg.head_fallback_fraction).clamp(8.0, f32::MAX);
                        let head_rect = Rect {
                            x0: p.bbox.x0,
                            y0: p.bbox.y0,
                            x1: p.bbox.x1,
                            y1: (p.bbox.y0 + head_h).min(h as f32),
                        };
                        let mask = mask_from_ellipse(w, h, head_rect, cfg.blur_ellipse_margin);
                        op.apply_masked(&mut rgba, &mask, cfg.mask_feather(sigma_for_box(head_rect.width())));
                        kept.push(p.clone());
                    }
                    if !persons.is_empty() {
                        tracing::debug!(
                            "head-fallback blurred {} person head(s) (face detector found none)",
                            persons.len()
                        );
                    }
                }
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

    /// Pre-optimization `box_blur_region` (re-reads the whole ±radius window
    /// per pixel). Kept as the reference for the byte-identical check below.
    fn box_blur_reference(img: &mut image::RgbaImage, region: Rect, radius: i64) {
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

    /// The sliding-window blur must be byte-identical to the previous
    /// implementation, including on partial windows (region touching the image
    /// border, radius larger than the region) and on every channel.
    #[test]
    fn sliding_window_blur_is_byte_identical_to_reference() {
        let mut img = DynamicImage::ImageRgb8(solid_img(160, 120, 30)).to_rgba8();
        for (i, p) in img.pixels_mut().enumerate() {
            let v = ((i * 37) % 251) as u8;
            *p = Rgba([v, v.wrapping_mul(3), (255 - v), 255]);
        }
        let regions = [
            Rect { x0: 10.0, y0: 8.0, x1: 150.0, y1: 112.0 }, // internal
            Rect { x0: -5.0, y0: -5.0, x1: 200.0, y1: 200.0 }, // clamped to frame
            Rect { x0: 40.0, y0: 40.0, x1: 41.0, y1: 41.0 },  // 8×8 minimum region
            Rect { x0: 0.0, y0: 0.0, x1: 159.0, y1: 119.0 },  // whole frame
        ];
        for region in regions {
            for radius in [2i64, 3, 8, 25, 75] {
                let mut fast = img.clone();
                let mut reference = img.clone();
                box_blur_region(&mut fast, region, radius);
                box_blur_reference(&mut reference, region, radius);
                assert_eq!(
                    fast.as_raw(),
                    reference.as_raw(),
                    "region {region:?} radius {radius}"
                );
            }
        }
    }

    /// Cost measurement, not part of the normal suite:
    /// `cargo test --release -- --ignored blur_cost`.
    #[test]
    #[ignore]
    fn blur_cost_sliding_window_vs_reference() {
        let mut img = DynamicImage::ImageRgb8(solid_img(1920, 1080, 30)).to_rgba8();
        let region = Rect { x0: 800.0, y0: 400.0, x1: 1100.0, y1: 700.0 };
        let t = std::time::Instant::now();
        for _ in 0..20 {
            blur_region_rgba(&mut img, region, 50.0); // sigma 50 ⇒ radius 75, 2 passate
        }
        let sliding = t.elapsed();
        let t = std::time::Instant::now();
        for _ in 0..20 {
            box_blur_reference(&mut img, region, 75);
            box_blur_reference(&mut img, region, 75);
        }
        let reference = t.elapsed();
        println!("20 volto 300x300 @ sigma 50 — sliding: {sliding:?}, riferimento: {reference:?}");
    }

    /// Cost measurement for the bbox-local ACTIVE mask path, not part of the
    /// normal suite: `cargo test --release -- --ignored active_bbox_mask_cost`.
    /// Absolute per-face cost of the *new* path (face-sized buffers, no
    /// full-frame scan); compare the same run on a previous build to quantify
    /// the bbox-local gain (the old path additionally allocated a `w×h` mask
    /// and scanned the full frame per face).
    #[test]
    #[ignore]
    fn active_bbox_mask_cost() {
        let mut img = DynamicImage::ImageRgb8(solid_img(1920, 1080, 30)).to_rgba8();
        let op = AnonOp::Blur(50.0);
        let t = std::time::Instant::now();
        for i in 0..50u32 {
            let cx = 140.0 + (i % 8) as f32 * 220.0;
            let cy = 160.0 + (i / 8) as f32 * 190.0;
            let face = Rect {
                x0: cx,
                y0: cy,
                x1: cx + 160.0,
                y1: cy + 240.0,
            };
            let region = mask_from_ellipse(1920, 1080, face, 0.08);
            op.apply_masked(&mut img, &region, 24.0);
        }
        println!(
            "50 masked 160×240 faces @1080p, sigma 50: {:?}",
            t.elapsed()
        );
    }

    #[test]
    fn sigma_formula_and_clamp() {
        assert!((sigma_for_box(80.0) - 20.0).abs() < 1e-5);
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
        apply_masked_blur(&mut img, &mask, 6.0, 6.0);
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
        assert_eq!(m.frame_pixel(50, 50).0[0], 255);
        assert_eq!(m.frame_pixel(26, 26).0[0], 0); // corner of bbox outside ellipse
    }

    #[test]
    fn polygon_mask_covers_hull() {
        let poly = vec![(10.0, 10.0), (90.0, 10.0), (50.0, 90.0)];
        let m = mask_from_polygon(100, 100, &poly, 0.0);
        assert_eq!(m.frame_pixel(50, 30).0[0], 255);
        assert_eq!(m.frame_pixel(5, 90).0[0], 0);
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
        let mask = segmenter_mask(w, h, &store, &img, &det, 0.05).unwrap();
        // Bbox-local: strictly smaller than the frame, always inside it.
        assert!(mask.mask.width() <= w && mask.mask.height() <= h);
        // Silhouette covers the face center.
        assert_eq!(mask.frame_pixel(665, 215).0[0], 255);
        // The union is a superset of the plain box ellipse.
        let ell = mask_from_ellipse(w, h, det.bbox, 0.05);
        let covered = mask.mask.iter().filter(|&&v| v > 0).count();
        let ell_covered = ell.mask.iter().filter(|&&v| v > 0).count();
        assert!(covered >= ell_covered, "{covered} < {ell_covered}");
        // Far from the face the frame stays uncovered.
        assert_eq!(mask.frame_pixel(100, 5).0[0], 0);
    }
}
