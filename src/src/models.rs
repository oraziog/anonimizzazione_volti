//! Inference layer (spec §8 `models.rs`): YOLOv8-Face detection with adaptive
//! output parsing (keypoints or no keypoints), binary classifier with ImageNet
//! normalization, and session lifecycle management.
//!
//! # Session ownership
//! `ort` 2.0 `Session::run` requires `&mut self` — ONNX Runtime sessions are
//! not safe for concurrent inference (see the `ort` docs). Sharing one session
//! behind `ArcSwap` (as older ort 1.x allowed) would be unsound here, so
//! concurrent workers get *exclusive* sessions from a [`SessionPool`] that
//! recycles idle sessions (one per busy worker ≈ the image semaphore).
//!
//! The YOLO model file never changes during a process lifetime, so its pool is
//! stable. The classifier is hot-swapped by the nightly retraining (§6): the
//! store keeps an `ArcSwap<Option<Arc<SessionPool>>>` and swapping replaces
//! the *whole pool* atomically — in-flight workers finish on the old sessions,
//! new workers acquire from the new pool.
//!
//! Tensor I/O uses ort's raw `(shape, data)` path (no `ndarray` feature
//! enabled) so the spec-pinned `ndarray = "0.15"` stays the only ndarray in
//! the dependency tree.

use std::path::PathBuf;
use std::sync::{Arc, Condvar, Mutex};

use anyhow::{anyhow, Result};

use crate::config::{Config, DetectorMode};

// ─── Session pooling ─────────────────────────────────────────────────────────

struct PoolState {
    idle: Vec<ort::session::Session>,
    /// Total sessions created (idle + checked out). Bounded by `capacity`.
    live: usize,
}

/// A pool of exclusive ONNX sessions over one model file.
///
/// `acquire()` blocks until a session is free, creating one lazily up to
/// `capacity`. Sessions must be used from blocking contexts (never while
/// holding an async executor thread hostage) and are automatically returned
/// to the pool on drop.
pub struct SessionPool {
    path: PathBuf,
    capacity: usize,
    state: Mutex<PoolState>,
    available: Condvar,
}

/// A checked-out session; returning it to the pool is automatic on drop.
pub struct PooledSession<'a> {
    session: Option<ort::session::Session>,
    pool: &'a SessionPool,
}

impl SessionPool {
    /// `capacity` bounds the number of live ONNX sessions (create-on-demand).
    pub fn new(path: PathBuf, capacity: usize) -> Self {
        Self {
            path,
            capacity: capacity.max(1),
            state: Mutex::new(PoolState {
                idle: Vec::new(),
                live: 0,
            }),
            available: Condvar::new(),
        }
    }

    /// The model file this pool loads sessions from.
    #[allow(dead_code)] // read by the retraining feature (training.rs)
    pub fn path(&self) -> &PathBuf {
        &self.path
    }

    /// Blocks until an exclusive session is available.
    ///
    /// Call only from blocking contexts (spawn_blocking / std threads).
    /// `live` is incremented *before* the (slow) load and only rolled back on
    /// failure, so concurrent callers can never overshoot `capacity`.
    pub fn acquire(&self) -> Result<PooledSession<'_>> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| anyhow!("session pool poisoned"))?;
        loop {
            if let Some(session) = state.idle.pop() {
                return Ok(PooledSession {
                    session: Some(session),
                    pool: self,
                });
            }
            if state.live < self.capacity {
                state.live += 1;
                drop(state); // never hold the lock during a file load
                let result = crate::model_loader::load_session(&self.path);
                let mut state = self
                    .state
                    .lock()
                    .map_err(|_| anyhow!("session pool poisoned"))?;
                match result {
                    Ok(session) => {
                        return Ok(PooledSession {
                            session: Some(session),
                            pool: self,
                        });
                    }
                    Err(e) => {
                        state.live -= 1;
                        self.available.notify_one();
                        return Err(e);
                    }
                }
            }
            state = self
                .available
                .wait(state)
                .map_err(|_| anyhow!("session pool poisoned"))?;
        }
    }

    fn release(&self, session: ort::session::Session) {
        if let Ok(mut state) = self.state.lock() {
            state.idle.push(session);
            self.available.notify_one();
        }
    }
}

impl Drop for PooledSession<'_> {
    fn drop(&mut self) {
        if let Some(session) = self.session.take() {
            self.pool.release(session);
        }
    }
}

impl std::ops::Deref for PooledSession<'_> {
    type Target = ort::session::Session;
    fn deref(&self) -> &Self::Target {
        self.session.as_ref().expect("session present while pooled")
    }
}

impl std::ops::DerefMut for PooledSession<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.session.as_mut().expect("session present while pooled")
    }
}

// ─── Session store (hot-swappable classifier, spec §3/§6) ───────────────────

/// Holds the live model pools. Cloneable (all interior `Arc`s).
#[derive(Clone)]
pub struct ModelStore {
    pub yolo: Arc<SessionPool>,
    classifier: Arc<arc_swap::ArcSwap<Option<Arc<SessionPool>>>>,
    /// Optional ACTIVE face-mask segmenter (MediaPipe Selfie, spec §4 —
    /// `MASK_SEGMENTER=mediapipe`). `None` keeps geometric masks.
    segmenter: Option<Arc<SessionPool>>,
}

impl ModelStore {
    pub fn new(
        yolo: SessionPool,
        classifier: Option<SessionPool>,
        segmenter: Option<SessionPool>,
    ) -> Self {
        Self {
            yolo: Arc::new(yolo),
            classifier: Arc::new(arc_swap::ArcSwap::from_pointee(classifier.map(Arc::new))),
            segmenter: segmenter.map(Arc::new),
        }
    }

    /// Current classifier pool, if one is configured/swapped in.
    pub fn classifier_pool(&self) -> Option<Arc<SessionPool>> {
        self.classifier.load().as_ref().clone()
    }

    /// The ACTIVE face-mask segmenter pool, if `MASK_SEGMENTER=mediapipe`.
    pub fn segmenter_pool(&self) -> Option<Arc<SessionPool>> {
        self.segmenter.clone()
    }

    /// Atomically swaps in a new classifier pool (pre-validated by the
    /// caller — spec §6 validation happens before the swap).
    #[allow(dead_code)] // called by the retraining feature (training.rs)
    pub fn swap_classifier_pool(&self, pool: SessionPool) {
        self.classifier.store(Arc::new(Some(Arc::new(pool))));
    }
}

// ─── Geometry helpers shared by pipeline & tests ─────────────────────────────

/// Axis-aligned rectangle in original-image pixel coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rect {
    pub x0: f32,
    pub y0: f32,
    pub x1: f32,
    pub y1: f32,
}

impl Rect {
    pub fn width(&self) -> f32 {
        (self.x1 - self.x0).max(0.0)
    }
    pub fn height(&self) -> f32 {
        (self.y1 - self.y0).max(0.0)
    }
    pub fn center(&self) -> (f32, f32) {
        ((self.x0 + self.x1) / 2.0, (self.y0 + self.y1) / 2.0)
    }

    /// Expands by `margin` on every side and clamps to image bounds.
    pub fn expand_clamped(&self, margin: f32, w: u32, h: u32) -> Rect {
        Rect {
            x0: (self.x0 - margin).max(0.0),
            y0: (self.y0 - margin).max(0.0),
            x1: (self.x1 + margin).min(w as f32),
            y1: (self.y1 + margin).min(h as f32),
        }
    }
}

/// Face keypoints in original-image coordinates (2 eyes, nose, 2 mouth corners).
pub type Keypoints = [(f32, f32); 5];

/// A YOLO detection mapped back to original-image coordinates.
#[derive(Debug, Clone)]
pub struct FaceDetection {
    pub bbox: Rect,
    pub confidence: f32,
    /// Present only if the loaded model exports landmarks (spec §10.2).
    pub keypoints: Option<Keypoints>,
}

// ─── YOLO preprocessing (letterbox) ──────────────────────────────────────────

/// Letterbox parameters for a given source size.
#[derive(Debug, Clone, Copy)]
pub struct LetterboxParams {
    pub scale: f32,
    pub pad_x: f32,
    pub pad_y: f32,
}

/// Computes letterbox parameters mapping source → `input`×`input`.
pub fn letterbox_params(src_w: u32, src_h: u32, input: u32) -> LetterboxParams {
    let scale = (input as f32 / src_w as f32).min(input as f32 / src_h as f32);
    let new_w = src_w as f32 * scale;
    let new_h = src_h as f32 * scale;
    LetterboxParams {
        scale,
        pad_x: (input as f32 - new_w) / 2.0,
        pad_y: (input as f32 - new_h) / 2.0,
    }
}

/// Builds the CHW f32 RGB input tensor `[1,3,input,input]` from an RGB8 image.
pub fn build_yolo_input(
    img: &image::ImageBuffer<image::Rgb<u8>, Vec<u8>>,
    input: u32,
) -> Vec<f32> {
    let (w, h) = (img.width(), img.height());
    let lb = letterbox_params(w, h, input);
    let n_in = input as usize;
    let mut out = vec![0f32; 3 * n_in * n_in];
    let dst_w = (w as f32 * lb.scale).round().max(1.0) as u32;
    let dst_h = (h as f32 * lb.scale).round().max(1.0) as u32;

    let resized = image::imageops::resize(img, dst_w, dst_h, image::imageops::FilterType::Triangle);

    let pad_x = lb.pad_x.round() as i64;
    let pad_y = lb.pad_y.round() as i64;
    for (x, y, p) in resized.enumerate_pixels() {
        let dx = x as i64 + pad_x;
        let dy = y as i64 + pad_y;
        if dx < 0 || dy < 0 || dx >= input as i64 || dy >= input as i64 {
            continue;
        }
        let (dx, dy) = (dx as usize, dy as usize);
        let [r, g, b] = p.0;
        out[dx + dy * n_in] = r as f32 / 255.0;
        out[n_in * n_in + dx + dy * n_in] = g as f32 / 255.0;
        out[2 * n_in * n_in + dx + dy * n_in] = b as f32 / 255.0;
    }
    out
}

/// Maps a point from letterboxed `input`×`input` coordinates back to source pixels.
pub fn letterbox_to_src(x: f32, y: f32, lb: &LetterboxParams) -> (f32, f32) {
    ((x - lb.pad_x) / lb.scale, (y - lb.pad_y) / lb.scale)
}

// ─── YOLO output parsing (adaptive, spec §10.2) ──────────────────────────────

/// Greedy NMS on IoU. Returns detections sorted by descending confidence.
pub fn nms(mut dets: Vec<FaceDetection>, iou_threshold: f32) -> Vec<FaceDetection> {
    dets.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut kept: Vec<FaceDetection> = Vec::new();
    'outer: for d in dets {
        for k in &kept {
            if iou(&d.bbox, &k.bbox) > iou_threshold {
                continue 'outer;
            }
        }
        kept.push(d);
    }
    kept
}

pub fn iou(a: &Rect, b: &Rect) -> f32 {
    let x0 = a.x0.max(b.x0);
    let y0 = a.y0.max(b.y0);
    let x1 = a.x1.min(b.x1);
    let y1 = a.y1.min(b.y1);
    let inter = (x1 - x0).max(0.0) * (y1 - y0).max(0.0);
    let area_a = a.width() * a.height();
    let area_b = b.width() * b.height();
    let union = area_a + area_b - inter;
    if union <= 0.0 {
        0.0
    } else {
        inter / union
    }
}

// ─── RetinaFace (mobilenetv1_0.25) preprocessing & output parsing ────────────

/// BGR mean used by the reference RetinaFace inference code (opencv order).
pub const RETINA_MEAN_BGR: [f32; 3] = [104.0, 117.0, 123.0];
/// FPN anchor scales per stride (mobile0.25 config, spec WIDER: [[16,32],...]).
const RETINA_MIN_SIZES: [[u32; 2]; 3] = [[16, 32], [64, 128], [256, 512]];
const RETINA_STEPS: [u32; 3] = [8, 16, 32];
const RETINA_VAR: [f32; 2] = [0.1, 0.2];

/// Generates the normalized priors `[cx, cy, s_kx, s_ky]` (16800 at 640×640)
/// for a given model input side, replicating `PriorBox.generate_anchors`.
pub fn generate_retinaface_priors(input: u32) -> Vec<[f32; 4]> {
    let perf_cells = RETINA_STEPS
        .iter()
        .map(|s| ((input / s) as usize) * ((input / s) as usize) * 2)
        .sum();
    let mut out = Vec::with_capacity(perf_cells);
    for (k, step) in RETINA_STEPS.iter().enumerate() {
        let cells = input / step;
        for i in 0..cells {
            for j in 0..cells {
                let cx = (j as f32 + 0.5) * *step as f32 / input as f32;
                let cy = (i as f32 + 0.5) * *step as f32 / input as f32;
                for ms in RETINA_MIN_SIZES[k] {
                    let size = ms as f32 / input as f32;
                    out.push([cx, cy, size, size]);
                }
            }
        }
    }
    out
}

/// Builds the CHW f32 input `[1,3,input,input]` for the ONNX RetinaFace
/// models: stretch-resize to square, BGR channel order, `mean` subtracted.
pub fn build_retinaface_input(
    img: &image::ImageBuffer<image::Rgb<u8>, Vec<u8>>,
    input: u32,
) -> Vec<f32> {
    let n = input as usize;
    let resized = image::imageops::resize(
        img,
        input,
        input,
        image::imageops::FilterType::Triangle,
    );
    let mut out = vec![0f32; 3 * n * n];
    for (x, y, p) in resized.enumerate_pixels() {
        let [r, g, b] = p.0;
        let idx = y as usize * n + x as usize;
        out[idx] = b as f32 - RETINA_MEAN_BGR[0];
        out[n * n + idx] = g as f32 - RETINA_MEAN_BGR[1];
        out[2 * n * n + idx] = r as f32 - RETINA_MEAN_BGR[2];
    }
    out
}

/// Decodes the three RetinaFace outputs (`loc`, `conf`, `landmarks`) using the
/// shared 640 priors and maps boxes back to source pixels. Confidence is the
/// already-softmaxed face score (`conf[:, 1]`); boxes are NMS-filtered.
pub fn decode_retinaface(
    loc: &[f32],
    conf: &[f32],
    landmarks: &[f32],
    priors: &[[f32; 4]],
    conf_threshold: f32,
    src_w: u32,
    src_h: u32,
) -> Vec<FaceDetection> {
    let n = priors.len();
    let (w, h) = (src_w as f32, src_h as f32);
    let mut dets: Vec<FaceDetection> = Vec::new();
    if loc.len() < n * 4 || conf.len() < n * 2 || landmarks.len() < n * 10 {
        return dets;
    }
    for (p, prior) in priors.iter().enumerate() {
        let score = conf[p * 2 + 1];
        if score < conf_threshold {
            continue;
        }
        let (pcx, pcy, pw, ph) = (prior[0], prior[1], prior[2], prior[3]);
        let cx = pcx + loc[p * 4] * RETINA_VAR[0] * pw;
        let cy = pcy + loc[p * 4 + 1] * RETINA_VAR[0] * ph;
        let bw = pw * (loc[p * 4 + 2] * RETINA_VAR[1]).exp();
        let bh = ph * (loc[p * 4 + 3] * RETINA_VAR[1]).exp();
        let x1 = ((cx - bw / 2.0) * w).clamp(0.0, w);
        let y1 = ((cy - bh / 2.0) * h).clamp(0.0, h);
        let x2 = ((cx + bw / 2.0) * w).clamp(0.0, w);
        let y2 = ((cy + bh / 2.0) * h).clamp(0.0, h);
        let bbox = Rect {
            x0: x1.min(x2),
            y0: y1.min(y2),
            x1: x1.max(x2),
            y1: y1.max(y2),
        };
        if bbox.width() < 1.0 || bbox.height() < 1.0 {
            continue;
        }
        let mut kps = [(0.0f32, 0.0f32); 5];
        for (k, slot) in kps.iter_mut().enumerate() {
            let lx = pcx + landmarks[p * 10 + k * 2] * RETINA_VAR[0] * pw;
            let ly = pcy + landmarks[p * 10 + k * 2 + 1] * RETINA_VAR[0] * ph;
            *slot = ((lx * w).clamp(0.0, w), (ly * h).clamp(0.0, h));
        }
        dets.push(FaceDetection {
            bbox,
            confidence: score,
            keypoints: Some(kps),
        });
    }
    dets
}

/// Runs the RetinaFace ONNX model on one image and returns NMS-filtered
/// detections in original-image coordinates (`session` exclusively owned).
pub fn run_retinaface(
    session: &mut ort::session::Session,
    img: &image::ImageBuffer<image::Rgb<u8>, Vec<u8>>,
    conf_threshold: f32,
    nms_iou: f32,
    input: u32,
) -> Result<Vec<FaceDetection>> {
    let (w, h) = img.dimensions();
    let input_data = build_retinaface_input(img, input);
    let tensor = ort::value::Tensor::from_array((
        vec![1i64, 3, input as i64, input as i64],
        input_data,
    ))
    .map_err(|e| anyhow!("build RetinaFace input tensor: {e}"))?;
    let outputs = session
        .run(ort::inputs![tensor])
        .map_err(|e| anyhow!("RetinaFace inference failed: {e}"))?;
    if outputs.len() < 3 {
        return Err(anyhow!("RetinaFace model returned fewer than 3 outputs"));
    }
    let mut tensors = Vec::new();
    for i in 0..3 {
        let (shape, data) = outputs[i]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow!("RetinaFace output {i} not f32 tensor: {e}"))?;
        let dims: Vec<i64> = shape.iter().copied().collect();
        tensors.push((dims, data));
    }
    let (loc, conf, lm) = (&tensors[0].1, &tensors[1].1, &tensors[2].1);
    let n_priors = tensors[1].0.get(1).copied().unwrap_or(0);
    if tensors[0].0.get(1).copied().unwrap_or(0) != n_priors
        || tensors[2].0.get(1).copied().unwrap_or(0) != n_priors
    {
        return Err(anyhow!("RetinaFace outputs disagree on anchor count"));
    }
    if n_priors == 0 {
        return Err(anyhow!("RetinaFace outputs carry no anchors"));
    }
    let priors = generate_retinaface_priors(input);
    let mut dets = decode_retinaface(loc, conf, lm, &priors, conf_threshold, w, h);
    dets = nms(dets, nms_iou);
    Ok(dets)
}

// ─── Inference instrumentation (operator /operator/gpu) ─────────────────────

use std::sync::atomic::{AtomicU64, Ordering};

/// Measured per-stage inference statistics. Counters are incremented by the
/// public inference entry points below (`run_detector`, `run_classifier`,
/// `run_selfie_segmenter`) and served by `GET /operator/gpu`; they are
/// process-lifetime accumulators over every image ever processed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InferenceStage {
    Detector,
    Classifier,
    Segmenter,
}

const INFER_STAGE_NAMES: [(&str, InferenceStage); 3] = [
    ("detector", InferenceStage::Detector),
    ("classifier", InferenceStage::Classifier),
    ("segmenter", InferenceStage::Segmenter),
];

static INFER_COUNT: [AtomicU64; 3] = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];
static INFER_NANOS: [AtomicU64; 3] = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];

fn infer_index(stage: InferenceStage) -> usize {
    match stage {
        InferenceStage::Detector => 0,
        InferenceStage::Classifier => 1,
        InferenceStage::Segmenter => 2,
    }
}

/// Records one inference run (`elapsed` includes the ONNX `Session::run`).
pub fn record_inference(stage: InferenceStage, elapsed: std::time::Duration) {
    let idx = infer_index(stage);
    INFER_COUNT[idx].fetch_add(1, Ordering::Relaxed);
    INFER_NANOS[idx].fetch_add(
        u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX),
        Ordering::Relaxed,
    );
}

/// Snapshot of the measured inference statistics, stable ordering
/// `[detector, classifier, segmenter]`, `(stage, count, avg_ms)`.
pub fn inference_stats() -> Vec<(&'static str, u64, f64)> {
    INFER_STAGE_NAMES
        .iter()
        .map(|(name, stage)| {
            let idx = infer_index(*stage);
            let count = INFER_COUNT[idx].load(Ordering::Relaxed);
            let nanos = INFER_NANOS[idx].load(Ordering::Relaxed);
            let avg_ms = if count == 0 {
                0.0
            } else {
                nanos as f64 / count as f64 / 1_000_000.0
            };
            (*name, count, avg_ms)
        })
        .collect()
}

/// Dispatches face detection to the configured detector (YOLO or RetinaFace).
pub fn run_detector(
    cfg: &Config,
    store: &ModelStore,
    img: &image::ImageBuffer<image::Rgb<u8>, Vec<u8>>,
    conf_threshold: f32,
) -> Result<Vec<FaceDetection>> {
    let started = std::time::Instant::now();
    let mut session = store.yolo.acquire()?;
    let dets = match cfg.detector_mode {
        DetectorMode::Yolo => run_yolo(
            &mut session,
            img,
            conf_threshold,
            cfg.yolo_nms_iou,
            cfg.yolo_input_size,
        ),
        DetectorMode::RetinaFace => run_retinaface(
            &mut session,
            img,
            conf_threshold,
            cfg.yolo_nms_iou,
            cfg.retinaface_input_size,
        ),
    }?;
    drop(session);
    record_inference(InferenceStage::Detector, started.elapsed());
    Ok(dets)
}

// ─── Classifier preprocessing ────────────────────────────────────────────────

/// ImageNet normalization constants.
pub const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
pub const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// Builds the classifier input `[1,3,224,224]` (spec §4) from an RGB8 crop.
pub fn build_classifier_input(img: &image::ImageBuffer<image::Rgb<u8>, Vec<u8>>) -> Vec<f32> {
    const SIZE: usize = 224;
    let resized = image::imageops::resize(
        img,
        SIZE as u32,
        SIZE as u32,
        image::imageops::FilterType::Triangle,
    );
    let mut out = vec![0f32; 3 * SIZE * SIZE];
    for (x, y, p) in resized.enumerate_pixels() {
        let [r, g, b] = p.0;
        let (x, y) = (x as usize, y as usize);
        out[x + y * SIZE] = (r as f32 / 255.0 - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
        out[SIZE * SIZE + x + y * SIZE] = (g as f32 / 255.0 - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
        out[2 * SIZE * SIZE + x + y * SIZE] =
            (b as f32 / 255.0 - IMAGENET_MEAN[2]) / IMAGENET_STD[2];
    }
    out
}

/// Softmax over 2 logits.
pub fn softmax2(a: f32, b: f32) -> (f32, f32) {
    let m = a.max(b);
    let ea = (a - m).exp();
    let eb = (b - m).exp();
    let s = ea + eb;
    (ea / s, eb / s)
}

// ─── Inference entry points (called under spawn_blocking) ────────────────────

/// Raw decoded detection in letterboxed (640×640) model pixels.
struct RawDet {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    conf: f32,
    keypoints: Option<Keypoints>,
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Decodes one multi-scale feature map in the native YOLOv8-face pose format:
/// `[1, 80, H, W]` where channel layout is
///   `0..64`  = DFL box distribution (4 sides × 16 bins),
///   `64`     = class objectness logit,
///   `65..80` = 5 keypoints × (x, y, visibility logits).
/// Boxes/keypoints are expressed in model pixels (stride-scaled grid), ready
/// for the inverse-letterbox mapping.
fn decode_dfl_face_grid(
    data: &[f32],
    grid_w: u32,
    grid_h: u32,
    conf_threshold: f32,
    input_px: f32,
) -> Vec<RawDet> {
    const CH: usize = 80;
    let hw = (grid_w * grid_h) as usize;
    if grid_w == 0 || grid_h == 0 || data.len() < CH * hw {
        return Vec::new();
    }
    let stride = input_px / grid_w as f32;
    let cell =
        |ch: usize, x: u32, y: u32| data[ch * hw + (y as usize) * grid_w as usize + x as usize];

    let mut out = Vec::new();
    for y in 0..grid_h {
        for x in 0..grid_w {
            let conf = sigmoid(cell(64, x, y));
            if conf < conf_threshold {
                continue;
            }

            // Decode DFL side distances: per side i, softmax over 16 bins.
            let mut side = [0f32; 4];
            for (i, slot) in side.iter_mut().enumerate() {
                let mut vals = [0f32; 16];
                let mut max = f32::NEG_INFINITY;
                for (b, v) in vals.iter_mut().enumerate() {
                    let value = cell(i * 16 + b, x, y);
                    *v = value;
                    max = max.max(value);
                }
                let mut denom = 0.0f32;
                let mut acc = 0.0f32;
                for (b, v) in vals.iter().enumerate() {
                    let e = (v - max).exp();
                    denom += e;
                    acc += b as f32 * e;
                }
                *slot = if denom > 0.0 { acc / denom } else { 0.0 };
            }

            let gx = x as f32 + 0.5;
            let gy = y as f32 + 0.5;
            let x1 = (gx - side[0]) * stride;
            let y1 = (gy - side[1]) * stride;
            let x2 = (gx + side[2]) * stride;
            let y2 = (gy + side[3]) * stride;

            // Keypoints: (raw * 2 + grid) * stride; visibility raw logits.
            let mut kps = [(0f32, 0f32); 5];
            let mut all_visible = true;
            for (k, slot) in kps.iter_mut().enumerate() {
                let kx = cell(65 + k * 3, x, y);
                let ky = cell(65 + k * 3 + 1, x, y);
                let vis = cell(65 + k * 3 + 2, x, y);
                if vis <= 0.0 {
                    all_visible = false;
                }
                *slot = (
                    (kx * 2.0 + x as f32) * stride,
                    (ky * 2.0 + y as f32) * stride,
                );
            }

            out.push(RawDet {
                x1,
                y1,
                x2,
                y2,
                conf,
                keypoints: if all_visible { Some(kps) } else { None },
            });
        }
    }
    out
}

/// Runs YOLOv8-Face on one image and returns NMS-filtered detections in
/// original-image coordinates. `session` must be exclusively owned (see
/// [`SessionPool::acquire`]).
///
/// Supports the default public export (three `[1,80,H,W]` pose-head maps, see
/// [`decode_dfl_face_grid`]) and, as a fallback, classic single-output models
/// with the `[1, C, anchors]` transposed layout (see [`parse_yolo_output`]).
pub fn run_yolo(
    session: &mut ort::session::Session,
    img: &image::ImageBuffer<image::Rgb<u8>, Vec<u8>>,
    conf_threshold: f32,
    nms_iou: f32,
    input: u32,
) -> Result<Vec<FaceDetection>> {
    let (w, h) = (img.width(), img.height());
    let input_data = build_yolo_input(img, input);
    let tensor = ort::value::Tensor::from_array((
        vec![1i64, 3, input as i64, input as i64],
        input_data,
    ))
    .map_err(|e| anyhow!("build YOLO input tensor: {e}"))?;
    let outputs = session
        .run(ort::inputs![tensor])
        .map_err(|e| anyhow!("YOLO inference failed: {e}"))?;

    if outputs.len() == 0 {
        return Err(anyhow!("YOLO model returned no outputs"));
    }
    let lb = letterbox_params(w, h, input);
    let mut raws: Vec<RawDet> = Vec::new();

    for value in outputs.values() {
        let (shape, data) = value
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow!("YOLO output not f32 tensor: {e}"))?;
        let dims: Vec<i64> = shape.iter().copied().collect();
        match dims.as_slice() {
            // Pose-head multi-scale map [1, 80, H, W] (default model).
            [1, 80, gh, gw] if (*gh > 0 && *gw > 0) => {
                raws.extend(decode_dfl_face_grid(
                    data,
                    *gw as u32,
                    *gh as u32,
                    conf_threshold,
                    input as f32,
                ));
            }
            // Classic transposed layout [1, C, anchors].
            [1, _c, _a] => {
                raws.extend(parse_yolo_output_raw(&dims, data, conf_threshold, &lb));
            }
            _ => {
                tracing::warn!(
                    "YOLO model returned an unrecognized output shape {dims:?}; ignoring it"
                );
            }
        }
    }

    let mut dets: Vec<FaceDetection> = Vec::new();
    for raw in raws {
        let (x0, y0) = letterbox_to_src(raw.x1, raw.y1, &lb);
        let (x1, y1) = letterbox_to_src(raw.x2, raw.y2, &lb);
        let bbox = Rect {
            x0: x0.min(x1).clamp(0.0, w as f32),
            y0: y0.min(y1).clamp(0.0, h as f32),
            x1: x0.max(x1).clamp(0.0, w as f32),
            y1: y0.max(y1).clamp(0.0, h as f32),
        };
        if bbox.width() < 1.0 || bbox.height() < 1.0 {
            continue;
        }
        let keypoints = raw.keypoints.map(|kps| {
            let mut m = [(0.0f32, 0.0f32); 5];
            for (i, (kx, ky)) in kps.iter().enumerate() {
                m[i] = letterbox_to_src(*kx, *ky, &lb);
            }
            m
        });
        dets.push(FaceDetection {
            bbox,
            confidence: raw.conf,
            keypoints,
        });
    }
    dets = nms(dets, nms_iou);
    Ok(dets)
}

/// Raw candidate extraction for classic `[1, C, anchors]` outputs.
fn parse_yolo_output_raw(
    shape: &[i64],
    data: &[f32],
    conf_threshold: f32,
    src_params: &LetterboxParams,
) -> Vec<RawDet> {
    let mut out = Vec::new();
    if shape.len() != 3 || shape[0] != 1 {
        return out;
    }
    let channels = shape[1] as usize;
    let anchors = shape[2] as usize;
    if channels < 5 || anchors == 0 || data.len() < channels * anchors {
        return out;
    }
    let n_kp = if channels >= 20 { 5 } else { 0 };
    let n_cls = channels - 4 - n_kp * 3;
    if n_cls == 0 {
        return out;
    }
    for a in 0..anchors {
        let col = |ch: usize| data[ch * anchors + a];
        let (conf, _) = if n_cls > 1 {
            let mut best = 0.0f32;
            for c in 0..n_cls {
                best = best.max(col(4 + n_kp * 3 + c));
            }
            (best, 0)
        } else {
            (col(4 + n_kp * 3), 0)
        };
        if conf < conf_threshold {
            continue;
        }
        let cx = col(0);
        let cy = col(1);
        let cw = col(2);
        let chh = col(3);
        if cw <= 0.0 || chh <= 0.0 {
            continue;
        }
        let mut kps = [(0.0f32, 0.0f32); 5];
        let mut all_visible = true;
        for (k, slot) in kps.iter_mut().enumerate() {
            let vis = col(4 + k * 3 + 2);
            if vis <= 0.0 {
                all_visible = false;
            }
            *slot = letterbox_to_src(col(4 + k * 3), col(4 + k * 3 + 1), src_params);
        }
        let (x0, y0) = letterbox_to_src(cx - cw / 2.0, cy - chh / 2.0, src_params);
        let (x1, y1) = letterbox_to_src(cx + cw / 2.0, cy + chh / 2.0, src_params);
        out.push(RawDet {
            x1: x0,
            y1: y0,
            x2: x1,
            y2: y1,
            conf,
            keypoints: if n_kp == 5 && all_visible {
                Some(kps)
            } else {
                None
            },
        });
    }
    out
}

/// Runs the binary classifier on a crop. Returns `(p_falso_positivo,
/// p_volto_reale)` after softmax over the two output logits (spec §4 layout
/// `[Falso_Positivo, Volto_Reale]`).
pub fn run_classifier(
    session: &mut ort::session::Session,
    crop: &image::ImageBuffer<image::Rgb<u8>, Vec<u8>>,
) -> Result<(f32, f32)> {
    let started = std::time::Instant::now();
    let result = (|| -> Result<(f32, f32)> {
        let input = build_classifier_input(crop);
        let tensor = ort::value::Tensor::from_array((vec![1i64, 3, 224, 224], input))
            .map_err(|e| anyhow!("build classifier input tensor: {e}"))?;
        let outputs = session
            .run(ort::inputs![tensor])
            .map_err(|e| anyhow!("classifier inference failed: {e}"))?;
        if outputs.len() == 0 {
            return Err(anyhow!("classifier returned no outputs"));
        }
        let value = &outputs[0];
        let (shape, data) = value
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow!("classifier output not f32 tensor: {e}"))?;
        if shape.iter().sum::<i64>() < 2 || data.len() < 2 {
            return Err(anyhow!("classifier output shape too small"));
        }
        Ok(softmax2(data[0], data[1]))
    })();
    if result.is_ok() {
        record_inference(InferenceStage::Classifier, started.elapsed());
    }
    result
}

// ─── Selfie segmentation (ACTIVE mask, MASK_SEGMENTER=mediapipe) ─────────────

/// Stretch-resize side of the MediaPipe Selfie Segmentation input (spec A/B:
/// 256×256, RGB, /255).
pub const SELFIE_SEG_INPUT: u32 = 256;

/// Builds the CHW f32 RGB input `[1,3,256,256]` from a face crop, normalized
/// to [0,1] (mediapipe convention; RGB order, no mean subtraction).
pub fn build_selfie_segmenter_input(
    img: &image::ImageBuffer<image::Rgb<u8>, Vec<u8>>,
) -> Vec<f32> {
    let n = SELFIE_SEG_INPUT as usize;
    let resized = image::imageops::resize(
        img,
        SELFIE_SEG_INPUT,
        SELFIE_SEG_INPUT,
        image::imageops::FilterType::Triangle,
    );
    let mut out = vec![0f32; 3 * n * n];
    for (x, y, p) in resized.enumerate_pixels() {
        let [r, g, b] = p.0;
        let idx = y as usize * n + x as usize;
        out[idx] = r as f32 / 255.0;
        out[n * n + idx] = g as f32 / 255.0;
        out[2 * n * n + idx] = b as f32 / 255.0;
    }
    out
}

/// Runs the MediaPipe Selfie Segmentation model on a face crop and returns the
/// binary silhouette resized back to crop dimensions (255 inside the person).
///
/// Pipeline mirrors the approved A/B (`scripts/mask_preview.py`):
/// input 256×256 → `alphas` sigmoid → threshold > 0.5 → resize back (linear)
/// → re-threshold > 127.
pub fn run_selfie_segmenter(
    session: &mut ort::session::Session,
    img: &image::ImageBuffer<image::Rgb<u8>, Vec<u8>>,
) -> Result<image::GrayImage> {
    let started = std::time::Instant::now();
    let result = (|| -> Result<image::GrayImage> {
        let (cw, ch) = img.dimensions();
        let input = build_selfie_segmenter_input(img);
        let tensor = ort::value::Tensor::from_array((
            vec![1i64, 3, SELFIE_SEG_INPUT as i64, SELFIE_SEG_INPUT as i64],
            input,
        ))
        .map_err(|e| anyhow!("build Selfie input tensor: {e}"))?;
        let outputs = session
            .run(ort::inputs![tensor])
            .map_err(|e| anyhow!("Selfie segmentation failed: {e}"))?;

        // The exported model emits a single `[1,1,256,256]` `alphas` map. Accept
        // any f32 output with ≥ 256² samples (some exports drop the channel dim),
        // copying the slice so the tensor borrow does not outlive the outputs.
        let mut alphas: Option<Vec<f32>> = None;
        for value in outputs.values() {
            let (shape, data) = value
                .try_extract_tensor::<f32>()
                .map_err(|e| anyhow!("Selfie output not f32 tensor: {e}"))?;
            let _ = shape; // dims not needed beyond the sample-count guard below
            if data.len() >= (SELFIE_SEG_INPUT as usize).pow(2) {
                alphas = Some(data[..(SELFIE_SEG_INPUT as usize).pow(2)].to_vec());
                break;
            }
        }
        let Some(alphas) = alphas else {
            return Err(anyhow!("Selfie segmentation returned no 256×256 output"));
        };
        let n = SELFIE_SEG_INPUT as usize;
        let mut mask256 = image::GrayImage::from_pixel(
            SELFIE_SEG_INPUT,
            SELFIE_SEG_INPUT,
            image::Luma([0u8]),
        );
        for y in 0..n {
            for x in 0..n {
                if alphas[y * n + x] > 0.5 {
                    mask256.put_pixel(x as u32, y as u32, image::Luma([255u8]));
                }
            }
        }
        let resized =
            image::imageops::resize(&mask256, cw, ch, image::imageops::FilterType::Triangle);
        let mut mask = image::GrayImage::from_pixel(cw, ch, image::Luma([0u8]));
        for (x, y, p) in resized.enumerate_pixels() {
            if p.0[0] > 127 {
                mask.put_pixel(x, y, image::Luma([255u8]));
            }
        }
        Ok(mask)
    })();
    if result.is_ok() {
        record_inference(InferenceStage::Segmenter, started.elapsed());
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inference_stats_record_and_snapshot() {
        // The counters are process-lifetime, so assert on deltas around two
        // recorded detector runs (25 ms + 75 ms → avg 50 ms) plus a stable
        // snapshot ordering.
        let before: Vec<_> = inference_stats();
        record_inference(
            InferenceStage::Detector,
            std::time::Duration::from_millis(25),
        );
        record_inference(
            InferenceStage::Detector,
            std::time::Duration::from_millis(75),
        );
        let after: Vec<_> = inference_stats();
        let by_name = |v: &Vec<(&'static str, u64, f64)>, n: &str| {
            *v.iter()
                .find(|(name, _, _)| *name == n)
                .expect("stage present")
        };
        let (name, count_b, avg_b) = by_name(&before, "detector");
        let (_, count_a, avg_a) = by_name(&after, "detector");
        assert_eq!(name, "detector");
        assert_eq!(count_a - count_b, 2);
        assert!(((avg_a * count_a as f64 - avg_b * count_b as f64) / 2.0 - 50.0).abs() < 1e-6);
        let names: Vec<&'static str> = after.iter().map(|(n, _, _)| *n).collect();
        assert_eq!(names, vec!["detector", "classifier", "segmenter"]);
    }

    fn rect(x0: f32, y0: f32, x1: f32, y1: f32) -> Rect {
        Rect { x0, y0, x1, y1 }
    }

    #[test]
    fn letterbox_roundtrip() {
        let lb = letterbox_params(1920, 1080, 640);
        assert!((lb.scale - 640.0 / 1920.0).abs() < 1e-6);
        let (sx, sy) = letterbox_to_src(320.0, 320.0, &lb);
        assert!((sx - 960.0).abs() < 1.0);
        assert!((sy - 540.0).abs() < 1.0);
    }

    #[test]
    fn decode_dfl_face_grid_geometry() {
        // One cell at grid (0,0), stride 8 (grid 80 → but decode is grid-size
        // agnostic; use 1×1 with stride 640 so math stays exact).
        let grid_w = 1u32;
        let grid_h = 1u32;
        let mut data = vec![-100.0f32; 80 * (grid_w * grid_h) as usize];
        let hw = (grid_w * grid_h) as usize;
        // Objectness logit strong.
        data[64 * hw] = 100.0;
        // One-hot DFL bins: side0→bin0, side1→bin0, side2→bin2, side3→bin1.
        data[0] = 8.0; // dist bin 0 of side 0 (channel 0)
        data[16 * hw] = 8.0; // dist bin 0 of side 1
        data[32 * hw + 2] = 8.0; // dist bin 2 of side 2
        data[48 * hw + 1] = 8.0; // dist bin 1 of side 3
                                 // Keypoints visible (raw kx=ky=0.5, vis=8).
        for k in 0..5usize {
            data[(65 + k * 3) * hw] = 0.5;
            data[(65 + k * 3 + 1) * hw] = 0.5;
            data[(65 + k * 3 + 2) * hw] = 8.0;
        }

        let raws = decode_dfl_face_grid(&data, grid_w, grid_h, 0.2, 640.0);
        assert_eq!(raws.len(), 1);
        let r = &raws[0];
        // stride = 640/1 → x1=(0.5-0)*640=320, x2=(0.5+2)*640=1600, etc.
        assert!((r.x1 - 320.0).abs() < 1e-3, "x1={}", r.x1);
        assert!((r.y1 - 320.0).abs() < 1e-3);
        assert!((r.x2 - 1600.0).abs() < 1e-3, "x2={}", r.x2);
        assert!((r.y2 - 960.0).abs() < 1e-3);
        assert!((r.conf - 1.0).abs() < 1e-4);
        let kps = r.keypoints.expect("visible keypoints");
        assert!((kps[0].0 - (0.5 * 2.0 + 0.0) * 640.0).abs() < 1e-3);
        assert!((kps[4].1 - 640.0).abs() < 1e-3);
    }

    #[test]
    fn decode_dfl_low_conf_filtered() {
        let grid_w = 1;
        let grid_h = 1;
        let mut data = vec![-100.0f32; 80];
        // objectness low → filtered.
        data[64] = -20.0;
        assert!(decode_dfl_face_grid(&data, grid_w, grid_h, 0.2, 640.0).is_empty());
    }

    #[test]
    fn parse_classic_transposed_output() {
        // Classic single-output model: [1, 20, 1] with layout
        // [cx,cy,w,h | 5 kp (x,y,vis) | 1 class score].
        let shape = [1i64, 20, 1];
        let mut data = vec![0f32; 20];
        data[0] = 320.0;
        data[1] = 320.0;
        data[2] = 100.0;
        data[3] = 100.0;
        for k in 0..5usize {
            data[4 + k * 3] = 300.0 + k as f32;
            data[4 + k * 3 + 1] = 310.0;
            data[4 + k * 3 + 2] = 0.9;
        }
        data[19] = 0.7; // class score (single class)

        let lb = letterbox_params(640, 640, 640);
        let raws = parse_yolo_output_raw(&shape, &data, 0.20, &lb);
        assert_eq!(raws.len(), 1);
        assert!((raws[0].conf - 0.7).abs() < 1e-6);
        let kps = raws[0].keypoints.unwrap();
        assert!((kps[0].0 - 300.0).abs() < 1e-4);
        assert!((raws[0].x1 - 270.0).abs() < 1e-4);
        assert!((raws[0].y2 - 370.0).abs() < 1e-4);
    }

    #[test]
    fn parse_raw_rejects_bad_shapes() {
        let lb = letterbox_params(640, 640, 640);
        assert!(parse_yolo_output_raw(&[1, 5], &[], 0.2, &lb).is_empty());
        assert!(parse_yolo_output_raw(&[2, 5, 10], &[], 0.2, &lb).is_empty());
    }

    #[test]
    fn nms_and_iou() {
        assert!((iou(&rect(0.0, 0.0, 10.0, 10.0), &rect(0.0, 0.0, 10.0, 10.0)) - 1.0).abs() < 1e-6);
        assert!(iou(&rect(0.0, 0.0, 10.0, 10.0), &rect(20.0, 20.0, 30.0, 30.0)) == 0.0);
        let dets = vec![
            FaceDetection {
                bbox: rect(0.0, 0.0, 10.0, 10.0),
                confidence: 0.9,
                keypoints: None,
            },
            FaceDetection {
                bbox: rect(1.0, 1.0, 11.0, 11.0),
                confidence: 0.8,
                keypoints: None,
            },
            FaceDetection {
                bbox: rect(100.0, 100.0, 120.0, 120.0),
                confidence: 0.7,
                keypoints: None,
            },
        ];
        let kept = nms(dets, 0.45);
        assert_eq!(kept.len(), 2);
        assert_eq!(kept[0].confidence, 0.9);
    }

    #[test]
    fn classifier_input_normalization() {
        let img: image::ImageBuffer<image::Rgb<u8>, Vec<u8>> =
            image::ImageBuffer::from_pixel(64, 64, image::Rgb([255, 128, 0]));
        let input = build_classifier_input(&img);
        assert_eq!(input.len(), 3 * 224 * 224);
        // First pixel red channel: (1.0 - 0.485) / 0.229
        let expect = (1.0 - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
        assert!((input[0] - expect).abs() < 1e-6);
    }

    #[test]
    fn selfie_segmenter_input_layout() {
        let img: image::ImageBuffer<image::Rgb<u8>, Vec<u8>> =
            image::ImageBuffer::from_pixel(4, 5, image::Rgb([10, 20, 30]));
        let input = build_selfie_segmenter_input(&img);
        assert_eq!(input.len(), 3 * 256 * 256);
        // Top-left pixel: RGB order, /255 normalization.
        assert!((input[0] - 10.0 / 255.0).abs() < 1e-6);
        assert!((input[256 * 256] - 20.0 / 255.0).abs() < 1e-6);
        assert!((input[2 * 256 * 256] - 30.0 / 255.0).abs() < 1e-6);
    }

    #[test]
    #[ignore = "requires the selfie segmentation model on disk"]
    fn real_selfie_segmenter_smoke() {
        let model = std::path::PathBuf::from("models_cache/model_quantized.onnx");
        let photo = std::path::PathBuf::from("testassets/0_Parade_marchingband_1_1004.jpg");
        if !model.exists() || !photo.exists() {
            eprintln!("selfie smoke assets missing; skipping");
            return;
        }
        let mut session = crate::model_loader::load_session(&model).unwrap();
        let img = image::open(&photo).unwrap().to_rgb8();
        // Parade photo: the reference top face box, crop of a real person.
        let crop = image::imageops::crop_imm(&img, 644, 193, 50, 50).to_image();
        let mask = run_selfie_segmenter(&mut session, &crop).unwrap();
        assert_eq!(mask.dimensions(), (50, 50));
        let covered = mask.iter().filter(|&&v| v > 0).count();
        eprintln!("selfie smoke: {covered}/{} masked pixels", 50 * 50);
        // A person's face crop must yield a non-empty silhouette.
        assert!(covered > 0, "expected a visible selfie silhouette in the crop");
    }

    #[test]
    fn softmax2_sums_to_one() {
        let (a, b) = softmax2(2.0, 1.0);
        assert!((a + b - 1.0).abs() < 1e-6);
        assert!(a > b);
    }

    #[test]
    fn rect_expand_clamps() {
        let r = rect(10.0, 10.0, 50.0, 50.0).expand_clamped(20.0, 55, 55);
        assert_eq!(r, rect(0.0, 0.0, 55.0, 55.0));
    }

    #[test]
    fn retinaface_priors_structure() {
        let priors = generate_retinaface_priors(640);
        assert_eq!(priors.len(), 16_800);
        // Level 0 (stride 8): cell (0,0), first size 16 → 640×640 normalized.
        let p0 = priors[0];
        assert!((p0[0] - (0.5 * 8.0 / 640.0)).abs() < 1e-6);
        assert!((p0[1] - (0.5 * 8.0 / 640.0)).abs() < 1e-6);
        assert!((p0[2] - 16.0 / 640.0).abs() < 1e-6);
        assert!((p0[3] - 16.0 / 640.0).abs() < 1e-6);
        // Second prior of the same cell → size 32.
        let p1 = priors[1];
        assert!((p1[2] - 32.0 / 640.0).abs() < 1e-6);
        // Last prior of level 0 must be cell (79,79) with size 32.
        let last_l0 = priors[2 * 80 * 80 - 1];
        assert!((last_l0[0] - (79.5 * 8.0 / 640.0)).abs() < 1e-4);
        assert!((last_l0[2] - 32.0 / 640.0).abs() < 1e-6);
        // Level boundaries: priors are grouped per level.
        let first_l1 = priors[2 * 80 * 80];
        assert!((first_l1[0] - (0.5 * 16.0 / 640.0)).abs() < 1e-6);
        assert!((first_l1[2] - 64.0 / 640.0).abs() < 1e-6);
    }

    #[test]
    fn retinaface_priors_scales_with_input() {
        // 512×512 → cells 64/32/16, two anchors per cell.
        let priors = generate_retinaface_priors(512);
        assert_eq!(priors.len(), 2 * (64 * 64 + 32 * 32 + 16 * 16));
        let p0 = priors[0];
        assert!((p0[0] - (0.5 * 8.0 / 512.0)).abs() < 1e-6);
        assert!((p0[2] - 16.0 / 512.0).abs() < 1e-6);
        let p_last = priors[2 * 64 * 64 - 1];
        assert!((p_last[0] - (63.5 * 8.0 / 512.0)).abs() < 1e-4);
    }

    #[test]
    fn retinaface_input_layout_bgr_mean() {
        let img: image::ImageBuffer<image::Rgb<u8>, Vec<u8>> =
            image::ImageBuffer::from_pixel(4, 4, image::Rgb([10, 20, 30]));
        let input = build_retinaface_input(&img, 640);
        assert_eq!(input.len(), 3 * 640 * 640);
        // BGR order with mean subtraction at the top-left pixel.
        assert!((input[0] - (30.0 - 104.0)).abs() < 1e-5);
        assert!((input[640 * 640] - (20.0 - 117.0)).abs() < 1e-5);
        assert!((input[2 * 640 * 640] - (10.0 - 123.0)).abs() < 1e-5);
    }

    #[test]
    fn retinaface_decode_synthetic() {
        let priors = generate_retinaface_priors(640);
        let n = priors.len();
        let loc = vec![0f32; n * 4];
        let mut conf = vec![0f32; n * 2];
        let lm = vec![0f32; n * 10];
        for p in 0..n {
            conf[p * 2 + 1] = 0.9;
        }
        // Zero loc → boxes equal to the prior box centered at the prior center.
        let dets = decode_retinaface(&loc, &conf, &lm, &priors, 0.05, 640, 640);
        assert_eq!(dets.len(), n);
        let d = &dets[0]; // prior: center (4,4), size 16×16
        assert!((d.bbox.x0 - 0.0).abs() < 1.0); // (4-8) → clamped at 0
        assert!((d.bbox.y1 - 12.0).abs() < 1.0); // (4+8)
        assert!((d.confidence - 0.9).abs() < 1e-6);
        assert!(d.keypoints.is_some());
    }

    #[test]
    fn retinaface_decode_filters_low_conf() {
        let priors = generate_retinaface_priors(640);
        let n = priors.len();
        let loc = vec![0f32; n * 4];
        let mut conf = vec![0f32; n * 2];
        let lm = vec![0f32; n * 10];
        conf.fill(0.01);
        let dets = decode_retinaface(&loc, &conf, &lm, &priors, 0.05, 640, 640);
        assert!(dets.is_empty());
    }

    #[test]
    fn retinaface_decode_mismatched_buffers() {
        let priors = generate_retinaface_priors(640);
        let dets = decode_retinaface(&[0.0], &[0.0, 0.0], &[0.0], &priors, 0.05, 640, 640);
        assert!(dets.is_empty());
    }

    #[test]
    #[ignore = "requires the model + sample photo under .test-assets"]
    fn real_face_model_smoke() {
        let model = std::path::PathBuf::from(".test-assets/yolov8n-face.onnx");
        let photo = std::path::PathBuf::from(".test-assets/face-sample.jpg");
        if !model.exists() || !photo.exists() {
            eprintln!("smoke assets missing; skipping");
            return;
        }
        let mut session = crate::model_loader::load_session(&model).unwrap();
        let img = image::open(&photo).unwrap().to_rgb8();
        let dets = run_yolo(&mut session, &img, 0.2, 0.45, 640).unwrap();
        eprintln!("smoke detections: {}", dets.len());
        for d in &dets {
            eprintln!(
                "  conf={:.3} bbox=({:.0},{:.0},{:.0},{:.0}) keypoints={}",
                d.confidence,
                d.bbox.x0,
                d.bbox.y0,
                d.bbox.x1,
                d.bbox.y1,
                d.keypoints.is_some()
            );
        }
        assert!(
            !dets.is_empty(),
            "expected at least one face in the model's demo photo"
        );
        assert!(
            dets.iter().any(|d| d.keypoints.is_some()),
            "pose-head export should provide 5 facial keypoints"
        );
    }

    #[test]
    #[ignore = "requires the model + sample photo on disk"]
    fn real_retinaface_model_smoke() {
        let model = std::path::PathBuf::from("models_cache/retinaface_mv1_0.25.onnx");
        let photo = std::path::PathBuf::from("testassets/0_Parade_marchingband_1_1004.jpg");
        if !model.exists() || !photo.exists() {
            eprintln!("retinaface smoke assets missing; skipping");
            return;
        }
        let mut session = crate::model_loader::load_session(&model).unwrap();
        let img = image::open(&photo).unwrap().to_rgb8();
        let dets = run_retinaface(&mut session, &img, 0.05, 0.45, 640).unwrap();
        eprintln!("retinaface smoke detections: {}", dets.len());
        for d in dets.iter().take(5) {
            eprintln!(
                "  conf={:.4} bbox=({:.2},{:.2},{:.2},{:.2})",
                d.confidence, d.bbox.x0, d.bbox.y0, d.bbox.x1, d.bbox.y1
            );
        }
        assert!(
            !dets.is_empty(),
            "expected at least one face in the parade photo"
        );
        // Reference (OpenCV-CUBIC) top detection at conf>0.05 iou<=0.45:
        // bbox (644.97,193.63)-(689.16,234.76) @ 0.9866
        let top = &dets[0];
        assert!(
            (top.confidence - 0.9866).abs() < 0.03,
            "top conf {}",
            top.confidence
        );
        assert!(
            (top.bbox.x0 - 644.97).abs() < 12.0 && (top.bbox.x1 - 689.16).abs() < 12.0,
            "top box x={},{}",
            top.bbox.x0,
            top.bbox.x1
        );
    }

    #[test]
    fn pool_fails_cleanly_without_file() {
        // A pool pointing at a missing file: acquire() must fail cleanly and
        // roll back its live counter (so a later retry still works, and a
        // waiter would be woken).
        let pool = SessionPool::new(PathBuf::from("/nonexistent/model.onnx"), 2);
        assert!(pool.acquire().is_err());
        assert!(pool.acquire().is_err());
        let state = pool.state.lock().unwrap();
        assert_eq!(state.live, 0);
        assert!(state.idle.is_empty());
    }

    /// Micro-benchmark of the ACTIVE per-face segmenter chain, to quantify
    /// what a hypothetical batched inference (one ONNX run for N faces) would
    /// actually save. Prints a per-call breakdown; run with `-- --ignored
    /// --nocapture`. Requires the model + parade photo on disk.
    #[test]
    #[ignore = "requires selfie model + parade photo on disk"]
    fn segmenter_cost_breakdown() {
        use std::time::Instant;
        let model = PathBuf::from("models_cache/model_quantized.onnx");
        let photo = PathBuf::from("testassets/0_Parade_marchingband_1_1004.jpg");
        if !model.exists() || !photo.exists() {
            eprintln!("assets missing; skipping");
            return;
        }
        let mut sess = crate::model_loader::load_session(&model).unwrap();
        let img = image::open(&photo).unwrap().to_rgb8();
        let (w, h) = img.dimensions();

        // Realistic YOLO face crop on this photo (matches detector smoke).
        let crop = crate::pipeline::crop_clamped(
            &img,
            Rect {
                x0: 644.0,
                y0: 193.0,
                x1: 689.0,
                y1: 235.0,
            },
        );
        let big = crate::pipeline::crop_clamped(
            &img,
            Rect {
                x0: 620.0,
                y0: 170.0,
                x1: 710.0,
                y1: 260.0,
            },
        );

        // Print session input shape → does the export support batch > 1?
        eprintln!("frame {}x{}  crop {}x{}  big {}x{}", w, h, crop.width(), crop.height(), big.width(), big.height());

        const RUNS: usize = 50;

        // (1) Resize + CHW fill alone.
        let c = crop.clone();
        let t0 = Instant::now();
        for _ in 0..RUNS {
            std::hint::black_box(build_selfie_segmenter_input(&c));
        }
        let inp_avg = t0.elapsed().as_secs_f64() * 1000.0 / RUNS as f64;

        // (2) Full per-face run (input + tensor + ONNX + resize-back + thr).
        let t0 = Instant::now();
        for _ in 0..RUNS {
            std::hint::black_box(run_selfie_segmenter(&mut sess, &crop).unwrap());
        }
        let full_avg = t0.elapsed().as_secs_f64() * 1000.0 / RUNS as f64;
        let t0 = Instant::now();
        for _ in 0..RUNS {
            std::hint::black_box(run_selfie_segmenter(&mut sess, &big).unwrap());
        }
        let full_avg_big = t0.elapsed().as_secs_f64() * 1000.0 / RUNS as f64;

        // (3) Dilate (the `rad` from a 45px-wide box → dilation 11px LInf).
        let mask = run_selfie_segmenter(&mut sess, &crop).unwrap();
        let t0 = Instant::now();
        for _ in 0..RUNS {
            std::hint::black_box(imageproc::morphology::dilate(
                &mask,
                imageproc::distance_transform::Norm::LInf,
                11u8,
            ));
        }
        let dilate_avg = t0.elapsed().as_secs_f64() * 1000.0 / RUNS as f64;

        eprintln!(
            "per-face: build_input={inp_avg:.1}ms  full_crop={full_avg:.1}ms  full_big={full_avg_big:.1}ms  dilate={dilate_avg:.2}ms"
        );

        // (4) Input build (already includes channel-split + CHW reordering), tensor
        // construction, single-run inference, output extraction, resize-back
        // + re-threshold. Prints shapes for single vs batch-2.
        let input = build_selfie_segmenter_input(&crop);
        let single_t = ort::value::Tensor::from_array((
            vec![1i64, 3, SELFIE_SEG_INPUT as i64, SELFIE_SEG_INPUT as i64],
            input.clone(),
        ))
        .unwrap();

        // ONNX run alone: 1× vs 2× vs 4× — the ONLY part batching amortizes.
        let mut t4 = input.clone();
        for _ in 0..3 {
            t4.extend_from_slice(&input);
        }
        let t4 = ort::value::Tensor::from_array((
            vec![4i64, 3, SELFIE_SEG_INPUT as i64, SELFIE_SEG_INPUT as i64],
            t4,
        ))
        .unwrap();
        let t0 = Instant::now();
        for _ in 0..RUNS {
            std::hint::black_box(sess.run(ort::inputs![single_t.clone()]).unwrap());
        }
        let run1 = t0.elapsed().as_secs_f64() * 1000.0 / RUNS as f64;
        let t0 = Instant::now();
        for _ in 0..RUNS {
            std::hint::black_box(sess.run(ort::inputs![t4.clone()]).unwrap());
        }
        let run4 = t0.elapsed().as_secs_f64() * 1000.0 / RUNS as f64;
        eprintln!("session.run: 1-face={run1:.1}ms  4-faces-in-one={run4:.1}ms  (amortized/face={:.1}ms)", run4 / 4.0);
        if let Ok(outs1) = sess.run(ort::inputs![single_t]) {
            for (k, v) in outs1.iter() {
                if let Ok((shape, _)) = v.try_extract_tensor::<f32>() {
                    eprintln!("single output '{k}' shape={shape:?}");
                }
            }
        }
        let mut batched = input.clone();
        batched.extend_from_slice(&input);
        let tensor = ort::value::Tensor::from_array((
            vec![2i64, 3, SELFIE_SEG_INPUT as i64, SELFIE_SEG_INPUT as i64],
            batched,
        ))
        .unwrap();
        match sess.run(ort::inputs![tensor]) {
            Ok(o) => {
                for (k, v) in o.iter() {
                    if let Ok((shape, data)) = v.try_extract_tensor::<f32>() {
                        eprintln!(
                            "BATCH-N output '{k}' shape={shape:?} len={}",
                            data.len()
                        );
                    }
                }
            }
            Err(e) => eprintln!("BATCH-N: REJECTED (static [1,3,256,256] input): {e}"),
        };
    }
}
