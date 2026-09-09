//! ZIP ingestion & output (spec §2, §8 `zip_worker.rs`).
//!
//! Uploaded archives are spooled to disk by the HTTP handlers, so neither the
//! input nor the output is ever held in RAM: `.zip` archives are opened as
//! file-backed `ZipArchive`s, `.7z`/`.rar` decompress from the spooled file
//! into a scratch dir under `DATA_DIR`. Every image is decoded, anonymized
//! and re-encoded by a blocking worker whose concurrency is bounded by the §9
//! semaphore (auto-detected from the core count), and the anonymized frames
//! are streamed into a zero-compression (`Stored`) output ZIP written
//! directly to disk under `DATA_DIR` (spec §8 "scrittura output STORE").

use std::collections::{HashMap, HashSet};
use std::io::{Cursor, Read, Seek, Write};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use tokio::sync::Semaphore;

use crate::config::Config;
use crate::db::{Camera, CameraState, Db, Detection};
use crate::models::ModelStore;
use crate::pipeline::{process_image, Branch};

// ─── Camera / entry parser (spec §2) ────────────────────────────────────────

/// Classification of one ZIP entry path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EntryKind {
    /// `CAM_001/foto.jpg` or `CAM_001_foto.jpg` style image.
    Image(EntryTarget),
    /// Directory entry — structural, never an error.
    Directory,
    /// Anything else (`.DS_Store`, `.txt`, unknown layout…) — an error per §2.
    Other,
}

/// A ZIP entry mapped to its camera.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EntryTarget {
    pub camera_id: String,
    /// Output name: the original relative path (mirrors the input layout).
    pub out_name: String,
    pub image_format: ImageFormat,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageFormat {
    Jpeg,
    Png,
}

impl ImageFormat {
    fn from_ext(ext: &str) -> Option<ImageFormat> {
        match ext.to_ascii_lowercase().as_str() {
            "jpg" | "jpeg" => Some(ImageFormat::Jpeg),
            "png" => Some(ImageFormat::Png),
            _ => None,
        }
    }
}

/// Robust camera-id grammar: 2–64 chars of `[A-Za-z0-9_-]`, starting with an
/// alphanumeric. Keeps DB keys and on-disk crop paths safe.
fn valid_camera_id(id: &str) -> bool {
    let mut chars = id.chars();
    match chars.next() {
        Some(c) if c.is_ascii_alphanumeric() => {}
        _ => return false,
    }
    let mut len = 1;
    for c in chars {
        if !(c.is_ascii_alphanumeric() || c == '_' || c == '-') {
            return false;
        }
        len += 1;
    }
    (2..=64).contains(&len)
}

fn is_image_name(name: &str) -> Option<ImageFormat> {
    let base = name.rsplit('/').next().unwrap_or(name);
    if base.is_empty() || base.starts_with('.') {
        return None;
    }
    let ext = base.rsplit_once('.')?;
    if ext.0.is_empty() {
        return None;
    }
    ImageFormat::from_ext(ext.1)
}

fn normalize_path(mut p: &str) -> String {
    let mut out = p.replace('\\', "/");
    while let Some(stripped) = out.strip_prefix("./") {
        out = stripped.to_string();
    }
    while let Some(stripped) = out.strip_prefix('/') {
        out = stripped.to_string();
    }
    let _ = &mut p;
    out
}

/// Output name with the extension replaced by `new_ext` (directory layout
/// preserved). Used when `OUTPUT_FORMAT` forces a conversion: an entry keeps
/// its path, only the suffix changes (e.g. `CAM_001/frame_1.PNG` → `.png`).
fn replace_ext(name: &str, new_ext: &str) -> String {
    match name.rsplit_once('.') {
        Some((stem, _)) => format!("{stem}.{new_ext}"),
        None => format!("{name}.{new_ext}"),
    }
}

/// Classifies an archive entry path (spec §2: folder style `CAM_001/foto.jpg`
/// or prefix style `CAM_001_foto.jpg`; anything else is an error — never a
/// panic).
pub fn classify_entry(path: &str) -> EntryKind {
    let p = normalize_path(path);
    // Reject path traversal outright (never trust zip paths, §2 robustness).
    if p.split('/').any(|seg| seg == "..") {
        return EntryKind::Other;
    }
    if p.is_empty() || p.ends_with('/') {
        return EntryKind::Directory;
    }

    // Folder style: the (last) folder component is the camera id.
    if let Some((folder, file)) = p.rsplit_once('/') {
        let camera = folder.rsplit('/').next().unwrap_or(folder);
        if valid_camera_id(camera) {
            if let Some(fmt) = is_image_name(file) {
                return EntryKind::Image(EntryTarget {
                    camera_id: camera.to_string(),
                    out_name: p.clone(),
                    image_format: fmt,
                });
            }
            // A real file inside a camera folder but not a supported image
            // (e.g. a log dropped beside the frames) → unsupported.
            return EntryKind::Other;
        }
        return EntryKind::Other;
    }

    // Prefix style: `CAM_001_foto.jpg` → id from the first two `_` segments.
    if let Some(fmt) = is_image_name(&p) {
        let segments: Vec<&str> = p.split('_').collect();
        if segments.len() >= 2 {
            let candidate = format!("{}_{}", segments[0], segments[1]);
            if valid_camera_id(&candidate) {
                return EntryKind::Image(EntryTarget {
                    camera_id: candidate,
                    out_name: p.clone(),
                    image_format: fmt,
                });
            }
        }
    }
    EntryKind::Other
}

// ─── Job-level types ────────────────────────────────────────────────────────

/// Immutable snapshot of everything a blocking image worker needs.
#[derive(Clone)]
pub struct CameraJob {
    pub camera_id: String,
    pub state: CameraState,
    pub roi_json: Option<String>,
    pub out_name: String,
    pub image_format: ImageFormat,
}

/// Result of one anonymized image.
pub struct ImageOutcome {
    pub out_name: String,
    /// FSM branch that anonymized this frame.
    pub branch: Branch,
    /// Encoded image bytes; `None` when the entry was dropped (error counted).
    pub payload: Option<Vec<u8>>,
    /// Detection centers persisted during LEARNING: `(x, y, confidence)`.
    pub detections: Vec<(f32, f32, f32)>,
    pub fp_crops_saved: u32,
    /// Decoded frame geometry (for the ROI pixel-space bookkeeping).
    pub frame_size: Option<(u32, u32)>,
    /// Set when this entry failed (undecodable / failsafe / encode error);
    /// surfaced in `_error.txt` + `X-Processing-Errors-Detail`.
    pub error: Option<String>,
}

/// Per-camera activity summary for one job (logs / operator visibility).
#[derive(Debug, Clone)]
pub struct CameraSummary {
    pub camera_id: String,
    pub branch: Branch,
    pub images: usize,
    pub detections_stored: usize,
    pub fp_crops_saved: usize,
}

/// One skipped/failed archive entry with the exact reason.
#[derive(Debug, Clone)]
pub struct ArchiveEntryError {
    /// Original entry path inside the uploaded archive.
    pub entry: String,
    /// Human-readable failure reason (same text as the log line).
    pub message: String,
}

/// Outcome of one whole archive job. The anonymized output ZIP lives on disk
/// under `DATA_DIR` (never in RAM) — the HTTP handler streams it back to the
/// client with the `X-Processing-Errors*` headers.
#[derive(Debug)]
pub struct ZipJobOutcome {
    /// Absolute path of the finished output ZIP on disk.
    pub output_path: std::path::PathBuf,
    /// Size in bytes of the finished output ZIP (for `Content-Length`).
    pub output_size: u64,
    /// Base file name served to the client (`<input>_elaborato.zip`).
    pub output_name: String,
    pub error_count: usize,
    pub processed_count: usize,
    pub camera_summaries: Vec<CameraSummary>,
    /// Per-entry failures (also mirrored inside the output archive as
    /// `<input>_error.txt` and exposed via `X-Processing-Errors-Detail`).
    pub errors: Vec<ArchiveEntryError>,
}

/// The bounded-concurrency ZIP processor (spec §2, §9).
#[derive(Clone)]
pub struct ZipProcessor {
    cfg: Arc<Config>,
    db: Db,
    store: ModelStore,
    semaphore: Arc<Semaphore>,
    errors: Arc<AtomicUsize>,
}

impl ZipProcessor {
    pub fn new(cfg: Arc<Config>, db: Db, store: ModelStore) -> Self {
        let capacity = cfg.effective_concurrency();
        Self {
            cfg,
            db,
            store,
            semaphore: Arc::new(Semaphore::new(capacity)),
            errors: Arc::new(AtomicUsize::new(0)),
        }
    }

    pub fn error_count(&self) -> usize {
        self.errors.load(Ordering::Relaxed)
    }

    /// Resolved data dir of this processor (the S3 worker stages its scratch
    /// files there so uploads/downloads never hit cross-device copies).
    #[cfg(feature = "s3")]
    pub fn data_dir(&self) -> &std::path::Path {
        &self.cfg.data_dir
    }

    /// Resolves a camera row once per camera id per job: the first image of a
    /// camera queries/creates the row, all subsequent frames reuse the cached
    /// snapshot. Saves one sqlite roundtrip (and a transaction) per frame.
    async fn camera_for(
        &self,
        cache: &mut HashMap<String, Camera>,
        id: &str,
    ) -> Result<Camera> {
        if let Some(cam) = cache.get(id) {
            return Ok(cam.clone());
        }
        let cam = self.db.get_or_create_camera(id).await?;
        cache.insert(id.to_string(), cam.clone());
        Ok(cam)
    }

    /// Dispatches an uploaded archive (already spooled to disk by the HTTP
    /// handler) by extension: .zip is opened as a file-backed `ZipArchive`
    /// (the main path), .7z / .rar decompress from the file into a scratch dir
    /// under DATA_DIR, then feed the same per-image pipeline. Anything else is
    /// rejected. The output ZIP is streamed to `DATA_DIR/<input>_elaborato.zip`.
    pub async fn process_archive_file(
        &self,
        input_name: &str,
        path: &std::path::Path,
    ) -> Result<ZipJobOutcome> {
        let lower = input_name.to_ascii_lowercase();
        if lower.ends_with(".zip") {
            let file = std::fs::File::open(path)
                .with_context(|| format!("cannot open spooled upload {}", path.display()))?;
            self.process_zip(input_name, file).await
        } else if lower.ends_with(".7z") {
            self.process_7z(input_name, path).await
        } else if lower.ends_with(".rar") {
            self.process_rar(input_name, path).await
        } else {
            anyhow::bail!("unsupported archive format '{input_name}' (supported: .zip, .7z, .rar)")
        }
    }

    /// Processes a ZIP whose archive is backed by `R` (a `std::fs::File` in
    /// production, an in-memory `Cursor` in tests). Entries are read one at a
    /// time and the anonymized frames are streamed into an on-disk output ZIP
    /// (spec §8 "scrittura output STORE"): memory stays bounded by the
    /// concurrent task buffers, never by archive or output size.
    async fn process_zip<R>(&self, input_name: &str, reader: R) -> Result<ZipJobOutcome>
    where
        R: Read + Seek + Send + 'static,
    {
        self.errors.store(0, Ordering::Relaxed);
        let mut entry_errors: Vec<ArchiveEntryError> = Vec::new();
        let mut archive =
            zip::ZipArchive::new(reader).context("uploaded file is not a readable ZIP archive")?;

        // Pass 1: classify every entry (no reads). Directories are structural;
        // anything unsupported is a logged, counted error that never aborts.
        let mut image_entries: Vec<(usize, EntryTarget)> = Vec::new();
        for i in 0..archive.len() {
            let name = archive
                .by_index(i)
                .map(|e| e.name().to_string())
                .unwrap_or_default();
            match classify_entry(&name) {
                EntryKind::Image(target) => image_entries.push((i, target)),
                EntryKind::Directory => {}
                EntryKind::Other => {
                    self.errors.fetch_add(1, Ordering::Relaxed);
                    entry_errors.push(ArchiveEntryError {
                        entry: name.clone(),
                        message: "unsupported entry (not an image in a camera layout)".into(),
                    });
                    tracing::warn!("skipping unsupported ZIP entry #{i} '{name}'");
                }
            }
        }

        let job_stamp = chrono::Utc::now().format("%Y%m%d_%H%M%S%3f").to_string();
        let total = image_entries.len();
        tracing::info!(
            "job '{input_name}': {total} image entries to process, {} pre-existing errors",
            self.error_count()
        );

        // Output ZIP writer: streamed straight to disk under DATA_DIR
        // (spec §8 STORE), zero compression — never buffered in RAM.
        let out_path = self
            .cfg
            .data_dir
            .join(sanitize_filename(&output_stem(input_name)));
        let out_file = std::fs::File::create(&out_path)
            .with_context(|| format!("create output archive {}", out_path.display()))?;
        let mut zip_out = zip::ZipWriter::new(out_file);

        // JoinSet items carry the entry path so per-entry failures can be
        // reported precisely: (out_name, Result<(camera_id, outcome)>).
        let mut set: tokio::task::JoinSet<(String, anyhow::Result<(String, ImageOutcome)>)> =
            tokio::task::JoinSet::new();
        // Pass 2: schedule all images. Reads are sequential (ZipArchive is
        // borrowed by this loop); decode + inference + blur run in parallel
        // under the semaphore.
        let mut cam_cache: HashMap<String, Camera> = HashMap::new();
        for (idx, target) in image_entries.into_iter() {
            let permit = self
                .semaphore
                .clone()
                .acquire_owned()
                .await
                .map_err(|_| anyhow!("semaphore closed"))?;

            // Reads are kept inside a synchronous helper: `ZipFile` owns a
            // boxed `dyn Read` (not Send) and must never cross an `.await`.
            let bytes = match read_entry_bytes(&mut archive, idx) {
                Ok(b) => b,
                Err(e) => {
                    self.errors.fetch_add(1, Ordering::Relaxed);
                    entry_errors.push(ArchiveEntryError {
                        entry: target.out_name.clone(),
                        message: format!("cannot read ZIP entry #{idx}: {e}"),
                    });
                    tracing::warn!("cannot read ZIP entry #{idx}: {e}");
                    drop(permit);
                    continue;
                }
            };

            // Camera FSM bookkeeping lives on the async side (sqlx).
            let cam = match self.camera_for(&mut cam_cache, &target.camera_id).await {
                Ok(c) => c,
                Err(e) => {
                    self.errors.fetch_add(1, Ordering::Relaxed);
                    entry_errors.push(ArchiveEntryError {
                        entry: target.out_name.clone(),
                        message: format!("DB error for camera {}: {e}", target.camera_id),
                    });
                    tracing::error!("DB error for camera {}: {e}", target.camera_id);
                    drop(permit);
                    continue;
                }
            };
            let job = CameraJob {
                camera_id: cam.id.clone(),
                state: cam.state,
                roi_json: cam.roi_json.clone(),
                out_name: target.out_name,
                image_format: target.image_format,
            };

            set.spawn(
                self.clone()
                    .image_task(permit, cam, job, bytes, job_stamp.clone()),
            );
        }

        // Pass 3: stream finished images into the output ZIP as they arrive.
        let (processed, pending_detections, summaries, entry_errors) = self
            .drain_results(&mut set, &mut zip_out, entry_errors)
            .await;

        // Batch-persist LEARNING detection coordinates (spec §4, background).
        for dets in pending_detections.values() {
            if let Err(e) = self.db.insert_detections(dets).await {
                tracing::error!("cannot persist detections: {e}");
            }
        }

        let output_size = finalize_zip(zip_out, input_name, &entry_errors)?;

        let camera_summaries: Vec<CameraSummary> = summaries
            .into_iter()
            .map(|(camera_id, (branch, images, dets, fp))| CameraSummary {
                camera_id,
                branch,
                images,
                detections_stored: dets,
                fp_crops_saved: fp,
            })
            .collect();

        let error_count = self.error_count();
        tracing::info!(
            "job '{input_name}': {processed}/{total} images written, {error_count} errors",
        );

        Ok(ZipJobOutcome {
            output_path: out_path,
            output_size,
            output_name: output_stem(input_name),
            error_count,
            processed_count: processed,
            camera_summaries,
            errors: entry_errors,
        })
    }

    /// Spawns one image task (camera FSM transition + blocking worker) into
    /// the shared JoinSet, returning `(out_name, Result<(camera_id, outcome)>)`
    /// so per-entry failures can be reported precisely. Takes `self` by value
    /// (ZipProcessor is cheap-to-clone) so the future is `'static`.
    async fn image_task(
        self,
        _permit: tokio::sync::OwnedSemaphorePermit,
        camera: Camera,
        job: CameraJob,
        bytes: Vec<u8>,
        job_stamp: String,
    ) -> (String, anyhow::Result<(String, ImageOutcome)>) {
        let db = self.db.clone();
        let cfg = self.cfg.clone();
        let store = self.store.clone();
        let errors = self.errors.clone();
        let out_name = job.out_name.clone();
        // INITIAL is a transitional fallback: after one cautious full-frame
        // pass the camera moves to LEARNING (spec §4).
        if job.state == CameraState::Initial {
            if let Err(e) = db.set_state(&job.camera_id, CameraState::Learning).await {
                tracing::error!("cannot transition {} to LEARNING: {e}", job.camera_id);
            }
        }
        let result = tokio::task::spawn_blocking(move || {
            process_one_image(&cfg, &store, &job, bytes, &job_stamp, errors)
        })
        .await
        .map_err(|e| anyhow!("image worker panicked: {e}"));
        (out_name, result.map(|outcome| (camera.id, outcome)))
    }

    /// Pass 3 shared by every archive format: consumes the JoinSet as results
    /// arrive, writes anonymized frames into `zip_out`, accumulates per-camera
    /// summaries and per-entry errors.
    #[allow(clippy::type_complexity)]
    async fn drain_results(
        &self,
        set: &mut tokio::task::JoinSet<(String, anyhow::Result<(String, ImageOutcome)>)>,
        zip_out: &mut zip::ZipWriter<std::fs::File>,
        mut entry_errors: Vec<ArchiveEntryError>,
    ) -> (
        usize,
        HashMap<String, Vec<Detection>>,
        HashMap<String, (Branch, usize, usize, usize)>,
        Vec<ArchiveEntryError>,
    ) {
        let mut processed = 0usize;
        let mut pending_detections: HashMap<String, Vec<Detection>> = HashMap::new();
        let mut summaries: HashMap<String, (Branch, usize, usize, usize)> = HashMap::new();
        let mut sized_cameras: HashSet<String> = HashSet::new();

        while let Some(res) = set.join_next().await {
            let (entry_name, task_result) = match res {
                Ok(r) => r,
                Err(e) => {
                    // Join handle itself panicked — no entry name available.
                    self.errors.fetch_add(1, Ordering::Relaxed);
                    entry_errors.push(ArchiveEntryError {
                        entry: "<unknown>".into(),
                        message: format!("image task join error: {e}"),
                    });
                    tracing::error!("image task join error: {e}");
                    continue;
                }
            };
            let (camera_id, mut outcome) = match task_result {
                Ok(r) => r,
                Err(e) => {
                    self.errors.fetch_add(1, Ordering::Relaxed);
                    entry_errors.push(ArchiveEntryError {
                        entry: entry_name,
                        message: format!("image task failed: {e:#}"),
                    });
                    tracing::error!("image task failed: {e:#}");
                    continue;
                }
            };

            if let Some(msg) = outcome.error.take() {
                entry_errors.push(ArchiveEntryError {
                    entry: outcome.out_name.clone(),
                    message: msg,
                });
            }

            if let Some(payload) = &outcome.payload {
                let opts = zip::write::FileOptions::<()>::default()
                    .compression_method(zip::CompressionMethod::Stored);
                let written = zip_out
                    .start_file(outcome.out_name.as_str(), opts)
                    .and_then(|_| {
                        zip_out
                            .write_all(payload)
                            .map_err(zip::result::ZipError::Io)
                    });
                match written {
                    Ok(()) => processed += 1,
                    Err(e) => {
                        self.errors.fetch_add(1, Ordering::Relaxed);
                        entry_errors.push(ArchiveEntryError {
                            entry: outcome.out_name.clone(),
                            message: format!("cannot write output ZIP entry: {e}"),
                        });
                        tracing::error!("cannot write output ZIP entry: {e}");
                    }
                }
            }

            if let Some((w, h)) = outcome.frame_size {
                if sized_cameras.insert(camera_id.clone()) {
                    if let Err(e) = self.db.ensure_camera_frame_size(&camera_id, w, h).await {
                        tracing::warn!("cannot persist frame size for {camera_id}: {e}");
                    }
                }
            }

            let det_count = outcome.detections.len();
            if det_count > 0 {
                let now = chrono::Utc::now();
                let list = pending_detections.entry(camera_id.clone()).or_default();
                list.extend(outcome.detections.drain(..).map(|(x, y, conf)| Detection {
                    camera_id: camera_id.clone(),
                    x,
                    y,
                    confidence: conf,
                    captured_at: now,
                }));
            }

            let e = summaries.entry(camera_id.clone()).or_insert((
                outcome.branch,
                0usize,
                0usize,
                0usize,
            ));
            e.1 += 1;
            e.2 += det_count;
            e.3 += outcome.fp_crops_saved as usize;
        }

        (processed, pending_detections, summaries, entry_errors)
    }

    /// .7z ingestion: pure-Rust `sevenz-rust` decompresses the spooled file
    /// to a scratch dir under DATA_DIR, then every extracted image runs
    /// through the same parallel pipeline as ZIP entries. The scratch dir is
    /// removed on exit.
    async fn process_7z(&self, input_name: &str, path: &std::path::Path) -> Result<ZipJobOutcome> {
        self.errors.store(0, Ordering::Relaxed);
        let entry_errors: Vec<ArchiveEntryError> = Vec::new();

        let job_stamp = chrono::Utc::now().format("%Y%m%d_%H%M%S%3f").to_string();
        let scratch = self.cfg.data_dir.join(format!("tmp_7z_{job_stamp}"));
        std::fs::create_dir_all(&scratch).context("create 7z scratch dir")?;

        // Decompression is CPU/IO-bound and never touches the async runtime.
        let path2 = path.to_path_buf();
        let scratch2 = scratch.clone();
        let decompressed =
            tokio::task::spawn_blocking(move || decompress_7z_file(&path2, &scratch2))
                .await
                .map_err(|e| anyhow!("7z extraction worker panicked: {e}"))??;

        let outcome = self
            .process_extracted_dir(input_name, &scratch, decompressed, job_stamp, entry_errors)
            .await;
        let _ = std::fs::remove_dir_all(&scratch);
        outcome
    }

    /// .rar ingestion: `unrar` (libunrar) extracts the spooled file to a
    /// scratch dir, then the same parallel pipeline as ZIP entries. The
    /// scratch dir is removed on exit.
    async fn process_rar(&self, input_name: &str, path: &std::path::Path) -> Result<ZipJobOutcome> {
        self.errors.store(0, Ordering::Relaxed);
        let entry_errors: Vec<ArchiveEntryError> = Vec::new();

        let job_stamp = chrono::Utc::now().format("%Y%m%d_%H%M%S%3f").to_string();
        let scratch = self.cfg.data_dir.join(format!("tmp_rar_{job_stamp}"));
        std::fs::create_dir_all(&scratch).context("create rar scratch dir")?;

        let path2 = path.to_path_buf();
        let scratch2 = scratch.clone();
        let decompressed = tokio::task::spawn_blocking(move || decompress_rar(&path2, &scratch2))
            .await
            .map_err(|e| anyhow!("rar extraction worker panicked: {e}"))??;

        let outcome = self
            .process_extracted_dir(input_name, &scratch, decompressed, job_stamp, entry_errors)
            .await;
        let _ = std::fs::remove_dir_all(&scratch);
        outcome
    }

    /// Shared driver for scratch-dir archives (.7z/.rar): walk the extracted
    /// tree, classify entries (same grammar as ZIP), then run the identical
    /// parallel pipeline with bytes read from disk.
    async fn process_extracted_dir(
        &self,
        input_name: &str,
        root: &std::path::Path,
        entries: Vec<(String, bool)>, // (relative path, is_directory)
        job_stamp: String,
        mut entry_errors: Vec<ArchiveEntryError>,
    ) -> Result<ZipJobOutcome> {
        // Classify every extracted path (no reads).
        let mut image_entries: Vec<(std::path::PathBuf, EntryTarget)> = Vec::new();
        for (rel, is_dir) in entries {
            let name = rel.replace('\\', "/");
            if is_dir || name.is_empty() || name.ends_with('/') {
                continue;
            }
            match classify_entry(&name) {
                EntryKind::Image(target) => {
                    // Rebuild the on-disk path by re-joining the same segments
                    // (avoids any path-traversal surprises on Windows).
                    let mut p = root.to_path_buf();
                    for seg in name.split('/') {
                        p.push(seg);
                    }
                    image_entries.push((p, target));
                }
                EntryKind::Directory => {}
                EntryKind::Other => {
                    self.errors.fetch_add(1, Ordering::Relaxed);
                    entry_errors.push(ArchiveEntryError {
                        entry: name.clone(),
                        message: "unsupported entry (not an image in a camera layout)".into(),
                    });
                    tracing::warn!("skipping unsupported archive entry '{name}'");
                }
            }
        }

        let total = image_entries.len();
        tracing::info!(
            "job '{input_name}': {total} image entries to process, {} pre-existing errors",
            self.error_count()
        );

        // Output ZIP writer: streamed straight to disk under DATA_DIR
        // (spec §8 STORE), zero compression — never buffered in RAM.
        let out_path = self
            .cfg
            .data_dir
            .join(sanitize_filename(&output_stem(input_name)));
        let out_file = std::fs::File::create(&out_path)
            .with_context(|| format!("create output archive {}", out_path.display()))?;
        let mut zip_out = zip::ZipWriter::new(out_file);
        let mut set: tokio::task::JoinSet<(String, anyhow::Result<(String, ImageOutcome)>)> =
            tokio::task::JoinSet::new();

        // Pass 2: read from disk + schedule under the semaphore.
        let mut cam_cache: HashMap<String, Camera> = HashMap::new();
        for (path, target) in image_entries.into_iter() {
            let permit = self
                .semaphore
                .clone()
                .acquire_owned()
                .await
                .map_err(|_| anyhow!("semaphore closed"))?;

            let bytes = match std::fs::read(&path) {
                Ok(b) => b,
                Err(e) => {
                    self.errors.fetch_add(1, Ordering::Relaxed);
                    entry_errors.push(ArchiveEntryError {
                        entry: target.out_name.clone(),
                        message: format!("cannot read extracted file: {e}"),
                    });
                    tracing::warn!("cannot read extracted file {}: {e}", path.display());
                    drop(permit);
                    continue;
                }
            };

            let cam = match self.camera_for(&mut cam_cache, &target.camera_id).await {
                Ok(c) => c,
                Err(e) => {
                    self.errors.fetch_add(1, Ordering::Relaxed);
                    entry_errors.push(ArchiveEntryError {
                        entry: target.out_name.clone(),
                        message: format!("DB error for camera {}: {e}", target.camera_id),
                    });
                    tracing::error!("DB error for camera {}: {e}", target.camera_id);
                    drop(permit);
                    continue;
                }
            };
            let job = CameraJob {
                camera_id: cam.id.clone(),
                state: cam.state,
                roi_json: cam.roi_json.clone(),
                out_name: target.out_name,
                image_format: target.image_format,
            };
            set.spawn(
                self.clone()
                    .image_task(permit, cam, job, bytes, job_stamp.clone()),
            );
        }

        // Pass 3: shared drain.
        let (processed, pending_detections, summaries, entry_errors) = self
            .drain_results(&mut set, &mut zip_out, entry_errors)
            .await;

        for dets in pending_detections.values() {
            if let Err(e) = self.db.insert_detections(dets).await {
                tracing::error!("cannot persist detections: {e}");
            }
        }

        let output_size = finalize_zip(zip_out, input_name, &entry_errors)?;

        let camera_summaries: Vec<CameraSummary> = summaries
            .into_iter()
            .map(|(camera_id, (branch, images, dets, fp))| CameraSummary {
                camera_id,
                branch,
                images,
                detections_stored: dets,
                fp_crops_saved: fp,
            })
            .collect();

        let error_count = self.error_count();
        tracing::info!(
            "job '{input_name}': {processed}/{total} images written, {error_count} errors",
        );

        Ok(ZipJobOutcome {
            output_path: out_path,
            output_size,
            output_name: output_stem(input_name),
            error_count,
            processed_count: processed,
            camera_summaries,
            errors: entry_errors,
        })
    }

    /// Merges several finished on-disk output ZIPs (all `Stored`) into one
    /// combined archive, streaming entry-by-entry (memory bounded by a single
    /// entry — an image, not the whole batch). `extra_errors` (archives that
    /// failed before producing any output) are written into a `batch_error.txt`
    /// entry. Used by `/anonymize/batch` so multi-archive jobs still return a
    /// single ZIP. Returns the combined file's path and size.
    pub async fn merge_zips(
        &self,
        combined_name: &str,
        paths: &[std::path::PathBuf],
        extra_errors: &[ArchiveEntryError],
    ) -> Result<(std::path::PathBuf, u64)> {
        let out_path = self.cfg.data_dir.join(sanitize_filename(combined_name));
        let out_path2 = out_path.clone();
        let paths2 = paths.to_vec();
        let extra2 = extra_errors.to_vec();
        let size = tokio::task::spawn_blocking(move || -> Result<u64> {
            let out_file = std::fs::File::create(&out_path2)
                .with_context(|| format!("create batch output {}", out_path2.display()))?;
            let mut zip_out = zip::ZipWriter::new(out_file);
            // The zip crate's writer rejects duplicate entry names outright
            // (InvalidArchive("Duplicate filename")), and repeated uploads of
            // the same camera archive are a normal batch scenario — so the
            // merge deduplicates by entry name (first archive wins) instead
            // of failing the whole batch.
            let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();

            for p in &paths2 {
                let mut archive = zip::ZipArchive::new(std::fs::File::open(p)?)
                    .with_context(|| format!("re-open merged output {}", p.display()))?;
                for i in 0..archive.len() {
                    let mut entry = archive
                        .by_index(i)
                        .map_err(|e| anyhow!("read merged entry #{i}: {e}"))?;
                    if entry.is_dir() {
                        continue;
                    }
                    let name = entry.name().to_string();
                    if !seen.insert(name.clone()) {
                        tracing::warn!(
                            "batch merge: skipping duplicate entry '{name}' (first copy wins)"
                        );
                        continue;
                    }
                    let opts = zip::write::FileOptions::<()>::default()
                        .compression_method(zip::CompressionMethod::Stored);
                    zip_out
                        .start_file(name.as_str(), opts)
                        .map_err(|e| anyhow!("start merged entry '{name}': {e}"))?;
                    std::io::copy(&mut entry, &mut zip_out)
                        .map_err(|e| anyhow!("copy merged entry '{name}': {e}"))?;
                }
            }

            if !extra2.is_empty() && seen.insert("batch_error.txt".to_string()) {
                let mut content = format!(
                    "Errori durante l'elaborazione del batch — {} archivio/i scartato/i:\n\n",
                    extra2.len()
                );
                for e in &extra2 {
                    content.push_str(&format!("{}: {}\n", e.entry, e.message));
                }
                let opts = zip::write::FileOptions::<()>::default()
                    .compression_method(zip::CompressionMethod::Stored);
                zip_out
                    .start_file("batch_error.txt", opts)
                    .map_err(|e| anyhow!("start batch_error.txt: {e}"))?;
                zip_out
                    .write_all(content.as_bytes())
                    .map_err(|e| anyhow!("write batch_error.txt: {e}"))?;
            }

            let file = zip_out
                .finish()
                .map_err(|e| anyhow!("finish batch ZIP: {e}"))?;
            Ok(file.metadata().map(|m| m.len()).unwrap_or(0))
        })
        .await
        .map_err(|e| anyhow!("batch merge worker panicked: {e}"))??;
        Ok((out_path, size))
    }
}

/// Finishes the on-disk output ZIP: mirrors every per-entry failure inside
/// the archive as `<input>_error.txt`, then finalizes and returns the size.
fn finalize_zip(
    mut zip_out: zip::ZipWriter<std::fs::File>,
    input_name: &str,
    entry_errors: &[ArchiveEntryError],
) -> Result<u64> {
    if !entry_errors.is_empty() {
        let err_name = error_file_name(input_name);
        let mut content = format!(
            "Errori durante l'elaborazione di '{}' — {} file scartati o degradati:\n\n",
            input_name,
            entry_errors.len()
        );
        for e in entry_errors {
            content.push_str(&format!("{}: {}\n", e.entry, e.message));
        }
        let opts = zip::write::FileOptions::<()>::default()
            .compression_method(zip::CompressionMethod::Stored);
        if let Err(e) = zip_out.start_file(err_name.as_str(), opts).and_then(|_| {
            zip_out
                .write_all(content.as_bytes())
                .map_err(zip::result::ZipError::Io)
        }) {
            tracing::error!("cannot write error report into output ZIP: {e}");
        }
    }
    let file = zip_out
        .finish()
        .map_err(|e| anyhow!("finish output ZIP: {e}"))?;
    Ok(file.metadata().map(|m| m.len()).unwrap_or(0))
}

/// Decompresses a spooled .7z file into `out_dir` using the pure-Rust
/// `sevenz-rust` crate (file-backed reader: the archive is never loaded in
/// RAM). Returns `(relative path, is_directory)` for every entry. Entry names
/// are re-joined segment-by-segment to reject path traversal; unsafe names
/// abort the whole extraction.
fn decompress_7z_file(
    path: &std::path::Path,
    out_dir: &std::path::Path,
) -> Result<Vec<(String, bool)>> {
    let file = std::fs::File::open(path).context("open spooled 7z archive")?;
    let len = file.metadata().map(|m| m.len()).unwrap_or(0);
    let mut reader = sevenz_rust::SevenZReader::new(file, len, sevenz_rust::Password::empty())
        .map_err(|e| anyhow!("invalid 7z archive: {e}"))?;
    let mut entries: Vec<(String, bool)> = Vec::new();
    let mut aborted: Option<String> = None;
    reader
        .for_each_entries(|entry, content| {
            let name = entry.name().to_string();
            if entry.is_directory() {
                entries.push((name, true));
                return Ok(true);
            }
            entries.push((name.clone(), false));

            let mut dest = out_dir.to_path_buf();
            for seg in name.split(['/', '\\']) {
                if seg.is_empty() || seg == "." {
                    continue;
                }
                if seg == ".." || seg.contains(':') {
                    aborted = Some(format!("unsafe entry name '{name}' in 7z archive"));
                    return Ok(false);
                }
                dest.push(seg);
            }
            if let Some(parent) = dest.parent() {
                if let Err(e) = std::fs::create_dir_all(parent) {
                    aborted = Some(format!("cannot create {}: {e}", parent.display()));
                    return Ok(false);
                }
            }
            let mut buf = Vec::new();
            if let Err(e) = content.read_to_end(&mut buf) {
                aborted = Some(format!("cannot read 7z entry '{name}': {e}"));
                return Ok(false);
            }
            if let Err(e) = std::fs::write(&dest, &buf) {
                aborted = Some(format!("cannot write {}: {e}", dest.display()));
                return Ok(false);
            }
            Ok(true)
        })
        .map_err(|e| anyhow!("7z extraction failed: {e}"))?;
    if let Some(reason) = aborted {
        anyhow::bail!("{reason}");
    }
    Ok(entries)
}

/// Decompresses a spooled .rar file into `out_dir` using libunrar (`unrar`
/// crate, C++ source bundled by `unrar_sys`). The `unrar` crate is path-based,
/// so the spooled file is passed straight through — no copy needed. Returns
/// `(relative path, is_directory)` for every entry.
///
/// Entries are extracted one at a time with a destination path rebuilt
/// segment-by-segment, rejecting path traversal (`..`) outright.
fn decompress_rar(
    rar_path: &std::path::Path,
    out_dir: &std::path::Path,
) -> Result<Vec<(String, bool)>> {
    let mut entries: Vec<(String, bool)> = Vec::new();
    let mut open = unrar::Archive::new(rar_path)
        .open_for_processing()
        .map_err(|e| anyhow!("invalid RAR archive: {e}"))?;

    loop {
        let Some(next) = open
            .read_header()
            .map_err(|e| anyhow!("cannot read RAR header: {e}"))?
        else {
            break;
        };
        let name = next.entry().filename.to_string_lossy().into_owned();
        let is_dir = next.entry().is_directory();
        entries.push((name.clone(), is_dir));
        if is_dir {
            open = next
                .skip()
                .map_err(|e| anyhow!("cannot skip RAR dir: {e}"))?;
            continue;
        }
        let (content, next_open) = next
            .read()
            .map_err(|e| anyhow!("cannot read RAR entry '{name}': {e}"))?;

        let mut dest = out_dir.to_path_buf();
        let mut safe = true;
        for seg in name.split(['/', '\\']) {
            if seg.is_empty() || seg == "." {
                continue;
            }
            if seg == ".." || seg.contains(':') {
                safe = false;
                break;
            }
            dest.push(seg);
        }
        if !safe {
            anyhow::bail!("unsafe entry name '{name}' in RAR archive");
        }
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create {}", parent.display()))?;
        }
        std::fs::write(&dest, &content).with_context(|| format!("write {}", dest.display()))?;
        open = next_open;
    }

    Ok(entries)
}

/// Reads one archive entry fully into memory on the calling (non-async) path.
///
/// The `zip` crate's pure-Rust decoders (LZMA via `lzma-rs`, deflate64) can
/// desynchronize on some real-world streams — e.g. `LzmaError("LZ distance 1
/// is beyond output size 0")` when the first token misreads as a match. When
/// the crate's own decode fails we re-read the entry *raw* and retry with an
/// independent decoder (liblzma for LZMA/XZ, libbz2, libzstd, the deflate64
/// crate), accepting only a decode whose CRC32 and size match the entry's
/// stored metadata — so archives from any tool always decode or fail loudly.
fn read_entry_bytes<R: Read + Seek>(
    archive: &mut zip::ZipArchive<R>,
    idx: usize,
) -> Result<Vec<u8>> {
    // First pass: the zip crate's own decoder. `entry` must be dropped before
    // we may touch `archive` again (ZipFile borrows it mutably).
    let (expected_crc, expected_size, method, first_err) = {
        let mut entry = archive
            .by_index(idx)
            .map_err(|e| anyhow!("cannot open ZIP entry #{idx}: {e}"))?;
        let expected_crc = entry.crc32();
        let expected_size = entry.size();
        let method = entry.compression();
        let mut bytes = Vec::new();
        match entry.read_to_end(&mut bytes) {
            Ok(_) => return Ok(bytes),
            Err(e) => (expected_crc, expected_size, method, e),
        }
    };

    // Entry whose in-crate decode failed: re-read the compressed payload raw
    // and retry with an independent decoder, accepting only a decode whose
    // CRC32 and size match the entry's stored metadata.
    let mut raw = Vec::new();
    archive
        .by_index_raw(idx)
        .map_err(|e| anyhow!("cannot re-open ZIP entry #{idx}: {e}"))?
        .read_to_end(&mut raw)
        .map_err(|e| anyhow!("cannot re-read ZIP entry #{idx}: {e}"))?;
    if let Some(out) = decode_fallback_verify(method, &raw, expected_crc, expected_size) {
        tracing::warn!(
            "ZIP entry #{idx}: {} decoder failed ({first_err}); recovered with independent decoder ({})",
            method_name(method),
            out.len()
        );
        return Ok(out);
    }
    // Surface the compression method in the error so operators can see which
    // decoder failed (e.g. "(LZMA)" vs "(deflate64)") in _error.txt / logs.
    Err(anyhow!(
        "cannot read ZIP entry #{idx} ({} method): {first_err}",
        method_name(method)
    ))
}

fn method_name(m: zip::CompressionMethod) -> &'static str {
    match m {
        zip::CompressionMethod::Lzma => "LZMA",
        zip::CompressionMethod::Xz => "XZ",
        zip::CompressionMethod::Bzip2 => "bzip2",
        zip::CompressionMethod::Zstd => "zstd",
        zip::CompressionMethod::Deflate64 => "deflate64",
        _ => "compressed",
    }
}

/// Decodes `raw` with an independent decoder for `method`, verifying the
/// output against the entry's stored CRC32 and size. Returns `None` when the
/// method has no fallback or every candidate fails.
fn decode_fallback_verify(
    method: zip::CompressionMethod,
    raw: &[u8],
    expected_crc: u32,
    expected_size: u64,
) -> Option<Vec<u8>> {
    let candidates: Vec<Vec<u8>> = match method {
        zip::CompressionMethod::Lzma => lzma_container_candidates(raw),
        zip::CompressionMethod::Xz => vec![raw.to_vec()],
        zip::CompressionMethod::Bzip2 => vec![raw.to_vec()],
        zip::CompressionMethod::Zstd => vec![raw.to_vec()],
        zip::CompressionMethod::Deflate64 => vec![raw.to_vec()],
        _ => return None,
    };
    for candidate in candidates {
        let out = match method {
            zip::CompressionMethod::Lzma => lzma_decode_liblzma(&candidate),
            zip::CompressionMethod::Xz => xz_decode(&candidate),
            zip::CompressionMethod::Bzip2 => bzip2_decode(&candidate),
            zip::CompressionMethod::Zstd => zstd_decode(&candidate),
            zip::CompressionMethod::Deflate64 => deflate64_decode(&candidate),
            _ => None,
        };
        if let Some(out) = out {
            if crc32fast::hash(&out) == expected_crc && out.len() as u64 == expected_size {
                return Some(out);
            }
        }
    }
    None
}

fn xz_decode(data: &[u8]) -> Option<Vec<u8>> {
    let mut out = Vec::new();
    xz2::read::XzDecoder::new(Cursor::new(data))
        .read_to_end(&mut out)
        .ok()?;
    Some(out)
}

fn bzip2_decode(data: &[u8]) -> Option<Vec<u8>> {
    let mut out = Vec::new();
    bzip2::bufread::BzDecoder::new(std::io::BufReader::new(Cursor::new(data)))
        .read_to_end(&mut out)
        .ok()?;
    Some(out)
}

fn zstd_decode(data: &[u8]) -> Option<Vec<u8>> {
    zstd::stream::decode_all(Cursor::new(data)).ok()
}

fn deflate64_decode(data: &[u8]) -> Option<Vec<u8>> {
    let mut out = Vec::new();
    deflate64::Deflate64Decoder::new(Cursor::new(data))
        .read_to_end(&mut out)
        .ok()?;
    Some(out)
}

/// Builds plausible LZMA-*alone* streams (the layout liblzma expects:
/// `[props 5][unpacked size u64][raw LZMA1 stream]`) out of the raw entry
/// payload, under the two container layouts found in the wild:
///
/// - **`[ver 2][prop-len 2][props …][raw stream]`** — the ZIP LZMA layout
///   (APPNOTE 4.4.5.6, what Python's `zipfile` and 7-Zip write). The raw
///   stream is EOS-terminated, so the rebuilt size field is `0xFF × 8`.
/// - **raw bytes as-is** — already an alone stream (`[props][size][stream]`,
///   the layout lzma-rs's size-in-header parser accepts).
fn lzma_container_candidates(raw: &[u8]) -> Vec<Vec<u8>> {
    let mut out = Vec::with_capacity(3);
    if raw.len() >= 4 {
        let prop_len = u16::from_le_bytes([raw[2], raw[3]]) as usize;
        // Sane property length ⇒ treat the first 4 bytes as
        // `[version u16][prop-len u16]`. The CRC+size gate rejects any wrong
        // reconstruction, so the version byte is not trusted (7-Zip, Python
        // and other tools write different values).
        if (1..=16).contains(&prop_len) && 4 + prop_len < raw.len() {
            let mut alone = Vec::with_capacity(raw.len() - 4 + 8);
            alone.extend_from_slice(&raw[4..4 + prop_len]);
            alone.extend_from_slice(&[0xFF; 8]); // unknown size ⇒ EOS-terminated
            alone.extend_from_slice(&raw[4 + prop_len..]);
            out.push(alone);

            // Some writers skip the version field: `[prop-len u16][props]…`.
            if prop_len >= 1 && 2 + prop_len < raw.len() {
                let mut alone2 = Vec::with_capacity(raw.len() - 2 + 8);
                alone2.extend_from_slice(&raw[2..2 + prop_len]);
                alone2.extend_from_slice(&[0xFF; 8]);
                alone2.extend_from_slice(&raw[2 + prop_len..]);
                out.push(alone2);
            }
        }
    }
    out.push(raw.to_vec()); // already an alone stream `[props][size][stream]`
    out
}

/// Decodes a classic LZMA-alone stream with liblzma (`xz2`).
fn lzma_decode_liblzma(data: &[u8]) -> Option<Vec<u8>> {
    use xz2::stream::{Action, Status, Stream};
    let mut stream = Stream::new_lzma_decoder(u64::MAX).ok()?;
    let mut out = Vec::with_capacity(1 << 16);
    let mut pos = 0usize; // absolute offset of unconsumed input
    loop {
        // `process_vec` only fills the Vec's *spare* capacity; once it is
        // exactly full the call makes no progress, so grow before each step.
        if out.len() == out.capacity() {
            out.reserve(out.capacity().max(1));
        }
        match stream
            .process_vec(&data[pos..], &mut out, Action::Run)
            .ok()?
        {
            Status::StreamEnd => break,
            _ => {
                let consumed = stream.total_in() as usize;
                if consumed <= pos {
                    return None; // stuck or no more input: not a valid stream
                }
                pos = consumed;
            }
        }
    }
    Some(out)
}

/// `<input>_elaborato.zip` naming (spec §2): "frames.zip" → "frames_elaborato.zip".
fn output_stem(input_name: &str) -> String {
    let base = input_name.rsplit('/').next().unwrap_or(input_name);
    let stem = match base.rsplit_once('.') {
        Some((s, _)) => s,
        None => base,
    };
    if stem.is_empty() {
        format!("{base}_elaborato.zip")
    } else {
        format!("{stem}_elaborato.zip")
    }
}

/// Sanitizes a file name to a safe `[A-Za-z0-9._-]` token (used for stored
/// outputs and spooled uploads under DATA_DIR).
pub fn sanitize_filename(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    for c in name.chars() {
        if c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_') {
            out.push(c);
        } else {
            out.push('_');
        }
    }
    if out.is_empty() {
        out.push_str("output");
    }
    out
}

/// Per-upload error report inside the output archive: "frames.zip" →
/// "frames_error.txt" (same name as the input, `_error` suffix — §2).
fn error_file_name(input_name: &str) -> String {
    let base = input_name.rsplit('/').next().unwrap_or(input_name);
    let stem = match base.rsplit_once('.') {
        Some((s, _)) => s,
        None => base,
    };
    if stem.is_empty() {
        format!("{base}_error.txt")
    } else {
        format!("{stem}_error.txt")
    }
}

/// Blocking per-image worker: decode → FSM-conditional anonymization → encode.
///
/// Never panics: every failure is logged and counted; a processing error
/// degrades to the cautious INITIAL full-frame blur so an un-anonymized face
/// is never emitted.
fn process_one_image(
    cfg: &Config,
    store: &ModelStore,
    job: &CameraJob,
    bytes: Vec<u8>,
    job_stamp: &str,
    errors: Arc<AtomicUsize>,
) -> ImageOutcome {
    // OUTPUT_FORMAT override: `keep` preserves the input format, otherwise the
    // frame is re-encoded as JPEG/PNG and its output entry renamed to `.jpg`/
    // `.png` (the directory layout is unchanged).
    let (out_name, format) = match cfg.output_format {
        crate::config::OutputFormat::Keep => (job.out_name.clone(), job.image_format),
        crate::config::OutputFormat::Jpeg => (replace_ext(&job.out_name, "jpg"), ImageFormat::Jpeg),
        crate::config::OutputFormat::Png => (replace_ext(&job.out_name, "png"), ImageFormat::Png),
    };
    let mut outcome = ImageOutcome {
        out_name,
        branch: Branch::Initial,
        payload: None,
        detections: Vec::new(),
        fp_crops_saved: 0,
        frame_size: None,
        error: None,
    };

    let t0 = std::time::Instant::now();
    let decoded = match image::load_from_memory(&bytes) {
        Ok(img) => img,
        Err(e) => {
            errors.fetch_add(1, Ordering::Relaxed);
            outcome.error = Some(format!("undecodable image: {e}"));
            tracing::warn!("undecodable image {}: {e}", job.out_name);
            return outcome;
        }
    };
    let t_decode = t0.elapsed();
    let rgb = decoded.to_rgb8();
    let (w, h) = rgb.dimensions();
    outcome.frame_size = Some((w, h));

    let t1 = std::time::Instant::now();
    let res = process_image(cfg, store, job.state, job.roi_json.as_deref(), &rgb);
    let (processed, detections, fp_crops, branch) = match res {
        Ok(out) => {
            let dets = if out.record_detections {
                out.detections
                    .iter()
                    .map(|d| {
                        let (cx, cy) = d.bbox.center();
                        (cx, cy, d.confidence)
                    })
                    .collect()
            } else {
                Vec::new()
            };
            (out.processed, dets, out.fp_crops, out.branch)
        }
        Err(e) => {
            // Failsafe: cautious full-frame blur keeps the GDPR guarantee even
            // when inference itself fails.
            errors.fetch_add(1, Ordering::Relaxed);
            outcome.error = Some(format!(
                "processing failed, failsafe full-frame blur applied: {e}"
            ));
            tracing::error!(
                "processing failed for {} (failsafe full blur): {e}",
                job.camera_id
            );
            let mut rgba = image::DynamicImage::ImageRgb8(rgb).to_rgba8();
            crate::pipeline::AnonOp::from_cfg(cfg).apply_full_frame(&mut rgba);
            (rgba, Vec::new(), Vec::new(), Branch::Initial)
        }
    };
    let t_process = t1.elapsed();
    outcome.branch = branch;

    // Persist false-positive crops for retraining (§4):
    // /app/dataset_falsi_positivi/{camera_id}/{job_stamp}_{seq}.jpg
    if !fp_crops.is_empty() {
        let dir = cfg.dataset_fp_dir.join(&job.camera_id);
        if std::fs::create_dir_all(&dir).is_ok() {
            for (i, (_cx, _cy, crop)) in fp_crops.into_iter().enumerate() {
                let path = dir.join(format!("{job_stamp}_{i:04}.jpg"));
                if crop.save(&path).is_ok() {
                    outcome.fp_crops_saved += 1;
                }
            }
        }
    }

    // OUTPUT_MAX_SIDE: downscale after anonymization so the archive payloads
    // shrink while the stored camera geometry (frame_size) stays the original.
    let max_side = cfg.output_max_side_px;
    let (processed, ew, eh) = if max_side > 0 && w.max(h) > max_side {
        let k = max_side as f64 / w.max(h) as f64;
        let (ew, eh) = (((w as f64 * k) as u32).max(1), ((h as f64 * k) as u32).max(1));
        let resized = image::imageops::resize(
            &processed,
            ew,
            eh,
            image::imageops::FilterType::Triangle,
        );
        (resized, ew, eh)
    } else {
        (processed, w, h)
    };

    // Encode the processed frame (converted format if OUTPUT_FORMAT is set).
    let t2 = std::time::Instant::now();
    let enc_result = encode_frame(&processed, ew, eh, format, cfg.jpeg_quality);
    let t_encode = t2.elapsed();
    match enc_result {
        Ok(payload) => outcome.payload = Some(payload),
        Err(e) => {
            errors.fetch_add(1, Ordering::Relaxed);
            outcome.error = Some(format!("encode failed: {e}"));
            tracing::error!("encode failed for {}: {e}", job.out_name);
        }
    }

    let total = t0.elapsed();
    let det = detections.len();
    tracing::info!(
        target: "perf",
        camera = %job.camera_id,
        detect = det,
        decode_ms = t_decode.as_secs_f64() * 1000.0,
        process_ms = t_process.as_secs_f64() * 1000.0,
        encode_ms = t_encode.as_secs_f64() * 1000.0,
        total_ms = total.as_secs_f64() * 1000.0,
        "per-image timing"
    );

    outcome.detections = detections;
    outcome
}

fn encode_frame(
    rgba: &image::RgbaImage,
    w: u32,
    h: u32,
    format: ImageFormat,
    jpeg_quality: u8,
) -> Result<Vec<u8>> {
    use image::ImageEncoder;
    // RGBA→RGB by direct copy (single allocation, no full-frame clone).
    let rgb: image::ImageBuffer<image::Rgb<u8>, Vec<u8>> =
        image::ImageBuffer::from_fn(w, h, |x, y| {
            let c = rgba.get_pixel(x, y).0;
            image::Rgb([c[0], c[1], c[2]])
        });
    match format {
        ImageFormat::Jpeg => {
            let mut buf = Vec::with_capacity((w as usize) * (h as usize) / 2);
            {
                let enc = image::codecs::jpeg::JpegEncoder::new_with_quality(
                    &mut buf,
                    jpeg_quality,
                );
                enc.write_image(rgb.as_raw(), w, h, image::ColorType::Rgb8)
                    .map_err(|e| anyhow!("jpeg encode: {e}"))?;
            }
            Ok(buf)
        }
        ImageFormat::Png => {
            let mut buf = Vec::with_capacity((w as usize) * (h as usize) / 2);
            {
                let enc = image::codecs::png::PngEncoder::new(&mut buf);
                enc.write_image(rgb.as_raw(), w, h, image::ColorType::Rgb8)
                    .map_err(|e| anyhow!("png encode: {e}"))?;
            }
            Ok(buf)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::StreamExt;
    use image::GenericImageView;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::AtomicUsize;
    use tokio_util::io::ReaderStream;

    static DIR_COUNTER: AtomicUsize = AtomicUsize::new(0);

    fn test_temp_dir(tag: &str) -> PathBuf {
        let n = DIR_COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "av_zip_worker_test_{tag}_{}_{n}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn test_cfg(data_dir: &Path) -> Config {
        let mut cfg = Config::test_default();
        cfg.data_dir = data_dir.to_path_buf();
        cfg.dataset_fp_dir = data_dir.join("fp");
        cfg.dataset_seed_real_faces_dir = data_dir.join("seed");
        cfg.models_backup_dir = data_dir.join("backup");
        cfg
    }

    /// A store whose YOLO pool points at a nonexistent model: every branch
    /// that needs a session (LEARNING/ACTIVE) fails at acquisition. The
    /// INITIAL branch never touches a session, and `process_one_image`
    /// degrades session failures to the failsafe full-frame blur — so the
    /// whole spool → process → stream cycle is testable with no ONNX model.
    fn store_without_models() -> ModelStore {
        ModelStore::new(
            crate::models::SessionPool::new(PathBuf::from("/nonexistent/test-model.onnx"), 1),
            None,
            None,
        )
    }

    fn synthetic_image(w: u32, h: u32, format: ImageFormat) -> Vec<u8> {
        use image::ImageEncoder;
        let rgb = image::RgbImage::from_fn(w, h, |x, y| {
            image::Rgb([
                (x.wrapping_mul(7).wrapping_add(y.wrapping_mul(3))) as u8,
                (x.wrapping_mul(3).wrapping_add(y.wrapping_mul(5))) as u8,
                128u8.wrapping_add((x ^ y) as u8),
            ])
        });
        let mut buf = Cursor::new(Vec::new());
        match format {
            ImageFormat::Jpeg => {
                let enc = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut buf, 90);
                enc.write_image(rgb.as_raw(), w, h, image::ColorType::Rgb8)
                    .unwrap();
            }
            ImageFormat::Png => {
                let enc = image::codecs::png::PngEncoder::new(&mut buf);
                enc.write_image(rgb.as_raw(), w, h, image::ColorType::Rgb8)
                    .unwrap();
            }
        }
        buf.into_inner()
    }

    fn build_zip(entries: &[(&str, &[u8])]) -> Vec<u8> {
        let mut buf = Cursor::new(Vec::new());
        {
            let mut w = zip::ZipWriter::new(&mut buf);
            for (name, data) in entries {
                w.start_file(
                    *name,
                    zip::write::FileOptions::<()>::default()
                        .compression_method(zip::CompressionMethod::Stored),
                )
                .unwrap();
                w.write_all(data).unwrap();
            }
            w.finish().unwrap();
        }
        buf.into_inner()
    }

    fn write_zip(path: &Path, entries: &[(&str, &[u8])]) {
        std::fs::write(path, build_zip(entries)).unwrap();
    }

    fn zip_entry_names(bytes: &[u8]) -> Vec<String> {
        let mut archive = zip::ZipArchive::new(Cursor::new(bytes)).unwrap();
        (0..archive.len())
            .map(|i| archive.by_index(i).unwrap().name().to_string())
            .collect()
    }

    #[test]
    fn folder_style_camera_parsing() {
        assert_eq!(
            classify_entry("CAM_001/foto.jpg"),
            EntryKind::Image(EntryTarget {
                camera_id: "CAM_001".into(),
                out_name: "CAM_001/foto.jpg".into(),
                image_format: ImageFormat::Jpeg,
            })
        );
        assert_eq!(
            classify_entry("CAM_001\\frame_1.PNG"),
            EntryKind::Image(EntryTarget {
                camera_id: "CAM_001".into(),
                out_name: "CAM_001/frame_1.PNG".into(),
                image_format: ImageFormat::Png,
            })
        );
    }

    #[test]
    fn prefix_style_camera_parsing() {
        match classify_entry("CAM_001_foto.jpg") {
            EntryKind::Image(t) => {
                assert_eq!(t.camera_id, "CAM_001");
                assert_eq!(t.out_name, "CAM_001_foto.jpg");
            }
            other => panic!("expected image, got {other:?}"),
        }
    }

    #[test]
    fn nested_layout_uses_leaf_folder() {
        match classify_entry("uploads/2024/CAM_007/x.jpeg") {
            EntryKind::Image(t) => assert_eq!(t.camera_id, "CAM_007"),
            other => panic!("expected image, got {other:?}"),
        }
    }

    #[test]
    fn permissive_safe_folder_becomes_camera() {
        // Any safe token folder with a supported image is treated as a camera;
        // this is deliberate (ids like TL_042 work too).
        match classify_entry("notacamera/foto.jpg") {
            EntryKind::Image(t) => assert_eq!(t.camera_id, "notacamera"),
            other => panic!("expected image, got {other:?}"),
        }
    }

    #[test]
    fn unsupported_entries_are_not_panics() {
        assert_eq!(classify_entry(".DS_Store"), EntryKind::Other);
        assert_eq!(classify_entry("readme.txt"), EntryKind::Other);
        assert_eq!(
            classify_entry("__MACOSX/CAM_001/._foto.jpg"),
            EntryKind::Other
        );
        assert_eq!(classify_entry("a/b/c/foto.jpg"), EntryKind::Other);
        assert_eq!(classify_entry("../CAM_001/evil.jpg"), EntryKind::Other);
    }

    #[test]
    fn directory_entries_recognized() {
        assert_eq!(classify_entry("CAM_001/"), EntryKind::Directory);
        assert_eq!(classify_entry(""), EntryKind::Directory);
        assert_eq!(classify_entry("./"), EntryKind::Directory);
    }

    #[test]
    fn camera_id_safety() {
        assert!(valid_camera_id("CAM_001"));
        assert!(valid_camera_id("CAM-0042"));
        assert!(valid_camera_id("TL_042"));
        assert!(!valid_camera_id("../../etc/passwd"));
        assert!(!valid_camera_id("x"));
        assert!(!valid_camera_id(""));
        assert!(!valid_camera_id("A"));
    }

    #[test]
    fn lzma_entries_recovered_with_liblzma_fallback() {
        // tests/assets/lzma_mixed.zip was written by Python's zipfile with
        // ZIP_LZMA (method 14) + stored + deflated entries. The pure-Rust
        // lzma-rs decoder in the `zip` crate desynchronizes on the Python
        // LZMA container (reproducing the live "LzmaError(LZ distance 1 is
        // beyond output size 0)" seen on customer archives); the liblzma
        // fallback must recover every entry byte-exactly.
        let zip_bytes = std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/assets/lzma_mixed.zip"
        ))
        .unwrap();
        let mut archive = zip::ZipArchive::new(Cursor::new(zip_bytes)).unwrap(); // Sanity: the LZMA entries are really method 14 and the plain
                                                                                 // zip-crate path rejects them (the documented reproducer).
        for idx in 0..3 {
            let mut entry = archive.by_index(idx).unwrap();
            assert_eq!(entry.compression(), zip::CompressionMethod::Lzma);
            let mut v = Vec::new();
            assert!(
                entry.read_to_end(&mut v).is_err(),
                "lzma-rs unexpectedly decoded entry #{idx}"
            );
        }

        let expected = [
            b"Hello from LZMA entry A - the quick brown fox jumps over the lazy dog. ".repeat(8),
            b"Hello from LZMA entry B - 0123456789 ".repeat(12),
            b"The large entry exercises the liblzma buffer-growth path: this sentence repeats. "
                .repeat(5000), // 405 KB: must grow past the initial 64 KiB buffer
        ];
        for (idx, want) in expected.iter().enumerate() {
            assert_eq!(
                read_entry_bytes(&mut archive, idx).unwrap(),
                *want,
                "liblzma fallback decoded entry #{idx} incorrectly"
            );
        }
        // Non-LZMA entries still take the plain path.
        assert_eq!(
            read_entry_bytes(&mut archive, 4).unwrap(),
            expected[1][..100]
        );
    }

    #[test]
    fn output_name_suffix() {
        assert_eq!(output_stem("frames.zip"), "frames_elaborato.zip");
        assert_eq!(output_stem("a/b/lotto1.ZIP"), "lotto1_elaborato.zip");
        assert_eq!(output_stem("nofile"), "nofile_elaborato.zip");
        assert_eq!(error_file_name("Mio_Test.zip"), "Mio_Test_error.txt");
        assert_eq!(error_file_name("a/b/lotto1.ZIP"), "lotto1_error.txt");
        assert_eq!(error_file_name("nofile"), "nofile_error.txt");
    }

    #[test]
    fn independent_decoder_fallbacks_verify_crc_and_size() {
        // Each fallback decoder must reproduce the payload byte-exactly and
        // pass the CRC+size gate against its own compressed stream.
        let payload = b"face data \x00\xff\x01 payload for the fallback decoder test ".repeat(200);
        let expected_crc = crc32fast::hash(&payload);

        // bzip2
        let mut bz = bzip2::write::BzEncoder::new(Vec::new(), bzip2::Compression::best());
        std::io::Write::write_all(&mut bz, &payload).unwrap();
        let bz_bytes = bz.finish().unwrap();
        let out = decode_fallback_verify(
            zip::CompressionMethod::Bzip2,
            &bz_bytes,
            expected_crc,
            payload.len() as u64,
        )
        .expect("bzip2 fallback must decode");
        assert_eq!(out, payload);

        // zstd
        let z_bytes = zstd::stream::encode_all(std::io::Cursor::new(&payload), 3).unwrap();
        let out = decode_fallback_verify(
            zip::CompressionMethod::Zstd,
            &z_bytes,
            expected_crc,
            payload.len() as u64,
        )
        .expect("zstd fallback must decode");
        assert_eq!(out, payload);

        // XZ
        let mut xz = xz2::write::XzEncoder::new(Vec::new(), 6);
        std::io::Write::write_all(&mut xz, &payload).unwrap();
        let xz_bytes = xz.finish().unwrap();
        let out = decode_fallback_verify(
            zip::CompressionMethod::Xz,
            &xz_bytes,
            expected_crc,
            payload.len() as u64,
        )
        .expect("xz fallback must decode");
        assert_eq!(out, payload);

        // Tampered CRC must be rejected (the gate, not the decoder, decides).
        assert!(decode_fallback_verify(
            zip::CompressionMethod::Bzip2,
            &bz_bytes,
            expected_crc ^ 0xFF,
            payload.len() as u64,
        )
        .is_none());
        // Unsupported methods have no fallback.
        assert!(decode_fallback_verify(
            zip::CompressionMethod::Stored,
            &payload,
            expected_crc,
            payload.len() as u64,
        )
        .is_none());
        // Deflate64 has no encoder available in the dependency tree to craft a
        // live stream; the fallback path is exercised by the other methods.
    }

    #[test]
    fn encode_roundtrips_jpeg_and_png() {
        let img = image::RgbaImage::from_pixel(8, 8, image::Rgba([10, 20, 30, 255]));
        let jpeg = encode_frame(&img, 8, 8, ImageFormat::Jpeg, 90).unwrap();
        let back = image::load_from_memory(&jpeg).unwrap();
        assert_eq!(back.dimensions(), (8, 8));

        let png = encode_frame(&img, 8, 8, ImageFormat::Png, 90).unwrap();
        let back = image::load_from_memory(&png).unwrap();
        assert_eq!(back.dimensions(), (8, 8));
    }

    #[tokio::test]
    async fn merge_zips_combines_stored_archives_and_error_report() {
        let dir = test_temp_dir("merge");
        let cfg = test_cfg(&dir);
        let db = Db::open(&dir.join("t.sqlite3")).await.unwrap();
        let processor = ZipProcessor::new(Arc::new(cfg.clone()), db, store_without_models());

        let zip1 = dir.join("a.zip");
        let zip2 = dir.join("b.zip");
        write_zip(
            &zip1,
            &[("CAM_001/x.jpg", b"jpeg-a"), ("readme.txt", b"hi")],
        );
        write_zip(&zip2, &[("CAM_002/y.png", b"png-b")]);

        let extra = vec![ArchiveEntryError {
            entry: "rotto.zip".into(),
            message: "invalid upload: not a readable archive".into(),
        }];
        let (merged, size) = processor
            .merge_zips("batch_elaborato.zip", &[zip1.clone(), zip2], &extra)
            .await
            .unwrap();
        assert_eq!(merged, dir.join("batch_elaborato.zip"));
        assert_eq!(size, std::fs::metadata(&merged).unwrap().len());

        let bytes = std::fs::read(&merged).unwrap();
        let names = zip_entry_names(&bytes);
        for want in ["CAM_001/x.jpg", "CAM_002/y.png", "batch_error.txt"] {
            assert!(
                names.iter().any(|n| n == want),
                "missing {want} in {names:?}"
            );
        }
        let mut archive = zip::ZipArchive::new(Cursor::new(&bytes)).unwrap();
        let mut err = archive.by_name("batch_error.txt").unwrap();
        let mut content = String::new();
        err.read_to_string(&mut content).unwrap();
        assert!(content.contains("rotto.zip"));
        assert!(content.contains("invalid upload"));

        // Without extra errors the combined archive has no batch_error.txt.
        let (merged2, _) = processor
            .merge_zips("batch2_elaborato.zip", &[zip1], &[])
            .await
            .unwrap();
        assert!(
            zip::ZipArchive::new(Cursor::new(std::fs::read(&merged2).unwrap()))
                .unwrap()
                .by_name("batch_error.txt")
                .is_err(),
            "no batch_error.txt expected when nothing failed"
        );
    }

    #[tokio::test]
    async fn merge_zips_deduplicates_repeated_upload_entries() {
        // Repeated uploads of the same camera archive are a normal batch
        // scenario: the merge must deduplicate identical entry names (first
        // copy wins) instead of failing the whole batch with the zip crate's
        // "Duplicate filename" error.
        let dir = test_temp_dir("merge_dup");
        let cfg = test_cfg(&dir);
        let db = Db::open(&dir.join("t.sqlite3")).await.unwrap();
        let processor = ZipProcessor::new(Arc::new(cfg.clone()), db, store_without_models());

        let zip1 = dir.join("lotto1.zip");
        write_zip(
            &zip1,
            &[("CAM_001/x.jpg", b"jpeg-1"), ("CAM_002/y.png", b"png-2")],
        );
        let (merged, size) = processor
            .merge_zips("batch_elaborato.zip", &[zip1.clone(), zip1], &[])
            .await
            .unwrap();
        assert_eq!(size, std::fs::metadata(&merged).unwrap().len());

        let names = zip_entry_names(&std::fs::read(&merged).unwrap());
        assert_eq!(names.len(), 2, "duplicates must be dropped: {names:?}");
        assert!(names.contains(&"CAM_001/x.jpg".to_string()));
        assert!(names.contains(&"CAM_002/y.png".to_string()));
    }

    #[tokio::test]
    async fn spool_process_stream_cycle_on_disk_without_onnx() {
        let dir = test_temp_dir("cycle");
        let cfg = test_cfg(&dir);
        let db = Db::open(&dir.join("t.sqlite3")).await.unwrap();
        // One camera per frame, pre-created and reset to INITIAL: every frame
        // runs the session-free INITIAL branch, so the whole cycle works with
        // a store whose model file does not exist.
        for cam in ["CAM_001", "CAM_002", "CAM_003"] {
            db.get_or_create_camera(cam).await.unwrap();
            db.reset_to_initial(cam).await.unwrap();
        }

        let jpeg = synthetic_image(48, 32, ImageFormat::Jpeg);
        let png = synthetic_image(40, 40, ImageFormat::Png);
        let zip_bytes = build_zip(&[
            ("CAM_001/frame1.jpg", jpeg.as_slice()),
            ("CAM_002/frame1.png", png.as_slice()),
            ("CAM_003/frame1.jpg", jpeg.as_slice()),
            ("readme.txt", b"not an image"),
        ]);

        // Spool exactly like the /anonymize handler does.
        let spool = dir.join("upload_20260101_test.zip");
        std::fs::write(&spool, &zip_bytes).unwrap();

        let processor = ZipProcessor::new(Arc::new(cfg.clone()), db, store_without_models());
        let outcome = processor
            .process_archive_file("test.zip", &spool)
            .await
            .unwrap();

        // Outcome metadata matches the on-disk artifact.
        assert_eq!(outcome.output_name, "test_elaborato.zip");
        let out_path = dir.join("test_elaborato.zip");
        assert_eq!(outcome.output_path, out_path);
        assert!(out_path.exists());
        assert_eq!(
            outcome.output_size,
            std::fs::metadata(&out_path).unwrap().len()
        );
        assert_eq!(outcome.processed_count, 3);
        assert_eq!(outcome.error_count, 1, "only readme.txt should fail");
        assert_eq!(outcome.errors.len(), 1);
        assert!(outcome.errors[0].entry.ends_with("readme.txt"));

        // The output ZIP holds the 3 anonymized frames + the error report,
        // and every frame is still a decodable image.
        let out_bytes = std::fs::read(&out_path).unwrap();
        let names = zip_entry_names(&out_bytes);
        for want in [
            "CAM_001/frame1.jpg",
            "CAM_002/frame1.png",
            "CAM_003/frame1.jpg",
            "test_error.txt",
        ] {
            assert!(
                names.iter().any(|n| n == want),
                "missing {want} in {names:?}"
            );
        }
        let mut archive = zip::ZipArchive::new(Cursor::new(&out_bytes)).unwrap();
        for name in [
            "CAM_001/frame1.jpg",
            "CAM_002/frame1.png",
            "CAM_003/frame1.jpg",
        ] {
            let mut e = archive.by_name(name).unwrap();
            let mut bytes = Vec::new();
            e.read_to_end(&mut bytes).unwrap();
            assert!(
                image::load_from_memory(&bytes).is_ok(),
                "{name} is not a decodable image"
            );
        }
        let mut err = archive.by_name("test_error.txt").unwrap();
        let mut content = String::new();
        err.read_to_string(&mut content).unwrap();
        assert!(content.contains("readme.txt"));

        // Stream the on-disk archive back exactly like respond_with_archive:
        // a ReaderStream over the file, byte-identical to Content-Length.
        let file = tokio::fs::File::open(&out_path).await.unwrap();
        let mut streamed: Vec<u8> = Vec::new();
        let mut reader = ReaderStream::new(file);
        while let Some(chunk) = reader.next().await {
            streamed.extend_from_slice(&chunk.unwrap());
        }
        assert_eq!(streamed.len() as u64, outcome.output_size);
        assert_eq!(&streamed[..2], b"PK", "streamed bytes must be a ZIP");
        assert_eq!(streamed, out_bytes);

        // The spool is transient: handler-style cleanup leaves DATA_DIR with
        // only the output + sqlite + ledger artifacts.
        std::fs::remove_file(&spool).unwrap();
        assert!(!spool.exists());
    }

    #[tokio::test]
    async fn no_model_learning_camera_falls_back_to_failsafe_blur() {
        let dir = test_temp_dir("failsafe");
        let cfg = test_cfg(&dir);
        let db = Db::open(&dir.join("t.sqlite3")).await.unwrap();
        // get_or_create_camera creates LEARNING; with no model the YOLO
        // session cannot be acquired → failsafe full-frame blur. The image
        // must still be written and the failure reported.
        let img = synthetic_image(32, 32, ImageFormat::Jpeg);
        let zip_bytes = build_zip(&[("CAM_001/frame1.jpg", img.as_slice())]);
        let spool = dir.join("upload_x.zip");
        std::fs::write(&spool, &zip_bytes).unwrap();

        let processor = ZipProcessor::new(Arc::new(cfg.clone()), db, store_without_models());
        let outcome = processor
            .process_archive_file("x.zip", &spool)
            .await
            .unwrap();

        assert_eq!(outcome.processed_count, 1, "failsafe frame must be written");
        assert_eq!(outcome.error_count, 1);
        assert!(
            outcome.errors[0].message.contains("failsafe"),
            "unexpected error: {}",
            outcome.errors[0].message
        );
        let names = zip_entry_names(&std::fs::read(&outcome.output_path).unwrap());
        assert!(names.contains(&"CAM_001/frame1.jpg".to_string()));
        assert!(names.contains(&"x_error.txt".to_string()));
    }

    #[tokio::test]
    async fn output_format_converts_formats_and_renames_entries() {
        // Jpeg: a PNG input is re-encoded as JPEG and its entry renamed to
        // .jpg; the JPEG input stays .jpg. INITIAL branch, no models needed.
        let dir = test_temp_dir("convert_jpeg");
        let mut cfg = test_cfg(&dir);
        cfg.output_format = crate::config::OutputFormat::Jpeg;
        let db = Db::open(&dir.join("t.sqlite3")).await.unwrap();
        for cam in ["CAM_001", "CAM_002"] {
            db.get_or_create_camera(cam).await.unwrap();
            db.reset_to_initial(cam).await.unwrap();
        }
        let jpeg = synthetic_image(48, 32, ImageFormat::Jpeg);
        let png = synthetic_image(40, 40, ImageFormat::Png);
        let zip_bytes = build_zip(&[
            ("CAM_001/a.jpg", jpeg.as_slice()),
            ("CAM_002/b.png", png.as_slice()),
        ]);
        let spool = dir.join("up.zip");
        std::fs::write(&spool, &zip_bytes).unwrap();
        let processor = ZipProcessor::new(Arc::new(cfg.clone()), db, store_without_models());
        let outcome = processor
            .process_archive_file("up.zip", &spool)
            .await
            .unwrap();
        assert_eq!(outcome.processed_count, 2);
        let out_bytes = std::fs::read(&outcome.output_path).unwrap();
        let names = zip_entry_names(&out_bytes);
        for want in ["CAM_001/a.jpg", "CAM_002/b.jpg"] {
            assert!(
                names.iter().any(|n| n == want),
                "missing {want} in {names:?}"
            );
        }
        assert!(
            !names.iter().any(|n| n == "CAM_002/b.png"),
            "old png name must be gone: {names:?}"
        );
        let mut archive = zip::ZipArchive::new(Cursor::new(&out_bytes)).unwrap();
        for name in ["CAM_001/a.jpg", "CAM_002/b.jpg"] {
            let mut e = archive.by_name(name).unwrap();
            let mut bytes = Vec::new();
            e.read_to_end(&mut bytes).unwrap();
            assert!(
                image::load_from_memory(&bytes).is_ok(),
                "{name} is not a decodable image"
            );
        }

        // Png: both inputs re-encoded lossless as .png.
        let dir2 = test_temp_dir("convert_png");
        let mut cfg2 = test_cfg(&dir2);
        cfg2.output_format = crate::config::OutputFormat::Png;
        let db2 = Db::open(&dir2.join("t.sqlite3")).await.unwrap();
        db2.get_or_create_camera("CAM_001").await.unwrap();
        db2.reset_to_initial("CAM_001").await.unwrap();
        let zip2 = build_zip(&[("CAM_001/a.jpg", jpeg.as_slice())]);
        let spool2 = dir2.join("up.zip");
        std::fs::write(&spool2, &zip2).unwrap();
        let p2 = ZipProcessor::new(Arc::new(cfg2.clone()), db2, store_without_models());
        let out2 = p2
            .process_archive_file("up.zip", &spool2)
            .await
            .unwrap();
        let names2 = zip_entry_names(&std::fs::read(&out2.output_path).unwrap());
        assert!(
            names2.iter().any(|n| n == "CAM_001/a.png"),
            "{names2:?}"
        );
        assert!(
            !names2.iter().any(|n| n == "CAM_001/a.jpg"),
            "old jpg name must be gone: {names2:?}"
        );
    }

    #[tokio::test]
    async fn output_downscales_to_max_side_after_anonymization() {
        let dir = test_temp_dir("downscale");
        let mut cfg = test_cfg(&dir);
        cfg.output_max_side_px = 16;
        let db = Db::open(&dir.join("t.sqlite3")).await.unwrap();
        db.get_or_create_camera("CAM_001").await.unwrap();
        db.reset_to_initial("CAM_001").await.unwrap();
        let jpeg = synthetic_image(48, 32, ImageFormat::Jpeg);
        let zip_bytes = build_zip(&[("CAM_001/frame1.jpg", jpeg.as_slice())]);
        let spool = dir.join("up.zip");
        std::fs::write(&spool, &zip_bytes).unwrap();
        let processor = ZipProcessor::new(Arc::new(cfg.clone()), db, store_without_models());
        let outcome = processor
            .process_archive_file("up.zip", &spool)
            .await
            .unwrap();
        assert_eq!(outcome.processed_count, 1);
        let out_bytes = std::fs::read(&outcome.output_path).unwrap();
        let names = zip_entry_names(&out_bytes);
        assert!(
            names.contains(&"CAM_001/frame1.jpg".to_string()),
            "keep format must not rename the entry: {names:?}"
        );
        let mut archive = zip::ZipArchive::new(Cursor::new(&out_bytes)).unwrap();
        let mut e = archive.by_name("CAM_001/frame1.jpg").unwrap();
        let mut bytes = Vec::new();
        e.read_to_end(&mut bytes).unwrap();
        let img = image::load_from_memory(&bytes).unwrap();
        assert_eq!(
            img.dimensions(),
            (16, 10),
            "48x32 downscaled to max side 16 keeps the aspect ratio"
        );
    }
}
