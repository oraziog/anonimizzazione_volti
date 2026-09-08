# Anonimizzazione Volti — batch face-anonymization service (Rust + ONNX)

Industrial-grade microservice that anonymizes (blurs) faces in batches of images
captured by fixed traffic/ZTL cameras, targeting GDPR compliance: **zero
visibly-unblurred real faces in the output**, surgical precision via a binary
classifier, and automated ROI learning per camera.

Implementation of the technical specification (Italian) in
`MD/anonimizzazione_volti.md`. The spec's section numbers are referenced
throughout the source.

## Architecture

| Module | Responsibility (spec) |
| ------ | --------------------- |
| `src/main.rs` | Runtime model download + fail-fast, SQLite pool, Axum server, operator endpoints, nightly ROI/retraining scheduler |
| `src/models.rs` | ONNX session pooling (per-worker exclusive sessions), YOLOv8-Face DFL/keypoint decoding, binary classifier preprocessing |
| `src/model_loader.rs` | Runtime download with backoff, SHA-256 verification, on-disk cache, loadability checks |
| `src/zip_worker.rs` | In-memory ZIP ingestion, robust `CAM_001/…`/`CAM_001_…` camera parser, bounded concurrency, `Stored`-compression output ZIP, FP-crop persistence |
| `src/pipeline.rs` | FSM-conditional anonymization: INITIAL full-frame blur, LEARNING box blur + data collection, ACTIVE ROI gate + classifier + hull/ellipse masked blur |
| `src/roi.rs` | DBSCAN → outlier rejection → convex hull → RDP smoothing → geometric validation → safety margin |
| `src/db.rs` | SQLite (WAL): camera FSM persistence, detection coordinates, ROI polygons, frame geometry |
| `src/config.rs` | Centralized env-var configuration (fail-fast on invalid values) |
| `src/training.rs` + `python/retrain.py` | Optional nightly PyO3 retraining bridge: fine-tune + ONNX export + pre-swap validation + backup + JSON audit |
| `src/eval_wider.rs` / `src/eval_fddb.rs` | Offline CLI evaluation of the detector against WIDER FACE / FDDB ground truth (AP, precision/recall, recall vs FP-per-image) |
| `python/prepare_seed.py` | Classifier-seed builder: crops face boxes from a YOLO dataset or from WIDER FACE (difficulty-filtered) |

### Session model (important)

`ort` 2.0 `Session::run` requires `&mut self` — ONNX Runtime sessions are **not
safe for concurrent inference** (per the `ort` docs). Instead of sharing one
`ArcSwap<Session>`, concurrent image workers each get an *exclusive* session
from a `SessionPool` (bounded by the §9 concurrency semaphore, one live session
per busy worker). The classifier is hot-swapped after nightly retraining by
atomically replacing the *whole pool* with one pointing at the new ONNX file;
in-flight workers finish on the old sessions.

### Camera FSM (spec §4)

- `INITIAL` — camera unknown or operator-reset; cautious **full-frame** blur,
  then an immediate transition to `LEARNING`.
- `LEARNING` — YOLO box + 15% margin blur (`σ = box_width/8`, clamped `[5,50]`);
  detection centers are persisted to SQLite and dubious crops
  (`conf < FP_CROP_CONF_MAX`) are saved to
  `dataset_falsi_positivi/{camera_id}/` for retraining. After
  `LEARNING_DAYS` the nightly task extracts a ROI and activates the camera.
- `ACTIVE` — only detections whose center falls inside the persisted ROI are
  blurred; the binary classifier re-checks the crop (fail-safe: classifier
  errors blur anyway), and the blur region is the **convex hull of the 5 facial
  keypoints** (ellipse fallback when the loaded model exports no landmarks),
  masked + gaussian-softened, margin `BLUR_HULL_MARGIN_PCT`.

### ROI extraction (spec §5)

Nightly task (and one catch-up pass at boot): for each `LEARNING` camera whose
window elapsed — DBSCAN (`eps`/`min_samples`) → drop unclustered noise →
convex hull → RDP smoothing → geometric validation (polygon area must be
10–90% of the frame) → centroid expansion by `ROI_MARGIN_PCT`. Success moves the
camera to `ACTIVE`; failure keeps it in `LEARNING` (anomaly logged for manual
review, spec §5.6).

## Build & run

### Local (no retraining)

```bash
cargo build --release
cargo test
```

Set the operational env vars from `.env.example`:

```bash
cp .env.example .env      # then edit paths / thresholds
export $(grep -v '^#' .env | xargs)   # or use your preferred loader
cargo run --release
```

The default YOLO model (`yolov8n-face.onnx`, public WIDERFace-trained model
with 5 landmarks) is downloaded automatically at startup to `MODEL_CACHE_DIR`
and verified against `MODEL_YOLO_SHA256`. **Startup fails fast** if a
configured model cannot be downloaded or loaded (spec §3).

### Docker (recommended — full service with nightly retraining)

```bash
docker compose up -d --build
```

The image build enables the `retraining` cargo feature (PyO3, Python 3.11) and
installs CPU PyTorch; set a real `OPERATOR_API_KEY` in `.env` first.

### Retraining build requirements

`cargo build` uses no default features, so a plain toolchain suffices. To build
the retraining bridge locally you need Python **3.11 or 3.12** with headers
(`python3-dev`) and PyTorch/torchvision/Pillow available in that interpreter:

```bash
cargo build --release --features retraining
```

pyo3 0.21 does not support Python 3.13+/3.14, which is why the Docker base
image pins Python 3.11.

## HTTP API

### `POST /anonymize` — batch ingest (spec §2)

Multipart field `file` containing a **.zip, .7z or .rar** archive (up to
`BODY_LIMIT_BYTES`, default 3.5 GB). One job at a time: a concurrent upload
gets **HTTP 429**. The upload is spooled to disk and the anonymized output
ZIP is **streamed to disk and back to the client** — neither the input nor
the output is ever held in RAM, so memory stays bounded regardless of
archive size. .7z/.rar are decompressed from the spooled file to a scratch
dir under `DATA_DIR` (removed afterwards) and fed through the same per-image
pipeline.

```bash
curl -fsS -F "file=@frames_2024.zip" http://localhost:8080/anonymize \
  -o frames_elaborato.zip -D - | grep -i "x-processing-errors"
```

Response: the anonymized ZIP (`<input>_elaborato.zip`,
`Content-Disposition: attachment`, compression **Stored**/zero) plus:

- `X-Processing-Errors: <count>` — number of skipped/corrupt/unsupported
  entries (`.DS_Store`, `.txt`, undecodable images, unknown layouts… never
  aborts the job);
- `X-Processing-Errors-Detail` — when > 0, a percent-encoded JSON array of
  `{"entry": "...", "error": "..."}` (first 50) with the exact per-file
  failure; decode with `decodeURIComponent`;
- `<input>_error.txt` **inside** the output archive — same list, human
  readable (e.g. `Mio_Test.zip` → `Mio_Test_error.txt`).

A copy of the output is stored under `DATA_DIR`.

### `POST /anonymize/batch` — volumi oltre `BODY_LIMIT_BYTES`

Same multipart protocol, but accepts **multiple archive fields in one request**
(any field named `file`, or whose name ends in .zip/.7z/.rar) — also works
with **chunked transfer encoding** (no `Content-Length`). The body limit is
disabled on this route: each archive is spooled to disk and processed **in
sequence** under the single-job lock, so the total volume is unbounded.

- With **one** archive the response is identical to `/anonymize`.
- With **several**, the per-archive outputs are merged (streaming, entry by
  entry) into a single `batch_elaborato.zip`; archives that fail before
  producing output are reported in `X-Processing-Errors-Detail` and written
  to `batch_error.txt` inside the response.

```bash
curl -fsS -F "file=@lotto1.zip" -F "file=@lotto2.zip" \
  http://localhost:8080/anonymize/batch -o batch_elaborato.zip
```

Robust decoding: LZMA / bzip2 / zstd / XZ / deflate64 entries whose in-crate
decoder fails are re-read raw and retried with an independent decoder
(liblzma, libbz2, libzstd), accepting only decodes whose CRC32 and size match
the entry metadata (see `docs/PITFALL-ZIP.md` for the full list of pitfalls).

Camera layout accepted:

```
CAM_001/foto.jpg            (folder style)
CAM_001_foto.jpg            (prefix style)
uploads/2024/CAM_007/x.jpeg (nested; leaf folder = camera)
```

Camera ids must match `[A-Za-z0-9_-]{2,64}` starting alphanumeric; any other
entry is logged as an error (path traversal like `..` is rejected outright).

### `GET /health`

### Operator endpoints (spec §4 "Intervento Operatore")

Gated by header `X-Operator-Key` (disable by leaving `OPERATOR_API_KEY`
empty → all operator routes answer 403).

```bash
KEY=...   # the configured OPERATOR_API_KEY

curl -H "X-Operator-Key: $KEY" http://localhost:8080/operator/cameras
curl -H "X-Operator-Key: $KEY" http://localhost:8080/operator/cameras/CAM_001
# Force a regression:
curl -H "X-Operator-Key: $KEY" -X POST \
     -H "Content-Type: application/json" \
     -d '{"target_state":"INITIAL"}' \
     http://localhost:8080/operator/cameras/CAM_001/reset
#   INITIAL  → zeroes the ROI and restarts the learning cycle
#   LEARNING → keeps the production ROI but reopens data collection

# Last nightly-retraining audit record (JSON, written by the retraining job):
curl -H "X-Operator-Key: $KEY" http://localhost:8080/operator/retrain-audit

# Last processed jobs (reads DATA_DIR/jobs.jsonl + the on-disk outputs):
# archive, images, errors, output size, timestamps; orphan *_elaborato.zip
# files are reported too. Optional ?limit=N (default 20, max 500).
curl -H "X-Operator-Key: $KEY" "http://localhost:8080/operator/jobs?limit=20"
```

The audit JSON contains `status` (`swapped` / `rejected` / `skipped` /
`failed`), python + Rust validation accuracy, sample counts, min accuracy, and
the candidate/backup ONNX paths — so operators can see *why* a candidate was
accepted or discarded.

## Configuration

Every knob lives in env vars — see `.env.example` for the full annotated list.
Key groups:

- **Models** — `YOLO_MODEL_URL`, `CLASSIFIER_MODEL_URL` (optional initial
  classifier), `MODEL_CACHE_DIR`, `MODEL_*_SHA256` integrity checks.
- **Pipeline** — `YOLO_CONF_THRESHOLD`, `YOLO_NMS_IOU`,
  `FP_CROP_CONF_MAX`, `INITIAL_BLUR_SIGMA`, `BLUR_HULL_MARGIN_PCT`,
  `JPEG_QUALITY`.
- **FSM/ROI** — `LEARNING_DAYS`, `ROI_EPS_PX`, `ROI_MIN_SAMPLES`,
  `ROI_RDP_EPSILON`, `ROI_AREA_MIN`, `ROI_AREA_MAX`, `ROI_MARGIN_PCT`.
- **Retraining** — `CRON_RETRAIN_SCHEDULE` (HH:MM local),
  `RETRAIN_MIN_ACCURACY`, `RETRAIN_HOLDOUT_FRACTION`, `RETRAIN_EPOCHS`,
  `RETRAIN_BATCH_SIZE`, `RETRAIN_SCRIPT_PATH`.
- **Ops** — `MAX_CONCURRENT_IMAGES` (default: auto-detected — cores ≤ 4 →
  all cores, cores > 4 → cores − 2; override only to tune),
  `BODY_LIMIT_BYTES`, `DATA_DIR`, `OPERATOR_API_KEY`.
- **Retention (STORE outputs)** — the anonymized `<input>_elaborato.zip`
  files accumulating in `DATA_DIR` are pruned by a background loop:
  `RETENTION_MAX_DAYS` (age rule), `RETENTION_MAX_GB` (size rule, oldest
  first), `RETENTION_INTERVAL_SECS`, `RETENTION_MIN_AGE_SECS` (in-flight
  guard). `RETENTION_ENABLED=false` (or both thresholds 0) disables it.

### Tuning misurato (stress test, 1000 immagini, 2 core fisici)

| Configurazione | Durata | Throughput | Picco RAM | CPU media |
|---|---|---|---|---|
| default (concorrenza 1 su 2 core) | 412 s | 2,4 img/s | ~570 MB | ~100% |
| `MAX_CONCURRENT_IMAGES=2` | 206 s | **4,9 img/s** | 636 MB | 188% |
| `MAX_CONCURRENT_IMAGES=4` (oversubscription) | 209 s | 4,8 img/s | 762 MB | 197% |
| `MAX_CONCURRENT_IMAGES=2` + `JPEG_QUALITY=80` | 50 s / 300 img | **6,0 img/s** | — | — |

Conclusioni pratiche:

- **Il default ora si adatta da solo**: con ≤ 4 core il servizio usa **tutti**
i core (il vecchio `core − 2` scendeva a 1 su macchine a 2 core, dimezzando
il throughput); sopra i 4 core mantiene il margine `core − 2`. Oltre il
numero di core non si guadagna nulla (solo RAM).
- **`JPEG_QUALITY=95` gonfia l'output ~2,8× rispetto all'input** (290 MB in
uscita da 103 MB in ingresso per 1000 foto): scendere a 80–85 riduce del
~25–30% la dimensione e accelera l'encode del ~20%, con qualità visiva più
che sufficiente per footage anonimizzato.
- **La RAM scala con i buffer dei task concorrenti (~60–130 MB per immagine),
non con la dimensione di input/output** (entrambi ora vivono su disco,
streaming): un singolo archive fino a `BODY_LIMIT_BYTES` va sempre bene; per
volumi maggiori usate `/anonymize/batch` con più archive (o upload chunked),
che non ha limiti di dimensione totale.

The provided default YOLO export outputs three `[1,80,H,W]` pose-head maps
(DFL box + objectness + 5 keypoints); `run_yolo` decodes that format natively
and falls back to classic `[1,C,anchors]` single-output models
(`YOLO_MODEL_URL` can point at your own export).

## Memory & concurrency (spec §9)

In-RAM budget: working buffers of the concurrent tasks (decode 6 MB + YOLO
tensor ~5 MB + classifier ~0.6 MB per task) + one encoded image per worker.
Uploads and outputs are spooled/streamed on disk, never buffered in RAM.
Minimum 16 GB RAM; use 32 GB when nightly retraining may overlap a ZIP job
(embedded PyTorch). Image processing runs under `tokio::task::spawn_blocking`
with a semaphore of `MAX_CONCURRENT_IMAGES` (auto-detected: cores ≤ 4 →
cores, cores > 4 → cores − 2).

## Testing

```bash
cargo test                       # unit tests (no model / network needed)
cargo test real_face_model_smoke -- --ignored
# end-to-end check against the real model + a photo:
#   downloads public yolov8n-face.onnx + demo photo into .test-assets first
```

## Offline detector evaluation (`eval-wider`, `eval-fddb`)

Both harnesses reuse the exact runtime decoder (`run_yolo`); no server or
config needed. `--det-conf` / `--nms-iou` are the *detector* knobs (mirroring
`YOLO_CONF_THRESHOLD` / `YOLO_NMS_IOU`); `--iou` / `--match-iou` are the
*evaluation* matching threshold (0.5, per the reference evaluators).

```bash
# WIDER FACE val: per-split AP + precision at operating points
./target/release/anonimizzazione_volti eval-wider \
    --model model_cache/yolov8n-face.onnx \
    --images WIDER_val/images \
    --gt wider_face_split/wider_face_val_bbx_gt.txt \
    --det-conf 0.20 --nms-iou 0.45

# FDDB: ellipse folds → recall at FP-per-image operating points
./target/release/anonimizzazione_volti eval-fddb \
    --model model_cache/yolov8n-face.onnx \
    --images fddb/images \
    --folds 'FDDB-fold-01.txt,FDDB-fold-02.txt,...' \
    --det-conf 0.20
```

`eval-wider` prints, per difficulty split (easy/medium/hard, attribute-based
rule), the AP and the recall/precision pairs at conf 0.5/0.25/0.1, computed
with the exact counting semantics of the reference evaluator
(wondervictor/WiderFace-Evaluation); `eval-fddb` prints recall at
FPPI = 0.05/0.1/0.25/0.5/1.0 (FDDB's official metric), approximating each GT
ellipse with its axis-aligned bounding box. Downloading the validation sets:

- **WIDER FACE val** — images ~365 MB (Google Drive id
  `1GUCogbp16PMGa39thoMMeWxp7Rp5oM8Q`, MD5 `dfa7d7e790efa35df3788964cf0bbaea`)
  + GT `wider_face_split.zip` from `shuoyang1213.me/WIDERFACE`; or
  `python prepare_seed.py --wider-download DIR` which fetches both.
- **FDDB** — folds + images from `vis-www.cs.umass.edu/fddb/` (images split
  into parts; ~2.7 GB total).

## Building the classifier seed

```bash
# From a YOLO-format dataset (e.g. Kaggle CC0 "Face Detection Dataset"):
python python/prepare_seed.py --source path/to/dataset --out dataset_seed/real_faces

# From WIDER FACE val (auto-download + extract, ~365 MB):
python python/prepare_seed.py --wider-download /data/wider --out dataset_seed/real_faces \
    --min-difficulty medium --max-images 2000
```

The WIDER mode applies the same per-face difficulty rule as `eval-wider` and by
default keeps only `easy` + `medium` faces (tiny/blurry `hard` crops would
poison the classifier seed); `--include-ignored` re-enables `ignore==1` faces,
`--margin`/`--min-side`/`--max-crops` control the crop geometry and budget.

## Spec open points — status

1. **Seed dataset** — external (mounted at `dataset_seed/real_faces`); the
   system never generates it. `python/prepare_seed.py` builds it from a YOLO
   dataset or WIDER FACE (CC0 / research-licensed sources: check the license
   of the images you ship into a production model).
2. **Keypoints** — resolved: the default public model exports 5 landmarks;
   ACTIVE uses hull blur, with ellipse fallback decided at runtime from the
   model's actual output.
3. **Operator auth** — API key via `X-Operator-Key` header.
4. **Formats** — JPEG/PNG only; everything else is a counted, logged skip.
5. **Concurrent uploads** — global job lock, `HTTP 429 Too Many Requests`.

## Layout note

This crate lives in the `src/` subfolder of the repository (next to the
specification in `MD/`); `[workspace]` is declared in its manifest so cargo
never walks up into the outer scaffold.
