# Installation and commissioning manual (Windows)

This document distills the **entire real-world journey** of installing,
running and tuning the service on Windows: every step below was actually
executed and verified on a Windows 10/11 Pro machine. For the Linux/Docker
path see `src/README.md` (quick start) and `src/Dockerfile`.

> Field lesson: **the order of the steps matters**. Environment first, then
> the binary, then configuration, then training data. Skipping a step always
> costs you more later.

## Contents

1. [Prerequisites](#1-prerequisites)
2. [Getting the code](#2-getting-the-code)
3. [Building the service](#3-building-the-service)
4. [Configuration (.env)](#4-configuration-env)
5. [First run and verification](#5-first-run-and-verification)
6. [Nightly retraining (surgical classifier)](#6-nightly-retraining)
7. [Commissioning: the camera FSM](#7-commissioning-the-camera-fsm)
8. [Operations tooling](#8-operations-tooling)
9. [Issues hit and fixes (lessons learned)](#9-issues-hit-and-fixes)

---

## 1. Prerequisites

| Component | Verified version | Notes |
|---|---|---|
| Rust (rustup + cargo) | 1.88+ | required by `ort 2.0.0-rc.13` |
| Python | **3.11 or 3.12** | for PyO3 (`retraining`) and helper scripts. **3.13/3.14 is NOT supported by PyO3 0.21** |
| Git | any | to clone the repository |
| 4–6 GB free disk | — | build + ONNX models + runtime data |
| CPU with AVX2 | — | ONNX Runtime |

Verified tip: with **uv** (`pip install uv` or winget) you get a managed,
per-user Python 3.12 in seconds, with no administrator privileges:

```
uv python install 3.12
uv venv --python 3.12 .venv312
uv pip install --python .venv312\Scripts\python.exe torch torchvision --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv312\Scripts\python.exe pillow onnx onnxscript onnxruntime
```

> Lesson learned: **torch 2.5.1** matches the Dockerfile; newer versions
> (2.14+) export ONNX weights into a separate `.onnx.data` file, a layout the
> runtime's backup/prune logic does not handle.

## 2. Getting the code

```
git clone https://github.com/oraziog/anonimizzazione_volti.git C:\Users\Admin\Music\anonimizzazione_volti
cd C:\Users\Admin\Music\anonimizzazione_volti
```

The repository root holds the Windows operations tooling (start/stop, logs,
reports); the Rust crate lives in `src/`.

## 3. Building the service

### Base build (no retraining)

```
cd src
cargo build --release
```

Produces `src\target\release\anonimizzazione_volti.exe`.

### Build with nightly retraining (recommended in production)

```
cd src
set PYO3_PYTHON=C:\Users\Admin\AppData\Roaming\uv\python\cpython-3.12-windows-x86_64-none\python.exe
cargo build --release --features retraining
```

> Lesson learned (important): with the `retraining` feature the binary embeds
> Python and needs **both `python312.dll` AND `python3.dll`** in its own
> folder. Managed distributions (uv) do not put them on the system PATH:
>
> ```
> copy "C:\Users\Admin\AppData\Roaming\uv\python\cpython-3.12-windows-x86_64-none\python312.dll" target\release\
> copy "C:\Users\Admin\AppData\Roaming\uv\python\cpython-3.12-windows-x86_64-none\python3.dll" target\release\
> ```
>
> Symptom without the fix: the exe exits immediately with code `0xC0000135`
> (STATUS_DLL_NOT_FOUND) **printing nothing**. And without `python3.dll`
> training succeeds but export fails with `DLL load failed while importing
> onnx_cpp2py_export`.

## 4. Configuration (.env)

```
cd src
copy .env.example .env
```

Variables to set at least once:

| Variable | Purpose |
|---|---|
| `OPERATOR_API_KEY` | operator API key. **Make it strong** (e.g. 48 chars from `secrets.token_urlsafe`); the server compares it in constant time |
| `CAMERA_ID_SOURCE` | `exif` = camera identity from the EXIF body serial number (automatic filename fallback); `filename` = folder/name prefix only |
| `MODEL_CACHE_DIR` | local folder where ONNX models are downloaded |
| `DATASET_SEED_REAL_FACES_DIR` | real-face seed (see §6) |
| `DATASET_FP_DIR` | automatically collected false positives |
| `CRON_RETRAIN_SCHEDULE` | nightly retraining time (default `03:00`) |
| `RUST_LOG` | log noise: `warn,anonimizzazione_volti=info` = service INFO only; `RUST_LOG=warn,perf=info` for the minimum |
| `NO_COLOR=1` | plain-text logs, no ANSI color codes |
| `PYTHONHOME` / `PYTHONPATH` | (only for the `retraining` build) Python 3.12 distribution home and the venv `site-packages` folder |

> Lesson learned: `start-server.ps1` loads the `.env` into the session before
> launching the exe, so any new variable takes effect immediately. The `.env`
> holds secrets and is git-ignored — never commit it.

## 5. First run and verification

**Double-click** `start-server.cmd` (or from a console: `powershell
-NoProfile -ExecutionPolicy Bypass -File start-server.ps1`). The script:

1. loads the `.env` (with an automatic fallback when run from subfolders);
2. starts the exe **hidden and independent** from the console window;
3. writes logs to `logs\server.log` and `logs\server.err`
   (the previous run is kept in `server.previous.log`);
4. saves the PID to `logs\server.pid`;
5. waits until port 8080 answers and prints `Server ONLINE`.

Immediate checks:

```
curl http://localhost:8080/health
curl -X POST http://localhost:8080/anonymize -F "file=@C:\path\test.zip" -o C:\path\out.zip
```

The `X-Processing-Errors` response header reports per-image errors.
Stop: double-click `stop-server.cmd`.

> Lesson learned: on PowerShell 5.1 `Start-Process -RedirectStandardOutput`
> **hangs forever** when launched from an interactive console; that is why
> the scripts generate an intermediate `.cmd` launcher for log redirection.
> The readiness probe uses a direct TCP test, not `Invoke-WebRequest`.

## 6. Nightly retraining

The "surgical" classifier (inside the ROI it blurs only confirmed faces)
needs three ingredients:

1. **Real-face seed** — built with `src/python/prepare_seed.py` from a YOLO
   dataset or WIDER FACE (100 val images → ~646 crops). Expected paths:
   `dataset_seed/real_faces/` (faces) and the optional
   `dataset_seed/real_faces_label/faces/` for manual review.
2. **False positives** — collected **automatically** by LEARNING cameras:
   every YOLO detection with confidence in
   `[YOLO_CONF_THRESHOLD, FP_CROP_CONF_MAX)` is saved as a crop under
   `dataset_falsi_positivi/{camera}/`.
3. **The `retraining` build** (§3) plus `PYTHONHOME`/`PYTHONPATH` in the `.env`.

At `03:00` the runtime: fine-tunes MobileNetV2 (fresh head, frozen backbone)
→ exports ONNX → **double validation** (Python accuracy ≥
`RETRAIN_MIN_ACCURACY=0.85` and a Rust-side ort test, not worse than the
deployed model) → hot swap with backup → writes `data/retrain_audit.json`
(served by `GET /operator/retrain-audit`). The swapped classifier
**survives restarts**: it is restored at startup from
`data/classifier_state.json` (log: `classifier restored from persisted state`).

Live status: `GET /operator/classifier` (`loaded`, `active_check_enabled`,
accuracy, inference counters).

> Lesson learned: on the first swap all consumed false positives are
> **deleted** (the model already learned on them); the dataset regrows
> automatically over the following days. Also mind the **sequence counter**:
> before the fix now included, multiple images of one job overwrote each
> other's crops (log said "saved 1670" vs 178 on disk).

## 7. Commissioning: the camera FSM

Lifecycle of every camera (identity from the EXIF serial or the filename):

```
INITIAL → LEARNING (box + 15% blur, detection/FP collection)
        → (after LEARNING_DAYS=30, nightly job) → ROI extraction (DBSCAN → hull)
        → ACTIVE (blur only inside the ROI, confirmed by the classifier)
```

- The FSM **catches up on startup**: a camera whose window expired while the
  service was down is promoted on the first useful run
  (log: `ROI extracted (…% of frame) → ACTIVE`).
- Promotion needs enough data: a dense cluster (DBSCAN `ROI_EPS_PX=50`,
  min 15 points) and an area between `ROI_AREA_MIN` (10%) and the max.
  With too few points the camera stays in LEARNING
  (`not enough detection data`).
- ACTIVE cameras re-evaluate the ROI nightly (window
  `ROI_REEXTRACT_WINDOW_DAYS`, adopted only when stable, IoU ≥ 0.6).
- An operator reset (`POST /operator/cameras/{id}/reset`) sends a camera back
  to LEARNING; `POST /operator/cameras/{id}/roi` with `{"type":"full"}`
  forces ACTIVE full-frame (handy for tests).

Live tuning: panel at `http://localhost:8080/operator/settings` (with the
`OPERATOR_API_KEY`) — every change is immediate and persisted to
`data/runtime_config.json`.

## 8. Operations tooling

(robust on Windows, all tested; run with
`powershell -NoProfile -ExecutionPolicy Bypass -File <script>`)

| Script | Purpose |
|---|---|
| `start-server.cmd` / `stop-server.cmd` | double-click start/stop, logs and PID |
| `anonimizza-cartella.ps1` | photo folder → ZIP → `/anonymize` → anonymized archive (drag & drop too) |
| `load-env.ps1` | loads the `.env` into the session (automatic parent-folder fallback) |
| `rotate-logs.ps1` | old logs → weekly ZIP, keeps the last 4 (task `AnonVolt_RotateLogs`) |
| `riassunto-log.ps1` | ERROR + FSM transitions per day (`-Giorni`, `-Dettaglio`) |
| `report-camere.ps1` | daily report: cameras, jobs, retraining audit, classifier health, **accuracy history with a text chart** (task `AnonVolt_ReportCamere` 08:00) |
| `report-settimanale.ps1` | weekly accuracy trend in Markdown (`reports\trend-accuracy-YYYY-Www.md`, task `AnonVolt_TrendAccuracy`) |
| `alert-retraining.ps1` | if the last retraining is `failed`/`rejected`, writes `alerts\RETRAINING-ATTENTION.txt` with cause and checklist (task `AnonVolt_RetrainAlert` 04:00); self-removes on success |
| `valida-classificatore.py` | validation with certain ground truth: clear/small-face recall, background FP, per-threshold table |
| `test_zip/make_exif_test.py` | generates images with an EXIF serial to test `CAMERA_ID_SOURCE=exif` |

## 9. Issues hit and fixes

A journal of the real issues met during installation and commissioning —
if you hit a similar symptom, the cure is here.

| # | Symptom | Cause | Fix |
|---|---|---|---|
| 1 | `load-env.ps1`: ".env file not found" | it looked for `.env` in the CWD | patch: fallback to the `.env` next to the script's parent folder (included) |
| 2 | Double-click start hangs / logs not written | `Start-Process -RedirectStandardOutput` hangs from a console | intermediate `.cmd` launcher generated by `start-server.ps1` |
| 3 | `[2m [0m` sequences in log files | ANSI codes from `tracing` | `NO_COLOR=1` in the `.env` (honored by tracing-subscriber ≥ 0.3.18) |
| 4 | Exe exits instantly, code `-1073741515` (0xC0000135), no output | `python312.dll` not found | copy `python312.dll` + `python3.dll` next to the exe (see §3) |
| 5 | Retraining audit: `Module onnx is not installed!` | missing packages in the in-process venv | `uv pip install pillow onnx onnxscript` in the venv and a correct `PYTHONPATH` in the `.env` |
| 6 | ONNX export produces a separate `.onnx.data` file | torch too new | align torch to 2.5.1 (CPU index) |
| 7 | `cargo test --features retraining` fails to build (`TempDir` has no `Clone`) | tests never compiled with the feature | fix included: `dir.to_path_buf()` in `training.rs` |
| 8 | FP crops lost: log reports 1670, only 178 on disk | per-image filename index → concurrent overwrites | fix included: global atomic counter `FP_CROP_SEQ` |
| 9 | Camera stays in LEARNING: "not enough detection data" | DBSCAN finds no dense cluster | a dense, realistic band of detections is required; check `ROI_EPS_PX`/`ROI_MIN_SAMPLES` |
| 10 | Camera stays in LEARNING: "ROI validation failed (area 0.1%)" | ROI smaller than the 10% minimum | it's a guard, not a bug: collect more data or review the scene |
| 11 | Unbalanced FP class → "always negative" classifier | weeks of collected FPs dwarf the seed | already handled: `MAX_CLASS_RATIO` 4:1 cap + FP clear after the swap |
| 12 | WSL2/Docker not installable (`Feature name is unknown`) | Windows image without the virtualization packages (modified install) | `DISM /RestoreHealth` does NOT restore removed packages: a repair-install from an original ISO is required. Alternative: native build (this manual) |

## Final installation checklist

Quick checklist after the first run:

- [ ] `curl http://localhost:8080/health` → 200
- [ ] Test upload → HTTP 200, `x-processing-errors: 0`
- [ ] `GET /operator/cameras` with `OPERATOR_API_KEY` → cameras listed in LEARNING
- [ ] (with `retraining`) the log contains `classifier restored from persisted state` after a restart
- [ ] `report-camere.ps1 -Stampa` → full report without errors
- [ ] First nightly retraining: `GET /operator/retrain-audit` → `status: swapped`
