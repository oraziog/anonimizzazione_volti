#!/usr/bin/env python3
"""Prepare the classifier seed dataset (spec §6 open point 1).

Takes a face-detection dataset and crops every annotated face box, with a
configurable margin, into `DATASET_SEED_REAL_FACES_DIR` (default
`dataset_seed/real_faces/`), ready for `python/retrain.py`.

Why crops and not full photos: `retrain.py` uses every file under the seed dir
as one 224×224 training sample (class 1 = real face), so tight face crops are
exactly what it needs.

Two data sources are supported:

1. YOLO-format dataset (e.g. the Kaggle "Face Detection Dataset", CC0 1.0):
   image plus a same-stem `.txt` label with lines `class_id cx cy w h`
   (normalized or absolute).

2. WIDER FACE validation set (`--wider-root` / `--wider-download`):
   images + `wider_face_val_bbx_gt.txt`, cropped with the same difficulty
   rule used by the `eval-wider` harness (`easy`/`medium`/`hard`, per-face
   attributes). By default only `easy` + `medium` faces are kept and the
   `ignore == 1` faces are skipped — the tiny/blurry `hard` crop would poison
   the classifier seed.

Usage:
    # 1. YOLO dataset already downloaded from Kaggle (extracted):
    python prepare_seed.py --source path/to/face-dataset \
        [--out dataset_seed/real_faces] [--margin 0.10] [--min-side 24] \
        [--max-crops 5000] [--class-id 0]

    # 2. WIDER FACE already downloaded + extracted (WIDER_val/ + wider_face_split/):
    python prepare_seed.py --wider-root path/to/wider \
        [--min-difficulty medium] [--max-images 0]

    # 3. WIDER FACE auto-download (images ~365 MB from Google Drive + GT):
    python prepare_seed.py --wider-download path/to/download_dir \
        [--max-images 0]

Coordinates (YOLO mode): label values are treated as *normalized*
(cx cy w h in [0,1]) and scaled by the image size; absolute pixel coordinates
are auto-detected when any value exceeds 1.5. Boxes are expanded by `margin`
(fraction of the box's own width/height), clamped to the image, and saved as
quality-90 JPEGs with a deterministic, collision-free naming scheme.
Identical crops (same pixels) are deduplicated.
"""

import argparse
import hashlib
import os
import re
import sys
import urllib.request
import zipfile

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
LABEL_EXT = ".txt"

# Official WIDER FACE files (same IDs as torchvision's WIDERFace dataset).
WIDER_VAL_DRIVE_ID = "1GUCogbp16PMGa39thoMMeWxp7Rp5oM8Q"
WIDER_VAL_FILENAME = "WIDER_val.zip"
WIDER_GT_URL = "http://shuoyang1213.me/WIDERFACE/support/bbx_annotation/wider_face_split.zip"
WIDER_GT_FILENAME = "wider_face_split.zip"

_DIFFICULTY_RANK = {"easy": 0, "medium": 1, "hard": 2}


def _collect_pairs(root):
    """Returns sorted [(image_path, label_path)] pairs found under root."""
    pairs = []
    for dirpath, _dirnames, filenames in os.walk(root):
        by_stem = {}
        for name in filenames:
            stem, ext = os.path.splitext(name)
            by_stem.setdefault(stem, {})[ext.lower()] = os.path.join(dirpath, name)
        for stem, files in by_stem.items():
            label = files.get(LABEL_EXT)
            image = next((files[e] for e in IMAGE_EXTS if e in files), None)
            if label and image:
                pairs.append((image, label))
    return sorted(pairs)


def _parse_labels(label_path, img_w, img_h, class_id):
    """Yields (x1, y1, x2, y2) pixel boxes for the requested class.

    Auto-detects normalized vs absolute coordinates. Lines with < 5 fields,
    comments or unknown classes are skipped.
    """
    try:
        with open(label_path, "r", encoding="utf-8", errors="replace") as fh:
            lines = fh.read().splitlines()
    except OSError as exc:
        print("  warn: cannot read %s: %s" % (label_path, exc))
        return
    normalized = None  # None = undecided, True/False once a box is seen
    for raw in lines:
        parts = raw.split()
        if len(parts) < 5:
            continue
        try:
            values = [float(p) for p in parts[:5]]
        except ValueError:
            continue
        if int(values[0]) != class_id:
            continue
        if normalized is None:
            normalized = max(abs(v) for v in values[1:]) <= 1.5
        if normalized:
            cx, cy, w, h = values[1:]
            x1 = (cx - w / 2.0) * img_w
            y1 = (cy - h / 2.0) * img_h
            x2 = (cx + w / 2.0) * img_w
            y2 = (cy + h / 2.0) * img_h
        else:
            cx, cy, w, h = values[1:]
            x1, y1, x2, y2 = cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0
        yield x1, y1, x2, y2


# ── WIDER FACE support ───────────────────────────────────────────────────────


def _wider_difficulty(w, h, blur, illum, occ, pose):
    """Same per-face difficulty rule as the `eval-wider` harness (Rust)."""
    min_side = min(w, h)
    if min_side >= 60.0 and blur == 0 and illum == 0 and occ == 0 and pose == 0:
        return "easy"
    if min_side < 30.0 or blur == 2 or occ == 2:
        return "hard"
    return "medium"


def _parse_wider_gt(gt_path):
    """Parses wider_face_val_bbx_gt.txt → {image_name: [box records]}.

    Each record: dict(x1, y1, x2, y2, difficulty, ignored).
    """
    images = {}
    with open(gt_path, "r", encoding="utf-8", errors="replace") as fh:
        lines = [l.strip() for l in fh if l.strip()]
    i = 0
    while i < len(lines):
        name = lines[i]
        i += 1
        if i >= len(lines):
            break
        try:
            n = int(lines[i])
        except ValueError:
            print("  warn: bad face count %r in %s" % (lines[i], gt_path))
            break
        i += 1
        boxes = []
        for _ in range(n):
            if i >= len(lines):
                break
            f = lines[i].split()
            i += 1
            if len(f) < 4:
                print("  warn: short GT line %r" % (lines[i - 1],))
                continue
            try:
                x1, y1, w, h = [float(v) for v in f[:4]]
                blur = float(f[4]) if len(f) > 4 else 0.0
                illum = float(f[6]) if len(f) > 6 else 0.0
                occ = float(f[7]) if len(f) > 7 else 0.0
                pose = float(f[8]) if len(f) > 8 else 0.0
                ignored = len(f) > 9 and f[9] == "1"
            except ValueError:
                continue
            boxes.append({
                "x1": x1, "y1": y1, "x2": x1 + w, "y2": y1 + h,
                "difficulty": _wider_difficulty(w, h, blur, illum, occ, pose),
                "ignored": ignored,
            })
        images[name] = boxes
    return images


def _download(url, dest, label):
    """Best-effort download with a browser User-Agent and resume on nothing."""
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print("%s already present: %s (%.1f MB)" % (
            label, dest, os.path.getsize(dest) / 1e6))
        return dest
    print("downloading %s → %s" % (label, dest))
    req = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0 (prepare_seed.py)"})
    tmp = dest + ".part"
    try:
        with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as fh:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
    except Exception as exc:  # noqa: BLE001
        if os.path.exists(tmp):
            os.remove(tmp)
        raise RuntimeError("%s download failed: %s" % (label, exc))
    os.replace(tmp, dest)
    print("downloaded %s (%.1f MB)" % (label, os.path.getsize(dest) / 1e6))
    return dest


def _download_wider(root):
    """Downloads WIDER_val.zip (Google Drive) + wider_face_split.zip."""
    os.makedirs(root, exist_ok=True)
    # Google Drive needs the usercontent endpoint (plain uc?id= returns a
    # virus-scan HTML page for large files).
    val_url = ("https://drive.usercontent.google.com/download?id=%s"
               "&export=download&confirm=t" % WIDER_VAL_DRIVE_ID)
    _download(val_url, os.path.join(root, WIDER_VAL_FILENAME), "WIDER_val.zip")
    _download(WIDER_GT_URL, os.path.join(root, WIDER_GT_FILENAME), "wider_face_split.zip")


def _extract_zip(zip_path, root):
    out_dir = os.path.join(root, os.path.splitext(os.path.basename(zip_path))[0])
    if os.path.isdir(out_dir):
        print("%s already extracted at %s" % (zip_path, out_dir))
        return out_dir
    print("extracting %s ..." % zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(root)
    return out_dir


def _wider_image_boxes(gt_map, min_difficulty, include_ignored):
    """Yields (image_abs_path, [boxes]) for WIDER images, GT-filtered."""
    images_dir = gt_map["images_dir"]
    for name, boxes in gt_map["images"].items():
        kept = []
        for b in boxes:
            if b["ignored"] and not include_ignored:
                continue
            if _DIFFICULTY_RANK[b["difficulty"]] < _DIFFICULTY_RANK[min_difficulty]:
                continue
            kept.append(b)
        if not kept:
            continue
        yield os.path.join(images_dir, name), kept


# ── Shared crop saving ───────────────────────────────────────────────────────


def _save_crops(rgb, boxes, out_dir, seen_hashes, counter, saved, skipped,
                args, prefix):
    """Crops+dedups+saves; returns (counter, saved, skipped, boxes_used)."""
    w, h = rgb.size
    boxes_used = 0
    for x1, y1, x2, y2 in boxes:
        if args.max_crops and saved >= args.max_crops:
            break
        bw, bh = x2 - x1, y2 - y1
        if bw <= 0 or bh <= 0:
            continue
        pad_x, pad_y = bw * args.margin, bh * args.margin
        cx0 = max(0, int(round(x1 - pad_x)))
        cy0 = max(0, int(round(y1 - pad_y)))
        cx1 = min(w, int(round(x2 + pad_x)))
        cy1 = min(h, int(round(y2 + pad_y)))
        if cx1 - cx0 < args.min_side or cy1 - cy0 < args.min_side:
            continue
        crop = rgb.crop((cx0, cy0, cx1, cy1))
        digest = hashlib.sha1(crop.tobytes()).hexdigest()
        if digest in seen_hashes:
            continue
        seen_hashes.add(digest)
        out_name = "%s_%05d.jpg" % (prefix, counter)
        crop.save(os.path.join(out_dir, out_name), "JPEG", quality=args.jpeg_quality)
        counter += 1
        saved += 1
        boxes_used += 1
        if saved % 250 == 0:
            print("  ... %d crops saved" % saved)
    return counter, saved, skipped, boxes_used


def _run_yolo(args, out_dir):
    pairs = _collect_pairs(args.source)
    if not pairs:
        sys.exit("no image+label pairs found under %s" % args.source)
    print("found %d image+label pairs under %s" % (len(pairs), args.source))
    saved = skipped = counter = total_boxes = 0
    seen_hashes = set()
    for image_path, label_path in pairs:
        if args.max_crops and saved >= args.max_crops:
            break
        try:
            with Image.open(image_path) as im:
                rgb = im.convert("RGB")
        except Exception as exc:  # noqa: BLE001
            print("  warn: cannot open %s: %s" % (image_path, exc))
            skipped += 1
            continue
        w, h = rgb.size
        boxes = list(_parse_labels(label_path, w, h, args.class_id))
        total_boxes += len(boxes)
        stem = re.sub(r"[^A-Za-z0-9._-]", "_", os.path.splitext(os.path.basename(image_path))[0])
        counter, saved, skipped, _ = _save_crops(
            rgb, boxes, out_dir, seen_hashes, counter, saved, skipped, args,
            "%s_%s" % (args.prefix, stem))
    print("done: %d crops saved to %s (from %d boxes in %d images, %d images "
          "skipped/unreadable)" % (saved, out_dir, total_boxes, len(pairs),
                                   skipped))
    if saved == 0:
        print("note: nothing was saved — check --source layout, --class-id "
              "and --min-side")


def _run_wider(args, out_dir):
    try:
        from PIL import Image
    except ImportError:
        sys.exit("Pillow is required: pip install pillow")
    if args.wider_download:
        _download_wider(args.wider_download)
        root = args.wider_download
    else:
        root = args.wider_root
    images_dir = os.path.join(root, "WIDER_val", "images")
    gt_path = os.path.join(root, "wider_face_split", "wider_face_val_bbx_gt.txt")
    if not os.path.isdir(images_dir):
        # Fallback: the zip is present but not extracted yet.
        val_zip = os.path.join(root, WIDER_VAL_FILENAME)
        if os.path.isfile(val_zip):
            images_dir = _extract_zip(val_zip, root)
            images_dir = os.path.join(images_dir, "images")
    if not os.path.isfile(gt_path):
        gt_zip = os.path.join(root, WIDER_GT_FILENAME)
        if os.path.isfile(gt_zip):
            _extract_zip(gt_zip, root)
    if not os.path.isdir(images_dir):
        sys.exit("cannot find WIDER images: expected %s" % images_dir)
    if not os.path.isfile(gt_path):
        sys.exit("cannot find WIDER GT: expected %s" % gt_path)

    gt_map = {"images_dir": images_dir, "images": _parse_wider_gt(gt_path)}
    print("WIDER GT: %d images, filter min-difficulty=%s include-ignored=%s"
          % (len(gt_map["images"]), args.min_difficulty, args.include_ignored))

    saved = skipped = counter = 0
    seen_hashes = set()
    kept_by_diff = {"easy": 0, "medium": 0, "hard": 0}
    total_boxes = 0
    n_images = 0
    for image_path, boxes in _wider_image_boxes(
            gt_map, args.min_difficulty, args.include_ignored):
        if args.max_images and n_images >= args.max_images:
            break
        n_images += 1
        try:
            with Image.open(image_path) as im:
                rgb = im.convert("RGB")
        except Exception as exc:  # noqa: BLE001
            print("  warn: cannot open %s: %s" % (image_path, exc))
            skipped += 1
            continue
        total_boxes += len(boxes)
        for b in boxes:
            kept_by_diff[b["difficulty"]] += 1
        stem = re.sub(r"[^A-Za-z0-9._-]", "_", image_path)
        counter, saved, skipped, _ = _save_crops(
            rgb, [(b["x1"], b["y1"], b["x2"], b["y2"]) for b in boxes],
            out_dir, seen_hashes, counter, saved, skipped, args,
            "%s_wider_%s" % (args.prefix, stem))
        if n_images % 250 == 0:
            print("  ... %d images processed, %d crops saved" % (n_images, saved))
    print("done: %d crops saved to %s (from %d boxes in %d images, %d images "
          "skipped/unreadable)" % (saved, out_dir, total_boxes, n_images,
                                   skipped))
    print("kept faces by difficulty: easy=%d medium=%d hard=%d"
          % (kept_by_diff["easy"], kept_by_diff["medium"], kept_by_diff["hard"]))
    if saved == 0:
        print("note: nothing was saved — check --min-difficulty, "
              "--include-ignored and --min-side")


def main():
    from PIL import Image  # noqa: F401  (imported early: required by both modes)
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--source", metavar="DIR",
                     help="YOLO-format dataset root (folders scanned recursively)")
    src.add_argument("--kaggle-slug", metavar="OWNER/DATASET",
                     help="download via kagglehub first (best-effort)")
    src.add_argument("--wider-root", metavar="DIR",
                     help="WIDER FACE root containing WIDER_val/ and wider_face_split/")
    src.add_argument("--wider-download", metavar="DIR",
                     help="auto-download WIDER FACE val (images + GT) into DIR")
    ap.add_argument("--out", metavar="DIR",
                    default=os.environ.get("DATASET_SEED_REAL_FACES_DIR",
                                           "dataset_seed/real_faces"),
                    help="output dir (default: dataset_seed/real_faces or "
                         "DATASET_SEED_REAL_FACES_DIR)")
    ap.add_argument("--margin", type=float, default=0.10,
                    help="expand each box by this fraction of its own size (0.10 = 10%%)")
    ap.add_argument("--min-side", type=float, default=24.0,
                    help="skip crops whose smallest side is below this (px)")
    ap.add_argument("--max-crops", type=int, default=5000,
                    help="stop after this many saved crops (0 = unlimited)")
    ap.add_argument("--class-id", type=int, default=0,
                    help="YOLO class id to crop (default 0 = face)")
    ap.add_argument("--jpeg-quality", type=int, default=90)
    ap.add_argument("--prefix", default="seed",
                    help="filename prefix for saved crops")
    # WIDER-only options.
    ap.add_argument("--min-difficulty", choices=["easy", "medium", "hard"],
                    default="medium",
                    help="keep faces at least this difficult (default: easy+medium; "
                         "hard faces are tiny/blurry and poison the seed)")
    ap.add_argument("--include-ignored", action="store_true",
                    help="also crop faces marked ignore==1 in the WIDER GT")
    ap.add_argument("--max-images", type=int, default=0,
                    help="WIDER: process at most N images (0 = all)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    if args.kaggle_slug:
        try:
            import kagglehub
        except ImportError:
            sys.exit("--kaggle-slug needs kagglehub: pip install kagglehub")
        print("downloading %s via kagglehub ..." % args.kaggle_slug)
        try:
            args.source = kagglehub.dataset_download(args.kaggle_slug)
        except Exception as exc:  # noqa: BLE001 - surface the real error
            sys.exit("kagglehub download failed (%s) — download the dataset "
                     "from Kaggle in the browser and use --source instead" % exc)
        print("dataset at %s" % args.source)

    if args.wider_root or args.wider_download:
        _run_wider(args, args.out)
    else:
        _run_yolo(args, args.out)


if __name__ == "__main__":
    main()