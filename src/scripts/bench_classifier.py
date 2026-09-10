#!/usr/bin/env python3
"""Quantitative benchmark YOLO-only vs YOLO+classifier on a WIDER_val ZIP.

Replicates the Rust ACTIVE pipeline (crop_clamped -> classifier, threshold
p_face >= 0.5, fail-safe blur on error) on the YOLOv8-Face detections:

  - YOLO solo:   every YOLO detection is blurred
  - YOLO+CLF:    a detection is blurred only if the classifier confirms it
                 (p_face >= 0.5); classifier errors count as "blur anyway"

Ground truth for residual false positives / missed faces: faces confirmed
when re-running YOLO at a very LOW confidence (0.001) + IoU matching —
i.e. we measure how many low-confidence YOLO faces (the probable FP pool)
survive each configuration, and whether any high-confidence face is lost.

Usage:
    .venv-train/Scripts/python.exe scripts/bench_classifier.py \
        --zip test-wider-sample/WIDER_val.zip \
        --face models_cache/yolov8m-face-lindevs.onnx \
        --clf models_cache/classifier_manual.onnx \
        --conf 0.02
"""
import argparse
import io
import time
import zipfile

import cv2
import numpy as np
import onnxruntime as ort

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def letterbox(img, size=640):
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    px, py = (size - nw) // 2, (size - nh) // 2
    canvas[py : py + nh, px : px + nw] = resized
    return canvas, scale, px, py


def run_face(sess, img, conf_threshold):
    lb, scale, px, py = letterbox(img)
    blob = lb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    out = sess.run(None, {sess.get_inputs()[0].name: blob})[0][0]
    dets = []
    for a in range(out.shape[1]):
        cx, cy, w, h, conf = out[0, a], out[1, a], out[2, a], out[3, a], out[4, a]
        if conf < conf_threshold or w <= 0 or h <= 0:
            continue
        x0 = (cx - w / 2 - px) / scale
        y0 = (cy - h / 2 - py) / scale
        x1 = (cx + w / 2 - px) / scale
        y1 = (cy + h / 2 - py) / scale
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, img.shape[1]), min(y1, img.shape[0])
        dets.append((x0, y0, x1, y1, conf))
    return dets


def iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def nms(dets, thr=0.45):
    kept = []
    for d in sorted(dets, key=lambda d: -d[4]):
        if all(iou(d, k) <= thr for k in kept):
            kept.append(d)
    return kept


def crop_clamped(img, det):
    x0, y0, x1, y1 = det[:4]
    return img[int(y0) : int(y1), int(x0) : int(x1)]


def classifier_confirm(clf_sess, crop):
    """Replicates build_classifier_input + run_classifier (threshold 0.5).

    Returns (confirmed, errored).
    """
    if crop.size == 0:
        return True, True  # fail-safe: blur anyway
    try:
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
        x = resized.astype(np.float32) / 255.0
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        x = x.transpose(2, 0, 1)[None]  # 1,3,224,224
        out = clf_sess.run(None, {clf_sess.get_inputs()[0].name: x})[0][0]
        # softmax2(logits) -> (p_fp, p_face); confirmed = p_face >= 0.5
        e = np.exp(out - out.max())
        s = e / e.sum()
        return s[1] >= 0.5, False
    except Exception:
        return True, True  # fail-safe: blur anyway


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default="test-wider-sample/WIDER_val.zip")
    ap.add_argument("--face", default="models_cache/yolov8m-face-lindevs.onnx")
    ap.add_argument("--clf", default="models_cache/classifier_manual.onnx")
    ap.add_argument("--conf", type=float, default=0.02,
                    help="YOLO_CONF_THRESHOLD_ACTIVE")
    ap.add_argument("--gt-conf", type=float, default=0.001,
                    help="low-conf pass used as FP-pool reference")
    ap.add_argument("--max-images", type=int, default=0)
    args = ap.parse_args()

    face_sess = ort.InferenceSession(args.face, providers=["CPUExecutionProvider"])
    clf_sess = ort.InferenceSession(args.clf, providers=["CPUExecutionProvider"])

    z = zipfile.ZipFile(args.zip)
    names = sorted(n for n in z.namelist() if n.lower().endswith(".jpg"))
    if args.max_images:
        names = names[: args.max_images]

    # Metrics
    yolo_only_blurs = 0
    yolo_clf_blurs = 0
    clf_rejected = 0
    clf_errors = 0
    # vs low-conf reference pool (probable FPs at low conf):
    gt_faces_total = 0          # reference faces (low-conf YOLO, matched to 0.02 dets or standalone)
    fp_pool_matched = 0         # reference dets NOT confirmed at 0.02 (the FP pool)
    fp_pool_blurred_yolo = 0    # of those, blurred by YOLO-only
    fp_pool_blurred_clf = 0     # of those, blurred by YOLO+CLF
    missed_yolo = 0             # reference faces missed by YOLO-only
    missed_clf = 0              # reference faces missed by YOLO+CLF
    detect_ms = 0.0
    clf_ms = 0.0

    for idx, name in enumerate(names):
        img = cv2.imdecode(np.frombuffer(z.read(name), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue

        t0 = time.perf_counter()
        dets = nms(run_face(face_sess, img, args.conf))
        t1 = time.perf_counter()
        detect_ms += (t1 - t0) * 1000

        # Reference pool at very low conf (probable-FP pool superset).
        ref = nms(run_face(face_sess, img, args.gt_conf))
        ref_matched = []
        for r in ref:
            best = max(
                (iou(r, d) for d in dets), default=0.0
            )
            if best > 0.45:
                ref_matched.append(r)
        # Reference faces = reference dets matching a 0.02 detection
        # (high recall assumption) — used for missed-face accounting.
        gt_faces_total += len(ref_matched)

        # Which reference dets are NOT in the 0.02 set → the FP pool
        # (YOLO at 0.02 already drops them; they are the faces a *lower*
        # threshold would add). For FP accounting we instead use the dets
        # at 0.02 with LOW confidence (< 0.10 = YOLO_CONF_THRESHOLD default),
        # which production treats as dubious (FP_CROP_CONF_MAX logic).
        dubious = [d for d in dets if d[4] < 0.10]
        fp_pool_matched += len(dubious)
        fp_pool_blurred_yolo += len(dubious)  # YOLO-only blurs everything

        yolo_only_blurs += len(dets)

        confirmed = 0
        for d in dets:
            crop = crop_clamped(img, d)
            t2 = time.perf_counter()
            ok, err = classifier_confirm(clf_sess, crop)
            t3 = time.perf_counter()
            clf_ms += (t3 - t2) * 1000
            if err:
                clf_errors += 1
            if ok:
                confirmed += 1
                if d[4] < 0.10:
                    fp_pool_blurred_clf += 1
            else:
                clf_rejected += 1
        yolo_clf_blurs += confirmed

        # Missed faces: reference-matched faces not covered by kept dets.
        kept_yolo = dets
        kept_clf = [d for d in dets if classifier_confirm(clf_sess, crop_clamped(img, d))[0]]
        for r in ref_matched:
            if all(iou(r, d) <= 0.45 for d in kept_yolo):
                missed_yolo += 1
            if all(iou(r, d) <= 0.45 for d in kept_clf):
                missed_clf += 1

        if (idx + 1) % 200 == 0:
            print(f"  [{idx+1}/{len(names)}] yolo_blurs={yolo_only_blurs} "
                  f"clf_blurs={yolo_clf_blurs} rifiuti={clf_rejected}", flush=True)

    n = max(len(names), 1)
    print()
    print(f"immagini: {len(names)}  (conf={args.conf})")
    print()
    print("── Regioni sfocate ──")
    print(f"YOLO solo:            {yolo_only_blurs}")
    print(f"YOLO+classificatore:  {yolo_clf_blurs}  "
          f"({100*(yolo_only_blurs-yolo_clf_blurs)/max(yolo_only_blurs,1):.1f}% in meno)")
    print(f"rifiuti classificatore: {clf_rejected}  errori (fail-safe blur): {clf_errors}")
    print()
    print("── FP potenziali (dubbi, conf < 0.10) ──")
    print(f"pool FP:                 {fp_pool_matched}")
    print(f"sfocati da YOLO solo:    {fp_pool_blurred_yolo} (tutti)")
    print(f"sfocati da YOLO+CLF:     {fp_pool_blurred_clf}  "
          f"({100*(fp_pool_blurred_yolo-fp_pool_blurred_clf)/max(fp_pool_matched,1):.1f}% filtrati)")
    print()
    print("── Volti persi (rif. low-conf) ──")
    print(f"volti riferimento:      {gt_faces_total}")
    print(f"persi YOLO solo:        {missed_yolo}")
    print(f"persi YOLO+CLF:         {missed_clf}")
    print()
    print("── Tempi ──")
    print(f"detector:  {detect_ms/n:.1f} ms/immagine")
    print(f"classifier: {clf_ms/max(yolo_only_blurs,1):.1f} ms/detection "
          f"({clf_ms/n:.1f} ms/immagine)")


if __name__ == "__main__":
    main()
