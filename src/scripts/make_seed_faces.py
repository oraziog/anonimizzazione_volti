#!/usr/bin/env python3
"""Generates the real-face seed for the binary classifier retraining.

Extracts face crops from a WIDER_val ZIP using the same YOLOv8-Face model the
service uses in production (models_cache/yolov8m-face-lindevs.onnx), applying
a HIGH confidence threshold so the seed contains as few false faces as
possible. The crop and preprocessing match the Rust pipeline
(crop_clamped → 224×224, the classifier transform then handles normalization).

Usage:
    .venv-train/Scripts/python.exe scripts/make_seed_faces.py \
        --zip test-wider-sample/WIDER_val.zip \
        --face models_cache/yolov8m-face-lindevs.onnx \
        --out dataset_seed/real_faces [--max-faces 400] [--conf 0.6]
"""
import argparse
import io
import os
import random
import zipfile

import cv2
import numpy as np
import onnxruntime as ort


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
    """YOLOv8-Face lindevs: output [1, 5, 8400] = cx,cy,w,h,conf."""
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
        x1 = min(x1, img.shape[1])
        y1 = min(y1, img.shape[0])
        if x1 - x0 < 24 or y1 - y0 < 24:  # faces that are too small → noise
            continue
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default="test-wider-sample/WIDER_val.zip")
    ap.add_argument("--face", default="models_cache/yolov8m-face-lindevs.onnx")
    ap.add_argument("--out", default="dataset_seed/real_faces")
    ap.add_argument("--conf", type=float, default=0.6,
                    help="HIGH face confidence threshold (clean seed)")
    ap.add_argument("--max-faces", type=int, default=400)
    ap.add_argument("--max-images", type=int, default=0,
                    help="cap on images examined (0 = all)")
    ap.add_argument("--per-image", type=int, default=4,
                    help="max faces taken from a single image (diversity)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    sess = ort.InferenceSession(args.face, providers=["CPUExecutionProvider"])
    z = zipfile.ZipFile(args.zip)
    names = [n for n in z.namelist() if n.lower().endswith(".jpg")]
    if args.max_images:
        names = names[: args.max_images]
    rng = random.Random(42)

    saved = 0
    for i, name in enumerate(names):
        if saved >= args.max_faces:
            break
        img = cv2.imdecode(np.frombuffer(z.read(name), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        dets = nms(run_face(sess, img, args.conf))
        rng.shuffle(dets)
        for (x0, y0, x1, y1, conf) in dets[: args.per_image]:
            if saved >= args.max_faces:
                break
            crop = img[int(y0) : int(y1), int(x0) : int(x1)]
            if crop.size == 0 or crop.shape[0] < 24 or crop.shape[1] < 24:
                continue
            fn = os.path.join(
                args.out, f"seed_{saved:05d}_conf{conf:.2f}.jpg"
            )
            cv2.imwrite(fn, crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
            saved += 1
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(names)}] faces salvate: {saved}", flush=True)

    print(f"completato: {saved} volti reali in {args.out}")


if __name__ == "__main__":
    main()
