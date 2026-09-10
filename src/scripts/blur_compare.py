#!/usr/bin/env python3
"""Genera confronti visivi dei metodi di anonimizzazione su volti reali WIDER.

Per N immagini selezionate disegna, per ogni volto rilevato da YOLO, una riga con:
  [originale] [gaussiana attuale] [fill solido] [pixelate] [blur forte x4]

Salva PNG concatenati in --out (default sweep/blur_compare).
"""
import argparse
import os
import zipfile
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

ZIP = "test-wider-sample/WIDER_val.zip"


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


def sigma_for_box(w):
    return max(5.0, min(50.0, w / 4.0))


def blur_gauss(img, region, sigma):
    roi = img[region[1] : region[3], region[0] : region[2]]
    k = int(round(sigma)) | 1
    out = cv2.GaussianBlur(roi, (k, k), sigma)
    img[region[1] : region[3], region[0] : region[2]] = out
    return img


def solid_fill(img, region):
    roi = img[region[1] : region[3], region[0] : region[2]]
    img[region[1] : region[3], region[0] : region[2]] = np.full_like(roi, (120, 120, 120))
    return img


def black_fill(img, region):
    roi = img[region[1] : region[3], region[0] : region[2]]
    img[region[1] : region[3], region[0] : region[2]] = np.zeros_like(roi)
    return img


def strong_blur(img, region, sigma, passes=4):
    for _ in range(passes):
        img = blur_gauss(img, region, sigma)
    return img


def pixelate(img, region, cell=12):
    roi = img[region[1] : region[3], region[0] : region[2]]
    h, w = roi.shape[:2]
    small = cv2.resize(roi, (max(1, w // cell), max(1, h // cell)), interpolation=cv2.INTER_LINEAR)
    big = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    img[region[1] : region[3], region[0] : region[2]] = big
    return img


def annotate(img, text, color=(255, 255, 255)):
    cv2.rectangle(img, (0, 0), (img.shape[1], 20), (0, 0, 0), -1)
    cv2.putText(img, text, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default=ZIP)
    ap.add_argument("--face", default="models_cache/yolov8m-face-lindevs.onnx")
    ap.add_argument("--conf", type=float, default=0.02)
    ap.add_argument("--images", nargs="+", default=None,
                    help="nomi immagine (solo la parte dopo /). default: prime 4 con volti")
    ap.add_argument("--out", default="sweep/blur_compare")
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    sess = ort.InferenceSession(args.face, providers=["CPUExecutionProvider"])

    z = zipfile.ZipFile(args.zip)
    names = sorted(n for n in z.namelist() if n.lower().endswith(".jpg"))

    if args.images:
        selected = [n for n in names if any(i in n for i in args.images)][: len(args.images)]
    else:
        selected = []
        for n in names:
            img = cv2.imdecode(np.frombuffer(z.read(n), np.uint8), cv2.IMREAD_COLOR)
            dets = nms(run_face(sess, img, args.conf))
            if len([d for d in dets if (d[2] - d[0]) >= 24]) >= 3:
                selected.append(n)
            if len(selected) >= 4:
                break

    for n in selected:
        img = cv2.imdecode(np.frombuffer(z.read(n), np.uint8), cv2.IMREAD_COLOR)
        dets = nms(run_face(sess, img, args.conf))
        faces = sorted([d for d in dets if (d[2] - d[0]) >= 24], key=lambda d: -(d[4]))
        if not faces:
            continue
        rows = []
        for (x0, y0, x1, y1, conf) in faces[:4]:
            sigma = sigma_for_box(x1 - x0)
            rect = (int(x0), int(y0), int(x1), int(y1))
            pad = int((x1 - x0) * 0.4)
            zx0 = max(0, int(x0) - pad); zy0 = max(0, int(y0) - pad)
            zx1 = min(img.shape[1], int(x1) + pad); zy1 = min(img.shape[0], int(y1) + pad)
            makers = (
                ("originale", lambda v, r: v, (0, 255, 0)),
                ("pixelate cell 6", lambda v, r: pixelate(v, r, 6), (255, 255, 255)),
                ("pixelate cell 8", lambda v, r: pixelate(v, r, 8), (255, 255, 255)),
                ("pixelate cell 10", lambda v, r: pixelate(v, r, 10), (255, 255, 255)),
                ("pixelate cell 12", lambda v, r: pixelate(v, r, 12), (255, 255, 255)),
                ("pixelate cell 16", lambda v, r: pixelate(v, r, 16), (255, 255, 255)),
            )
            variants = []
            for label, maker, color in makers:
                v = maker(img.copy(), rect)
                crop = v[zy0:zy1, zx0:zx1]
                target = 200
                scale = target / crop.shape[0]
                crop = cv2.resize(crop, (int(crop.shape[1] * scale), target), interpolation=cv2.INTER_LINEAR)
                variants.append(annotate(crop, label, color))
            row = np.hstack([pad_border(v, 4) for v in variants])
            rows.append(row)
        wmax = max(r.shape[1] for r in rows)
        rows = [pad_border(r, 0) if r.shape[1] == wmax else np.hstack([r, np.full((r.shape[0], wmax - r.shape[1], 3), 20, np.uint8)]) for r in rows]
        canvas = np.vstack([pad_border(r, 6) for r in rows])
        fn = outdir / (n.split("/")[-1].replace(".jpg", "") + "_compare.png")
        cv2.imwrite(str(fn), canvas)
        print(f"Saved {fn}  ({len(faces)} volti | sigma@32px={sigma_for_box(32.0):.0f}")
    z.close()


def pad_border(img, px, color=(0, 0, 0)):
    return cv2.copyMakeBorder(img, px, px, px, px, cv2.BORDER_CONSTANT, value=color)


if __name__ == "__main__":
    main()