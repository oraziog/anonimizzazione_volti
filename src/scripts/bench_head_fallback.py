#!/usr/bin/env python3
"""Benchmark del fallback testa/persona (HEAD_FALLBACK_ENABLED).

Confronta, sulle immagini di test (WIDER_val sample, 305 jpg), la pipeline
ACTIVE con il solo detector facciale YOLOv8-Face vs YOLO + fallback COCO:

  - per ogni immagine: YOLO facciale (soglia = YOLO_CONF_THRESHOLD_ACTIVE)
    -> se 0 volti rilevati, COCO person detector -> sfoca la parte superiore
       (HEAD_FALLBACK_FRACTION) di ogni box persona (regione testa).

Metriche:
  - immagini con >=1 volto YOLO (blur facciale "prima")
  - immagini senza volti YOLO che scatenano il fallback (blur teste "in più")
  - totale regioni sfocate prima vs dopo
  - distribuzione teste per immagine nel fallback

Uso:
    python scripts/bench_head_fallback.py \
        --zip test-wider-sample/WIDER_val.zip \
        --face models_cache/yolov8m-face-lindevs.onnx \
        --coco models_cache/yolov8n-coco.onnx \
        --conf 0.02 --frac 0.30 [--max-images N]
"""
import argparse
import io
import sys
import zipfile

import cv2
import numpy as np
import onnxruntime as ort


def letterbox(img, size=640):
    """Letterbox centrato, identico a `build_yolo_input` del codice Rust."""
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 0, dtype=np.uint8)
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
        dets.append((max(x0, 0), max(y0, 0), min(x1, img.shape[1]), min(y1, img.shape[0]), conf))
    return dets


def run_persons(sess, img, conf_threshold):
    """YOLOv8n COCO: output [1, 84, 8400] = cx,cy,w,h + 80 classi; classe 0 = person."""
    lb, scale, px, py = letterbox(img)
    blob = lb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    out = sess.run(None, {sess.get_inputs()[0].name: blob})[0][0]
    dets = []
    for a in range(out.shape[1]):
        conf = out[4, a]  # classe 0 (person)
        cx, cy, w, h = out[0, a], out[1, a], out[2, a], out[3, a]
        if conf < conf_threshold or w <= 0 or h <= 0:
            continue
        x0 = (cx - w / 2 - px) / scale
        y0 = (cy - h / 2 - py) / scale
        x1 = (cx + w / 2 - px) / scale
        y1 = (cy + h / 2 - py) / scale
        dets.append((max(x0, 0), max(y0, 0), min(x1, img.shape[1]), min(y1, img.shape[0]), conf))
    return dets


def nms(dets, iou_thresh=0.45):
    dets = sorted(dets, key=lambda d: -d[4])
    kept = []
    for d in dets:
        x0, y0, x1, y1 = d[:4]
        if any(_iou(d, k) > iou_thresh for k in kept):
            continue
        kept.append(d)
    return kept


def _iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default="test-wider-sample/WIDER_val.zip")
    ap.add_argument("--face", default="models_cache/yolov8m-face-lindevs.onnx")
    ap.add_argument("--coco", default="models_cache/yolov8n-coco.onnx")
    ap.add_argument("--conf", type=float, default=0.02)
    ap.add_argument("--frac", type=float, default=0.30)
    ap.add_argument("--max-images", type=int, default=0)
    args = ap.parse_args()

    face_sess = ort.InferenceSession(args.face, providers=["CPUExecutionProvider"])
    coco_sess = ort.InferenceSession(args.coco, providers=["CPUExecutionProvider"])

    z = zipfile.ZipFile(args.zip)
    names = [n for n in z.namelist() if n.lower().endswith(".jpg")]
    if args.max_images:
        names = names[: args.max_images]

    total_faces = 0        # regioni sfocate dal solo YOLO ("prima")
    total_heads = 0        # teste aggiunte dal fallback COCO ("dopo" - "prima")
    fb_images = 0          # immagini che hanno attivato il fallback
    with_faces = 0         # immagini con >=1 volto YOLO
    fb_dist = {}           # teste per immagine nel fallback

    for i, name in enumerate(names):
        raw = z.read(name)
        arr = np.frombuffer(raw, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            continue
        faces = nms(run_face(face_sess, img, args.conf))
        if faces:
            with_faces += 1
            total_faces += len(faces)
        else:
            # Fallback COCO: teste = parte superiore (HEAD_FALLBACK_FRACTION)
            persons = nms(run_persons(coco_sess, img, args.conf))
            heads = sum(1 for p in persons if (p[3] - p[1]) > 0)  # tutte le persone -> testa
            if persons:
                fb_images += 1
                total_heads += heads
                fb_dist[heads] = fb_dist.get(heads, 0) + 1
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(names)}] face_imgs={with_faces} fb_imgs={fb_images}", flush=True)

    print()
    print(f"immagini: {len(names)}  (conf={args.conf}, frac={args.frac})")
    print(f"prima (solo YOLO):      {with_faces} immagini con volto, {total_faces} regioni sfocate")
    print(f"dopo (YOLO+COCO):       {with_faces + fb_images} immagini con blur")
    print(f"fallback COCO attivato: {fb_images} immagini ({100*fb_images/max(len(names),1):.1f}%)")
    print(f"teste aggiunte:         {total_heads} (in media {total_heads/max(fb_images,1):.2f} per immagine)")
    print(f"totale regioni sfocate: {total_faces} -> {total_faces + total_heads} (+{100*total_heads/max(total_faces,1):.0f}%)")
    if fb_dist:
        print("distribuzione teste/immagine nel fallback:", dict(sorted(fb_dist.items())))


if __name__ == "__main__":
    main()