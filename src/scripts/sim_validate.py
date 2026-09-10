#!/usr/bin/env python3
"""Simula validate_exported_model (training.rs:355) + A/B gate:
accuracy del candidato e del deployato sullo stesso holdout determinista
(sorted, 60 per class, p_face >= p_fp)."""
import glob
import os
import sys

import cv2
import numpy as np
import onnxruntime as ort

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def sample_images(root, limit=60):
    out = []
    for ext in ("jpg", "jpeg", "png"):
        out += glob.glob(os.path.join(root, "**", "*." + ext), recursive=True)
    out.sort()
    return out[:limit]


def accuracy(onnx_path, seed_dir, fp_dir, limit=60):
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    reals = sample_images(seed_dir, limit)
    fps = sample_images(fp_dir, limit)
    correct = total = 0
    for p in reals + fps:
        img = cv2.imread(p)
        if img is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        x = x.transpose(2, 0, 1)[None]
        out = sess.run(None, {sess.get_inputs()[0].name: x})[0][0]
        e = np.exp(out - out.max())
        s = e / e.sum()
        predicted_real = s[1] >= s[0]
        is_real = p.startswith(seed_dir)
        correct += (predicted_real == is_real)
        total += 1
    return correct / total if total else 0.0


if __name__ == "__main__":
    seed = sys.argv[1]
    fp = sys.argv[2]
    cand = sys.argv[3]
    deployed = sys.argv[4]
    cand_acc = accuracy(cand, seed, fp)
    dep_acc = accuracy(deployed, seed, fp)
    print(f"CANDIDATO    ({os.path.basename(cand)}):  {cand_acc:.4f}")
    print(f"DEPLOYATO    ({os.path.basename(deployed)}): {dep_acc:.4f}")
    min_acc = 0.85
    ok = cand_acc >= min_acc and cand_acc >= dep_acc
    print(f"min_acc={min_acc}  candidato>=min: {cand_acc >= min_acc}  candidato>=deployato: {cand_acc >= dep_acc}")
    print("decision: " + ("SWAP (accettato)" if ok else "REJECT (tenuto il deployato)"))