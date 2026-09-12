#!/usr/bin/env python3
"""Validazione della soglia CLASSIFIER_CONFIRM_THRESHOLD con verita' certa.

Tre insiemi con etichette affidabili (il dataset dei falsi positivi del
servizio NON e' usato come verita': contiene anche veri volti piccoli):

  A) 200 volti chiari (box GT WIDER >= 28 px)          - popolazione tipica
  B) 200 volti piccoli (box GT 12..27 px)              - caso peggiore: volti
     lontani che YOLO in ACTIVE (conf >= 0.05) rileva a bassa confidenza
  C) 200 sfondi certi (crop senza intersezione GT)     - non-volti veri

Per ogni soglia riporta recall per A/B, FP su C e propone la soglia piu'
alta che mantiene recall_A >= 98% e recall_B >= 95% (il danno asimmetrico:
un volto non sfocato e' una violazione GDPR, un blur in piu' e' solo estetico).

Uso (una riga):
  C:/Users/Admin/Music/anonimizzazione_volti/.venv312/Scripts/python.exe C:/Users/Admin/Music/anonimizzazione_volti/valida-classificatore.py
"""
import json
import os
import random
import sys

ROOT = r"C:/Users/Admin/Music/anonimizzazione_volti"
GT = os.path.join(ROOT, "wider_seed", "wider_face_split", "wider_face_val_bbx_gt.txt")
IMAGES = os.path.join(ROOT, "wider_seed", "WIDER_val", "images")
MODELS = os.path.join(ROOT, "models", "cache")
OUT_JSON = os.path.join(ROOT, "test_zip", "validazione_classificatore.json")

N_PER_SET = 200
MIN_FACE = 28          # set A
SMALL_MIN, SMALL_MAX = 12, 27  # set B
SPLIT_SEED = 20260912

import numpy as np
from PIL import Image
import onnxruntime as ort

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def parse_gt():
    entries = []
    cur = None
    with open(GT, "r", encoding="utf-8") as fh:
        lines = [ln.rstrip("\n") for ln in fh]
    i = 0
    while i < len(lines):
        name = lines[i].strip()
        if not name.endswith(".jpg"):
            i += 1
            continue
        i += 1
        count = int(lines[i].split()[0])
        boxes = []
        j = i + 1
        for _ in range(max(count, 0)):
            parts = lines[j].split()
            x, y, w, h = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
            if w >= 1 and h >= 1:
                boxes.append((x, y, w, h))
            j += 1
        entries.append((name, boxes))
        i = j
    return entries


def crop_face(img, box, pad=0.15):
    x, y, w, h = box
    pw, ph = max(int(w * pad), 2), max(int(h * pad), 2)
    left, top = max(0, int(x) - pw), max(0, int(y) - ph)
    right, bottom = min(img.width, int(x + w) + pw), min(img.height, int(y + h) + ph)
    if right - left < 8 or bottom - top < 8:
        return None
    return img.crop((left, top, right, bottom))


def collect_faces(size_range):
    """Volti GT con lato minimo dentro size_range (lo, hi)."""
    lo, hi = size_range
    rng = random.Random(SPLIT_SEED)
    entries = [e for e in parse_gt() if e[1]]
    rng.shuffle(entries)
    out = []
    for name, boxes in entries:
        if len(out) >= N_PER_SET:
            break
        path = os.path.join(IMAGES, name.replace("/", os.sep))
        if not os.path.exists(path):
            continue
        valid = [b for b in boxes if lo <= min(b[2], b[3]) and max(b[2], b[3]) < (hi or 10**9)]
        if not valid:
            continue
        try:
            with Image.open(path) as im:
                im = im.convert("RGB")
                for box in valid[:6]:
                    c = crop_face(im, box)
                    if c is None:
                        continue
                    out.append((name, c))
                    if len(out) >= N_PER_SET:
                        break
        except Exception:
            continue
    return out


def collect_backgrounds():
    """Crop di sfondo certi: nessuna intersezione (margine 20 px) con box GT."""
    rng = random.Random(SPLIT_SEED + 1)
    entries = [e for e in parse_gt() if e[1]]
    rng.shuffle(entries)
    out = []
    for name, boxes in entries:
        if len(out) >= N_PER_SET:
            break
        path = os.path.join(IMAGES, name.replace("/", os.sep))
        if not os.path.exists(path):
            continue
        try:
            with Image.open(path) as im:
                im = im.convert("RGB")
                for _ in range(40):
                    cw = rng.randint(80, 220)
                    ch = rng.randint(80, 220)
                    if im.width < cw or im.height < ch:
                        continue
                    left = rng.randint(0, im.width - cw)
                    top = rng.randint(0, im.height - ch)
                    clash = False
                    for (x, y, w, h) in boxes:
                        if min(w, h) < 10:
                            continue  # micro-box: visivamente trascurabili
                        m = 8
                        if not (left + cw < x - m or left > x + w + m or
                                top + ch < y - m or top > y + h + m):
                            clash = True
                            break
                    if clash:
                        continue
                    out.append((os.path.basename(path), im.crop((left, top, left + cw, top + ch))))
                    break
        except Exception:
            continue
    return out


def to_input(img):
    x = img.resize((224, 224)).convert("RGB")
    arr = np.asarray(x, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    chw = arr.transpose(2, 0, 1)[None, ...]
    return np.ascontiguousarray(chw)


def latest_classifier():
    cands = [f for f in os.listdir(MODELS) if f.startswith("classifier_") and f.endswith(".onnx")]
    cands.sort()
    return os.path.join(MODELS, cands[-1])


def score_set(sess, inp, items):
    scores = []
    for name, c in items:
        p = float(sess.run(None, {inp: to_input(c)})[0][0][1])
        scores.append((p, name))
    return scores


def main():
    print("costruzione insiemi (verita' certa da GT WIDER)...")
    setA = collect_faces((MIN_FACE, 0))
    setB = collect_faces((SMALL_MIN, SMALL_MAX))
    setC = collect_backgrounds()
    print(f"A volti chiari: {len(setA)}   B volti piccoli: {len(setB)}   C sfondi certi: {len(setC)}")
    if len(setA) < 100 or len(setC) < 80:
        print("set insufficienti, interrotto", file=sys.stderr)
        return 1

    model = latest_classifier()
    print("modello:", os.path.basename(model))
    sess = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name

    sA = score_set(sess, inp, setA)
    sB = score_set(sess, inp, setB) if setB else []
    sC = score_set(sess, inp, setC)

    soglie = (0.25, 0.5, 0.6, 0.75, 0.9)
    tabella = {}
    print("\n{:>7} {:>16} {:>16} {:>22} {:>10}".format(
        "soglia", "recall A", "recall B", "FP su C (su 200)", "accuracy A+C"))
    for thr in soglie:
        recA = sum(1 for p, _ in sA if p >= thr) / len(sA)
        recB = (sum(1 for p, _ in sB if p >= thr) / len(sB)) if sB else None
        fpC = sum(1 for p, _ in sC if p >= thr)
        accAC = ((sum(1 for p, _ in sA if p >= thr) + sum(1 for p, _ in sC if p < thr))
                 / (len(sA) + len(sC)))
        tabella[str(thr)] = {
            "recall_volti_chiari": round(recA, 4),
            "recall_volti_piccoli": round(recB, 4) if recB is not None else None,
            "fp_su_sfondi_certi": fpC,
            "accuracy_A_più_C": round(accAC, 4),
        }
        print("{:>7} {:>16} {:>16} {:>22} {:>10}".format(
            thr, f"{recA*100:.1f}%",
            f"{recB*100:.1f}%" if recB is not None else "-",
            f"{fpC} ({fpC/len(sC)*100:.0f}%)",
            f"{accAC*100:.1f}%"))

    # Raccomandazione: la soglia piu' alta con recall_A >= 98% e recall_B >= 95%
    raccomandata = 0.5
    for thr in soglie:
        t = tabella[str(thr)]
        rb = t["recall_volti_piccoli"]
        ok = t["recall_volti_chiari"] >= 0.98 and (rb is None or rb >= 0.95)
        if ok:
            raccomandata = thr
        else:
            break
    print(f"\nsoglia raccomandata (GDPR-safe): {raccomandata}")

    risultati = {
        "model": os.path.basename(model),
        "set": {"volti_chiari_A": len(sA), "volti_piccoli_B": len(sB), "sfondi_certi_C": len(sC)},
        "soglie": tabella,
        "soglia_raccomandata": raccomandata,
    }
    with open(OUT_JSON, "w") as fh:
        json.dump(risultati, fh, indent=2, ensure_ascii=False)
    print("dettaglio:", OUT_JSON)
    return 0


if __name__ == "__main__":
    sys.exit(main())
