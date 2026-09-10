# eval_wider_output.py - valuta la QUALITA' di un output della pipeline rispetto
# al ground truth WIDER-val (box dei volti). Usato per confrontare configurazioni:
#
#   python scripts/eval_wider_output.py --in test-active/wider_active_in.zip \
#       --out test-active/out_X.zip --label "BASE"
#
# Metriche per ogni volto GT (ignore=0):
#   - ratio = var(Laplacian(out_box)) / var(Laplacian(in_box))   -> sfocado/chiaro
#   - alpha = 1 - |out - blur31(in)| / |in - blur31(in)|          -> opacita' blur
#   stato: BLURRED (alpha>0.6 e ratio<0.45) | PARTIAL | CLEAR (volto non sfocato)
# bucket per larghezza volto: SMALL<40, MED<120, BIG>=120 px.
#
# Stampa ASCII-safe (console Windows). Salva confronti PNG per i primi CLEAR.
import argparse
import os
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(r"C:\Users\Admin\Music\anonimizzazione_volti\src")
GT = ROOT / ".test-assets" / "wider_face_split" / "wider_face_val_bbx_gt.txt"


def parse_gt(path):
    imgs = {}
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    i = 0
    while i < len(lines):
        fname = lines[i].strip().replace("/", "\\")
        i += 1
        if i >= len(lines):
            break
        try:
            n = int(lines[i].strip())
        except ValueError:
            continue
        i += 1
        boxes = []
        for _ in range(n):
            if i >= len(lines):
                break
            p = lines[i].split()
            i += 1
            if len(p) < 5:
                continue
            x1, y1, w, h = (float(p[0]), float(p[1]), float(p[2]), float(p[3]))
            ignore = int(p[9]) if len(p) > 9 else 0
            boxes.append((x1, y1, x1 + w, y1 + h, ignore))
        imgs[fname] = boxes
    by_leaf = {}
    for k, v in imgs.items():
        by_leaf[leaf(k)] = v
    return by_leaf


def leaf(fname):
    return fname.replace("/", "\\").split("\\")[-1]


def lap_var(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def alpha_measure(orig, out):
    k = np.ones((31, 31), np.float32) / (31 * 31)
    blur = cv2.filter2D(orig.astype(np.float32), -1, k)
    d_out = np.abs(out.astype(np.float32) - blur).mean(axis=2).mean()
    d_orig = np.abs(orig.astype(np.float32) - blur).mean(axis=2).mean() + 1e-6
    return float(np.clip(1.0 - d_out / d_orig, 0.0, 1.0))


def box_state(orig, out, x1, y1, x2, y2):
    h, w = orig.shape[:2]
    x0, y0 = max(0, int(x1)), max(0, int(y1))
    x1i, y1i = min(w, int(x2)), min(h, int(y2))
    if x1i <= x0 or y1i <= y0:
        return None
    o = orig[y0:y1i, x0:x1i]
    t = out[y0:y1i, x0:x1i]
    lv_in = lap_var(o)
    lv_out = lap_var(t)
    ratio = lv_out / lv_in if lv_in > 1e-3 else 0.0
    alpha = alpha_measure(o, t)
    if alpha > 0.6 and ratio < 0.45:
        return "BLURRED", alpha, ratio
    if ratio >= 0.8:
        return "CLEAR", alpha, ratio
    return "PARTIAL", alpha, ratio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inzip")
    ap.add_argument("--out", dest="outzip")
    ap.add_argument("--label", default="RUN")
    ap.add_argument("--png", default="test-active/eval")
    args = ap.parse_args()

    gt = parse_gt(GT)
    zout = zipfile.ZipFile(args.outzip)
    out_entries = {entry.split("/")[-1]: entry for entry in zout.namelist()}
    zin = zipfile.ZipFile(args.inzip)

    stats = {"BLURRED": 0, "PARTIAL": 0, "CLEAR": 0}
    bucket = {"SMALL": {"BLURRED": 0, "PARTIAL": 0, "CLEAR": 0},
              "MED": {"BLURRED": 0, "PARTIAL": 0, "CLEAR": 0},
              "BIG": {"BLURRED": 0, "PARTIAL": 0, "CLEAR": 0}}
    alphas = []
    clear_examples = []
    total_faces = 0
    no_gt = 0
    pngdir = Path(args.png)
    pngdir.mkdir(parents=True, exist_ok=True)

    for entry in zin.namelist():
        base = entry.split("/")[-1]
        if not base.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        # CAM_001_frame_%04d_<leaf orig>
        parts = base.split("_", 4)
        orig_name = parts[4] if len(parts) == 5 else base
        if orig_name not in gt:
            no_gt += 1
            continue
        if base not in out_entries:
            print(f"  ! manca in output: {base}")
            continue
        arr_in = cv2.imdecode(np.frombuffer(zin.read(entry), np.uint8), cv2.IMREAD_COLOR)
        arr_out = cv2.imdecode(np.frombuffer(zout.read(out_entries[base]), np.uint8), cv2.IMREAD_COLOR)
        if arr_in is None or arr_out is None:
            continue
        for (x1, y1, x2, y2, ign) in gt[orig_name]:
            if ign != 0:
                continue
            total_faces += 1
            st = box_state(arr_in, arr_out, x1, y1, x2, y2)
            if st is None:
                continue
            state, alpha, ratio = st
            stats[state] += 1
            alphas.append(alpha)
            bw = x2 - x1
            b = "SMALL" if bw < 40 else ("MED" if bw < 120 else "BIG")
            bucket[b][state] += 1
            if state == "CLEAR" and len(clear_examples) < 8:
                clear_examples.append((orig_name, x1, y1, x2, y2))

    tot = sum(stats.values()) or 1
    print(f"== {args.label}: {args.inzip}")
    print(f"volti GT (non-ignore): {total_faces}   (entry senza GT: {no_gt})")
    print(f"BLURRED : {stats['BLURRED']:4d}  ({100*stats['BLURRED']/tot:5.1f}%)   <= ok")
    print(f"PARTIAL : {stats['PARTIAL']:4d}  ({100*stats['PARTIAL']/tot:5.1f}%)   <= blur trasparente/incompleto")
    print(f"CLEAR   : {stats['CLEAR']:4d}  ({100*stats['CLEAR']/tot:5.1f}%)   <= volto NON sfocato (perso!)")
    print(f"opacita' media blur (alpha, 0=chiaro 1=opaco): {np.mean(alphas):.3f}")
    print("bucket:")
    for b in ("SMALL", "MED", "BIG"):
        bb = bucket[b]
        t = sum(bb.values()) or 1
        print(f"  {b:5s}: tot {t:4d} | ok {bb['BLURRED']:4d} | parz {bb['PARTIAL']:3d} | persi {bb['CLEAR']:3d} "
              f"({100*bb['CLEAR']/t:4.1f}%)")

    for name, x1, y1, x2, y2 in clear_examples:
        # crop orig | out del primo volto CLEAR trovato, con box disegnata
        src = None
        for e in zin.namelist():
            b = e.split("/")[-1]
            if b.lower().endswith((".jpg", ".jpeg")) and b.split("_", 4)[-1] == name:
                src = e
                break
        if src is None:
            continue
        arr_in = cv2.imdecode(np.frombuffer(zin.read(src), np.uint8), cv2.IMREAD_COLOR)
        arr_out = cv2.imdecode(np.frombuffer(zout.read(out_entries[src.split('/')[-1]]), np.uint8), cv2.IMREAD_COLOR)
        if arr_in is None:
            continue
        c1 = arr_in.copy()
        c2 = arr_out.copy()
        cv2.rectangle(c1, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        cv2.rectangle(c2, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        canvas = np.hstack([c1, c2])
        fn = pngdir / ("clear_" + name.replace("/", "_").replace("\\", "_"))
        cv2.imwrite(str(fn), canvas)
        print(f"   esempio CLEAR salvato: {fn.name}")

    zin.close()
    zout.close()


if __name__ == "__main__":
    main()