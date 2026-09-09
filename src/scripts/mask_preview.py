"""A/B estetico: blur ACTIVE attuale (ellisse) vs maschera MediaPipe Selfie (feathered).

Replica fedelmente la pipeline Rust:
  - sigma = box_w/8 clamp [5, 50]
  - blur   = 2 passate box blur, radius = round(sigma*1.5)
  - feather = gaussian blur della maschera con sigma (apply_masked)
  - blend alpha: out = a*blur + (1-a)*orig

Genera pagine PNG (6 volti, 3 pannelli: origine | ellisse | selfie) in testassets/mask_preview/.
"""
import argparse
import hashlib
import os
import random
import zipfile

import cv2
import numpy as np

from pathlib import Path

ROOT = Path(r"C:\Users\Admin\Music\anonimizzazione_volti\src")
GT = ROOT / ".test-assets" / "wider_face_split" / "wider_face_val_bbx_gt.txt"
ZIP = ROOT / "WIDER_val.zip"
CACHE = Path(os.environ.get("TEMP", r"C:\Users\Admin\AppData\Local\Temp")) / "opencode" / "wider_preview"
OUT = ROOT / "testassets" / "mask_preview"
MODEL = ROOT / "models_cache" / "selfie_seg.onnx"

import onnxruntime as ort


def parse_gt(path):
    """Ritorna dict: immagine -> lista box (x1,y1,x2,y2,ignore)."""
    imgs = {}
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    i = 0
    while i < len(lines):
        fname = lines[i].strip()
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
            if len(p) < 2:
                continue
            x1, y1, w, h = (float(p[0]), float(p[1]), float(p[2]), float(p[3]))
            ignore = int(p[9]) if len(p) > 9 else 0
            boxes.append((x1, y1, x1 + w, y1 + h, ignore))
        imgs[fname] = boxes
    return imgs


def sigma_for_box(w):
    return min(50.0, max(5.0, w / 8.0))


def box_blur2(img, sigma):
    r = max(2, int(round(sigma * 1.5)))
    k = 2 * r + 1
    return cv2.blur(cv2.blur(img, (k, k)), (k, k))


def feather(mask, sigma):
    s = max(0.001, sigma)
    return cv2.GaussianBlur(mask.astype(np.float32), (0, 0), s)


def blend(orig, blured, alpha):
    a = alpha[..., None] / 255.0
    return (blured.astype(np.float32) * a + orig.astype(np.float32) * (1.0 - a)).astype(np.uint8)


def load_face(path_fname, box):
    """Estrae l'immagine dalla zip (con cache su disco) e ritorna crop con margine 15%."""
    entry = f"WIDER_val/images/{path_fname}"
    cache_path = CACHE / hashlib.md5(entry.encode()).hexdigest()[:8] / Path(path_fname).name
    if not cache_path.exists():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(ZIP) as z:
            with z.open(entry) as fi, open(cache_path, "wb") as fo:
                fo.write(fi.read())
    img = cv2.imread(str(cache_path))
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box[:4]]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    side = max(x2 - x1, y2 - y1)
    if x2 <= x1 or y2 <= y1:
        return None
    m = int(0.15 * side)
    cx1, cy1 = max(0, x1 - m), max(0, y1 - m)
    cx2, cy2 = min(w, x2 + m), min(h, y2 + m)
    return img[cy1:cy2, cx1:cx2], (x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--volti", type=int, default=150)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--tune", type=int, nargs="*", default=[])
    args = ap.parse_args()

    imgs = parse_gt(GT)
    candidates = []
    for fname, boxes in imgs.items():
        for b in boxes:
            w, h = b[2] - b[0], b[3] - b[1]
            if b[4] == 0 and 50 <= max(w, h) <= 460 and min(w, h) > 20:
                candidates.append((fname, b))
    rnd = random.Random(args.seed)
    rnd.shuffle(candidates)

    sess = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    out = sess.get_outputs()[0].name

    OUT.mkdir(parents=True, exist_ok=True)
    pages = {}
    seen = set()
    n = 0
    for fname, b in candidates:
        if n >= args.volti:
            break
        if fname in seen:
            continue
        seen.add(fname)
        r = load_face(fname, b)
        if r is None:
            continue
        crop, (bx1, by1, bx2, by2) = r
        n += 1
        ch, cw = crop.shape[:2]

        # pannello originale
        orig = crop.copy()

        # ellisse attuale (fallback ACTIVE): inscritta nel box, margin 5%
        ell = np.zeros((ch, cw), np.uint8)
        bw = bx2 - bx1
        bh = by2 - by1
        rw = bw / 2 * 1.05
        rh = (by2 - by1) / 2 * 1.05
        cx = (bx1 + bx2) / 2
        cy = (by1 + by2) / 2
        cv2.ellipse(ell, (int(cx), int(cy)), (max(1, int(rw)), max(1, int(rh))), 0, 0, 360, 255, -1)
        sigma = sigma_for_box(bw)
        blurred = box_blur2(crop, sigma)
        a_ell = feather(ell, sigma)
        out_ell = blend(crop, blurred, a_ell)

        out_mp = maschera_blur(sess, inp, out, crop, ell, bw, bh, sigma)

        if n in args.tune:
            save_tune(OUT, n, orig, sess, inp, out, crop, ell, bw, bh)

        page_idx = (n - 1) // 6
        row = (n - 1) % 6
        panel = {0: orig, 1: out_ell, 2: out_mp}
        pages.setdefault(page_idx, {})[row] = panel
        print(f"[{n:4d}] {fname} box=({bx1},{by1},{bx2 - bx1}x{by2 - by1}) sigma={sigma:.0f}")

    # rendi le pagine
    for pi, rows in sorted(pages.items()):
        canvas = np.full((6 * 280 + 7 * 20 + 40, 3 * 340 + 4 * 20 + 120, 3), 255, np.uint8)
        for row in range(6):
            if row not in rows:
                continue
            for col in range(3):
                im = rows[row][col]
                ih, iw = im.shape[:2]
                scale = min(300 / iw, 260 / ih)
                nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
                im = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_AREA)
                x = 20 + 20 + col * (340 + 20)
                y = 40 + 20 + row * (280 + 20) + (260 - nh) // 2
                canvas[y:y + nh, x:x + nw] = im
        cv2.imwrite(str(OUT / f"page_{pi + 1:02d}.png"), canvas)
    print(f"\n-> {len(pages)} pagine in {OUT}")


def maschera_blur(sess, inp, out, crop, ell, bw, bh, sigma_eff):
    """Colonna proposta: maschera selfie estesa (1/4 lato) in unione con l'ellisse."""
    ch, cw = crop.shape[:2]
    t = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    t = cv2.resize(t, (256, 256), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    t = np.transpose(t, (2, 0, 1))[None].astype(np.float32)
    m = sess.run([out], {inp: t})[0][0, 0]
    mask = (m > 0.5).astype(np.uint8) * 255
    mask = cv2.resize(mask, (cw, ch), interpolation=cv2.INTER_LINEAR)
    mask = (mask > 127).astype(np.uint8) * 255
    rad = max(2, int(round(0.25 * min(bw, bh))))
    kern = np.ones((2 * rad + 1, 2 * rad + 1), np.uint8)
    dilmask = np.maximum(cv2.dilate(mask, kern), ell)
    blurred = box_blur2(crop, sigma_eff)
    a = feather(dilmask, sigma_eff)
    return blend(crop, blurred, a)


def save_tune(OUT, vid, orig, sess, inp, out, crop, ell, bw, bh):
    """Foglio di taratura dell'intensita': ORIG + sigma x1/x1.5/x2/x3."""
    factors = (1, 1.5, 2, 3)
    canvas = np.full((320, (1 + len(factors)) * 340 + (len(factors) + 2) * 20, 3), 255, np.uint8)
    ims = [orig]
    for k in factors:
        ims.append(maschera_blur(sess, inp, out, crop, ell, bw, bh, sigma_for_box(bw) * k))
    for col, im in enumerate(ims):
        ih, iw = im.shape[:2]
        scale = min(300 / iw, 260 / ih)
        nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
        im = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_AREA)
        x = 20 + col * (340 + 20)
        y = (320 - nh) // 2
        canvas[y:y + nh, x:x + nw] = im
    cv2.imwrite(str(OUT / f"tune_{vid}.png"), canvas)
    print(f"-> foglio taratura tune_{vid}.png (ORIG | x1 | x1.5 | x2 | x3)")


if __name__ == "__main__":
    main()