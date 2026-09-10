#!/usr/bin/env python3
"""Benchmark mostrava il costo relativo dei 3 metodi di anonimizzazione
(gaussiana attuale / fill-nero / pixelate / blur-forte x4) su ROI reali."""
import time
import zipfile

import cv2
import numpy as np

ZIP = "test-wider-sample/WIDER_val.zip"


def blur_gauss(img, region, sigma, passes=1):
    for _ in range(passes):
        roi = img[region[1] : region[3], region[0] : region[2]]
        k = int(round(sigma * 1.5)) * 2 + 1
        img[region[1] : region[3], region[0] : region[2]] = cv2.GaussianBlur(roi, (k, k), sigma)
    return img


def pixelate(img, region, cell=12):
    roi = img[region[1] : region[3], region[0] : region[2]]
    h, w = roi.shape[:2]
    small = cv2.resize(roi, (max(1, w // cell), max(1, h // cell)), interpolation=cv2.INTER_LINEAR)
    img[region[1] : region[3], region[0] : region[2]] = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    return img


def main():
    z = zipfile.ZipFile(ZIP)
    names = [n for n in z.namelist() if n.lower().endswith(".jpg")][:40]
    imgs = [cv2.imdecode(np.frombuffer(z.read(n), np.uint8), cv2.IMREAD_COLOR) for n in names]
    imgs = [i for i in imgs if i is not None]
    z.close()

    # ROI realistici: distribuzione larghezze ~ WIDER (molti piccoli, pochi grandi)
    pairs = []
    rng = np.random.default_rng(0)
    for img in imgs:
        n = min(40, 10 + int(rng.integers(0, 20)))
        for _ in range(n):
            # wr: 40% piccoli 15-35, 40% medi 35-70, 20% grandi 70-160
            r = rng.random()
            w = int(rng.integers(15, 35)) if r < 0.4 else (int(rng.integers(35, 70)) if r < 0.8 else int(rng.integers(70, 160)))
            h = int(w * rng.uniform(1.0, 1.4))
            w = min(w, img.shape[1] - 2)
            h = min(h, img.shape[0] - 2)
            x0 = int(rng.integers(0, img.shape[1] - w + 1))
            y0 = int(rng.integers(0, img.shape[0] - h + 1))
            pairs.append((img, (x0, y0, x0 + w, y0 + h)))
    pairs = pairs[:800]
    regions = [r for _, r in pairs]

    def bench(fn, repeat=2):
        total = 0.0
        for _ in range(repeat):
            t0 = time.perf_counter()
            for img, reg in pairs:
                img = img.copy()
                fn(img, reg)
            total += time.perf_counter() - t0
        return total / repeat / len(pairs) * 1000  # ms per volto

    sigma_avg = sum(max(5, min(50, (r[2] - r[0]) / 4.0)) for r in regions) / len(regions)
    print(f"Regioni: {len(regions)}  volto medio = ~{(regions[0][2]-regions[0][0])}px..{(regions[-1][2]-regions[-1][0])}px  sigma_avg={sigma_avg:.1f}")
    print(f"gaussiana attuale (x2 passi):  {bench(lambda i, r: blur_gauss(i, r, max(5, min(50, (r[2]-r[0])/4.0)), passes=2)):.4f} ms/volto")
    print(f"blur forte x4 (x4 passi):      {bench(lambda i, r: blur_gauss(i, r, max(5, min(50, (r[2]-r[0])/4.0))*2, passes=4)):.4f} ms/volto")
    print(f"pixelate (cell 12):            {bench(lambda i, r: pixelate(i, r, 12)):.4f} ms/volto")
    print(f"pixelate (cell 8):             {bench(lambda i, r: pixelate(i, r, 8)):.4f} ms/volto")


if __name__ == "__main__":
    main()