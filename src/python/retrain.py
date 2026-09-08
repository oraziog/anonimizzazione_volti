#!/usr/bin/env python3
"""Nightly binary-classifier retraining (spec §6).

Fine-tunes a MobileNetV2 (frozen backbone, fresh 2-class head) on
    class 0: false positives auto-collected during LEARNING
             (/app/dataset_falsi_positivi/{camera_id}/*.jpg)
    class 1: curated real-face seed (/app/dataset_seed/real_faces/)
and exports the head as ONNX with inputs in [1, 3, 224, 224] float32.

Preprocessing contract with the Rust runtime (src/models.rs):
the runtime feeds ImageNet-normalized CHW data
    x = (pixel/255 - mean) / std
so the training transform applies the SAME normalization. No Normalize is
ever applied twice.

The script is invoked by Rust through PyO3 (function `retrain`) and can also
be run standalone from the shell for debugging:
    python python/retrain.py SEED_DIR FP_DIR OUT.onnx METRICS.json \
        [--epochs 5] [--batch-size 32] [--holdout 0.10] [--max-class-ratio 4.0]

Class balancing: the runtime auto-collects false positives 24/7, so after a
few days the FP set can dwarf the curated seed. The larger class is capped at
`MAX_CLASS_RATIO` × the smaller one (deterministic sampling), otherwise the
classifier degenerates into "always negative".
"""

import argparse
import json
import os
import random
import sys
from datetime import datetime, timezone

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def _collect_images(root):
    found = []
    if not root or not os.path.isdir(root):
        return found
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            if os.path.splitext(name)[1].lower() in IMAGE_EXTS:
                found.append(os.path.join(dirpath, name))
    return sorted(found)


def retrain(seed_dir, fp_dir, out_onnx, metrics_json,
            epochs=5, batch_size=32, holdout=0.10, max_ratio=4.0):
    """Returns a metrics dict; raises on any failure (Rust discards then)."""
    # ── Dataset ─────────────────────────────────────────────────────────────
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, Dataset
        from torchvision import models, transforms
        from PIL import Image
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "PyTorch/torchvision/Pillow not importable — retraining requires "
            "the Python image (Dockerfile): %s" % exc
        ) from exc

    reals = _collect_images(seed_dir)
    fps = _collect_images(fp_dir)
    if len(reals) < 2 or len(fps) < 2:
        raise ValueError(
            "insufficient data for retraining: %d real, %d false positives"
            % (len(reals), len(fps))
        )

    rng = random.Random(0x5EED)

    # ── Class balance (spec §6) ─────────────────────────────────────────────
    # Cap the dominant class at MAX_CLASS_RATIO × the minority so a huge
    # auto-collected FP backlog cannot drown the curated seed.
    if max_ratio > 0.0 and len(reals) > 1 and len(fps) > 1:
        cap = max(int(round(min(len(reals), len(fps)) * max_ratio)), 2)
        if len(reals) > cap:
            reals = sorted(rng.sample(reals, cap))
            print("class balance: capped real faces to %d" % cap, file=sys.stderr)
        if len(fps) > cap:
            fps = sorted(rng.sample(fps, cap))
            print("class balance: capped false positives to %d" % cap, file=sys.stderr)

    class FaceDataset(Dataset):
        def __init__(self, items, transform):
            self.items = items  # list of (path, label)
            self.transform = transform

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            path, label = self.items[idx]
            with Image.open(path) as im:
                img = im.convert("RGB")
            return self.transform(img), label

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    labeled = [(p, 1) for p in reals] + [(p, 0) for p in fps]
    rng.shuffle(labeled)
    n_val = max(1, int(round(len(labeled) * holdout)))
    val_items, train_items = labeled[:n_val], labeled[n_val:]
    if not train_items:
        raise ValueError("holdout consumed the whole dataset")

    train_ds = FaceDataset(train_items, transform)
    val_ds = FaceDataset(val_items, transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0)

    # ── Model: frozen backbone + new head (spec §6) ────────────────────────
    device = torch.device("cpu")
    try:
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        frozen_backbone = True
    except Exception:
        # Offline fallback: random backbone, still fine-tuned end-to-end.
        model = models.mobilenet_v2(weights=None)
        frozen_backbone = False
        print("WARN: could not download ImageNet weights; training from scratch",
              file=sys.stderr)

    for param in model.features.parameters():
        param.requires_grad = False
    model.classifier = nn.Sequential(
        nn.Dropout(0.2, inplace=False),
        nn.Linear(model.last_channel, 2),
    )
    model = model.to(device)

    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # ── Training ───────────────────────────────────────────────────────────
    best_acc = 0.0
    best_state = None
    for epoch in range(1, int(epochs) + 1):
        model.train()
        total, correct = 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            preds = logits.argmax(dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
        train_acc = correct / total if total else 0.0

        model.eval()
        v_total, v_correct = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                preds = model(images).argmax(dim=1)
                v_total += labels.size(0)
                v_correct += (preds == labels).sum().item()
        val_acc = v_correct / v_total if v_total else 0.0
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        print("epoch %d/%d train_acc=%.4f val_acc=%.4f"
              % (epoch, int(epochs), train_acc, val_acc), file=sys.stderr)

    if best_state is not None:
        model.load_state_dict(best_state)

    # ── Export ─────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(out_onnx)), exist_ok=True)
    model.eval()
    dummy = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        torch.onnx.export(
            model, dummy, out_onnx,
            input_names=["input"], output_names=["output"],
            opset_version=17,
            dynamic_axes=None,
        )
    if not os.path.exists(out_onnx):
        raise RuntimeError("ONNX export produced no file at %s" % out_onnx)

    metrics = {
        "val_accuracy": round(float(best_acc), 6),
        "val_samples": v_total,
        "train_samples": len(train_items),
        "epochs_run": int(epochs),
        "real_samples_used": len(reals),
        "fp_samples_used": len(fps),
        "max_class_ratio": max_ratio,
        "backbone": "imagenet-pretrained" if frozen_backbone else "random",
        "exported_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(metrics_json, "w") as fh:
        json.dump(metrics, fh)
    print("retraining done: %s (val_accuracy=%.4f)" % (out_onnx, best_acc))
    return metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("seed_dir")
    ap.add_argument("fp_dir")
    ap.add_argument("out_onnx")
    ap.add_argument("metrics_json")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--holdout", type=float, default=0.10)
    ap.add_argument("--max-class-ratio", type=float, default=4.0,
                    help="cap the dominant class at RATIO x the minority (0 = no cap)")
    args = ap.parse_args()
    retrain(args.seed_dir, args.fp_dir, args.out_onnx, args.metrics_json,
            args.epochs, args.batch_size, args.holdout, args.max_class_ratio)


if __name__ == "__main__":
    main()
