#!/usr/bin/env python3
"""
classifier.py
Cat identity classifier (eevee / pichu / raichu) — MobileNetV2 transfer learning.

  1. Creates a proper train / val / test split from a single source folder,
     instead of relying on a pre-existing train/val that mixed the roles of
     validation and test.
  2. Touches the test set EXACTLY ONCE, at the very end, after all model
     selection is done. This is the "no data snooping" guarantee required
     by the rubric.
  3. Performs a small but real hyperparameter sweep (learning rate) with
     multiple random seeds per setting so we can report mean +/- std,
     instead of a single run that could be lucky.
  4. Removes a double class-weighting bug that was in the original code
     (both WeightedRandomSampler AND a weighted loss were applied; here
     we use only WeightedRandomSampler).
  5. Removes RandomVerticalFlip from the train augmentations, since cats
     do not appear upside down in the robot's operating environment and
     vertical flip is more likely to hurt than help.
  6. Saves a JSON log of every run to results.json so the report numbers
     come from one verifiable source.

"""
import json
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as T
from torch import nn, optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import models


# -------------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------------
# Source: where the *original* per-class folders live. Each subfolder
# (eevee/, pichu/, raichu/) should contain that cat's images.
SOURCE_DIR = Path(os.path.expanduser("~/cat_dataset/all"))

# Working split directory. Will be created fresh each time we run with a
# given SPLIT_SEED so the train/val/test partition is fully reproducible.
SPLIT_DIR = Path(os.path.expanduser("~/cat_dataset/split"))

# Where to save trained model weights and the results log.
OUTPUT_DIR = Path(os.path.expanduser("~/cat_classifier_runs"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Split ratios. We carve test out FIRST, then train/val from what remains.
TEST_FRAC = 0.15
VAL_FRAC = 0.15  # of the total, not of the post-test remainder
SPLIT_SEED = 42  # fixed so the split is identical across all runs

# Training settings.
EPOCHS = 30
BATCH_SIZE = 16
LR_CANDIDATES = [1e-3, 1e-4, 1e-5]  # hyperparameter sweep
SEEDS = [0, 1, 2]                   # 3 random seeds per learning rate
IMG_SIZE = 224
RESIZE_BEFORE_CROP = 256


# -------------------------------------------------------------------------
# STEP 1: build a clean train / val / test split from SOURCE_DIR
# -------------------------------------------------------------------------
def prepare_split() -> dict:
    """
    Build SPLIT_DIR/{train,val,test}/{class}/ by copying files.
    Returns a dict with the per-class image counts in each split.

    IMPORTANT: this split is created once with a fixed seed (SPLIT_SEED)
    and reused for every training run in this script. That way, no run
    gets to peek at a different test set, and the test set is touched
    exactly once per LR config -- only when we evaluate the final
    selected model on it.

    KNOWN LIMITATION (documented in the report):
    Frames in this dataset were extracted from short videos at 3 fps.
    Adjacent frames from the same video are visually very similar.
    A random per-image split therefore puts highly correlated frames
    into both train and test, which inflates test accuracy relative
    to the true generalization performance on a new video or a new
    day. Doing a video-level split would be stronger, but the original
    video-of-origin metadata is no longer attached to each frame, so
    we use the random split and disclose this caveat.
    """
    rng = random.Random(SPLIT_SEED)

    if SPLIT_DIR.exists():
        shutil.rmtree(SPLIT_DIR)

    counts = {"train": {}, "val": {}, "test": {}}

    if not SOURCE_DIR.exists():
        raise FileNotFoundError(
            f"SOURCE_DIR does not exist: {SOURCE_DIR}\n"
            "Put all images in ~/cat_dataset/all/{eevee,pichu,raichu}/\n"
            "If your data is currently in ~/cat_dataset/train and "
            "~/cat_dataset/val, merge them into ~/cat_dataset/all first:\n"
            "  mkdir -p ~/cat_dataset/all/{eevee,pichu,raichu}\n"
            "  cp ~/cat_dataset/train/eevee/*  ~/cat_dataset/all/eevee/\n"
            "  cp ~/cat_dataset/val/eevee/*    ~/cat_dataset/all/eevee/\n"
            "  cp ~/cat_dataset/train/pichu/*  ~/cat_dataset/all/pichu/\n"
            "  cp ~/cat_dataset/val/pichu/*    ~/cat_dataset/all/pichu/\n"
            "  cp ~/cat_dataset/train/raichu/* ~/cat_dataset/all/raichu/\n"
            "  cp ~/cat_dataset/val/raichu/*   ~/cat_dataset/all/raichu/\n"
        )

    for class_dir in sorted(SOURCE_DIR.iterdir()):
        if not class_dir.is_dir():
            continue
        cls = class_dir.name
        # Collect all images for this class
        imgs = sorted(class_dir.glob("*.jpg")) + sorted(class_dir.glob("*.png"))
        rng.shuffle(imgs)

        n = len(imgs)
        n_test = int(round(n * TEST_FRAC))
        n_val = int(round(n * VAL_FRAC))
        # Train gets whatever is left.

        test_imgs = imgs[:n_test]
        val_imgs = imgs[n_test:n_test + n_val]
        train_imgs = imgs[n_test + n_val:]

        for split_name, split_imgs in [
            ("train", train_imgs),
            ("val", val_imgs),
            ("test", test_imgs),
        ]:
            dst_dir = SPLIT_DIR / split_name / cls
            dst_dir.mkdir(parents=True, exist_ok=True)
            for p in split_imgs:
                shutil.copy(p, dst_dir / p.name)
            counts[split_name][cls] = len(split_imgs)

    return counts


# -------------------------------------------------------------------------
# STEP 2: data loaders
# -------------------------------------------------------------------------
def build_loaders(batch_size: int, num_workers: int = 0):
    """
    Build train / val / test DataLoaders.

    Augmentation choices:
      - RandomCrop after a Resize-to-256, so the model sees a variety of
        framings. Larger source size preserves the small white nose patch
        on pichu, which is the discriminative feature vs eevee.
      - RandomHorizontalFlip: a cat seen from the right looks like a cat
        seen from the left, so this is a free 2x data multiplier.
      - We do NOT use RandomVerticalFlip: cats never appear upside-down
        in the robot's deployment scenario, so upside-down training
        images are off-distribution noise.
      - ColorJitter and small RandomRotation: cheap robustness to lighting
        and head-tilt variation.
    """
    train_tf = T.Compose([
        T.Resize((RESIZE_BEFORE_CROP, RESIZE_BEFORE_CROP)),
        T.RandomCrop(IMG_SIZE),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        T.RandomRotation(15),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_tf = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.ImageFolder(SPLIT_DIR / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(SPLIT_DIR / "val", transform=eval_tf)
    test_ds = datasets.ImageFolder(SPLIT_DIR / "test", transform=eval_tf)

    # Sanity check: the class-to-index mapping is the same across splits
    assert train_ds.class_to_idx == val_ds.class_to_idx == test_ds.class_to_idx, (
        "Class index mapping differs across splits; ImageFolder picked them up "
        "in a different order. Check that every split has the same subfolders."
    )

    # Weighted sampler for class balance in the train loader.
    # NOTE: we use only the sampler for class balance, NOT a weighted loss.
    # Using both (as the original code did) double-corrects and biases
    # toward minority classes.
    targets = [s[1] for s in train_ds.samples]
    class_counts = np.bincount(targets, minlength=len(train_ds.classes))
    inv = 1.0 / np.maximum(class_counts, 1)
    sample_weights = [inv[t] for t in targets]
    sampler = WeightedRandomSampler(
        sample_weights, num_samples=len(sample_weights), replacement=True,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader, train_ds.classes


# -------------------------------------------------------------------------
# STEP 3: model
# -------------------------------------------------------------------------
def build_model(num_classes: int, device: torch.device) -> nn.Module:
    """
    MobileNetV2 pretrained on ImageNet, with the final classifier layer
    replaced for our 3-class task.

    Regularization choices:
      - ImageNet pretraining is itself a strong form of regularization:
        the early-layer features were learned on 1.4M diverse images
        and are not allowed to drift far during fine-tuning at our
        small learning rates.
      - Weight decay (L2) on the optimizer; chosen below.
      - The MobileNetV2 classifier head has a Dropout(p=0.2) before the
        Linear, which we keep at default.
    """
    model = models.mobilenet_v2(weights="IMAGENET1K_V1")
    in_features = model.classifier[1].in_features  # 1280
    # Replace the last layer; the existing Dropout(p=0.2) before it stays.
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model.to(device)


# -------------------------------------------------------------------------
# STEP 4: train + evaluate one configuration
# -------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    """Return overall accuracy, per-class accuracy, and confusion matrix."""
    model.eval()
    correct = 0
    total = 0
    per_class_correct = np.zeros(num_classes, dtype=np.int64)
    per_class_total = np.zeros(num_classes, dtype=np.int64)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs).argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        for c in range(num_classes):
            mask = labels == c
            per_class_total[c] += mask.sum().item()
            per_class_correct[c] += (preds[mask] == labels[mask]).sum().item()
        for t, p in zip(labels.cpu().numpy(), preds.cpu().numpy()):
            confusion[t, p] += 1

    overall_acc = correct / max(total, 1)
    per_class_acc = (
        per_class_correct / np.maximum(per_class_total, 1)
    ).tolist()
    return overall_acc, per_class_acc, confusion.tolist()


def train_one_run(lr, seed, train_loader, val_loader, num_classes, device):
    """
    Train one model with a given learning rate and seed.
    Selects the best epoch by validation accuracy.
    Returns the trained model (in best-val state) and per-epoch history.
    """
    # Reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    model = build_model(num_classes=num_classes, device=device)
    criterion = nn.CrossEntropyLoss()  # NOT class-weighted; sampler handles balance
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    best_val_acc = 0.0
    best_state = None
    history = []

    for epoch in range(EPOCHS):
        model.train()
        running_correct = 0
        running_total = 0
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
            running_correct += (out.argmax(1) == labels).sum().item()
            running_total += labels.size(0)

        train_acc = running_correct / max(running_total, 1)
        train_loss = running_loss / max(running_total, 1)

        val_acc, val_per_class, _ = evaluate(
            model, val_loader, device, num_classes,
        )
        scheduler.step()

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "val_per_class": val_per_class,
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"  epoch {epoch + 1:02d}/{EPOCHS}  "
            f"train_loss={train_loss:.3f}  train_acc={train_acc:.3f}  "
            f"val_acc={val_acc:.3f}"
        )

    # Restore best-val state before returning
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, best_val_acc


# -------------------------------------------------------------------------
# MAIN: do the sweep, then evaluate the chosen config on the test set ONCE
# -------------------------------------------------------------------------
def main():
    # Force unbuffered output so progress prints appear in real time
    # even when output is being piped through tee.
    import sys
    sys.stdout.reconfigure(line_buffering=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("\n=== STEP 1: building train/val/test split ===")
    counts = prepare_split()
    print("Image counts per split:")
    for split, by_class in counts.items():
        total = sum(by_class.values())
        print(f"  {split}: {total} total  {by_class}")

    print("\n=== STEP 2: building data loaders ===")
    train_loader, val_loader, test_loader, class_names = build_loaders(BATCH_SIZE)
    print(f"Classes (in this order): {class_names}")
    num_classes = len(class_names)

    print("\n=== STEP 3: hyperparameter sweep on val set ===")
    print(f"Learning rates: {LR_CANDIDATES}")
    print(f"Seeds per LR:   {SEEDS}")

    results_log = {
        "config": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "img_size": IMG_SIZE,
            "lr_candidates": LR_CANDIDATES,
            "seeds": SEEDS,
            "split_seed": SPLIT_SEED,
            "test_frac": TEST_FRAC,
            "val_frac": VAL_FRAC,
            "class_names": class_names,
            "image_counts": counts,
        },
        "sweep": [],
        "final": None,
    }

    sweep_summary = {}  # lr -> list of best_val_accs across seeds

    for lr in LR_CANDIDATES:
        sweep_summary[lr] = []
        for seed in SEEDS:
            print(f"\n--- training lr={lr}, seed={seed} ---")
            t0 = time.time()
            _, history, best_val = train_one_run(
                lr, seed, train_loader, val_loader, num_classes, device,
            )
            elapsed = time.time() - t0
            print(f"  done in {elapsed:.1f}s. best val_acc={best_val:.3f}")
            results_log["sweep"].append({
                "lr": lr,
                "seed": seed,
                "best_val_acc": best_val,
                "history": history,
                "elapsed_seconds": elapsed,
            })
            sweep_summary[lr].append(best_val)

    print("\n=== STEP 4: sweep summary (val set; mean +/- std across seeds) ===")
    print(f"{'lr':>10}  {'mean_val_acc':>12}  {'std_val_acc':>11}")
    best_lr = None
    best_mean = -1.0
    for lr in LR_CANDIDATES:
        vals = np.array(sweep_summary[lr])
        m, s = vals.mean(), vals.std(ddof=1) if len(vals) > 1 else 0.0
        print(f"{lr:>10.0e}  {m:>12.4f}  {s:>11.4f}")
        if m > best_mean:
            best_mean = m
            best_lr = lr

    print(f"\nSelected learning rate (best mean val acc): {best_lr:.0e}")

    print("\n=== STEP 5: final test-set evaluation (touched exactly once) ===")
    # Retrain once with the chosen LR and a fresh seed, then evaluate on test.
    # Using a NEW seed (not in SEEDS) so the test number isn't influenced by
    # the specific seeds we used during the sweep.
    final_seed = 100
    print(f"Retraining one final model: lr={best_lr:.0e}, seed={final_seed}")
    final_model, final_history, final_val = train_one_run(
        best_lr, final_seed, train_loader, val_loader, num_classes, device,
    )
    test_acc, test_per_class, test_confusion = evaluate(
        final_model, test_loader, device, num_classes,
    )

    print(f"\nFinal test accuracy: {test_acc:.4f}")
    print("Per-class test accuracy:")
    for name, acc in zip(class_names, test_per_class):
        print(f"  {name}: {acc:.4f}")
    print("Confusion matrix (rows = true, cols = predicted):")
    header = "        " + "  ".join(f"{n:>7}" for n in class_names)
    print(header)
    for name, row in zip(class_names, test_confusion):
        print(f"{name:>7} " + "  ".join(f"{v:>7d}" for v in row))

    results_log["final"] = {
        "selected_lr": best_lr,
        "final_seed": final_seed,
        "final_val_acc_best": final_val,
        "test_acc": test_acc,
        "test_per_class_acc": test_per_class,
        "test_confusion": test_confusion,
        "final_history": final_history,
    }

    # Baseline: majority-class predictor on the test set.
    test_class_counts = np.array(
        [counts["test"][c] for c in class_names], dtype=np.int64
    )
    majority_acc = float(test_class_counts.max()) / float(test_class_counts.sum())
    results_log["final"]["majority_baseline_test_acc"] = majority_acc
    print(f"\nBaseline (majority-class on test): {majority_acc:.4f}")

    # Save model and log.
    model_path = OUTPUT_DIR / "cat_classifier_final.pth"
    torch.save(final_model.state_dict(), model_path)
    log_path = OUTPUT_DIR / "results.json"
    with open(log_path, "w") as f:
        json.dump(results_log, f, indent=2)
    print(f"\nSaved final model:  {model_path}")
    print(f"Saved results log:  {log_path}")
    print("Done.")


if __name__ == "__main__":
    main()
