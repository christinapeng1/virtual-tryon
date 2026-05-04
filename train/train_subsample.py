#!/usr/bin/env python3
"""
Train PoseNet on a random subsample of preprocessed SURREAL data.

Identical to train.py except for the --subsample argument, which randomly
selects a fraction of the training set each run.  Validation is never
subsampled so losses remain comparable across experiments.

Typical usage on OSCAR
-----------------------
  python train_subsample.py \\
    --train_csv /users/dzhu36/scratch/surreal_preprocessed/index_train.csv \\
    --val_csv   /users/dzhu36/scratch/surreal_preprocessed/index_val.csv   \\
    --out_dir   /users/dzhu36/scratch/posenet_checkpoints_sub              \\
    --subsample 0.20   \\
    --epochs    30     \\
    --batch     128    \\
    --workers   8

Checkpoints saved
------------------
  <out_dir>/last.pt   — overwritten every epoch (safe to interrupt + resume)
  <out_dir>/best.pt   — best validation loss seen so far

Resume after interruption
--------------------------
  Add  --resume <out_dir>/last.pt  to the command above.
  Note: resuming resamples the training set with the same --subsample fraction
  but a different random draw.  This is intentional — each run sees different
  clips, acting as a form of data augmentation across runs.
"""

import argparse
import os
import random
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from dataset import SURREALDataset
from model import PoseNet
from torch.optim.lr_scheduler import CosineAnnealingLR


# ── Argument parsing ───────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train PoseNet on a subsample of SURREAL")
    p.add_argument("--train_csv", required=True, help="index_train.csv from preprocess.py")
    p.add_argument("--val_csv",   required=True, help="index_val.csv from preprocess.py")
    p.add_argument("--out_dir",   required=True, help="Directory for checkpoints")
    p.add_argument(
        "--subsample",
        type=float,
        default=0.20,
        metavar="FRAC",
        help="Fraction of training samples to use per run, e.g. 0.20 for 20%% (default: 0.20)",
    )
    p.add_argument(
        "--backbone",
        default="mobilenet_v3_small",
        choices=["mobilenet_v3_small", "mobilenet_v3_large", "mobilenet_v2"],
        help="Feature extractor backbone (default: mobilenet_v3_small)",
    )
    p.add_argument("--epochs",  type=int,   default=30,   help="Training epochs")
    p.add_argument("--batch",   type=int,   default=128,  help="Batch size")
    p.add_argument("--lr",      type=float, default=1e-3, help="Initial learning rate")
    p.add_argument("--workers", type=int,   default=8,    help="DataLoader workers")
    p.add_argument(
        "--resume",
        default=None,
        metavar="CHECKPOINT",
        help="Path to a .pt checkpoint to resume training from",
    )
    return p.parse_args()


# ── Training / validation steps ───────────────────────────────────────────────

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> float:
    model.train()
    total_loss = 0.0
    log_interval = max(1, len(loader) // 10)  # print ~10 times per epoch

    print(f"  Epoch {epoch+1}: starting {len(loader)} batches...", flush=True)
    for batch_idx, (imgs, joints) in enumerate(loader):
        imgs   = imgs.to(device)
        joints = joints.to(device)

        preds = model(imgs)
        loss  = criterion(preds, joints)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)

        if (batch_idx + 1) % log_interval == 0:
            pct = 100.0 * (batch_idx + 1) / len(loader)
            avg = total_loss / ((batch_idx + 1) * imgs.size(0))
            print(f"  Epoch {epoch+1} [{pct:5.1f}%]  batch_loss={avg:.6f}", flush=True)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def val_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    for imgs, joints in loader:
        imgs   = imgs.to(device)
        joints = joints.to(device)
        preds  = model(imgs)
        total_loss += criterion(preds, joints).item() * imgs.size(0)
    return total_loss / len(loader.dataset)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    if not (0.0 < args.subsample <= 1.0):
        raise ValueError(f"--subsample must be in (0, 1], got {args.subsample}")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Training on CPU is not practical for this dataset.\n"
            "Check that: (1) you requested a GPU node (--gres=gpu:1), and\n"
            "            (2) PyTorch was installed with a CUDA version matching the driver:\n"
            "                pip install torch torchvision "
            "--index-url https://download.pytorch.org/whl/cu124"
        )
    device = torch.device("cuda")
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Device  : {device} ({torch.cuda.get_device_name(0)})")
    print(f"Backbone: {args.backbone}")

    # ── Datasets ──────────────────────────────────────────────────────────────
    print("Loading datasets...", flush=True)
    train_ds_full = SURREALDataset(args.train_csv, augment=True)
    val_ds        = SURREALDataset(args.val_csv,   augment=False)

    # Randomly subsample the training set
    n_full   = len(train_ds_full)
    n_sub    = max(1, int(n_full * args.subsample))
    indices  = random.sample(range(n_full), n_sub)
    train_ds = Subset(train_ds_full, indices)

    print(f"Train   : {n_sub:,} / {n_full:,} samples ({args.subsample*100:.0f}% subsample)")
    print(f"Val     : {len(val_ds):,} samples")

    # ── DataLoaders ───────────────────────────────────────────────────────────
    print(f"Building DataLoaders (workers={args.workers})...", flush=True)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=False,
        persistent_workers=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False,
        persistent_workers=False,
    )
    print("DataLoaders ready.", flush=True)

    # ── Model, optimiser, scheduler, loss ─────────────────────────────────────
    print("Building model...", flush=True)
    model     = PoseNet(backbone=args.backbone, pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    criterion = nn.SmoothL1Loss()

    start_epoch   = 0
    best_val_loss = float("inf")

    # ── Optional resume ───────────────────────────────────────────────────────
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch   = ckpt["epoch"] + 1
        best_val_loss = ckpt["val_loss"]
        print(f"Resumed from epoch {ckpt['epoch']}  (best val_loss={best_val_loss:.6f})")

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        t0         = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        print(f"Train loss: {train_loss:.6f}", flush=True)
        val_loss   = val_epoch(model, val_loader, criterion, device)
        print(f"Val loss: {val_loss:.6f}", flush=True)
        scheduler.step()
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch+1:3d}/{args.epochs}"
            f"  train={train_loss:.6f}"
            f"  val={val_loss:.6f}"
            f"  lr={scheduler.get_last_lr()[0]:.2e}"
            f"  {elapsed:.0f}s",
            flush=True,
        )

        ckpt = {
            "epoch":      epoch,
            "backbone":   args.backbone,
            "subsample":  args.subsample,
            "model":      model.state_dict(),
            "optimizer":  optimizer.state_dict(),
            "scheduler":  scheduler.state_dict(),
            "val_loss":   val_loss,
        }

        torch.save(ckpt, os.path.join(args.out_dir, "last.pt"))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(ckpt, os.path.join(args.out_dir, "best.pt"))
            print(f"         ↑ new best val_loss={best_val_loss:.6f}", flush=True)

    print(f"\nFinished.  Best val_loss={best_val_loss:.6f}")
    print(f"Best checkpoint: {os.path.join(args.out_dir, 'best.pt')}")


if __name__ == "__main__":
    main()
