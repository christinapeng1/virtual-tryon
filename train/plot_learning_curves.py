#!/usr/bin/env python3
"""
Parse SLURM .out files from training runs and plot learning curves.

Usage
-----
  # Single run:
  python train/plot_learning_curves.py posenet_sub_2196619.out

  # Multiple runs overlaid (add --label for legend names):
  python train/plot_learning_curves.py \\
      posenet_sub_2196619.out --label "20% subsample" \\
      posenet_sub_2214139.out --label "40% subsample" \\
      --out figures/learning_curves.png

Output
------
  Three-panel figure:
    - Train loss vs epoch
    - Validation loss vs epoch
    - Learning rate vs epoch (log scale)
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless — no display needed on Oscar
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ── Parsing ───────────────────────────────────────────────────────────────────

# Matches: "Epoch   1/30  train=0.000683  val=0.000387  lr=9.97e-04  1019s"
EPOCH_RE = re.compile(
    r"Epoch\s+(\d+)/\d+\s+train=([\d.e+-]+)\s+val=([\d.e+-]+)\s+lr=([\d.e+-]+)"
)


def parse_log(path: str) -> dict:
    """Return {epochs, train_loss, val_loss, lr} lists parsed from a .out file."""
    epochs, train_loss, val_loss, lr = [], [], [], []
    with open(path) as f:
        for line in f:
            m = EPOCH_RE.search(line)
            if m:
                epochs.append(int(m.group(1)))
                train_loss.append(float(m.group(2)))
                val_loss.append(float(m.group(3)))
                lr.append(float(m.group(4)))
    if not epochs:
        print(f"[WARN] No epoch lines found in {path}", file=sys.stderr)
    return {"epochs": epochs, "train": train_loss, "val": val_loss, "lr": lr}


def merge_runs(run_dicts: list[dict]) -> dict:
    """
    Merge multiple parsed logs into a single run by epoch number.

    Useful when a run spans multiple SLURM jobs/logs due to preemption or resume.
    If an epoch appears multiple times, the later occurrence wins.
    """
    by_epoch: dict[int, tuple[float, float, float]] = {}
    for d in run_dicts:
        for ep, tr, va, l in zip(d["epochs"], d["train"], d["val"], d["lr"]):
            by_epoch[int(ep)] = (float(tr), float(va), float(l))

    epochs = sorted(by_epoch.keys())
    train = [by_epoch[e][0] for e in epochs]
    val = [by_epoch[e][1] for e in epochs]
    lr = [by_epoch[e][2] for e in epochs]
    return {"epochs": epochs, "train": train, "val": val, "lr": lr}


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Plot learning curves from SLURM .out files",
        epilog=(
            "Example:\n"
            "  python train/plot_learning_curves.py \\\n"
            "      posenet_sub_2196619.out posenet_sub_2214139.out \\\n"
            "      --labels '20%% subsample' '40%% subsample' \\\n"
            "      --out figures/learning_curves.png"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "files",
        nargs="+",
        metavar="FILE",
        help="One or more .out log files to plot",
    )
    p.add_argument(
        "--labels",
        nargs="*",
        metavar="LABEL",
        default=[],
        help="Legend label for each file, in the same order (optional)",
    )
    p.add_argument(
        "--out",
        default="learning_curves.png",
        metavar="PATH",
        help="Output image path (default: learning_curves.png)",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Output DPI (default: 150)",
    )
    return p.parse_args()


# ── Plotting ──────────────────────────────────────────────────────────────────

COLORS = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0"]


def main():
    args = parse_args()

    # Pad labels if fewer were given than files
    labels = list(args.labels)
    while len(labels) < len(args.files):
        labels.append(Path(args.files[len(labels)]).stem)

    # Group files by label so one run can span multiple logs (resume/requeue)
    grouped: dict[str, list[str]] = {}
    for path, label in zip(args.files, labels):
        grouped.setdefault(label, []).append(path)

    runs = []
    for label, paths in grouped.items():
        parsed = []
        for path in paths:
            data = parse_log(path)
            if data["epochs"]:
                parsed.append(data)
                print(f"  {label}: {len(data['epochs'])} epochs parsed from {path}")
            else:
                print(f"  [SKIP] {path} — no epoch data found")

        if not parsed:
            continue

        merged = merge_runs(parsed) if len(parsed) > 1 else parsed[0]
        runs.append((label, merged))
        if len(parsed) > 1:
            print(f"  {label}: merged {len(paths)} files -> {len(merged['epochs'])} epochs")

    if not runs:
        sys.exit("No data to plot.")

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    fig.suptitle("PoseNet Training — Learning Curves", fontsize=14, fontweight="bold")

    ax_train, ax_val, ax_lr = axes

    for i, (label, data) in enumerate(runs):
        color = COLORS[i % len(COLORS)]
        ep = data["epochs"]

        ax_train.plot(ep, data["train"], color=color, linewidth=1.8, label=label)
        ax_val.plot(ep, data["val"],   color=color, linewidth=1.8, label=label,
                    linestyle="--")

        # Mark best val epoch
        best_idx = data["val"].index(min(data["val"]))
        ax_val.scatter(
            ep[best_idx], data["val"][best_idx],
            color=color, s=80, zorder=5,
            label=f"{label} best (ep {ep[best_idx]}, {data['val'][best_idx]:.5f})",
        )

        ax_lr.plot(ep, data["lr"], color=color, linewidth=1.8, label=label)

    # ── Train loss panel ──────────────────────────────────────────────────────
    ax_train.set_ylabel("Train Loss (SmoothL1)", fontsize=11)
    ax_train.legend(fontsize=9)
    ax_train.grid(True, alpha=0.3)
    ax_train.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.5f"))

    # ── Val loss panel ────────────────────────────────────────────────────────
    ax_val.set_ylabel("Val Loss (SmoothL1)", fontsize=11)
    ax_val.legend(fontsize=9)
    ax_val.grid(True, alpha=0.3)
    ax_val.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.5f"))

    # ── LR panel ─────────────────────────────────────────────────────────────
    ax_lr.set_ylabel("Learning Rate", fontsize=11)
    ax_lr.set_xlabel("Epoch", fontsize=11)
    ax_lr.set_yscale("log")
    ax_lr.legend(fontsize=9)
    ax_lr.grid(True, alpha=0.3)

    # Integer x-ticks
    all_epochs = max(max(d["epochs"]) for _, d in runs)
    ax_lr.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=10))
    ax_lr.set_xlim(1, all_epochs)

    plt.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
