#!/usr/bin/env python3
"""
One-time preprocessing: extract SURREAL frames as JPEGs and build index CSVs.

Run this once before training.  It walks your surreal_data/ directory,
extracts every frame from every .mp4, applies the selfie flip, resizes to
192x256, and writes:
  <out_dir>/frames/   -- JPEG images, one per frame
  <out_dir>/index_train.csv
  <out_dir>/index_val.csv

Each CSV row: jpeg_path, j0_x, j0_y, ..., j23_x, j23_y  (48 label columns)

Selfie flip applied here (once, baked in)
------------------------------------------
  image   : cv2.flip(frame, 1)         -- horizontal mirror
  labels  : joints_x = 1.0 - joints_x -- mirror x coordinate
  NO pair swap needed (see model.py SMPL_FLIP_PAIRS comment for why)

Frames filtered out
--------------------
  Any frame where a shoulder, elbow, or wrist joint falls outside [0, 1]
  after normalization.  All other joints are clamped to [0, 1].

Typical usage on OSCAR
-----------------------
  python preprocess.py \\
    /users/dzhu36/scratch/surreal_data \\
    /users/dzhu36/scratch/surreal_preprocessed \\
    --splits train val \\
    --workers 8
"""

import argparse
import csv
import glob
import multiprocessing as mp
import os
import sys

import cv2
import numpy as np
import scipy.io

# SMPL indices that must be in-frame for a sample to be kept
REQUIRED_JOINTS = [16, 17, 18, 19, 20, 21]  # shoulders, elbows, wrists

# Native SURREAL frame dimensions
SURREAL_H, SURREAL_W = 240, 320

# Output frame dimensions (same 4:3 aspect ratio -- no distortion)
OUTPUT_H, OUTPUT_W = 192, 256

# ── Per-clip worker ───────────────────────────────────────────────────────────

def _process_clip(args: tuple) -> tuple[list, int]:
    """
    Worker function called by each process in the multiprocessing pool.

    Returns (rows, n_skipped) where rows is a list of CSV rows
    [jpeg_path, j0_x, j0_y, ..., j23_x, j23_y].
    """
    mp4_path, mat_path, out_frames_dir, prefix = args

    # ── Load joints ──────────────────────────────────────────────────────────
    try:
        info     = scipy.io.loadmat(mat_path)
        joints2d = info["joints2D"].astype(np.float32)  # (2, 24, T)
    except Exception as exc:
        print(f"[WARN] skipping {mat_path}: {exc}", file=sys.stderr, flush=True)
        return [], 0

    # MATLAB squeezes single-frame clips from (2, 24, 1) to (2, 24) — restore the T dim.
    if joints2d.ndim == 2:
        joints2d = joints2d[:, :, np.newaxis]

    T = joints2d.shape[2]

    # Normalize pixels -> [0, 1]:  x / width,  y / height
    joints_norm = np.zeros((T, 24, 2), dtype=np.float32)
    joints_norm[:, :, 0] = joints2d[0, :, :].T / SURREAL_W   # x
    joints_norm[:, :, 1] = joints2d[1, :, :].T / SURREAL_H   # y

    # Apply selfie flip: mirror x (no pair swap -- see module docstring)
    joints_norm[:, :, 0] = 1.0 - joints_norm[:, :, 0]

    # ── Load all video frames at once (faster than per-frame seek) ───────────
    cap    = cv2.VideoCapture(mp4_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        print(f"[WARN] no frames decoded from {mp4_path}", file=sys.stderr, flush=True)
        return [], 0

    actual_T  = min(T, len(frames))
    rows      = []
    n_skipped = 0

    for t in range(actual_T):
        jt = joints_norm[t].copy()  # (24, 2)

        # Drop frames where any required joint is out of the image
        req = jt[REQUIRED_JOINTS]
        if np.any(req < 0.0) or np.any(req > 1.0):
            n_skipped += 1
            continue

        # Clamp all other joints to [0, 1]
        jt = np.clip(jt, 0.0, 1.0)

        # Flip, resize, save JPEG
        frame = cv2.flip(frames[t], 1)
        frame = cv2.resize(frame, (OUTPUT_W, OUTPUT_H), interpolation=cv2.INTER_LINEAR)

        jpeg_name = f"{prefix}_f{t:04d}.jpg"
        jpeg_path = os.path.join(out_frames_dir, jpeg_name)
        cv2.imwrite(jpeg_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])

        row = [jpeg_path] + jt.flatten().tolist()
        rows.append(row)

    return rows, n_skipped


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess SURREAL dataset for PoseNet training"
    )
    parser.add_argument(
        "data_root",
        help="Path to surreal_data/ (should contain a cmu/ subdirectory)",
    )
    parser.add_argument(
        "out_dir",
        help="Output directory; frames/ subfolder + index CSVs created here",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        help="Which splits to process (default: train val)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel worker processes (default: 4; match --cpus-per-task in sbatch)",
    )
    args = parser.parse_args()

    out_frames_dir = os.path.join(args.out_dir, "frames")
    os.makedirs(out_frames_dir, exist_ok=True)

    # CSV header
    header = ["jpeg_path"] + [
        f"j{j}_{ax}" for j in range(24) for ax in ("x", "y")
    ]

    for split in args.splits:
        mp4_paths = sorted(
            glob.glob(os.path.join(args.data_root, "**", "*.mp4"), recursive=True)
        )
        sep = os.sep
        mp4_paths = [
            p for p in mp4_paths
            if f"{sep}{split}{sep}" in p or f"/{split}/" in p
        ]
        print(f"\n[{split}]  {len(mp4_paths):,} mp4 files  ({args.workers} workers)")

        # Build the arg list for pool workers, skipping already-processed clips.
        # A clip is considered done if its first frame (f0000) exists on disk.
        worker_args = []
        n_already_done = 0
        for mp4_path in mp4_paths:
            mat_path = mp4_path.replace(".mp4", "_info.mat")
            if not os.path.exists(mat_path):
                continue
            parts  = mp4_path.replace("\\", "/").split("/")
            prefix = "_".join(parts[-4:]).replace(".mp4", "")
            first_frame = os.path.join(out_frames_dir, f"{prefix}_f0000.jpg")
            if os.path.exists(first_frame):
                n_already_done += 1
                continue
            worker_args.append((mp4_path, mat_path, out_frames_dir, prefix))

        if n_already_done:
            print(f"  Resuming: {n_already_done:,} clips already done, "
                  f"{len(worker_args):,} remaining")

        csv_path = os.path.join(args.out_dir, f"index_{split}.csv")

        total_samples = 0
        total_skipped = 0

        # Append to existing CSV if resuming; write fresh header only for new files.
        csv_exists = os.path.exists(csv_path)
        with open(csv_path, "a" if csv_exists else "w", newline="") as f:
            writer = csv.writer(f)
            if not csv_exists:
                writer.writerow(header)

            with mp.Pool(processes=args.workers) as pool:
                for i, (rows, n_skipped) in enumerate(
                    pool.imap(_process_clip, worker_args, chunksize=8)
                ):
                    writer.writerows(rows)
                    total_samples += len(rows)
                    total_skipped += n_skipped

                    if (i + 1) % 500 == 0 or (i + 1) == len(worker_args):
                        print(
                            f"  {i+1:6,}/{len(worker_args):,} clips"
                            f"  {total_samples:,} kept"
                            f"  {total_skipped:,} skipped (OOF)"
                        )
                        sys.stdout.flush()

        print(f"[{split}]  {total_samples:,} samples -> {csv_path}")


if __name__ == "__main__":
    # Required on some Linux HPC environments when using multiprocessing
    mp.set_start_method("fork", force=True)
    main()
