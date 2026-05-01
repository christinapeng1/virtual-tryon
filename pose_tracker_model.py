#!/usr/bin/env python3
"""
Model-based drop-in replacement for pose_tracker.py.

Outputs the IDENTICAL 12-value CSV format per frame:
  lsx,lsy,rsx,rsy,lex,ley,rex,rey,lwx,lwy,rwx,rwy

To switch from MediaPipe to this model, change ONE line in pose_tracker.cpp:

  // Before (MediaPipe):
  const char* scriptPath = "../pose_tracker.py";

  // After (trained model):
  const char* scriptPath = "../pose_tracker_model.py";

Coordinate convention vs. MediaPipe
-------------------------------------
MediaPipe's landmark 11 = anatomical left shoulder.  On a horizontally-flipped
(selfie) frame the anatomical left appears on the VISUAL RIGHT, so pose_tracker.py
had to explicitly swap indices:
    left_shoulder  = landmark[12]   # visual-left  is actually landmark 12
    right_shoulder = landmark[11]   # visual-right is actually landmark 11

This model was trained on pre-flipped frames (selfie orientation), so:
    SMPL joint 16 → visual LEFT  shoulder  (no swap needed)
    SMPL joint 17 → visual RIGHT shoulder
    SMPL joint 18 → visual LEFT  elbow
    SMPL joint 19 → visual RIGHT elbow
    SMPL joint 20 → visual LEFT  wrist
    SMPL joint 21 → visual RIGHT wrist

Setup
------
  export POSENET_CHECKPOINT=/path/to/train/checkpoints/best.pt
  # or edit MODEL_CHECKPOINT below
"""

import os
import sys

os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout = open(sys.stdout.fileno(), "w", 1)   # line-buffered stdout
sys.stderr = open(sys.stderr.fileno(), "w", 1)

import cv2
import torch
import torchvision.transforms as T
from PIL import Image

# Add train/ to path so we can import model.py without installing it
_TRAIN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train")
sys.path.insert(0, _TRAIN_DIR)

from model import PoseNet, POSE_TRACKER_JOINTS  # noqa: E402

# ── Checkpoint path ────────────────────────────────────────────────────────────
# Override via environment variable or edit this string directly.
MODEL_CHECKPOINT = os.environ.get(
    "POSENET_CHECKPOINT",
    os.path.join(_TRAIN_DIR, "checkpoints", "best.pt"),
)

if not os.path.exists(MODEL_CHECKPOINT):
    print(
        f"Error: model checkpoint not found at '{MODEL_CHECKPOINT}'.\n"
        f"Set the POSENET_CHECKPOINT environment variable or train the model first.",
        file=sys.stderr,
        flush=True,
    )
    sys.exit(1)

# ── Load model ─────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ckpt     = torch.load(MODEL_CHECKPOINT, map_location=device)
backbone = ckpt.get("backbone", "mobilenet_v3_small")
model    = PoseNet(backbone=backbone, pretrained=False).to(device)
model.load_state_dict(ckpt["model"])
model.eval()

print(f"PoseNet ({backbone}) loaded from {MODEL_CHECKPOINT}", file=sys.stderr, flush=True)
print(f"Device: {device}", file=sys.stderr, flush=True)

# ── Pre-processing (must match training) ──────────────────────────────────────
# Resize to 192×256 (H×W), convert to tensor, apply ImageNet normalisation.
preprocess = T.Compose([
    T.Resize((192, 256), antialias=True),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

# ── Camera ─────────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera", file=sys.stderr, flush=True)
    sys.exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

# ── Inference loop ─────────────────────────────────────────────────────────────
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Selfie flip — exactly as in training data (preprocess.py)
    frame = cv2.flip(frame, 1)

    # BGR → RGB PIL image → normalised tensor
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = preprocess(Image.fromarray(rgb)).unsqueeze(0).to(device)

    with torch.no_grad():
        joints = model(tensor)[0].cpu().numpy()  # (24, 2)

    # Index directly by SMPL joint number — no swapping needed.
    # (See module docstring for why this differs from pose_tracker.py.)
    ls = joints[POSE_TRACKER_JOINTS["left_shoulder"]]   # SMPL 16
    rs = joints[POSE_TRACKER_JOINTS["right_shoulder"]]  # SMPL 17
    le = joints[POSE_TRACKER_JOINTS["left_elbow"]]      # SMPL 18
    re = joints[POSE_TRACKER_JOINTS["right_elbow"]]     # SMPL 19
    lw = joints[POSE_TRACKER_JOINTS["left_wrist"]]      # SMPL 20
    rw = joints[POSE_TRACKER_JOINTS["right_wrist"]]     # SMPL 21

    # Output format is identical to pose_tracker.py — pose_tracker.cpp reads
    # this with sscanf(buf, "%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f", ...)
    print(
        f"{ls[0]:.4f},{ls[1]:.4f},"
        f"{rs[0]:.4f},{rs[1]:.4f},"
        f"{le[0]:.4f},{le[1]:.4f},"
        f"{re[0]:.4f},{re[1]:.4f},"
        f"{lw[0]:.4f},{lw[1]:.4f},"
        f"{rw[0]:.4f},{rw[1]:.4f}"
    )

cap.release()
