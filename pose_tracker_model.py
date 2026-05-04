#!/usr/bin/env python3
"""
Drop-in replacement for pose_tracker.py that uses the trained PoseNet
checkpoint instead of MediaPipe.

Outputs the same 21-float CSV line per frame:
  nose_x, nose_y,
  l_shoulder_x, l_shoulder_y, r_shoulder_x, r_shoulder_y,
  l_elbow_x,    l_elbow_y,    r_elbow_x,    r_elbow_y,
  l_wrist_x,    l_wrist_y,    r_wrist_x,    r_wrist_y,
  l_hip_x,      l_hip_y,      r_hip_x,      r_hip_y,
  yaw, visibility, world_shoulder_width

SMPL joint index mapping used here (selfie/mirror convention):
  0=Pelvis, 1=L_Hip, 2=R_Hip, ..., 15=Head,
  16=L_Shoulder, 17=R_Shoulder, 18=L_Elbow, 19=R_Elbow,
  20=L_Wrist,    21=R_Wrist
"""

import sys
import os
import math
import traceback

os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout = open(sys.stdout.fileno(), "w", 1)
sys.stderr = open(sys.stderr.fileno(), "w", 1)

# Add train/ to path so we can import model.py
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR  = os.path.join(SCRIPT_DIR, "train")
sys.path.insert(0, TRAIN_DIR)

try:
    import torch
    import torchvision.transforms as T
    import cv2
    from model import PoseNet, INPUT_SIZE
except ImportError as e:
    print(f"Error: missing dependency. {e}", file=sys.stderr, flush=True)
    traceback.print_exc()
    sys.exit(1)

# ── Config ────────────────────────────────────────────────────────────────────

CHECKPOINT = os.path.join(SCRIPT_DIR, "train", "checkpoints-full", "best.pt")
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
DEBUG      = os.environ.get("POSE_DEBUG", "0") == "1"

# ImageNet normalisation — must match training preprocessing
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

TRANSFORM = T.Compose([
    T.ToTensor(),
    T.Normalize(_MEAN, _STD),
])

# ── Load model ────────────────────────────────────────────────────────────────

print(f"Loading PoseNet from {CHECKPOINT} …", file=sys.stderr, flush=True)

try:
    ckpt     = torch.load(CHECKPOINT, map_location=DEVICE)
    backbone = ckpt.get("backbone", "mobilenet_v3_small")
    print(f"Checkpoint backbone: {backbone}", file=sys.stderr, flush=True)

    model = PoseNet(backbone=backbone, pretrained=False).to(DEVICE)
    state = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
    model.load_state_dict(state)
    model.eval()
    print("Model loaded.", file=sys.stderr, flush=True)
except Exception as e:
    print(f"Error loading checkpoint: {e}", file=sys.stderr, flush=True)
    traceback.print_exc()
    sys.exit(1)

# ── Camera ────────────────────────────────────────────────────────────────────

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera", file=sys.stderr, flush=True)
    sys.exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

H_IN, W_IN = INPUT_SIZE   # (192, 256) — must match training

# ── Inference loop ────────────────────────────────────────────────────────────

with torch.inference_mode():
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Selfie view — matches training data orientation
        frame = cv2.flip(frame, 1)

        # Resize to model input size (H×W = 192×256)
        resized = cv2.resize(frame, (W_IN, H_IN))
        rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # PIL-free path: convert numpy HWC uint8 → CHW float tensor manually
        from PIL import Image
        pil = Image.fromarray(rgb)
        inp = TRANSFORM(pil).unsqueeze(0).to(DEVICE)   # (1, 3, H, W)

        joints = model(inp).squeeze(0).cpu().numpy()   # (24, 2)  values in [0,1]

        # ── Extract required joints ───────────────────────────────────────────
        # SMPL joint indices in selfie/mirror convention (training data pre-flipped).
        # joint 16 = L_Shoulder should be on visual LEFT of selfie frame.
        # If the shirt appears mirrored, swap LS/RS and LE/RE and LW/RW below.
        HEAD  = 15
        LS, RS = 17, 16   # R_Shoulder appears on visual left after selfie flip
        LE, RE = 19, 18
        LW, RW = 21, 20
        LH, RH =  2,  1

        nx,  ny  = joints[HEAD]
        lsx, lsy = joints[LS]
        rsx, rsy = joints[RS]
        lex, ley = joints[LE]
        rex, rey = joints[RE]
        lwx, lwy = joints[LW]
        rwx, rwy = joints[RW]
        lhx, lhy = joints[LH]
        rhx, rhy = joints[RH]

        # ── Derived fields ────────────────────────────────────────────────────
        # Yaw: approximate from shoulder horizontal separation relative to
        # a reference width.  Positive = turned right.
        shoulder_dx = rsx - lsx
        shoulder_dy = rsy - lsy
        yaw = math.degrees(math.atan2(shoulder_dy, shoulder_dx))

        visibility = 1.0   # no per-landmark confidence from this model

        # World shoulder width: rough estimate — wider pixel gap ≈ facing camera
        # Normalise pixel width by frame width fraction; scale to ~0.4 m typical
        world_shoulder_width = max(shoulder_dx * 0.8, 0.15)

        print(
            f"{nx:.4f},{ny:.4f},"
            f"{lsx:.4f},{lsy:.4f},"
            f"{rsx:.4f},{rsy:.4f},"
            f"{lex:.4f},{ley:.4f},"
            f"{rex:.4f},{rey:.4f},"
            f"{lwx:.4f},{lwy:.4f},"
            f"{rwx:.4f},{rwy:.4f},"
            f"{lhx:.4f},{lhy:.4f},"
            f"{rhx:.4f},{rhy:.4f},"
            f"{yaw:.4f},{visibility:.4f},{world_shoulder_width:.4f}"
        )
        sys.stdout.flush()

        # ── Debug visualization ───────────────────────────────────────────────
        if DEBUG:
            vis = cv2.resize(frame, (W_IN, H_IN)).copy()
            h_vis, w_vis = vis.shape[:2]

            JOINT_COLORS = {
                "head":       (255, 255,   0),
                "l_shoulder": (  0, 255,   0),
                "r_shoulder": (  0, 255,   0),
                "l_elbow":    (  0, 255, 255),
                "r_elbow":    (  0, 255, 255),
                "l_wrist":    (  0, 128, 255),
                "r_wrist":    (  0, 128, 255),
                "l_hip":      (255,   0, 255),
                "r_hip":      (255,   0, 255),
            }
            named = [
                ("head",       nx,  ny),
                ("l_shoulder", lsx, lsy),
                ("r_shoulder", rsx, rsy),
                ("l_elbow",    lex, ley),
                ("r_elbow",    rex, rey),
                ("l_wrist",    lwx, lwy),
                ("r_wrist",    rwx, rwy),
                ("l_hip",      lhx, lhy),
                ("r_hip",      rhx, rhy),
            ]
            # Skeleton edges to draw
            edges = [
                ("l_shoulder", "r_shoulder"),
                ("l_shoulder", "l_elbow"),
                ("l_elbow",    "l_wrist"),
                ("r_shoulder", "r_elbow"),
                ("r_elbow",    "r_wrist"),
                ("l_shoulder", "l_hip"),
                ("r_shoulder", "r_hip"),
                ("l_hip",      "r_hip"),
                ("head",       "l_shoulder"),
                ("head",       "r_shoulder"),
            ]
            coords = {name: (int(x * w_vis), int(y * h_vis)) for name, x, y in named}

            for a, b in edges:
                cv2.line(vis, coords[a], coords[b], (180, 180, 180), 1)

            for name, x, y in named:
                px, py = int(x * w_vis), int(y * h_vis)
                color  = JOINT_COLORS.get(name, (255, 255, 255))
                cv2.circle(vis, (px, py), 5, color, -1)
                cv2.putText(vis, f"{name} ({x:.2f},{y:.2f})",
                            (px + 6, py), cv2.FONT_HERSHEY_SIMPLEX,
                            0.32, color, 1, cv2.LINE_AA)

            cv2.putText(vis, f"yaw={yaw:.1f} wsw={world_shoulder_width:.2f}",
                        (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
            cv2.imshow("PoseNet debug", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

cap.release()
if DEBUG:
    cv2.destroyAllWindows()
