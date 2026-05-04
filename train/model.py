"""
PoseNet — MobileNet backbone + regression head for 24 SMPL joint prediction.

Output: (B, 24, 2) tensor of (x, y) coordinates, each in [0, 1].
Coordinates are in *selfie/mirror* space (training data was pre-flipped),
so joint 16 is always on the visual LEFT of the frame.
"""

import torch
import torch.nn as nn
import torchvision.models as models

# ── Constants ─────────────────────────────────────────────────────────────────

NUM_JOINTS = 24

# Model input size (H, W).  Matches SURREAL's 240×320 aspect ratio (4:3)
# and is the size frames are saved as by preprocess.py.
INPUT_SIZE = (192, 256)

# ── SMPL joint reference ──────────────────────────────────────────────────────

SMPL_JOINT_NAMES = [
    "Pelvis",     # 0
    "L_Hip",      # 1
    "R_Hip",      # 2
    "Spine1",     # 3
    "L_Knee",     # 4
    "R_Knee",     # 5
    "Spine2",     # 6
    "L_Ankle",    # 7
    "R_Ankle",    # 8
    "Spine3",     # 9
    "L_Foot",     # 10
    "R_Foot",     # 11
    "Neck",       # 12
    "L_Collar",   # 13
    "R_Collar",   # 14
    "Head",       # 15
    "L_Shoulder", # 16  ← visual left  in selfie view
    "R_Shoulder", # 17  ← visual right in selfie view
    "L_Elbow",    # 18
    "R_Elbow",    # 19
    "L_Wrist",    # 20
    "R_Wrist",    # 21
    "L_Hand",     # 22
    "R_Hand",     # 23
]

# Pairs to swap when applying an *additional* horizontal flip on top of
# already-flipped (selfie) training frames.  Not used in the main pipeline
# (the initial selfie flip only needs x-mirroring, no pair swap), but
# included here for completeness and any future augmentation.
SMPL_FLIP_PAIRS = [
    (1,  2),  # L_Hip      ↔ R_Hip
    (4,  5),  # L_Knee     ↔ R_Knee
    (7,  8),  # L_Ankle    ↔ R_Ankle
    (10, 11), # L_Foot     ↔ R_Foot
    (13, 14), # L_Collar   ↔ R_Collar
    (16, 17), # L_Shoulder ↔ R_Shoulder
    (18, 19), # L_Elbow    ↔ R_Elbow
    (20, 21), # L_Wrist    ↔ R_Wrist
    (22, 23), # L_Hand     ↔ R_Hand
]

# The six joints forwarded to pose_tracker_model.py → pose_tracker.cpp.
# Because training images are pre-flipped, "L_" here means VISUAL left
# (what appears on the left of the screen in selfie view).
# No index-swapping is needed — contrast with the MediaPipe version which
# had to swap 11↔12 etc. because of a coordinate-convention mismatch.
POSE_TRACKER_JOINTS = {
    "left_shoulder":  16,
    "right_shoulder": 17,
    "left_elbow":     18,
    "right_elbow":    19,
    "left_wrist":     20,
    "right_wrist":    21,
}

# ── Model ─────────────────────────────────────────────────────────────────────

class PoseNet(nn.Module):
    """
    Single-frame, single-person pose estimator.

    Architecture
    ------------
    MobileNetV2/V3 feature extractor (ImageNet-pretrained, fine-tuned)
        → AdaptiveAvgPool2d(1)          # collapse spatial dims
        → Linear(C, 512) + ReLU + Dropout(0.2)
        → Linear(512, 48) + Sigmoid     # 24 joints × 2 coords, in [0, 1]

    Backbone options and their feature channel counts
    --------------------------------------------------
    mobilenet_v3_small  576   fastest,  ~2.5 M params
    mobilenet_v3_large  960   balanced, ~5.4 M params
    mobilenet_v2       1280   largest,  ~3.4 M params
    """

    _BACKBONES: dict = {
        "mobilenet_v3_small": (
            models.mobilenet_v3_small,
            models.MobileNet_V3_Small_Weights.DEFAULT,
            576,
        ),
        "mobilenet_v3_large": (
            models.mobilenet_v3_large,
            models.MobileNet_V3_Large_Weights.DEFAULT,
            960,
        ),
        "mobilenet_v2": (
            models.mobilenet_v2,
            models.MobileNet_V2_Weights.DEFAULT,
            1280,
        ),
    }

    def __init__(self, backbone: str = "mobilenet_v3_small", pretrained: bool = True):
        super().__init__()

        if backbone not in self._BACKBONES:
            raise ValueError(
                f"backbone must be one of {list(self._BACKBONES)}, got '{backbone}'"
            )

        model_fn, weights_enum, feature_dim = self._BACKBONES[backbone]
        base = model_fn(weights=weights_enum if pretrained else None)

        # Strip the original classifier; keep only the conv feature extractor.
        self.backbone = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)   # (B, C, H, W) → (B, C, 1, 1)

        self.head = nn.Sequential(
            nn.Flatten(),                          # (B, C, 1, 1) → (B, C)
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, NUM_JOINTS * 2),        # (B, 48)
            nn.Sigmoid(),                          # normalize output to [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x       : (B, 3, H, W)  ImageNet-normalized input
        returns : (B, 24, 2)    (x, y) per SMPL joint, values in [0, 1]
        """
        x = self.backbone(x)
        x = self.pool(x)
        x = self.head(x)
        return x.view(-1, NUM_JOINTS, 2)
