#!/usr/bin/env python3

import cv2
import sys
import os
import math
import traceback

# Disable buffering so output appears immediately
os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout = open(sys.stdout.fileno(), "w", 1)
sys.stderr = open(sys.stderr.fileno(), "w", 1)

try:
    from mediapipe import solutions
    import mediapipe as mp
except ImportError as e:
    print(f"Error: MediaPipe not installed properly. {e}", file=sys.stderr, flush=True)
    traceback.print_exc()
    sys.exit(1)

mp_pose = solutions.pose

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera", file=sys.stderr, flush=True)
    sys.exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

with mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
) as pose:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Selfie view so it matches the C++ display
        frame = cv2.flip(frame, 1)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        # left + right shoulder x and y coordinates
        lsx=lsy=rsx=rsy=0.0
        
        # left + right shoulder x and y world coordinates
        wlsx=wlsy=wlsz=wrsx=wrsy=wrsz=0.0

        # rotation and visibility
        yaw=vis=0.0
        
        world_shoulder_width = 0.0

        if results.pose_landmarks and results.pose_world_landmarks:
            lm  = results.pose_landmarks.landmark
            wlm = results.pose_world_landmarks.landmark


            # MediaPipe Pose landmarks:
            # 11 = left shoulder
            # 12 = right shoulder
            # 23 = left hip
            # 24 = right hip
            ls, rs = lm[11], lm[12]
            wls, wrs = wlm[11], wlm[12]

            lsx, lsy = ls.x, ls.y
            rsx, rsy = rs.x, rs.y

            # 3D world shoulder positions
            wlsx, wlsy, wlsz = wls.x, wls.y, wls.z
            wrsx, wrsy, wrsz = wrs.x, wrs.y, wrs.z

            # True 3D shoulder width so it doesn't collapse on side view
            world_shoulder_width = math.dist((wls.x, wls.y, wls.z), (wrs.x, wrs.y, wrs.z))

            # Chest yaw
            wlh, wrh = wlm[23], wlm[24]

            sv = (wrs.x-wls.x, wrs.y-wls.y, wrs.z-wls.z)
            ms = ((wls.x+wrs.x)*0.5, (wls.y+wrs.y)*0.5, (wls.z+wrs.z)*0.5)
            mh = ((wlh.x+wrh.x)*0.5, (wlh.y+wrh.y)*0.5, (wlh.z+wrh.z)*0.5)
            spv = (mh[0]-ms[0], mh[1]-ms[1], mh[2]-ms[2])

            cn_x = sv[1]*spv[2] - sv[2]*spv[1]
            cn_z = sv[0]*spv[1] - sv[1]*spv[0]
            yaw = math.degrees(math.atan2(cn_x, cn_z))
            vis = (ls.visibility + rs.visibility) * 0.5

        print(
            f"{lsx:.4f},{lsy:.4f},{rsx:.4f},{rsy:.4f},"
            f"{wlsx:.4f},{wlsy:.4f},{wlsz:.4f},"
            f"{wrsx:.4f},{wrsy:.4f},{wrsz:.4f},"
            f"{yaw:.2f},{vis:.3f},{world_shoulder_width:.4f}"
        )
        sys.stdout.flush()

cap.release()