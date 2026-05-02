#!/usr/bin/env python3

import cv2
import sys
import os
import traceback
import math

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

        left_shoulder_x = 0.0
        left_shoulder_y = 0.0
        right_shoulder_x = 0.0
        right_shoulder_y = 0.0
        left_elbow_x = 0.0
        left_elbow_y = 0.0
        right_elbow_x = 0.0
        right_elbow_y = 0.0
        left_wrist_x = 0.0
        left_wrist_y = 0.0
        right_wrist_x = 0.0
        right_wrist_y = 0.0
        left_hip_x = 0.0
        left_hip_y = 0.0
        right_hip_x = 0.0
        right_hip_y = 0.0
        nose_x = 0.0
        nose_y = 0.0

        yaw = 0.0
        visibility = 0.0
        world_shoulder_width = 0.0

        if results.pose_landmarks and results.pose_world_landmarks:
            # MediaPipe Pose landmarks:
            # 0 = nose
            # 11 = left shoulder, 12 = right shoulder
            # 13 = left elbow, 14 = right elbow
            # 15 = left wrist, 16 = right wrist
            # 23 = left hip, 24 = right hip
            # NOTE: Frame is flipped for selfie view, so visually LEFT and RIGHT are swapped
            nose = results.pose_landmarks.landmark[0]              # Nose for rotation direction
            left_shoulder = results.pose_landmarks.landmark[12]   # Visually left is landmark 12
            right_shoulder = results.pose_landmarks.landmark[11]  # Visually right is landmark 11
            left_elbow = results.pose_landmarks.landmark[14]      # Visually left is landmark 14
            right_elbow = results.pose_landmarks.landmark[13]     # Visually right is landmark 13
            left_wrist = results.pose_landmarks.landmark[16]      # Visually left is landmark 16
            right_wrist = results.pose_landmarks.landmark[15]     # Visually right is landmark 15
            left_hip = results.pose_landmarks.landmark[24]        # Visually left is landmark 24
            right_hip = results.pose_landmarks.landmark[23]       # Visually right is landmark 23

            nose_x = nose.x
            nose_y = nose.y
            left_shoulder_x = left_shoulder.x
            left_shoulder_y = left_shoulder.y
            right_shoulder_x = right_shoulder.x
            right_shoulder_y = right_shoulder.y
            left_elbow_x = left_elbow.x
            left_elbow_y = left_elbow.y
            right_elbow_x = right_elbow.x
            right_elbow_y = right_elbow.y
            left_wrist_x = left_wrist.x
            left_wrist_y = left_wrist.y
            right_wrist_x = right_wrist.x
            right_wrist_y = right_wrist.y
            left_hip_x = left_hip.x
            left_hip_y = left_hip.y
            right_hip_x = right_hip.x
            right_hip_y = right_hip.y

            wlm = results.pose_world_landmarks.landmark

            wls = wlm[12]   # visually left after flip
            wrs = wlm[11]   # visually right after flip
            wlh = wlm[24]
            wrh = wlm[23]

            world_shoulder_width = math.dist(
                (wls.x, wls.y, wls.z),
                (wrs.x, wrs.y, wrs.z)
            )

            sv = (wrs.x - wls.x, wrs.y - wls.y, wrs.z - wls.z)
            ms = ((wls.x + wrs.x) * 0.5, (wls.y + wrs.y) * 0.5, (wls.z + wrs.z) * 0.5)
            mh = ((wlh.x + wrh.x) * 0.5, (wlh.y + wrh.y) * 0.5, (wlh.z + wrh.z) * 0.5)
            spv = (mh[0] - ms[0], mh[1] - ms[1], mh[2] - ms[2])

            cn_x = sv[1] * spv[2] - sv[2] * spv[1]
            cn_z = sv[0] * spv[1] - sv[1] * spv[0]

            yaw = math.degrees(math.atan2(cn_x, cn_z))
            visibility = (left_shoulder.visibility + right_shoulder.visibility) * 0.5

        print(
            f"{nose_x:.4f},{nose_y:.4f},"
            f"{left_shoulder_x:.4f},{left_shoulder_y:.4f},"
            f"{right_shoulder_x:.4f},{right_shoulder_y:.4f},"
            f"{left_elbow_x:.4f},{left_elbow_y:.4f},"
            f"{right_elbow_x:.4f},{right_elbow_y:.4f},"
            f"{left_wrist_x:.4f},{left_wrist_y:.4f},"
            f"{right_wrist_x:.4f},{right_wrist_y:.4f},"
            f"{left_hip_x:.4f},{left_hip_y:.4f},"
            f"{right_hip_x:.4f},{right_hip_y:.4f},"
            f"{yaw:.2f},{visibility:.3f},{world_shoulder_width:.4f}"
        )
        sys.stdout.flush()

cap.release()