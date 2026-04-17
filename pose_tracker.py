#!/usr/bin/env python3

import cv2
import sys
import os
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
    model_complexity=0,
    smooth_landmarks=False,
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

        if results.pose_landmarks:
            # MediaPipe Pose landmarks:
            # 11 = left shoulder
            # 12 = right shoulder
            left_shoulder = results.pose_landmarks.landmark[11]
            right_shoulder = results.pose_landmarks.landmark[12]

            left_shoulder_x = left_shoulder.x
            left_shoulder_y = left_shoulder.y
            right_shoulder_x = right_shoulder.x
            right_shoulder_y = right_shoulder.y

        print(
            f"{left_shoulder_x:.4f},{left_shoulder_y:.4f},"
            f"{right_shoulder_x:.4f},{right_shoulder_y:.4f}"
        )
        sys.stdout.flush()

cap.release()