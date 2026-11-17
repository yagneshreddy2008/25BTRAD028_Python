"""
Hand Detector Module for SignBridge AI.
Wraps MediaPipe HandLandmarker for detecting Left and Right hand landmarks in OpenCV frames.
"""

import os
from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Default model path
MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")


class HandDetector:
    def __init__(self, model_path: Optional[str] = None, num_hands: int = 2, min_confidence: float = 0.5):
        """
        Initializes MediaPipe HandLandmarker.
        """
        self.model_path = model_path or MODEL_PATH
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"HandLandmarker model not found at {self.model_path}. "
                "Ensure hand_landmarker.task is present in the backend directory."
            )

        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=num_hands,
            min_hand_detection_confidence=min_confidence,
            min_hand_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

        # 21 Landmark connection pairs for drawing
        self.connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),        # Index
            (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
            (9, 13), (13, 14), (14, 15), (15, 16), # Ring
            (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
            (0, 17)                                 # Palm base
        ]

    def detect_landmarks(
        self,
        frame: np.ndarray,
        draw: bool = False
    ) -> Tuple[Optional[List[Dict[str, float]]], Optional[List[Dict[str, float]]], np.ndarray, int]:
        """
        Detects hand landmarks from an OpenCV BGR frame.

        Returns:
            (left_hand_landmarks, right_hand_landmarks, annotated_frame, hands_count)
        """
        if frame is None or frame.size == 0:
            return None, None, frame, 0

        h, w = frame.shape[:2]
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Run inference
        detection_result = self.detector.detect(mp_image)

        left_hand = None
        right_hand = None
        annotated = frame.copy() if draw else frame
        hands_count = 0

        if detection_result and detection_result.hand_landmarks:
            hands_count = len(detection_result.hand_landmarks)

            for i, hand_lms in enumerate(detection_result.hand_landmarks):
                # Handedness label: 'Left' or 'Right'
                handedness = "Right"
                if detection_result.handedness and i < len(detection_result.handedness):
                    handedness = detection_result.handedness[i][0].category_name

                # Extract landmark list
                lm_list = [{"x": float(lm.x), "y": float(lm.y), "z": float(lm.z)} for lm in hand_lms]

                if handedness == "Left":
                    left_hand = lm_list
                else:
                    right_hand = lm_list

                # Optional visualization
                if draw:
                    color = (255, 200, 0) if handedness == "Left" else (0, 200, 255)
                    # Draw skeletal connections
                    for start_idx, end_idx in self.connections:
                        if start_idx < len(hand_lms) and end_idx < len(hand_lms):
                            pt1 = (int(hand_lms[start_idx].x * w), int(hand_lms[start_idx].y * h))
                            pt2 = (int(hand_lms[end_idx].x * w), int(hand_lms[end_idx].y * h))
                            cv2.line(annotated, pt1, pt2, color, 2)

                    # Draw landmark points
                    for lm in hand_lms:
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(annotated, (cx, cy), 4, (255, 255, 255), -1)
                        cv2.circle(annotated, (cx, cy), 5, color, 1)

        return left_hand, right_hand, annotated, hands_count

    def close(self):
        """Release underlying detector resources."""
        if hasattr(self, "detector") and self.detector:
            self.detector.close()
