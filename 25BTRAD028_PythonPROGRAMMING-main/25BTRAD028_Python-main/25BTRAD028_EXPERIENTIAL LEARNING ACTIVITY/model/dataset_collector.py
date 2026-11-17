"""
Dataset Collector Module for SignBridge AI.
Supports both live webcam collection with MediaPipe and ISL kinematic dataset generation.
Includes 20 rich Indian Sign Language (ISL) gestures with full ambidextrous (Left/Right)
and static-hold / dynamic-transition variation.
"""

import os
import sys
import time
import json
import argparse
from typing import List, Dict, Optional, Tuple
import cv2
import numpy as np

# Ensure backend modules can be imported
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BACKEND_DIR = os.path.join(PROJECT_ROOT, "backend")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from preprocessing import (
    process_frame_landmarks,
    normalize_hand_landmarks,
    augment_sequence,
    TOTAL_FRAME_FEATURES,
    DEFAULT_SEQUENCE_LENGTH,
)
from hand_detector import HandDetector

DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
RAW_DIR = os.path.join(DATASET_DIR, "raw")
PROCESSED_DIR = os.path.join(DATASET_DIR, "processed")
LABELS_FILE = os.path.join(PROJECT_ROOT, "model", "labels.json")

# 20 Target Indian Sign Language Vocabulary Classes
ISL_GESTURES_20 = [
    "HELLO",
    "NAMASTE",
    "THANK YOU",
    "PLEASE",
    "YES",
    "NO",
    "HELP",
    "HOW",
    "ARE",
    "YOU",
    "GOOD",
    "BAD",
    "LOVE",
    "PEACE",
    "STOP",
    "OK",
    "SORRY",
    "WATER",
    "WELCOME",
    "NEUTRAL",
]


def ensure_directories():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)


def load_labels() -> List[str]:
    if os.path.exists(LABELS_FILE):
        try:
            with open(LABELS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return list(ISL_GESTURES_20)


def save_labels(labels: List[str]):
    os.makedirs(os.path.dirname(LABELS_FILE), exist_ok=True)
    with open(LABELS_FILE, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2)


# ──────────────────────────────────────────────────────────
#  Anatomical 3D Hand Topology Builder (MediaPipe 21 Landmarks)
# ──────────────────────────────────────────────────────────

def build_hand_landmarks(
    thumb: str = "EXTENDED",
    index: str = "EXTENDED",
    middle: str = "EXTENDED",
    ring: str = "EXTENDED",
    pinky: str = "EXTENDED",
    rot_pitch: float = 0.0,
    rot_yaw: float = 0.0,
    rot_roll: float = 0.0,
    palm_curl: float = 0.0,
) -> np.ndarray:
    """
    Constructs an anatomically accurate 21-landmark 3D hand coordinates array
    matching real MediaPipe Hands topology and coordinate distributions.
    """
    pts = np.zeros((21, 3), dtype=np.float32)

    # 0: Wrist
    pts[0] = [0.0, 0.0, 0.0]

    # Palm MCP joints
    # 1: Thumb CMC, 2: Thumb MCP
    pts[1] = [-0.22, 0.18, -0.04]
    pts[2] = [-0.38, 0.32, -0.06]

    # 5: Index MCP
    pts[5] = [-0.28, 0.85, 0.00]
    # 9: Middle MCP (reference palm length = 1.0)
    pts[9] = [0.00, 1.00, 0.00]
    # 13: Ring MCP
    pts[13] = [0.26, 0.88, -0.02]
    # 17: Pinky MCP
    pts[17] = [0.48, 0.75, -0.04]

    # ── Thumb Joints (3: IP, 4: TIP) ──
    if thumb == "THUMB_UP":
        pts[3] = [-0.35, 0.65, 0.02]
        pts[4] = [-0.35, 1.05, 0.05]
    elif thumb == "THUMB_DOWN":
        pts[3] = [-0.35, -0.15, 0.02]
        pts[4] = [-0.35, -0.55, 0.05]
    elif thumb == "THUMB_OUT":
        pts[3] = [-0.65, 0.40, 0.00]
        pts[4] = [-0.95, 0.45, 0.00]
    elif thumb == "THUMB_WRAP":
        pts[3] = [-0.20, 0.45, 0.20]
        pts[4] = [-0.05, 0.48, 0.22]
    elif thumb == "PINCH":
        pts[3] = [-0.28, 0.70, 0.10]
        pts[4] = [-0.22, 0.90, 0.12]
    else:  # EXTENDED
        pts[3] = [-0.45, 0.55, 0.00]
        pts[4] = [-0.60, 0.80, 0.02]

    # ── Index Joints (6: PIP, 7: DIP, 8: TIP) ──
    if index == "EXTENDED":
        pts[6] = [-0.30, 1.25, 0.00]
        pts[7] = [-0.30, 1.60, 0.00]
        pts[8] = [-0.30, 1.95, 0.00]
    elif index == "FORWARD":
        pts[6] = [-0.28, 0.95, 0.35]
        pts[7] = [-0.28, 1.05, 0.70]
        pts[8] = [-0.28, 1.15, 1.05]
    elif index == "PINCH":
        pts[6] = [-0.28, 1.15, 0.15]
        pts[7] = [-0.26, 1.00, 0.18]
        pts[8] = [-0.22, 0.90, 0.12]
    elif index == "CURVED":
        pts[6] = [-0.28, 1.15, 0.25]
        pts[7] = [-0.28, 1.30, 0.45]
        pts[8] = [-0.28, 1.35, 0.60]
    else:  # CURLED
        pts[6] = [-0.28, 1.05, 0.22]
        pts[7] = [-0.26, 0.85, 0.28]
        pts[8] = [-0.24, 0.65, 0.20]

    # ── Middle Joints (10: PIP, 11: DIP, 12: TIP) ──
    if middle == "EXTENDED":
        pts[10] = [0.00, 1.45, 0.00]
        pts[11] = [0.00, 1.85, 0.00]
        pts[12] = [0.00, 2.20, 0.00]
    elif middle == "FORWARD":
        pts[10] = [0.00, 1.10, 0.35]
        pts[11] = [0.00, 1.20, 0.70]
        pts[12] = [0.00, 1.30, 1.05]
    elif middle == "CURVED":
        pts[10] = [0.00, 1.30, 0.25]
        pts[11] = [0.00, 1.45, 0.45]
        pts[12] = [0.00, 1.50, 0.60]
    else:  # CURLED
        pts[10] = [0.00, 1.20, 0.22]
        pts[11] = [0.00, 0.95, 0.28]
        pts[12] = [0.00, 0.70, 0.20]

    # ── Ring Joints (14: PIP, 15: DIP, 16: TIP) ──
    if ring == "EXTENDED":
        pts[14] = [0.28, 1.30, -0.02]
        pts[15] = [0.28, 1.65, -0.02]
        pts[16] = [0.28, 2.00, -0.02]
    elif ring == "CURVED":
        pts[14] = [0.28, 1.15, 0.25]
        pts[15] = [0.28, 1.30, 0.45]
        pts[16] = [0.28, 1.35, 0.60]
    else:  # CURLED
        pts[14] = [0.26, 1.08, 0.20]
        pts[15] = [0.25, 0.88, 0.25]
        pts[16] = [0.24, 0.65, 0.18]

    # ── Pinky Joints (18: PIP, 19: DIP, 20: TIP) ──
    if pinky == "EXTENDED":
        pts[18] = [0.54, 1.10, -0.04]
        pts[19] = [0.58, 1.40, -0.04]
        pts[20] = [0.62, 1.70, -0.04]
    elif pinky == "CURVED":
        pts[18] = [0.50, 0.95, 0.25]
        pts[19] = [0.52, 1.10, 0.45]
        pts[20] = [0.54, 1.15, 0.60]
    else:  # CURLED
        pts[18] = [0.48, 0.92, 0.18]
        pts[19] = [0.46, 0.75, 0.22]
        pts[20] = [0.44, 0.58, 0.15]

    # Apply 3D pitch/roll/yaw rotations
    if rot_pitch != 0.0:
        c, s = np.cos(rot_pitch), np.sin(rot_pitch)
        Rx = np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)
        pts = pts @ Rx.T

    if rot_yaw != 0.0:
        c, s = np.cos(rot_yaw), np.sin(rot_yaw)
        Ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
        pts = pts @ Ry.T

    if rot_roll != 0.0:
        c, s = np.cos(rot_roll), np.sin(rot_roll)
        Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
        pts = pts @ Rz.T

    return pts.astype(np.float32)


def mirror_hand_to_left(right_pts: np.ndarray) -> np.ndarray:
    """Mirrors a right hand landmark configuration to a left hand."""
    left = right_pts.copy()
    left[:, 0] = -left[:, 0]  # Flip X-axis
    return left


def normalize_pts_single(pts: np.ndarray) -> np.ndarray:
    """Applies standard normalization (wrist at origin, unit palm scale, clamped)."""
    if pts is None or np.allclose(pts, 0, atol=1e-5):
        return np.zeros(63, dtype=np.float32)
    pts = pts.copy()
    wrist = pts[0].copy()
    pts -= wrist
    palm_size = float(np.linalg.norm(pts[9]))
    if palm_size < 1e-3:
        palm_size = float(np.max(np.linalg.norm(pts, axis=1)))
    if palm_size < 1e-3:
        return np.zeros(63, dtype=np.float32)
    pts /= palm_size
    res = pts.flatten().astype(np.float32)
    np.clip(res, -5.0, 5.0, out=res)
    return res


# ──────────────────────────────────────────────────────────
#  20 Gesture Pose Generators (Time t in [0, 1])
# ──────────────────────────────────────────────────────────

def get_gesture_frames(
    gesture: str,
    t: float,
    is_left_dominant: bool = False,
    is_static_hold: bool = False,
    var: dict = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (left_pts, right_pts) for any of the 20 ISL gestures.
    Supports left-dominant hand mirroring and static hold vs dynamic motion.
    """
    ZERO = np.zeros((21, 3), dtype=np.float32)
    v = var or {}
    micro_jitter = v.get("jitter", 0.0)

    # Effective motion phase
    phase = 0.5 if is_static_hold else t

    if gesture == "HELLO":
        # Two upright open hands raised higher, distinct wide waving roll
        wave_r = 0.30 * np.sin(phase * np.pi * 3)
        wave_l = -wave_r
        R = build_hand_landmarks("EXTENDED", "EXTENDED", "EXTENDED", "EXTENDED", "EXTENDED", rot_roll=wave_r, rot_pitch=-0.2)
        R[:, 1] += 0.35  # Hands held higher (shoulder/head height)
        R[:, 0] += 0.25  # Spread wider
        L = mirror_hand_to_left(R)
        return L, R

    elif gesture == "NAMASTE":
        # Both hands pressed flat together tightly at chest center (zero lateral spread)
        R = build_hand_landmarks("EXTENDED", "EXTENDED", "EXTENDED", "EXTENDED", "EXTENDED", rot_yaw=0.15)
        R[:, 0] = 0.05   # Right right at center
        L = mirror_hand_to_left(R)
        L[:, 0] = -0.05  # Left right at center
        return L, R

    elif gesture == "THANK YOU":
        # Flat hand from chin moving forward and down
        pitch = -0.4 + 0.6 * phase
        fwd = 0.50 * phase
        H = build_hand_landmarks("EXTENDED", "EXTENDED", "EXTENDED", "EXTENDED", "EXTENDED", rot_pitch=pitch)
        H[:, 2] += fwd
        H[:, 1] -= 0.20 * phase
        if is_left_dominant:
            return mirror_hand_to_left(H), ZERO
        return ZERO, H

    elif gesture == "PLEASE":
        # Two hands: flat non-dominant palm, dominant hand rubbing circularly
        L = mirror_hand_to_left(build_hand_landmarks("EXTENDED", "EXTENDED", "EXTENDED", "EXTENDED", "EXTENDED", rot_pitch=0.4))
        rub_x = 0.08 * np.cos(phase * np.pi * 4)
        rub_y = 0.08 * np.sin(phase * np.pi * 4)
        R = build_hand_landmarks("EXTENDED", "EXTENDED", "EXTENDED", "EXTENDED", "EXTENDED")
        R[:, 0] += rub_x
        R[:, 1] += rub_y
        return L, R

    elif gesture == "YES":
        # Closed fist nodding up and down from wrist
        nod = 0.30 * np.sin(phase * np.pi * 4)
        H = build_hand_landmarks("THUMB_WRAP", "CURLED", "CURLED", "CURLED", "CURLED", rot_pitch=nod)
        if is_left_dominant:
            return mirror_hand_to_left(H), ZERO
        return ZERO, H

    elif gesture == "NO":
        # Index & Middle extended together, pronounced horizontal side-to-side wag
        wag = 0.35 * np.sin(phase * np.pi * 4.0)
        H = build_hand_landmarks("THUMB_WRAP", "EXTENDED", "EXTENDED", "CURLED", "CURLED", rot_yaw=wag)
        H[8, 0] = -0.15  # Index and middle close together
        H[12, 0] = -0.05
        if is_left_dominant:
            return mirror_hand_to_left(H), ZERO
        return ZERO, H

    elif gesture == "HELP":
        # Two hands: horizontal open flat palm, dominant thumbs-up fist lifting on top
        L = mirror_hand_to_left(build_hand_landmarks("EXTENDED", "EXTENDED", "EXTENDED", "EXTENDED", "EXTENDED", rot_pitch=0.7))
        lift = 0.25 * phase
        R = build_hand_landmarks("THUMB_UP", "CURLED", "CURLED", "CURLED", "CURLED")
        L[:, 1] += lift
        R[:, 1] += lift
        return L, R

    elif gesture == "HOW":
        # Both hands curved, rotating outward
        twist = 0.35 * np.sin(phase * np.pi * 2)
        R = build_hand_landmarks("EXTENDED", "CURVED", "CURVED", "CURVED", "CURVED", rot_roll=twist, rot_yaw=twist)
        L = mirror_hand_to_left(R)
        return L, R

    elif gesture == "ARE":
        # Flat hand sweeping smoothly forward
        fwd = 0.40 * phase
        H = build_hand_landmarks("EXTENDED", "EXTENDED", "EXTENDED", "EXTENDED", "EXTENDED")
        H[:, 2] += fwd
        if is_left_dominant:
            return mirror_hand_to_left(H), ZERO
        return ZERO, H

    elif gesture == "YOU":
        # Pointing index finger directly forward at camera
        thrust = 0.15 * phase
        H = build_hand_landmarks("THUMB_WRAP", "FORWARD", "CURLED", "CURLED", "CURLED")
        H[:, 2] += thrust
        if is_left_dominant:
            return mirror_hand_to_left(H), ZERO
        return ZERO, H

    elif gesture == "GOOD":
        # Thumbs up! Thumb pointing straight up, other 4 curled
        pump = 0.08 * np.sin(phase * np.pi * 2.5) if not is_static_hold else 0.0
        H = build_hand_landmarks("THUMB_UP", "CURLED", "CURLED", "CURLED", "CURLED")
        H[:, 1] += pump
        if is_left_dominant:
            return mirror_hand_to_left(H), ZERO
        return ZERO, H

    elif gesture == "BAD":
        # Thumbs down! Thumb pointing straight down, other 4 curled
        pump = -0.08 * np.sin(phase * np.pi * 2.5) if not is_static_hold else 0.0
        H = build_hand_landmarks("THUMB_DOWN", "CURLED", "CURLED", "CURLED", "CURLED")
        H[:, 1] += pump
        if is_left_dominant:
            return mirror_hand_to_left(H), ZERO
        return ZERO, H

    elif gesture == "LOVE":
        # 'I Love You': Thumb, Index, Pinky extended; Middle and Ring curled
        H = build_hand_landmarks("THUMB_OUT", "EXTENDED", "CURLED", "CURLED", "EXTENDED")
        if is_left_dominant:
            return mirror_hand_to_left(H), ZERO
        return ZERO, H

    elif gesture == "PEACE":
        # Victory V: Index and Middle spread widely in V; Thumb, Ring, Pinky curled; held steady
        H = build_hand_landmarks("THUMB_WRAP", "EXTENDED", "EXTENDED", "CURLED", "CURLED")
        # Distinct V-shape spread
        H[6, 0] -= 0.15; H[7, 0] -= 0.25; H[8, 0] -= 0.38
        H[10, 0] += 0.15; H[11, 0] += 0.25; H[12, 0] += 0.38
        if is_left_dominant:
            return mirror_hand_to_left(H), ZERO
        return ZERO, H

    elif gesture == "STOP":
        # Open vertical flat palm facing forward at camera
        H = build_hand_landmarks("THUMB_OUT", "EXTENDED", "EXTENDED", "EXTENDED", "EXTENDED")
        if is_left_dominant:
            return mirror_hand_to_left(H), ZERO
        return ZERO, H

    elif gesture == "OK":
        # Thumb & Index pinch circle; Middle, Ring, Pinky extended upright
        H = build_hand_landmarks("PINCH", "PINCH", "EXTENDED", "EXTENDED", "EXTENDED")
        if is_left_dominant:
            return mirror_hand_to_left(H), ZERO
        return ZERO, H

    elif gesture == "SORRY":
        # Fist rubbing chest in circular motion
        circle_x = 0.08 * np.cos(phase * np.pi * 3.5)
        circle_y = 0.08 * np.sin(phase * np.pi * 3.5)
        H = build_hand_landmarks("THUMB_WRAP", "CURLED", "CURLED", "CURLED", "CURLED")
        H[:, 0] += circle_x
        H[:, 1] += circle_y
        if is_left_dominant:
            return mirror_hand_to_left(H), ZERO
        return ZERO, H

    elif gesture == "WATER":
        # 'W' handshape: Index, Middle, Ring extended upright; Thumb holds Pinky
        H = build_hand_landmarks("THUMB_WRAP", "EXTENDED", "EXTENDED", "EXTENDED", "CURLED")
        if is_left_dominant:
            return mirror_hand_to_left(H), ZERO
        return ZERO, H

    elif gesture == "WELCOME":
        # Both hands open with palms facing upward, sweeping inward toward chest
        sweep = 0.20 * phase
        R = build_hand_landmarks("EXTENDED", "EXTENDED", "EXTENDED", "EXTENDED", "EXTENDED", rot_pitch=0.6, rot_roll=-0.2)
        R[:, 0] -= sweep
        L = mirror_hand_to_left(R)
        return L, R

    else:  # "NEUTRAL"
        # Idle resting state: no hands or relaxed low hands
        if np.random.rand() > 0.4:
            R = build_hand_landmarks("THUMB_WRAP", "CURLED", "CURLED", "CURLED", "CURLED", rot_pitch=1.2) * 0.4
            return ZERO, R
        return ZERO, ZERO


# ──────────────────────────────────────────────────────────
#  Dataset Generation Routine
# ──────────────────────────────────────────────────────────

def generate_isl_dataset(
    samples_per_gesture: int = 160,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    save_raw: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates a balanced, realistic ISL dataset for all 20 target vocabulary signs.
    Includes:
    - 50% Right-handed and 50% Left-handed variations for all single-handed signs.
    - 50% Static-hold variations and 50% Dynamic-transition variations.
    - Full landmark normalization matching real-time inference.
    - Gaussian jitter & temporal scale augmentation.
    """
    ensure_directories()
    labels = load_labels()
    save_labels(labels)

    print("=" * 65)
    print(f" SignBridge AI – ISL Dataset Generator (20 Vocabulary Classes)")
    print("=" * 65)
    print(f"[Dataset Generator] Classes ({len(labels)}): {labels}")
    print(f"[Dataset Generator] Generating {samples_per_gesture} samples per class...")

    all_sequences = []
    all_labels = []

    for class_idx, label in enumerate(labels):
        label_dir = os.path.join(RAW_DIR, label)
        if save_raw:
            os.makedirs(label_dir, exist_ok=True)

        for sample_i in range(samples_per_gesture):
            seq = np.zeros((sequence_length, TOTAL_FRAME_FEATURES), dtype=np.float32)

            # Diversity variations
            is_left_dom = (sample_i % 2 == 1)
            is_static_hold = ((sample_i // 2) % 2 == 1)

            speed = np.random.uniform(0.85, 1.15)
            scale = np.random.uniform(0.90, 1.10)
            jitter_amp = np.random.uniform(0.005, 0.015)

            var = {"jitter": jitter_amp}

            for t_idx in range(sequence_length):
                t_norm = (t_idx / (sequence_length - 1)) * speed
                t_norm = min(1.0, max(0.0, t_norm))

                L_pts, R_pts = get_gesture_frames(
                    gesture=label,
                    t=t_norm,
                    is_left_dominant=is_left_dom,
                    is_static_hold=is_static_hold,
                    var=var,
                )

                # Micro-jitter and scale
                if not np.allclose(L_pts, 0):
                    L_pts = (L_pts * scale) + np.random.normal(0, jitter_amp, L_pts.shape).astype(np.float32)
                if not np.allclose(R_pts, 0):
                    R_pts = (R_pts * scale) + np.random.normal(0, jitter_amp, R_pts.shape).astype(np.float32)

                left_norm = normalize_pts_single(L_pts)
                right_norm = normalize_pts_single(R_pts)

                seq[t_idx] = np.concatenate([left_norm, right_norm])

            # Augmentation
            seq = augment_sequence(seq, jitter_std=0.012, scale_range=(0.95, 1.05))

            if save_raw:
                sample_file = os.path.join(label_dir, f"sample_{sample_i:04d}.npy")
                np.save(sample_file, seq)

            all_sequences.append(seq)
            all_labels.append(class_idx)

    X = np.array(all_sequences, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int32)

    # Shuffle dataset
    perm = np.random.permutation(len(X))
    X, y = X[perm], y[perm]

    np.save(os.path.join(PROCESSED_DIR, "X.npy"), X)
    np.save(os.path.join(PROCESSED_DIR, "y.npy"), y)

    print(f"\n[Dataset Generator] Success! Generated {len(X)} sequences of shape {X.shape}.")
    print(f"[Dataset Generator] Processed dataset saved to: {PROCESSED_DIR}")
    return X, y


def load_dataset() -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Loads compiled dataset or generates it if missing."""
    x_path = os.path.join(PROCESSED_DIR, "X.npy")
    y_path = os.path.join(PROCESSED_DIR, "y.npy")
    labels = load_labels()

    if os.path.exists(x_path) and os.path.exists(y_path):
        X = np.load(x_path)
        y = np.load(y_path)
        # Check if dataset matches labels count
        if len(np.unique(y)) == len(labels):
            print(f"[Dataset] Loaded existing processed dataset: X={X.shape}, y={y.shape} ({len(labels)} classes)")
            return X, y, labels

    print(f"[Dataset] Generating new 20-class dataset...")
    X, y = generate_isl_dataset(samples_per_gesture=160, save_raw=True)
    return X, y, labels


# ──────────────────────────────────────────────────────────
#  Live Webcam Collection Feature
# ──────────────────────────────────────────────────────────

def collect_live_webcam(
    gesture_name: str,
    num_sequences: int = 30,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    camera_id: int = 0,
):
    """Records live landmark sequences via webcam using OpenCV and MediaPipe HandDetector."""
    ensure_directories()
    gesture_dir = os.path.join(RAW_DIR, gesture_name)
    os.makedirs(gesture_dir, exist_ok=True)

    print(f"\n[Collector] Initializing webcam for gesture: {gesture_name}")
    print(f"[Collector] Planned sequences: {num_sequences}, Frames per sequence: {sequence_length}")

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"[Collector] Error: Could not open camera {camera_id}")
        return

    detector = HandDetector()
    existing = [f for f in os.listdir(gesture_dir) if f.endswith(".npy")]
    seq_idx = len(existing)

    print("\n---------------------------------------------------------")
    print(f" Ready to record: {gesture_name}")
    print(" Controls: Press SPACE to record, Q to quit.")
    print("---------------------------------------------------------\n")

    while seq_idx < (len(existing) + num_sequences):
        waiting = True
        while waiting:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            left, right, annotated, count = detector.detect_landmarks(frame, draw=True)

            cv2.putText(annotated, f"Gesture: {gesture_name} [Sample {seq_idx + 1}]", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(annotated, f"Hands: {count}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if count > 0 else (0, 0, 255), 2)
            cv2.putText(annotated, "Press SPACE to start recording", (20, 440),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2)

            cv2.imshow("SignBridge AI – Collector", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                waiting = False
            elif key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                detector.close()
                return

        # 3-second countdown
        for countdown in [3, 2, 1]:
            t_end = time.time() + 0.5
            while time.time() < t_end:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                cv2.putText(frame, f"GET READY: {countdown}", (200, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 165, 255), 4)
                cv2.imshow("SignBridge AI – Collector", frame)
                cv2.waitKey(1)

        # Record sequence
        sequence_data = []
        for frame_num in range(sequence_length):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            left, right, annotated, count = detector.detect_landmarks(frame, draw=True)

            feats = process_frame_landmarks(left, right, normalize=True)
            sequence_data.append(feats)

            cv2.putText(annotated, f"RECORDING [{frame_num + 1}/{sequence_length}]",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow("SignBridge AI – Collector", annotated)
            cv2.waitKey(1)

        if len(sequence_data) == sequence_length:
            save_path = os.path.join(gesture_dir, f"seq_{seq_idx:04d}.npy")
            np.save(save_path, np.array(sequence_data, dtype=np.float32))
            print(f"[Collector] Saved sample {seq_idx + 1} to {save_path}")
            seq_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print(f"\n[Collector] Completed recording for {gesture_name}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SignBridge AI ISL Dataset Collector")
    parser.add_argument("--generate", action="store_true", help="Generate full 20-class ISL dataset")
    parser.add_argument("--samples", type=int, default=160, help="Number of samples per class")
    parser.add_argument("--live", action="store_true", help="Record live gestures via webcam")
    parser.add_argument("--gesture", type=str, default="HELLO", help="Gesture name for live collection")

    args = parser.parse_args()

    if args.live:
        collect_live_webcam(gesture_name=args.gesture.upper(), num_sequences=args.samples)
    else:
        generate_isl_dataset(samples_per_gesture=args.samples, save_raw=True)
