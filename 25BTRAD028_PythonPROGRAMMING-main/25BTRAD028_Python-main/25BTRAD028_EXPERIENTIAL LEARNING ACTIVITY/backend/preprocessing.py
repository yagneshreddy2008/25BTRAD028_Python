"""
Preprocessing and Normalization Module for SignBridge AI.
Transforms MediaPipe hand landmarks into invariant, normalized sequence arrays for the LSTM model.
"""

from typing import List, Optional, Tuple, Dict, Any
import numpy as np


# Hand landmark constants
NUM_LANDMARKS_PER_HAND = 21
NUM_COORDS_PER_LANDMARK = 3  # (x, y, z)
HAND_FEATURE_DIM = NUM_LANDMARKS_PER_HAND * NUM_COORDS_PER_LANDMARK  # 63
TOTAL_FRAME_FEATURES = HAND_FEATURE_DIM * 2  # 126 (Left + Right)
DEFAULT_SEQUENCE_LENGTH = 30  # Number of frames in a gesture sequence


def extract_hand_features(landmarks: Optional[List[Dict[str, float]]]) -> np.ndarray:
    """
    Extracts raw coordinates from a single hand's landmarks.
    Expected input: list of 21 objects with 'x', 'y', 'z'.
    Returns: (63,) float32 array. Returns zeros if hand is missing.
    """
    if not landmarks or len(landmarks) < NUM_LANDMARKS_PER_HAND:
        return np.zeros(HAND_FEATURE_DIM, dtype=np.float32)

    coords = []
    for lm in landmarks[:NUM_LANDMARKS_PER_HAND]:
        coords.extend([float(lm.get("x", 0.0)), float(lm.get("y", 0.0)), float(lm.get("z", 0.0))])
    return np.array(coords, dtype=np.float32)


def normalize_hand_landmarks(coords_63: np.ndarray) -> np.ndarray:
    """
    Normalizes a 63-element hand vector for translation and scale invariance.
    - Translation: Shifts wrist landmark (landmark 0) to origin (0, 0, 0).
    - Scale: Normalizes by the Euclidean distance between wrist (0) and middle finger MCP (9).
    """
    if coords_63 is None or np.all(coords_63 == 0.0):
        return np.zeros(HAND_FEATURE_DIM, dtype=np.float32)

    pts = coords_63.reshape(NUM_LANDMARKS_PER_HAND, NUM_COORDS_PER_LANDMARK).copy()

    # Wrist is landmark 0
    wrist = pts[0].copy()
    pts_centered = pts - wrist

    # MediaPipe camera screen coordinates have y=0 at top and y=1 at bottom.
    # Therefore, fingers pointing UP have negative y relative to the wrist in screen coordinates.
    # Invert Y so that upright fingers have positive Y (+Y), matching standard 3D hand space:
    pts_centered[:, 1] = -pts_centered[:, 1]

    # Scale reference: distance between wrist (0) and middle finger MCP (9)
    palm_size = float(np.linalg.norm(pts_centered[9]))
    if palm_size < 1e-3:
        # Fallback to max span if palm_size is degenerate
        palm_size = float(np.max(np.linalg.norm(pts_centered, axis=1)))

    if palm_size < 1e-3:
        # Hand data is degenerate — treat as absent
        return np.zeros(HAND_FEATURE_DIM, dtype=np.float32)

    pts_normalized = pts_centered / palm_size

    result = pts_normalized.flatten().astype(np.float32)
    np.clip(result, -10.0, 10.0, out=result)
    return result


def process_frame_landmarks(
    left_hand: Optional[List[Dict[str, float]]],
    right_hand: Optional[List[Dict[str, float]]],
    normalize: bool = True
) -> np.ndarray:
    """
    Combines Left and Right hands into a 126-dimensional feature vector.
    Indices 0..62: Left hand (normalized)
    Indices 63..125: Right hand (normalized)
    """
    left_raw = extract_hand_features(left_hand)
    right_raw = extract_hand_features(right_hand)

    if normalize:
        left_feat = normalize_hand_landmarks(left_raw)
        right_feat = normalize_hand_landmarks(right_raw)
    else:
        left_feat = left_raw
        right_feat = right_raw

    return np.concatenate([left_feat, right_feat]).astype(np.float32)


class LandmarkSequenceBuffer:
    """
    Maintains a rolling temporal sequence of landmark frames for real-time LSTM inference.
    """
    def __init__(self, sequence_length: int = DEFAULT_SEQUENCE_LENGTH, feature_dim: int = TOTAL_FRAME_FEATURES):
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.buffer: List[np.ndarray] = []
        self._hand_present_history: List[bool] = []

    def append_frame(self, frame_features: np.ndarray, hands_detected: bool):
        """
        Appends a 126-D frame feature vector. Maintains fixed max window length.
        """
        if frame_features.shape[0] != self.feature_dim:
            raise ValueError(f"Expected feature dim {self.feature_dim}, got {frame_features.shape[0]}")

        self.buffer.append(frame_features.astype(np.float32))
        self._hand_present_history.append(hands_detected)

        if len(self.buffer) > self.sequence_length:
            self.buffer.pop(0)
            self._hand_present_history.pop(0)

    def is_full(self) -> bool:
        """Returns True if buffer has accumulated the required sequence length."""
        return len(self.buffer) >= self.sequence_length

    def has_active_hands(self, min_ratio: float = 0.3) -> bool:
        """
        Checks if hands were detected in at least a minimum fraction of recent frames.
        Avoids running inference on empty/noise frames.
        """
        if not self._hand_present_history:
            return False
        active_count = sum(1 for h in self._hand_present_history if h)
        return (active_count / len(self._hand_present_history)) >= min_ratio

    def get_sequence_array(self) -> np.ndarray:
        """
        Returns (1, sequence_length, feature_dim) float32 batch array for model prediction.
        Pads with initial frame or zeros if buffer is not yet full.
        """
        if not self.buffer:
            return np.zeros((1, self.sequence_length, self.feature_dim), dtype=np.float32)

        curr_len = len(self.buffer)
        if curr_len < self.sequence_length:
            # Pad front with repeat of first available frame
            pad_count = self.sequence_length - curr_len
            first_frame = self.buffer[0]
            padded = [first_frame] * pad_count + list(self.buffer)
            seq = np.array(padded, dtype=np.float32)
        else:
            seq = np.array(self.buffer[-self.sequence_length:], dtype=np.float32)

        return np.expand_dims(seq, axis=0)

    def clear(self):
        """Resets the sequence buffer."""
        self.buffer.clear()
        self._hand_present_history.clear()


def augment_sequence(
    sequence: np.ndarray,
    jitter_std: float = 0.015,
    scale_range: Tuple[float, float] = (0.92, 1.08),
    time_warp: bool = True
) -> np.ndarray:
    """
    Data augmentation for landmark sequences during model training:
    - Additive Gaussian noise (simulates hand tremor, tracking noise)
    - Random scaling (simulates user distance variations)
    - Optional temporal interpolation / speed variation
    """
    augmented = sequence.copy()

    # Scale augmentation
    scale = np.random.uniform(scale_range[0], scale_range[1])
    augmented = augmented * scale

    # Gaussian jitter (applied only to non-zero landmark values)
    mask = (augmented != 0.0)
    noise = np.random.normal(0, jitter_std, augmented.shape).astype(np.float32)
    augmented[mask] += noise[mask]

    # Ensure no extreme values
    np.clip(augmented, -10.0, 10.0, out=augmented)

    return augmented
