"""
Predictor Module for SignBridge AI.
Combines deep temporal PyTorch LSTM sequence inference with
robust anatomical hand landmark geometry verification to ensure high accuracy
and eliminate false predictions (such as spurious 'BAD' or 'THANK YOU').
"""

import os
import sys
import json
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

# Path configurations
MODEL_DIR           = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model"))
DEFAULT_MODEL_PATH  = os.path.join(MODEL_DIR, "trained_model.pt")
DEFAULT_LABELS_PATH = os.path.join(MODEL_DIR, "labels.json")

if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)


def analyze_hand_geometry(pts: Optional[List[Dict[str, float]]]) -> Optional[Dict[str, Any]]:
    """
    Extracts anatomical posture metrics from a 21-landmark hand list.
    Screen coordinate conventions: x in [0, 1] (left to right), y in [0, 1] (top to bottom).
    Therefore:
      - A landmark is higher in space (closer to head/ceiling) if landmark.y < other.y
      - A landmark is lower in space (closer to ground/desk) if landmark.y > other.y
    """
    if not pts or len(pts) < 21:
        return None

    try:
        coords = np.array(
            [[float(p.get("x", 0.0)), float(p.get("y", 0.0)), float(p.get("z", 0.0))] for p in pts[:21]],
            dtype=np.float32
        )
    except Exception:
        return None

    wrist = coords[0]
    palm_size = float(np.linalg.norm(coords[9] - wrist))
    if palm_size < 1e-3:
        return None

    # Check if a finger is extended (tip is clearly further from wrist and knuckle than middle joint)
    def check_extended(tip_idx: int, pip_idx: int, mcp_idx: int) -> bool:
        d_tip_w = np.linalg.norm(coords[tip_idx] - wrist)
        d_pip_w = np.linalg.norm(coords[pip_idx] - wrist)
        d_tip_m = np.linalg.norm(coords[tip_idx] - coords[mcp_idx])
        d_pip_m = np.linalg.norm(coords[pip_idx] - coords[mcp_idx])
        return bool(d_tip_w > d_pip_w * 1.05 and d_tip_m > d_pip_m * 1.05)

    def check_curled(tip_idx: int, pip_idx: int, mcp_idx: int) -> bool:
        d_tip_w = np.linalg.norm(coords[tip_idx] - wrist)
        d_pip_w = np.linalg.norm(coords[pip_idx] - wrist)
        d_tip_m = np.linalg.norm(coords[tip_idx] - coords[mcp_idx])
        d_pip_m = np.linalg.norm(coords[pip_idx] - coords[mcp_idx])
        return bool(d_tip_w < d_pip_w * 1.05 or d_tip_m < d_pip_m * 1.05)

    index_up  = check_extended(8, 6, 5)
    middle_up = check_extended(12, 10, 9)
    ring_up   = check_extended(16, 14, 13)
    pinky_up  = check_extended(20, 18, 17)

    index_curled  = check_curled(8, 6, 5)
    middle_curled = check_curled(12, 10, 9)
    ring_curled   = check_curled(16, 14, 13)
    pinky_curled  = check_curled(20, 18, 17)

    # Thumb analysis
    thumb_tip = coords[4]
    thumb_mcp = coords[2]

    # Thumb pointing UP toward top of screen (tip y is significantly smaller than mcp y)
    thumb_up = bool(thumb_tip[1] < (thumb_mcp[1] - 0.03) and thumb_tip[1] < (wrist[1] - 0.04))

    # Thumb pointing DOWN toward bottom of screen (tip y is significantly larger than mcp y)
    thumb_down = bool(thumb_tip[1] > (thumb_mcp[1] + 0.04) and thumb_tip[1] > (wrist[1] + 0.02))

    # Thumb extended outward horizontally (lateral extension)
    thumb_out = bool(abs(thumb_tip[0] - thumb_mcp[0]) > (palm_size * 0.45))

    # Pinch: distance between thumb tip (4) and index tip (8)
    pinch_dist = float(np.linalg.norm(thumb_tip - coords[8]))
    is_pinch   = bool(pinch_dist < (palm_size * 0.40))

    # V-spread: separation between index tip (8) and middle tip (12)
    v_dist   = float(np.linalg.norm(coords[8] - coords[12]))
    v_spread = bool(v_dist > (palm_size * 0.28))

    return {
        "index_up":      index_up,
        "middle_up":     middle_up,
        "ring_up":       ring_up,
        "pinky_up":      pinky_up,
        "index_curled":  index_curled,
        "middle_curled": middle_curled,
        "ring_curled":   ring_curled,
        "pinky_curled":  pinky_curled,
        "thumb_up":      thumb_up,
        "thumb_down":    thumb_down,
        "thumb_out":     thumb_out,
        "is_pinch":      is_pinch,
        "v_spread":      v_spread,
        "wrist":         wrist,
        "palm_size":     palm_size
    }


def extract_anatomical_signature(
    raw_left: Optional[List[Dict[str, float]]],
    raw_right: Optional[List[Dict[str, float]]]
) -> Tuple[Optional[str], float]:
    """
    Evaluates definitive anatomical hand gesture rules.
    Returns (gesture_label, confidence_score) if a clear physical signature is present.
    """
    left_g  = analyze_hand_geometry(raw_left)
    right_g = analyze_hand_geometry(raw_right)

    hands_count = (1 if left_g else 0) + (1 if right_g else 0)
    if hands_count == 0:
        return "NEUTRAL", 0.95

    # ── TWO-HANDED GESTURES ──
    if left_g and right_g:
        wrist_dist = float(np.linalg.norm(left_g["wrist"] - right_g["wrist"]))

        # NAMASTE: both hands upright touching at chest level
        if wrist_dist < 0.28 and (left_g["index_up"] or left_g["middle_up"]) and (right_g["index_up"] or right_g["middle_up"]):
            return "NAMASTE", 0.98

        # HELP: one fist with thumb up resting on opposite flat horizontal palm
        if (left_g["thumb_up"] and not left_g["index_up"]) or (right_g["thumb_up"] and not right_g["index_up"]):
            return "HELP", 0.95

        # HELLO: both hands open, upright and raised
        if left_g["index_up"] and right_g["index_up"]:
            return "HELLO", 0.95

        # PLEASE: hands close together in front of body
        if wrist_dist < 0.28:
            return "PLEASE", 0.90

        # WELCOME: both open palms facing upward
        return "WELCOME", 0.88

    # ── SINGLE-HANDED GESTURES ──
    g = right_g or left_g
    if g:
        num_ext = sum([g["index_up"], g["middle_up"], g["ring_up"], g["pinky_up"]])

        # 1. GOOD: Decisive Thumbs Up (Thumb points UP, 4 fingers curled)
        if g["thumb_up"] and (g["middle_curled"] and g["ring_curled"] and g["pinky_curled"]) and not g["index_up"]:
            return "GOOD", 0.98

        # 2. BAD: Decisive Thumbs Down (Thumb points DOWN, 4 fingers curled)
        # CRITICAL: BAD is ONLY returned if thumb is explicitly pointing DOWN!
        if g["thumb_down"] and (g["middle_curled"] and g["ring_curled"] and g["pinky_curled"]) and not g["index_up"]:
            return "BAD", 0.98

        # 3. PEACE: Victory V-sign (Index & Middle UP in V, Ring & Pinky curled)
        if g["index_up"] and g["middle_up"] and g["ring_curled"] and g["pinky_curled"] and g["v_spread"]:
            return "PEACE", 0.98

        # 4. YOU: Pointing directly forward (Index UP, other fingers curled)
        if g["index_up"] and g["middle_curled"] and g["ring_curled"] and g["pinky_curled"]:
            return "YOU", 0.98

        # 5. LOVE: Universal I Love You (Thumb OUT or UP, Index UP, Pinky UP, Middle & Ring curled)
        if (g["thumb_out"] or g["thumb_up"]) and g["index_up"] and g["middle_curled"] and g["ring_curled"] and g["pinky_up"]:
            return "LOVE", 0.98

        # 6. OK: Thumb & Index pinch circle, other fingers upright
        if g["is_pinch"] and (g["middle_up"] or g["ring_up"] or g["pinky_up"]):
            return "OK", 0.96

        # 7. WATER: Three fingers upright in 'W' shape (Index, Middle, Ring UP, Pinky curled)
        if g["index_up"] and g["middle_up"] and g["ring_up"] and g["pinky_curled"]:
            return "WATER", 0.96

        # 8. NO: Index and Middle extended upright together (no V-spread)
        if g["index_up"] and g["middle_up"] and g["ring_curled"] and g["pinky_curled"] and not g["v_spread"]:
            return "NO", 0.92

        # 9. OPEN PALM (STOP vs HELLO vs THANK YOU):
        if num_ext >= 3 and g["index_up"] and g["middle_up"] and (g["ring_up"] or g["pinky_up"]) and not g["is_pinch"]:
            # If held high near head / ear level (wrist y < 0.45) -> HELLO (single-hand wave)
            if g["wrist"][1] < 0.45:
                return "HELLO", 0.95
            # If held near chin / lips (wrist y between 0.45 and 0.60) -> THANK YOU
            if 0.45 <= g["wrist"][1] <= 0.62:
                return "THANK YOU", 0.92
            # Otherwise, held open in front facing camera -> STOP
            return "STOP", 0.95

        # 10. FIST (YES vs SORRY):
        if num_ext == 0 and not g["thumb_up"] and not g["thumb_down"]:
            # If fist is placed over chest area (wrist y < 0.60) -> SORRY
            if g["wrist"][1] < 0.60:
                return "SORRY", 0.90
            # Otherwise -> YES (fist nodding)
            return "YES", 0.90

    return None, 0.0


class SignPredictor:
    def __init__(self, model_path: Optional[str] = None, labels_path: Optional[str] = None):
        self.model_path  = model_path  or DEFAULT_MODEL_PATH
        self.labels_path = labels_path or DEFAULT_LABELS_PATH

        self.labels:     List[str] = []
        self.model:      Any       = None
        self.device:     Any       = None
        self.is_loaded:  bool      = False
        self.load_error: Optional[str] = None

        self._load_labels()
        self._load_model()

    def _load_labels(self):
        """Loads gesture label mappings from labels.json."""
        fallback = [
            "HELLO", "NAMASTE", "THANK YOU", "PLEASE", "YES", "NO", "HELP", "HOW",
            "ARE", "YOU", "GOOD", "BAD", "LOVE", "PEACE", "STOP", "OK", "SORRY",
            "WATER", "WELCOME", "NEUTRAL"
        ]
        if os.path.exists(self.labels_path):
            try:
                with open(self.labels_path, "r", encoding="utf-8") as f:
                    self.labels = json.load(f)
                print(f"[Predictor] Loaded {len(self.labels)} labels from {self.labels_path}")
            except Exception as e:
                print(f"[Predictor] Warning: Could not read labels file: {e}")
                self.labels = fallback
        else:
            self.labels = fallback

    def _load_model(self):
        """Loads the trained PyTorch LSTM model from trained_model.pt."""
        if not os.path.exists(self.model_path):
            self.load_error = (
                f"Model file not found at {self.model_path}. "
                "Run model/train_model.py first."
            )
            print(f"[Predictor] {self.load_error}")
            return

        try:
            import torch
            from train_model import ISLLSTMModel

            checkpoint  = torch.load(self.model_path, map_location="cpu", weights_only=False)
            input_size  = checkpoint.get("input_size",  126)
            num_classes = checkpoint.get("num_classes", len(self.labels))

            saved_labels = checkpoint.get("labels")
            if saved_labels:
                self.labels = saved_labels

            m = ISLLSTMModel(input_size=input_size, num_classes=num_classes)
            m.load_state_dict(checkpoint["model_state_dict"])
            m.eval()

            self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model     = m.to(self.device)
            self.is_loaded = True
            self.load_error = None
            print(f"[Predictor] Model loaded from {self.model_path} on {self.device} ({num_classes} classes)")
        except Exception as e:
            self.is_loaded  = False
            self.load_error = f"Failed to load model: {e}"
            print(f"[Predictor] Error: {self.load_error}")

    def predict(
        self,
        sequence_array: np.ndarray,
        raw_left: Optional[List[Dict[str, float]]] = None,
        raw_right: Optional[List[Dict[str, float]]] = None
    ) -> Tuple[str, float, Dict[str, float]]:
        """
        Hybrid inference combining:
        1. PyTorch LSTM temporal probabilities across rolling 30-frame sequence
        2. Anatomical landmark geometry verification to prevent false triggers (e.g. spurious 'BAD' or 'THANK YOU')

        Returns:
            (predicted_label, confidence_score, probabilities_dict)
        """
        if not self.is_loaded or self.model is None:
            if os.path.exists(self.model_path):
                self._load_model()
            if not self.is_loaded:
                return "MODEL_NOT_READY", 0.0, {}

        if len(sequence_array.shape) != 3:
            raise ValueError(
                f"Expected 3-D input (batch, time, features), got shape {sequence_array.shape}"
            )

        try:
            import torch
            import torch.nn.functional as F

            # 1. Run LSTM forward pass
            with torch.no_grad():
                x      = torch.tensor(sequence_array, dtype=torch.float32).to(self.device)
                logits = self.model(x)
                raw_probs = F.softmax(logits, dim=1)[0].cpu().numpy()

            probs_dict = {
                (self.labels[i] if i < len(self.labels) else f"CLASS_{i}"): float(raw_probs[i])
                for i in range(len(raw_probs))
            }

            # 2. Extract anatomical physical signature
            geom_label, geom_conf = extract_anatomical_signature(raw_left, raw_right)

            # 3. Fuse anatomical evidence
            if geom_label and geom_label in probs_dict:
                # Strong physical signature heavily boosts the detected class
                for k in probs_dict:
                    probs_dict[k] *= 0.10
                probs_dict[geom_label] = max(probs_dict.get(geom_label, 0.0), geom_conf)

            # 4. Strict domain sanity filters:
            left_g  = analyze_hand_geometry(raw_left)
            right_g = analyze_hand_geometry(raw_right)
            active_g = right_g or left_g

            if active_g:
                # - Filter 'BAD': BAD is ONLY possible if thumb is explicitly pointing DOWN
                if not active_g["thumb_down"] and "BAD" in probs_dict:
                    probs_dict["BAD"] = 0.0

                # - Filter 'THANK YOU': ONLY possible if hand is near chin/lips (wrist y < 0.65)
                #   If hand is in mid-frame or lower, heavily suppress THANK YOU so it never steals predictions
                if active_g["wrist"][1] > 0.65 and "THANK YOU" in probs_dict:
                    probs_dict["THANK YOU"] *= 0.05

                # - Filter 'GOOD': ONLY possible if thumb is pointing UP and fingers are curled
                if (not active_g["thumb_up"] or active_g["index_up"] or active_g["pinky_up"]) and "GOOD" in probs_dict:
                    probs_dict["GOOD"] *= 0.05

            # - If only 1 hand is present, suppress 2-handed classes:
            two_handed_signs = {"NAMASTE", "HELP", "HOW", "PLEASE", "WELCOME"}
            if not (raw_left and raw_right and len(raw_left) >= 21 and len(raw_right) >= 21):
                for ths in two_handed_signs:
                    if ths in probs_dict:
                        probs_dict[ths] *= 0.05

            # Re-normalize probabilities
            total_prob = sum(probs_dict.values())
            if total_prob > 1e-6:
                for k in probs_dict:
                    probs_dict[k] /= total_prob

            # Extract final predicted sign and confidence
            sorted_items = sorted(probs_dict.items(), key=lambda kv: kv[1], reverse=True)
            top_label, top_conf = sorted_items[0]

            return top_label, float(top_conf), probs_dict

        except Exception as e:
            print(f"[Predictor] Inference error: {e}")
            return "ERROR", 0.0, {}

    def get_num_classes(self) -> int:
        return len(self.labels)
