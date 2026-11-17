import sys
import os
import numpy as np

sys.path.insert(0, r"C:\Users\teega\.gemini\antigravity\scratch\signbridge-ai\backend")
from predictor import SignPredictor

predictor = SignPredictor()
print(f"Model loaded: {predictor.is_loaded}")
print(f"Labels: {predictor.labels}")

def make_hand(wrist=(0.5, 0.7, 0.0), extended_fingers=(False, False, False, False, False), thumb_dir="up"):
    # extended_fingers: (thumb, index, middle, ring, pinky)
    wx, wy, wz = wrist
    pts = [{"x": wx, "y": wy, "z": wz}] # 0: wrist
    
    # Thumb: 1, 2, 3, 4
    if thumb_dir == "up":
        pts.extend([
            {"x": wx - 0.04, "y": wy - 0.04, "z": 0.0},
            {"x": wx - 0.07, "y": wy - 0.08, "z": 0.0},
            {"x": wx - 0.09, "y": wy - 0.12, "z": 0.0},
            {"x": wx - 0.10, "y": wy - 0.17, "z": 0.0} # Tip well above wrist
        ])
    elif thumb_dir == "down":
        pts.extend([
            {"x": wx - 0.04, "y": wy + 0.04, "z": 0.0},
            {"x": wx - 0.07, "y": wy + 0.08, "z": 0.0},
            {"x": wx - 0.09, "y": wy + 0.12, "z": 0.0},
            {"x": wx - 0.10, "y": wy + 0.18, "z": 0.0} # Tip well below wrist
        ])
    else: # curled across palm
        pts.extend([
            {"x": wx - 0.03, "y": wy - 0.03, "z": 0.0},
            {"x": wx - 0.04, "y": wy - 0.05, "z": 0.0},
            {"x": wx - 0.02, "y": wy - 0.05, "z": 0.0},
            {"x": wx, "y": wy - 0.04, "z": 0.0}
        ])

    # 4 fingers: Index (5-8), Middle (9-12), Ring (13-16), Pinky (17-20)
    finger_xs = [-0.04, 0.0, 0.04, 0.07]
    for i, fx in enumerate(finger_xs):
        is_ext = extended_fingers[i+1]
        mcp_x = wx + fx
        mcp_y = wy - 0.09
        if is_ext:
            pts.extend([
                {"x": mcp_x, "y": mcp_y, "z": 0.0},
                {"x": mcp_x, "y": mcp_y - 0.05, "z": 0.0},
                {"x": mcp_x, "y": mcp_y - 0.09, "z": 0.0},
                {"x": mcp_x, "y": mcp_y - 0.14, "z": 0.0} # Tip extended upwards
            ])
        else: # curled into palm
            pts.extend([
                {"x": mcp_x, "y": mcp_y, "z": 0.0},
                {"x": mcp_x, "y": mcp_y - 0.03, "z": 0.0},
                {"x": mcp_x, "y": mcp_y - 0.01, "z": 0.0},
                {"x": mcp_x, "y": mcp_y + 0.01, "z": 0.0} # Tip curled down near palm
            ])
    return pts

def make_pinch_hand():
    pts = make_hand(wrist=(0.5, 0.7, 0.0), extended_fingers=(False, False, True, True, True))
    # Bring thumb tip (4) close to index tip (8)
    pts[4] = {"x": 0.46, "y": 0.60, "z": 0.0}
    pts[8] = {"x": 0.46, "y": 0.60, "z": 0.0}
    return pts

tests = [
    ("GOOD (Thumbs Up)", make_hand(thumb_dir="up", extended_fingers=(True, False, False, False, False)), None),
    ("BAD (Thumbs Down)", make_hand(thumb_dir="down", extended_fingers=(False, False, False, False, False)), None),
    ("STOP (Open Palm, Mid-Height)", make_hand(wrist=(0.5, 0.65, 0.0), extended_fingers=(True, True, True, True, True)), None),
    ("PEACE (V-Sign)", make_hand(extended_fingers=(False, True, True, False, False)), None),
    ("LOVE (ILY Sign)", make_hand(thumb_dir="up", extended_fingers=(True, True, False, False, True)), None),
    ("YOU (Pointing Index)", make_hand(extended_fingers=(False, True, False, False, False)), None),
    ("WATER (W 3 Fingers)", make_hand(extended_fingers=(False, True, True, True, False)), None),
    ("OK (Pinch circle)", make_pinch_hand(), None),
    ("THANK YOU (Hand near chin/lips)", make_hand(wrist=(0.5, 0.52, 0.0), extended_fingers=(True, True, True, True, True)), None),
    ("HELLO (Hand raised near head)", make_hand(wrist=(0.5, 0.38, 0.0), extended_fingers=(True, True, True, True, True)), None),
    ("NAMASTE (Two Palms Together)", make_hand(wrist=(0.48, 0.65, 0.0), extended_fingers=(True, True, True, True, True)), make_hand(wrist=(0.52, 0.65, 0.0), extended_fingers=(True, True, True, True, True))),
    ("HELP (Fist on Flat Palm)", make_hand(wrist=(0.48, 0.65, 0.0), thumb_dir="up", extended_fingers=(True, False, False, False, False)), make_hand(wrist=(0.52, 0.65, 0.0), extended_fingers=(True, True, True, True, True))),
    ("NEUTRAL (No Hands)", None, None),
]

print("\n--- RUNNING PREDICTOR ACCURACY VERIFICATION ---")
passed = 0
for name, left, right in tests:
    # Dummy sequence (15, 126) of zeros - testing hybrid anatomical & LSTM
    seq = np.zeros((1, 15, 126), dtype=np.float32)
    label, conf, details = predictor.predict(seq, raw_left=left, raw_right=right)
    expected = name.split()[0]
    is_match = (expected in label)
    status = "PASS" if is_match else "FAIL"
    if is_match:
        passed += 1
    print(f"[{status}] {name:30} -> Predicted: '{label}' ({conf*100:.1f}%)")

print(f"\nResult: {passed}/{len(tests)} tests passed successfully!")
