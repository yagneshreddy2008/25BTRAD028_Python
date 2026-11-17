"""
Model Evaluation Module for SignBridge AI.
Calculates test accuracy, classification metrics (precision, recall, F1),
and generates a confusion matrix plot. Uses PyTorch backend.
"""

import os
import sys
import json
import argparse
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR    = os.path.join(PROJECT_ROOT, "model")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "dataset", "processed")

DEFAULT_MODEL_PATH  = os.path.join(MODEL_DIR, "trained_model.pt")
DEFAULT_LABELS_PATH = os.path.join(MODEL_DIR, "labels.json")
CONFUSION_MATRIX_PATH = os.path.join(MODEL_DIR, "confusion_matrix.png")

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def evaluate_saved_model(
    model_path: str = DEFAULT_MODEL_PATH,
    labels_path: str = DEFAULT_LABELS_PATH,
    output_image_path: str = CONFUSION_MATRIX_PATH
):
    """
    Evaluates the trained PyTorch LSTM model on the test dataset and
    outputs metrics and a confusion matrix plot.
    """
    import torch

    if not os.path.exists(model_path):
        print(f"[Evaluation] Error: Model not found at {model_path}. Run train_model.py first.")
        return

    # ── Load model ──────────────────────────────────────────────
    # Import ISLLSTMModel from train_model.py
    if MODEL_DIR not in sys.path:
        sys.path.insert(0, MODEL_DIR)
    from train_model import ISLLSTMModel

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    input_size  = checkpoint.get("input_size",  126)
    num_classes = checkpoint.get("num_classes", 11)
    labels      = checkpoint.get("labels", None)

    # Fallback: load labels from JSON
    if labels is None:
        with open(labels_path, "r", encoding="utf-8") as f:
            labels = json.load(f)

    model = ISLLSTMModel(input_size=input_size, num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"[Evaluation] Loaded model from {model_path}")
    print(f"[Evaluation] Labels ({num_classes} classes): {labels}")

    # ── Load test data ───────────────────────────────────────────
    x_test_path = os.path.join(PROCESSED_DIR, "X_test.npy")
    y_test_path = os.path.join(PROCESSED_DIR, "y_test.npy")

    if os.path.exists(x_test_path) and os.path.exists(y_test_path):
        X_test       = np.load(x_test_path)
        y_test_labels = np.load(y_test_path)
        # If one-hot encoded, collapse
        if y_test_labels.ndim > 1:
            y_test_labels = np.argmax(y_test_labels, axis=1)
        print(f"[Evaluation] Test samples: {len(X_test)}")
    else:
        print("[Evaluation] Test split not found — loading full dataset as test set.")
        X_full = np.load(os.path.join(PROCESSED_DIR, "X.npy"))
        y_full = np.load(os.path.join(PROCESSED_DIR, "y.npy"))
        X_test, y_test_labels = X_full, y_full

    # ── Inference ────────────────────────────────────────────────
    with torch.no_grad():
        X_tensor = torch.tensor(X_test, dtype=torch.float32)
        logits   = model(X_tensor)
        y_pred   = logits.argmax(dim=1).numpy()

    acc    = accuracy_score(y_test_labels, y_pred)
    report = classification_report(
        y_test_labels,
        y_pred,
        target_names=labels,
        digits=4,
        zero_division=0
    )
    cm = confusion_matrix(y_test_labels, y_pred)

    print("\n" + "=" * 65)
    print(" SignBridge AI – Model Evaluation Report")
    print("=" * 65)
    print(f"Overall Test Accuracy: {acc * 100:.2f}%\n")
    print("Classification Report:")
    print(report)

    # ── Confusion Matrix Plot ────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")          # Non-interactive backend (no display needed)
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(11, 9))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels
        )
        plt.title(
            f"ISL Gesture Recognition – Confusion Matrix  (Acc: {acc * 100:.2f}%)",
            fontsize=14,
            pad=15
        )
        plt.xlabel("Predicted Sign", fontsize=12, labelpad=10)
        plt.ylabel("True Sign",      fontsize=12, labelpad=10)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()

        os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
        plt.savefig(output_image_path, dpi=200)
        plt.close()
        print(f"[Evaluation] Confusion matrix saved to: {output_image_path}")
    except Exception as e:
        print(f"[Evaluation] Warning: Could not render confusion matrix plot: {e}")

    print("=" * 65 + "\n")
    return acc, cm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SignBridge AI Model Evaluator")
    parser.add_argument("--model",  type=str, default=DEFAULT_MODEL_PATH,
                        help="Path to trained_model.pt")
    parser.add_argument("--output", type=str, default=CONFUSION_MATRIX_PATH,
                        help="Output path for confusion matrix PNG")
    args = parser.parse_args()

    evaluate_saved_model(model_path=args.model, output_image_path=args.output)
