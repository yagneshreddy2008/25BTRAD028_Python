"""
Model Training Module for SignBridge AI.
Builds, trains, and saves the LSTM neural network for real-time ISL sequential gesture recognition.
Uses PyTorch as the deep learning backend.
"""

import os
import sys
import json
import argparse
import numpy as np

# Ensure root paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BACKEND_DIR = os.path.join(PROJECT_ROOT, "backend")
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)
if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.model_selection import train_test_split

from dataset_collector import load_dataset, load_labels, DEFAULT_SEQUENCE_LENGTH, TOTAL_FRAME_FEATURES


# ──────────────────────────────────────────────────────────
#  Model Architecture
# ──────────────────────────────────────────────────────────

class ISLLSTMModel(nn.Module):
    """
    3-layer LSTM classifier for real-time ISL gesture recognition.
    Input:  (batch, sequence_length, feature_dim)  — 30 frames × 126 features
    Output: (batch, num_classes)                   — softmax probabilities
    """

    def __init__(self, input_size: int = 126, num_classes: int = 11):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, 64,  batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(64,  128, batch_first=True)
        self.dropout2 = nn.Dropout(0.2)
        self.lstm3 = nn.LSTM(128,  64, batch_first=True)
        self.dropout3 = nn.Dropout(0.2)
        self.fc1    = nn.Linear(64,  64)
        self.relu1  = nn.ReLU()
        self.dropout4 = nn.Dropout(0.15)
        self.fc2    = nn.Linear(64,  32)
        self.relu2  = nn.ReLU()
        self.fc3    = nn.Linear(32, num_classes)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        out = self.dropout2(out)
        out, _ = self.lstm3(out)
        # Take only last time-step output
        out = out[:, -1, :]
        out = self.dropout3(out)
        out = self.relu1(self.fc1(out))
        out = self.dropout4(out)
        out = self.relu2(self.fc2(out))
        out = self.fc3(out)          # raw logits — CrossEntropyLoss handles softmax
        return out


# ──────────────────────────────────────────────────────────
#  Training Routine
# ──────────────────────────────────────────────────────────

def train(
    epochs: int = 40,
    batch_size: int = 32,
    model_save_path: str = None,
    patience: int = 12
):
    """
    Full PyTorch model training workflow:
      1. Loads dataset
      2. Performs 70/15/15 Train/Validation/Test split
      3. Trains LSTM with EarlyStopping and LR scheduling
      4. Evaluates on the held-out test set
      5. Saves trained model weights as trained_model.pt
    """
    if model_save_path is None:
        model_save_path = os.path.join(MODEL_DIR, "trained_model.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 65)
    print(" SignBridge AI – ISL Gesture Recognition Model Training")
    print("=" * 65)
    print(f"[Training] Device: {device}")

    # 1. Load data
    X, y, labels = load_dataset()
    num_classes   = len(labels)
    sequence_length = X.shape[1]
    feature_dim     = X.shape[2]

    print(f"[Training] Dataset shape: X={X.shape}, y={y.shape}")
    print(f"[Training] Target vocabulary ({num_classes} classes): {labels}")

    # 2. Train / Val / Test split  (70 / 15 / 15)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )
    print(f"[Training] Splits — Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # Convert to PyTorch tensors
    def to_tensors(Xa, ya):
        return (
            torch.tensor(Xa, dtype=torch.float32),
            torch.tensor(ya, dtype=torch.long),
        )

    Xt, yt   = to_tensors(X_train, y_train)
    Xv, yv   = to_tensors(X_val,   y_val)
    Xte, yte = to_tensors(X_test,  y_test)

    train_loader = DataLoader(TensorDataset(Xt, yt),   batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(Xv, yv),   batch_size=batch_size)
    test_loader  = DataLoader(TensorDataset(Xte, yte), batch_size=batch_size)

    # 3. Build model
    model     = ISLLSTMModel(input_size=feature_dim, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4, min_lr=1e-5
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Training] Model parameters: {total_params:,}")

    # 4. Training loop with EarlyStopping
    best_val_loss = float("inf")
    best_weights  = None
    stagnant      = 0

    print(f"\n[Training] Starting training for up to {epochs} epochs...")
    print("-" * 65)

    for epoch in range(1, epochs + 1):
        # ── Train ──
        model.train()
        train_loss = 0.0
        train_correct = 0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(Xb)
            loss   = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss   += loss.item() * Xb.size(0)
            train_correct += (logits.argmax(1) == yb).sum().item()

        train_loss /= len(Xt)
        train_acc   = train_correct / len(Xt)

        # ── Validate ──
        model.eval()
        val_loss = 0.0
        val_correct = 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                logits  = model(Xb)
                val_loss     += criterion(logits, yb).item() * Xb.size(0)
                val_correct  += (logits.argmax(1) == yb).sum().item()

        val_loss /= len(Xv)
        val_acc   = val_correct / len(Xv)

        scheduler.step(val_loss)

        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc * 100:6.2f}% | "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc * 100:6.2f}%"
        )

        # EarlyStopping: save best weights
        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_weights  = {k: v.clone() for k, v in model.state_dict().items()}
            stagnant      = 0
        else:
            stagnant += 1
            if stagnant >= patience:
                print(f"\n[Training] Early stopping triggered at epoch {epoch}.")
                break

    # Restore best weights
    if best_weights:
        model.load_state_dict(best_weights)
        print("[Training] Restored best model weights.")

    # 5. Evaluate on test set
    model.eval()
    test_correct = 0
    with torch.no_grad():
        for Xb, yb in test_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            test_correct += (model(Xb).argmax(1) == yb).sum().item()

    test_acc = test_correct / len(Xte)
    print("=" * 65)
    print(f" Test Set Accuracy: {test_acc * 100:.2f}%  |  Samples: {len(Xte)}")
    print("=" * 65)

    # 6. Save model
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    # Save full state dict + metadata for easy reload
    save_dict = {
        "model_state_dict": model.state_dict(),
        "input_size":       feature_dim,
        "num_classes":      num_classes,
        "sequence_length":  sequence_length,
        "labels":           labels,
    }
    torch.save(save_dict, model_save_path)
    print(f"[Training] Saved model to: {model_save_path}")

    # Save test split for evaluate_model.py
    processed_dir = os.path.join(PROJECT_ROOT, "dataset", "processed")
    np.save(os.path.join(processed_dir, "X_test.npy"), X_test)
    np.save(os.path.join(processed_dir, "y_test.npy"), y_test)
    print(f"[Training] Saved test split to: {processed_dir}")

    return model, labels, test_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SignBridge AI Model Trainer (PyTorch)")
    parser.add_argument("--epochs",     type=int, default=40, help="Max training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--patience",   type=int, default=12, help="EarlyStopping patience")
    args = parser.parse_args()

    train(epochs=args.epochs, batch_size=args.batch_size, patience=args.patience)
