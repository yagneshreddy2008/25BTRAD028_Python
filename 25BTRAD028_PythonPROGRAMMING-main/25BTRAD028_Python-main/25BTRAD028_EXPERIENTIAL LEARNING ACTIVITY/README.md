# SignBridge AI — Real-Time ISL to Text Translator

> **AI-powered Indian Sign Language (ISL) recognition website** — webcam → MediaPipe landmarks → LSTM → real-time text + TTS.

---

## Quick Start

```bat
# 1. Start the server (double-click or run in terminal)
start.bat

# 2. Open browser
http://localhost:8000
```

> **Python environment:** `C:\signenv` (pre-configured with all packages)

---

## Project Structure

```
signbridge-ai/
├── backend/
│   ├── main.py                 FastAPI app — REST + WebSocket
│   ├── websocket_handler.py    Real-time landmark stream handler
│   ├── hand_detector.py        MediaPipe HandLandmarker (Tasks API)
│   ├── predictor.py            PyTorch LSTM inference wrapper
│   ├── preprocessing.py        Landmark normalization + sequence buffer
│   ├── sentence_builder.py     Debounce / hold / grammar logic
│   ├── hand_landmarker.task    MediaPipe model binary (7.8 MB)
│   └── requirements.txt
├── model/
│   ├── train_model.py          PyTorch LSTM trainer (ISLLSTMModel)
│   ├── evaluate_model.py       Confusion matrix + classification report
│   ├── dataset_collector.py    Live webcam + synthetic ISL generator
│   ├── trained_model.pt        ✅ Trained model weights
│   ├── labels.json             11 ISL class labels
│   └── confusion_matrix.png    ✅ Generated after evaluation
├── dataset/
│   ├── raw/                    Per-gesture .npy sequence files
│   └── processed/
│       ├── X.npy               (770, 30, 126)
│       ├── y.npy               (770,)
│       ├── X_test.npy          Held-out test split
│       └── y_test.npy
├── frontend/
│   ├── index.html              Full dashboard UI
│   ├── style.css               Dark glassmorphism theme
│   └── script.js               WebRTC + MediaPipe + WebSocket client
├── start.bat                   ▶  One-click server launcher
├── train.bat                   ▶  Retrain model from scratch
└── README.md
```

---

## Supported ISL Signs (20 Gestures)

| Sign | Type | Hands | Description |
|------|------|-------|-------------|
| HELLO | Greeting | 2 | Open upright hands waving |
| NAMASTE | Respect / Greeting | 2 | Both palms pressed flat together at chest |
| THANK YOU | Courtesy | 1 | Flat hand from chin moving forward & down |
| PLEASE | Politeness | 2 | Dominant palm rubbing circular on flat palm |
| YES | Affirmation | 1 | Closed fist nodding up and down from wrist |
| NO | Negation | 1 | Index & Middle fingers waving side to side |
| HELP | Assistance | 2 | Fist on flat palm lifting upward together |
| HOW | Question | 2 | Curved hands rotating upward |
| ARE | Auxiliary | 1 | Flat hand sweeping smoothly forward |
| YOU | Pronoun | 1 | Index finger pointing forward at camera |
| GOOD | Descriptor | 1 | Decisive thumbs up with 4 fingers curled |
| BAD | Disapproval | 1 | Decisive thumbs down with 4 fingers curled |
| LOVE | Affection | 1 | 'I Love You': Thumb, Index, Pinky extended |
| PEACE | Victory | 1 | Victory V-sign: Index & Middle spread in V |
| STOP | Command | 1 | Open vertical palm facing camera firmly |
| OK | Agreement | 1 | Thumb & Index pinch circle, 3 fingers up |
| SORRY | Apology | 1 | Fist rubbing circular on chest / heart |
| WATER | Sustenance | 1 | 3 fingers upright in 'W' shape near mouth |
| WELCOME | Courtesy | 2 | Both palms up, sweeping inward toward body |
| NEUTRAL | Resting State | 0 | Resting hands / transitions / idle |

> **Note:** All single-handed gestures support both **Left** and **Right** hands!

---

## Model Performance & Metrics

> **Evaluated Test Accuracy: 86.25%** across 480 held-out test sequences on all 20 classes.
> Confusion matrix visualization saved to `model/confusion_matrix.png`.

---

## Live Recording & Dataset Expansion (Optional)

You can record additional custom live gesture data using your webcam at any time:

---

## Installation (Already Done)

The virtual environment `C:\signenv` is already set up. If you need to reinstall:

```bat
python -m venv C:\signenv
C:\signenv\Scripts\pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
C:\signenv\Scripts\pip install fastapi uvicorn[standard] websockets mediapipe opencv-python numpy scikit-learn matplotlib seaborn pyttsx3 pydantic
```

> **Note:** Use `C:\signenv` (short path) to avoid Windows MAX_PATH errors with deep package paths.

---

## Running the Application

### Start the server

```bat
start.bat
# or manually:
cd C:\Users\teega\.gemini\antigravity\scratch\signbridge-ai
C:\signenv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

### Open in browser

```
http://localhost:8000
```

> Requires a modern browser (Chrome, Edge, Firefox) with webcam access. Use `localhost` or `https://` — webcam requires a secure context.

---

## Retraining the Model

```bat
# Full retrain + evaluation
train.bat

# Or manually with options
C:\signenv\Scripts\python.exe model\train_model.py --epochs 60 --batch_size 32 --patience 15
C:\signenv\Scripts\python.exe model\evaluate_model.py
```

---

## REST API Reference

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/status` | Server health, model status, vocabulary |
| GET | `/api/vocabulary` | Detailed gesture descriptions |
| POST | `/api/speak` | Server-side TTS (pyttsx3 fallback) |
| POST | `/api/model/reload` | Hot-reload newly trained model |
| WS  | `/ws/landmarks` | Real-time landmark → prediction stream |

---

## WebSocket Protocol

**Client → Server** (`type: "landmarks"`):
```json
{
  "type": "landmarks",
  "left_hand": [{"x":0.5,"y":0.4,"z":-0.1}, ...],
  "right_hand": null,
  "client_timestamp": 1700000000000
}
```

**Server → Client** (`type: "prediction"`):
```json
{
  "type": "prediction",
  "predicted_word": "HELLO",
  "confidence": 0.87,
  "candidate_word": "HELLO",
  "candidate_progress": 0.6,
  "is_holding": true,
  "last_confirmed": null
}
```

**Server → Client** (`type: "word_confirmed"`):
```json
{
  "type": "word_confirmed",
  "word": "HELLO",
  "sentence_state": {"words": ["HELLO"], "sentence": "Hello", ...}
}
```
