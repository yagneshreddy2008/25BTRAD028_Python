"""
Main FastAPI Application for SignBridge AI.
Serves static frontend, WebSocket endpoints for real-time inference, and REST APIs for vocabulary & TTS.
"""

import os
import sys
import uuid
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# Ensure local backend imports resolve
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from predictor import SignPredictor
from websocket_handler import SignWebSocketHandler
from sentence_builder import SentenceBuilder

# Paths
FRONTEND_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "frontend"))
MODEL_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "model"))
LABELS_PATH = os.path.join(MODEL_DIR, "labels.json")

# Global instances
predictor: Optional[SignPredictor] = None
ws_handler: Optional[SignWebSocketHandler] = None
translation_history = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor, ws_handler
    print("[SignBridge AI] Initializing server and ML components...")
    predictor = SignPredictor()
    ws_handler = SignWebSocketHandler(predictor)
    print(f"[SignBridge AI] Model loaded: {predictor.is_loaded}")
    print(f"[SignBridge AI] Vocabulary: {predictor.labels}")
    yield
    print("[SignBridge AI] Shutting down server...")


app = FastAPI(
    title="SignBridge AI API",
    description="Real-Time Indian Sign Language to Text & Speech Translator",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for all local development origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic Schemas
class TTSRequest(BaseModel):
    text: str
    rate: Optional[int] = 150
    volume: Optional[float] = 1.0


class ActionRequest(BaseModel):
    action: str  # "delete_last", "clear_sentence"


# -------------------------------------------------------------
# REST Endpoints
# -------------------------------------------------------------
@app.get("/api/status")
async def get_system_status():
    """Returns current health, model status, and supported vocabulary."""
    return {
        "status": "online",
        "service": "SignBridge AI",
        "model_loaded": predictor.is_loaded if predictor else False,
        "model_error": predictor.load_error if predictor else None,
        "vocabulary_count": len(predictor.labels) if predictor else 0,
        "vocabulary": predictor.labels if predictor else [],
        "target_language": "Indian Sign Language (ISL)"
    }


@app.get("/api/vocabulary")
async def get_vocabulary_details():
    """Returns detailed information and performance tips for supported ISL signs."""
    vocabulary_info = [
        {
            "label": "HELLO",
            "type": "Two-Handed / Greeting",
            "description": "Open upright hands waving gently in conversational space.",
            "hands_required": 2
        },
        {
            "label": "NAMASTE",
            "type": "Two-Handed / Indian Greeting",
            "description": "Both palms pressed flat together vertically at chest level in prayer gesture.",
            "hands_required": 2
        },
        {
            "label": "THANK YOU",
            "type": "Single-Handed / Courtesy",
            "description": "Flat hand touches chin or lips, then moves forward and slightly downward.",
            "hands_required": 1
        },
        {
            "label": "PLEASE",
            "type": "Two-Handed / Politeness",
            "description": "Open flat dominant palm gently rubbing in circles over the non-dominant palm.",
            "hands_required": 2
        },
        {
            "label": "YES",
            "type": "Single-Handed / Affirmation",
            "description": "Closed fist nodding up and down from the wrist like a nodding head.",
            "hands_required": 1
        },
        {
            "label": "NO",
            "type": "Single-Handed / Negation",
            "description": "Index and middle fingers extended waving side-to-side horizontally.",
            "hands_required": 1
        },
        {
            "label": "HELP",
            "type": "Two-Handed / Assistance",
            "description": "Fist with thumb up resting on open upward-facing opposite palm, lifting together.",
            "hands_required": 2
        },
        {
            "label": "HOW",
            "type": "Two-Handed / Question",
            "description": "Both hands curved with knuckles facing outward, rotating upward together until palms face up.",
            "hands_required": 2
        },
        {
            "label": "ARE",
            "type": "Single-Handed / Auxiliary",
            "description": "Dominant flat hand moving smoothly forward from chest into signing space.",
            "hands_required": 1
        },
        {
            "label": "YOU",
            "type": "Single-Handed / Pronoun",
            "description": "Index finger pointing directly forward toward conversational partner (webcam).",
            "hands_required": 1
        },
        {
            "label": "GOOD",
            "type": "Single-Handed / Descriptor",
            "description": "Decisive thumbs-up with thumb extended straight up and 4 fingers curled.",
            "hands_required": 1
        },
        {
            "label": "BAD",
            "type": "Single-Handed / Disapproval",
            "description": "Decisive thumbs-down with thumb extended downward and 4 fingers curled.",
            "hands_required": 1
        },
        {
            "label": "LOVE",
            "type": "Single-Handed / Affection",
            "description": "Universal 'I Love You' sign: Thumb, Index, and Pinky extended; Middle and Ring curled.",
            "hands_required": 1
        },
        {
            "label": "PEACE",
            "type": "Single-Handed / Victory",
            "description": "Victory V-sign: Index and Middle fingers extended in a spread V; others curled.",
            "hands_required": 1
        },
        {
            "label": "STOP",
            "type": "Single-Handed / Command",
            "description": "Open vertical palm facing camera firmly with all 5 fingers upright.",
            "hands_required": 1
        },
        {
            "label": "OK",
            "type": "Single-Handed / Agreement",
            "description": "Thumb and index tips touching to form a circle, other 3 fingers extended upright.",
            "hands_required": 1
        },
        {
            "label": "SORRY",
            "type": "Single-Handed / Apology",
            "description": "Closed fist making a gentle circular rubbing motion over chest or heart.",
            "hands_required": 1
        },
        {
            "label": "WATER",
            "type": "Single-Handed / Sustenance",
            "description": "Three fingers (Index, Middle, Ring) extended upright in a 'W' handshape near mouth.",
            "hands_required": 1
        },
        {
            "label": "WELCOME",
            "type": "Two-Handed / Courtesy",
            "description": "Both open palms facing upward, sweeping smoothly inward toward the body.",
            "hands_required": 2
        },
        {
            "label": "NEUTRAL",
            "type": "Resting State",
            "description": "Hands relaxed, resting on desk, or moving between signs. Prevents accidental word activations.",
            "hands_required": 0
        }
    ]
    return {"vocabulary": vocabulary_info}


@app.post("/api/speak")
async def server_tts(req: TTSRequest):
    """
    Fallback server-side TTS synthesis using pyttsx3.
    """
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", req.rate)
        engine.setProperty("volume", req.volume)
        # Run speak non-blocking or in thread pool
        import threading
        def run_voice(t):
            eng = pyttsx3.init()
            eng.say(t)
            eng.runAndWait()
        threading.Thread(target=run_voice, args=(req.text,), daemon=True).start()
        return {"status": "speaking", "text": req.text}
    except Exception as e:
        return {"status": "client_fallback_recommended", "error": str(e)}


@app.post("/api/model/reload")
async def reload_model():
    """Reloads modules and model weights if newly updated."""
    global predictor, ws_handler
    import importlib
    import preprocessing
    import predictor as predictor_mod
    import websocket_handler as ws_mod
    importlib.reload(preprocessing)
    importlib.reload(predictor_mod)
    importlib.reload(ws_mod)
    predictor = predictor_mod.SignPredictor()
    ws_handler = ws_mod.SignWebSocketHandler(predictor)
    return {
        "status": "reloaded",
        "is_loaded": predictor.is_loaded,
        "vocabulary_count": len(predictor.labels),
        "error": predictor.load_error
    }


# -------------------------------------------------------------
# WebSocket Endpoint
# -------------------------------------------------------------
@app.websocket("/ws/landmarks")
async def websocket_landmarks(websocket: WebSocket):
    """Real-time bidirectional WebSocket stream for landmark arrays."""
    client_id = str(uuid.uuid4())[:8]
    if ws_handler:
        await ws_handler.handle_connection(websocket, client_id)
    else:
        await websocket.close()


# -------------------------------------------------------------
# Static Files & Frontend Routing
# -------------------------------------------------------------
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/")
    async def serve_index():
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

    @app.get("/{full_path:path}")
    async def serve_frontend_files(full_path: str):
        file_path = os.path.join(FRONTEND_DIR, full_path)
        if os.path.exists(file_path) and os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
