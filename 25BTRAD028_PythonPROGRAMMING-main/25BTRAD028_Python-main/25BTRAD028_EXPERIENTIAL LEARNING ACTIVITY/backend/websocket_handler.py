"""
WebSocket Handler Module for SignBridge AI.
Handles real-time landmark streaming, sequence management, model inference, and sentence synchronization.
"""

import json
from typing import Dict, Any, Optional
from fastapi import WebSocket, WebSocketDisconnect
import numpy as np

from preprocessing import process_frame_landmarks, LandmarkSequenceBuffer
from sentence_builder import SentenceBuilder
from predictor import SignPredictor


class SignWebSocketHandler:
    def __init__(self, predictor: SignPredictor):
        self.predictor = predictor
        self.active_sessions: Dict[str, Dict[str, Any]] = {}

    async def handle_connection(self, websocket: WebSocket, client_id: str):
        """
        Manages the lifecycle of a real-time translation WebSocket connection.
        """
        await websocket.accept()
        print(f"[WebSocket] Client connected: {client_id}")

        # Initialize per-client session state
        buffer = LandmarkSequenceBuffer(sequence_length=30)
        sentence_builder = SentenceBuilder(
            confidence_threshold=0.50,
            confirmation_frames=4,
            cooldown_seconds=1.0,
            idle_label="NEUTRAL"
        )
        is_paused = False

        self.active_sessions[client_id] = {
            "buffer": buffer,
            "sentence_builder": sentence_builder,
            "is_paused": is_paused
        }

        # Send initial state and available labels
        await websocket.send_json({
            "type": "session_init",
            "status": "ready",
            "model_ready": self.predictor.is_loaded,
            "labels": self.predictor.labels,
            "sentence_state": sentence_builder.get_state()
        })

        try:
            while True:
                raw_text = await websocket.receive_text()
                data = json.loads(raw_text)
                msg_type = data.get("type", "")

                if msg_type == "landmarks":
                    if is_paused:
                        continue

                    left_hand = data.get("left_hand")
                    right_hand = data.get("right_hand")
                    client_ts = data.get("client_timestamp")

                    hands_detected = bool(left_hand or right_hand)

                    # Extract and normalize (126 features)
                    frame_feats = process_frame_landmarks(left_hand, right_hand, normalize=True)
                    buffer.append_frame(frame_feats, hands_detected=hands_detected)

                    # Only run prediction when hands are actually active in the buffer
                    if buffer.has_active_hands(min_ratio=0.15):
                        seq_batch = buffer.get_sequence_array()
                        predicted_label, confidence, prob_dict = self.predictor.predict(
                            seq_batch, raw_left=left_hand, raw_right=right_hand
                        )

                        # Process through sentence confirmation stabilizer
                        is_confirmed, confirmed_word = sentence_builder.process_prediction(
                            predicted_label, confidence
                        )

                        # If word was just confirmed, notify UI
                        if is_confirmed and confirmed_word:
                            await websocket.send_json({
                                "type": "word_confirmed",
                                "word": confirmed_word,
                                "sentence_state": sentence_builder.get_state(),
                                "client_timestamp": client_ts
                            })

                        # Send continuous real-time telemetry
                        state = sentence_builder.get_state()
                        active_candidate = state["current_candidate"] or (predicted_label if confidence >= 0.45 and predicted_label != "NEUTRAL" else None)
                        await websocket.send_json({
                            "type": "prediction",
                            "predicted_word": predicted_label,
                            "confidence": round(confidence, 3),
                            "candidate_word": active_candidate,
                            "candidate_progress": state["candidate_progress"],
                            "is_holding": state["is_holding"],
                            "last_confirmed": state["last_confirmed_word"],
                            "client_timestamp": client_ts
                        })
                    else:
                        # Feed neutral/idle to stabilizer when no hands detected
                        sentence_builder.process_prediction("NEUTRAL", 1.0)
                        await websocket.send_json({
                            "type": "prediction",
                            "predicted_word": "NO_HANDS",
                            "confidence": 0.0,
                            "candidate_word": None,
                            "candidate_progress": 0.0,
                            "is_holding": False,
                            "last_confirmed": sentence_builder.last_confirmed_word,
                            "client_timestamp": client_ts
                        })

                elif msg_type == "action":
                    action = data.get("action", "")
                    if action == "delete_last":
                        sentence_builder.delete_last_word()
                        await websocket.send_json({
                            "type": "sentence_update",
                            "sentence_state": sentence_builder.get_state()
                        })
                    elif action == "clear_sentence":
                        sentence_builder.clear()
                        buffer.clear()
                        await websocket.send_json({
                            "type": "sentence_update",
                            "sentence_state": sentence_builder.get_state()
                        })
                    elif action == "pause":
                        is_paused = True
                    elif action == "resume":
                        is_paused = False
                    elif action == "reset_session":
                        sentence_builder.clear()
                        buffer.clear()
                        await websocket.send_json({
                            "type": "sentence_update",
                            "sentence_state": sentence_builder.get_state()
                        })

        except WebSocketDisconnect:
            print(f"[WebSocket] Client disconnected: {client_id}")
        except Exception as e:
            print(f"[WebSocket] Session error ({client_id}): {e}")
        finally:
            if client_id in self.active_sessions:
                del self.active_sessions[client_id]
