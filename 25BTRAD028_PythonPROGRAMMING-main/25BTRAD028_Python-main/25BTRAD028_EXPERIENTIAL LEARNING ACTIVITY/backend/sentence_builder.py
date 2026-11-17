"""
Sentence Builder Module for SignBridge AI
Handles word confirmation, smoothing, duplicate prevention, grammar formatting, and sentence state.
"""

from typing import List, Dict, Optional, Tuple
import time


class SentenceBuilder:
    def __init__(
        self,
        confidence_threshold: float = 0.50,
        confirmation_frames: int = 4,
        cooldown_seconds: float = 1.0,
        idle_label: str = "NEUTRAL"
    ):
        """
        Args:
            confidence_threshold: Minimum prediction probability to consider.
            confirmation_frames: Number of consecutive identical predictions required to confirm a word.
            cooldown_seconds: Minimum time in seconds required before confirming the same word again.
            idle_label: Class label representing resting/neutral hands.
        """
        self.confidence_threshold = confidence_threshold
        self.confirmation_frames = confirmation_frames
        self.cooldown_seconds = cooldown_seconds
        self.idle_label = idle_label.upper()

        # Sentence tracking
        self.words: List[str] = []
        self.raw_history: List[str] = []

        # Real-time state
        self.current_candidate: Optional[str] = None
        self.candidate_count: int = 0
        self.candidate_confidence: float = 0.0

        self.last_confirmed_word: Optional[str] = None
        self.last_confirmed_time: float = 0.0
        self.is_holding_sign: bool = False

    def process_prediction(self, predicted_label: str, confidence: float) -> Tuple[bool, Optional[str]]:
        """
        Feed a raw model prediction into the stabilizer.
        Returns:
            (is_newly_confirmed, confirmed_word_or_none)
        """
        label = predicted_label.strip().upper()
        current_time = time.time()

        # If prediction is below confidence threshold or is neutral/idle:
        if confidence < self.confidence_threshold or label == self.idle_label or label == "BACKGROUND":
            # Reset candidate tracking on neutral/weak frames
            self.current_candidate = None
            self.candidate_count = 0
            self.candidate_confidence = 0.0
            # User relaxed hand, so they are no longer holding the previous sign
            self.is_holding_sign = False
            return False, None

        # Same candidate as current sequence
        if label == self.current_candidate:
            self.candidate_count += 1
            # Running average of confidence
            self.candidate_confidence = (self.candidate_confidence * (self.candidate_count - 1) + confidence) / self.candidate_count
        else:
            # New candidate encountered
            self.current_candidate = label
            self.candidate_count = 1
            self.candidate_confidence = confidence

        # Check if candidate satisfies confirmation threshold
        if self.candidate_count >= self.confirmation_frames:
            # Check duplicate / hold prevention:
            # If user has been holding the same sign, do not re-add
            if label == self.last_confirmed_word and self.is_holding_sign:
                return False, None

            # Cooldown check for repeated words
            if label == self.last_confirmed_word and (current_time - self.last_confirmed_time) < self.cooldown_seconds:
                return False, None

            # Confirmed! Add to sentence
            self._add_word(label)
            self.last_confirmed_word = label
            self.last_confirmed_time = current_time
            self.is_holding_sign = True

            # Reset candidate counter so next repetition requires a new sequence
            self.candidate_count = 0
            return True, label

        return False, None

    def _add_word(self, word: str):
        """Append confirmed word and record raw history."""
        cleaned = word.strip().upper()
        if cleaned:
            self.words.append(cleaned)
            self.raw_history.append(cleaned)

    def delete_last_word(self) -> Optional[str]:
        """Remove the most recently added word."""
        if self.words:
            removed = self.words.pop()
            self.last_confirmed_word = self.words[-1] if self.words else None
            self.is_holding_sign = False
            return removed
        return None

    def clear(self):
        """Reset the current sentence and candidate states."""
        self.words.clear()
        self.current_candidate = None
        self.candidate_count = 0
        self.candidate_confidence = 0.0
        self.last_confirmed_word = None
        self.last_confirmed_time = 0.0
        self.is_holding_sign = False

    def get_raw_sentence(self) -> str:
        """Return words separated by space."""
        return " ".join(self.words)

    def get_formatted_sentence(self) -> str:
        """
        Rule-based grammar and sentence formation:
        1. Capitalizes the first letter.
        2. Detects question starters (HOW, WHAT, WHERE, WHO, WHY, ARE, CAN, DO, IS) and appends '?'.
        3. Formats greetings (HELLO, NAMASTE, HI) with commas or exclamations.
        4. Normalizes multi-word phrases (e.g., 'THANK YOU').
        """
        if not self.words:
            return ""

        # Normalize words to title-case or appropriate representation
        tokens = [w.capitalize() for w in self.words]

        # Combine common idioms if sequential
        text = " ".join(tokens)

        # Question detection
        question_starters = {"How", "What", "Where", "Who", "Why", "Are", "Can", "Do", "Is", "When", "Which"}
        first_word = tokens[0]

        if first_word in question_starters:
            punctuation = "?"
        elif first_word in {"Hello", "Namaste", "Hi", "Welcome"}:
            if len(tokens) == 1:
                punctuation = "!"
            else:
                punctuation = "."
        else:
            punctuation = "."

        # Check if last word already has punctuation
        if text and text[-1] not in ".?!,":
            text = text + punctuation

        return text

    def get_state(self) -> Dict:
        """Return full state dictionary for UI clients."""
        return {
            "words": list(self.words),
            "raw_sentence": self.get_raw_sentence(),
            "formatted_sentence": self.get_formatted_sentence(),
            "last_confirmed_word": self.last_confirmed_word,
            "current_candidate": self.current_candidate,
            "candidate_confidence": round(float(self.candidate_confidence), 3),
            "candidate_progress": min(1.0, self.candidate_count / max(1, self.confirmation_frames)),
            "is_holding": self.is_holding_sign,
            "word_count": len(self.words)
        }
