"""Conversation-level features for emotion-shift experiments.

This module deliberately sits outside the single-message classifier. It lets
the project compare sentence-only predictions with a lightweight speaker-aware
history baseline once conversation labels are available.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from .schemas import SentimentLabel


@dataclass(frozen=True)
class TurnRecord:
    conversation_id: str
    turn_id: int
    speaker_id: str
    text: str
    sentiment: SentimentLabel
    confidence: float


@dataclass(frozen=True)
class EmotionShift:
    previous: SentimentLabel | None
    current: SentimentLabel
    changed: bool
    direction: Literal["none", "positive_to_negative", "negative_to_positive", "to_neutral", "from_neutral"]


class ConversationMemory:
    """Keep a bounded history and report interpretable emotion changes."""

    def __init__(self, max_turns: int = 8) -> None:
        if max_turns < 1:
            raise ValueError("max_turns must be positive")
        self.max_turns = max_turns
        self._turns: list[TurnRecord] = []

    @property
    def turns(self) -> tuple[TurnRecord, ...]:
        return tuple(self._turns)

    def add(self, turn: TurnRecord) -> EmotionShift:
        previous = self._previous_for_speaker(turn.speaker_id)
        shift = EmotionShift(previous, turn.sentiment, previous != turn.sentiment, self._direction(previous, turn.sentiment))
        self._turns.append(turn)
        self._turns = self._turns[-self.max_turns :]
        return shift

    def to_dict(self) -> dict[str, object]:
        return {"max_turns": self.max_turns, "turns": [asdict(turn) for turn in self._turns]}

    def _previous_for_speaker(self, speaker_id: str) -> SentimentLabel | None:
        for turn in reversed(self._turns):
            if turn.speaker_id == speaker_id:
                return turn.sentiment
        return None

    @staticmethod
    def _direction(previous: SentimentLabel | None, current: SentimentLabel) -> str:
        if previous is None or previous == current:
            return "none"
        if previous == "positive" and current == "negative":
            return "positive_to_negative"
        if previous == "negative" and current == "positive":
            return "negative_to_positive"
        if current == "neutral":
            return "to_neutral"
        return "from_neutral"
