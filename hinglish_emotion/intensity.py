"""Transparent emotion-intensity features for future valence/arousal labels."""
from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class IntensityEstimate:
    valence: float
    arousal: float
    strength: float
    cues: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {"valence": self.valence, "arousal": self.arousal, "strength": self.strength, "cues": list(self.cues)}


class IntensityEstimator:
    """A reproducible heuristic baseline; supervised V/A is a later ablation."""

    def estimate(self, text: str) -> IntensityEstimate:
        lowered = text.lower()
        cues: list[str] = []
        valence = 0.0
        arousal = 0.0
        if re.search(r"!{1,}", text):
            arousal += 0.20; cues.append("exclamation")
        if re.search(r"\?{2,}", text):
            arousal += 0.15; cues.append("repeated_question")
        if re.search(r"(.)\1{2,}", lowered):
            arousal += 0.20; cues.append("elongation")
        positive = ("accha", "acchi", "mast", "love", "happy", "great", "wah")
        negative = ("bad", "bakwaas", "bekaar", "hate", "sad", "worst", "nahi")
        for word in positive:
            if word in lowered:
                valence += 0.20; cues.append(word)
        for word in negative:
            if word in lowered:
                valence -= 0.20; cues.append(word)
        emoji_arousal = sum(symbol in text for symbol in ("😂", "😭", "😡", "🔥"))
        if emoji_arousal:
            arousal += min(0.4, 0.15 * emoji_arousal); cues.append("emotional_emoji")
        valence = max(-1.0, min(1.0, valence))
        arousal = max(0.0, min(1.0, arousal))
        return IntensityEstimate(valence, arousal, max(abs(valence), arousal), tuple(dict.fromkeys(cues)))
