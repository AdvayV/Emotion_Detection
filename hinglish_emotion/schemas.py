from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


SentimentLabel = Literal["positive", "neutral", "negative"]


@dataclass(frozen=True)
class ClassifierResult:
    label: SentimentLabel
    probabilities: dict[SentimentLabel, float]
    sarcasm_probability: float = 0.0
    evidence: tuple[str, ...] = ()

    @property
    def confidence(self) -> float:
        return self.probabilities[self.label]

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "probabilities": dict(self.probabilities),
            "confidence": self.confidence,
            "sarcasm_probability": self.sarcasm_probability,
            "evidence": list(self.evidence),
        }


@dataclass(frozen=True)
class ReviewerResult:
    sentiment: SentimentLabel
    sarcasm: bool
    text_emoji_relation: Literal["agreement", "conflict", "none", "uncertain"]
    evidence: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "ReviewerResult":
        sentiment = str(value.get("sentiment", "")).lower()
        relation = str(value.get("text_emoji_relation", "")).lower()
        if sentiment not in {"positive", "neutral", "negative"}:
            raise ValueError("Reviewer returned an invalid sentiment label")
        if relation not in {"agreement", "conflict", "none", "uncertain"}:
            raise ValueError("Reviewer returned an invalid text_emoji_relation")
        evidence_value = value.get("evidence", [])
        if isinstance(evidence_value, str):
            evidence = (evidence_value,)
        elif isinstance(evidence_value, list) and all(isinstance(item, str) for item in evidence_value):
            evidence = tuple(evidence_value)
        else:
            raise ValueError("Reviewer evidence must be text or a list of text spans")
        return cls(
            sentiment=sentiment,  # type: ignore[arg-type]
            sarcasm=bool(value.get("sarcasm", False)),
            text_emoji_relation=relation,  # type: ignore[arg-type]
            evidence=evidence,
        )

