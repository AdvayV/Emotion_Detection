from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Protocol

from .normalization import HinglishNormalizer, NormalizedMessage
from .routing import ReviewDecision, ReviewRouter
from .schemas import ClassifierResult, ReviewerResult, SentimentLabel


class PrimaryClassifier(Protocol):
    def predict(self, message: NormalizedMessage) -> ClassifierResult: ...


class LocalReviewer(Protocol):
    def review(self, raw_text: str, normalized_text: str, primary: ClassifierResult) -> ReviewerResult: ...


POSITIVE_WORDS = {"accha", "acchi", "good", "happy", "love", "mast", "great", "best", "finally", "wah"}
NEGATIVE_WORDS = {"bad", "sad", "hate", "bakwaas", "bekaar", "late", "fail", "failed", "worst"}
POSITIVE_SYMBOLS = {"❤", "❤️", "😊", "😍", "🔥", "👍", "🎉"}
NEGATIVE_SYMBOLS = {"😡", "😒", "😞", "😢", "👎"}
AMBIGUOUS_SYMBOLS = {"😭", "😂"}


@dataclass(frozen=True)
class PipelinePrediction:
    normalized: NormalizedMessage
    primary: ClassifierResult
    review_decision: ReviewDecision
    reviewer: ReviewerResult | None
    final_label: SentimentLabel
    text_emoji_conflict: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "normalized": self.normalized.to_dict(),
            "primary": self.primary.to_dict(),
            "review_decision": self.review_decision.to_dict(),
            "reviewer": None if self.reviewer is None else self.reviewer.to_dict(),
            "final_label": self.final_label,
            "text_emoji_conflict": self.text_emoji_conflict,
        }


class RuleBasedClassifier:
    """Transparent smoke-test classifier, not the final research model."""

    def predict(self, message: NormalizedMessage) -> ClassifierResult:
        words = [token.normalized for token in message.tokens if token.kind in {"word", "hashtag"}]
        score = 0.0
        evidence: list[str] = []
        for index, word in enumerate(words):
            if word in {"nahi", "not", "never", "no", "mat", "without"}:
                evidence.append(word)
                continue
            contribution = 1.0 if word in POSITIVE_WORDS else -1.0 if word in NEGATIVE_WORDS else 0.0
            nearby_words = words[max(0, index - 3) : index] + words[index + 1 : index + 4]
            nearby_negation = any(
                candidate in {"nahi", "not", "never", "no", "mat", "without"}
                for candidate in nearby_words
            )
            if contribution and nearby_negation:
                contribution *= -1
            if contribution:
                score += contribution
                evidence.append(word)

        emoji_score = sum(symbol in POSITIVE_SYMBOLS for symbol in message.emojis) - sum(
            symbol in NEGATIVE_SYMBOLS for symbol in message.emojis
        )
        score += 0.7 * emoji_score
        evidence.extend(symbol for symbol in message.emojis if symbol in POSITIVE_SYMBOLS | NEGATIVE_SYMBOLS | AMBIGUOUS_SYMBOLS)

        # Softmax over deliberately small heuristic logits. This gives the
        # router realistic uncertainty during integration tests.
        logits = {"positive": score, "neutral": 0.75 - abs(score) * 0.25, "negative": -score}
        probabilities = self._softmax(logits)
        label = max(probabilities, key=probabilities.get)

        literal_positive = any(word in POSITIVE_WORDS for word in words)
        negative_situation = any(word in NEGATIVE_WORDS for word in words)
        laughter = any(symbol == "😂" for symbol in message.emojis)
        sarcasm_probability = 0.75 if literal_positive and negative_situation and laughter else 0.10

        return ClassifierResult(
            label=label,  # type: ignore[arg-type]
            probabilities=probabilities,  # type: ignore[arg-type]
            sarcasm_probability=sarcasm_probability,
            evidence=tuple(dict.fromkeys(evidence)),
        )

    @staticmethod
    def _softmax(logits: dict[str, float]) -> dict[str, float]:
        maximum = max(logits.values())
        exponentials = {key: math.exp(value - maximum) for key, value in logits.items()}
        total = sum(exponentials.values())
        return {key: value / total for key, value in exponentials.items()}


class EmotionPipeline:
    def __init__(
        self,
        classifier: PrimaryClassifier | None = None,
        normalizer: HinglishNormalizer | None = None,
        router: ReviewRouter | None = None,
        reviewer: LocalReviewer | None = None,
    ) -> None:
        self.normalizer = normalizer or HinglishNormalizer()
        self.classifier = classifier or RuleBasedClassifier()
        self.router = router or ReviewRouter()
        self.reviewer = reviewer

    def predict(self, text: str) -> PipelinePrediction:
        normalized = self.normalizer.normalize(text)
        primary = self.classifier.predict(normalized)
        conflict = self._text_emoji_conflict(normalized, primary)
        decision = self.router.decide(
            confidence=primary.confidence,
            sarcasm_probability=primary.sarcasm_probability,
            text_emoji_conflict=conflict,
            has_negation=normalized.has_negation,
        )
        reviewer_result = None
        if decision.should_review and self.reviewer is not None:
            reviewer_result = self.reviewer.review(text, normalized.normalized_text, primary)
        final_label = reviewer_result.sentiment if reviewer_result is not None else primary.label
        return PipelinePrediction(normalized, primary, decision, reviewer_result, final_label, conflict)

    @staticmethod
    def _text_emoji_conflict(message: NormalizedMessage, result: ClassifierResult) -> bool:
        has_positive = any(symbol in POSITIVE_SYMBOLS for symbol in message.emojis)
        has_negative = any(symbol in NEGATIVE_SYMBOLS for symbol in message.emojis)
        has_ambiguous = any(symbol in AMBIGUOUS_SYMBOLS for symbol in message.emojis)
        if result.label == "positive" and has_negative:
            return True
        if result.label == "negative" and has_positive:
            return True
        return has_ambiguous and result.confidence < 0.75
