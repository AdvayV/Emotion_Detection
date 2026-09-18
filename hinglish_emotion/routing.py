from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RouterConfig:
    confidence_threshold: float = 0.62
    sarcasm_threshold: float = 0.60
    route_text_emoji_conflicts: bool = True
    route_low_confidence_negation: bool = True


@dataclass(frozen=True)
class ReviewDecision:
    should_review: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class ReviewRouter:
    def __init__(self, config: RouterConfig | None = None) -> None:
        self.config = config or RouterConfig()

    def decide(
        self,
        *,
        confidence: float,
        sarcasm_probability: float = 0.0,
        text_emoji_conflict: bool = False,
        has_negation: bool = False,
        classifier_disagreement: bool = False,
    ) -> ReviewDecision:
        reasons: list[str] = []
        if confidence < self.config.confidence_threshold:
            reasons.append("low_confidence")
        if sarcasm_probability >= self.config.sarcasm_threshold:
            reasons.append("possible_sarcasm")
        if text_emoji_conflict and self.config.route_text_emoji_conflicts:
            reasons.append("text_emoji_conflict")
        if classifier_disagreement:
            reasons.append("classifier_disagreement")
        if (
            has_negation
            and self.config.route_low_confidence_negation
            and confidence < min(0.75, self.config.confidence_threshold + 0.10)
        ):
            reasons.append("uncertain_negation")
        return ReviewDecision(bool(reasons), tuple(dict.fromkeys(reasons)))

