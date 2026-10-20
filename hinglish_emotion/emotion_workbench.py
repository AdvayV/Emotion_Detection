"""Join phrase prediction and transparent diagnostics without mixing score types."""
from __future__ import annotations

from dataclasses import dataclass
import math

from .emotions import EmotionResult
from .intensity import IntensityEstimator
from .normalization import HinglishNormalizer
from .pipeline import RuleBasedClassifier, POSITIVE_SYMBOLS, NEGATIVE_SYMBOLS, AMBIGUOUS_SYMBOLS
from .ollama_emotion_reviewer import OllamaEmotionReviewer


@dataclass(frozen=True)
class WorkbenchResult:
    primary: EmotionResult
    final_emotion: str
    normalization: dict
    cues: dict
    review: dict
    context: str = ""

    def to_dict(self):
        return {"primary": self.primary.to_dict(), "final_emotion": self.final_emotion,
                "normalization": self.normalization, "cues": self.cues,
                "review": self.review, "context": self.context}


class EmotionWorkbench:
    def __init__(self, detector, reviewer=None, sarcasm_classifier=None):
        self.detector = detector
        self.reviewer = reviewer
        self.sarcasm_classifier = sarcasm_classifier
        self.normalizer = HinglishNormalizer()

    def analyze(self, text, language_pair="hi-en", context=""):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Enter a nonempty phrase")
        if len(text) > 20000 or len(context) > 8000:
            raise ValueError("Use a phrase under 20,000 characters and context under 8,000 characters")
        primary = self.detector.predict(text, language_pair)
        normalized = self.normalizer.normalize(text)
        baseline = RuleBasedClassifier().predict(normalized)
        # Compare words alone with emoji polarity; do not call the emotion model
        # again on modified text or present heuristic scores as learned scores.
        words = " ".join(token.raw for token in normalized.tokens if token.kind != "symbol")
        word_sentiment = RuleBasedClassifier().predict(self.normalizer.normalize(words)).label
        pos = any(symbol in text for symbol in POSITIVE_SYMBOLS)
        neg = any(symbol in text for symbol in NEGATIVE_SYMBOLS)
        amb = any(symbol in text for symbol in AMBIGUOUS_SYMBOLS)
        if not (pos or neg or amb):
            relation = "none"
        elif amb or (pos and neg) or word_sentiment == "neutral":
            relation = "uncertain"
        elif (pos and word_sentiment == "negative") or (neg and word_sentiment == "positive"):
            relation = "conflict"
        else:
            relation = "agreement"
        score = baseline.sarcasm_probability
        source, sarcasm_message = "heuristic", "Rule cue score; not a calibrated sarcasm probability."
        if self.sarcasm_classifier is not None:
            try:
                score = float(self.sarcasm_classifier.predict_text(text, language_pair=language_pair))
                if not math.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError("Invalid sarcasm score")
                source = "supervised"
                sarcasm_message = "Separate binary classifier; calibration and accuracy depend on its dataset."
            except (RuntimeError, ValueError, OSError) as exc:
                score = baseline.sarcasm_probability
                sarcasm_message = f"Sarcasm checkpoint unavailable; using rule cues. {exc}"
        reasons = []
        if primary.needs_context:
            reasons.append("Low confidence, close scores, or a truncated phrase")
        if relation == "conflict":
            reasons.append("Word sentiment and emoji polarity conflict")
        if score >= .6:
            reasons.append("Possible sarcasm")
        if context.strip():
            reasons.append("Earlier conversation supplied for review")
        if primary.experimental:
            reasons.append("Experimental model or language scope")
        cues = {"sarcasm_score": score, "sarcasm_source": source,
                "sarcasm_message": sarcasm_message, "emoji_relation": relation,
                "evidence": list(baseline.evidence), "evidence_source": "lexical cues, not model attribution",
                "intensity": {**IntensityEstimator().estimate(text).to_dict(), "source": "heuristic"},
                "review_reasons": reasons}
        review = {"status": "disabled", "suggestion": None, "evidence": [],
                  "message": "Local Qwen review is off. The primary prediction is retained."}
        final = primary.emotion
        if self.reviewer is not None:
            if not reasons:
                review.update(status="not_needed", message="No review trigger; primary prediction retained.")
            else:
                try:
                    ready, message = self.reviewer.availability()
                    if not ready:
                        review.update(status="unavailable", message=message)
                    else:
                        response = self.reviewer.review(text, normalized.normalized_text, primary,
                                                       context=context, language_pair=language_pair, cues=cues)
                        response = OllamaEmotionReviewer.validate(response, text)
                        final = response["emotion"]
                        review.update(status="applied", suggestion=final, evidence=response["evidence"],
                                      sarcasm=response["sarcasm"], message="Local review applied as a second opinion; primary scores unchanged.")
                except (RuntimeError, ValueError, OSError, KeyError, TypeError) as exc:
                    review.update(status="error", message=f"Review failed; primary prediction retained. {exc}")
        return WorkbenchResult(primary, final, normalized.to_dict(), cues, review, context)
