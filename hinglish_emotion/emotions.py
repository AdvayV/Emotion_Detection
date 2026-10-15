"""Seven-emotion classification, separate from three-way sentiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
import re

from .normalization import HinglishNormalizer, NormalizedMessage
from .tasks import EMOTION_LABELS, LANGUAGE_PAIRS
from .transformer_classifier import LocalTransformerClassifier
from .checkpoint_metadata import emotion_checkpoint_metadata
from .text_preparation import prepare_text


@dataclass(frozen=True)
class EmotionResult:
    emotion: str
    probabilities: dict[str, float]
    evidence: tuple[str, ...] = ()
    source: str = "lexical_demo"
    model_name: str = "Demo lexical rules"
    experimental: bool = False
    warnings: tuple[str, ...] = ()
    truncated: bool = False

    @property
    def confidence(self) -> float:
        return self.probabilities[self.emotion]

    @property
    def needs_context(self) -> bool:
        scores = sorted(self.probabilities.values(), reverse=True)
        return self.experimental or self.truncated or scores[0] < 0.55 or scores[0] - scores[1] < 0.15

    def to_dict(self) -> dict[str, object]:
        return {**asdict(self), "confidence": self.confidence, "needs_context": self.needs_context}


# A transparent demo vocabulary. These scores are not calibrated model probabilities.
EMOTION_CUES = {
    "joy": ("happy", "joy", "delighted", "excited", "khush", "khushi", "maza", "mast", "खुश", "खुशी", "heureux", "heureuse", "content", "contente", "ravi", "ravie", "joie", "😊", "🎉"),
    "sadness": ("sad", "sadness", "heartbroken", "miss you", "dukhi", "udaas", "rona", "दुखी", "उदास", "triste", "tristesse", "déçu", "déçue", "chagrin", "😢", "💔"),
    "anger": ("angry", "furious", "rage", "gussa", "naraz", "गुस्सा", "नाराज", "colère", "furieux", "furieuse", "énervé", "énervée", "😡"),
    "fear": ("afraid", "scared", "terrified", "fear", "dar", "darr", "darna", "डर", "डरता", "peur", "effrayé", "effrayée", "terrifié", "terrifiée", "😨"),
    "surprise": ("surprised", "surprise", "unexpected", "shocked", "hairan", "achanak", "हैरान", "अचानक", "surpris", "surprise", "étonné", "étonnée", "inattendu", "😲"),
    "disgust": ("disgusted", "disgusting", "revolting", "gross", "ghin", "ghinauna", "घिन", "घिनौना", "dégoût", "dégoûté", "dégoûtée", "dégoûtant", "écœurant", "🤢"),
}
NEGATED_BEFORE = {"not", "never", "no", "nahi", "mat", "pas", "jamais", "sans", "नहीं", "don't", "can't", "cannot", "isn't", "wasn't"}
NEGATED_AFTER = {"nahi", "nahin", "नहीं"}


class RuleEmotionClassifier:
    """Lexical demo for trying both language mixes before real training."""

    def predict(self, message: NormalizedMessage) -> EmotionResult:
        text = message.normalized_text.lower().replace("’", "'")
        logits = {label: 0.0 for label in EMOTION_LABELS}
        logits["neutral"] = 1.0
        evidence: list[str] = []
        for label, phrases in EMOTION_CUES.items():
            for phrase in set(phrases):
                pattern = r"(?<!\w)" + re.escape(phrase) + r"(?!\w)"
                for match in re.finditer(pattern, text):
                    before = text[:match.start()].split()[-3:]
                    after = text[match.end():].split()[:2]
                    if any(word in NEGATED_BEFORE for word in before) or any(word in NEGATED_AFTER for word in after):
                        continue
                    logits[label] += 2.5
                    evidence.append(match.group())
        maximum = max(logits.values())
        values = {label: math.exp(value - maximum) for label, value in logits.items()}
        total = sum(values.values())
        probabilities = {label: value / total for label, value in values.items()}
        emotion = max(probabilities, key=probabilities.get)
        return EmotionResult(emotion, probabilities, tuple(dict.fromkeys(evidence)))


class LocalEmotionClassifier(LocalTransformerClassifier):
    def __init__(self, model_path, max_length: int | None = None, device: str | None = None) -> None:
        self.metadata = emotion_checkpoint_metadata(model_path)
        limit = self.metadata["max_length"] if max_length is None else max_length
        super().__init__(model_path, limit, device, required_labels=EMOTION_LABELS)

    def predict_text(self, text: str) -> EmotionResult:
        prepared = prepare_text(text, self.metadata["text_mode"], self.metadata["preparation_version"])
        token_ids = self.tokenizer(prepared, add_special_tokens=True, truncation=False)["input_ids"]
        truncated = len(token_ids) > self.max_length
        probabilities = self.predict_text_probabilities(prepared)
        warnings = []
        synthetic_evaluation = self.metadata.get("evaluation_scope") in {"synthetic_only", "mixed_test"}
        if synthetic_evaluation:
            warnings.append("Synthetic or mixed synthetic test data: real-world human-labelled accuracy is not established.")
        if self.metadata["legacy"]:
            warnings.append("Legacy checkpoint: training provenance and language validation are unknown.")
        if truncated:
            warnings.append(f"Input was truncated to {self.max_length} tokens.")
        return EmotionResult(
            max(probabilities, key=probabilities.get), probabilities,
            source="trained_checkpoint", model_name=self.metadata.get("base_model", self.model_path),
            warnings=tuple(warnings), truncated=truncated, experimental=synthetic_evaluation,
        )

    def predict(self, message: NormalizedMessage) -> EmotionResult:
        return self.predict_text(message.raw_text)


class EmotionDetector:
    def __init__(self, classifier=None) -> None:
        self.classifier = classifier or RuleEmotionClassifier()
        self.normalizer = HinglishNormalizer()

    def predict(self, text: str, language_pair: str = "hi-en") -> EmotionResult:
        if not text.strip():
            raise ValueError("Enter a non-empty phrase")
        if language_pair not in LANGUAGE_PAIRS:
            raise ValueError("Language mix must be hi-en or fr-en")
        if hasattr(self.classifier, "predict_text"):
            result = self.classifier.predict_text(text)
            verified = self.classifier.metadata.get("validated_language_pairs", [])
            if language_pair not in verified:
                message = f"{LANGUAGE_PAIRS[language_pair]} has no verified seven-class test coverage for this checkpoint."
                result = replace(result, experimental=True, warnings=(*result.warnings, message))
            return result
        result = self.classifier.predict(self.normalizer.normalize(text))
        if language_pair == "fr-en":
            result = replace(result, experimental=True, warnings=("French–English is experimental until bilingual review is available.",))
        return result
