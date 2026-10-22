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
        return self.truncated or scores[0] < 0.55 or scores[0] - scores[1] < 0.15

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
        self.fusion = None
        if self.metadata.get("architecture") == "text_emoji_multitask_v1":
            from .fusion import LocalFusionClassifier
            self.fusion = LocalFusionClassifier(model_path, device=device)
            self.model_path = str(model_path)
            self.model = self.fusion.model
            self.tokenizer = self.fusion.tokenizer
            self.torch = self.fusion.torch
            self.device = self.fusion.device
            self.required_labels = set(EMOTION_LABELS)
            self.max_length = self.metadata["max_length"]
            return
        limit = self.metadata["max_length"] if max_length is None else max_length
        super().__init__(model_path, limit, device, required_labels=EMOTION_LABELS)

    def predict_with_context(self, text: str, context: str) -> EmotionResult:
        if self.fusion is not None:
            return self.fusion.predict_with_context(text, context)
        result = self.predict_text(text)
        return replace(result, warnings=(*result.warnings,
            "This encoder was trained on individual phrases; supplied context is only available to the optional reviewer."))

    def predict_text(self, text: str) -> EmotionResult:
        if self.fusion is not None:
            return self.fusion.predict_text(text)
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

    def predict_batch(self, texts: list[str], batch_size: int = 8) -> list[EmotionResult]:
        if self.fusion is not None:
            return self.fusion.predict_batch(texts, batch_size)
        if batch_size < 1 or any(not text.strip() for text in texts):
            raise ValueError("Provide non-empty phrases and a positive batch size")
        results = []
        id2label = {int(key): value.lower() for key, value in self.model.config.id2label.items()}
        if set(id2label.values()) != self.required_labels:
            raise RuntimeError("The checkpoint does not have the seven required emotion labels")
        for offset in range(0, len(texts), batch_size):
            prepared = [prepare_text(text, self.metadata["text_mode"], self.metadata["preparation_version"]) for text in texts[offset:offset + batch_size]]
            lengths = [len(ids) for ids in self.tokenizer(prepared, truncation=False)["input_ids"]]
            encoded = self.tokenizer(prepared, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length)
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with self.torch.no_grad():
                values = self.torch.softmax(self.model(**encoded).logits, dim=-1).cpu().tolist()
            for scores, length in zip(values, lengths):
                probabilities = {id2label[index]: float(value) for index, value in enumerate(scores)}
                synthetic = self.metadata.get("evaluation_scope") in {"synthetic_only", "mixed_test"}
                warnings = []
                if synthetic:
                    warnings.append("Synthetic or mixed synthetic test data: real-world human-labelled accuracy is not established.")
                if self.metadata["legacy"]:
                    warnings.append("Legacy checkpoint: training provenance and language validation are unknown.")
                if length > self.max_length:
                    warnings.append(f"Input was truncated to {self.max_length} tokens.")
                results.append(EmotionResult(max(probabilities, key=probabilities.get), probabilities, source="trained_checkpoint", model_name=self.metadata.get("base_model", self.model_path), experimental=synthetic, warnings=tuple(warnings), truncated=length > self.max_length))
        return results


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

    def predict_context(self, text: str, language_pair: str = "hi-en", context: str = "") -> EmotionResult:
        if not text.strip() or language_pair not in LANGUAGE_PAIRS:
            raise ValueError("Use a non-empty phrase and supported language mix")
        if not context.strip() or not hasattr(self.classifier, "predict_with_context"):
            return self.predict(text, language_pair)
        result = self.classifier.predict_with_context(text, context)
        if language_pair not in self.classifier.metadata.get("validated_language_pairs", []):
            result = replace(result, experimental=True, warnings=(*result.warnings,
                f"{LANGUAGE_PAIRS[language_pair]} has no verified seven-class test coverage for this checkpoint."))
        return result

    def predict_many(self, texts: list[str], language_pairs: list[str], batch_size: int = 8) -> list[EmotionResult]:
        if len(texts) != len(language_pairs) or any(pair not in LANGUAGE_PAIRS for pair in language_pairs):
            raise ValueError("Every phrase needs a supported language mix")
        if not hasattr(self.classifier, "predict_batch"):
            return [self.predict(text, pair) for text, pair in zip(texts, language_pairs)]
        results = self.classifier.predict_batch(texts, batch_size)
        verified = self.classifier.metadata.get("validated_language_pairs", [])
        return [replace(result, experimental=True, warnings=(*result.warnings, f"{LANGUAGE_PAIRS[pair]} has no verified seven-class test coverage for this checkpoint.")) if pair not in verified else result for result, pair in zip(results, language_pairs)]
