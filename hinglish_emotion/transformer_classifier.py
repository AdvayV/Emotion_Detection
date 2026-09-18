from __future__ import annotations

from pathlib import Path

from .normalization import NormalizedMessage
from .schemas import ClassifierResult


class LocalTransformerClassifier:
    """Inference adapter for a locally saved Hugging Face classifier.

    Model loading is lazy so the normalization and evaluation tools remain
    usable on machines without PyTorch or Transformers.
    """

    def __init__(self, model_path: str | Path, max_length: int = 128, device: str | None = None) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install the optional model dependencies before loading a transformer") from exc

        self.torch = torch
        self.model_path = str(model_path)
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path, local_files_only=True
        ).to(self.device)
        self.model.eval()

    def predict(self, message: NormalizedMessage) -> ClassifierResult:
        encoded = self.tokenizer(
            message.normalized_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with self.torch.no_grad():
            logits = self.model(**encoded).logits[0]
            values = self.torch.softmax(logits, dim=-1).detach().cpu().tolist()

        id2label = {int(key): value.lower() for key, value in self.model.config.id2label.items()}
        probabilities = {id2label[index]: float(value) for index, value in enumerate(values)}
        required = {"positive", "neutral", "negative"}
        if set(probabilities) != required:
            raise RuntimeError(f"Local model label mapping must be {sorted(required)}, got {sorted(probabilities)}")
        label = max(probabilities, key=probabilities.get)
        return ClassifierResult(label=label, probabilities=probabilities)  # type: ignore[arg-type]

