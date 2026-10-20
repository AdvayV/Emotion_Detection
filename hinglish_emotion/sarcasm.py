"""Local-only adapter for the separately supervised binary sarcasm task."""
from __future__ import annotations

import json
from pathlib import Path


class LocalSarcasmClassifier:
    def __init__(self, path):
        path = Path(path)
        self.metadata = json.loads((path / "sarcasm_metadata.json").read_text(encoding="utf-8"))
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
        expected = ["not_sarcastic", "sarcastic"]
        if (self.metadata.get("task") != "sarcasm_binary" or self.metadata.get("training_status") != "complete"
                or self.metadata.get("labels") != expected or self.metadata.get("human_reviewed") is not True
                or self.metadata.get("base_model") != "FacebookAI/xlm-roberta-base"
                or config.get("model_type") != "xlm-roberta"
                or config.get("id2label") != {str(i): label for i, label in enumerate(expected)}
                or not list(path.glob("*.safetensors"))):
            raise ValueError("Use a completed, reviewed binary sarcasm SafeTensors checkpoint")
        if not set(self.metadata.get("language_pairs", [])) <= {"hi-en"} or not self.metadata.get("language_pairs"):
            raise ValueError("Sarcasm checkpoint must declare its Hindi-English language scope")
        max_length = self.metadata.get("max_length", 128)
        if type(max_length) is not int or not 1 <= max_length <= 512:
            raise ValueError("Invalid sarcasm checkpoint maximum sequence length")
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(path), local_files_only=True, trust_remote_code=False, use_safetensors=True).eval()

    def predict_text(self, text, language_pair="hi-en"):
        if language_pair not in self.metadata["language_pairs"]:
            raise ValueError("No trained sarcasm classifier for this language pair")
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True,
                                max_length=int(self.metadata.get("max_length", 128)))
        with self.torch.inference_mode():
            return self.model(**inputs).logits.softmax(-1)[0, 1].item()
