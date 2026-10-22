"""Optional learned XLM-R text/emoji fusion; all deployed weights load locally.

Emoji sequence IDs use versioned SHA-256 buckets, not sentiment rules. Bucket
embeddings, the vector gate and auxiliary head learn only during fine-tuning.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from .tasks import EMOTION_LABELS
from .text_preparation import PREPARATION_VERSION, prepare_text

ARCHITECTURE = "text_emoji_multitask_v1"
BASE_MODEL = "FacebookAI/xlm-roberta-base"
EMOJI_BUCKETS = 4096
MAX_EMOJI = 32
CONTEXT_INPUT_POLICY = "target_first_budget_v1"


def emoji_sequence_ids(text: str, buckets: int = EMOJI_BUCKETS, limit: int = MAX_EMOJI) -> list[int]:
    """Keep Unicode emoji sequences (including ZWJ, modifiers and flags) intact."""
    import emoji
    if buckets < 2 or limit < 1:
        raise ValueError("Emoji vocabulary and sequence limit must be positive")
    sequences = [item["emoji"] for item in emoji.emoji_list(text)][:limit]
    return [1 + int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big") % (buckets - 1)
            for value in sequences]


def emoji_batch(texts, device=None, buckets=EMOJI_BUCKETS, limit=MAX_EMOJI):
    rows = [emoji_sequence_ids(text, buckets, limit) for text in texts]
    width = max(1, max(map(len, rows), default=0))
    return torch.tensor([row + [0] * (width - len(row)) for row in rows], dtype=torch.long, device=device)


def encode_fusion_inputs(tokenizer, texts, contexts=None, max_length=128):
    """Allocate token budget to current text first, then supplied context.

    The same bounded token-ID construction is used by training and inference.
    An overlong current message is safely truncated to its own available budget;
    context never causes additional current-message tokens to be discarded.
    Returns padded tensors and original total lengths for truncation disclosure.
    """
    if contexts is not None and len(contexts) != len(texts):
        raise ValueError("Provide one context per current message")
    paired = contexts is not None
    special = tokenizer.num_special_tokens_to_add(pair=paired)
    budget = max_length - special
    if budget < 1:
        raise ValueError("Token limit must leave room for current text and special tokens")
    target_ids = tokenizer(texts, add_special_tokens=False, truncation=False, verbose=False)["input_ids"]
    context_ids = (tokenizer(contexts, add_special_tokens=False, truncation=False, verbose=False)["input_ids"]
                   if paired else [None] * len(texts))
    rows, lengths = [], []
    for target, context in zip(target_ids, context_ids):
        lengths.append(len(target) + (len(context) if context is not None else 0) + special)
        target = target[:budget]
        context = context[:max(0, budget - len(target))] if context is not None else None
        # XLM-R layout is <s>A</s> or <s>A</s></s>B</s>. Constructing this
        # explicitly supports both Transformers 4 and 5 tokenizer backends.
        if special != (4 if paired else 2) or tokenizer.bos_token_id is None or tokenizer.eos_token_id is None:
            raise ValueError("The fusion encoder requires the XLM-R special-token layout")
        ids = [tokenizer.bos_token_id] + target + [tokenizer.eos_token_id]
        if context is not None:
            ids += [tokenizer.eos_token_id] + context + [tokenizer.eos_token_id]
        rows.append({"input_ids": ids, "attention_mask": [1] * len(ids)})
    return tokenizer.pad(rows, padding=True, return_tensors="pt"), lengths


class EmotionHead(nn.Module):
    def __init__(self, hidden_size, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, len(EMOTION_LABELS))

    def forward(self, vector):
        return self.out_proj(self.dropout(torch.tanh(self.dense(self.dropout(vector)))))


class TextEmojiFusionModel(nn.Module):
    """Learn a residual vector gate over current-message emoji representations."""
    def __init__(self, config, encoder=None):
        super().__init__()
        from transformers import AutoModel
        self.config = config
        self.encoder = encoder if encoder is not None else AutoModel.from_config(config, trust_remote_code=False)
        hidden = config.hidden_size
        buckets = getattr(config, "emoji_buckets", EMOJI_BUCKETS)
        self.emoji_embeddings = nn.Embedding(buckets, hidden, padding_idx=0)
        self.emoji_projection = nn.Linear(hidden, hidden, bias=False)
        self.gate = nn.Linear(hidden * 2, hidden)
        self.classifier = EmotionHead(hidden, getattr(config, "hidden_dropout_prob", 0.1))
        self.sarcasm_head = nn.Linear(hidden, 1)

    def forward(self, input_ids, attention_mask=None, emoji_ids=None, **kwargs):
        text = self.encoder(input_ids=input_ids, attention_mask=attention_mask, **kwargs).last_hidden_state[:, 0]
        if emoji_ids is None:
            emoji_ids = input_ids.new_zeros((input_ids.shape[0], 1))
        mask = emoji_ids.ne(0).unsqueeze(-1)
        embedded = self.emoji_embeddings(emoji_ids)
        vector = (embedded * mask).sum(1) / mask.sum(1).clamp_min(1)
        projected = self.emoji_projection(vector)
        gate = torch.sigmoid(self.gate(torch.cat((text, projected), -1)))
        # Force no-emoji contribution and gate to exactly zero, even after training.
        gate = gate * mask.any(1).to(gate.dtype)
        fused = text + gate * projected
        return SimpleNamespace(logits=self.classifier(fused), sarcasm_logits=self.sarcasm_head(fused).squeeze(-1),
                               emoji_gate=gate, fused_state=fused, text_state=text)

    @classmethod
    def from_base_pretrained(cls, base_model=BASE_MODEL, **kwargs):
        from transformers import AutoModel
        if base_model != BASE_MODEL:
            raise ValueError("The advanced architecture supports XLM-R base only")
        encoder = AutoModel.from_pretrained(base_model, trust_remote_code=False, use_safetensors=True, **kwargs)
        config = encoder.config
        config.emoji_buckets = EMOJI_BUCKETS
        config.id2label = dict(enumerate(EMOTION_LABELS))
        config.label2id = {label: index for index, label in enumerate(EMOTION_LABELS)}
        config.architectures = ["TextEmojiFusionModel"]
        config.fusion_architecture = ARCHITECTURE
        config.emoji_hash_version = "sha256_sequence_v1"
        config.max_emoji_sequences = MAX_EMOJI
        return cls(config, encoder)

    def save_pretrained(self, directory):
        from safetensors.torch import save_file
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.config.save_pretrained(directory)
        temporary = directory / "model.safetensors.tmp"
        save_file({name: value.detach().cpu().contiguous() for name, value in self.state_dict().items()}, str(temporary))
        temporary.replace(directory / "model.safetensors")

    @classmethod
    def from_pretrained(cls, directory):
        """Never downloads code, configuration, tokenizer or weights."""
        from transformers import AutoConfig
        from safetensors.torch import load_file
        directory = Path(directory)
        config = AutoConfig.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
        if config.model_type != "xlm-roberta" or getattr(config, "fusion_architecture", None) != ARCHITECTURE:
            raise ValueError("Unsupported fusion model configuration")
        if getattr(config, "emoji_hash_version", None) != "sha256_sequence_v1":
            raise ValueError("Unsupported emoji hash version")
        if getattr(config, "emoji_buckets", None) != EMOJI_BUCKETS or getattr(config, "max_emoji_sequences", None) != MAX_EMOJI:
            raise ValueError("Unsupported emoji vocabulary settings")
        model = cls(config)
        model.load_state_dict(load_file(str(directory / "model.safetensors")), strict=True)
        return model


def multitask_loss(output, emotion_targets, sarcasm_targets=None, sarcasm_weight=0.2, class_weights=None):
    """Missing independent sarcasm annotations use -100 and have zero head gradient."""
    emotion = nn.functional.cross_entropy(output.logits.float(), emotion_targets, weight=class_weights)
    auxiliary = output.sarcasm_logits.sum() * 0.0
    if sarcasm_targets is not None:
        mask = sarcasm_targets.ne(-100)
        if mask.any():
            auxiliary = nn.functional.binary_cross_entropy_with_logits(
                output.sarcasm_logits[mask].float(), sarcasm_targets[mask].float())
    return emotion + sarcasm_weight * auxiliary


class LocalFusionClassifier:
    def __init__(self, model_path, device=None):
        from transformers import AutoTokenizer
        from .checkpoint_metadata import emotion_checkpoint_metadata
        self.model_path = str(Path(model_path).resolve())
        self.metadata = validate_fusion_checkpoint(self.model_path)
        if (self.metadata.get("architecture") != ARCHITECTURE or self.metadata.get("base_model") != BASE_MODEL
                or self.metadata.get("training_status") != "complete"):
            raise ValueError("Use a completed learned text-emoji fusion checkpoint")
        for field in ("trained_sarcasm_head", "trained_with_context"):
            if type(self.metadata.get(field)) is not bool:
                raise ValueError(f"Fusion checkpoint must declare {field}")
        if self.metadata["trained_sarcasm_head"] and not self.metadata.get("sarcasm_supervision", {}).get("human_reviewed"):
            raise ValueError("The auxiliary sarcasm head requires reviewed supervision provenance")
        self.max_length = self.metadata["max_length"]
        self.required_labels = set(EMOTION_LABELS)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True, trust_remote_code=False)
        self.model = TextEmojiFusionModel.from_pretrained(self.model_path).to(self.device).eval()
        self.labels = [self.model.config.id2label[index].lower() for index in range(7)]
        if set(self.labels) != self.required_labels:
            raise ValueError("Fusion checkpoint requires the seven emotion labels")

    def _infer(self, texts, contexts=None):
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Provide non-empty phrases")
        if contexts is not None and len(contexts) != len(texts):
            raise ValueError("Provide one context for each phrase")
        mode, version = self.metadata["text_mode"], self.metadata["preparation_version"]
        prepared = [prepare_text(text, mode, version) for text in texts]
        context_values = [prepare_text(context or "", mode, version) for context in contexts] if contexts is not None else None
        ignored_context = context_values is not None and any(context_values) and not self.metadata["trained_with_context"]
        if ignored_context:
            context_values = None
        if self.metadata["trained_with_context"] and context_values is None:
            context_values = [""] * len(texts)
        inputs, lengths = encode_fusion_inputs(self.tokenizer, prepared, context_values, self.max_length)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        inputs["emoji_ids"] = emoji_batch(texts, self.device)
        with torch.inference_mode():
            output = self.model(**inputs)
            probabilities = output.logits.softmax(-1).cpu().tolist()
            sarcasm = output.sarcasm_logits.sigmoid().cpu().tolist() if self.metadata["trained_sarcasm_head"] else None
        from .emotions import EmotionResult
        results = []
        for values, length in zip(probabilities, lengths):
            scores = dict(zip(self.labels, map(float, values)))
            warnings = []
            synthetic = self.metadata.get("evaluation_scope") in {"synthetic_only", "mixed_test"}
            if synthetic:
                warnings.append("Synthetic or mixed synthetic test data: real-world human-labelled accuracy is not established.")
            if ignored_context:
                warnings.append("Context was ignored: this fusion checkpoint was not trained with context.")
            if length > self.max_length:
                warnings.append(f"Input was truncated to {self.max_length} tokens.")
            results.append(EmotionResult(max(scores, key=scores.get), scores, source="trained_checkpoint",
                                        model_name="XLM-R learned text-emoji fusion", experimental=synthetic,
                                        warnings=tuple(warnings), truncated=length > self.max_length))
        return results, sarcasm

    def predict_text(self, text):
        return self._infer([text])[0][0]

    def predict_with_context(self, text, context):
        return self._infer([text], [context])[0][0]

    def predict(self, message):
        return self.predict_text(message.raw_text)

    def predict_batch(self, texts, batch_size=8):
        if batch_size < 1:
            raise ValueError("Batch size must be positive")
        return [result for start in range(0, len(texts), batch_size)
                for result in self._infer(texts[start:start + batch_size])[0]]

    def predict_sarcasm(self, text, context=None, language_pair=None):
        if not self.metadata["trained_sarcasm_head"]:
            raise ValueError("This checkpoint has no supervised sarcasm head")
        scope = self.metadata.get("sarcasm_supervision", {})
        if language_pair is not None and language_pair not in scope.get("language_pairs", []):
            raise ValueError("No reviewed auxiliary sarcasm training for this language pair")
        _, scores = self._infer([text], [context] if context else None)
        verified = language_pair is not None and language_pair in scope.get("validated_language_pairs", [])
        return {"sarcasm_probability": float(scores[0]), "source": "trained_auxiliary_head",
                "experimental": not verified,
                "warnings": [] if verified else ["Auxiliary sarcasm has no verified binary test coverage for the supplied language scope."]}

def validate_fusion_checkpoint(directory):
    """Inspect deployment identity and tensor shapes without allocating XLM-R."""
    from safetensors import safe_open
    from .checkpoint_metadata import emotion_checkpoint_metadata
    directory = Path(directory)
    metadata = emotion_checkpoint_metadata(directory)
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    if (metadata.get("architecture") != ARCHITECTURE or metadata.get("base_model") != BASE_MODEL
            or metadata.get("training_status") != "complete" or config.get("model_type") != "xlm-roberta"
            or config.get("fusion_architecture") != ARCHITECTURE or config.get("hidden_size") != 768
            or config.get("emoji_hash_version") != "sha256_sequence_v1"
            or config.get("emoji_buckets") != EMOJI_BUCKETS or config.get("max_emoji_sequences") != MAX_EMOJI):
        raise ValueError("Unsupported or incomplete XLM-R fusion checkpoint")
    if type(metadata.get("trained_sarcasm_head")) is not bool or type(metadata.get("trained_with_context")) is not bool:
        raise ValueError("Fusion supervision flags must be booleans")
    if metadata["trained_sarcasm_head"] and metadata.get("sarcasm_supervision", {}).get("human_reviewed") is not True:
        raise ValueError("Reviewed sarcasm supervision provenance is required")
    if metadata["max_length"] > 512 or metadata["max_length"] < 5:
        raise ValueError("Invalid fusion token limit")
    if metadata.get("context_input_policy") != CONTEXT_INPUT_POLICY:
        raise ValueError("Unsupported fusion context token policy")
    expected = {"classifier.out_proj.weight": (7, 768), "classifier.out_proj.bias": (7,),
                "emoji_embeddings.weight": (EMOJI_BUCKETS, 768), "gate.weight": (768, 1536),
                "emoji_projection.weight": (768, 768), "sarcasm_head.weight": (1, 768)}
    with safe_open(str(directory / "model.safetensors"), framework="pt", device="cpu") as weights:
        for name, shape in expected.items():
            if name not in weights.keys() or tuple(weights.get_slice(name).get_shape()) != shape:
                raise ValueError(f"Invalid fusion tensor: {name}")
        if "encoder.embeddings.word_embeddings.weight" not in weights.keys():
            raise ValueError("Fusion checkpoint has no XLM-R encoder weights")
    return metadata
