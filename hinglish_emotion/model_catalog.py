"""Named encoder choices and their local sentiment checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EncoderSpec:
    key: str
    name: str
    model_id: str
    folder: str


ENCODERS = (
    EncoderSpec("mbert", "mBERT", "google-bert/bert-base-multilingual-cased", "mbert_sentiment"),
    EncoderSpec("muril", "MuRIL", "google/muril-base-cased", "muril_sentiment"),
    EncoderSpec("xlmr", "XLM-R", "FacebookAI/xlm-roberta-base", "xlmr_sentiment"),
)
ENCODER_BY_KEY = {encoder.key: encoder for encoder in ENCODERS}


def checkpoint_path(encoder: EncoderSpec, root: str | Path = "models") -> Path:
    return Path(root) / encoder.folder


def checkpoint_ready(path: str | Path) -> bool:
    """A folder created by a failed training run is not a usable checkpoint."""
    path = Path(path)
    weights = ("model.safetensors", "pytorch_model.bin", "model.safetensors.index.json", "pytorch_model.bin.index.json")
    tokenizer = ("tokenizer.json", "vocab.txt", "sentencepiece.bpe.model", "spiece.model")
    return (path / "config.json").is_file() and any((path / name).is_file() for name in weights) and any((path / name).is_file() for name in tokenizer)
