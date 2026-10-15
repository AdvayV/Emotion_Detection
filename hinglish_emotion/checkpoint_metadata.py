"""Inspect checkpoint identity and preprocessing before loading its weights."""

from __future__ import annotations

import json
from pathlib import Path

from .tasks import EMOTION_LABELS, LANGUAGE_PAIRS
from .text_preparation import PREPARATION_VERSION, TEXT_MODES


def emotion_checkpoint_metadata(path: str | Path) -> dict:
    path = Path(path)
    try:
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("A readable config.json is required for an emotion checkpoint") from exc
    labels = list(config.get("id2label", {}).values())
    if len(labels) != 7 or {str(label).lower() for label in labels} != set(EMOTION_LABELS):
        raise ValueError("This is not a seven-emotion checkpoint; sentiment models cannot be used here")
    metadata = config.get("emotion_metadata")
    if metadata is None:
        # Preserve old checkpoints' actual lowercasing/normalization behaviour.
        return {
            "schema_version": 0, "task": "emotion", "labels": list(EMOTION_LABELS),
            "text_mode": "normalized", "preparation_version": 1, "max_length": 128,
            "trained_language_pairs": [], "validated_language_pairs": [],
            "training_status": "legacy_unknown", "legacy": True,
        }
    if not isinstance(metadata, dict) or metadata.get("schema_version") != 1:
        raise ValueError("Unsupported emotion checkpoint metadata")
    if metadata.get("task") != "emotion" or set(metadata.get("labels", [])) != set(EMOTION_LABELS):
        raise ValueError("Checkpoint task or emotion labels are inconsistent")
    if metadata.get("text_mode") not in TEXT_MODES or metadata.get("preparation_version") != PREPARATION_VERSION:
        raise ValueError("Unsupported checkpoint text preparation settings")
    if not isinstance(metadata.get("max_length"), int) or metadata["max_length"] < 1:
        raise ValueError("Invalid checkpoint token limit")
    for field in ("trained_language_pairs", "validated_language_pairs"):
        if not set(metadata.get(field, [])) <= set(LANGUAGE_PAIRS):
            raise ValueError(f"Unknown language mix in checkpoint {field}")
    return {**metadata, "legacy": False}
