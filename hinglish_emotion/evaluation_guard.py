"""Leakage checks using phrase hashes without exporting training text."""
import csv
import hashlib
import json
from pathlib import Path
from .text_preparation import duplicate_key


def phrase_hash(text):
    return hashlib.sha256(duplicate_key(text).encode("utf-8")).hexdigest()


def write_guard(path, rows, metadata):
    seen = sorted({phrase_hash(row["text"]) for row in rows if row.get("split") in {"train", "validation"}})
    guard = {"schema_version": 1, "dataset_sha256": metadata["dataset_sha256"], "split_sha256": metadata["split_sha256"], "training_validation_hashes": seen}
    Path(path).write_text(json.dumps(guard, indent=2), encoding="utf-8")
    return guard


def training_hashes(detector):
    classifier = getattr(detector, "classifier", None)
    model_path = getattr(classifier, "model_path", None)
    if not model_path:
        return None
    path = Path(model_path)
    guard_file = path / "evaluation_guard.json"
    if guard_file.is_file():
        guard = json.loads(guard_file.read_text(encoding="utf-8"))
        metadata = getattr(classifier, "metadata", {})
        if guard.get("schema_version") != 1 or any(guard.get(key) != metadata.get(key) for key in ("dataset_sha256", "split_sha256")):
            raise ValueError("Checkpoint evaluation guard does not match its metadata")
        return set(guard["training_validation_hashes"])
    manifest = path / "split_manifest.csv"
    if manifest.is_file():
        with manifest.open(encoding="utf-8-sig", newline="") as handle:
            return {phrase_hash(row["text"]) for row in csv.DictReader(handle) if row.get("split") in {"train", "validation"}}
    return None
