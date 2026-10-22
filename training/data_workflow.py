"""Reproducible preparation and frozen splits, independent of model loading."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from hinglish_emotion.data_validation import validate_csv
from hinglish_emotion.emotion_data import read_emotion_csv
from hinglish_emotion.tasks import EMOTION_LABELS, SENTIMENT_LABELS
from hinglish_emotion.text_preparation import prepare_text, duplicate_key
from training.splits import split_data


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def records_hash(frame: pd.DataFrame) -> str:
    records = frame.fillna("").astype(str).to_dict("records")
    return hashlib.sha256(json.dumps(records, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def prepare_data(data, task, text_mode, split_seed=42, split_manifest=None, data_audit=None, language_pair="all", allow_synthetic=False):
    labels = EMOTION_LABELS if task == "emotion" else SENTIMENT_LABELS
    if task == "emotion":
        frame = pd.DataFrame(read_emotion_csv(data))
        sources = frame.get("source", pd.Series("", index=frame.index)).astype(str).str.strip().str.lower()
        if sources.isin({"illustrative", "assistant_draft"}).any():
            raise ValueError("Illustrative or unreviewed draft data cannot be used for research training")
        statuses = frame.get("annotation_status", pd.Series("", index=frame.index)).astype(str).str.lower()
        synthetic = sources.str.contains("synthetic", regex=False) | statuses.eq("synthetic")
        if synthetic.any() and not allow_synthetic:
            raise ValueError("Synthetic data requires --allow-synthetic for an explicitly experimental run")
        if synthetic.any() and data_audit:
            audit = json.loads(Path(data_audit).read_text(encoding="utf-8"))
            if audit.get("dataset_kind") != "synthetic_template" or audit.get("prepared_csv_sha256") != sha256_file(data):
                raise ValueError("Synthetic dataset audit does not match this CSV")
        if sources.str.contains("masac|ediref", regex=True).any():
            if not data_audit:
                raise ValueError("MaSaC training requires its approved dataset audit")
            audit = json.loads(Path(data_audit).read_text(encoding="utf-8"))
            if not audit.get("reuse_conditions_verified") or audit.get("prepared_csv_sha256") != sha256_file(data):
                raise ValueError("Dataset audit is unapproved or does not match this CSV")
        if language_pair != "all":
            frame = frame[frame["language_pair"] == language_pair].copy()
            if frame.empty:
                raise ValueError(f"No rows for {language_pair}")
        frame["label"] = frame["emotion"]
    else:
        report = validate_csv(data)
        if not report.valid:
            raise ValueError(f"Dataset validation failed: {report.to_dict()}")
        frame = pd.read_csv(data, dtype=str, keep_default_na=False)
        frame["label"] = frame["label"].str.strip().str.lower()
    frame = frame.reset_index(drop=True).fillna("")
    counts = {label: int(frame["label"].eq(label).sum()) for label in labels}
    if min(counts.values()) < 6:
        raise ValueError("Each class needs at least six examples and enough independent groups for every split")
    if "row_id" not in frame:
        frame["row_id"] = [f"row_{index:08d}" for index in range(len(frame))]
    if frame["row_id"].eq("").any() or frame["row_id"].duplicated().any():
        raise ValueError("row_id values must be non-empty and unique")
    dataset_hash = records_hash(frame)
    frame["normalized_text"] = frame["text"].map(lambda text: prepare_text(text, text_mode))
    frame["duplicate_key"] = frame["text"].map(duplicate_key)
    manifest_path = Path(split_manifest) if split_manifest else None
    if manifest_path and manifest_path.exists():
        stored = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
        if stored["row_id"].duplicated().any() or set(stored["row_id"]) != set(frame["row_id"]):
            raise ValueError("Frozen split manifest does not contain exactly these dataset rows")
        info = json.loads(manifest_path.with_suffix(".metadata.json").read_text(encoding="utf-8"))
        if info.get("dataset_sha256") != dataset_hash or info.get("manifest_sha256") != sha256_file(manifest_path):
            raise ValueError("Dataset or frozen split manifest has changed")
        if "split" in frame:
            existing = frame.set_index("row_id")["split"]
            frozen = stored.set_index("row_id")["split"]
            if any(str(existing[key]) != str(frozen[key]) for key in existing.index):
                raise ValueError("Declared split conflicts with the frozen manifest")
        frame["split"] = frame["row_id"].map(stored.set_index("row_id")["split"])
    parts = tuple(part.sort_values("row_id").reset_index(drop=True) for part in split_data(frame, split_seed))
    keep = [column for column in frame if column not in {"normalized_text", "duplicate_key", "split"}]
    manifest = pd.concat([part[keep].assign(split=name) for part, name in zip(parts, ("train", "validation", "test"))], ignore_index=True).sort_values(["split", "row_id"]).reset_index(drop=True)
    manifest_csv = manifest.to_csv(index=False, lineterminator="\n")
    manifest_hash = hashlib.sha256(manifest_csv.encode("utf-8")).hexdigest()
    if manifest_path and not manifest_path.exists():
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(manifest_csv, encoding="utf-8", newline="")
        manifest_path.with_suffix(".metadata.json").write_text(json.dumps({
            "dataset_sha256": dataset_hash, "manifest_sha256": sha256_file(manifest_path),
            "split_seed": split_seed, "task": task,
        }, indent=2), encoding="utf-8")
    verified_pairs = []
    test = parts[2]
    train_sources = parts[0].get("source", pd.Series("", index=parts[0].index)).astype(str)
    test_statuses = test.get("annotation_status", pd.Series("", index=test.index)).astype(str)
    test_sources = test.get("source", pd.Series("", index=test.index)).astype(str)
    synthetic_train = train_sources.str.contains("synthetic", case=False, regex=False) | parts[0].get("annotation_status", pd.Series("", index=parts[0].index)).astype(str).eq("synthetic")
    synthetic_test = test_sources.str.contains("synthetic", case=False, regex=False) | test_statuses.eq("synthetic")
    training_kind = "synthetic_only" if synthetic_train.all() else "mixed_with_synthetic" if synthetic_train.any() else "human_or_unverified"
    evaluation_scope = "synthetic_only" if synthetic_test.all() else "mixed_test" if synthetic_test.any() else "human_annotated" if test_statuses.isin({"public_annotation", "human_adjudicated", "human_reviewed"}).all() else "unverified"
    if task == "emotion":
        for pair in sorted(test["language_pair"].unique()):
            subset = test[test["language_pair"] == pair]
            statuses = subset.get("annotation_status", pd.Series("", index=subset.index))
            source_is_synthetic = subset.get("source", pd.Series("", index=subset.index)).astype(str).str.contains("synthetic", case=False, regex=False).any()
            if not source_is_synthetic and set(subset["label"]) == set(labels) and statuses.isin({"public_annotation", "human_adjudicated", "human_reviewed"}).all():
                verified_pairs.append(pair)
    return {
        "parts": parts, "manifest": manifest, "manifest_sha256": manifest_hash,
        "dataset_sha256": dataset_hash, "labels": labels,
        "report": {"rows": len(frame), "label_counts": counts, "valid": True},
        "trained_language_pairs": sorted(frame["language_pair"].unique()) if "language_pair" in frame else [],
        "validated_language_pairs": verified_pairs,
        "training_data_kind": training_kind, "evaluation_scope": evaluation_scope,
    }
