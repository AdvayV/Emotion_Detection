"""Validate and evaluate mixed-language, seven-emotion datasets."""

from __future__ import annotations

from collections import Counter
import csv
from pathlib import Path

from .evaluation import classification_metrics
from .tasks import EMOTION_LABELS, LANGUAGE_PAIRS, canonical_emotion
from .text_preparation import duplicate_key


def validate_emotion_rows(rows: list[dict]) -> list[dict]:
    if not rows:
        raise ValueError("The emotion dataset is empty")
    required = {"text", "emotion", "language_pair"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError("Missing emotion columns: " + ", ".join(sorted(missing)))
    cleaned = []
    for index, original in enumerate(rows, start=2):
        row = {**original}
        row["text"] = str(row.get("text") or "").strip()
        row["emotion"] = canonical_emotion(str(row.get("emotion") or ""))
        row["language_pair"] = str(row.get("language_pair") or "").strip().lower()
        if not row["text"]:
            raise ValueError(f"Row {index}: message text is empty")
        if row["emotion"] not in EMOTION_LABELS:
            raise ValueError(f"Row {index}: unknown emotion {row['emotion']!r}; use {', '.join(EMOTION_LABELS)}")
        if row["language_pair"] not in LANGUAGE_PAIRS:
            raise ValueError(f"Row {index}: language_pair must be hi-en or fr-en")
        cleaned.append(row)
    return cleaned


def read_emotion_csv(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return validate_emotion_rows(list(csv.DictReader(handle)))


def evaluation_rows(rows: list[dict]) -> tuple[list[dict], str]:
    """Use only test rows when a file includes a declared train/validation/test split."""
    splits = {str(row.get("split") or "").strip().lower() for row in rows}
    if splits <= {"", "demo"}:
        return rows, "demo" if "demo" in splits else "provided evaluation file"
    if not splits <= {"train", "validation", "test"}:
        raise ValueError("split must be train, validation or test for every row, or demo for an illustrative file")
    test_rows = [row for row in rows if str(row["split"]).strip().lower() == "test"]
    if not test_rows:
        raise ValueError("This split dataset contains no test rows")
    return test_rows, "declared test split"


def evaluate_emotions(rows: list[dict], detector) -> dict[str, object]:
    rows = validate_emotion_rows(rows)
    rows, scope = evaluation_rows(rows)
    records = []
    for row in rows:
        result = detector.predict(row["text"], row["language_pair"])
        records.append({
            "text": row["text"], "language_pair": row["language_pair"],
            "actual": row["emotion"], "predicted": result.emotion,
            "confidence": result.confidence, "correct": result.emotion == row["emotion"],
            "needs_context": result.needs_context,
            "experimental": result.experimental, "source": result.source,
            "truncated": result.truncated,
        })
    def metrics(records):
        labels = [row["actual"] for row in records]
        return {
            **classification_metrics(labels, [row["predicted"] for row in records], EMOTION_LABELS),
            "rows": len(records),
            "missing_emotions": [label for label in EMOTION_LABELS if label not in labels],
        }
    overlap = 0
    classifier = getattr(detector, "classifier", None)
    model_path = getattr(classifier, "model_path", None)
    if model_path and (Path(model_path) / "split_manifest.csv").is_file():
        with (Path(model_path) / "split_manifest.csv").open(encoding="utf-8-sig", newline="") as handle:
            seen = {duplicate_key(row["text"]) for row in csv.DictReader(handle) if row["split"] in {"train", "validation"}}
        overlap = sum(duplicate_key(row["text"]) in seen for row in rows)
    return {
        "synthetic_dataset": any("synthetic" in str(row.get("source", "")).lower() or str(row.get("annotation_status", "")).lower() == "synthetic" for row in rows),
        "training_overlap_rows": overlap,
        "experimental_language_pairs": [pair for pair in LANGUAGE_PAIRS if any(record["language_pair"] == pair and record["experimental"] for record in records)],
        "scope": scope, "illustrative": any(row.get("source") == "illustrative" for row in rows),
        "overall": metrics(records),
        "by_language": {pair: metrics([row for row in records if row["language_pair"] == pair])
                        for pair in LANGUAGE_PAIRS if any(row["language_pair"] == pair for row in records)},
        "label_counts": dict(Counter(row["actual"] for row in records)),
        "predictions": records,
    }
