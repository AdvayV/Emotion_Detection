from __future__ import annotations

from dataclasses import asdict

import pandas as pd

from hinglish_emotion.data_validation import DatasetReport, OPTIONAL_COLUMNS, VALID_LABELS


def dataframe_report(frame: pd.DataFrame) -> DatasetReport:
    """Apply the project's CSV rules to an in-memory uploaded dataset."""
    missing = {"text", "label"} - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing))}")

    texts = frame["text"].fillna("").astype(str).str.strip()
    labels = frame["label"].fillna("").astype(str).str.strip().str.lower()
    label_counts = {label: int((labels == label).sum()) for label in sorted(VALID_LABELS)}
    optional = tuple(sorted(set(frame.columns) & OPTIONAL_COLUMNS))
    return DatasetReport(
        rows=len(frame),
        label_counts=label_counts,
        empty_text_rows=int((texts == "").sum()),
        invalid_label_rows=int((~labels.isin(VALID_LABELS)).sum()),
        duplicate_text_rows=int(texts.duplicated().sum()),
        optional_columns=optional,
        conversation_ready={"conversation_id", "speaker_id", "turn_id"}.issubset(frame.columns),
    )


def report_dict(report: DatasetReport) -> dict[str, object]:
    result = asdict(report)
    result["valid"] = report.valid
    return result


def sarcasm_rate(frame: pd.DataFrame) -> float | None:
    if "sarcasm" not in frame.columns or frame.empty:
        return None
    values = frame["sarcasm"].fillna(False).astype(str).str.strip().str.lower()
    return float(values.isin({"true", "1", "yes"}).mean())
