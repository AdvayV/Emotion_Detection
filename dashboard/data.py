from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import BinaryIO

import pandas as pd

from hinglish_emotion.data_validation import DatasetReport, OPTIONAL_COLUMNS, VALID_LABELS


def read_dataset(source: str | Path | BinaryIO) -> pd.DataFrame:
    """Read either supported interchange format using the same schema."""
    name = getattr(source, "name", source)
    suffix = Path(str(name)).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(source)
    if suffix == ".xlsx":
        return pd.read_excel(source, sheet_name="phrases")
    raise ValueError("Unsupported dataset format. Use a .csv or .xlsx file.")


def dataframe_report(frame: pd.DataFrame) -> DatasetReport:
    """Apply the project's tabular-data rules to an in-memory dataset."""
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
