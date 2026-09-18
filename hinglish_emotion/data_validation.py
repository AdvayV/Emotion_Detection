from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path


VALID_LABELS = {"positive", "neutral", "negative"}
OPTIONAL_COLUMNS = {
    "sarcasm",
    "valence",
    "arousal",
    "conversation_id",
    "speaker_id",
    "turn_id",
    "evidence",
    "pair_id",
}


@dataclass(frozen=True)
class DatasetReport:
    rows: int
    label_counts: dict[str, int]
    empty_text_rows: int
    invalid_label_rows: int
    duplicate_text_rows: int
    optional_columns: tuple[str, ...]
    conversation_ready: bool

    @property
    def valid(self) -> bool:
        return self.rows > 0 and self.empty_text_rows == 0 and self.invalid_label_rows == 0

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["valid"] = self.valid
        return value


def validate_csv(path: str | Path) -> DatasetReport:
    path = Path(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing = {"text", "label"} - fieldnames
        if missing:
            raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing))}")
        rows = list(reader)

    label_counts = {label: 0 for label in sorted(VALID_LABELS)}
    empty_text_rows = 0
    invalid_label_rows = 0
    seen: set[str] = set()
    duplicates = 0
    for row in rows:
        text = (row.get("text") or "").strip()
        label = (row.get("label") or "").strip().lower()
        if not text:
            empty_text_rows += 1
        if label not in VALID_LABELS:
            invalid_label_rows += 1
        else:
            label_counts[label] += 1
        normalized_key = text.casefold()
        if normalized_key in seen:
            duplicates += 1
        seen.add(normalized_key)

    optional = tuple(sorted(fieldnames & OPTIONAL_COLUMNS))
    conversation_ready = {"conversation_id", "speaker_id", "turn_id"}.issubset(fieldnames)
    return DatasetReport(
        rows=len(rows),
        label_counts=label_counts,
        empty_text_rows=empty_text_rows,
        invalid_label_rows=invalid_label_rows,
        duplicate_text_rows=duplicates,
        optional_columns=optional,
        conversation_ready=conversation_ready,
    )

