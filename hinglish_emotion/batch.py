"""File prediction and labelled evaluation shared by the dashboard and CLI."""
from collections import Counter
import csv
import io
from pathlib import Path
from .tasks import EMOTION_LABELS, LANGUAGE_PAIRS, canonical_emotion
from .evaluation import classification_metrics
from .emotion_data import evaluation_rows
from .evaluation_guard import training_hashes, phrase_hash

ALIASES = {"en-hi": "hi-en", "hindi-english": "hi-en", "english-hindi": "hi-en", "hinglish": "hi-en", "en-fr": "fr-en", "french-english": "fr-en", "english-french": "fr-en", "frenglish": "fr-en"}


def language_pair(value):
    value = str(value).strip().lower().replace("_", "-").replace("–", "-").replace(" ", "")
    value = ALIASES.get(value, value)
    if value not in LANGUAGE_PAIRS:
        raise ValueError("Use hi-en (Hindi–English) or fr-en / en-fr (English–French)")
    return value


def read_frame(content: bytes, filename: str):
    import pandas as pd
    if len(content) > 25 * 1024 * 1024:
        raise ValueError("Upload a file smaller than 25 MB")
    if Path(filename).suffix.lower() == ".xlsx":
        frame = pd.read_excel(io.BytesIO(content), dtype=str, keep_default_na=False)
    elif Path(filename).suffix.lower() == ".csv":
        decoded = content.decode("utf-8-sig")
        header = next(csv.reader(io.StringIO(decoded)), [])
        if len({name.strip() for name in header}) != len(header):
            raise ValueError("The CSV has duplicate column names")
        frame = pd.read_csv(io.StringIO(decoded), dtype=str, keep_default_na=False)
    else:
        raise ValueError("Use a UTF-8 CSV or XLSX file")
    frame.columns = [str(column).strip() for column in frame.columns]
    if frame.empty or frame.columns.duplicated().any():
        raise ValueError("The dataset is empty or has duplicate column names")
    if len(frame) > 50000:
        raise ValueError("Use at most 50,000 rows per upload")
    return frame.fillna("")


def normalize_records(records, default_language="hi-en", text_column=None, emotion_column=None, language_column=None):
    if not records:
        raise ValueError("The dataset is empty")
    columns = list(records[0])
    def choose(explicit, candidates, required=False):
        if explicit is not None:
            if explicit == "": return None
            if explicit not in columns: raise ValueError(f"Missing column: {explicit}")
            return explicit
        selected = next((column for name in candidates for column in columns if column.strip().lower() == name), None)
        if required and not selected: raise ValueError("Choose the column containing phrases (text, message or phrase)")
        return selected
    text_column = choose(text_column, ("text", "message", "phrase", "sentence"), True)
    emotion_column = choose(emotion_column, ("emotion", "label", "expected_emotion", "actual"))
    language_column = choose(language_column, ("language_pair", "language", "language_mix"))
    default_language = language_pair(default_language)
    rows = []
    for index, original in enumerate(records, 2):
        text = str(original.get(text_column) or "").strip()
        if not text: raise ValueError(f"Row {index}: phrase text is empty")
        raw_label = str(original.get(emotion_column) or "").strip() if emotion_column else ""
        emotion = canonical_emotion(raw_label) if raw_label else ""
        if emotion and emotion not in EMOTION_LABELS:
            raise ValueError(f"Row {index}: {raw_label!r} is not a seven-emotion label. Use {', '.join(EMOTION_LABELS)}")
        pair = language_pair(original.get(language_column) or default_language) if language_column else default_language
        rows.append({**original, "row_id": str(original.get("row_id") or f"upload_{index-1:06d}"), "text": text, "emotion": emotion, "language_pair": pair})
    return rows


def predict_dataset(rows, detector, test_only=True, batch_size=8, progress=None):
    rows, scope = evaluation_rows(rows) if test_only else (rows, "all supplied rows including any training/validation rows")
    if len(rows) > 10000:
        raise ValueError("Predict at most 10,000 selected rows in one run; use the declared test split or a smaller file")
    results = [None] * len(rows)
    completed = 0
    checks = {}
    for pair in LANGUAGE_PAIRS:
        indices = [i for i, row in enumerate(rows) if row["language_pair"] == pair]
        if not indices: continue
        engine = detector(pair) if callable(detector) else detector
        known_hashes = training_hashes(engine)
        checks[pair] = known_hashes is not None
        for offset in range(0, len(indices), batch_size):
            chunk = indices[offset:offset+batch_size]
            guesses = engine.predict_many([rows[i]["text"] for i in chunk], [pair]*len(chunk), batch_size)
            if len(guesses) != len(chunk): raise RuntimeError("Classifier returned an incomplete batch")
            for index, guess in zip(chunk, guesses):
                row = rows[index]
                results[index] = {
                    "row_id": row["row_id"], "text": row["text"], "language_pair": pair,
                    "split": row.get("split", ""),
                    "dataset_source": row.get("dataset_source") or row.get("source", ""),
                    "annotation_status": row.get("annotation_status", ""),
                    "actual": row.get("emotion", ""), "predicted": guess.emotion,
                    "confidence": guess.confidence,
                    "correct": guess.emotion == row["emotion"] if row.get("emotion") else None,
                    "source": guess.source, "model": guess.model_name,
                    "needs_context": guess.needs_context, "experimental": guess.experimental,
                    "truncated": guess.truncated,
                    "warnings": " | ".join(guess.warnings),
                    "training_overlap": phrase_hash(row["text"]) in known_hashes if known_hashes is not None else None,
                    **{f"score_{label}": guess.probabilities[label] for label in EMOTION_LABELS},
                }
            completed += len(chunk)
            if progress: progress(completed, len(rows))
    labelled = [row for row in results if row["actual"]]
    def metrics(subset):
        if not subset: return None
        return {**classification_metrics([row["actual"] for row in subset], [row["predicted"] for row in subset], EMOTION_LABELS), "rows": len(subset), "missing_emotions": [label for label in EMOTION_LABELS if label not in {row["actual"] for row in subset}]}
    synthetic = any("synthetic" in str(row.get("dataset_source") or row.get("source", "")).lower() or str(row.get("annotation_status", "")).lower() == "synthetic" for row in rows)
    return {
        "scope": scope, "total_rows": len(rows), "labelled_rows": len(labelled),
        "unlabelled_rows": len(rows)-len(labelled), "overall": metrics(labelled),
        "by_language": {pair: metrics([row for row in labelled if row["language_pair"] == pair]) for pair in LANGUAGE_PAIRS if any(row["language_pair"] == pair for row in results)},
        "predicted_counts": dict(Counter(row["predicted"] for row in results)),
        "models_used": [{"language_pair": pair, "source": source, "model": model, "rows": count} for (pair,source,model),count in Counter((row["language_pair"],row["source"],row["model"]) for row in results).items()],
        "training_overlap_rows": sum(row["training_overlap"] is True for row in results),
        "overlap_check_available": checks, "synthetic_dataset": synthetic,
        "illustrative": any((row.get("dataset_source") or row.get("source")) in {"illustrative", "assistant_draft"} for row in rows),
        "experimental_language_pairs": [pair for pair in LANGUAGE_PAIRS if any(row["language_pair"] == pair and row["experimental"] for row in results)],
        "predictions": results,
    }
