"""Matched-data comparison of local seven-emotion checkpoints."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import gc
import hashlib
from itertools import combinations
import json
from pathlib import Path
from time import perf_counter

from .batch import normalize_records, predict_dataset
from .disagreement import probability_summary
from .emotion_data import evaluation_rows
from .emotions import EmotionDetector
from .evaluation import classification_metrics
from .tasks import EMOTION_LABELS, LANGUAGE_PAIRS


def comparison_rows(records, pair, test_only=True):
    """Select one language and partition before every model sees identical rows."""
    if pair not in LANGUAGE_PAIRS:
        raise ValueError("Choose hi-en or fr-en")
    normalized = normalize_records(list(records), default_language=pair)
    matched = [row for row in normalized if row["language_pair"] == pair]
    if not matched:
        raise ValueError("No phrases match the selected language mix")
    selected, scope = evaluation_rows(matched) if test_only else (matched, "all supplied rows, including any training/validation rows")
    if len(selected) > 10000:
        raise ValueError("Compare at most 10,000 selected rows; use a smaller file or its test split")
    if any(len(row["text"]) > 20000 for row in selected):
        raise ValueError("Each phrase must contain at most 20,000 characters")
    return selected, scope, len(normalized) - len(matched)


def _metrics(predictions):
    if not predictions:
        return None
    truth = [row["actual"] for row in predictions]
    return {
        **classification_metrics(truth, [row["predicted"] for row in predictions], EMOTION_LABELS),
        "rows": len(predictions),
        "correct": sum(row["correct"] is True for row in predictions),
        "missing_emotions": [label for label in EMOTION_LABELS if label not in truth],
    }


def compare_models(records, pair, checkpoints, *, classifier_factory,
                   test_only=True, exclude_overlap=True, batch_size=4,
                   clear_shared_cache=None, progress=None):
    """Load one model at a time, then score all on the same labelled rows.

    Unknown labels remain unscored. If any selected model flags training overlap,
    that row is excluded from *every* model's score when exclude_overlap is true.
    Predictions for excluded rows are retained for inspection. A selected model
    failure aborts the run; it is never replaced by demo rules or a zero score.
    """
    selected, scope, skipped = comparison_rows(records, pair, test_only)
    checkpoints = [(str(name), Path(path), str(version)) for name, path, version in checkpoints]
    if len(checkpoints) < 2 or len({m[0] for m in checkpoints}) != len(checkpoints):
        raise ValueError("Choose at least two distinct named checkpoints")
    if len({str(path.resolve()).casefold() for _, path, _ in checkpoints}) != len(checkpoints):
        raise ValueError("The same checkpoint cannot be entered twice")
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("Use a positive batch size")
    if pair != "hi-en" and any(name.casefold() == "muril" for name, _, _ in checkpoints):
        raise ValueError("MuRIL comparison is available for Hindi-English only")
    if clear_shared_cache:
        clear_shared_cache()
    collected = []
    total = len(checkpoints) * len(selected)
    for model_index, (name, path, version) in enumerate(checkpoints):
        classifier = engine = None
        try:
            if progress:
                progress(model_index * len(selected), total, f"Loading {name}")
            start = perf_counter()
            classifier = classifier_factory(path)
            load_seconds = perf_counter() - start
            metadata = dict(getattr(classifier, "metadata", {}))
            if pair not in metadata.get("trained_language_pairs", []):
                raise ValueError(f"{name} is not trained for {pair}")
            engine = EmotionDetector(classifier)
            start = perf_counter()
            report = predict_dataset(
                selected, engine, test_only=False, batch_size=batch_size,
                progress=(lambda done, count: progress(model_index * len(selected) + done, total, f"Predicting with {name}")) if progress else None,
            )
            prediction_seconds = perf_counter() - start
            for original, prediction in zip(selected, report["predictions"]):
                if prediction["source"] != "trained_checkpoint":
                    raise ValueError(f"{name} returned a fallback instead of trained predictions")
                if prediction["text"] != original["text"] or prediction["actual"] != original["emotion"]:
                    raise ValueError(f"{name} returned misaligned predictions")
                probabilities = {label: prediction[f"score_{label}"] for label in EMOTION_LABELS}
                probability_summary(probabilities)
                if prediction["predicted"] not in probabilities or probabilities[prediction["predicted"]] < max(probabilities.values()):
                    raise ValueError(f"{name} returned an inconsistent predicted emotion")
            collected.append({
                "name": name, "checkpoint": str(path), "version": version,
                "base_model": metadata.get("base_model", "unknown"),
                "trained_language_pairs": metadata.get("trained_language_pairs", []),
                "training_data_kind": metadata.get("training_data_kind", "unknown"),
                "evaluation_scope": metadata.get("evaluation_scope", "unknown"),
                "text_mode": metadata.get("text_mode", "unknown"),
                "load_seconds": load_seconds, "prediction_seconds": prediction_seconds,
                "milliseconds_per_phrase": 1000 * prediction_seconds / len(selected),
                "overlap_check_available": all(report["overlap_check_available"].values()),
                "training_overlap_rows": report["training_overlap_rows"],
                "truncated_rows": sum(row["truncated"] for row in report["predictions"]),
                "predictions": report["predictions"],
            })
        except (ValueError, RuntimeError, OSError, TypeError, ImportError) as exc:
            raise RuntimeError(f"{name} comparison failed: {exc}") from exc
        finally:
            engine = classifier = None
            gc.collect()

    overlaps = {i for model in collected for i, row in enumerate(model["predictions"]) if row["training_overlap"] is True}
    scored = {i for i, row in enumerate(selected) if row["emotion"] and not (exclude_overlap and i in overlaps)}
    models = []
    for model in collected:
        metrics = _metrics([row for i, row in enumerate(model["predictions"]) if i in scored])
        models.append({**{key: value for key, value in model.items() if key != "predictions"}, "metrics": metrics})
    wide = []
    for i, row in enumerate(selected):
        labels = [model["predictions"][i]["predicted"] for model in collected]
        item = {
            "row_number": i + 1, "row_id": row["row_id"], "text": row["text"],
            "language_pair": pair, "expected_emotion": row["emotion"],
            "dataset_source": row.get("dataset_source") or row.get("source", ""),
            "annotation_status": row.get("annotation_status", ""),
            "included_in_scoring": i in scored, "training_overlap_any_model": i in overlaps,
            "models_disagree": len(set(labels)) > 1,
        }
        for model in collected:
            prediction = model["predictions"][i]
            name = model["name"]
            for field in ("predicted", "confidence", "correct", "source", "model", "training_overlap", "truncated", "warnings"):
                item[f"{name} {field}"] = prediction[field]
            item.update({f"{name} score_{label}": prediction[f"score_{label}"] for label in EMOTION_LABELS})
        wide.append(item)
    paired = []
    for left, right in combinations(collected, 2):
        counts = Counter((left["predictions"][i]["correct"], right["predictions"][i]["correct"]) for i in scored)
        paired.append({
            "left": left["name"], "right": right["name"], "scored_rows": len(scored),
            "both_correct": counts[(True, True)], "left_only_correct": counts[(True, False)],
            "right_only_correct": counts[(False, True)], "both_wrong": counts[(False, False)],
            "agreement_all_selected_rows": sum(a["predicted"] == b["predicted"] for a, b in zip(left["predictions"], right["predictions"])) / len(selected),
        })
    canonical = json.dumps(selected, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return {
        "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "language_pair": pair, "scope": scope, "test_only": test_only,
        "dataset_sha256": hashlib.sha256(canonical).hexdigest(),
        "selected_rows": len(selected), "other_language_rows_skipped": skipped,
        "labelled_rows": sum(bool(row["emotion"]) for row in selected), "scored_rows": len(scored),
        "label_counts": dict(Counter(selected[i]["emotion"] for i in sorted(scored))),
        "exclude_overlap": exclude_overlap, "training_overlap_rows": len(overlaps),
        "excluded_overlap_rows": len(overlaps) if exclude_overlap else 0,
        "synthetic_dataset": any("synthetic" in str(row.get("dataset_source") or row.get("source", "")).lower() or row.get("annotation_status") == "synthetic" for row in selected),
        "illustrative_labels": any((row.get("dataset_source") or row.get("source")) in {"illustrative", "assistant_draft"} or row.get("annotation_status") == "assistant_draft" for row in selected),
        "all_labels_declared_human_reviewed": bool(scored) and all(str(selected[i].get("human_reviewed", "")).lower() in {"true", "yes", "1"} for i in scored),
        "models": models, "paired_comparisons": paired, "predictions": wide,
        "scoring_note": "Accuracy and seven-class macro-F1 use identical labelled eligible rows for every selected model. Confidence and agreement are not accuracy. Scores are conditional on supplied label quality and dataset scope.",
    }
