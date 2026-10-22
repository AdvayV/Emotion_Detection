"""Verify saved multilingual emotion encoder results without loading weights."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from hinglish_emotion.tasks import EMOTION_LABELS

CONTRACT = Path("colab/mbert_study/study_contract.json")
MBERT = Path("models/mbert_multilingual_emotion")
XLMR = Path("models/xlmr_multilingual_emotion")
BASELINE = Path("colab/mbert_study/xlmr_baseline_metrics.json")
MBERT_BASE = "google-bert/bert-base-multilingual-cased"
XLMR_BASE = "FacebookAI/xlm-roberta-base"


def _read(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read {path.as_posix()}: {exc}") from exc


def _score(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"Invalid {name}; expected a finite score from 0 to 1")
    return float(value)


def _validate_test(test, expected_support, *, rows=None):
    if not isinstance(test, dict):
        raise ValueError("Saved run has no test metrics")
    if "rows" in test and (isinstance(test["rows"], bool) or not isinstance(test["rows"], int) or test["rows"] < 1):
        raise ValueError("Test row count must be a positive integer")
    labels = list(EMOTION_LABELS)
    per_class = test.get("per_class")
    matrix = test.get("confusion_matrix")
    if not isinstance(per_class, dict) or set(per_class) != set(labels):
        raise ValueError("Test metrics must include all seven ordered emotions")
    if not isinstance(matrix, dict) or set(matrix) != set(labels):
        raise ValueError("Test metrics must include a seven-class confusion matrix")
    supports, diag, predicted = {}, {}, {label: 0 for label in labels}
    total = 0
    for actual in labels:
        line = matrix[actual]
        if not isinstance(line, dict) or set(line) != set(labels):
            raise ValueError("Confusion matrix labels do not match the study")
        supports[actual] = 0
        for predicted_label in labels:
            count = line[predicted_label]
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError("Confusion matrix must contain nonnegative integer counts")
            supports[actual] += count
            predicted[predicted_label] += count
            total += count
        diag[actual] = line[actual]
    if supports != expected_support or (rows is not None and total != rows) or test.get("rows", total) != total:
        raise ValueError("Test support or row total differs from the study")
    stats = {}
    for label in labels:
        item = per_class[label]
        if not isinstance(item, dict) or item.get("support") != supports[label]:
            raise ValueError("Per-emotion support does not match confusion matrix")
        tp = diag[label]
        precision = tp / predicted[label] if predicted[label] else 0.0
        recall = tp / supports[label] if supports[label] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        for key, calculated in (("precision", precision), ("recall", recall), ("f1", f1)):
            actual_value = _score(item.get(key), f"{label} {key}")
            if not math.isclose(actual_value, calculated, abs_tol=1e-8):
                raise ValueError(f"{label} {key} does not reconcile with the confusion matrix")
        stats[label] = {"precision": precision, "recall": recall, "f1": f1, "support": supports[label]}
    accuracy = _score(test.get("accuracy"), "accuracy")
    macro_f1 = _score(test.get("macro_f1"), "macro-F1")
    derived_accuracy = sum(diag.values()) / total
    derived_f1 = sum(stats[label]["f1"] for label in labels) / len(labels)
    if not math.isclose(accuracy, derived_accuracy, abs_tol=1e-8) or not math.isclose(macro_f1, derived_f1, abs_tol=1e-8):
        raise ValueError("Accuracy or macro-F1 does not reconcile with per-class results")
    return {"accuracy": accuracy, "macro_f1": macro_f1, "rows": total, "per_class": stats}


def _validate_run(metrics, expected, base_model):
    if metrics.get("task") != "emotion" or metrics.get("base_model") != base_model:
        raise ValueError("Unexpected task or base encoder")
    if metrics.get("labels") != list(EMOTION_LABELS):
        raise ValueError("Saved run has different emotion labels")
    for key in ("dataset_sha256", "split_sha256"):
        if metrics.get(key) != expected[key]:
            raise ValueError(f"Saved run {key} differs from the study contract")
    if metrics.get("split_sizes") != expected["split_sizes"]:
        raise ValueError("Saved run split sizes differ from the study contract")
    if metrics.get("evaluation_scope") != "synthetic_only" or metrics.get("training_data_kind") != "synthetic_only":
        raise ValueError("Study contract only accepts synthetic-only evaluation")
    for key, value in expected["settings"].items():
        if metrics.get(key) != value:
            raise ValueError(f"Saved run setting {key} differs from the study contract")
    return metrics


def _combined_matrix(scores):
    """Add confusion matrices from the disjoint bilingual test subsets."""
    labels = list(EMOTION_LABELS)
    matrix = {actual: {guess: 0 for guess in labels} for actual in labels}
    for part in scores:
        for actual in labels:
            for guess in labels:
                matrix[actual][guess] += part["confusion_matrix"][actual][guess]
    return matrix


def _same_matrix(left, right):
    return all(left[actual][guess] == right[actual][guess]
               for actual in EMOTION_LABELS for guess in EMOTION_LABELS)


def load_encoder_results(root: Path):
    """Return verified scores and notices; absent mBERT metrics remain pending."""
    root = Path(root)
    contract = _read(root / CONTRACT)
    if contract is None:
        return {"status": "pending", "message": "The multilingual encoder study contract is missing."}
    if contract.get("schema_version") != 1 or contract.get("job") != "mbert_combined":
        raise ValueError("Unsupported multilingual encoder study contract")
    baseline = _read(root / BASELINE)
    installed_xlmr = _read(root / XLMR / "metrics.json")
    if baseline is None or installed_xlmr is None:
        raise ValueError("The saved XLM-R baseline or installed XLM-R metrics are missing")
    expected_hash = hashlib.sha256(json.dumps(baseline, sort_keys=True).encode()).hexdigest()
    if expected_hash != contract["baseline_metrics_sha256"]:
        raise ValueError("The saved XLM-R baseline metrics hash differs from the contract")
    if hashlib.sha256(json.dumps(installed_xlmr, sort_keys=True).encode()).hexdigest() != expected_hash:
        raise ValueError("Installed XLM-R metrics differ from the contract baseline")
    baseline = _validate_run(baseline, contract, XLMR_BASE)
    baseline_score = _validate_test(baseline["test"], contract["test_support"], rows=contract["split_sizes"]["test"])
    baseline_by_language = {}
    for language in ("hi-en", "fr-en"):
        expected_language_support = {label: (143 if label != "neutral" else 142) for label in EMOTION_LABELS}
        baseline_by_language[language] = _validate_test(baseline.get("test_by_language", {}).get(language),
                                                        expected_language_support, rows=1000)

    mbert_path = root / MBERT
    mbert_metrics = _read(mbert_path / "metrics.json")
    if mbert_metrics is None:
        return {"status": "pending", "baseline": baseline_score,
                "message": "mBERT training metrics are not installed yet."}
    metadata = _read(mbert_path / "emotion_metadata.json")
    if metadata is None:
        raise ValueError("mBERT emotion metadata is missing")
    for key in ("task", "base_model", "labels", "dataset_sha256", "split_sha256"):
        if metadata.get(key) != mbert_metrics.get(key):
            raise ValueError(f"mBERT metadata and metrics differ for {key}")
    if metadata.get("training_status") != "complete" or set(metadata.get("trained_language_pairs", [])) != {"fr-en", "hi-en"}:
        raise ValueError("mBERT metadata does not describe a completed bilingual run")
    _validate_run(mbert_metrics, contract, MBERT_BASE)
    for language in ("hi-en", "fr-en"):
        source = mbert_metrics.get("test_by_language", {}).get(language)
        if not isinstance(source, dict):
            raise ValueError(f"mBERT {language} test results are missing")
        expected_support = {label: contract["test_support"][label] // 2 for label in EMOTION_LABELS}
        if expected_support["neutral"] != 142:
            expected_support["neutral"] = 142
        checked = _validate_test(source, expected_support, rows=1000)
        for key in ("accuracy", "macro_f1"):
            if not math.isclose(_score(source.get(key), key), checked[key], abs_tol=1e-8):
                raise ValueError(f"mBERT {language} aggregate {key} is inconsistent")
    mbert_overall = _validate_test(mbert_metrics["test"], contract["test_support"], rows=2000)
    if not _same_matrix(_combined_matrix([mbert_metrics["test_by_language"][pair] for pair in ("hi-en", "fr-en")]),
                        mbert_metrics["test"]["confusion_matrix"]):
        raise ValueError("mBERT bilingual test matrix differs from its language-specific test matrices")
    if not _same_matrix(_combined_matrix([baseline["test_by_language"][pair] for pair in ("hi-en", "fr-en")]),
                        baseline["test"]["confusion_matrix"]):
        raise ValueError("XLM-R bilingual test matrix differs from its language-specific test matrices")
    if mbert_metrics.get("emotion_metadata") != metadata:
        raise ValueError("Embedded and installed mBERT emotion metadata differ")
    return {"status": "verified", "baseline": baseline_score, "baseline_by_language": baseline_by_language,
            "mbert": mbert_overall,
            "mbert_by_language": {pair: _validate_test(mbert_metrics["test_by_language"][pair],
                {label: (143 if label != "neutral" else 142) for label in EMOTION_LABELS}, rows=1000)
                for pair in ("hi-en", "fr-en")},
            "history": mbert_metrics.get("history", []), "contract": contract}
