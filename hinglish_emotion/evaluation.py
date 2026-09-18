from __future__ import annotations

from collections import Counter
import math
from typing import Iterable, Sequence


LABELS = ("positive", "neutral", "negative")


def classification_metrics(y_true: Sequence[str], y_pred: Sequence[str]) -> dict[str, object]:
    if len(y_true) != len(y_pred) or not y_true:
        raise ValueError("y_true and y_pred must be non-empty and have equal length")
    per_class: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    for label in LABELS:
        tp = sum(actual == label and predicted == label for actual, predicted in zip(y_true, y_pred))
        fp = sum(actual != label and predicted == label for actual, predicted in zip(y_true, y_pred))
        fn = sum(actual == label and predicted != label for actual, predicted in zip(y_true, y_pred))
        support = sum(actual == label for actual in y_true)
        precision = _safe_divide(tp, tp + fp)
        recall = _safe_divide(tp, tp + fn)
        f1 = _safe_divide(2 * precision * recall, precision + recall)
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        f1_values.append(f1)
    return {
        "accuracy": sum(actual == predicted for actual, predicted in zip(y_true, y_pred)) / len(y_true),
        "macro_f1": sum(f1_values) / len(f1_values),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }


def confusion_matrix(y_true: Sequence[str], y_pred: Sequence[str]) -> dict[str, dict[str, int]]:
    matrix = {actual: {predicted: 0 for predicted in LABELS} for actual in LABELS}
    for actual, predicted in zip(y_true, y_pred):
        if actual not in matrix or predicted not in LABELS:
            raise ValueError(f"Unknown label pair: {actual!r}, {predicted!r}")
        matrix[actual][predicted] += 1
    return matrix


def expected_calibration_error(
    confidences: Sequence[float], correct: Sequence[bool], bins: int = 10
) -> float:
    if len(confidences) != len(correct) or not confidences:
        raise ValueError("confidences and correct must be non-empty and have equal length")
    if bins < 1:
        raise ValueError("bins must be positive")
    ece = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        members = [i for i, value in enumerate(confidences) if lower <= value <= upper if index == bins - 1 or value < upper]
        if not members:
            continue
        accuracy = sum(correct[i] for i in members) / len(members)
        confidence = sum(confidences[i] for i in members) / len(members)
        ece += len(members) / len(confidences) * abs(accuracy - confidence)
    return ece


def variation_consistency(prediction_groups: Iterable[Sequence[str]]) -> float:
    groups = [tuple(group) for group in prediction_groups if group]
    if not groups:
        raise ValueError("At least one prediction group is required")
    return sum(len(set(group)) == 1 for group in groups) / len(groups)


def sentiment_flip_accuracy(pairs: Iterable[tuple[str, str, str, str]]) -> float:
    """Fraction where both original and meaning-changing predictions are correct."""
    pairs = list(pairs)
    if not pairs:
        raise ValueError("At least one counterfactual pair is required")
    correct = sum(
        original_prediction == original_label and changed_prediction == changed_label
        for original_label, original_prediction, changed_label, changed_prediction in pairs
    )
    return correct / len(pairs)


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0

