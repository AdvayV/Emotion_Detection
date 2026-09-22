from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from hinglish_emotion.evaluation import classification_metrics, expected_calibration_error


@dataclass(frozen=True)
class ModelEvaluation:
    """Evaluation data used by the dashboard's model-effectiveness charts."""

    model: str
    summary: dict[str, float | int | str]
    per_class: pd.DataFrame
    confusion: pd.DataFrame
    calibration: pd.DataFrame
    predictions: pd.DataFrame


def evaluate_pipeline(frame: pd.DataFrame, pipeline: Any, model_name: str) -> ModelEvaluation:
    """Run a project pipeline on labelled rows and return chart-ready metrics."""
    required = {"text", "label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing))}")

    evaluation_frame = frame.dropna(subset=["text", "label"]).copy()
    evaluation_frame["text"] = evaluation_frame["text"].astype(str).str.strip()
    evaluation_frame["label"] = evaluation_frame["label"].astype(str).str.strip().str.lower()
    evaluation_frame = evaluation_frame[
        evaluation_frame["text"].ne("")
        & evaluation_frame["label"].isin({"positive", "neutral", "negative"})
    ]
    if evaluation_frame.empty:
        raise ValueError("No valid labelled rows are available for evaluation")

    records: list[dict[str, object]] = []
    for _, row in evaluation_frame.iterrows():
        result = pipeline.predict(row["text"])
        records.append(
            {
                "text": row["text"],
                "actual": row["label"],
                "predicted": result.final_label,
                "confidence": result.primary.confidence,
                "correct": result.final_label == row["label"],
                "routed_for_review": result.review_decision.should_review,
                "review_applied": result.reviewer is not None,
                "sarcasm_probability": result.primary.sarcasm_probability,
            }
        )

    predictions = pd.DataFrame(records)
    scores = classification_metrics(
        predictions["actual"].tolist(), predictions["predicted"].tolist()
    )
    ece = expected_calibration_error(
        predictions["confidence"].tolist(), predictions["correct"].tolist()
    )
    summary: dict[str, float | int | str] = {
        "model": model_name,
        "rows": len(predictions),
        "accuracy": float(scores["accuracy"]),
        "macro_f1": float(scores["macro_f1"]),
        "calibration_error": float(ece),
        "average_confidence": float(predictions["confidence"].mean()),
        "review_route_rate": float(predictions["routed_for_review"].mean()),
        "review_applied_rate": float(predictions["review_applied"].mean()),
    }

    per_class_rows = []
    for label, values in scores["per_class"].items():
        per_class_rows.append(
            {
                "model": model_name,
                "sentiment": label,
                "precision": float(values["precision"]),
                "recall": float(values["recall"]),
                "f1": float(values["f1"]),
                "support": int(values["support"]),
            }
        )

    labels = ["positive", "neutral", "negative"]
    confusion = pd.DataFrame(scores["confusion_matrix"]).T.loc[labels, labels]
    confusion.index.name = "Actual"
    confusion.columns.name = "Predicted"

    return ModelEvaluation(
        model=model_name,
        summary=summary,
        per_class=pd.DataFrame(per_class_rows),
        confusion=confusion,
        calibration=calibration_table(predictions),
        predictions=predictions,
    )


def calibration_table(predictions: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    """Aggregate model confidence into reliability-diagram bins."""
    if bins < 1:
        raise ValueError("bins must be positive")
    working = predictions.copy()
    edges = [index / bins for index in range(bins + 1)]
    working["bin"] = pd.cut(
        working["confidence"], bins=edges, include_lowest=True, right=True
    )
    rows: list[dict[str, float | int]] = []
    for interval, group in working.groupby("bin", observed=False):
        if group.empty:
            continue
        rows.append(
            {
                "bin_midpoint": float((interval.left + interval.right) / 2),
                "mean_confidence": float(group["confidence"].mean()),
                "empirical_accuracy": float(group["correct"].mean()),
                "count": len(group),
            }
        )
    return pd.DataFrame(rows)
