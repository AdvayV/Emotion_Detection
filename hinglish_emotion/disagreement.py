"""Sequential comparison helpers for the human-reviewed model disagreement queue."""

from __future__ import annotations

import gc
import math
from pathlib import Path
from typing import Callable, Iterable

from .tasks import EMOTION_LABELS, LANGUAGE_PAIRS

MAX_PHRASES = 20


def probability_summary(probabilities: dict[str, float]) -> dict[str, object]:
    """Validate and summarize base probabilities without recalibrating them."""
    if set(probabilities) != set(EMOTION_LABELS):
        raise ValueError("Each checkpoint must return exactly the seven emotion labels")
    values = {label: float(probabilities[label]) for label in EMOTION_LABELS}
    if any(not math.isfinite(value) or value < 0 or value > 1 for value in values.values()):
        raise ValueError("Checkpoint probabilities must be finite values between zero and one")
    if not math.isclose(sum(values.values()), 1.0, rel_tol=1e-5, abs_tol=1e-5):
        raise ValueError("Checkpoint probabilities must sum to one")
    ranked = sorted(values.values(), reverse=True)
    return {
        "probabilities": values,
        "emotion": max(values, key=values.get),
        "score": values[max(values, key=values.get)],
        "uncertainty": ranked[0] < 0.55 or ranked[0] - ranked[1] < 0.15,
    }


def jensen_shannon(left: dict[str, float], right: dict[str, float]) -> tuple[float, float]:
    """Return base-2 Jensen-Shannon divergence and its square-root distance."""
    a = [max(float(left[label]), 0.0) for label in EMOTION_LABELS]
    b = [max(float(right[label]), 0.0) for label in EMOTION_LABELS]
    sa, sb = sum(a), sum(b)
    if sa <= 0 or sb <= 0:
        raise ValueError("Probability distributions must have positive mass")
    a, b = [v / sa for v in a], [v / sb for v in b]
    midpoint = [(x + y) / 2 for x, y in zip(a, b)]

    def kl(p, q):
        return sum(x * math.log2(x / y) for x, y in zip(p, q) if x > 0)

    divergence = (kl(a, midpoint) + kl(b, midpoint)) / 2
    return divergence, math.sqrt(max(0.0, divergence))


def compare_phrases(
    phrases: Iterable[str],
    language_pair: str,
    checkpoints: Iterable[tuple[str, str | Path, str]],
    *,
    classifier_factory: Callable[[str | Path], object],
    clear_shared_cache: Callable[[], None] | None = None,
) -> list[dict[str, object]]:
    """Run compatible checkpoints one at a time and release each before the next."""
    texts = [str(text) for text in phrases]
    models = list(checkpoints)
    if language_pair not in LANGUAGE_PAIRS:
        raise ValueError("Choose a supported language mix")
    if not 1 <= len(texts) <= MAX_PHRASES or any(not text.strip() for text in texts):
        raise ValueError(f"Provide between 1 and {MAX_PHRASES} non-empty phrases")
    if any(len(text) > 20_000 for text in texts):
        raise ValueError("Each phrase must be at most 20,000 characters")
    if len(models) < 2 or len({name for name, _, _ in models}) != len(models):
        raise ValueError("Choose at least two distinct trained checkpoints")

    if clear_shared_cache:
        clear_shared_cache()
    collected: list[list[dict[str, object]]] = [[] for _ in texts]
    for name, path, version in models:
        classifier = None
        try:
            classifier = classifier_factory(path)
            for index, text in enumerate(texts):
                result = classifier.predict_text(text)
                if result.source != "trained_checkpoint":
                    raise ValueError(f"{name} did not return a trained-checkpoint prediction")
                summary = probability_summary(dict(result.probabilities))
                warnings = tuple(getattr(result, "warnings", ()))
                collected[index].append({
                    "model": name,
                    "checkpoint": str(path),
                    "version": version,
                    "source": result.source,
                    "warnings": warnings,
                    "truncated": bool(getattr(result, "truncated", False)),
                    "experimental": bool(getattr(result, "experimental", False)),
                    "model_name": str(getattr(result, "model_name", name)),
                    **summary,
                })
        finally:
            if classifier is not None:
                del classifier
            gc.collect()

    rows = []
    for text, predictions in zip(texts, collected):
        labels = {str(item["emotion"]) for item in predictions}
        divergences = []
        for left_index in range(len(predictions)):
            for right_index in range(left_index + 1, len(predictions)):
                div, distance = jensen_shannon(
                    predictions[left_index]["probabilities"],
                    predictions[right_index]["probabilities"],
                )
                divergences.append({
                    "models": (predictions[left_index]["model"], predictions[right_index]["model"]),
                    "divergence": div,
                    "distance": distance,
                })
        rows.append({
            "text": text,
            "language_pair": language_pair,
            "predictions": predictions,
            "disagreement": len(labels) > 1,
            "uncertainty": any(bool(item["uncertainty"] or item["truncated"]) for item in predictions),
            "pairwise_js": divergences,
        })
    return rows


QUEUE_COLUMNS = (
    "text", "emotion", "language_pair", "group_id", "human_reviewed",
    "synthetic", "source", "review_notes", "provenance", "split",
)


def queue_csv(rows: Iterable[dict[str, object]]) -> str:
    """Serialize reviewed candidates with CSV quoting and unspecified split."""
    import csv
    import io

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=QUEUE_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({**row, "split": ""})
    return buffer.getvalue()


def add_reviewed_candidate(queue: list[dict[str, object]], candidate: dict[str, object]) -> list[dict[str, object]]:
    """Return a queue with text/language deduplication; callers store it in session state."""
    text = str(candidate.get("text", "")).strip()
    pair = str(candidate.get("language_pair", ""))
    emotion = str(candidate.get("emotion", ""))
    if not text or pair not in LANGUAGE_PAIRS or emotion not in EMOTION_LABELS:
        raise ValueError("A reviewed candidate needs text, a supported language mix and a seven-emotion label")
    duplicate_key = " ".join(text.casefold().split())
    if any(" ".join(str(item.get("text", "")).casefold().split()) == duplicate_key and item.get("language_pair") == pair for item in queue):
        return queue
    return [*queue, {
        "text": text, "emotion": emotion, "language_pair": pair,
        "group_id": str(candidate.get("group_id") or ""),
        "human_reviewed": True, "synthetic": candidate.get("synthetic"),
        "source": str(candidate.get("source", "model_disagreement_review")),
        "review_notes": str(candidate.get("review_notes", "")),
        "provenance": str(candidate.get("provenance", "")), "split": "",
    }]
