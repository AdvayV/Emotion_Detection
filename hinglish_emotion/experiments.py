"""Reproducible, per-row emotion preprocessing and review experiments."""
from __future__ import annotations

from collections import defaultdict
import re
import time
from typing import Callable, Iterable
import emoji as _emoji

from .normalization import HinglishNormalizer
from .evaluation_guard import phrase_hash, training_hashes
from .emotions import EmotionResult
from .tasks import EMOTION_LABELS, LANGUAGE_PAIRS

VARIANTS = ("baseline", "normalized", "emoji_removed", "context", "no_context")
_REVIEW_MODES = {"none", "selective", "all"}
def _remove_emoji(text: str) -> str:
    text = _emoji.replace_emoji(text, replace="")
    return re.sub(r"\s+", " ", text).strip()


def _prediction(value):
    if isinstance(value, dict):
        label = value.get("emotion", value.get("label"))
        scores = value.get("probabilities", value.get("scores", {}))
        source = value.get("source", value.get("model_name", "predictor"))
        return label, scores if isinstance(scores, dict) else {}, source
    label = getattr(value, "emotion", getattr(value, "label", None))
    scores = getattr(value, "probabilities", {})
    return label, scores if isinstance(scores, dict) else {}, getattr(value, "source", type(value).__name__)


def _call_predictor(predictor, text, language, context):
    if hasattr(predictor, "analyze"):
        result = predictor.analyze(text, language_pair=language, context=context)
        primary = result.primary
        return primary.emotion, dict(primary.probabilities), primary.source, getattr(result, "final_emotion", primary.emotion), getattr(result, "review", {})
    if context and hasattr(predictor, "predict_context"):
        value=predictor.predict_context(text,language_pair=language,context=context)
    elif hasattr(predictor, "predict"):
        value = predictor.predict(text, language_pair=language)
    else:
        try:
            value = predictor(text, language_pair=language, context=context)
        except TypeError:
            try:
                value = predictor(text, language, context)
            except TypeError:
                value = predictor(text)
    label, scores, source = _prediction(value)
    return label, scores, source, label, {}


def _expected(row, variant):
    explicit = row.get(f"expected_label_{variant}") or row.get(f"{variant}_expected_label")
    if explicit:
        return str(explicit).strip().lower(), "explicit_variant_label"
    invariant = row.get("label_invariant") is True or str(row.get("label_invariant", "")).strip().lower() in {"1", "true", "yes"}
    if variant == "baseline" or invariant:
        label = row.get("expected_label", row.get("label", row.get("emotion")))
        if label:
            return str(label).strip().lower(), "declared_label"
        if invariant:
            return None, "invariance_declared_without_label"
        return None, "unlabelled"
    return None, "label_not_declared_for_variant"


def run_experiments(rows: Iterable[dict], predictor: Callable, variants=VARIANTS, *, review_mode="none", reviewer=None, row_limit=None, require_test_split=True):
    """Run paired input variants; rows may contain text, language_pair, label/split/context.

    Reviewer is called only when explicitly enabled. It may be an object with
    ``review(text, normalized_text, primary, context=..., language_pair=...)`` or
    a callable. Reviewer outputs are suggestions; base labels and scores remain intact.
    """
    variants = tuple(dict.fromkeys(variants))
    if not variants or any(item not in VARIANTS for item in variants):
        raise ValueError(f"variants must be selected from {VARIANTS}")
    if review_mode not in _REVIEW_MODES:
        raise ValueError("review_mode must be none, selective or all")
    if review_mode != "none" and reviewer is None:
        raise ValueError("A reviewer is required when review_mode is enabled")
    source_rows = list(rows)
    all_rows = list(source_rows)
    overlap_hashes = {
        phrase_hash(str(row.get("text", ""))) for row in all_rows
        if str(row.get("split", "")).strip().lower() in {"train", "validation"} and str(row.get("text", "")).strip()
    }
    overlap_warning = None
    try:
        guard_hashes = training_hashes(predictor)
        if guard_hashes:
            overlap_hashes.update(guard_hashes)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        overlap_warning = f"Checkpoint overlap guard could not be read: {exc}"
    if require_test_split:
        source_rows = [row for row in source_rows if str(row.get("split", "test")).strip().lower() == "test"]
    if row_limit is not None:
        if row_limit < 1:
            raise ValueError("row_limit must be positive")
        source_rows = source_rows[:row_limit]
    normalizer = HinglishNormalizer()
    output = []
    for index, row in enumerate(source_rows):
        text = str(row.get("text", ""))
        if not text.strip():
            continue
        row_id = row.get("row_id", row.get("id", index))
        language = str(row.get("language_pair") or row.get("language") or "hi-en")
        if language not in LANGUAGE_PAIRS:
            for variant in variants:
                expected, label_source = _expected(row, variant)
                output.append({"row_id": row_id, "row_instance_id": index, "text": text, "input_text": text,
                    "input_variant": variant, "language_pair": language, "context": "", "actual_label": expected,
                    "label_source": label_source, "predicted_label": None, "base_label": None,
                    "review_suggestion": None, "review_status": "unsupported_language", "source": None,
                    "label_changed_by_review": False, "reviewer_evidence": [],
                    "scores": {}, "latency_ms": 0.0, "prediction_error": f"Unsupported language mix: {language}",
                    "review_error": "", "error": f"Unsupported language mix: {language}", "heldout_overlap": phrase_hash(text) in overlap_hashes})
            continue
        normalized = normalizer.normalize(text)
        for variant in variants:
            emoji_free = _remove_emoji(text)
            variant_text = {"baseline": text, "normalized": normalized.normalized_text,
                            "emoji_removed": emoji_free,
                            "context": text, "no_context": text}[variant]
            context = str(row.get("context", "") or "") if variant == "context" else ""
            started = time.perf_counter()
            label = source = None
            scores = {}
            prediction_error = ""
            try:
                label, scores, source, _, _ = _call_predictor(predictor, variant_text, language, context)
                if label not in EMOTION_LABELS:
                    raise ValueError(f"Predictor returned unsupported emotion: {label!r}")
                scores = {str(k): float(v) for k, v in (scores or {}).items()}
            except Exception as exc:
                prediction_error = f"{type(exc).__name__}: {exc}"
                label, source, scores = None, None, {}
            elapsed = (time.perf_counter() - started) * 1000
            expected, label_source = _expected(row, variant)
            review_suggestion = None
            review_status = "disabled"
            review_error = ""
            reviewer_evidence = []
            review_latency = 0.0
            if not prediction_error and review_mode != "none":
                should_review = review_mode == "all" or (review_mode == "selective" and (not scores or max(scores.values(), default=0) < .60))
                if should_review:
                    if sum(1 for item in output if item["review_status"] in {"suggested", "error"}) >= 5:
                        review_status = "limit_reached"
                    else:
                        review_start = time.perf_counter()
                        try:
                            primary = EmotionResult(label, scores, source=str(source or "unknown"))
                            if hasattr(reviewer, "review"):
                                if hasattr(reviewer, "availability"):
                                    ready, message = reviewer.availability()
                                    if not ready:
                                        raise RuntimeError(message)
                                result = reviewer.review(variant_text, normalizer.normalize(variant_text).normalized_text,
                                    primary, context=context, language_pair=language)
                                if hasattr(reviewer, "validate"):
                                    result = reviewer.validate(result, variant_text)
                            else:
                                result = reviewer(variant_text, label, context)
                            review_suggestion = result.get("emotion", result.get("suggestion")) if isinstance(result, dict) else getattr(result, "emotion", None)
                            reviewer_evidence = result.get("evidence", []) if isinstance(result, dict) else []
                            if review_suggestion not in EMOTION_LABELS:
                                raise ValueError("Reviewer returned unsupported emotion")
                            review_status = "suggested"
                        except Exception as exc:
                            review_status = "error"
                            review_error = f"{type(exc).__name__}: {exc}"
                        review_latency = (time.perf_counter() - review_start) * 1000
                        elapsed += review_latency
            output.append({"row_id": row_id, "row_instance_id": index, "text": text, "input_text": variant_text,
                "input_variant": variant, "language_pair": language, "context": context,
                "heldout_overlap": phrase_hash(text) in overlap_hashes,
                "actual_label": expected, "label_source": label_source, "predicted_label": label,
                "base_label": label, "review_suggestion": review_suggestion, "review_status": review_status,
                "label_changed_by_review": bool(review_suggestion and label and review_suggestion != label),
                "reviewer_evidence": reviewer_evidence, "source": source, "scores": scores,
                "latency_ms": round(elapsed, 3), "primary_latency_ms": round(elapsed - review_latency, 3),
                "review_latency_ms": round(review_latency, 3), "prediction_error": prediction_error,
                "review_error": review_error, "error": "; ".join(item for item in (prediction_error, review_error) if item)})
    grouped = defaultdict(list)
    for item in output:
        grouped[item["input_variant"]].append(item)
    summaries = {}
    for variant, items in grouped.items():
        evaluated = [item for item in items if item["actual_label"] and item["label_source"] in {"declared_label", "explicit_variant_label"} and not item["prediction_error"]]
        stable = [item for item in items if item["predicted_label"] and not item["prediction_error"]]
        baseline_by_id = {b["row_instance_id"]: b["predicted_label"] for b in grouped.get("baseline", [])}
        paired = [(item, baseline_by_id[item["row_instance_id"]]) for item in stable if baseline_by_id.get(item["row_instance_id"])]
        changes = sum(item["predicted_label"] != baseline for item, baseline in paired)
        has_baseline = bool(grouped.get("baseline"))
        summaries[variant] = {"rows": len(items), "successful": sum(not item["prediction_error"] for item in items),
            "errors": sum(bool(item["prediction_error"]) for item in items),
            "review_errors": sum(bool(item["review_error"]) for item in items),
            "mean_latency_ms": round(sum(item["latency_ms"] for item in items) / max(len(items), 1), 3),
            "mean_primary_latency_ms": round(sum(item.get("primary_latency_ms", 0) for item in items) / max(len(items), 1), 3),
            "mean_review_latency_ms": round(sum(item.get("review_latency_ms", 0) for item in items) / max(len(items), 1), 3),
            "label_stability_vs_baseline": round(1 - changes / len(paired), 4) if has_baseline and paired else None,
            "accuracy": round(sum(item["actual_label"] == item["predicted_label"] for item in evaluated) / len(evaluated), 4) if evaluated else None,
            "accuracy_rows": len(evaluated), "label_scope": "declared labels only; meaning-changing variants excluded without explicit labels or invariance"}
    reviews = [item for item in output if item["review_status"] in {"suggested", "error"}]
    reviewed_labelled = [item for item in reviews if item["review_status"] == "suggested" and item["actual_label"] and item["label_source"] in {"declared_label", "explicit_variant_label"} and item["predicted_label"]]
    review_eval_rows = len(reviewed_labelled)
    before_correct = sum(item["base_label"] == item["actual_label"] for item in reviewed_labelled)
    after_correct = sum(item["review_suggestion"] == item["actual_label"] for item in reviewed_labelled)
    return {"rows": output, "summaries": summaries, "review_summary": {"mode": review_mode, "attempted": len(reviews),
            "coverage": round(len(reviews) / max(len(output), 1), 4), "suggestions": sum(item["review_status"] == "suggested" for item in reviews),
            "errors": sum(item["review_status"] == "error" for item in reviews),
            "mean_review_latency_ms": round(sum(item["review_latency_ms"] for item in reviews) / max(len(reviews), 1), 3),
            "changed_labels": sum(item["label_changed_by_review"] for item in output),
            "labelled_rows_compared": review_eval_rows,
            "accuracy_before": round(before_correct / review_eval_rows, 4) if review_eval_rows else None,
            "accuracy_after": round(after_correct / review_eval_rows, 4) if review_eval_rows else None,
            "reviewer_is_suggestion_only": True},
        "heldout_overlap_rows": sum(item.get("heldout_overlap", False) for item in output),
        "warnings": ["Metrics describe only explicitly labelled rows. Assistant-authored drafts are not independent benchmark evidence.",
                     "Preprocessing variants may change intended emotion; stability is not accuracy."] + ([overlap_warning] if overlap_warning else [])}
