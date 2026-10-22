"""Conservative temperature scaling for seven-class emotion scores.

Calibration changes confidence values only. It does not repair incorrect labels,
model predictions, or the model's learned decision boundaries.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .tasks import EMOTION_LABELS, canonical_emotion

PROFILE_SCHEMA = 1
MIN_TEMPERATURE = 0.05
MAX_TEMPERATURE = 20.0


def _validated_probs(probabilities: Mapping[str, float]) -> dict[str, float]:
    if not isinstance(probabilities, Mapping) or set(probabilities) != set(EMOTION_LABELS):
        raise ValueError("Probabilities must contain exactly the seven emotion labels")
    values = {}
    for label in EMOTION_LABELS:
        if isinstance(probabilities[label], bool):
            raise ValueError("Probabilities must be numeric values, not booleans")
        try:
            value = float(probabilities[label])
        except (TypeError, ValueError) as exc:
            raise ValueError("Probabilities must be finite numbers") from exc
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError("Probabilities must be finite values between zero and one")
        values[label] = value
    if not math.isclose(sum(values.values()), 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("Seven emotion probabilities must sum to one")
    return values


def apply_temperature(probabilities: Mapping[str, float], temperature: float) -> dict[str, float]:
    """Apply temperature scaling to probabilities, retaining their argmax."""
    probs = _validated_probs(probabilities)
    t = float(temperature)
    if not math.isfinite(t) or not MIN_TEMPERATURE <= t <= MAX_TEMPERATURE:
        raise ValueError(f"Temperature must be between {MIN_TEMPERATURE} and {MAX_TEMPERATURE}")
    # p**(1/T) is softmax(logits/T) without needing logits. Zero remains zero.
    logs = {label: (math.log(p) / t if p > 0 else -math.inf) for label, p in probs.items()}
    peak = max(logs.values())
    scaled = {label: (math.exp(value - peak) if math.isfinite(value) else 0.0) for label, value in logs.items()}
    total = sum(scaled.values())
    result = {label: value / total for label, value in scaled.items()}
    if max(probs, key=probs.get) != max(result, key=result.get):
        raise ValueError("Temperature scaling unexpectedly changed the predicted class")
    return result


def _nll(rows: Sequence[Mapping[str, float]], indices: Sequence[int], t: float) -> float:
    total = 0.0
    for probs, index in zip(rows, indices):
        p = apply_temperature(probs, t)[EMOTION_LABELS[index]]
        total -= math.log(max(p, 1e-300))
    return total / len(rows)


def _metrics(rows: Sequence[Mapping[str, float]], indices: Sequence[int], t: float, bins: int = 10) -> dict:
    scaled = [apply_temperature(row, t) for row in rows]
    nll = sum(-math.log(max(p[EMOTION_LABELS[i]], 1e-300)) for p, i in zip(scaled, indices)) / len(rows)
    brier = sum(sum((p[label] - float(label == EMOTION_LABELS[i])) ** 2 for label in EMOTION_LABELS) for p, i in zip(scaled, indices)) / len(rows)
    ece = 0.0
    for b in range(bins):
        group = [(p, i) for p, i in zip(scaled, indices) if b / bins <= max(p.values()) < (b + 1) / bins or (b == bins - 1 and max(p.values()) == 1.0)]
        if group:
            confidence = sum(max(p.values()) for p, _ in group) / len(group)
            accuracy = sum(max(p, key=p.get) == EMOTION_LABELS[i] for p, i in group) / len(group)
            ece += len(group) / len(scaled) * abs(accuracy - confidence)
    return {"nll": nll, "brier": brier, "ece": ece}


def _identity(model_identity: Mapping | str) -> dict:
    if isinstance(model_identity, str):
        model_identity = {"model_path": model_identity}
    if not isinstance(model_identity, Mapping):
        raise ValueError("A checkpoint identity is required")
    identity = dict(model_identity)
    if not identity.get("model_fingerprint"):
        if identity.get("checkpoint_version"):
            identity["model_fingerprint"] = str(identity["checkpoint_version"])
        else:
            source = str(identity.get("model_path", "")).strip()
            if not source:
                raise ValueError("model_identity must include model_fingerprint, checkpoint_version, or model_path")
            path = Path(source)
            digest = hashlib.sha256()
            if path.is_dir():
                files = sorted(p for p in path.rglob("*") if p.is_file() and not any(part.startswith(".") for part in p.parts))
                if not files:
                    raise ValueError("Checkpoint directory has no files to fingerprint")
                for file in files:
                    digest.update(file.relative_to(path).as_posix().encode())
                    digest.update(file.read_bytes())
            elif path.is_file():
                digest.update(path.read_bytes())
            else:
                raise ValueError("model_path must identify an existing checkpoint")
            identity["model_fingerprint"] = digest.hexdigest()
            identity["model_path"] = str(path.resolve())
    if identity.get("labels", list(EMOTION_LABELS)) != list(EMOTION_LABELS):
        raise ValueError("Checkpoint label order must match EMOTION_LABELS")
    identity["labels"] = list(EMOTION_LABELS)
    return identity


def fit_temperature(probability_rows: Iterable[Mapping[str, float]], labels: Iterable[str], *, model_identity: Mapping | str,
                    split: str = "validation", scope: str = "unspecified", synthetic: bool = False,
                    allow_synthetic: bool = False, human_reviewed: bool | None = None,
                    allow_unreviewed: bool = False) -> dict:
    """Fit a bounded scalar on held-out validation/calibration labels."""
    split = str(split).strip().lower()
    if split not in {"validation", "calibration"}:
        raise ValueError("Calibration requires validation rows or an explicitly declared calibration file")
    if synthetic and not allow_synthetic:
        raise ValueError("Synthetic labels require explicit allow_synthetic=True")
    if human_reviewed is not True and not allow_unreviewed:
        raise ValueError("Unknown or unreviewed labels require explicit allow_unreviewed=True")
    rows = [_validated_probs(row) for row in probability_rows]
    normalized_labels = [canonical_emotion(str(label)) for label in labels]
    if not rows or len(rows) != len(normalized_labels):
        raise ValueError("Provide matching, nonempty probability rows and labels")
    if any(label not in EMOTION_LABELS for label in normalized_labels):
        raise ValueError("Labels must be among the seven emotion classes")
    identity = _identity(model_identity)
    index = [EMOTION_LABELS.index(label) for label in normalized_labels]
    baseline = _metrics(rows, index, 1.0)
    # Golden-section minimization in log-temperature; bounded scalar optimization.
    lo, hi = math.log(MIN_TEMPERATURE), math.log(MAX_TEMPERATURE)
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    x1, x2 = hi - ratio * (hi - lo), lo + ratio * (hi - lo)
    f1, f2 = _nll(rows, index, math.exp(x1)), _nll(rows, index, math.exp(x2))
    for _ in range(120):
        if f1 <= f2:
            hi, x2, f2 = x2, x1, f1
            x1 = hi - ratio * (hi - lo)
            f1 = _nll(rows, index, math.exp(x1))
        else:
            lo, x1, f1 = x1, x2, f2
            x2 = lo + ratio * (hi - lo)
            f2 = _nll(rows, index, math.exp(x2))
    candidates = [1.0, math.exp((lo + hi) / 2), MIN_TEMPERATURE, MAX_TEMPERATURE]
    temperature = min(candidates, key=lambda t: _nll(rows, index, t))
    calibrated = _metrics(rows, index, temperature)
    return {
        "schema_version": PROFILE_SCHEMA, "temperature": temperature,
        "model_fingerprint": identity["model_fingerprint"], "model_identity": identity,
        "labels": list(EMOTION_LABELS), "scope": str(scope), "split": split,
        "rows": len(rows), "synthetic": bool(synthetic),
        "human_reviewed": False, "labels_declared_human_reviewed": human_reviewed is True,
        "status": "experimental_unreviewed",
        "metrics_before": baseline, "metrics_after": calibrated,
        "method": "temperature_scaling", "changes_class_predictions": False,
    }


def validate_profile(profile: Mapping, expected_model_fingerprint: str | Mapping) -> dict:
    """Validate profile structure and ensure it belongs to this checkpoint."""
    if not isinstance(profile, Mapping) or profile.get("schema_version") != PROFILE_SCHEMA:
        raise ValueError("Unsupported calibration profile")
    expected = expected_model_fingerprint.get("model_fingerprint") if isinstance(expected_model_fingerprint, Mapping) else expected_model_fingerprint
    if not expected or profile.get("model_fingerprint") != expected:
        raise ValueError("Calibration profile checkpoint fingerprint does not match")
    if profile.get("labels") != list(EMOTION_LABELS) or profile.get("status") != "experimental_unreviewed":
        raise ValueError("Calibration profile labels or review status are invalid")
    t = float(profile.get("temperature", float("nan")))
    if not math.isfinite(t) or not MIN_TEMPERATURE <= t <= MAX_TEMPERATURE:
        raise ValueError("Calibration profile temperature is invalid")
    if profile.get("split") not in {"validation", "calibration"}:
        raise ValueError("Calibration profile must use held-out validation/calibration data")
    if profile.get("human_reviewed") is not False:
        raise ValueError("Calibration profiles cannot claim human review")
    return dict(profile)


def profile_from_records(records: Iterable[Mapping], *, model_identity: Mapping | str, split: str,
                         scope: str, synthetic: bool = False, allow_synthetic: bool = False,
                         human_reviewed: bool | None = None, allow_unreviewed: bool = False) -> dict:
    """Fit directly from records containing probabilities and a gold label."""
    records = list(records)
    return fit_temperature([r["probabilities"] for r in records], [r.get("label", r.get("emotion")) for r in records],
                           model_identity=model_identity, split=split, scope=scope,
                           synthetic=synthetic, allow_synthetic=allow_synthetic,
                           human_reviewed=human_reviewed, allow_unreviewed=allow_unreviewed)


def frozen_phrase_hashes(project_root: str | Path) -> set[str]:
    """Hashes from frozen training/test partitions and challenge evaluation sets."""
    from .evaluation_guard import phrase_hash
    root = Path(project_root)
    paths = [*root.glob("data/training/**/train.csv"), *root.glob("data/training/**/test.csv"), *sorted((root / "evaluation_set").glob("*.csv"))]
    hashes: set[str] = set()
    for path in paths:
        if not path.is_file():
            continue
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                text = row.get("text") or row.get("phrase")
                if text:
                    hashes.add(phrase_hash(text))
    return hashes
