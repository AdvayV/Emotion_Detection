import math

import pytest

from hinglish_emotion.calibration import (
    apply_temperature,
    fit_temperature,
    validate_profile,
)
from hinglish_emotion.tasks import EMOTION_LABELS


def _row(label, confidence=0.55):
    other = (1 - confidence) / 6
    return {item: confidence if item == label else other for item in EMOTION_LABELS}


def test_fit_improves_nll_and_preserves_argmax():
    labels = [EMOTION_LABELS[i % 7] for i in range(140)]
    # Low confidence on reliably correct predictions calls for T < 1.
    rows = [_row(label, .35) for label in labels]
    profile = fit_temperature(rows, labels, model_identity={"model_fingerprint": "model-v1"}, split="validation", human_reviewed=True)
    assert profile["metrics_after"]["nll"] <= profile["metrics_before"]["nll"] + 1e-10
    assert profile["temperature"] < 1
    for row in rows:
        scaled = apply_temperature(row, profile["temperature"])
        assert max(scaled, key=scaled.get) == max(row, key=row.get)
        assert math.isclose(sum(scaled.values()), 1.0)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.1, 1.1, True])
def test_rejects_invalid_probability_values(bad):
    row = _row("joy")
    row["joy"] = bad
    with pytest.raises(ValueError):
        apply_temperature(row, 1.0)


def test_rejects_bad_sum_and_temperature():
    with pytest.raises(ValueError, match="sum to one"):
        apply_temperature({label: .1 for label in EMOTION_LABELS}, 1)
    with pytest.raises(ValueError):
        apply_temperature(_row("joy"), float("nan"))


def test_fit_requires_real_labelled_probabilities_and_validation_split():
    with pytest.raises(ValueError, match="validation"):
        fit_temperature([_row("joy")], ["joy"], model_identity={"model_fingerprint": "model-v1"}, split="test", human_reviewed=True)
    with pytest.raises(ValueError, match="Synthetic"):
        fit_temperature([_row("joy")], ["joy"], model_identity={"model_fingerprint": "model-v1"}, synthetic=True, allow_unreviewed=True)
    with pytest.raises(ValueError, match="unreviewed"):
        fit_temperature([_row("joy")], ["joy"], model_identity={"model_fingerprint": "model-v1"})


def test_profile_is_bound_to_checkpoint_fingerprint_and_unreviewed():
    profile = fit_temperature([_row("joy")], ["joy"], model_identity={"model_fingerprint": "model-v1"}, human_reviewed=True)
    assert profile["human_reviewed"] is False
    assert validate_profile(profile, "model-v1")["temperature"] == profile["temperature"]
    with pytest.raises(ValueError, match="fingerprint"):
        validate_profile(profile, "different-model")
