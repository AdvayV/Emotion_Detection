"""Quality and leakage-contract checks for assistant-authored challenge drafts."""
import csv
import json
from pathlib import Path

import pytest

from hinglish_emotion.evaluation_guard import phrase_hash
from hinglish_emotion.tasks import EMOTION_LABELS
from training.create_challenge_drafts import OUT, ROOT, build

TRAINING = ROOT / "data/training/multilingual_bootstrap_v1/all_splits.csv"
HAS_AUDIT_INPUTS = TRAINING.is_file() and any((ROOT / "models").glob("*/evaluation_guard.json"))


def read_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


@pytest.mark.parametrize(("filename", "pair"), [
    ("hindi_english_100.csv", "hi-en"),
    ("english_french_100.csv", "fr-en"),
])
def test_labelled_drafts_have_100_diverse_unreviewed_rows(filename, pair):
    rows = read_rows(OUT / filename)
    assert len(rows) == 100
    assert {row["language_pair"] for row in rows} == {pair}
    assert {row["emotion"] for row in rows} == set(EMOTION_LABELS)
    assert all(row["annotation_status"] == "assistant_draft" for row in rows)
    assert all(row["source"] == "assistant_draft" and row["human_reviewed"] == "false" for row in rows)
    assert len({phrase_hash(row["text"]) for row in rows}) == 100


@pytest.mark.parametrize(("filename", "pair"), [
    ("hindi_english_100_unlabelled.csv", "hi-en"),
    ("english_french_100_unlabelled.csv", "fr-en"),
])
def test_unlabelled_prediction_copies_have_blank_emotion(filename, pair):
    rows = read_rows(OUT / filename)
    assert len(rows) == 100
    assert {row["language_pair"] for row in rows} == {pair}
    assert all(not row["emotion"] for row in rows)
    assert all(row["human_reviewed"] == "false" for row in rows)


@pytest.mark.skipif(not HAS_AUDIT_INPUTS, reason="requires the frozen training CSV and installed model evaluation guards")
def test_overlap_audit_covers_all_training_rows_and_installed_guards(tmp_path):
    report = build(tmp_path / "drafts")
    assert report["training_source_rows"] == 20000
    assert len(report["installed_model_guards"]) >= 1
    assert "does not establish independent authorship" in report["warning"]
    for name, item in report["files"].items():
        if name.endswith("_unlabelled.csv"):
            continue
        assert item["rows"] == 100
        assert item["training_overlap_rows"] == 0
        assert item["guard_overlap_rows"]
        assert all(value == 0 for value in item["guard_overlap_rows"].values())
        assert len(item["sha256"]) == 64


@pytest.mark.skipif(not HAS_AUDIT_INPUTS, reason="requires the frozen training CSV and installed model evaluation guards")
def test_committed_audit_records_zero_overlap_for_every_model():
    report = json.loads((OUT / "overlap_audit.json").read_text(encoding="utf-8"))
    assert report["training_source_rows"] == 20000
    assert report["installed_model_guards"]
    for name, item in report["files"].items():
        if name.endswith("_unlabelled.csv"):
            continue
        assert item["training_overlap_rows"] == 0
        assert set(item["guard_overlap_rows"]) == set(report["installed_model_guards"])
        assert all(value == 0 for value in item["guard_overlap_rows"].values())
