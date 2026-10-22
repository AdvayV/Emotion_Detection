"""Saved encoder metric checks; never opens or downloads model weights."""
from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
import sys
import tempfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.encoder_results import load_encoder_results


@pytest.fixture
def tmp_path():
    with tempfile.TemporaryDirectory() as directory:
        yield Path(directory)


def install_fixture(root: Path):
    files = [
        ("colab/mbert_study/study_contract.json", "colab/mbert_study/study_contract.json"),
        ("colab/mbert_study/xlmr_baseline_metrics.json", "colab/mbert_study/xlmr_baseline_metrics.json"),
        ("colab/mbert_study/xlmr_baseline_metrics.json", "models/xlmr_multilingual_emotion/metrics.json"),
        ("colab/mbert_study/completed_metrics.json", "models/mbert_multilingual_emotion/metrics.json"),
        ("colab/mbert_study/completed_metadata.json", "models/mbert_multilingual_emotion/emotion_metadata.json"),
    ]
    for source_relative, target_relative in files:
        source = ROOT / source_relative
        target = root / target_relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def test_installed_mbert_and_xlmr_metrics_verify_without_weights(tmp_path):
    install_fixture(tmp_path)
    result = load_encoder_results(tmp_path)
    assert result["status"] == "verified"
    assert result["mbert"]["rows"] == 2000
    assert result["mbert"]["accuracy"] == 0.814
    assert result["mbert_by_language"]["hi-en"]["accuracy"] == 0.843
    assert result["mbert_by_language"]["fr-en"]["macro_f1"] == pytest.approx(0.770796074268007)
    assert result["baseline_by_language"]["hi-en"]["rows"] == 1000


@pytest.mark.parametrize("mutation", ["wrong_split", "wrong_metadata", "bad_support", "bad_matrix", "bool_score"])
def test_corrupted_mbert_run_is_refused(tmp_path, mutation):
    install_fixture(tmp_path)
    metrics_path = tmp_path / "models/mbert_multilingual_emotion/metrics.json"
    metrics = read_json(metrics_path)
    metadata_path = tmp_path / "models/mbert_multilingual_emotion/emotion_metadata.json"
    metadata = read_json(metadata_path)
    if mutation == "wrong_split":
        metrics["split_sha256"] = "changed"
    elif mutation == "wrong_metadata":
        metadata["split_sha256"] = "changed"
        write_json(metadata_path, metadata)
        metrics["emotion_metadata"] = metadata
    elif mutation == "bad_support":
        metrics["test"]["per_class"]["joy"]["support"] += 1
    elif mutation == "bad_matrix":
        metrics["test_by_language"]["hi-en"]["confusion_matrix"]["joy"]["joy"] += 1
    else:
        metrics["test"]["accuracy"] = True
    write_json(metrics_path, metrics)
    with pytest.raises(ValueError):
        load_encoder_results(tmp_path)


def test_missing_mbert_metrics_remain_pending(tmp_path):
    install_fixture(tmp_path)
    (tmp_path / "models/mbert_multilingual_emotion/metrics.json").unlink()
    result = load_encoder_results(tmp_path)
    assert result["status"] == "pending"
    assert "not installed" in result["message"]
