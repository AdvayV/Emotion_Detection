"""Integrity checks for the generated mBERT Colab handoff."""
import ast
import hashlib
import json
import os
import subprocess
import sys
import zipfile

import pytest

import training.package_mbert_study as packager
from training.package_mbert_study import (ROOT, SNAPSHOT_BASELINE, SNAPSHOT_CONTRACT,
                                          SNAPSHOT_METADATA, build, notebook, study_contract)

HAS_STUDY_INPUTS = (SNAPSHOT_BASELINE.is_file() and SNAPSHOT_METADATA.is_file() and SNAPSHOT_CONTRACT.is_file()
                    and (ROOT / "data/training/multilingual_bootstrap_v1/all_splits.csv").is_file())


def source_baseline():
    return (json.loads(SNAPSHOT_BASELINE.read_text(encoding="utf-8")),
            json.loads(SNAPSHOT_METADATA.read_text(encoding="utf-8")))


@pytest.mark.skipif(not HAS_STUDY_INPUTS, reason="requires the saved XLM-R baseline snapshots and frozen study data")
def test_baseline_snapshots_are_bound_to_saved_contract():
    baseline, metadata = source_baseline()
    contract = json.loads(SNAPSHOT_CONTRACT.read_text(encoding="utf-8"))
    assert packager._canonical_hash(baseline) == contract["baseline_metrics_sha256"]
    assert packager._canonical_hash(metadata) == contract["baseline_metadata_sha256"]
    assert metadata["dataset_sha256"] == contract["dataset_sha256"]
    assert metadata["split_sha256"] == contract["split_sha256"]


@pytest.mark.skipif(not HAS_STUDY_INPUTS, reason="requires the saved XLM-R baseline snapshots and frozen study data")
def test_study_contract_matches_installed_xlmr_data_and_settings():
    baseline, metadata = source_baseline()
    contract = study_contract(baseline, metadata)
    assert contract["split_sizes"] == {"train": 16000, "validation": 2000, "test": 2000}
    assert contract["dataset_sha256"] == metadata["dataset_sha256"]
    assert contract["split_sha256"] == metadata["split_sha256"]
    assert contract["settings"]["precision"] == "fp16"
    assert contract["cross_split_group_overlaps"] == 0
    assert contract["cross_split_template_family_overlaps"] == 0
    assert contract["cross_split_normalized_duplicate_overlaps"] == 0
    assert contract["independent_human_evaluation_set_available"] is False


@pytest.mark.skipif(not HAS_STUDY_INPUTS, reason="requires the saved XLM-R baseline snapshots and frozen study data")
@pytest.mark.parametrize("which", ["metrics", "metadata"])
def test_contract_rejects_mismatched_installed_baseline(which):
    baseline, metadata = source_baseline()
    target = baseline if which == "metrics" else metadata
    target["dataset_sha256"] = "changed"
    with pytest.raises(ValueError, match="dataset_sha256"):
        study_contract(baseline, metadata)


def test_notebook_python_is_valid_and_holds_the_combined_gpu_job():
    doc = notebook({})
    code = "\n".join("".join(cell["source"]) for cell in doc["cells"] if cell["cell_type"] == "code")
    ast.parse(code)
    assert 'JOB = "mbert_combined"' in code
    assert '"--models", "mbert"' in code
    assert '"--language-pair", "all"' in code
    assert '"--precision", "fp16"' in code
    assert '"--resume"' in code
    assert '"train": 16000, "validation": 2000, "test": 2000' in code
    assert "test_fingerprint(prepared[\"parts\"][2])" in code
    assert 'contract["baseline_metrics_sha256"]' in code
    assert 'contract["baseline_metadata_sha256"]' in code
    assert "def test_fingerprint(frame):" in code
    assert "independent human-labelled evaluation set" in "".join(doc["cells"][0]["source"]).lower()


@pytest.mark.skipif(not HAS_STUDY_INPUTS, reason="requires the saved XLM-R baseline snapshots and frozen study data")
def test_built_zip_matches_manifest_and_contains_no_weights(tmp_path, monkeypatch):
    output = tmp_path / "handoff"
    monkeypatch.setattr(packager, "BASELINE", tmp_path / "missing_model_metrics.json")
    monkeypatch.setattr(packager, "MODEL_METADATA", tmp_path / "missing_model_metadata.json")
    result = build(output)
    assert result["baseline_verified"]
    assert result["baseline_source"] == "snapshot"
    assert result["split_sizes"] == {"train": 16000, "validation": 2000, "test": 2000}
    assert (output / "TRAIN_MBERT_STUDY.ipynb").is_file()
    assert (output / "START_HERE.md").is_file()
    assert (output / "package" / "study" / "study_contract.json").is_file()
    assert (output / "xlmr_baseline_metadata.json").is_file()
    with zipfile.ZipFile(result["zip"]) as archive:
        prefix = "hinglish_project_training/"
        names = archive.namelist()
        assert not any(name.endswith((".pt", ".safetensors", ".bin")) for name in names)
        manifest = json.loads(archive.read(prefix + "MANIFEST.json"))
        files = {name[len(prefix):] for name in names if name != prefix + "MANIFEST.json"}
        assert files == set(manifest["sha256"])
        for name, digest in manifest["sha256"].items():
            assert hashlib.sha256(archive.read(prefix + name)).hexdigest() == digest
        contract = json.loads(archive.read(prefix + "study/study_contract.json"))
        assert contract["dataset_sha256"] == manifest_baseline_hash(contract, archive, prefix)
        assert prefix + "data/all_splits.csv" in names
        assert prefix + "source/training/package_project.py" not in names

    # Modules used by the Colab cells must import from the export itself, with
    # the repository checkout unavailable through cwd or PYTHONPATH.
    source = output / "package/source"
    isolated = tmp_path / "isolated_colab_cwd"
    isolated.mkdir()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(source)
    subprocess.run([sys.executable, "-c", (
        "import training.data_workflow, training.compare_encoders, training.cloud_bundle, "
        "training.install_checkpoint, training.train_transformer"
    )], cwd=isolated, env=env, check=True, capture_output=True, text=True)


def manifest_baseline_hash(contract, archive, prefix):
    baseline = json.loads(archive.read(prefix + "study/xlmr_baseline_metrics.json"))
    return baseline["dataset_sha256"]
