"""Study integrity and handoff checks without downloading or training an encoder."""
import ast
import copy
import csv
import hashlib
import io
import json
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from training.muril_study import compare_runs, test_fingerprint as fingerprint, validate_baseline, write_report
from training.package_muril_study import build, notebook

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "colab/muril_study"


@pytest.fixture
def runs():
    contract = json.loads((STUDY / "study_contract.json").read_text())
    baseline = json.loads((STUDY / "xlmr_baseline_metrics.json").read_text())
    muril = copy.deepcopy(baseline)
    muril.update(base_model="google/muril-base-cased")
    for key in ("dataset_sha256", "split_sha256", "split_sizes"):
        muril[key] = contract["matched"][key]
    muril.update(contract["matched"]["settings"])
    return contract, baseline, muril


def test_comparison_uses_hindi_test_not_overall_score(runs):
    contract, baseline, muril = runs
    report = compare_runs(contract, baseline, muril)
    assert report["rows"][0]["Accuracy"] == baseline["test_by_language"]["hi-en"]["accuracy"]
    assert report["rows"][0]["Accuracy"] != baseline["test"]["accuracy"]
    assert [row["Training phrases"] for row in report["rows"]] == [16000, 8000]
    assert [row["Test phrases"] for row in report["rows"]] == [1000, 1000]
    assert len(report["per_class"]) == 14
    assert report["comparison_type"] == "same_test_different_training_scope"
    assert any("does not isolate" in note for note in report["notes"])
    assert any("synthetic" in note.lower() for note in report["notes"])


@pytest.mark.parametrize("mutation", ["dataset", "split", "settings", "labels", "scope", "rows", "support", "nan", "missing_language"])
def test_refuses_incompatible_or_invalid_saved_run(runs, mutation):
    contract, baseline, muril = runs
    if mutation == "dataset":
        muril["dataset_sha256"] = "changed"
    elif mutation == "split":
        muril["split_sha256"] = "changed"
    elif mutation == "settings":
        muril["seed"] = 999
    elif mutation == "labels":
        muril["labels"] = muril["labels"][::-1]
    elif mutation == "scope":
        muril["evaluation_scope"] = "human_annotated"
    elif mutation == "rows":
        muril["test_by_language"]["hi-en"]["rows"] = 999
    elif mutation == "support":
        muril["test_by_language"]["hi-en"]["per_class"]["joy"]["support"] += 1
    elif mutation == "nan":
        muril["test_by_language"]["hi-en"]["macro_f1"] = float("nan")
    else:
        del muril["test_by_language"]["hi-en"]
    with pytest.raises(ValueError):
        compare_runs(contract, baseline, muril)


def test_baseline_and_optional_matched_run_are_verified(runs):
    contract, baseline, muril = runs
    assert validate_baseline(contract, baseline)["rows"] == 1000
    matched = copy.deepcopy(muril)
    matched["base_model"] = "FacebookAI/xlm-roberta-base"
    report = compare_runs(contract, baseline, muril, matched)
    assert len(report["rows"]) == 3
    assert report["comparison_type"] == "same_test_with_matched_pair"
    matched["base_model"] = "google-bert/bert-base-multilingual-cased"
    with pytest.raises(ValueError):
        compare_runs(contract, baseline, muril, matched)
    baseline["test"]["accuracy"] = 0.1
    with pytest.raises(ValueError, match="baseline has changed"):
        validate_baseline(contract, baseline)


def test_reports_export_as_valid_csv_json_and_svg(runs, tmp_path):
    report = compare_runs(*runs)
    output = write_report(report, tmp_path / "report")
    assert json.loads((output / "comparison.json").read_text()) == report
    rows = list(csv.DictReader(io.StringIO((output / "comparison.csv").read_text(encoding="utf-8-sig"))))
    assert len(rows) == 2 and rows[1]["Model"] == "MuRIL Hindi-English"
    assert len(pd.read_csv(output / "per_emotion.csv")) == 14
    svg = ET.fromstring((output / "comparison.svg").read_text())
    assert svg.tag.endswith("svg")
    assert "synthetic" in (output / "READ_ME.txt").read_text().lower()


def test_fingerprint_binds_text_labels_and_ids_but_ignores_order_and_extra_columns():
    frame = pd.DataFrame([{"row_id": "a", "text": "accha laga", "emotion": "joy", "language_pair": "hi-en"},
                          {"row_id": "b", "text": "mood kharab", "emotion": "sadness", "language_pair": "hi-en"}])
    assert fingerprint(frame) == fingerprint(frame.iloc[::-1].assign(extra="ignored"))
    for column, value in (("text", "changed phrase"), ("emotion", "fear"), ("row_id", "different")):
        changed = frame.copy()
        changed.loc[0, column] = value
        assert fingerprint(frame) != fingerprint(changed)


def test_colab_notebook_is_executable_python_and_isolates_training_jobs():
    doc = notebook()
    code = "\n".join("".join(cell["source"]) for cell in doc["cells"] if cell["cell_type"] == "code")
    ast.parse(code)
    assert 'JOB = "muril_hi_en"' in code
    assert '"xlmr_hi_en_matched"' in code
    assert "torch.cuda.is_available()" in code and '"--resume"' in code
    assert '"--language-pair", "hi-en"' in code
    assert "contract[\"hi_en_test_sha256\"]" in code
    assert '"--allow-cpu"' not in code
    # Inference export happens before the separate aggregate report.
    assert code.index("export_checkpoint(model_dir") < code.index("report = compare_runs")


def test_built_archive_is_complete_verified_and_has_no_weights(tmp_path):
    output = tmp_path / "handoff"
    output.mkdir()
    guide = output / "START_HERE.md"
    guide.write_text("Keep the authored guide")
    result = build(output)
    assert result["baseline_verified"] and result["test_phrases"] == 1000
    assert guide.read_text() == "Keep the authored guide"
    assert not list(output.glob("muril_build_*"))
    with zipfile.ZipFile(result["zip"]) as archive:
        prefix = "hinglish_project_training/"
        names = archive.namelist()
        assert not any(name.endswith((".pt", ".safetensors", ".bin")) for name in names)
        manifest = json.loads(archive.read(prefix + "MANIFEST.json"))
        files = {name[len(prefix):] for name in names if not name.endswith("/MANIFEST.json")}
        assert files == set(manifest["sha256"])
        for name, digest in manifest["sha256"].items():
            assert hashlib.sha256(archive.read(prefix + name)).hexdigest() == digest
        assert prefix + "source/training/muril_study.py" in names
        frame = pd.read_csv(io.BytesIO(archive.read(prefix + "data/all_splits.csv")))
        hindi = frame[frame.language_pair.eq("hi-en")]
        assert hindi["split"].value_counts().to_dict() == {"train": 8000, "validation": 1000, "test": 1000}
        contract = json.loads(archive.read(prefix + "study/study_contract.json"))
        assert fingerprint(hindi[hindi.split.eq("test")]) == contract["hi_en_test_sha256"]
        bad_baseline = json.loads((STUDY / "xlmr_baseline_metrics.json").read_text())
        bad_baseline["split_sha256"] = "a different trained dataset"
        bad_path = tmp_path / "bad_metrics.json"
        bad_path.write_text(json.dumps(bad_baseline))
    with pytest.raises(ValueError, match="split_sha256"):
        build(tmp_path / "bad_handoff", bad_path)
