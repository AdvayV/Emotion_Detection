"""Build a Colab handoff for mBERT on the installed combined XLM-R split."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

from hinglish_emotion.tasks import EMOTION_LABELS
from training.data_workflow import prepare_data
from training.package_project import PACKAGE_NAME, ROOT, build as build_project, notebook as project_notebook

OUT = ROOT / "colab/mbert_study"
BASELINE = ROOT / "models/xlmr_multilingual_emotion/metrics.json"
MODEL_METADATA = ROOT / "models/xlmr_multilingual_emotion/emotion_metadata.json"
SNAPSHOT_BASELINE = OUT / "xlmr_baseline_metrics.json"
SNAPSHOT_METADATA = OUT / "xlmr_baseline_metadata.json"
SNAPSHOT_CONTRACT = OUT / "study_contract.json"
SETTINGS = ("text_mode", "preparation_version", "max_length", "seed", "split_seed", "batch_size",
            "gradient_accumulation", "learning_rate", "class_weights", "freeze_encoder", "precision",
            "patience", "training_mode", "training_data_kind", "allow_synthetic")


def test_fingerprint(frame):
    """Fingerprint held-out identities, text, labels, and language pairs."""
    rows = sorted((str(row.row_id), str(row.text), str(row.emotion), str(row.language_pair))
                  for row in frame.itertuples(index=False))
    return hashlib.sha256(json.dumps(rows, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def _canonical_hash(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def resolve_baselines(baseline_path=None, metadata_path=None):
    """Load installed baseline files or verified, small metadata snapshots."""
    baseline_source = Path(baseline_path) if baseline_path is not None else (BASELINE if BASELINE.is_file() else SNAPSHOT_BASELINE)
    metadata_source = Path(metadata_path) if metadata_path is not None else (MODEL_METADATA if MODEL_METADATA.is_file() else SNAPSHOT_METADATA)
    if not baseline_source.is_file():
        raise FileNotFoundError(f"No installed XLM-R metrics or baseline snapshot: {baseline_source}")
    if not metadata_source.is_file():
        raise FileNotFoundError(f"No installed XLM-R metadata or baseline snapshot: {metadata_source}")
    baseline = json.loads(baseline_source.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_source.read_text(encoding="utf-8"))
    using_snapshot = baseline_source.resolve() == SNAPSHOT_BASELINE.resolve() or metadata_source.resolve() == SNAPSHOT_METADATA.resolve()
    if using_snapshot:
        if not SNAPSHOT_CONTRACT.is_file():
            raise FileNotFoundError(f"Cannot verify baseline snapshots without {SNAPSHOT_CONTRACT}")
        contract = json.loads(SNAPSHOT_CONTRACT.read_text(encoding="utf-8"))
        if _canonical_hash(baseline) != contract.get("baseline_metrics_sha256"):
            raise ValueError("Baseline metrics snapshot hash differs from the saved mBERT study contract")
        if _canonical_hash(metadata) != contract.get("baseline_metadata_sha256"):
            raise ValueError("Baseline metadata snapshot hash differs from the saved mBERT study contract")
        if (baseline.get("dataset_sha256") != contract.get("dataset_sha256")
                or metadata.get("dataset_sha256") != contract.get("dataset_sha256")
                or baseline.get("split_sha256") != contract.get("split_sha256")
                or metadata.get("split_sha256") != contract.get("split_sha256")):
            raise ValueError("Baseline snapshots do not match the saved dataset and split contract")
    return baseline, metadata, using_snapshot


def study_contract(baseline, metadata):
    data = ROOT / "data/training/multilingual_bootstrap_v1"
    prepared = prepare_data(data / "all_splits.csv", "emotion", "raw", 42,
                           data_audit=data / "dataset_audit.json", allow_synthetic=True)
    sizes = dict(zip(("train", "validation", "test"), (len(part) for part in prepared["parts"])))
    if sizes != {"train": 16000, "validation": 2000, "test": 2000}:
        raise ValueError(f"Expected the frozen 16,000/2,000/2,000 combined split, got {sizes}")
    if tuple(baseline.get("labels", ())) != tuple(EMOTION_LABELS):
        raise ValueError("Installed XLM-R baseline has unexpected emotion labels")
    for key in ("dataset_sha256", "split_sha256"):
        expected = prepared["dataset_sha256"] if key == "dataset_sha256" else prepared["manifest_sha256"]
        if baseline.get(key) != expected or metadata.get(key) != expected:
            raise ValueError(f"Installed XLM-R {key} does not match the current frozen dataset")
    if metadata.get("base_model") != "FacebookAI/xlm-roberta-base" or metadata.get("task") != "emotion":
        raise ValueError("Installed checkpoint is not the expected combined XLM-R emotion baseline")
    if metadata.get("training_status") != "complete" or metadata.get("labels") != list(EMOTION_LABELS):
        raise ValueError("Installed XLM-R metadata is incomplete or incompatible")
    required = dict(text_mode="raw", preparation_version=1, max_length=128, seed=42, split_seed=42,
                    batch_size=2, gradient_accumulation=8, learning_rate=2e-5, class_weights="balanced",
                    freeze_encoder=False, precision="fp16", patience=2, training_mode="full_fine_tuning",
                    training_data_kind="synthetic_only", allow_synthetic=True)
    actual = {key: baseline.get(key) for key in SETTINGS}
    if actual != required:
        raise ValueError("Installed XLM-R training settings differ from the declared mBERT study settings")
    frame = prepared["parts"][2]
    manifest = prepared["manifest"]
    cross_split_overlap = lambda column: int((manifest.groupby(column)["split"].nunique() > 1).sum())
    from hinglish_emotion.text_preparation import duplicate_key
    manifest = manifest.assign(_duplicate_key=manifest["text"].map(duplicate_key))
    return {
        "schema_version": 1,
        "job": "mbert_combined",
        "base_model": "google-bert/bert-base-multilingual-cased",
        "baseline_model": "FacebookAI/xlm-roberta-base",
        "labels": list(EMOTION_LABELS),
        "dataset_sha256": prepared["dataset_sha256"],
        "split_sha256": prepared["manifest_sha256"],
        "split_sizes": sizes,
        "test_sha256": test_fingerprint(frame),
        "test_support": {label: int(frame.emotion.eq(label).sum()) for label in EMOTION_LABELS},
        "baseline_metrics_sha256": hashlib.sha256(json.dumps(baseline, sort_keys=True).encode()).hexdigest(),
        "baseline_metadata_sha256": hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest(),
        "settings": required,
        "synthetic_data": True,
        "independent_human_evaluation_set_available": False,
        "cross_split_group_overlaps": cross_split_overlap("group_id"),
        "cross_split_scenario_family_overlaps": cross_split_overlap("scenario_family"),
        "cross_split_template_family_overlaps": cross_split_overlap("template_family"),
        "cross_split_normalized_duplicate_overlaps": cross_split_overlap("_duplicate_key"),
    }


def notebook(contract):
    cells = []

    def add(kind, source):
        cell = {"cell_type": kind, "id": f"mbert_{len(cells):02d}", "metadata": {}, "source": source.splitlines(keepends=True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    add("markdown", """# mBERT and XLM-R: combined multilingual emotion run

Fine-tune mBERT on Google Colab with a GPU using the same seven emotion labels, Hindi-English plus English-French data, and exact 16,000/2,000/2,000 frozen split as the installed combined XLM-R model. This is supervised classifier fine-tuning, not language-model pretraining. Select **Runtime > Change runtime type > T4 GPU**, then run all cells and upload `mbert_study.zip` when prompted.

The dataset is synthetic and template-based. The held-out test split is intentionally the same one used by XLM-R, so this is a paired model comparison, not an independent evaluation. No independent human-labelled evaluation set is present in the project. Computed overlap counts report zero cross-split group, template-family and normalized-duplicate overlaps; this does not establish statistical independence. Inspect the bundled audit before interpreting results.
""")
    add("code", "".join(project_notebook()["cells"][1]["source"]))
    add("code", '''JOB = "mbert_combined"
EPOCHS = 3
assert JOB == "mbert_combined" and EPOCHS == 3
drive.mount("/content/drive")
STORAGE = Path("/content/drive/MyDrive/HinglishEmotionProject/MBERTStudy")
run = STORAGE / JOB
run.mkdir(parents=True, exist_ok=True)
DATA = PACKAGE / "data/all_splits.csv"
AUDIT = PACKAGE / "data/dataset_audit.json"
contract = json.loads((PACKAGE / "study/study_contract.json").read_text())
baseline = json.loads((PACKAGE / "study/xlmr_baseline_metrics.json").read_text())
baseline_metadata = json.loads((PACKAGE / "study/xlmr_baseline_metadata.json").read_text())
assert hashlib.sha256(json.dumps(baseline, sort_keys=True).encode()).hexdigest() == contract["baseline_metrics_sha256"]
assert hashlib.sha256(json.dumps(baseline_metadata, sort_keys=True).encode()).hexdigest() == contract["baseline_metadata_sha256"]
assert baseline_metadata["training_status"] == "complete"
assert baseline_metadata["task"] == "emotion" and baseline_metadata["labels"] == contract["labels"]
assert baseline_metadata["dataset_sha256"] == contract["dataset_sha256"]
assert baseline_metadata["split_sha256"] == contract["split_sha256"]
''')
    add("markdown", """## Verify the baseline and freeze the split

The notebook checks the dataset hash, split hash, test identities, labels, split sizes and training settings against the bundled XLM-R metrics and metadata snapshots before training. The current data must match exactly. It then saves the frozen split manifest to Drive, so a resumed run cannot silently use a different split.
""")
    add("code", '''from training.data_workflow import prepare_data
def test_fingerprint(frame):
    rows = sorted((str(row.row_id), str(row.text), str(row.emotion), str(row.language_pair))
                  for row in frame.itertuples(index=False))
    return hashlib.sha256(json.dumps(rows, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
prepared = prepare_data(DATA, "emotion", "raw", 42, run / "frozen_split.csv", AUDIT,
                        "all", allow_synthetic=True)
assert prepared["dataset_sha256"] == contract["dataset_sha256"], "Dataset changed"
assert prepared["manifest_sha256"] == contract["split_sha256"], "Frozen split changed"
assert test_fingerprint(prepared["parts"][2]) == contract["test_sha256"], "Held-out test set changed"
sizes = dict(zip(("train", "validation", "test"), (len(part) for part in prepared["parts"])))
assert sizes == contract["split_sizes"] == {"train": 16000, "validation": 2000, "test": 2000}
print("Verified data and splits:", sizes)
''')
    add("markdown", """## Train mBERT on the GPU

The run uses raw text, max length 128, batch size 2, gradient accumulation 8, learning rate 2e-5, seed 42, balanced class weights and fp16. It fine-tunes the full encoder and emotion head for up to three epochs; validation macro-F1 selects the checkpoint with patience 2. Drive stores the frozen split and epoch recovery state. If Colab disconnects, reopen this notebook, upload the same ZIP, keep the same job folder, and run all cells again to resume.

Model cards: [mBERT](https://huggingface.co/google-bert/bert-base-multilingual-cased), [XLM-R](https://huggingface.co/FacebookAI/xlm-roberta-base). CPU training is disabled.
""")
    add("code", '''command = [sys.executable, "-u", "-m", "training.compare_encoders",
    "--task", "emotion", "--models", "mbert", "--data", str(DATA), "--language-pair", "all",
    "--output-root", str(run / "models"), "--split-manifest", str(run / "frozen_split.csv"),
    "--epochs", str(EPOCHS), "--batch-size", "2", "--gradient-accumulation", "8",
    "--max-length", "128", "--learning-rate", "2e-5", "--precision", "fp16",
    "--text-mode", "raw", "--seed", "42", "--split-seed", "42", "--resume",
    "--data-audit", str(AUDIT), "--allow-synthetic"]
subprocess.run(command, cwd=PROJECT, check=True)
''')
    add("markdown", """## Compare and export

The comparison cell refuses metrics from another dataset, split, label set, model, or setting. Accuracy is the share of correct predictions; macro-F1 gives equal weight to each emotion. Both scores are from the shared synthetic test set and do not establish real-world performance. Test scores must not be used to select training settings.
""")
    add("code", '''from training.cloud_bundle import export_checkpoint
from IPython.display import display
import pandas as pd
model_dir = run / "models/mbert_emotion"
metrics = json.loads((model_dir / "metrics.json").read_text())
assert metrics["base_model"] == contract["base_model"]
assert metrics["dataset_sha256"] == contract["dataset_sha256"]
assert metrics["split_sha256"] == contract["split_sha256"]
assert metrics["labels"] == contract["labels"]
assert metrics["split_sizes"] == contract["split_sizes"]
for key, value in contract["settings"].items():
    assert metrics[key] == value, f"Training setting changed: {key}"
assert test_fingerprint(prepared["parts"][2]) == contract["test_sha256"]
rows = []
for name, item in (("XLM-R baseline", baseline), ("mBERT", metrics)):
    rows.append({"model": name, "best_validation_macro_f1": item["best_validation_macro_f1"],
                 "test_accuracy": item["test"]["accuracy"], "test_macro_f1": item["test"]["macro_f1"],
                 "test_rows": item["split_sizes"]["test"]})
display(pd.DataFrame(rows))
print("Shared synthetic test set; not an independent human evaluation.")
export_path = STORAGE / "exports/mbert_combined.zip"
export_checkpoint(model_dir, export_path)
print("Download this inference checkpoint:", export_path)
files.download(str(export_path))
''')
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "colab": {"name": "TRAIN_MBERT_STUDY.ipynb", "provenance": []}}, "nbformat": 4, "nbformat_minor": 5}


def build(output=None, baseline_path=None, metadata_path=None):
    output = Path(output or OUT).resolve()
    baseline, metadata, using_snapshot = resolve_baselines(baseline_path, metadata_path)
    contract = study_contract(baseline, metadata)
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mbert_build_", dir=output) as temporary:
        scratch = Path(temporary)
        build_project(scratch)
        package = scratch / "package"
        study = package / "study"
        study.mkdir()
        baseline_text = json.dumps(baseline, indent=2)
        metadata_text = json.dumps(metadata, indent=2)
        contract_text = json.dumps(contract, indent=2)
        (study / "study_contract.json").write_text(contract_text, encoding="utf-8")
        (study / "xlmr_baseline_metrics.json").write_text(baseline_text, encoding="utf-8")
        (study / "xlmr_baseline_metadata.json").write_text(metadata_text, encoding="utf-8")
        manifest = json.loads((package / "MANIFEST.json").read_text(encoding="utf-8"))
        for file in study.iterdir():
            manifest["sha256"][file.relative_to(package).as_posix()] = hashlib.sha256(file.read_bytes()).hexdigest()
        (package / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        final_package = output / "package"
        final_package.mkdir(exist_ok=True)
        for file in package.rglob("*"):
            if file.is_file():
                destination = final_package / file.relative_to(package)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(file, destination)
        archive_path = output / "mbert_study.zip"
        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
            for rel in sorted(manifest["sha256"]):
                archive.write(final_package / rel, PACKAGE_NAME + "/" + rel)
            archive.write(final_package / "MANIFEST.json", PACKAGE_NAME + "/MANIFEST.json")
    (output / "study_contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    (output / "xlmr_baseline_metrics.json").write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    (output / "xlmr_baseline_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output / "TRAIN_MBERT_STUDY.ipynb").write_text(json.dumps(notebook(contract), indent=2, ensure_ascii=False), encoding="utf-8")
    guide = """# mBERT combined Colab study

1. In a fresh clone, regenerate the deterministic multilingual bootstrap data from the project root: `python -c "from training.multilingual_data import save_multilingual; save_multilingual('data/training/multilingual_bootstrap_v1')"`.
2. Rebuild this handoff with `python -m training.package_mbert_study`. The builder uses installed XLM-R metrics/metadata when available, or verifies the small snapshots in this folder against `study_contract.json` when model files are absent.
3. Upload `TRAIN_MBERT_STUDY.ipynb` to Google Colab, select **Runtime > Change runtime type > T4 GPU**, and run all cells.
4. Upload `mbert_study.zip` at the first prompt. The notebook path-checks extraction, verifies every file hash, installs the project dependencies, and confirms CUDA is enabled.
5. The default job fine-tunes mBERT on the existing combined Hindi-English and English-French seven-emotion dataset. It requires the exact frozen 16,000 train / 2,000 validation / 2,000 test split and SHA-256 hashes recorded by the XLM-R baseline.
6. Approve the Google Drive mount. Frozen split, model checkpoint and resume state save under `MyDrive/HinglishEmotionProject/MBERTStudy/mbert_combined`. Reopen the same notebook and upload the same ZIP to resume after a runtime disconnect.
7. The last notebook cell compares validation-selected mBERT against the combined XLM-R baseline on the same held-out synthetic test examples, then downloads `mbert_combined.zip`. From the project root, install it with `python -m training.install_checkpoint "C:/path/to/mbert_combined.zip"`.

The phrases are synthetic and template-based. The test split is shared intentionally with XLM-R; it is not an independent evaluation. The project contains no separate human-labelled evaluation set. The dataset audit reports zero cross-split group and template-family overlap. Report the synthetic limitation whenever sharing scores. Do not tune hyperparameters against test results.
"""
    (output / "START_HERE.md").write_text(guide, encoding="utf-8")
    return {"notebook": str(output / "TRAIN_MBERT_STUDY.ipynb"), "zip": str(output / "mbert_study.zip"),
            "package": str(output / "package"), "split_sizes": contract["split_sizes"],
            "baseline_verified": True, "baseline_source": "snapshot" if using_snapshot else "installed model metadata"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--metadata", type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.output, args.baseline, args.metadata), indent=2))
