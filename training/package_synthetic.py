"""Create a self-contained Colab package with 8,000 synthetic training phrases."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

from training.synthetic_phrases import COUNTS, save_dataset

ROOT = Path(__file__).resolve().parents[1]
DATASET_FILES = ("train.csv", "validation.csv", "test.csv", "all_splits.csv", "dataset_audit.json", "DATASET_CARD.md")


def snapshot_dataset(destination, dataset_dir=None):
    """Package the working CSVs without regenerating or silently replacing them."""
    destination = Path(destination)
    if dataset_dir is None:
        return save_dataset(destination)
    source = Path(dataset_dir).resolve()
    if not source.exists():
        save_dataset(source)
    if not all((source / name).is_file() for name in DATASET_FILES):
        raise ValueError("Working dataset is incomplete; preserve it and restore its missing files before packaging")
    report = json.loads((source / "dataset_audit.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256((source / "all_splits.csv").read_bytes()).hexdigest()
    if report.get("dataset_kind") != "synthetic_template" or report.get("prepared_csv_sha256") != digest:
        raise ValueError("Working dataset audit does not match its CSV; do not overwrite or silently relabel the data")
    with (source / "all_splits.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != sum(COUNTS.values()) or report.get("split_sizes") != COUNTS:
        raise ValueError("The 8000-phrase package requires its fixed 8000/1000/1000 partitions")
    for split, count in COUNTS.items():
        with (source / f"{split}.csv").open(encoding="utf-8", newline="") as handle:
            part = list(csv.DictReader(handle))
        if len(part) != count or part != [row for row in rows if row["split"] == split]:
            raise ValueError(f"Working {split}.csv does not match all_splits.csv")
    destination.mkdir(parents=True, exist_ok=True)
    for name in DATASET_FILES:
        if (source / name).resolve() != (destination / name).resolve():
            shutil.copyfile(source / name, destination / name)
    return report


def notebook(package_name):
    cells = []
    def add(kind, source):
        cell = {"cell_type": kind, "id": f"step_{len(cells):02d}", "metadata": {}, "source": source.splitlines(keepends=True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    add("markdown", """# Train on 8,000 generated Hinglish phrases

This package contains exactly 8,000 training rows, 1,000 validation rows and
1,000 synthetic test rows. Labels are generated from original templates and
have not been independently reviewed. Scores measure synthetic examples only.
No MaSaC data, publisher approval claim or human-review claim is included.

Run cells one at a time in a fresh Google Colab GPU runtime.
The dataset and current project source are already inside the package ZIP.
""")
    add("code", """from pathlib import Path
import hashlib
import json
import zipfile
from google.colab import files

uploaded = files.upload()  # Select colab_ready_8000.zip
archives = [name for name in uploaded if name.lower().endswith(".zip")]
assert len(archives) == 1, "Upload exactly one final package ZIP."
destination = Path("/content").resolve()
with zipfile.ZipFile(archives[0]) as archive:
    assert sum(item.file_size for item in archive.infolist()) < 64 * 1024 * 1024
    for item in archive.infolist():
        assert (destination / item.filename).resolve().is_relative_to(destination)
        assert (item.external_attr >> 16) & 0o170000 != 0o120000
    archive.extractall(destination)
PACKAGE = destination / """ + repr(package_name) + """
manifest = json.loads((PACKAGE / "PACKAGE_MANIFEST.json").read_text())
for relative, expected in manifest["sha256"].items():
    path = (PACKAGE / relative).resolve()
    assert path.is_relative_to(PACKAGE.resolve())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, relative
print("Package files verified:", len(manifest["sha256"]))
""")
    add("markdown", "## Install the model dependencies and check the GPU\nChoose Runtime → Change runtime type → GPU, selecting T4 if offered. If none is assigned, wait or use another free notebook provider. No paid AWS service is needed.\n")
    add("code", """import os
import subprocess
import sys

PROJECT = PACKAGE / "source"
os.chdir(PROJECT)
subprocess.run([sys.executable, "-m", "pip", "install", "-e", ".[model]"], check=True)
sys.path.insert(0, str(PROJECT))
import torch
assert torch.cuda.is_available(), "Choose a GPU runtime before training."
print("Assigned GPU:", torch.cuda.get_device_name(0))
""")
    add("markdown", "## Persist checkpoints in your own Drive\nThe latest completed epoch can be recovered after a disconnected runtime. Keep sufficient storage for model weights and optimizer state. This notebook initially trains XLM-R only to conserve your free GPU allocation.\n")
    add("code", """from google.colab import drive
drive.mount("/content/drive")
STORAGE = Path("/content/drive/MyDrive/HinglishEmotionSynthetic8000")
STORAGE.mkdir(parents=True, exist_ok=True)
MODELS = ["xlmr"]  # Change to ["xlmr", "muril"] for the matched two-model experiment.
""")
    add("markdown", "## Confirm the exact split counts\nThe combined CSV is the trainer's input. Its split column keeps validation and test phrases out of weight updates. Synthetic training is explicitly opted into; that does not imply human validation.\n")
    add("code", """from training.data_workflow import prepare_data
DATA = PACKAGE / "data" / "all_splits.csv"
AUDIT = PACKAGE / "data" / "dataset_audit.json"
MANIFEST = STORAGE / "frozen_split.csv"
prepared = prepare_data(DATA, "emotion", "raw", 42, MANIFEST, AUDIT, "hi-en", allow_synthetic=True)
sizes = [len(part) for part in prepared["parts"]]
assert sizes == [8000, 1000, 1000], sizes
assert prepared["validated_language_pairs"] == []
print("Training / validation / synthetic test:", sizes)
print("Scope:", prepared["evaluation_scope"])
""")
    add("markdown", "## Fine-tune the model\nStart with three epochs, batch size 2 and accumulation 8. This usually combines 16 phrases per optimizer update. The checkpoint with the best validation macro-F1 is saved. Runtime depends on GPU availability and throughput. Re-run this cell after reconnecting with unchanged data/settings to resume the latest completed epoch.\n")
    add("code", """MODEL_ROOT = STORAGE / "models"
command = [
    sys.executable, "-m", "training.compare_encoders",
    "--task", "emotion", "--models", *MODELS,
    "--data", str(DATA), "--data-audit", str(AUDIT),
    "--output-root", str(MODEL_ROOT), "--language-pair", "hi-en",
    "--split-manifest", str(MANIFEST), "--epochs", "3",
    "--batch-size", "2", "--gradient-accumulation", "8",
    "--max-length", "128", "--learning-rate", "2e-5",
    "--precision", "auto", "--text-mode", "raw",
    "--seed", "42", "--split-seed", "42",
    "--allow-synthetic", "--resume",
]
subprocess.run(command, cwd=PROJECT, check=True)
""")
    add("markdown", "## Inspect synthetic-only results\nHigh scores here do not establish real-world emotion recognition. Inspect each emotion's support and the lexical baseline. A human-labelled independent test set is still required for research claims.\n")
    add("code", """import pandas as pd
display(pd.read_csv(MODEL_ROOT / "emotion_encoder_comparison.csv"))
for key in MODELS:
    metrics = json.loads((MODEL_ROOT / f"{key}_emotion" / "metrics.json").read_text())
    print(key, "evaluation scope:", metrics["evaluation_scope"])
    display(pd.DataFrame(metrics["test"]["per_class"]).T)
""")
    add("markdown", "## Export to your laptop\nThe ZIP excludes optimizer state and raw dataset phrases. Extract the model folder into your VIT project's models folder, then select it in Emotion lab. The dashboard will identify its synthetic-only, experimental scope.\n")
    add("code", """from training.cloud_bundle import export_checkpoint
for key in MODELS:
    path = export_checkpoint(MODEL_ROOT / f"{key}_emotion", STORAGE / f"{key}_emotion.zip")
    print("Saved in Drive:", path)
files.download(str(STORAGE / f"{MODELS[0]}_emotion.zip"))
""")
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}, "colab": {"name": "TRAIN_ON_COLAB.ipynb", "provenance": []}}, "nbformat": 4, "nbformat_minor": 5}


def build(output_dir, overwrite=False, dataset_dir=None):
    output = Path(output_dir).resolve()
    if output.exists() and not overwrite:
        raise ValueError("Output folder exists; use --overwrite to refresh only the known package files")
    output.mkdir(parents=True, exist_ok=True)
    report = snapshot_dataset(output / "data", dataset_dir)
    paths = [output / "data" / name for name in DATASET_FILES]
    source = output / "source"
    for directory in ("hinglish_emotion", "training"):
        target = source / directory
        target.mkdir(parents=True, exist_ok=True)
        for file in sorted((ROOT / directory).glob("*.py")):
            copied = target / file.name
            shutil.copyfile(file, copied)
            paths.append(copied)
    shutil.copyfile(ROOT / "pyproject.toml", source / "pyproject.toml")
    paths.append(source / "pyproject.toml")
    notebook_path = output / "TRAIN_ON_COLAB.ipynb"
    notebook_path.write_text(json.dumps(notebook(output.name), indent=2, ensure_ascii=False), encoding="utf-8")
    paths.append(notebook_path)
    instructions = """# Colab-ready experimental training package

## What is included

| File | Rows |
| --- | ---: |
| data/train.csv | 8,000 training phrases |
| data/validation.csv | 1,000 validation phrases |
| data/test.csv | 1,000 synthetic test phrases |
| data/all_splits.csv | All 10,000 rows with fixed partitions |

Seven emotions: joy, sadness, anger, fear, surprise, disgust, neutral.
Language: Hindi–English. French–English is not part of this package.
Labels are generated template-intent labels, not human annotations. This is an
experimental alternative to the pending human-reviewed research dataset.
The 10,000 rows are compositional variants, not 10,000 independently authored
natural utterances. Read data/DATASET_CARD.md and data/dataset_audit.json.

In the local project, the working datasets are under
data/training/synthetic_hinglish_8000. This upload folder contains a verified
snapshot of those files. Keep demo data separate from training data.

## Exact upload steps

1. Open https://colab.research.google.com/ and sign in.
2. File → Upload notebook → select TRAIN_ON_COLAB.ipynb from this folder.
3. Runtime → Change runtime type → GPU; select T4 if offered.
4. Run the first code cell, then upload the provided colab_ready_8000.zip.
5. Run each following cell in order. Dependencies and data are already specified.
6. Mount your own Drive when prompted. Leave MODELS = ["xlmr"] for the first run.
7. Training uses 8,000 phrases; validation/test rows do not update model weights.
8. Download xlmr_emotion.zip from the final cell or from your Drive.
9. Extract xlmr_emotion into your VIT project's models folder and select XLM-R.

Set MODELS = ["xlmr", "muril"] in the notebook to train the matched comparison
sequentially. Free GPU resources are availability-limited. No AWS paid plan or
MaSaC permission file is required: no external dataset was used in this package.

Three epochs mean 24,000 training phrase exposures per model, still only 8,000
distinct training rows. Keep checkpoints on Drive; reconnect and rerun setup with
the same package and settings before resuming an interrupted run.

## What the scores mean

Synthetic validation/test accuracy measures these generated templates only.
It is not research accuracy, evidence of human annotation, or proof of semantic
understanding. The exported checkpoint retains synthetic-only scope and the
dashboard warns that real-world performance is unverified.

The explicit --allow-synthetic option is intentional. Original demo and MaSaC
reuse safeguards remain separate. Replace or supplement this set with properly
reviewed real data for the eventual research experiment.
"""
    start = output / "START_HERE.md"
    start.write_text(instructions, encoding="utf-8")
    paths.append(start)
    manifest = {"package": output.name, "training_phrases": 8000, "data_kind": "synthetic_only", "sha256": {file.relative_to(output).as_posix(): hashlib.sha256(file.read_bytes()).hexdigest() for file in paths}}
    manifest_path = output / "PACKAGE_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    paths.append(manifest_path)
    archive_path = output.with_suffix(".zip")
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(paths):
            archive.write(file, output.name + "/" + file.relative_to(output).as_posix())
    return output, archive_path, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="colab/colab_ready_8000")
    parser.add_argument("--dataset-dir", default="data/training/synthetic_hinglish_8000", help="Working dataset folder; generated only if it does not exist")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    folder, archive, report = build(args.output_dir, args.overwrite, args.dataset_dir)
    print(json.dumps({"folder": str(folder), "zip": str(archive), "split_sizes": report["split_sizes"], "source": "synthetic_only"}, indent=2))


if __name__ == "__main__":
    main()
