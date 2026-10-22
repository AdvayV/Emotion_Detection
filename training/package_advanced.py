"""Build the standalone advanced XLM-R Colab handoff without fabricating weights."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "xlmr_advanced_training"
DATA_FILES = ("all_splits.csv", "train.csv", "validation.csv", "test.csv", "dataset_audit.json", "DATASET_CARD.md")

GUIDE = """# Optional advanced XLM-R training

1. Upload TRAIN_ADVANCED.ipynb to https://colab.research.google.com/.
2. Select Runtime > Change runtime type > GPU, then run the notebook cells.
3. Upload xlmr_advanced_training.zip from this folder at the first prompt.
4. The default experiment trains an actual learned text-emoji gate using XLM-R,
   on both Hindi-English and French-English. GPU training happens in your runtime.
5. Mount your own Drive for completed-epoch recovery. Keep EPOCHS=3 initially.
6. Inspect per-language and per-emotion support and scores. Download the inference ZIP.
7. Install from the laptop project root:
   python -m training.install_checkpoint "C:/path/to/xlmr_fusion_emotion.zip"
8. Refresh the Emotion lab and select the optional learned fusion profile.

The existing base XLM-R, mBERT and MuRIL checkpoints remain available. This package
contains source and data, no trained fusion weights. The learned profile becomes
available only after a completed GPU run is exported and installed.

The default data is the existing synthetic bootstrap: 16,000 train / 2,000
validation / 2,000 test, half from each language pair. It is not human-labelled.
Its scores measure generated examples and do not establish real-world accuracy
or improvement over the base checkpoints. Use the same independent held-out labels
and task when comparing models. No accuracy target or published-paper parity is claimed.

## Optional independently supervised tasks

Default USE_CONTEXT=False and SARCASM_COLUMN=None are intentional: the bootstrap
has no independently reviewed context or sarcasm annotations. No labels are derived
from emoji, sentiment, emotion classes or model guesses.

For your own canonical CSV, set USE_OWN_DATA=True. Required columns:
text,emotion,language_pair; labels joy,sadness,anger,fear,surprise,disgust,neutral;
language_pair hi-en or fr-en. Optional row_id, group_id, conversation_id and split
follow the existing frozen split/audit workflow. A source requiring an approved
matching data audit must include its audit JSON in the upload.

To train context conditioning, supply context and human_reviewed=true for every
row with context, plus group_id or conversation_id tying related turns together.
Set USE_CONTEXT=True. Every split must contain reviewed contextual emotion rows.
The emotion label must describe the current message with the provided context.
A corrected CSV retaining any synthetic-origin rows remains mixed/synthetic data:
set ALLOW_SYNTHETIC=True explicitly and provide its matching audit when required.
Human correction of selected rows does not make all generated rows human-labelled.
Do not paste held-out future turns into training context. Current text is the first
XLM-R tokenizer sequence and context is the second. Current text receives the
available token budget first; context receives the remaining budget. An overlong
current message is bounded safely and truncation is disclosed at inference.

To train the auxiliary sarcasm head, provide an independent sarcasm column of
0/1 or not_sarcastic/sarcastic and human_reviewed=true for its labelled rows.
Set SARCASM_COLUMN="sarcasm". Leave missing labels blank; their auxiliary loss
is masked. Both binary classes are required in train and validation. A trained
emotion model does not imply a trained sarcasm head. Context and sarcasm require
suitable real labels before they can be represented as trained capabilities.

## Kaggle alternative when Colab cannot assign a GPU

Open TRAIN_ADVANCED_KAGGLE.ipynb in Kaggle Notebooks. Upload this same training ZIP
as a private Kaggle Dataset and attach it with Add Input. In notebook Settings,
select a GPU accelerator and enable Internet to fetch the public base model and
Python dependencies. No credential API or paid service is used by the notebook.
The notebook locates the ZIP below /kaggle/input and saves the run below
/kaggle/working/CodeMood_Advanced_XLMR. Use Save Version with output persistence;
retain its model/recovery files and the inference ZIP. For a later session, attach
the saved notebook output as input, set RESTORE_RUN_FROM to that prior run folder,
and rerun the same settings. Kaggle runtime resets do not preserve unsaved working
files. Availability, account verification and quotas depend on Kaggle.

## Architecture and recovery

Text CLS plus mean learned emoji-sequence embeddings are fused through a learned
residual vector gate. Emoji sequences (including ZWJ, modifiers and flags) use
versioned SHA256 buckets, so embeddings may collide; they are trained parameters,
not hand-authored sentiment scores. Missing emoji contribution is exactly zero.
The optional binary head shares the fused representation. Emoji embeddings, gate,
classification head and encoder receive gradients during training.

Only CUDA training is permitted. --dry-run validates data, groups, frozen hashes
and supervision before importing Torch, querying CUDA or downloading a base model.
Validation macro-F1 chooses the best seven-emotion checkpoint. Test is evaluated
once after selection. A completed run returns saved results when resumed; tuning
after inspecting test requires a separately preregistered experiment and fresh holdout.

Drive recovery saves the newest fully completed epoch with model/optimizer/RNG
SafeTensors plus JSON, without pickle. A partial epoch repeats after interruption.
Rerun the notebook from the upload cell after a runtime reset, with the same data
and training settings. EPOCHS is the total target count. Keep the best checkpoint
and recovery folder together. The inference ZIP excludes optimizer state, source
training rows and recovery files. GPU allocation and runtime durations vary.
"""


def notebook():
    cells = []
    def add(kind, source):
        cell = {"cell_type": kind, "id": f"step_{len(cells):02d}", "metadata": {}, "source": source.splitlines(keepends=True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    add("markdown", """# Train optional learned text-emoji XLM-R

Select a GPU runtime and run cells in order. This notebook runs real fine-tuning
on your assigned GPU. The packaged 16k/2k/2k bootstrap is synthetic, experimental
and cannot establish human-labelled accuracy. Optional context and sarcasm
supervision are disabled until you supply independently reviewed labels.
""")
    add("code", '''from pathlib import Path, PurePosixPath
import hashlib, json, zipfile, os, subprocess, sys
from google.colab import files
uploaded = files.upload()  # Select xlmr_advanced_training.zip
archives = [name for name in uploaded if name.lower().endswith(".zip")]
assert len(archives) == 1, "Upload exactly one training ZIP"
base = Path("/content").resolve()
PACKAGE = base / "xlmr_advanced_training"
with zipfile.ZipFile(archives[0]) as archive:
    names = archive.namelist()
    assert len(names) == len(set(names)) and len(names) < 1000
    assert sum(item.file_size for item in archive.infolist()) < 128 * 1024**2
    for item in archive.infolist():
        name = item.filename
        assert "\\\\" not in name and not PurePosixPath(name).is_absolute()
        assert ".." not in PurePosixPath(name).parts
        assert name.startswith("xlmr_advanced_training/")
        assert (base / name).resolve().is_relative_to(PACKAGE.resolve())
        assert (item.external_attr >> 16) & 0o170000 != 0o120000
    manifest = json.loads(archive.read("xlmr_advanced_training/MANIFEST.json"))
    expected = {"xlmr_advanced_training/" + name for name in manifest["sha256"]}
    assert set(names) == expected | {"xlmr_advanced_training/MANIFEST.json"}
    for name, digest in manifest["sha256"].items():
        assert hashlib.sha256(archive.read("xlmr_advanced_training/" + name)).hexdigest() == digest
    archive.extractall(base)
PROJECT = PACKAGE / "source"
os.chdir(PROJECT)
subprocess.run([sys.executable, "-m", "pip", "install", "-e", ".[model]"], check=True)
sys.path.insert(0, str(PROJECT))
print("Verified source/data package. No model weights were included.")
''')
    add("markdown", """## Choose data and optional reviewed supervision

Use defaults to train text/emoji fusion only. Your own emotion CSV can include
reviewed context and binary sarcasm labels: consult START_HERE.md before enabling
either option. Do not manufacture sarcasm labels from emotion or emoji.
""")
    add("code", '''USE_OWN_DATA = False
ALLOW_SYNTHETIC = not USE_OWN_DATA  # Set True explicitly for your mixed CSV retaining synthetic-origin rows
USE_CONTEXT = False
SARCASM_COLUMN = None  # "sarcasm" only with independent reviewed binary labels
EPOCHS = 3
DATA = PACKAGE / "data/all_splits.csv"
AUDIT = PACKAGE / "data/dataset_audit.json"
if USE_OWN_DATA:
    custom = files.upload()  # one canonical CSV, plus approved audit JSON if required
    csvs = [name for name in custom if name.lower().endswith(".csv")]
    audits = [name for name in custom if name.lower().endswith(".json")]
    assert len(csvs) == 1 and len(audits) <= 1
    DATA = Path(csvs[0]).resolve()
    AUDIT = Path(audits[0]).resolve() if audits else None
else:
    assert not USE_CONTEXT and SARCASM_COLUMN is None, "The synthetic bootstrap has no reviewed context/sarcasm labels"
from google.colab import drive
drive.mount("/content/drive")
STORAGE = Path("/content/drive/MyDrive/CodeMood_Advanced_XLMR")
RUN = STORAGE / "xlmr_fusion_emotion"
RUN.mkdir(parents=True, exist_ok=True)
command = [sys.executable, "-u", "-m", "training.train_advanced", "--data", str(DATA),
           "--output", str(RUN), "--split-manifest", str(RUN / "frozen_split.csv"),
           "--epochs", str(EPOCHS), "--batch-size", "2", "--gradient-accumulation", "8",
           "--max-length", "128", "--learning-rate", "2e-5", "--precision", "auto"]
if AUDIT: command += ["--data-audit", str(AUDIT)]
if ALLOW_SYNTHETIC: command += ["--allow-synthetic"]
if USE_CONTEXT: command += ["--use-context"]
if SARCASM_COLUMN: command += ["--sarcasm-column", SARCASM_COLUMN]
subprocess.run(command + ["--dry-run"], cwd=PROJECT, check=True)
if not USE_OWN_DATA:
    from training.data_workflow import prepare_data
    prepared = prepare_data(DATA, "emotion", "raw", 42, RUN / "frozen_split.csv", AUDIT, "all", True)
    assert [len(part) for part in prepared["parts"]] == [16000, 2000, 2000]
''')
    add("markdown", """## Train and recover completed epochs

The encoder, seven-emotion head, emoji embeddings and gate are updated. The best
validation macro-F1 checkpoint is retained. Drive holds the newest completed
epoch and safe optimizer/RNG recovery state; no CPU training is attempted.
""")
    add("code", '''import torch
assert torch.cuda.is_available(), "Select a GPU runtime before training"
print("GPU:", torch.cuda.get_device_name(0))
if (RUN / "latest_recovery.json").is_file():
    command += ["--resume"]
subprocess.run(command, cwd=PROJECT, check=True)
''')
    add("markdown", """## Inspect the recorded scope and download inference weights

Synthetic-only scores do not establish real-world accuracy. Keep the test split
untouched while making model choices. No improvement over the base checkpoints
is claimed without a matched independent comparison.
""")
    add("code", '''import pandas as pd
metrics = json.loads((RUN / "metrics.json").read_text())
print("Evaluation scope:", metrics["evaluation_scope"])
print("Context trained:", metrics["emotion_metadata"]["trained_with_context"])
print("Supervised sarcasm head:", metrics["emotion_metadata"]["trained_sarcasm_head"])
display(pd.DataFrame(metrics["history"]))
display(pd.DataFrame(metrics["test_by_language"]).T)
display(pd.DataFrame(metrics["test"]["per_class"]).T)
from training.cloud_bundle import export_checkpoint
archive = export_checkpoint(RUN, STORAGE / "xlmr_fusion_emotion.zip")
print("Saved inference ZIP:", archive)
files.download(str(archive))
''')
    add("markdown", """## Install locally

From the laptop project root:
`python -m training.install_checkpoint "C:/path/to/xlmr_fusion_emotion.zip"`

Refresh the Emotion lab and choose the optional learned fusion profile. Existing
base checkpoints are retained. No trained head or capability is implied by this
source package; availability follows the installed completed checkpoint metadata.
""")
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "colab": {"name": "TRAIN_ADVANCED.ipynb", "provenance": []}}, "nbformat": 4, "nbformat_minor": 5}


def kaggle_notebook():
    """Same trainer with Kaggle inputs/working outputs and no Colab APIs."""
    book = notebook()
    book["metadata"]["colab"] = {}
    book["metadata"]["kernelspec"]["display_name"] = "Python 3"
    first = "".join(book["cells"][1]["source"])
    first = first.replace('from google.colab import files\nuploaded = files.upload()  # Select xlmr_advanced_training.zip\narchives = [name for name in uploaded if name.lower().endswith(".zip")]\nassert len(archives) == 1, "Upload exactly one training ZIP"\nbase = Path("/content").resolve()',
                          'archives = sorted(Path("/kaggle/input").rglob("xlmr_advanced_training.zip"))\nassert len(archives) == 1, "Attach the private Dataset containing exactly one training ZIP using Add Input"\nbase = Path("/kaggle/working").resolve()')
    book["cells"][1]["source"] = first.splitlines(keepends=True)
    settings = "".join(book["cells"][3]["source"])
    upload = settings.index('if USE_OWN_DATA:')
    end = settings.index('else:', upload)
    settings = settings[:upload] + '''OWN_DATA_PATH = None  # e.g. "/kaggle/input/my-reviewed-data/all_splits.csv"
OWN_AUDIT_PATH = None  # matching audit JSON, if required by the source
RESTORE_RUN_FROM = None  # e.g. "/kaggle/input/prior-notebook-output/CodeMood_Advanced_XLMR/xlmr_fusion_emotion"
if USE_OWN_DATA:
    assert OWN_DATA_PATH, "Set OWN_DATA_PATH to your attached canonical CSV"
    DATA = Path(OWN_DATA_PATH).resolve()
    AUDIT = Path(OWN_AUDIT_PATH).resolve() if OWN_AUDIT_PATH else None
''' + settings[end:]
    settings = settings.replace('from google.colab import drive\ndrive.mount("/content/drive")\nSTORAGE = Path("/content/drive/MyDrive/CodeMood_Advanced_XLMR")\nRUN = STORAGE / "xlmr_fusion_emotion"\nRUN.mkdir(parents=True, exist_ok=True)',
                                '''STORAGE = Path("/kaggle/working/CodeMood_Advanced_XLMR")
RUN = STORAGE / "xlmr_fusion_emotion"
if RESTORE_RUN_FROM:
    import shutil
    source = Path(RESTORE_RUN_FROM).resolve()
    assert source.is_relative_to(Path("/kaggle/input")) and source.is_dir()
    assert not RUN.exists(), "Restore into a fresh working run folder"
    shutil.copytree(source, RUN)
RUN.mkdir(parents=True, exist_ok=True)''')
    book["cells"][3]["source"] = settings.splitlines(keepends=True)
    export = "".join(book["cells"][7]["source"])
    export = export.replace('files.download(str(archive))', 'from IPython.display import FileLink\ndisplay(FileLink(str(archive)))\nprint("Save Version with outputs before ending this session; retain the run folder for completed-epoch recovery.")')
    book["cells"][7]["source"] = export.splitlines(keepends=True)
    book["cells"][0]["source"] = ("# Train advanced XLM-R on Kaggle\n\nAttach the private training ZIP Dataset with Add Input. Enable a GPU and Internet\nin Settings, then run cells in order. Working outputs are saved under /kaggle/working;\nSave Version with outputs before stopping. The default data is synthetic and has no\nreviewed sarcasm/context labels. No model improvement is claimed without independent evaluation.\n").splitlines(keepends=True)
    return book


def build(output=None, data_dir=None):
    output = Path(output or ROOT / "colab/advanced_training").resolve()
    data = Path(data_dir or ROOT / "data/training/multilingual_bootstrap_v1").resolve()
    if not all((data / name).is_file() for name in DATA_FILES):
        raise ValueError("The existing multilingual bootstrap is required; it is never silently regenerated")
    audit = json.loads((data / "dataset_audit.json").read_text(encoding="utf-8"))
    if (audit.get("dataset_kind") != "synthetic_template"
            or audit.get("prepared_csv_sha256") != hashlib.sha256((data / "all_splits.csv").read_bytes()).hexdigest()):
        raise ValueError("The source dataset audit does not match the existing CSV")
    from training.data_workflow import prepare_data
    prepared = prepare_data(data / "all_splits.csv", "emotion", "raw", 42, None, data / "dataset_audit.json", "all", True)
    if [len(part) for part in prepared["parts"]] != [16000, 2000, 2000]:
        raise ValueError("The default package requires fixed 16000/2000/2000 synthetic partitions")
    package = output / "package"
    package.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in DATA_FILES:
        target = package / "data" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(data / name, target)
        paths.append(target)
    for folder in ("hinglish_emotion", "training"):
        for path in sorted((ROOT / folder).glob("*.py")):
            target = package / "source" / folder / path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            paths.append(target)
    target = package / "source/pyproject.toml"
    shutil.copyfile(ROOT / "pyproject.toml", target)
    paths.append(target)
    output.mkdir(parents=True, exist_ok=True)
    (output / "TRAIN_ADVANCED.ipynb").write_text(json.dumps(notebook(), ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "START_HERE.md").write_text(GUIDE, encoding="utf-8")
    (output / "TRAIN_ADVANCED_KAGGLE.ipynb").write_text(json.dumps(kaggle_notebook(), ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {"package": PACKAGE, "architecture": "text_emoji_multitask_v1", "evaluation_scope": "synthetic_only",
                "split_sizes": dict(zip(("train", "validation", "test"), map(len, prepared["parts"]))),
                "sha256": {path.relative_to(package).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths)}}
    manifest_path = package / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    archive_path = output / "xlmr_advanced_training.zip"
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted([*paths, manifest_path]):
            archive.write(path, PACKAGE + "/" + path.relative_to(package).as_posix())
    return {"folder": str(output), "zip": str(archive_path), "split_sizes": manifest["split_sizes"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output")
    parser.add_argument("--data-dir")
    args = parser.parse_args()
    print(json.dumps(build(args.output, args.data_dir), indent=2))


if __name__ == "__main__":
    main()
