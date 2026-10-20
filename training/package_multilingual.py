"""Build the separate Colab handoff for both language mixes using XLM-R."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
from training.multilingual_data import save_multilingual

ROOT=Path(__file__).resolve().parents[1]
PACKAGE="xlmr_multilingual_training"


def notebook():
    cells=[]
    def add(kind,text):
        cell={"cell_type":kind,"id":f"step_{len(cells):02d}","metadata":{},"source":text.splitlines(keepends=True)}
        if kind=="code":cell.update(execution_count=None,outputs=[])
        cells.append(cell)
    add("markdown","""# XLM-R emotion training for Hindi-English and English-French

Start with Runtime > Change runtime type > GPU. Run cells in order or Run all,
responding to the upload and Drive prompts. This is task fine-tuning, not
training a language model from scratch. The provided bootstrap is synthetic;
its scores do not establish accuracy on naturally written or human-reviewed text.
""")
    add("code",'''from pathlib import Path
import hashlib, json, zipfile, subprocess, sys, os
from google.colab import files
uploaded = files.upload()  # Select xlmr_multilingual.zip
archives = [name for name in uploaded if name.lower().endswith(".zip")]
assert len(archives) == 1, "Upload exactly one training ZIP"
base = Path("/content").resolve()
with zipfile.ZipFile(archives[0]) as archive:
    assert sum(item.file_size for item in archive.infolist()) < 128 * 1024**2
    for item in archive.infolist():
        assert "\\\\" not in item.filename
        assert (base / item.filename).resolve().is_relative_to(base)
        assert (item.external_attr >> 16) & 0o170000 != 0o120000
    archive.extractall(base)
PACKAGE = base / "xlmr_multilingual_training"
manifest = json.loads((PACKAGE / "MANIFEST.json").read_text())
for name, expected in manifest["sha256"].items():
    path = (PACKAGE / name).resolve()
    assert path.is_relative_to(PACKAGE)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, name
PROJECT = PACKAGE / "source"
os.chdir(PROJECT)
subprocess.run([sys.executable, "-m", "pip", "install", "-e", ".[model]"], check=True)
sys.path.insert(0, str(PROJECT))
import torch
assert torch.cuda.is_available(), "Select a GPU runtime, then restart from the first cell"
print("GPU:", torch.cuda.get_device_name(0))
''')
    add("markdown","""## Choose the experiment

`combined` trains one checkpoint on both language mixes. `separate` trains two
XLM-R checkpoints sequentially. Leave the defaults for the first run.
The original installed Hindi-only model is preserved on your laptop.

To use reviewed data, set USE_OWN_DATA=True and upload a canonical CSV containing
text,emotion,language_pair, optionally group_id and split. Use hi-en and fr-en.
Each language needs all seven emotions and enough independent groups per split.
The test split must remain untouched while choosing training settings.
""")
    add("code",'''MODE = "combined"  # "combined" or "separate"
USE_OWN_DATA = False
ALLOW_SYNTHETIC = not USE_OWN_DATA
EPOCHS = 3  # Total target epochs, including already completed epochs on resume
DATA = PACKAGE / "data" / "all_splits.csv"
AUDIT = PACKAGE / "data" / "dataset_audit.json"
if USE_OWN_DATA:
    custom = files.upload()
    csvs = [name for name in custom if name.endswith(".csv")]
    assert len(csvs) == 1, "Upload one labelled canonical CSV"
    DATA = Path(csvs[0]).resolve()
    AUDIT = None
    # For a source requiring its matching audit, upload the audit JSON too.
    audits = [name for name in custom if name.endswith(".json")]
    if audits:
        assert len(audits) == 1
        AUDIT = Path(audits[0]).resolve()
assert MODE in {"combined", "separate"}
from google.colab import drive
drive.mount("/content/drive")
STORAGE = Path("/content/drive/MyDrive/CodeMood_XLMR_v2")
STORAGE.mkdir(parents=True, exist_ok=True)
''')
    add("markdown","""## Validate and train

The bootstrap contains 16,000 training, 2,000 validation and 2,000 synthetic test
rows across the two languages. Each language contributes 8,000/1,000/1,000.
The same frozen split is retained after disconnection. Checkpoints go to Drive.
Batch size 2 and accumulation 8 usually give 16 phrases per optimizer update.
The best validation macro-F1 checkpoint is used for test evaluation.
""")
    add("code",'''from training.data_workflow import prepare_data
jobs = [("all", "xlmr_multilingual_emotion")] if MODE == "combined" else [("hi-en", "xlmr_hi_en_emotion"), ("fr-en", "xlmr_fr_en_emotion")]
trained = []
for language, export_name in jobs:
    run = STORAGE / MODE / export_name
    manifest_path = run / "frozen_split.csv"
    prepared = prepare_data(DATA, "emotion", "raw", 42, manifest_path, AUDIT, language, allow_synthetic=ALLOW_SYNTHETIC)
    print(export_name, "split sizes:", [len(part) for part in prepared["parts"]])
    if not USE_OWN_DATA:
        expected = [16000, 2000, 2000] if language == "all" else [8000, 1000, 1000]
        assert [len(part) for part in prepared["parts"]] == expected
    command = [sys.executable, "-u", "-m", "training.compare_encoders", "--task", "emotion", "--models", "xlmr", "--data", str(DATA), "--language-pair", language, "--output-root", str(run / "models"), "--split-manifest", str(manifest_path), "--epochs", str(EPOCHS), "--batch-size", "2", "--gradient-accumulation", "8", "--max-length", "128", "--learning-rate", "2e-5", "--precision", "auto", "--text-mode", "raw", "--seed", "42", "--split-seed", "42", "--resume"]
    if AUDIT: command += ["--data-audit", str(AUDIT)]
    if ALLOW_SYNTHETIC: command += ["--allow-synthetic"]
    subprocess.run(command, cwd=PROJECT, check=True)
    trained.append((run / "models" / "xlmr_emotion", export_name))
''')
    add("markdown","""## Inspect results and export

Report per-language and per-emotion support, accuracy, macro-F1 and errors.
Bootstrap scores are synthetic-only. Do not compare these seven-class results
directly with Team 5 G2's three-class results. Real improvement needs the same
independent test labels and comparable tasks.
""")
    add("code",'''import pandas as pd
from training.cloud_bundle import export_checkpoint
exports = []
for model_dir, export_name in trained:
    metrics = json.loads((model_dir / "metrics.json").read_text())
    print(export_name, "scope:", metrics["evaluation_scope"])
    display(pd.DataFrame(metrics["test_by_language"]).T)
    display(pd.DataFrame(metrics["test"]["per_class"]).T)
    display(pd.DataFrame(metrics["history"]))
    output = export_checkpoint(model_dir, STORAGE / f"{export_name}.zip")
    exports.append(output)
    print("Saved inference export:", output)
for output in exports:
    files.download(str(output))
''')
    add("markdown","""## Install on the laptop

From the personal project root, run:
`python -m training.install_checkpoint "C:/path/to/downloaded/xlmr_multilingual_emotion.zip"`

For separate runs, install each ZIP the same way. The installer selects the
language-specific folder from metadata, rejects incomplete/unsafe exports and
preserves existing models. Restart or refresh the Streamlit app. XLM-R routes
to a checkpoint trained for the selected language. The ZIP excludes optimizer
state and raw training phrases. Keep Drive recovery files if resuming training.
""")
    return {"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"colab":{"name":"TRAIN_XLMR.ipynb","provenance":[]}},"nbformat":4,"nbformat_minor":5}


def build(output=None,data_dir=None):
    output=Path(output or ROOT/"colab/xlmr_multilingual").resolve()
    data=Path(data_dir or ROOT/"data/training/multilingual_bootstrap_v1").resolve()
    audit=save_multilingual(data)
    package=output/"package"
    package.mkdir(parents=True,exist_ok=True)
    paths=[]
    for file in data.rglob("*"):
        if file.is_file() and file.name in {"train.csv","validation.csv","test.csv","all_splits.csv","dataset_audit.json","DATASET_CARD.md"}:
            target=package/"data"/file.relative_to(data);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(file,target);paths.append(target)
    for folder in ("hinglish_emotion","training"):
        target=package/"source"/folder;target.mkdir(parents=True,exist_ok=True)
        for file in (ROOT/folder).glob("*.py"):
            shutil.copyfile(file,target/file.name);paths.append(target/file.name)
    shutil.copyfile(ROOT/"pyproject.toml",package/"source/pyproject.toml")
    paths.append(package/"source/pyproject.toml")
    (output/"TRAIN_XLMR.ipynb").write_text(json.dumps(notebook(),ensure_ascii=False,indent=2),encoding="utf-8")
    guide="""# XLM-R training handoff

1. Open https://colab.research.google.com/ and upload TRAIN_XLMR.ipynb.
2. Runtime > Change runtime type > GPU (T4 if available). Run all.
3. At the upload prompt choose xlmr_multilingual.zip from this same folder.
4. Leave MODE="combined" for one model supporting both language mixes. Use
   MODE="separate" only if you want two independent XLM-R runs.
5. Approve your own Drive mount. Training and recovery state save under
   MyDrive/CodeMood_XLMR_v2. Free GPU availability and session length vary.
6. Inspect per-language/per-emotion results. Download the inference ZIP(s).
7. On the laptop, from HinglishEmotionProject, run:
   python -m training.install_checkpoint "C:/path/to/downloaded/model.zip"
8. Open http://127.0.0.1:8501, select Emotion lab, XLM-R, and the language.
   The app discovers language-compatible checkpoints. Existing models are preserved.

## Dataset and limitations

The supplied bootstrap is synthetic: 8,000 training, 1,000 validation and 1,000
synthetic test phrases per language mix. Combined: 16,000/2,000/2,000. It is not
independently human-labelled and cannot establish real-world or published-paper
accuracy. The existing Hinglish examples are retained; French-English examples
are authored template compositions. No MaSaC corpus is included.

Working CSVs: data/training/multilingual_bootstrap_v1 from the project root.
Use all_splits.csv for training. Language codes: hi-en and fr-en (UI also accepts
en-fr uploads). Seven labels: joy, sadness, anger, fear, surprise, disgust, neutral.
USE_OWN_DATA=True accepts a canonical reviewed CSV and optional source audit.
Do not automatically relabel positive/negative sentiment as the seven emotions.

The package folder is a source/data snapshot for reproducibility; upload only
the notebook and ZIP. The ZIP contains no model weights, credentials or Drive data.

## Recovery and troubleshooting

After a reset, rerun from the first cell and upload the same package again.
PACKAGE undefined means the upload cell has not run. CUDA unavailable means the
runtime has no visible GPU. A partial epoch is repeated after disconnection.
Resume requires unchanged data and settings. Increasing EPOCHS sets the total,
not additional epochs. Keep both best checkpoint and latest epoch state on Drive.

Unlabelled CSV/XLSX files can be predicted in the dashboard, but accuracy needs
expected labels. Test rows are selected by default when a split column exists.
The model may confidently misclassify everyday phrases; training data diversity
and independently reviewed evaluation are still required. Optional Qwen review
is available for phrases; file evaluation scores the primary emotion model.
"""
    (output/"START_HERE.md").write_text(guide,encoding="utf-8")
    manifest={"package":PACKAGE,"evaluation_scope":"synthetic_only","sha256":{p.relative_to(package).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}}
    (package/"MANIFEST.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    archive=output/"xlmr_multilingual.zip"
    with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED) as z:
        for file in sorted([*paths,package/"MANIFEST.json"]):z.write(file,PACKAGE+"/"+file.relative_to(package).as_posix())
    return {"folder":str(output),"zip":str(archive),"split_sizes":audit["split_sizes"]}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output");parser.add_argument("--data-dir")
    args=parser.parse_args();print(json.dumps(build(args.output,args.data_dir),indent=2))
