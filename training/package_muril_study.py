"""Build a Colab-only MuRIL study tied to the installed XLM-R test split."""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import zipfile
from pathlib import Path

from hinglish_emotion.tasks import EMOTION_LABELS
from training.data_workflow import prepare_data
from training.muril_study import test_fingerprint, validate_baseline
from training.package_project import PACKAGE_NAME, ROOT, build as build_project, notebook as project_notebook

OUT = ROOT / "colab/muril_study"
BASELINE = ROOT / "models/xlmr_multilingual_emotion/metrics.json"
SETTING_KEYS = ("text_mode", "preparation_version", "max_length", "seed", "split_seed", "batch_size",
                "gradient_accumulation", "learning_rate", "class_weights", "freeze_encoder", "precision",
                "patience", "training_mode", "training_data_kind", "allow_synthetic")


def study_contract(baseline):
    data = ROOT / "data/training/multilingual_bootstrap_v1"
    common = dict(data=data / "all_splits.csv", task="emotion", text_mode="raw", split_seed=42,
                  data_audit=data / "dataset_audit.json", allow_synthetic=True)
    combined = prepare_data(**common)
    hindi = prepare_data(**common, language_pair="hi-en")
    test = hindi["parts"][2]
    full_test = combined["parts"][2]
    if test_fingerprint(test) != test_fingerprint(full_test[full_test.language_pair.eq("hi-en")]):
        raise ValueError("The Hindi-English subset does not reproduce XLM-R's held-out phrases")
    if tuple(len(part) for part in hindi["parts"]) != (8000, 1000, 1000):
        raise ValueError("The study requires the original 8,000/1,000/1,000 Hindi-English split")
    settings = {key: baseline[key] for key in SETTING_KEYS}
    required = dict(text_mode="raw", preparation_version=1, max_length=128, seed=42, split_seed=42,
                    batch_size=2, gradient_accumulation=8, learning_rate=2e-5, class_weights="balanced",
                    freeze_encoder=False, precision="fp16", patience=2, training_mode="full_fine_tuning",
                    training_data_kind="synthetic_only", allow_synthetic=True)
    if settings != required:
        raise ValueError("Installed baseline settings differ from the notebook's declared study settings")

    def expected(prepared):
        return {"dataset_sha256": prepared["dataset_sha256"], "split_sha256": prepared["manifest_sha256"],
                "split_sizes": dict(zip(("train", "validation", "test"), (len(part) for part in prepared["parts"]))),
                "hi_en_test_rows": len(test), "hi_en_test_support": {label: int(test.emotion.eq(label).sum()) for label in EMOTION_LABELS},
                "settings": settings}

    contract = {"schema_version": 1, "language_pair": "hi-en", "max_epochs": 3,
                "baseline_metrics_sha256": hashlib.sha256(json.dumps(baseline, sort_keys=True).encode()).hexdigest(),
                "hi_en_test_sha256": test_fingerprint(test), "baseline": expected(combined), "matched": expected(hindi)}
    validate_baseline(contract, baseline)
    return contract


def notebook():
    cells = []

    def add(kind, source):
        cell = {"cell_type": kind, "id": f"muril_{len(cells):02d}", "metadata": {}, "source": source.splitlines(keepends=True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    add("markdown", """# MuRIL and XLM-R: Hindi-English emotion study

Fine-tune MuRIL in **Colab with a GPU**, using the same 1,000 Hindi-English test phrases as the existing combined XLM-R run. Select **Runtime > Change runtime type > T4 GPU**, then **Runtime > Run all**. Upload `muril_study.zip` when prompted. This is supervised seven-emotion fine-tuning, not pretraining a language model from scratch.

The data is synthetic and template-based. These scores do not demonstrate real-world accuracy. The existing combined XLM-R had 16,000 training phrases across Hindi-English and English-French; MuRIL uses 8,000 Hindi-English phrases. An optional matched Hindi-English XLM-R run helps separate the encoder difference from training-language scope. Run MuRIL first, then the optional matched job in a separate session.
""")
    # Reuse the project's path-checked extraction, SHA validation and GPU check.
    add("code", "".join(project_notebook()["cells"][1]["source"]))
    add("code", '''# One job per session. For the optional matched run, change JOB only.
JOB = "muril_hi_en"  # muril_hi_en | xlmr_hi_en_matched
EPOCHS = 3
assert JOB in {"muril_hi_en", "xlmr_hi_en_matched"}
assert EPOCHS == 3, "Keep the study's three-epoch maximum (early stopping uses validation only)."
drive.mount("/content/drive")
STORAGE = Path("/content/drive/MyDrive/HinglishEmotionProject/MuRILStudy")
run = STORAGE / JOB
run.mkdir(parents=True, exist_ok=True)
model = "muril" if JOB == "muril_hi_en" else "xlmr"
DATA = PACKAGE / "data/all_splits.csv"
AUDIT = PACKAGE / "data/dataset_audit.json"
contract = json.loads((PACKAGE / "study/study_contract.json").read_text())
baseline = json.loads((PACKAGE / "study/xlmr_baseline_metrics.json").read_text())
''')
    add("markdown", """## Freeze and check the data

Only Hindi-English rows enter either new job. The stored hashes bind the text, labels, row identities and splits to the installed XLM-R baseline. Training updates weights on 8,000 rows; 1,000 validation rows select the best checkpoint; 1,000 held-out test rows are evaluated after selection. Do not tune settings using the test scores. The labels are joy, sadness, anger, fear, surprise, disgust and neutral.
""")
    add("code", '''from training.data_workflow import prepare_data
from training.muril_study import test_fingerprint, validate_baseline
validate_baseline(contract, baseline)
prepared = prepare_data(DATA, "emotion", "raw", 42, run / "frozen_split.csv", AUDIT, "hi-en", allow_synthetic=True)
expected = contract["matched"]
assert prepared["dataset_sha256"] == expected["dataset_sha256"], "Dataset changed"
assert prepared["manifest_sha256"] == expected["split_sha256"], "Split changed"
assert test_fingerprint(prepared["parts"][2]) == contract["hi_en_test_sha256"], "Test phrases changed"
sizes = dict(zip(("train", "validation", "test"), (len(part) for part in prepared["parts"])))
assert sizes == expected["split_sizes"]
print("Job:", JOB, "| Synthetic Hindi-English splits:", sizes)
''')
    add("markdown", """## Train on the GPU

Both new runs use raw text, length 128, batch size 2, eight accumulation steps, learning rate 0.00002, seed 42, balanced class weights and fp16. All encoder weights and the emotion head are fine-tuned. Validation macro-F1 selects the best checkpoint, with patience 2 and at most three epochs. Epoch recovery state and the frozen split save to Drive. After a runtime reset, use the same ZIP and JOB, then Run all; completed epochs can be resumed. CPU training is intentionally disabled.

Model cards: [MuRIL](https://huggingface.co/google/muril-base-cased), [XLM-R](https://huggingface.co/FacebookAI/xlm-roberta-base). MuRIL is restricted to Hindi-English in this project.
""")
    add("code", '''command = [sys.executable, "-u", "-m", "training.compare_encoders",
    "--task", "emotion", "--models", model, "--data", str(DATA), "--language-pair", "hi-en",
    "--output-root", str(run / "models"), "--split-manifest", str(run / "frozen_split.csv"),
    "--epochs", str(EPOCHS), "--batch-size", "2", "--gradient-accumulation", "8",
    "--max-length", "128", "--learning-rate", "2e-5", "--precision", "fp16",
    "--text-mode", "raw", "--seed", "42", "--split-seed", "42", "--resume",
    "--data-audit", str(AUDIT), "--allow-synthetic"]
subprocess.run(command, cwd=PROJECT, check=True)
''')
    add("markdown", """## Download the trained checkpoint

The export contains the best inference weights, tokenizer, configuration and metrics. Training data and optimizer recovery files remain outside the export. Keep the full Drive job folder if you want to resume. On your Windows computer, install `muril_hi_en.zip` with `python -m training.install_checkpoint "C:/Users/Advay/Downloads/muril_hi_en.zip"` from the project folder. The dashboard will discover it under MuRIL. The optional matched XLM-R export installs separately from the existing combined model.
""")
    add("code", '''from training.cloud_bundle import export_checkpoint
model_dir = run / "models" / (model + "_emotion")
export_path = STORAGE / "exports" / (JOB + ".zip")
export_checkpoint(model_dir, export_path)
print("Inference checkpoint:", export_path)
files.download(str(export_path))
''')
    add("markdown", """## Compare the saved runs

Accuracy counts correct test predictions. Macro-F1 gives each emotion equal weight. The report includes both metrics, per-emotion precision/recall/F1/support, a graph and the study limitations. It uses the baseline's Hindi-English score, not its combined-language overall score. Mismatched data, splits, labels or declared settings are rejected. This is a single-seed synthetic study; no statistical-significance or real-world-performance claim follows from it.
""")
    add("code", '''from training.muril_study import compare_runs, write_report
from IPython.display import SVG, display
import pandas as pd
muril_path = STORAGE / "muril_hi_en/models/muril_emotion/metrics.json"
matched_path = STORAGE / "xlmr_hi_en_matched/models/xlmr_emotion/metrics.json"
if not muril_path.is_file():
    print("Train the muril_hi_en job first. No MuRIL score is available yet.")
else:
    muril = json.loads(muril_path.read_text())
    matched = json.loads(matched_path.read_text()) if matched_path.is_file() else None
    report = compare_runs(contract, baseline, muril, matched)
    report_dir = write_report(report, run / "comparison")
    display(pd.DataFrame(report["rows"]))
    display(SVG(filename=str(report_dir / "comparison.svg")))
    display(pd.DataFrame(report["per_class"]))
    for note in report["notes"]:
        print(note)
    report_zip = STORAGE / "exports" / (JOB + "_comparison.zip")
    with zipfile.ZipFile(report_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for file in sorted(report_dir.iterdir()):
            if file.is_file():
                z.write(file, file.name)
    files.download(str(report_zip))
''')
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "colab": {"name": "TRAIN_MURIL_STUDY.ipynb", "provenance": []}}, "nbformat": 4, "nbformat_minor": 5}


def build(output=None, baseline_path=None):
    output = Path(output or OUT).resolve()
    baseline = json.loads(Path(baseline_path or BASELINE).read_text(encoding="utf-8"))
    contract = study_contract(baseline)
    output.mkdir(parents=True, exist_ok=True)
    contract_text = json.dumps(contract, indent=2)
    baseline_text = json.dumps(baseline, indent=2)
    # Build in an isolated temporary folder; never overwrite the authored guide.
    with tempfile.TemporaryDirectory(prefix="muril_build_", dir=output) as temporary:
        scratch = Path(temporary)
        build_project(scratch)
        package = scratch / "package"
        study = package / "study"
        study.mkdir()
        (study / "study_contract.json").write_text(contract_text, encoding="utf-8")
        (study / "xlmr_baseline_metrics.json").write_text(baseline_text, encoding="utf-8")
        manifest = json.loads((package / "MANIFEST.json").read_text(encoding="utf-8"))
        for file in study.iterdir():
            manifest["sha256"][file.relative_to(package).as_posix()] = hashlib.sha256(file.read_bytes()).hexdigest()
        (package / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        with zipfile.ZipFile(output / "muril_study.zip", "w", zipfile.ZIP_DEFLATED) as z:
            for file in sorted(package.rglob("*")):
                if file.is_file():
                    z.write(file, PACKAGE_NAME + "/" + file.relative_to(package).as_posix())
    (output / "study_contract.json").write_text(contract_text, encoding="utf-8")
    (output / "xlmr_baseline_metrics.json").write_text(baseline_text, encoding="utf-8")
    (output / "TRAIN_MURIL_STUDY.ipynb").write_text(json.dumps(notebook(), indent=2, ensure_ascii=False), encoding="utf-8")
    return {"notebook": str(output / "TRAIN_MURIL_STUDY.ipynb"), "zip": str(output / "muril_study.zip"),
            "test_phrases": contract["matched"]["hi_en_test_rows"], "baseline_verified": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.output, args.baseline), indent=2))
