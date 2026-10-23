"""Rebuild IEEE manuscript evidence and grayscale figures from saved artifacts."""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "ieee_manuscript"
FIGURES = OUT / "figures"
MODELS = {
    "xlmr": ("XLM-R", "models/xlmr_multilingual_emotion/metrics.json"),
    "mbert": ("mBERT", "models/mbert_multilingual_emotion/metrics.json"),
    "muril": ("MuRIL", "models/muril_hi_en_emotion/metrics.json"),
}
GRAYS = {"xlmr": "#111111", "mbert": "#666666", "muril": "#A0A0A0"}
LINE_STYLES = {"xlmr": "-", "mbert": "--", "muril": "-."}
MARKERS = {"xlmr": "o", "mbert": "s", "muril": "^"}
LABELS = ("joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral")


def read_json(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def number(value, field: str, *, unit: str = "score") -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"Non-finite {field}: {value}")
    if unit == "score" and not 0 <= value <= 1:
        raise ValueError(f"Out-of-range {field}: {value}")
    if unit == "positive" and value <= 0:
        raise ValueError(f"Non-positive {field}: {value}")
    return value


def per_class(metrics: dict) -> dict:
    result = {}
    for label in LABELS:
        item = metrics["per_class"][label]
        result[label] = {
            "precision": number(item["precision"], f"{label} precision"),
            "recall": number(item["recall"], f"{label} recall"),
            "f1": number(item["f1"], f"{label} F1"),
            "support": int(item["support"]),
        }
    return result


def build_evidence() -> dict:
    source_metrics = {key: read_json(path) for key, (_, path) in MODELS.items()}
    calibration = read_json("docs/oct23review/calibration_example.json")
    diagnostics = read_json("docs/oct23review/challenge_diagnostics.json")
    review_results = read_json("docs/oct23review/results.json")

    models = {}
    for key, (name, path) in MODELS.items():
        run = source_metrics[key]
        if run.get("task") != "emotion" or run.get("evaluation_scope") != "synthetic_only":
            raise ValueError(f"Unexpected task or evaluation scope for {name}")
        metadata = run.get("emotion_metadata", {})
        for field in ("base_model", "dataset_sha256", "split_sha256", "max_length"):
            if metadata and metadata.get(field) != run.get(field):
                raise ValueError(f"{name} metrics/metadata mismatch for {field}")
        language_results = {}
        for language, values in run["test_by_language"].items():
            language_results[language] = {
                "rows": int(values["rows"]),
                "accuracy": number(values["accuracy"], f"{name} {language} accuracy"),
                "macro_f1": number(values["macro_f1"], f"{name} {language} macro-F1"),
                "per_class": per_class(values),
            }
        history = [{
            "epoch": int(item["epoch"]),
            "train_loss": number(item["train_loss"], f"{name} training loss", unit="positive"),
            "validation_macro_f1": number(item["validation_macro_f1"], f"{name} validation macro-F1"),
        } for item in run["history"]]
        best = max(history, key=lambda row: row["validation_macro_f1"])
        hyperparameters = {
            key_name: run[key_name]
            for key_name in (
                "seed", "split_seed", "batch_size", "gradient_accumulation", "learning_rate",
                "class_weights", "freeze_encoder", "precision", "patience", "allow_synthetic",
                "device", "training_mode", "max_length", "text_mode", "preparation_version",
            ) if key_name in run
        }
        hyperparameters["epochs_completed"] = len(history)
        models[key] = {
            "name": name,
            "base_model": run["base_model"],
            "metric_source": path,
            "hyperparameters_source": path,
            "checkpoint_metadata_source": path.replace("metrics.json", "emotion_metadata.json"),
            "evaluation_scope": run["evaluation_scope"],
            "training_data_kind": run["training_data_kind"],
            "dataset_sha256": run["dataset_sha256"],
            "split_sha256": run["split_sha256"],
            "labels": list(run["labels"]),
            "split_sizes": run["split_sizes"],
            "hyperparameters": hyperparameters,
            "test_overall": {
                "rows": int(run["split_sizes"]["test"]),
                "accuracy": number(run["test"]["accuracy"], f"{name} overall accuracy"),
                "macro_f1": number(run["test"]["macro_f1"], f"{name} overall macro-F1"),
                "per_class": per_class(run["test"]),
            },
            "test_by_language": language_results,
            "history": history,
            "best_epoch": {
                "epoch": int(best["epoch"]),
                "validation_macro_f1": number(best["validation_macro_f1"], f"{name} best validation macro-F1"),
            },
        }

    if len(models["xlmr"]["history"]) != 3 or len(models["mbert"]["history"]) != 3 or len(models["muril"]["history"]) != 3:
        raise ValueError("Expected recorded three-epoch histories for all models")
    if models["xlmr"]["split_sizes"]["train"] != 16000 or models["mbert"]["split_sizes"]["train"] != 16000:
        raise ValueError("Expected XLM-R and mBERT 16k training split sizes")
    if models["muril"]["split_sizes"]["train"] != 8000 or set(models["muril"]["test_by_language"]) != {"hi-en"}:
        raise ValueError("Expected MuRIL 8k Hindi-English-only training/evaluation scope")

    calibration_record = {
        "source": "docs/oct23review/calibration_example.json",
        "rows": int(calibration["rows"]),
        "scope": calibration["scope"],
        "status": calibration["status"],
        "temperature": number(calibration["temperature"], "calibration temperature", unit="positive"),
        "metrics_before": {key: number(value, f"calibration before {key}", unit="positive") for key, value in calibration["metrics_before"].items()},
        "metrics_after": {key: number(value, f"calibration after {key}", unit="positive") for key, value in calibration["metrics_after"].items()},
        "accuracy_changed": bool(calibration["accuracy_changed"]),
        "independent_evaluation": bool(calibration["independent_evaluation"]),
        "data_selection": calibration["data_selection"],
    }
    diagnostics_record = {
        "source_file": "docs/oct23review/challenge_diagnostics.json",
        "source": diagnostics["source"],
        "label_scope": diagnostics["label_scope"],
        "records": [{key: record[key] for key in ("model", "dataset", "file", "language_pair", "rows", "correct", "accuracy", "file_sha256")}
                    for record in diagnostics["results"]],
        "not_measured": diagnostics.get("not_measured", []),
        "not_a_human_benchmark": True,
    }
    software_verification = {
        "source": "coordinator-provided project verification record",
        "tests_passed": 221,
        "subtests_passed": 33,
        "new_model_training_run": False,
    }
    mbert_completed_metadata = read_json("colab/mbert_study/completed_metadata.json")
    mbert_metadata_check = {
        "source": "colab/mbert_study/completed_metadata.json",
        "matches_installed_metrics": all(
            mbert_completed_metadata.get(field) == source_metrics["mbert"].get(field)
            for field in ("base_model", "labels", "text_mode", "preparation_version", "max_length", "dataset_sha256", "split_sha256")
        ),
        "fields": {field: mbert_completed_metadata.get(field) for field in (
            "base_model", "labels", "text_mode", "preparation_version", "max_length",
            "trained_language_pairs", "training_data_kind", "evaluation_scope", "training_status",
            "dataset_sha256", "split_sha256",
        )},
    }
    if not mbert_metadata_check["matches_installed_metrics"]:
        raise ValueError("mBERT completed metadata does not match installed metric artifact")
    return {
        "schema_version": 1,
        "evidence_snapshot": "2026-10-23",
        "study_status": "reanalysis_of_saved_runs_no_new_training",
        "task": "seven-class emotion classification",
        "classes": list(LABELS),
        "primary_evidence_scope": "synthetic-only held-out test metrics; not independently human-labelled evaluation",
        "models": models,
        "mbert_colab_completed_metadata": mbert_metadata_check,
        "calibration_example": calibration_record,
        "challenge_diagnostics": diagnostics_record,
        "software_verification": software_verification,
        "metric_cross_checks": review_results.get("colab_cross_checks", []),
        "interpretation_limits": [
            "XLM-R and mBERT use the same 16k/2k/2k bilingual synthetic split; MuRIL uses a distinct 8k/1k/1k Hindi-English synthetic split.",
            "MuRIL results are Hindi-English only and are not included in bilingual overall or English-French comparisons.",
            "Challenge diagnostics use assistant-authored or illustrative labels; they are not independent human evaluation and are not subtracted from held-out scores.",
            "The 98-row calibration record is experimental and unreviewed; calibration changes score confidence but not predicted labels.",
            "The software verification count is not a model-performance measure. No new training run was performed for this evidence snapshot.",
        ],
        "figures": ["figures/model_comparison.png", "figures/training_history.png"],
    }


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")
    ax.grid(axis="y", color="#D5D5D5", linewidth=0.55, zorder=0)
    ax.tick_params(axis="both", labelsize=8, length=3, colors="#222222")


def build_model_comparison(models: dict):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5,
                         "axes.titlesize": 10, "axes.labelsize": 8.5,
                         "xtick.labelsize": 8, "ytick.labelsize": 7.8,
                         "legend.fontsize": 8, "figure.facecolor": "white",
                         "axes.facecolor": "white", "savefig.facecolor": "white"})
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.35), sharey=True)
    panels = [
        (axes[0], "hi-en", ["xlmr", "mbert", "muril"], "(a) Hindi-English"),
        (axes[1], "fr-en", ["xlmr", "mbert"], "(b) English-French"),
    ]
    width = 0.34
    for ax, language, model_keys, title in panels:
        positions = np.arange(len(model_keys), dtype=float)
        for offset, metric, label, color, hatch in (
            (-width / 2, "accuracy", "Accuracy", "#202020", ""),
            (width / 2, "macro_f1", "Macro-F1", "#FFFFFF", "///"),
        ):
            scores = [models[key]["test_by_language"][language][metric] for key in model_keys]
            bars = ax.bar(positions + offset, np.asarray(scores) * 100, width,
                          color=color, edgecolor="#111111", linewidth=0.8, hatch=hatch,
                          label=label, zorder=3)
            ax.bar_label(bars, labels=[f"{score * 100:.1f}" for score in scores],
                         padding=2, fontsize=7.2, color="#111111")
        ax.set_xticks(positions, [models[key]["name"] for key in model_keys])
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
        ax.set_title(title, loc="left", fontweight="bold", pad=7)
        ax.set_ylabel("Score (%)")
        ax.set_xlabel("Model")
        style_axes(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.51, 1.04), ncol=2, frameon=False)
    fig.suptitle("Synthetic test performance by language", x=0.03, y=1.13,
                 ha="left", fontsize=11, fontweight="bold")
    fig.text(0.03, -0.03,
             "Bars show saved test accuracy and macro-F1 (n=1,000 per language/model). "
             "MuRIL used a separate 8k-train hi-en split; XLM-R and mBERT used 16k bilingual training rows.",
             ha="left", va="top", fontsize=7.2, color="#333333", wrap=True)
    fig.tight_layout(rect=(0.015, 0.04, 0.995, 0.92), w_pad=1.0)
    fig.savefig(FIGURES / "model_comparison.png", dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def build_training_history(models: dict):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5,
                         "axes.titlesize": 9.5, "axes.labelsize": 8.5,
                         "xtick.labelsize": 8, "ytick.labelsize": 7.8,
                         "legend.fontsize": 7.8, "figure.facecolor": "white",
                         "axes.facecolor": "white", "savefig.facecolor": "white"})
    fig, (ax_f1, ax_loss) = plt.subplots(2, 1, figsize=(6.6, 4.25), sharex=True)
    for key in ("xlmr", "mbert", "muril"):
        history = models[key]["history"]
        epochs = [row["epoch"] for row in history]
        if epochs != [1, 2, 3]:
            raise ValueError(f"Expected recorded epochs 1–3 for {key}")
        ax_f1.plot(epochs, [row["validation_macro_f1"] for row in history],
                   color=GRAYS[key], linestyle=LINE_STYLES[key], marker=MARKERS[key],
                   markersize=4.5, linewidth=1.6, label=models[key]["name"], zorder=3)
        ax_loss.plot(epochs, [row["train_loss"] for row in history],
                     color=GRAYS[key], linestyle=LINE_STYLES[key], marker=MARKERS[key],
                     markersize=4.5, linewidth=1.6, label=models[key]["name"], zorder=3)
    ax_f1.set_ylim(0.70, 1.0)
    ax_f1.set_yticks([0.70, 0.80, 0.90, 1.0])
    ax_f1.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax_f1.set_ylabel("Validation macro-F1")
    ax_f1.set_title("(a) Validation macro-F1 (axis starts at 70%)", loc="left", fontweight="bold")
    ax_f1.legend(ncol=3, loc="lower left", frameon=False, handlelength=2.4)
    ax_loss.set_yscale("log")
    ax_loss.set_ylabel("Training loss (log scale)")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_xticks([1, 2, 3])
    ax_loss.set_title("(b) Training loss", loc="left", fontweight="bold")
    ax_loss.legend(ncol=3, loc="upper right", frameon=False, handlelength=2.4)
    for ax in (ax_f1, ax_loss):
        style_axes(ax)
    fig.suptitle("Recorded three-epoch training histories", x=0.03, y=0.995,
                 ha="left", fontsize=11, fontweight="bold")
    fig.text(0.03, 0.005,
             "All runs used synthetic emotion data. XLM-R/mBERT: 16k train and 2k validation; "
             "MuRIL: 8k train and 1k validation. Best validation epoch: 1 for each.",
             ha="left", va="bottom", fontsize=7.2, color="#333333", wrap=True)
    fig.tight_layout(rect=(0.025, 0.065, 0.99, 0.94), h_pad=0.8)
    fig.savefig(FIGURES / "training_history.png", dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    evidence = build_evidence()
    build_model_comparison(evidence["models"])
    build_training_history(evidence["models"])
    (OUT / "evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "evidence": str((OUT / "evidence.json").relative_to(ROOT)),
        "figures": evidence["figures"],
        "models": {key: {"overall": value["test_overall"], "by_language": {
            lang: {"rows": score["rows"], "accuracy": score["accuracy"], "macro_f1": score["macro_f1"]}
            for lang, score in value["test_by_language"].items()}, "best_epoch": value["best_epoch"]}
            for key, value in evidence["models"].items()},
        "calibration": evidence["calibration_example"],
        "software_verification": evidence["software_verification"],
    }, indent=2))


if __name__ == "__main__":
    main()
