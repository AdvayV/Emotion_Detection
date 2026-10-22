"""Build print-ready model and validation-history figures for the Oct 23 review."""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "oct23review"
FIGURES = OUT / "figures"
LABELS = {"xlmr": "XLM-R", "mbert": "mBERT", "muril": "MuRIL"}
COLORS = {"xlmr": "#009E73", "mbert": "#3B4CC0", "muril": "#E69F00"}
METRIC_COLORS = {"accuracy": "#009E73", "macro_f1": "#3B4CC0"}
CHALLENGE_DIAGNOSTICS = {
    "source": "coordinator CPU diagnostic on unreviewed draft labels; not independent human benchmark",
    "source_file": "docs/oct23review/challenge_diagnostics.json (not present at chart build time)",
    "records": [
        {"model": "mBERT", "dataset": "Hindi-English assistant challenge drafts", "language_pair": "hi-en", "rows": 100, "accuracy": 0.50},
        {"model": "mBERT", "dataset": "English-French assistant challenge drafts", "language_pair": "fr-en", "rows": 100, "accuracy": 0.41},
        {"model": "mBERT", "dataset": "Bundled hi-en demo subset", "language_pair": "hi-en", "rows": 42, "accuracy": 0.57142},
        {"model": "MuRIL", "dataset": "Hindi-English assistant challenge drafts", "language_pair": "hi-en", "rows": 100, "accuracy": 0.51},
        {"model": "MuRIL", "dataset": "Bundled hi-en demo subset", "language_pair": "hi-en", "rows": 42, "accuracy": 0.59523},
    ],
    "not_measured": [{"model": "XLM-R", "dataset": "Assistant challenge drafts", "reason": "No verified diagnostic score supplied."}],
    "interpretation": "Diagnostic point estimates on assistant-authored labels; sets are unmatched to the saved synthetic held-out tests, so no generalization-gap subtraction or human-accuracy claim is made.",
}
_DIAGNOSTIC_PATH=OUT/"challenge_diagnostics.json"
if _DIAGNOSTIC_PATH.is_file():
    _recorded=json.loads(_DIAGNOSTIC_PATH.read_text(encoding="utf-8"))
    CHALLENGE_DIAGNOSTICS={"source":_recorded["source"],"source_file":"docs/oct23review/challenge_diagnostics.json",
        "records":_recorded["results"],"label_scope":_recorded["label_scope"],
        "not_measured":[{"model":"XLM-R","dataset":"Assistant challenge drafts","reason":"No measured challenge score recorded."}]}



def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def finite_metric(value: object, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"Invalid {name}: {value}")
    return value


def find_best_epoch(history: list[dict]) -> dict:
    if not history:
        raise ValueError("Training history is empty")
    best = max(history, key=lambda item: item["validation_macro_f1"])
    return {"epoch": int(best["epoch"]), "validation_macro_f1": finite_metric(best["validation_macro_f1"], "validation macro-F1")}


def cross_check(left: dict, right: dict, left_name: str, right_name: str) -> dict:
    for key in ("dataset_sha256", "split_sha256", "history", "test", "test_by_language", "split_sizes"):
        if left.get(key) != right.get(key):
            raise ValueError(f"Saved metric copies differ for {key}: {left_name} vs {right_name}")
    return {"files": [left_name, right_name], "matched": True,
            "dataset_sha256": left["dataset_sha256"], "split_sha256": left["split_sha256"]}


def build_data() -> tuple[dict, dict]:
    paths = {
        "xlmr": ROOT / "models/xlmr_multilingual_emotion/metrics.json",
        "mbert": ROOT / "models/mbert_multilingual_emotion/metrics.json",
        "muril": ROOT / "models/muril_hi_en_emotion/metrics.json",
    }
    metrics = {key: load(path) for key, path in paths.items()}
    xlm_colab = load(ROOT / "colab/mbert_study/xlmr_baseline_metrics.json")
    mb_colab = load(ROOT / "colab/mbert_study/completed_metrics.json")
    mb_metadata = load(ROOT / "colab/mbert_study/completed_metadata.json")
    xlm_muril_study = load(ROOT / "colab/muril_study/xlmr_baseline_metrics.json")
    crosschecks = [
        cross_check(metrics["xlmr"], xlm_colab, "models/xlmr_multilingual_emotion/metrics.json", "colab/mbert_study/xlmr_baseline_metrics.json"),
        cross_check(metrics["xlmr"], xlm_muril_study, "models/xlmr_multilingual_emotion/metrics.json", "colab/muril_study/xlmr_baseline_metrics.json"),
        cross_check(metrics["mbert"], mb_colab, "models/mbert_multilingual_emotion/metrics.json", "colab/mbert_study/completed_metrics.json"),
    ]
    if mb_metadata.get("dataset_sha256") != metrics["mbert"].get("dataset_sha256") or mb_metadata.get("split_sha256") != metrics["mbert"].get("split_sha256"):
        raise ValueError("mBERT completed metadata does not match the installed checkpoint metrics")

    for key, data in metrics.items():
        if data.get("evaluation_scope") != "synthetic_only":
            raise ValueError(f"Unexpected evaluation scope for {key}: {data.get('evaluation_scope')}")
        if not data.get("history"):
            raise ValueError(f"Missing epoch history for {key}")
        for score in (data["test"]["accuracy"], data["test"]["macro_f1"]):
            finite_metric(score, f"{key} overall test score")
        for language, score in data["test_by_language"].items():
            finite_metric(score["accuracy"], f"{key} {language} accuracy")
            finite_metric(score["macro_f1"], f"{key} {language} macro-F1")

    models = {}
    for key, data in metrics.items():
        history = [{"epoch": int(item["epoch"]),
                    "train_loss": float(item["train_loss"]),
                    "validation_macro_f1": finite_metric(item["validation_macro_f1"], "validation macro-F1")}
                   for item in data["history"]]
        models[key] = {
            "name": LABELS[key],
            "base_model": data.get("base_model"),
            "metrics_file": paths[key].relative_to(ROOT).as_posix(),
            "evaluation_scope": data.get("evaluation_scope"),
            "training_data_kind": data.get("training_data_kind"),
            "split_sizes": data.get("split_sizes"),
            "dataset_sha256": data.get("dataset_sha256"),
            "split_sha256": data.get("split_sha256"),
            "test_overall": {"accuracy": finite_metric(data["test"]["accuracy"], "accuracy"),
                             "macro_f1": finite_metric(data["test"]["macro_f1"], "macro-F1"),
                             "rows": sum(item["rows"] for item in data["test_by_language"].values())},
            "test_by_language": {
                language: {"accuracy": finite_metric(score["accuracy"], "accuracy"),
                           "macro_f1": finite_metric(score["macro_f1"], "macro-F1"),
                           "rows": int(score["rows"])}
                for language, score in data["test_by_language"].items()
            },
            "history": history,
            "best_epoch": find_best_epoch(history),
        }
    if models["xlmr"]["split_sizes"]["train"] != models["mbert"]["split_sizes"]["train"]:
        raise ValueError("XLM-R and mBERT training split sizes differ from expected shared study")
    if models["muril"]["test_by_language"].keys() != {"hi-en"}:
        raise ValueError("MuRIL should have Hindi-English-only results")
    return models, {"metric_cross_checks": crosschecks,
                    "mbert_metadata_matches_metrics": True}


def model_comparison(models: dict) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5,
                         "axes.titlesize": 12, "axes.labelsize": 10.5,
                         "xtick.labelsize": 10, "ytick.labelsize": 9.5,
                         "legend.fontsize": 9.5, "figure.facecolor": "white",
                         "axes.facecolor": "white", "savefig.facecolor": "white"})
    fig, axes = plt.subplots(3, 1, figsize=(6.8, 8.45), sharey=True)
    panel_specs = [
        ("hi-en", ["xlmr", "mbert", "muril"], "A  Hindi–English test", 1000),
        ("fr-en", ["xlmr", "mbert"], "B  English–French test", 1000),
        ("overall", ["xlmr", "mbert"], "C  Bilingual overall (XLM-R and mBERT)", 2000),
    ]
    width = 0.34
    for ax, (scope, encoders, title, n) in zip(axes, panel_specs):
        if scope == "overall":
            get_score = lambda encoder, metric: models[encoder]["test_overall"][metric]
        else:
            get_score = lambda encoder, metric: models[encoder]["test_by_language"][scope][metric]
        x = list(range(len(encoders)))
        for offset, metric in ((-width / 2, "accuracy"), (width / 2, "macro_f1")):
            values = [get_score(encoder, metric) for encoder in encoders]
            bars = ax.bar([position + offset for position in x], values, width=width,
                          color=METRIC_COLORS[metric], edgecolor="white", linewidth=0.8,
                          label="Accuracy" if metric == "accuracy" else "Macro-F1", zorder=3)
            ax.bar_label(bars, labels=[f"{value:.1%}" for value in values], padding=2,
                         fontsize=8.5, color="#243044")
        ax.set_xticks(x, [LABELS[encoder] for encoder in encoders])
        ax.set_ylim(0, 1.0)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.set_ylabel("Score")
        ax.set_title(f"{title}  ·  n = {n:,} per model", loc="left", fontweight="bold", pad=8)
        ax.grid(axis="y", color="#DCE2EA", linewidth=0.7, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#9AA5B1")
        ax.tick_params(axis="x", length=0, pad=5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 0.98), frameon=False)
    fig.suptitle("Saved synthetic test results", x=0.04, y=0.998, ha="left", fontsize=14, fontweight="bold")
    fig.text(0.04, 0.008,
             "All scores are from generated test phrases, not an independent human benchmark. "
             "XLM-R and mBERT used 16k training phrases; MuRIL used 8k and has hi-en results only.",
             ha="left", va="bottom", fontsize=8.1, color="#4B5563", wrap=True)
    fig.tight_layout(rect=(0.035, 0.045, 0.99, 0.955), h_pad=1.2)
    fig.savefig(FIGURES / "model_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def validation_history(models: dict) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5,
                         "axes.titlesize": 12, "axes.labelsize": 10.5,
                         "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
                         "legend.fontsize": 9.3, "figure.facecolor": "white",
                         "axes.facecolor": "white", "savefig.facecolor": "white"})
    fig, (ax_f1, ax_loss) = plt.subplots(2, 1, figsize=(6.8, 5.1), sharex=True)
    epochs = [item["epoch"] for item in models["xlmr"]["history"]]
    for key in ("xlmr", "mbert", "muril"):
        history = models[key]["history"]
        if [item["epoch"] for item in history] != epochs:
            raise ValueError("Epoch histories do not share a comparable 3-epoch schedule")
        color = COLORS[key]
        values = [item["validation_macro_f1"] for item in history]
        ax_f1.plot(epochs, values, marker="o", markersize=5, linewidth=2, color=color,
                   label=LABELS[key], zorder=3)
        best = models[key]["best_epoch"]
        ax_f1.scatter([best["epoch"]], [best["validation_macro_f1"]], marker="*", s=105,
                      color="#D55E00", edgecolor="white", linewidth=0.7, zorder=5)
        ax_loss.plot(epochs, [item["train_loss"] for item in history], marker="o", markersize=5,
                     linewidth=2, color=color, label=LABELS[key], zorder=3)
    ax_f1.set_ylim(0.70, 1.0)
    ax_f1.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax_f1.set_ylabel("Validation macro-F1")
    ax_f1.set_title("Validation macro-F1 · star marks each model’s best epoch", loc="left", fontweight="bold")
    ax_loss.set_yscale("log")
    ax_loss.set_ylabel("Training loss (log scale)")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_xticks(epochs)
    ax_loss.set_title("Training loss", loc="left", fontweight="bold")
    for ax in (ax_f1, ax_loss):
        ax.grid(axis="y", color="#DCE2EA", linewidth=0.7, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#9AA5B1")
    ax_f1.legend(ncol=3, loc="lower left", frameon=False)
    ax_loss.legend(ncol=3, loc="upper right", frameon=False)
    fig.suptitle("Recorded training histories", x=0.04, y=0.995, ha="left", fontsize=14, fontweight="bold")
    fig.text(0.04, 0.006,
             "The runs used synthetic data. XLM-R/mBERT: 16k train, 2k validation; "
             "MuRIL: 8k train, 1k validation. The validation axis spans 70–100%.",
             ha="left", va="bottom", fontsize=8.2, color="#4B5563")
    fig.tight_layout(rect=(0.04, 0.055, 0.99, 0.95), h_pad=1.25)
    fig.savefig(FIGURES / "validation_history.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    models, verification = build_data()
    model_comparison(models)
    validation_history(models)
    result = {
        "title": "October 23 model review figures",
        "metric_scope": "saved synthetic-only test metrics; not a real-world benchmark",
        "sources": [model["metrics_file"] for model in models.values()],
        "colab_cross_checks": verification["metric_cross_checks"],
        "mbert_metadata_matches_metrics": verification["mbert_metadata_matches_metrics"],
        "models": models,
        "chart_artifacts": ["figures/model_comparison.png", "figures/validation_history.png"],
        "challenge_diagnostics": {**CHALLENGE_DIAGNOSTICS, "included_in_figures": False},
        "interpretation_limits": [
            "All saved test scores and training histories are from synthetic-only data.",
            "MuRIL used a separate 8k/1k/1k hi-en synthetic split; XLM-R and mBERT used 16k/2k/2k bilingual synthetic splits.",
            "MuRIL is not shown in the bilingual overall or English-French panels because no corresponding results are recorded.",
            "These figures do not establish accuracy on independently reviewed human data.",
            "Assistant-authored diagnostic draft scores are reported separately and are not subtracted from saved test scores because the sets are unmatched.",
        ],
    }
    (OUT / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"artifacts": result["chart_artifacts"], "best_epochs": {key: value["best_epoch"] for key, value in models.items()},
                      "scores": {key: {"hi-en": value["test_by_language"]["hi-en"],
                                       "fr-en": value["test_by_language"].get("fr-en"),
                                       "overall": value["test_overall"]} for key, value in models.items()}}, indent=2))


if __name__ == "__main__":
    main()
