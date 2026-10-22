"""Read-only multilingual emotion encoder results, backed by saved metrics."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from hinglish_emotion.tasks import EMOTION_LABELS
from training.encoder_results import MBERT, XLMR, load_encoder_results

MURIL_CONTRACT = Path("colab/muril_study/study_contract.json")
MURIL_BASELINE = Path("colab/muril_study/xlmr_baseline_metrics.json")
MURIL = Path("models/muril_hi_en_emotion/metrics.json")


def _render_table_and_download(rows, label, filename):
    frame = pd.DataFrame(rows)
    st.dataframe(frame, hide_index=True, width="stretch")
    st.download_button(label, frame.to_csv(index=False).encode("utf-8-sig"),
                       file_name=filename, mime="text/csv", key=f"encoder_results_{filename}")


def _metrics_view(rows, per_class, *, chart_style=None):
    summary = pd.DataFrame(rows)
    wide = summary.melt(id_vars="Model", value_vars=["Accuracy", "Macro-F1"],
                        var_name="Metric", value_name="Score")
    fig = px.bar(wide, x="Model", y="Score", color="Metric", barmode="group", range_y=[0, 1],
                 color_discrete_map={"Accuracy": "#365f91", "Macro-F1": "#238579"},
                 title="Held-out test scores")
    if chart_style:
        try:
            chart_style(fig, 360)
        except (TypeError, AttributeError):
            pass
    st.plotly_chart(fig, width="stretch")
    _render_table_and_download(rows, "Download summary (CSV)", "encoder_results_summary.csv")
    per_frame = pd.DataFrame(per_class)
    st.bar_chart(per_frame.pivot(index="Emotion", columns="Model", values="F1"), y_label="F1", x_label="Emotion")
    _render_table_and_download(per_class, "Download per-emotion results (CSV)", "encoder_results_per_emotion.csv")


def _read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read {path.as_posix()}: {exc}") from exc


def _muril_hi_en(root, contract):
    """Use the existing strict MuRIL study comparator, which binds test identities."""
    muril_contract = _read_json(root / MURIL_CONTRACT)
    baseline = _read_json(root / MURIL_BASELINE)
    metrics = _read_json(root / MURIL)
    if not muril_contract or not baseline or not metrics:
        return None
    digest = hashlib.sha256(json.dumps(baseline, sort_keys=True).encode()).hexdigest()
    if digest != contract["baseline_metrics_sha256"]:
        raise ValueError("MuRIL study XLM-R baseline is not the baseline verified by the multilingual study")
    from training.muril_study import compare_runs
    # That module validates against its own canonical hi-en test contract.
    report = compare_runs(muril_contract, baseline, metrics)
    row = next(item for item in report["rows"] if item["Model"] == "MuRIL Hindi-English")
    classes = [item for item in report["per_class"] if item["Model"] == "MuRIL Hindi-English"]
    return row, classes


def render_encoder_results(root: Path, chart_style=None) -> None:
    """Display verified XLM-R/mBERT scores and optional MuRIL hi-en results."""
    root = Path(root)
    st.subheader("Multilingual emotion encoder results")
    st.caption("Saved test metrics only. This panel does not load model weights or run predictions.")
    try:
        result = load_encoder_results(root)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        st.warning(f"Encoder results could not be verified: {exc}")
        return
    if result["status"] == "pending":
        if "baseline" in result:
            st.metric("Verified XLM-R bilingual accuracy", f"{result['baseline']['accuracy']:.1%}")
            st.metric("Verified XLM-R bilingual macro-F1", f"{result['baseline']['macro_f1']:.4f}")
        st.info(result["message"])
        st.warning("These scores use synthetic phrases and do not establish real-world emotion accuracy.")
        return

    language = st.selectbox("Test language", ["hi-en", "fr-en", "Bilingual overall"],
                            format_func={"hi-en": "Hindi-English", "fr-en": "English-French",
                                         "Bilingual overall": "Bilingual overall"}.get,
                            key="encoder_results_language")
    muril_included = False
    if language == "Bilingual overall":
        xlmr_metrics = _read_json(root / XLMR / "metrics.json")
        rows = [
            {"Model": "XLM-R", "Accuracy": result["baseline"]["accuracy"], "Macro-F1": result["baseline"]["macro_f1"]},
            {"Model": "mBERT", "Accuracy": result["mbert"]["accuracy"], "Macro-F1": result["mbert"]["macro_f1"]},
        ]
        # Include verified MuRIL only in hi-en comparisons; it has no bilingual score.
        per_class = []
        for model, metrics in (("XLM-R", xlmr_metrics["test"]), ("mBERT", _read_json(root / MBERT / "metrics.json")["test"])):
            per_class.extend({"Model": model, "Emotion": label, "Precision": values["precision"],
                              "Recall": values["recall"], "F1": values["f1"], "Support": values["support"]}
                             for label, values in metrics["per_class"].items())
    else:
        pair = language
        xlmr_metrics = _read_json(root / XLMR / "metrics.json")
        if pair not in xlmr_metrics.get("test_by_language", {}):
            st.warning(f"Verified XLM-R {pair} test metrics are unavailable.")
            return
        xlmr = result["baseline_by_language"][pair]
        mbert = result["mbert_by_language"][pair]
        rows = [{"Model": model, "Accuracy": score["accuracy"], "Macro-F1": score["macro_f1"]}
                for model, score in (("XLM-R", xlmr), ("mBERT", mbert))]
        per_class = [{"Model": model, "Emotion": label, "Precision": score["per_class"][label]["precision"],
                      "Recall": score["per_class"][label]["recall"], "F1": score["per_class"][label]["f1"],
                      "Support": score["per_class"][label]["support"]}
                     for model, score in (("XLM-R", xlmr), ("mBERT", mbert)) for label in EMOTION_LABELS]
        if pair == "hi-en":
            try:
                muril = _muril_hi_en(root, result["contract"])
                if muril:
                    muril_included = True
                    rows.append({"Model": "MuRIL", "Accuracy": muril[0]["Accuracy"], "Macro-F1": muril[0]["Macro-F1"]})
                    per_class.extend({"Model": "MuRIL", "Emotion": item["Emotion"], "Precision": item["Precision"],
                                      "Recall": item["Recall"], "F1": item["F1"], "Support": item["Support"]}
                                     for item in muril[1])
            except (ValueError, KeyError, TypeError, OSError) as exc:
                st.info(f"MuRIL was excluded because its separate Hindi-English study did not verify: {exc}")

    for row in rows:
        cols = st.columns(2)
        cols[0].metric(f"{row['Model']} accuracy", f"{row['Accuracy']:.1%}")
        cols[1].metric(f"{row['Model']} macro-F1", f"{row['Macro-F1']:.4f}")
    _metrics_view(rows, per_class, chart_style=chart_style)
    st.warning("The results are synthetic-only. The mBERT and XLM-R runs use one seed, so this is not a significance study.")
    if muril_included:
        st.warning("MuRIL's verified comparison, when available, uses 8,000 Hindi-English training phrases; the combined mBERT and XLM-R runs use 16,000 phrases across Hindi-English and English-French. That is not a controlled encoder comparison.")
    if result.get("history"):
        trace = pd.DataFrame(result["history"])
        if {"epoch", "validation_macro_f1"}.issubset(trace.columns):
            st.subheader("mBERT validation macro-F1 by epoch")
            st.line_chart(trace.set_index("epoch")["validation_macro_f1"], y_label="Macro-F1")
    st.caption("Training accuracy by epoch was not saved in this run's metrics; test accuracy is shown above.")
