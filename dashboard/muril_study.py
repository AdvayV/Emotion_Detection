"""Read-only display for the controlled XLM-R and MuRIL emotion study."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


STUDY_DIR = Path("colab/muril_study")
BASELINE_PATH = Path("colab/muril_study/xlmr_baseline_metrics.json")
CONTRACT_PATH = Path("colab/muril_study/study_contract.json")
MURIL_PATH = Path("models/muril_hi_en_emotion/metrics.json")
MATCHED_PATH = Path("models/xlmr_hi_en_emotion/metrics.json")
COMPARISON_EXPORT = Path("Documents for review/Supporting results/muril_hi_en_comparison.zip")
MURIL_CHECKPOINT = Path("models/muril_hi_en_emotion")


def _read_json(path: Path):
    try:
        with path.open(encoding="utf-8") as source:
            return json.load(source)
    except FileNotFoundError:
        return None
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read {path.as_posix()}: {exc}") from exc


def _downloads(root: Path) -> None:
    st.subheader("MuRIL study training files")
    st.write("Train the Hindi–English MuRIL run in Colab, then install its metrics and checkpoint to compare it here.")
    folder = root / STUDY_DIR
    files = [
        ("TRAIN_MURIL_STUDY.ipynb", "Download MuRIL study notebook", "application/x-ipynb+json"),
        ("muril_study.zip", "Download MuRIL study package", "application/zip"),
        ("START_HERE.md", "Download study guide", "text/markdown"),
    ]
    for name, label, mime in files:
        path = folder / name
        if path.is_file():
            st.download_button(label, path.read_bytes(), file_name=name, mime=mime, key=f"muril_study_{name}")


def _show_comparison(result, chart_style, *, report_only=False, checkpoint_installed=False):
    rows = result["rows"]
    per_class = result["per_class"]
    notes = result["notes"]
    if not rows:
        raise ValueError("The verified study returned no comparison rows.")
    summary = pd.DataFrame(rows)
    expected = {"Model", "Training phrases", "Validation phrases", "Test phrases", "Accuracy", "Macro-F1", "Training scope"}
    if not expected.issubset(summary.columns):
        raise ValueError("The study comparison is missing required summary fields.")

    if report_only:
        st.success("Colab results available; MuRIL checkpoint is installed" if checkpoint_installed else
                   "Colab results available; MuRIL checkpoint not installed")
        if not checkpoint_installed:
            st.info("The comparison report contains saved test metrics, so scores are available without model weights. To install the model, download the separate **muril_hi_en.zip** checkpoint from Colab, then run `python -m training.install_checkpoint \"C:/Users/Advay/Downloads/muril_hi_en.zip\"` from the project root.")

    warnings = [str(note) for note in notes if "synthetic" in str(note).lower() or "does not isolate" in str(note).lower() or "different training" in str(note).lower()]
    other_notes = [str(note) for note in notes if note not in warnings]
    for note in warnings:
        st.warning(note)
    if other_notes:
        st.info(" ".join(other_notes))
    st.caption(f"Comparison type: {result.get('comparison_type', 'verified study')}")

    cards = st.columns(min(4, len(rows) * 2))
    # Keep the two main runs visible even when an optional matched run is present.
    for index, row in enumerate(rows[:2]):
        cards[index * 2].metric(f"{row['Model']} accuracy", f"{row['Accuracy']:.1%}")
        cards[index * 2 + 1].metric(f"{row['Model']} macro-F1", f"{row['Macro-F1']:.4f}")

    chart_data = summary.melt(id_vars="Model", value_vars=["Accuracy", "Macro-F1"], var_name="Metric", value_name="Score")
    chart = px.bar(chart_data, x="Model", y="Score", color="Metric", barmode="group", text="Score", range_y=[0, 1], color_discrete_sequence=["#6264CF", "#46A390"])
    chart.update_traces(texttemplate="%{y:.1%}", textposition="outside", cliponaxis=False)
    chart.update_yaxes(tickformat=".0%", range=[0, 1], title=None)
    chart.update_xaxes(title=None)
    st.plotly_chart(chart_style(chart, 380), width="stretch", config={"displayModeBar": False})
    st.dataframe(summary, hide_index=True, width="stretch", column_config={
        "Accuracy": st.column_config.NumberColumn(format="percent"),
        "Macro-F1": st.column_config.NumberColumn(format="%.4f"),
    })
    if per_class:
        detail = pd.DataFrame(per_class)
        required_detail = {"Model", "Emotion", "F1", "Precision", "Recall", "Support"}
        if not required_detail.issubset(detail.columns):
            raise ValueError("The study comparison is missing per-emotion scores.")
        st.subheader("Per-emotion results")
        f1_chart = px.bar(detail, x="Emotion", y="F1", color="Model", barmode="group", range_y=[0, 1],
                          color_discrete_sequence=["#6264CF", "#46A390", "#D48A42"],
                          hover_data=["Support", "Precision", "Recall"])
        f1_chart.update_yaxes(tickformat=".0%", range=[0, 1], title="F1 score")
        f1_chart.update_xaxes(title=None)
        st.plotly_chart(chart_style(f1_chart, 380), width="stretch", config={"displayModeBar": False})
        st.dataframe(detail, hide_index=True, width="stretch", column_config={
            "Precision": st.column_config.NumberColumn(format="%.4f"),
            "Recall": st.column_config.NumberColumn(format="%.4f"),
            "F1": st.column_config.NumberColumn(format="%.4f"),
        })
    st.download_button("Download comparison (CSV)", summary.to_csv(index=False).encode("utf-8"), file_name="xlmr_vs_muril_comparison.csv", mime="text/csv")
    if per_class:
        st.download_button("Download per-emotion results (CSV)", pd.DataFrame(per_class).to_csv(index=False).encode("utf-8"), file_name="xlmr_vs_muril_per_emotion.csv", mime="text/csv")
    st.caption("Scores come from saved test results on synthetic phrases and do not establish real-world accuracy. Combined XLM-R used 16,000 phrases across Hindi–English and English–French; MuRIL used 8,000 Hindi–English phrases, so training scope differs.")


def render_muril_study(root: Path, chart_style) -> None:
    """Render study metrics only after training metrics pass the shared contract."""
    root = Path(root)
    st.subheader("Hindi–English encoder comparison")
    st.write("Compare the saved XLM-R baseline with a MuRIL run on the study’s shared Hindi–English test set.")
    st.caption("This view reads saved metrics only. It does not load model weights or start training.")

    contract_path, baseline_path = root / CONTRACT_PATH, root / BASELINE_PATH
    muril_path, matched_path = root / MURIL_PATH, root / MATCHED_PATH
    try:
        contract = _read_json(contract_path)
        baseline = _read_json(baseline_path)
        muril = _read_json(muril_path)
        matched = _read_json(matched_path)
    except ValueError as exc:
        st.warning(str(exc))
        _downloads(root)
        return

    if contract is None or baseline is None:
        st.info("The saved study contract or XLM-R baseline metrics are unavailable, so baseline scores cannot be verified.")
        st.warning("The study uses synthetic phrases. Its scores do not establish real-world accuracy or generalization.")
        _downloads(root)
        return

    try:
        from training.muril_study import compare_runs, validate_baseline
        baseline_score = validate_baseline(contract, baseline)
    except (ValueError, KeyError, TypeError, OSError, ImportError) as exc:
        st.warning(f"The XLM-R baseline could not be verified: {exc}")
        st.warning("The study uses synthetic phrases. Its scores do not establish real-world accuracy or generalization.")
        _downloads(root)
        return

    if muril is None:
        export_path = root / COMPARISON_EXPORT
        if not export_path.is_file():
            export_path = root / "muril_hi_en_comparison.zip"  # Older project layouts.
        if export_path.is_file():
            try:
                from training.muril_study import load_comparison_export
                result = load_comparison_export(export_path, contract, baseline)
                from hinglish_emotion.model_catalog import checkpoint_ready
                installed = checkpoint_ready(root / MURIL_CHECKPOINT)
                _show_comparison(result, chart_style, report_only=True, checkpoint_installed=installed)
                _downloads(root)
                return
            except (ValueError, KeyError, TypeError, OSError, ImportError) as exc:
                st.warning(f"The Colab comparison report could not be verified: {exc}")
                _downloads(root)
                return
        st.info("MuRIL has not been trained or installed for this study yet. Missing MuRIL results are not treated as zero.")
        metrics = st.columns(3)
        metrics[0].metric("Verified XLM-R hi-en accuracy", f"{baseline_score['accuracy']:.1%}")
        metrics[1].metric("Verified XLM-R hi-en macro-F1", f"{baseline_score['macro_f1']:.4f}")
        metrics[2].metric("Verified test phrases", baseline_score["rows"])
        st.warning("The study uses synthetic phrases. Its scores do not establish real-world accuracy or generalization.")
        st.warning("Combined XLM-R training used 16,000 phrases across two language mixes; MuRIL uses 8,000 Hindi–English phrases. The shared test alone does not isolate the encoder effect.")
        _downloads(root)
        return

    try:
        result = compare_runs(contract, baseline, muril, matched)
        from hinglish_emotion.model_catalog import checkpoint_ready
        if checkpoint_ready(root / MURIL_CHECKPOINT):
            st.success("MuRIL checkpoint installed; saved training results verified. Choose MuRIL with Hindi–English in Try a phrase for live predictions.")
        else:
            st.info("Saved MuRIL training results verified; model weights are unavailable for live predictions.")
        _show_comparison(result, chart_style)
        _downloads(root)
    except (ValueError, KeyError, TypeError, OSError, ImportError) as exc:
        st.warning(f"The study results could not be verified: {exc}")
        _downloads(root)
