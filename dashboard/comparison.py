"""One place to score compatible encoders on identical supplied phrases."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.disagreement import available_checkpoints
from hinglish_emotion.batch import normalize_records, read_frame
from hinglish_emotion.emotions import LocalEmotionClassifier
from hinglish_emotion.model_comparison import comparison_rows, compare_models
from hinglish_emotion.tasks import EMOTION_LABELS, EMOTION_NAMES, LANGUAGE_PAIRS


def local_comparison_classifier(path):
    return LocalEmotionClassifier(path, device="cpu")


def _clear_emotion_caches():
    from dashboard.emotions import trained_classifier, trained_sarcasm_classifier
    trained_classifier.clear()
    trained_sarcasm_classifier.clear()


def render_model_comparison(header, chart_style, root=None, clear_sentiment_cache=None):
    root = Path(root) if root else Path(__file__).resolve().parents[1]
    header("Model comparison", "Run the same dataset through the installed emotion models, score them against supplied labels, and inspect every prediction together.")
    st.caption("Local CPU inference, one encoder at a time. This run compares base model outputs; Qwen and demo-rule fallback are not used.")
    pair = st.selectbox("Comparison language", list(LANGUAGE_PAIRS), format_func=LANGUAGE_PAIRS.get, key="comparison_language")
    installed = {name: (name, path, version) for name, path, version in available_checkpoints(root, pair)}
    names = [name for name in ("XLM-R", "mBERT", "MuRIL") if name in installed]
    selected_names = st.multiselect("Models to score", names, default=names, key=f"comparison_models_{pair}")
    st.caption("Available for this language: " + (", ".join(names) or "no trained checkpoints"))
    if pair == "fr-en":
        st.info("English-French compares XLM-R and mBERT. MuRIL is restricted to Hindi-English; it is never given an invented French score.")
    elif len(names) < 3:
        st.info("Install the missing compatible checkpoints to compare all three encoders. Missing models are not scored as zero.")

    upload = st.file_uploader("Comparison dataset (CSV or XLSX)", type=["csv", "xlsx"], key="comparison_upload")
    language_folder = "hindi_english" if pair == "hi-en" else "english_french"
    sources = {
        "Saved synthetic test": root / f"data/training/multilingual_bootstrap_v1/{language_folder}/test.csv",
        "Separate checking phrases (unreviewed draft labels)": root / f"evaluation_set/{language_folder}_100.csv",
        "Small demonstration": root / "data/demo/emotions/mixed_emotions_demo.csv",
    }
    if upload is not None:
        content, source = upload.getvalue(), upload.name
    else:
        available = {name: path for name, path in sources.items() if path.is_file()}
        if not available:
            st.info("Upload a CSV or XLSX with a text column. Add an emotion column to measure accuracy and macro-F1.")
            return
        selected_source = st.selectbox("Comparison dataset", list(available), key=f"comparison_dataset_{pair}")
        file = available[selected_source]
        content, source = file.read_bytes(), file.name
    digest = hashlib.sha256(content).hexdigest()
    try:
        frame = read_frame(content, source)
        columns = list(frame.columns)
        def default_index(options, candidates):
            return next((i for i, option in enumerate(options) if option.lower() in candidates), 0)
        with st.expander("Map comparison columns"):
            text_col = st.selectbox("Phrase column", columns, index=default_index(columns, {"text", "message", "phrase", "sentence"}), key=f"comparison_text_{digest}")
            labels = ["No expected labels", *columns]
            label_col = st.selectbox("Gold emotion column", labels, index=default_index(labels, {"emotion", "label", "expected_emotion", "actual"}), key=f"comparison_label_{digest}")
            languages = ["Use comparison language", *columns]
            pair_col = st.selectbox("Comparison language column", languages, index=default_index(languages, {"language_pair", "language", "language_mix"}), key=f"comparison_pair_{digest}")
        rows = normalize_records(frame.to_dict("records"), pair, text_col, "" if label_col == "No expected labels" else label_col, "" if pair_col == "Use comparison language" else pair_col)
        test_only = st.checkbox("Compare test rows only when split metadata is present", value=True, key="comparison_test_only")
        exclude_overlap = st.checkbox("Exclude phrases found in any selected model's training/validation data from every model's score", value=True, key="comparison_exclude_overlap")
        selected_rows, scope, skipped = comparison_rows(rows, pair, test_only)
    except (ValueError, OSError, UnicodeError) as exc:
        st.error(f"Could not prepare comparison: {exc}")
        return
    with st.expander("CPU settings"):
        batch_size = st.selectbox("Comparison batch size", [1, 2, 4, 8], index=2, key="comparison_batch_size")
        threads = st.selectbox("CPU threads", [1, 2, 4], index=1, key="comparison_threads")
    first, second, third = st.columns(3)
    first.metric("Identical phrases per model", len(selected_rows))
    second.metric("Rows with expected emotions", sum(bool(row["emotion"]) for row in selected_rows))
    third.metric("Selected models", len(selected_names))
    st.caption(f"Source: {source} | scope: {scope} | other-language rows skipped: {skipped}")
    with st.expander("Preview comparison input"):
        st.dataframe(pd.DataFrame(selected_rows).head(50), hide_index=True, width="stretch")
    if not any(row["emotion"] for row in selected_rows):
        st.info("No expected labels: model outputs and disagreement can be compared, but accuracy and F1 cannot be calculated.")
    selected = [installed[name] for name in selected_names]
    if len(selected) < 2:
        st.info("Choose at least two installed models to enable comparison.")
    signature = (digest, source, pair, text_col, label_col, pair_col, test_only, exclude_overlap, batch_size, threads,
                 tuple((name, str(path), version) for name, path, version in selected))
    if st.button("Run and score selected models", type="primary", disabled=len(selected) < 2, key="comparison_run"):
        st.session_state.pop("model_comparison_result", None)
        progress = st.progress(0.0, text="Preparing shared dataset...")
        old_threads = None
        try:
            import torch
            old_threads = torch.get_num_threads()
            torch.set_num_threads(threads)
            if clear_sentiment_cache:
                clear_sentiment_cache()
            result = compare_models(rows, pair, selected, classifier_factory=local_comparison_classifier,
                test_only=test_only, exclude_overlap=exclude_overlap, batch_size=batch_size,
                clear_shared_cache=_clear_emotion_caches,
                progress=lambda done, total, message: progress.progress(done / total, text=f"{message}: {done:,} / {total:,} model-phrase predictions"))
            result["input_file_sha256"] = digest
            result["input_filename"] = source
            result["runtime_settings"] = {"device": "cpu", "threads": threads, "batch_size": batch_size}
            st.session_state.model_comparison_result = {"report": result, "signature": signature}
            st.session_state.comparison_phrase_number = 1
            st.session_state.comparison_search = ""
        except (ImportError, ValueError, RuntimeError, OSError, TypeError) as exc:
            st.error(f"Comparison could not finish: {exc}. No model was replaced by demo rules.")
        finally:
            if old_threads is not None:
                torch.set_num_threads(old_threads)
            progress.empty()
    saved = st.session_state.get("model_comparison_result")
    if not saved:
        return
    if saved["signature"] != signature:
        st.info("Data, labels, language, models or settings changed. Run comparison again for matching results.")
        return
    _render_report(saved["report"], chart_style)


def _render_report(report, chart_style):
    st.divider()
    st.subheader("Results on the same phrases")
    st.caption(f"Every model is scored on the same {report['scored_rows']:,} labelled eligible rows. {report['selected_rows']:,} rows were predicted by every selected model.")
    if report["synthetic_dataset"] or report["illustrative_labels"]:
        st.warning("These are synthetic or unreviewed draft labels. Results describe this supplied dataset and do not establish real-world accuracy.")
    elif not report["all_labels_declared_human_reviewed"]:
        st.info("The supplied labels have no complete human-review declaration. Scores depend on their quality.")
    if report["training_overlap_rows"]:
        st.warning(f"{report['training_overlap_rows']:,} phrases match at least one model's training/validation guard. Excluded from every score: {report['excluded_overlap_rows']:,}.")
    missing_guards = [model["name"] for model in report["models"] if not model["overlap_check_available"]]
    if missing_guards:
        st.warning("Training-overlap checks are unavailable for " + ", ".join(missing_guards) + ". Held-out status cannot be confirmed for those models.")
    summary = pd.DataFrame([{
        "Model": model["name"], "Scored rows": model["metrics"]["rows"] if model["metrics"] else 0,
        "Accuracy": model["metrics"]["accuracy"] if model["metrics"] else None,
        "Macro-F1": model["metrics"]["macro_f1"] if model["metrics"] else None,
        "Correct": model["metrics"]["correct"] if model["metrics"] else None,
        "Truncated phrases": model["truncated_rows"], "Load seconds": model["load_seconds"],
        "Prediction seconds": model["prediction_seconds"], "ms per phrase": model["milliseconds_per_phrase"],
    } for model in report["models"]])
    score_tab, rows_tab, emotion_tab, pair_tab, details_tab = st.tabs(["Scoreboard", "All model outputs", "Per-emotion metrics", "Paired mistakes", "Run details"])
    with score_tab:
        if report["scored_rows"]:
            ranked = summary.sort_values(["Macro-F1", "Accuracy"], ascending=False, kind="stable")
            st.dataframe(ranked, hide_index=True, width="stretch", column_config={"Accuracy":st.column_config.NumberColumn(format="percent"),"Macro-F1":st.column_config.NumberColumn(format="%.4f")})
            st.caption("Compare macro-F1 across all seven classes alongside accuracy. A higher score on this file is not a claim of universal superiority; ties and small differences need independent evaluation.")
            chart = summary.melt(id_vars="Model", value_vars=["Accuracy", "Macro-F1"], var_name="Metric", value_name="Score")
            st.plotly_chart(chart_style(px.bar(chart, x="Model", y="Score", color="Metric", barmode="group", range_y=[0,1], text_auto=".1%")), width="stretch", key="comparison_score_chart")
            missing = [label for label in EMOTION_LABELS if label not in report["label_counts"]]
            if missing:
                st.info("Missing scored classes: " + ", ".join(missing) + ". Macro-F1 still averages all seven classes, including zero-support classes.")
        else:
            st.info("No eligible labelled rows remain. Predictions are available, but no accuracy or F1 score is reported.")
            st.dataframe(summary, hide_index=True, width="stretch")
        st.caption("Timing includes CPU prediction and preprocessing; checkpoint loading is measured separately. Neither is a hardware-independent encoder benchmark.")
    predictions = pd.DataFrame(report["predictions"])
    with rows_tab:
        search = st.text_input("Search compared phrases", key="comparison_search")
        view = st.radio("Rows to inspect", ["All phrases", "Models disagree", "At least one wrong prediction"], horizontal=True, key="comparison_row_filter")
        visible = predictions.copy()
        if search:
            visible = visible[visible["text"].str.contains(search, case=False, regex=False)]
        if view == "Models disagree":
            visible = visible[visible["models_disagree"]]
        elif view == "At least one wrong prediction":
            visible = visible[visible[[f"{m['name']} correct" for m in report["models"]]].eq(False).any(axis=1)]
        primary_columns = ["row_number", "text", "expected_emotion", *[f"{m['name']} {field}" for m in report["models"] for field in ("predicted", "confidence", "correct")], "models_disagree", "included_in_scoring", "training_overlap_any_model"]
        st.dataframe(visible[primary_columns], hide_index=True, width="stretch")
        st.caption("Confidence is the base model's top score, not measured accuracy. Full exports include all seven scores, provenance and overlap flags.")
        with st.expander("Inspect one phrase's seven emotion scores"):
            number = st.number_input("Phrase number", min_value=1, max_value=len(predictions), value=1, step=1, key="comparison_phrase_number")
            row = report["predictions"][number-1]
            st.write(row["text"])
            st.caption("Expected emotion: " + (row["expected_emotion"] or "not supplied"))
            distributions = pd.DataFrame({"Emotion":[EMOTION_NAMES[label] for label in EMOTION_LABELS], **{m["name"]:[row[f"{m['name']} score_{label}"] for label in EMOTION_LABELS] for m in report["models"]}})
            st.dataframe(distributions, hide_index=True, width="stretch")
    with emotion_tab:
        scored_models = [model for model in report["models"] if model["metrics"]]
        if not scored_models:
            st.info("Per-emotion scores and confusion matrices require eligible expected labels.")
        else:
            class_rows = [{"Model":model["name"],"Emotion":label,**values} for model in scored_models for label,values in model["metrics"]["per_class"].items()]
            st.dataframe(pd.DataFrame(class_rows), hide_index=True, width="stretch")
            st.plotly_chart(chart_style(px.bar(pd.DataFrame(class_rows), x="Emotion", y="f1", color="Model", barmode="group", range_y=[0,1])), width="stretch", key="comparison_class_chart")
            tabs = st.tabs([model["name"] for model in scored_models])
            for tab, model in zip(tabs, scored_models):
                with tab:
                    matrix = pd.DataFrame(model["metrics"]["confusion_matrix"]).T.loc[list(EMOTION_LABELS),list(EMOTION_LABELS)]
                    st.plotly_chart(chart_style(px.imshow(matrix, text_auto=True, aspect="auto", color_continuous_scale="Purples", labels={"x":"Predicted emotion","y":"Expected emotion"})), width="stretch", key="comparison_matrix_"+model["name"])
    with pair_tab:
        st.dataframe(pd.DataFrame(report["paired_comparisons"]), hide_index=True, width="stretch")
        st.caption("Left-only and right-only correct counts use the same scored rows. Agreement uses all selected phrases and does not show which model is correct. These counts are descriptive, not a significance test or a voting classifier.")
    with details_tab:
        st.json({key:value for key,value in report.items() if key not in {"predictions","models","paired_comparisons"}})
        st.dataframe(pd.DataFrame([{key:value for key,value in model.items() if key!="metrics"} for model in report["models"]]), hide_index=True, width="stretch")
    st.download_button("Download comparison scores (CSV)", summary.to_csv(index=False).encode("utf-8-sig"), "model_comparison_scores.csv", "text/csv", key="comparison_scores_download", on_click="ignore")
    st.download_button("Download aligned model outputs (CSV)", predictions.to_csv(index=False).encode("utf-8-sig"), "model_comparison_predictions.csv", "text/csv", key="comparison_predictions_download", on_click="ignore")
    st.download_button("Download complete comparison report (JSON)", json.dumps(report,indent=2,ensure_ascii=False).encode("utf-8"), "model_comparison_report.json", "application/json", key="comparison_report_download", on_click="ignore")
