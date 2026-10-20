"""Dataset upload, prediction and accuracy display for seven emotions."""
import hashlib
import json
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st
from hinglish_emotion.batch import read_frame, normalize_records, predict_dataset
from hinglish_emotion.tasks import EMOTION_LABELS, LANGUAGE_PAIRS


def render_file_predictions(root, choice, current_pair, resolver, chart_style):
    st.subheader("Predict a dataset and check its labels")
    st.write("Upload CSV or XLSX. A text column is enough for predictions. Add expected emotion labels to calculate accuracy and macro-F1.")
    upload = st.file_uploader("Dataset file (CSV or XLSX)", type=["csv", "xlsx"], key="emotion_file")
    sources = {
        "Demo examples": root / "data/demo/emotions/mixed_emotions_demo.csv",
        "Current Hinglish synthetic test": root / "data/training/synthetic_hinglish_8000/test.csv",
        "Multilingual synthetic test": root / "data/training/multilingual_bootstrap_v1/test.csv",
    }
    if upload:
        content, name = upload.getvalue(), upload.name
    else:
        available = {label: path for label, path in sources.items() if path.is_file()}
        if not available:
            st.info("Upload a CSV or XLSX file to start prediction.")
            return
        selected = st.selectbox("Bundled dataset", list(available))
        content, name = available[selected].read_bytes(), available[selected].name
    digest = hashlib.sha256(content).hexdigest()
    try:
        frame = read_frame(content, name)
        columns = list(frame.columns)
        def index_for(options, candidates):
            return next((i for i, option in enumerate(options) if option.lower() in candidates), 0)
        with st.expander("Map file columns", expanded=False):
            text_col = st.selectbox("Text column", columns, index=index_for(columns, {"text", "message", "phrase", "sentence"}), key=f"text_column_{digest}")
            label_options = ["No expected labels", *columns]
            label_col = st.selectbox("Expected emotion column", label_options, index=index_for(label_options, {"emotion", "label", "expected_emotion", "actual"}), key=f"label_column_{digest}")
            pair_options = ["Use selected language", *columns]
            pair_col = st.selectbox("Language column", pair_options, index=index_for(pair_options, {"language_pair", "language", "language_mix"}), key=f"pair_column_{digest}")
        rows = normalize_records(frame.to_dict("records"), current_pair, text_col, "" if label_col == "No expected labels" else label_col, "" if pair_col == "Use selected language" else pair_col)
    except Exception as exc:
        st.error(f"Could not use this file: {exc}")
        st.code("text,emotion,language_pair\nAaj main bahut khush hoon,joy,hi-en\nJe suis triste today,sadness,fr-en")
        return
    present = [pair for pair in LANGUAGE_PAIRS if any(row["language_pair"] == pair for row in rows)]
    defaults = present if upload else [current_pair] if current_pair in present else present
    selected_pairs = st.multiselect("Evaluate language mixes", present, default=defaults, format_func=LANGUAGE_PAIRS.get, key=f"file_pairs_{digest}")
    selected_rows = [row for row in rows if row["language_pair"] in selected_pairs]
    test_only = st.checkbox("Use test rows only when the file declares train/validation/test splits", value=True, key="file_test_only")
    fallback = st.checkbox("Allow clearly labelled demo-rule fallback for an unavailable language model", value=False, key="batch_fallback")
    selected_models, issues = {}, []
    for pair in selected_pairs:
        path, metadata, issue = resolver(choice, pair)
        selected_models[pair] = path
        if issue:
            if fallback:
                st.warning(f"{LANGUAGE_PAIRS[pair]}: {issue} Demo rules will be used for these rows, not XLM-R.")
            else: issues.append(f"{LANGUAGE_PAIRS[pair]}: {issue}")
    for issue in issues: st.warning(issue)
    labelled = sum(bool(row["emotion"]) for row in selected_rows)
    c = st.columns(3)
    c[0].metric("Rows selected", len(selected_rows))
    c[1].metric("Rows with expected labels", labelled)
    c[2].metric("Language mixes", len(selected_pairs))
    with st.expander("Preview selected file"):
        st.dataframe(pd.DataFrame(selected_rows).head(100), hide_index=True, width="stretch")
        st.caption("Preview limited to 100 rows; all selected rows are processed, subject to the test-split option.")
    if not labelled:
        st.info("Prediction-only file: accuracy cannot be calculated without expected labels.")
    signature = (choice, current_pair, digest, tuple(selected_pairs), test_only, fallback, tuple((pair, str(path)) for pair, path in selected_models.items()))
    from dashboard.emotions import detector_for
    from hinglish_emotion.checkpoints import checkpoint_version
    signature += (tuple((pair, checkpoint_version(path) if path else None) for pair, path in selected_models.items()),)
    run_label = "Evaluate seven emotions" if labelled else "Predict file"
    if st.button(run_label, type="primary", disabled=bool(issues) or not selected_rows, key="run_emotion_file"):
        st.session_state.pop("emotion_evaluation", None)
        progress = st.progress(0, text="Preparing local predictions…")
        def provider(pair):
            path = selected_models[pair]
            try:
                return detector_for(str(path) if path else "")
            except (ValueError, RuntimeError, OSError) as exc:
                if not fallback: raise
                st.warning(f"{LANGUAGE_PAIRS[pair]} checkpoint could not load ({exc}). Actual source for these predictions: demo rules.")
                return detector_for("")
        try:
            report = predict_dataset(selected_rows, provider, test_only, progress=lambda done,total: progress.progress(done/total, text=f"Processed {done} of {total} phrases"))
            st.session_state.emotion_evaluation = {"report": report, "signature": signature, "choice": choice}
        except (ValueError, RuntimeError, OSError) as exc:
            st.error(f"Prediction could not finish: {exc}")
        finally:
            progress.empty()
    saved = st.session_state.get("emotion_evaluation")
    if not saved: return
    if saved["signature"] != signature:
        st.info("File or settings changed. Run prediction again to see results for the current selection.")
        return
    report = saved["report"]
    st.divider()
    st.subheader("Results against the supplied labels")
    st.caption(f"Scope: {report['scope']}. Each prediction records the actual model or fallback used.")
    sources = pd.DataFrame(report["models_used"]).rename(columns={"language_pair":"Language", "source":"Actual source", "model":"Model", "rows":"Phrases"})
    sources["Language"] = sources["Language"].map(LANGUAGE_PAIRS)
    sources["Actual source"] = sources["Actual source"].map({"trained_checkpoint":"Trained checkpoint", "lexical_demo":"Demo rules"}).fillna(sources["Actual source"])
    st.dataframe(sources,hide_index=True,width="stretch")
    if choice != "Demo lexical rules" and any(item["source"] == "lexical_demo" for item in report["models_used"]):
        st.warning("Some rows used demo-rule fallback. The overall score combines the actual sources listed above; it is not an XLM-R-only benchmark.")
    if report["synthetic_dataset"] or report["illustrative"]:
        st.warning("Generated or illustrative labels: these scores do not establish independent real-world accuracy.")
    if report["training_overlap_rows"]:
        st.warning(f"{report['training_overlap_rows']} phrases overlap this checkpoint's training/validation data. Scores on those phrases are not held-out accuracy.")
    if not all(report["overlap_check_available"].values()):
        st.caption("Training-overlap checks are unavailable for one or more selected models. Absence of a detected overlap is not proof of independence.")
    overall = report["overall"]
    if overall:
        c = st.columns(3)
        c[0].metric("Emotion accuracy", f"{overall['accuracy']:.1%}", help="Correct predictions divided by rows with expected labels.")
        c[1].metric("Seven-class macro-F1", f"{overall['macro_f1']:.1%}", help="Mean F1 over the seven fixed emotion classes; missing classes are listed below.")
        c[2].metric("Phrases evaluated", overall["rows"])
        if overall["missing_emotions"]: st.info("Missing test classes: " + ", ".join(overall["missing_emotions"]))
        summary = [{"Language": LANGUAGE_PAIRS[pair], "Accuracy": scores["accuracy"], "Macro-F1": scores["macro_f1"], "Labelled rows": scores["rows"]} for pair,scores in report["by_language"].items() if scores]
        st.dataframe(pd.DataFrame(summary), hide_index=True, width="stretch", column_config={"Accuracy": st.column_config.NumberColumn(format="percent"), "Macro-F1": st.column_config.NumberColumn(format="percent")})
        left,right = st.columns(2)
        with left:
            matrix = pd.DataFrame(overall["confusion_matrix"]).T.loc[list(EMOTION_LABELS), list(EMOTION_LABELS)]
            st.plotly_chart(chart_style(px.imshow(matrix,text_auto=True,aspect="auto",color_continuous_scale="Purples",labels={"x":"Predicted emotion","y":"Expected emotion"}),370),width="stretch")
        with right:
            st.dataframe(pd.DataFrame(overall["per_class"]).T.reset_index(names="Emotion"),hide_index=True,width="stretch",column_config={**{name:st.column_config.NumberColumn(format="%.3f") for name in ("precision","recall","f1")},"support":st.column_config.NumberColumn(format="%d")})
        st.caption("Confusion matrix: correct predictions lie on the diagonal. Precision measures label reliability; recall measures how many examples of an emotion were found.")
    else:
        st.metric("Phrases predicted", report["total_rows"])
        st.info("No expected labels were supplied, so no accuracy or F1 is reported.")
    predictions = pd.DataFrame(report["predictions"])
    with st.expander("Inspect incorrect predictions and actual model sources"):
        st.dataframe(predictions[predictions["correct"].eq(False)] if overall else predictions,hide_index=True,width="stretch")
    st.download_button("Download emotion predictions", predictions.to_csv(index=False).encode("utf-8"), file_name="emotion_predictions.csv",mime="text/csv")
    st.download_button("Download evaluation report", json.dumps(report,indent=2,ensure_ascii=False).encode("utf-8"),file_name="emotion_evaluation.json",mime="application/json")
