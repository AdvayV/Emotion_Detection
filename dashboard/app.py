from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.data import dataframe_report, read_dataset, report_dict, sarcasm_rate
from dashboard.evaluation import ModelEvaluation, evaluate_pipeline
from hinglish_emotion.intensity import IntensityEstimator
from hinglish_emotion.ollama_reviewer import DEFAULT_OLLAMA_MODEL, OllamaReviewer
from hinglish_emotion.pipeline import EmotionPipeline, RuleBasedClassifier
from hinglish_emotion.transformer_classifier import LocalTransformerClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DATA = PROJECT_ROOT / "data" / "hinglish_emotion_phrases.xlsx"
EXAMPLES = {
    "Positive with emphasis": "movie acchaaa thi 😊",
    "Negation": "service acchi nahi thi 😞",
    "Possible sarcasm": "wah kya service hai, 2 ghante late 😂",
    "Neutral": "theek tha",
}
LABEL_COLORS = {"positive": "#1B7F5A", "neutral": "#49627A", "negative": "#B5475A"}


def inject_theme() -> None:
    st.markdown(
        """
        <style>
          .stApp { background: #F6F8FC; }
          .block-container { max-width: 1180px; padding-top: 2.1rem; padding-bottom: 3rem; }
          h1, h2, h3 { color: #14233B; letter-spacing: -0.02em; }
          [data-testid="stMetric"] { background: #FFFFFF; border: 1px solid #DCE4F0; border-radius: 12px; padding: 0.8rem 1rem; }
          [data-testid="stMetricLabel"] { color: #52657D; }
          div[data-testid="stExpander"] { background: #FFFFFF; border: 1px solid #DCE4F0; border-radius: 12px; }
          .research-note { color: #52657D; font-size: 0.96rem; margin-bottom: 1.4rem; }
          .status-chip { display: inline-block; padding: 0.28rem 0.62rem; border-radius: 999px; font-size: 0.82rem; font-weight: 600; margin-right: 0.35rem; }
          .chip-positive { background: #DDF4E9; color: #126143; }
          .chip-neutral { background: #E8EFF7; color: #36516F; }
          .chip-negative { background: #FBE4E7; color: #8B2639; }
          .chip-warning { background: #FFF1D7; color: #8C5A00; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def status_chip(label: str, kind: str) -> str:
    return f'<span class="status-chip chip-{kind}">{label}</span>'


def build_pipeline(model_path: str, use_reviewer: bool, ollama_model: str) -> tuple[EmotionPipeline, str]:
    classifier = None
    mode = "Rule-based baseline"
    if model_path.strip():
        path = Path(model_path.strip())
        if not path.is_dir():
            raise RuntimeError("The local model folder was not found. Leave this empty to use the baseline.")
        classifier = LocalTransformerClassifier(path)
        mode = f"Local transformer: {path.name}"
    reviewer = OllamaReviewer(model=ollama_model) if use_reviewer else None
    return EmotionPipeline(classifier=classifier, reviewer=reviewer), mode


def render_overview() -> None:
    st.title("Hinglish Emotion Research Dashboard")
    st.markdown('<p class="research-note">A local workspace for transparent Hinglish sentiment analysis, sarcasm review, and dataset inspection.</p>', unsafe_allow_html=True)

    left, right = st.columns([1.25, 1])
    with left:
        st.subheader("Research flow")
        st.graphviz_chart(
            """digraph { graph [bgcolor=transparent, rankdir=LR, nodesep=0.45];
            node [shape=box, style=rounded, color=\"#9DB2CE\", fontcolor=\"#14233B\"];
            input [label=\"Hinglish message\"]; normalize [label=\"Normalize\nspelling + emphasis\"]; classify [label=\"Sentiment classifier\"]; route [label=\"Uncertainty + sarcasm router\"]; review [label=\"Optional local Qwen review\"]; result [label=\"Interpretable result\"];
            input -> normalize -> classify -> route -> review -> result; }""",
            width="stretch",
        )
    with right:
        st.subheader("Current local workflow")
        st.markdown("Use **Analyze text** for a single message, then open **Dataset explorer** to review labels and data quality. The baseline is always available; local MuRIL and Qwen can be switched on when installed.")
        st.info("The bundled dataset is illustrative. Report research results only after running experiments on the full labelled dataset.")

    st.subheader("Five proposed improvements")
    improvements = [
        ("1", "Hinglish-aware encoder", "Compare MuRIL, mBERT, and XLM-R with the original English-oriented baselines."),
        ("2", "Spelling and emphasis", "Normalise variants such as acchaaa while retaining expressive elongation."),
        ("3", "Emoji + sarcasm", "Use emoji context and route literal-positive, context-negative messages for review."),
        ("4", "Local Qwen reviewer", "Ask Qwen3 locally only for uncertain or conflicting cases."),
        ("5", "Stronger evaluation", "Use macro-F1, calibration, class recall, and targeted ablation studies."),
    ]
    for row in (improvements[:3], improvements[3:]):
        columns = st.columns(len(row))
        for column, (number, title, detail) in zip(columns, row):
            with column:
                st.markdown(f"### {number}. {title}")
                st.write(detail)


def render_analysis(model_path: str, use_reviewer: bool, ollama_model: str) -> None:
    st.title("Analyze text")
    st.markdown('<p class="research-note">Inspect the final prediction and the signals that influenced it. All analysis remains on this computer.</p>', unsafe_allow_html=True)

    selected_example = st.selectbox("Load an example", ["Custom text", *EXAMPLES], label_visibility="collapsed")
    default_text = EXAMPLES.get(selected_example, "")
    text = st.text_area("Hinglish message", value=default_text, placeholder="Type a Hinglish message, for example: movie acchaaa thi 😊", height=120)
    analyze = st.button("Analyze message", type="primary", width="content")

    if not analyze:
        st.caption("Try an example or enter your own message to see the analysis.")
        return
    if not text.strip():
        st.warning("Enter a message before running analysis.")
        return

    try:
        pipeline, mode = build_pipeline(model_path, use_reviewer, ollama_model)
        prediction = pipeline.predict(text)
    except RuntimeError as exc:
        st.error(str(exc))
        return

    intensity = IntensityEstimator().estimate(text)
    if "history" not in st.session_state:
        st.session_state.history = []
    st.session_state.history.append(
        {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "text": text,
            "sentiment": prediction.final_label,
            "confidence": round(prediction.primary.confidence, 3),
            "sarcasm_probability": round(prediction.primary.sarcasm_probability, 3),
            "routed_for_review": prediction.review_decision.should_review,
        }
    )

    st.caption(f"Active mode: {mode}")
    result_kind = prediction.final_label
    st.markdown(status_chip(f"Final sentiment: {prediction.final_label.title()}", result_kind), unsafe_allow_html=True)
    if prediction.review_decision.should_review:
        st.markdown(status_chip("Review route triggered", "warning"), unsafe_allow_html=True)
    st.write("")

    metric_columns = st.columns(4)
    metric_columns[0].metric("Confidence", f"{prediction.primary.confidence:.0%}")
    metric_columns[1].metric("Sarcasm likelihood", f"{prediction.primary.sarcasm_probability:.0%}")
    metric_columns[2].metric("Valence", f"{intensity.valence:+.2f}")
    metric_columns[3].metric("Arousal", f"{intensity.arousal:.2f}")

    chart_col, details_col = st.columns([1.05, 1])
    with chart_col:
        probability_frame = pd.DataFrame(
            {"Sentiment": list(prediction.primary.probabilities), "Probability": list(prediction.primary.probabilities.values())}
        )
        chart = px.bar(
            probability_frame,
            x="Sentiment",
            y="Probability",
            color="Sentiment",
            color_discrete_map=LABEL_COLORS,
            range_y=[0, 1],
            text_auto=".0%",
            title="Primary classifier probabilities",
        )
        chart.update_layout(showlegend=False, margin=dict(l=0, r=0, t=45, b=0), yaxis_tickformat=".0%")
        st.plotly_chart(chart, width="stretch")
    with details_col:
        st.subheader("Decision signals")
        reasons = prediction.review_decision.reasons or ("No review route triggered",)
        st.write("**Router:** " + ", ".join(reason.replace("_", " ") for reason in reasons))
        st.write("**Emoji relation:** " + ("possible conflict" if prediction.text_emoji_conflict else "no conflict detected"))
        st.write("**Intensity cues:** " + (", ".join(intensity.cues) if intensity.cues else "none detected"))
        reviewer = prediction.reviewer
        if reviewer is not None:
            st.success(f"Local Qwen review: {reviewer.sentiment.title()} | sarcasm: {'yes' if reviewer.sarcasm else 'no'}")
        elif use_reviewer and prediction.review_decision.should_review:
            st.warning("Qwen was selected but did not return a review. Confirm Ollama is running with the selected model.")

    with st.expander("Normalization and evidence", expanded=True):
        token_rows = [
            {
                "Original": token.raw,
                "Normalized": token.normalized,
                "Language": token.language,
                "Emphasis retained": "Yes" if token.elongated else "",
                "Negation": "Yes" if token.is_negation else "",
            }
            for token in prediction.normalized.tokens
        ]
        st.write(f"**Normalized message:** {prediction.normalized.normalized_text}")
        st.write("**Evidence:** " + (", ".join(prediction.primary.evidence) if prediction.primary.evidence else "No lexical evidence was identified by the baseline."))
        st.dataframe(pd.DataFrame(token_rows), width="stretch", hide_index=True)

    history = pd.DataFrame(st.session_state.history)
    with st.expander(f"Session history ({len(history)})"):
        st.dataframe(history.iloc[::-1], width="stretch", hide_index=True)
        st.download_button(
            "Download session history as CSV",
            data=history.to_csv(index=False).encode("utf-8"),
            file_name="hinglish_emotion_session_history.csv",
            mime="text/csv",
        )


def load_dataset(uploaded_file: object | None) -> tuple[pd.DataFrame, str]:
    if uploaded_file is not None:
        return read_dataset(uploaded_file), f"Uploaded dataset: {uploaded_file.name}"
    return read_dataset(SAMPLE_DATA), "Bundled 75-phrase workbook"


def render_dataset() -> None:
    st.title("Dataset explorer")
    st.markdown('<p class="research-note">Review dataset balance and annotations before training a model.</p>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Upload a compatible CSV or XLSX",
        type=["csv", "xlsx"],
        help="Required columns: text and label. XLSX files must contain a 'phrases' sheet.",
    )
    try:
        frame, source = load_dataset(uploaded)
        report = dataframe_report(frame)
    except Exception as exc:
        st.error(f"Could not read this dataset: {exc}")
        return

    st.caption(f"Source: {source}")
    if not report.valid:
        st.error("This dataset needs attention before training.")
        st.json(report_dict(report))
        return
    st.success("Dataset passed the required text and label checks.")

    if uploaded is None:
        st.download_button(
            "Download the 75-phrase XLSX",
            data=SAMPLE_DATA.read_bytes(),
            file_name=SAMPLE_DATA.name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    sarcasm = sarcasm_rate(frame)
    metrics = st.columns(4)
    metrics[0].metric("Rows", report.rows)
    metrics[1].metric("Unique messages", report.rows - report.duplicate_text_rows)
    metrics[2].metric("Sarcasm rate", "Not labelled" if sarcasm is None else f"{sarcasm:.0%}")
    metrics[3].metric("Optional fields", len(report.optional_columns))

    first_col, second_col = st.columns(2)
    with first_col:
        balance = pd.DataFrame({"Sentiment": list(report.label_counts), "Messages": list(report.label_counts.values())})
        balance_chart = px.bar(
            balance,
            x="Sentiment",
            y="Messages",
            color="Sentiment",
            color_discrete_map=LABEL_COLORS,
            title="Class balance",
            text="Messages",
        )
        balance_chart.update_layout(showlegend=False, margin=dict(l=0, r=0, t=45, b=0))
        st.plotly_chart(balance_chart, width="stretch")
    with second_col:
        length_frame = frame.copy()
        length_frame["Message length"] = length_frame["text"].fillna("").astype(str).str.len()
        length_chart = px.histogram(length_frame, x="Message length", nbins=min(20, max(5, len(length_frame))), title="Message-length distribution")
        length_chart.update_traces(marker_color="#315F97")
        length_chart.update_layout(margin=dict(l=0, r=0, t=45, b=0), yaxis_title="Messages")
        st.plotly_chart(length_chart, width="stretch")

    st.subheader("Source data")
    query = st.text_input("Search messages", placeholder="Search text, label, or annotation")
    visible = frame.copy()
    if query.strip():
        searchable = visible.fillna("").astype(str).apply(lambda row: row.str.contains(query, case=False, regex=False).any(), axis=1)
        visible = visible[searchable]
    st.dataframe(visible, width="stretch", hide_index=True)


def evaluation_models(
    model_path: str, use_reviewer: bool, ollama_model: str
) -> tuple[list[tuple[str, EmotionPipeline]], list[str]]:
    """Build only model variants that are genuinely available on this machine."""
    models: list[tuple[str, EmotionPipeline]] = [
        ("Rule-based baseline", EmotionPipeline(classifier=RuleBasedClassifier()))
    ]
    notices: list[str] = []
    transformer = None
    transformer_name = ""
    if model_path.strip():
        path = Path(model_path.strip())
        if not path.is_dir():
            notices.append("The configured transformer folder was not found, so it was skipped.")
        else:
            transformer = LocalTransformerClassifier(path)
            transformer_name = f"Local transformer · {path.name}"
            models.append((transformer_name, EmotionPipeline(classifier=transformer)))

    if use_reviewer:
        reviewer = OllamaReviewer(model=ollama_model)
        ready, message = reviewer.availability()
        if ready:
            primary = transformer or RuleBasedClassifier()
            primary_name = transformer_name or "Rule baseline"
            models.append(
                (
                    f"{primary_name} + Qwen review",
                    EmotionPipeline(classifier=primary, reviewer=reviewer),
                )
            )
        else:
            notices.append(f"Qwen evaluation was skipped: {message}")
    return models, notices


def render_model_panel(evaluation: ModelEvaluation) -> None:
    summary = evaluation.summary
    metrics = st.columns(5)
    metrics[0].metric("Accuracy", f"{float(summary['accuracy']):.1%}")
    metrics[1].metric("Macro-F1", f"{float(summary['macro_f1']):.1%}")
    metrics[2].metric("Calibration error", f"{float(summary['calibration_error']):.3f}")
    metrics[3].metric("Avg confidence", f"{float(summary['average_confidence']):.1%}")
    metrics[4].metric("Review routed", f"{float(summary['review_route_rate']):.1%}")

    left, right = st.columns(2)
    with left:
        class_frame = evaluation.per_class.melt(
            id_vars=["sentiment", "support"],
            value_vars=["precision", "recall", "f1"],
            var_name="Metric",
            value_name="Score",
        )
        class_chart = px.bar(
            class_frame,
            x="sentiment",
            y="Score",
            color="Metric",
            barmode="group",
            range_y=[0, 1],
            title=f"{evaluation.model}: per-class effectiveness",
            labels={"sentiment": "Actual sentiment"},
        )
        class_chart.update_layout(margin=dict(l=0, r=0, t=55, b=0), yaxis_tickformat=".0%")
        st.plotly_chart(class_chart, width="stretch")
    with right:
        confusion_chart = px.imshow(
            evaluation.confusion,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="Blues",
            title=f"{evaluation.model}: confusion matrix",
            labels={"x": "Predicted", "y": "Actual", "color": "Rows"},
        )
        confusion_chart.update_layout(margin=dict(l=0, r=0, t=55, b=0))
        st.plotly_chart(confusion_chart, width="stretch")

    calibration = go.Figure()
    calibration.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration",
            line={"dash": "dash", "color": "#6B7280"},
        )
    )
    if not evaluation.calibration.empty:
        calibration.add_trace(
            go.Scatter(
                x=evaluation.calibration["mean_confidence"],
                y=evaluation.calibration["empirical_accuracy"],
                mode="lines+markers+text",
                text=evaluation.calibration["count"].map(lambda value: f"n={value}"),
                textposition="top center",
                name=evaluation.model,
                marker={"size": 9},
            )
        )
    calibration.update_layout(
        title=f"{evaluation.model}: reliability by confidence bin",
        xaxis={"title": "Mean confidence", "range": [0, 1], "tickformat": ".0%"},
        yaxis={"title": "Observed accuracy", "range": [0, 1], "tickformat": ".0%"},
        margin=dict(l=0, r=0, t=55, b=0),
    )
    st.plotly_chart(calibration, width="stretch")

    errors = evaluation.predictions[~evaluation.predictions["correct"]]
    with st.expander(f"Misclassified phrases ({len(errors)})"):
        st.dataframe(errors, width="stretch", hide_index=True)


def render_evaluation(model_path: str, use_reviewer: bool, ollama_model: str) -> None:
    st.title("Model effectiveness")
    st.markdown(
        '<p class="research-note">Run every available model variant on the same labelled phrases. '
        "All scores shown here are measured live; unavailable checkpoints are never assigned placeholder scores.</p>",
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader(
        "Evaluation dataset (CSV or XLSX)",
        type=["csv", "xlsx"],
        key="evaluation_dataset",
        help="Leave empty to use the balanced 75-phrase workbook.",
    )
    try:
        frame, source = load_dataset(uploaded)
        report = dataframe_report(frame)
    except Exception as exc:
        st.error(f"Could not read this evaluation dataset: {exc}")
        return
    if not report.valid:
        st.error("The evaluation dataset has empty text or invalid labels.")
        st.json(report_dict(report))
        return

    st.caption(f"Source: {source} · {report.rows} labelled rows")
    st.info(
        "The bundled workbook is an illustrative smoke-test set. Its results verify this implementation, "
        "but they are not a substitute for a held-out research benchmark."
    )
    evaluation_signature = (source, model_path.strip(), use_reviewer, ollama_model)
    if st.button("Run model evaluation", type="primary"):
        try:
            models, notices = evaluation_models(model_path, use_reviewer, ollama_model)
            for notice in notices:
                st.warning(notice)
            progress = st.progress(0, text="Preparing evaluation…")
            evaluations = []
            for index, (name, pipeline) in enumerate(models, start=1):
                progress.progress((index - 1) / len(models), text=f"Evaluating {name}…")
                evaluations.append(evaluate_pipeline(frame, pipeline, name))
            progress.progress(1.0, text="Evaluation complete")
            st.session_state.model_evaluations = evaluations
            st.session_state.model_evaluation_signature = evaluation_signature
        except Exception as exc:
            st.error(f"Evaluation could not be completed: {exc}")
            return

    evaluations = st.session_state.get("model_evaluations", [])
    if not evaluations:
        st.caption("Run the evaluation to generate comparison and per-model graphs.")
        return
    if st.session_state.get("model_evaluation_signature") != evaluation_signature:
        st.warning("The graphs below use previous data or model settings. Run evaluation again to refresh them.")

    summary = pd.DataFrame([item.summary for item in evaluations])
    comparison = summary.melt(
        id_vars=["model"],
        value_vars=["accuracy", "macro_f1"],
        var_name="Metric",
        value_name="Score",
    )
    comparison["Metric"] = comparison["Metric"].map(
        {"accuracy": "Accuracy", "macro_f1": "Macro-F1"}
    )
    comparison_chart = px.bar(
        comparison,
        x="model",
        y="Score",
        color="Metric",
        barmode="group",
        text_auto=".1%",
        range_y=[0, 1],
        title="Available model comparison on the selected dataset",
        labels={"model": "Model"},
    )
    comparison_chart.update_layout(margin=dict(l=0, r=0, t=55, b=0), yaxis_tickformat=".0%")
    st.plotly_chart(comparison_chart, width="stretch")

    tabs = st.tabs([item.model for item in evaluations])
    for tab, evaluation in zip(tabs, evaluations):
        with tab:
            render_model_panel(evaluation)


def main() -> None:
    st.set_page_config(page_title="Hinglish Emotion Research", page_icon="💬", layout="wide")
    inject_theme()
    if "history" not in st.session_state:
        st.session_state.history = []

    with st.sidebar:
        st.header("Local model settings")
        model_path = st.text_input("Local transformer folder", placeholder="models/muril_sentiment")
        use_reviewer = st.toggle("Use local Qwen reviewer", value=False)
        ollama_model = st.text_input("Ollama model", value=DEFAULT_OLLAMA_MODEL, disabled=not use_reviewer)
        if use_reviewer:
            ready, message = OllamaReviewer(model=ollama_model).availability()
            (st.success if ready else st.warning)(message)
        st.caption("Leave the model folder blank to use the transparent rule baseline.")

    overview, analysis, evaluation, dataset = st.tabs(
        ["Overview", "Analyze text", "Model effectiveness", "Dataset explorer"]
    )
    with overview:
        render_overview()
    with analysis:
        render_analysis(model_path, use_reviewer, ollama_model)
    with evaluation:
        render_evaluation(model_path, use_reviewer, ollama_model)
    with dataset:
        render_dataset()


if __name__ == "__main__":
    main()
