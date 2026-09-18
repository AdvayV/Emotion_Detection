from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.data import dataframe_report, report_dict, sarcasm_rate
from hinglish_emotion.intensity import IntensityEstimator
from hinglish_emotion.ollama_reviewer import OllamaReviewer
from hinglish_emotion.pipeline import EmotionPipeline
from hinglish_emotion.transformer_classifier import LocalTransformerClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DATA = PROJECT_ROOT / "data" / "sample_messages.csv"
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


def load_dataset(uploaded_file: object | None) -> tuple[pd.DataFrame | None, str]:
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file), "Uploaded dataset"
    return pd.read_csv(SAMPLE_DATA), "Bundled sample dataset"


def render_dataset() -> None:
    st.title("Dataset explorer")
    st.markdown('<p class="research-note">Review dataset balance and annotations before training a model.</p>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload a compatible CSV", type=["csv"], help="Required columns: text and label")
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


def main() -> None:
    st.set_page_config(page_title="Hinglish Emotion Research", page_icon="💬", layout="wide")
    inject_theme()
    if "history" not in st.session_state:
        st.session_state.history = []

    with st.sidebar:
        st.header("Local model settings")
        model_path = st.text_input("Local transformer folder", placeholder="models/muril_sentiment")
        use_reviewer = st.toggle("Use local Qwen reviewer", value=False)
        ollama_model = st.text_input("Ollama model", value="qwen3:4b", disabled=not use_reviewer)
        st.caption("Leave the model folder blank to use the transparent rule baseline.")

    overview, analysis, dataset = st.tabs(["Overview", "Analyze text", "Dataset explorer"])
    with overview:
        render_overview()
    with analysis:
        render_analysis(model_path, use_reviewer, ollama_model)
    with dataset:
        render_dataset()


if __name__ == "__main__":
    main()
