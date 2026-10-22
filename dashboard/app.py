from __future__ import annotations

from datetime import datetime
import hashlib
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.data import dataframe_report, read_dataset, report_dict, sarcasm_rate
from dashboard.evaluation import ModelEvaluation, evaluate_pipeline
from dashboard.theme import apply_theme
from hinglish_emotion.intensity import IntensityEstimator
from hinglish_emotion.model_catalog import ENCODERS, checkpoint_path, checkpoint_ready
from hinglish_emotion.ollama_reviewer import DEFAULT_OLLAMA_MODEL, OllamaReviewer
from hinglish_emotion.pipeline import EmotionPipeline, RuleBasedClassifier
from hinglish_emotion.transformer_classifier import LocalTransformerClassifier
from hinglish_emotion.checkpoints import emotion_checkpoint
from hinglish_emotion.tasks import LANGUAGE_PAIRS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DATA = PROJECT_ROOT / "data" / "demo" / "sentiment" / "hinglish_emotion_phrases.xlsx"
PAGES = ["Emotion lab", "Model disagreement", "Analyze text", "Model effectiveness", "Dataset explorer", "Overview"]
EXAMPLES = {
    "Positive with emphasis": "movie acchaaa thi 😊",
    "Negation": "service acchi nahi thi 😞",
    "Possible sarcasm": "wah kya service hai, 2 ghante late 😂",
    "Neutral": "theek tha",
}
LABEL_COLORS = {"positive": "#087F83", "neutral": "#6B7888", "negative": "#B85C69"}
SENTIMENT_ICONS = {"positive": "🙂", "neutral": "😐", "negative": "🙁"}
SENTIMENT_EXPLANATIONS = {
    "positive": "The message expresses a positive opinion or feeling.",
    "neutral": "The message does not lean clearly positive or negative.",
    "negative": "The message expresses a complaint or negative feeling.",
}
REVIEW_REASONS = {
    "low_confidence": "The classifier is not confident enough to decide on its own.",
    "possible_sarcasm": "Praise and a negative situation may be used sarcastically.",
    "text_emoji_conflict": "The words and emoji may point to different meanings.",
    "classifier_disagreement": "The classifiers disagree about the sentiment.",
    "uncertain_negation": "A negation such as 'nahi' makes this message harder to judge.",
}
MODEL_DESCRIPTIONS = {
    "Rule-based baseline": "Uses sentiment words, negation and emoji rules. Ready to try immediately.",
    "mBERT": "A multilingual BERT encoder. Needs an installed three-class sentiment checkpoint.",
    "MuRIL": "An encoder for Indian languages and transliterated text. Needs an installed sentiment checkpoint.",
    "XLM-R": "A multilingual RoBERTa encoder. Needs an installed three-class sentiment checkpoint.",
    "Custom checkpoint": "Use another locally saved three-class sentiment model.",
}


def page_header(title: str, description: str) -> None:
    st.caption("HINGLISH LAB  /  LOCAL RESEARCH WORKSPACE")
    st.title(title)
    st.write(description)


def style_chart(figure, height: int = 320):
    figure.update_layout(
        template="plotly_white", height=height,
        font={"family": "Arial, sans-serif", "color": "#34475A", "size": 13},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 5, "r": 10, "t": 30, "b": 5}, legend_title_text="",
    )
    figure.update_xaxes(gridcolor="#E2E9EC")
    figure.update_yaxes(gridcolor="#E2E9EC")
    return figure


def classifier_name(model_path: str) -> str:
    if not model_path:
        return "Rule-based baseline"
    for encoder in ENCODERS:
        if Path(model_path).name == encoder.folder:
            return encoder.name
    return "Custom checkpoint"


def navigate_to(page: str) -> None:
    st.session_state.workspace = page


def select_baseline() -> None:
    st.session_state.encoder_choice = "Rule-based baseline"


def load_example() -> None:
    st.session_state.message_input = EXAMPLES.get(st.session_state.analysis_example, "")


@st.cache_resource(max_entries=1, show_spinner=False)
def local_classifier(path: str, checkpoint_version: int) -> LocalTransformerClassifier:
    return LocalTransformerClassifier(path)


def build_pipeline(model_path: str, use_reviewer: bool, ollama_model: str) -> tuple[EmotionPipeline, str]:
    classifier = None
    mode = classifier_name(model_path)
    if model_path.strip():
        path = Path(model_path.strip())
        if not checkpoint_ready(path):
            raise RuntimeError("This model needs a trained checkpoint. Choose the rule-based baseline to try analysis now.")
        classifier = local_classifier(str(path), (path / "config.json").stat().st_mtime_ns)
    reviewer = OllamaReviewer(model=ollama_model) if use_reviewer else None
    return EmotionPipeline(classifier=classifier, reviewer=reviewer), mode


def render_sidebar() -> tuple[str, str, bool, str]:
    with st.sidebar:
        st.title("💬 Hinglish Lab")
        st.caption("A local workspace for model exploration")
        page = st.radio("Workspace", PAGES, key="workspace")
        if page in {"Emotion lab", "Model disagreement"}:
            st.divider()
            st.info("Seven emotions for Hindi–English and English–French. The selected language determines which trained checkpoint is available.")
            st.subheader("Seven-emotion settings")
            st.caption("Choose the language mix and emotion checkpoint in the workspace. Sentiment checkpoints use a different label set; Qwen review supports both tasks.")
            return page, "", False, DEFAULT_OLLAMA_MODEL
        if page not in {"Analyze text", "Model effectiveness"}:
            st.divider()
            st.subheader("Explore the project")
            st.caption("Choose a workspace above to inspect data, predictions or evaluation results.")
            return page, "", False, DEFAULT_OLLAMA_MODEL
        st.divider()
        st.subheader("Sentiment settings")
        encoder_choice = st.selectbox(
            "Primary classifier",
            ["Rule-based baseline", *(encoder.name for encoder in ENCODERS), "Custom checkpoint"],
            key="encoder_choice",
            help="This model predicts positive, neutral or negative sentiment.",
        )
        st.caption(MODEL_DESCRIPTIONS[encoder_choice])
        model_path = ""
        if encoder_choice == "Custom checkpoint":
            custom = st.text_input("Local transformer folder", placeholder="models/my_checkpoint")
            if custom.strip():
                path = Path(custom.strip())
                model_path = str(path if path.is_absolute() else PROJECT_ROOT / path)
        elif encoder_choice != "Rule-based baseline":
            encoder = next(item for item in ENCODERS if item.name == encoder_choice)
            model_path = str(checkpoint_path(encoder, PROJECT_ROOT / "models"))
        if encoder_choice == "Custom checkpoint" and not model_path:
            st.info("Enter your checkpoint folder to use a custom model.")
        elif not model_path or checkpoint_ready(model_path):
            st.success("Ready to analyze", icon="✅")
        else:
            st.warning(f"{encoder_choice} is not trained here yet.", icon="⚠️")
            st.button("Use rule-based baseline", on_click=select_baseline, width="stretch")

        st.divider()
        use_reviewer = st.toggle(
            "Ask Qwen for a second opinion", value=False,
            help="Only uncertain or conflicting messages are sent to your local Ollama model.",
        )
        st.caption("Optional · runs locally through Ollama. CPU review may take a little longer.")
        with st.expander("Advanced model settings"):
            ollama_model = st.text_input("Ollama model", value=DEFAULT_OLLAMA_MODEL)
            if model_path:
                st.caption("Selected checkpoint")
                st.code(model_path, language=None)
        if use_reviewer:
            ready, message = OllamaReviewer(model=ollama_model).availability()
            (st.success if ready else st.warning)(message)
        st.divider()
        st.caption("Private by default · live analysis stays in this browser session.")
    return page, model_path, use_reviewer, ollama_model


def render_result(saved: dict) -> None:
    prediction = saved["prediction"]
    intensity = saved["intensity"]
    st.caption("02  /  YOUR RESULT")
    st.caption(f"Analyzed with {saved['mode']}")
    label = prediction.final_label
    st.subheader(f"{SENTIMENT_ICONS[label]} {label.title()}")
    st.write(SENTIMENT_EXPLANATIONS[label])
    st.caption("Analyzed message")
    st.write(saved["text"])
    score_col, emoji_col = st.columns(2)
    score_col.metric(
        "Primary confidence", f"{prediction.primary.confidence:.0%}",
        help="The primary classifier's score for its own prediction. Qwen may change the final sentiment without changing this score.",
    )
    emoji_col.metric(
        "Emoji check",
        "Needs context" if prediction.text_emoji_conflict else "No conflict" if prediction.normalized.emojis else "No emoji",
        help="Checks whether the emoji and predicted sentiment may conflict.",
    )
    if not saved["model_path"]:
        st.caption("Confidence is a rule-based estimate, not a measured probability of being correct.")

    st.divider()
    st.markdown("**How the classifier scored the message**")
    probability_frame = pd.DataFrame({
        "Sentiment": list(prediction.primary.probabilities),
        "Score": list(prediction.primary.probabilities.values()),
    })
    chart = px.bar(
        probability_frame, x="Score", y="Sentiment", orientation="h",
        color="Sentiment", color_discrete_map=LABEL_COLORS, range_x=[0, 1],
        text=probability_frame["Score"].map(lambda score: f"{score:.0%}"),
    )
    chart.update_traces(textposition="outside", cliponaxis=False)
    chart.update_layout(showlegend=False)
    chart.update_xaxes(title=None, tickformat=".0%", range=[0, 1.12])
    chart.update_yaxes(title=None, categoryorder="array", categoryarray=["negative", "neutral", "positive"])
    st.plotly_chart(style_chart(chart, 215), width="stretch", config={"displayModeBar": False})

    st.markdown("**Does it need a second look?**")
    if prediction.reviewer is not None:
        reviewer = prediction.reviewer
        st.success("Qwen reviewed this message locally.")
        if reviewer.sentiment != prediction.primary.label:
            st.write(f"Qwen changed **{prediction.primary.label} → {reviewer.sentiment}**.")
        else:
            st.write("Qwen agreed with the primary sentiment.")
        st.write(f"**Sarcasm according to Qwen:** {'Yes' if reviewer.sarcasm else 'No'}")
    elif prediction.review_decision.should_review:
        st.warning("This message has signals that deserve a second look.")
        for reason in prediction.review_decision.reasons:
            st.write("• " + REVIEW_REASONS.get(reason, reason.replace("_", " ")))
        if saved["use_reviewer"]:
            st.caption("Qwen was requested but no review was returned. Check the Ollama status in the sidebar.")
        else:
            st.caption("Switch on the Qwen second opinion in the sidebar to review difficult messages.")
    else:
        st.success("No additional review was requested.")
        st.caption("The primary classifier did not trigger the uncertainty checks.")

    with st.expander("Evidence, spelling changes and research details"):
        evidence = prediction.reviewer.evidence if prediction.reviewer is not None else prediction.primary.evidence
        st.markdown("**Words and emoji picked up by the model**")
        if evidence:
            st.write(", ".join(evidence))
        else:
            st.caption("This classifier does not provide word-level evidence.")
        st.markdown("**Text after normalization**")
        st.code(prediction.normalized.normalized_text, language=None)
        changes = [token for token in prediction.normalized.tokens if token.raw != token.normalized]
        st.caption(f"{len(changes)} token(s) changed. Emphasis and negation are kept as separate features.")
        token_rows = [{
            "Original": token.raw, "Normalized": token.normalized,
            "Language": token.language,
            "Emphasis retained": "Yes" if token.elongated else "",
            "Negation": "Yes" if token.is_negation else "",
        } for token in prediction.normalized.tokens]
        st.dataframe(pd.DataFrame(token_rows), width="stretch", hide_index=True)
        st.divider()
        st.markdown("**Sarcasm signal**")
        if saved["model_path"]:
            st.caption("The transformer checkpoint predicts sentiment only; it has no trained sarcasm head.")
        else:
            st.write("Possible sarcasm flagged" if prediction.primary.sarcasm_probability >= 0.60 else "No strong sarcasm signal")
            st.caption(f"Rule-based signal: {prediction.primary.sarcasm_probability:.0%}. This uses praise, a negative situation and laughter cues.")
        st.markdown("**Emotion intensity · heuristic estimate**")
        valence_col, arousal_col = st.columns(2)
        valence_col.metric("Valence", f"{intensity.valence:+.2f}", help="Negative to positive tone, from -1 to +1.")
        arousal_col.metric("Arousal", f"{intensity.arousal:.2f}", help="Estimated emotional energy, from 0 to 1.")
        st.caption("These lexical estimates are independent of the sentiment classifier and may disagree with it.")
        if intensity.cues:
            st.write("Intensity cues: " + ", ".join(cue.replace("_", " ") for cue in intensity.cues))


def render_history() -> None:
    history = pd.DataFrame(st.session_state.get("history", []))
    if history.empty:
        return
    with st.expander(f"Recent analyses · {len(history)} in this session"):
        st.caption("Kept in memory for this browser session. Download a copy before closing it.")
        st.dataframe(history.iloc[::-1], width="stretch", hide_index=True)
        st.download_button(
            "Download analysis history", data=history.to_csv(index=False).encode("utf-8"),
            file_name="hinglish_emotion_session_history.csv", mime="text/csv",
        )


def render_analysis(model_path: str, use_reviewer: bool, ollama_model: str) -> None:
    page_header("Analyze text", "Understand the sentiment of a Hinglish message and see what influenced the prediction.")
    st.caption("This page predicts three-class sentiment. The trained seven-emotion XLM-R checkpoint is used in Emotion lab.")
    st.button("Open trained emotion workspace", on_click=navigate_to, args=("Emotion lab",), key="analysis_to_emotions")
    input_col, result_col = st.columns([1, 1.15], gap="large")
    selected = st.session_state.get("encoder_choice", "Rule-based baseline")
    signature = (selected, model_path, use_reviewer, ollama_model)
    ready = (not model_path and selected == "Rule-based baseline") or bool(model_path and checkpoint_ready(model_path))
    with input_col:
        with st.container(border=True):
            st.caption("01  /  YOUR MESSAGE")
            st.subheader("What would you like to analyze?")
            st.selectbox("Try an example", ["Custom text", *EXAMPLES], key="analysis_example", on_change=load_example)
            if "message_input" not in st.session_state:
                st.session_state.message_input = ""
            text = st.text_area(
                "Hinglish message", key="message_input", height=170,
                placeholder="movie acchaaa thi 😊",
                help="Keep the original spelling and emoji. The pipeline handles normalization.",
            )
            st.caption(f"Classifier: {selected} · Qwen second opinion: {'on' if use_reviewer else 'off'}")
            analyze = st.button("Analyze message", type="primary", width="stretch", disabled=not ready)
            if not ready:
                st.info("Choose the ready rule-based baseline in the sidebar, or train this encoder first.")
        with st.container(border=True):
            st.markdown("**Three things to try**")
            st.write("**Spelling:** accha and acchaaa")
            st.write("**Negation:** acchi thi and acchi nahi thi")
            st.write("**Sarcasm:** wah kya service hai, 2 ghante late 😂")
            st.caption("The default baseline uses rules. A trained encoder can later be compared on the same examples.")

        if analyze:
            if not text.strip():
                st.warning("Write a message or choose an example before analyzing.")
            else:
                try:
                    with st.spinner("Reading your message…"):
                        pipeline, mode = build_pipeline(model_path, use_reviewer, ollama_model)
                        prediction = pipeline.predict(text)
                    st.session_state.last_analysis = {
                        "prediction": prediction, "intensity": IntensityEstimator().estimate(text),
                        "text": text, "mode": mode, "model_path": model_path,
                        "use_reviewer": use_reviewer, "signature": signature,
                    }
                    st.session_state.history.append({
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "text": text, "classifier": mode, "sentiment": prediction.final_label,
                        "confidence": round(prediction.primary.confidence, 3),
                        "sarcasm_probability": round(prediction.primary.sarcasm_probability, 3) if not model_path else None,
                        "routed_for_review": prediction.review_decision.should_review,
                    })
                except (RuntimeError, ValueError, OSError) as exc:
                    st.error(f"Analysis could not finish: {exc}")

    with result_col:
        with st.container(border=True):
            saved = st.session_state.get("last_analysis")
            if saved:
                if saved["text"] != text or saved["signature"] != signature:
                    st.info("Showing your previous result. Analyze again to use the current message and settings.")
                render_result(saved)
            else:
                st.caption("02  /  YOUR RESULT")
                st.subheader("Your result will appear here")
                st.write("Choose an example or write a message, then select **Analyze message**.")
                st.divider()
                st.markdown("**You will see**")
                st.write("A sentiment label: positive, neutral or negative.")
                st.write("The classifier's scores and any emoji conflict.")
                st.write("Why a second review may be useful.")
                st.caption("No prediction has been run yet.")
    render_history()


def render_overview() -> None:
    page_header("How the project works", "A local workflow for Hinglish sentiment, spelling variation, emoji and difficult messages.")
    ready_encoders = sum(checkpoint_ready(checkpoint_path(encoder, PROJECT_ROOT / "models")) for encoder in ENCODERS)
    columns = st.columns(3)
    columns[0].metric("Default classifier", "Ready", help="The rule-based baseline runs without downloading model weights.", border=True)
    columns[1].metric("Trained sentiment encoders", f"{ready_encoders} / 3", help="Three-class checkpoints only; the seven-emotion XLM-R checkpoint is counted separately below.", border=True)
    columns[2].metric("Review option", "Local Qwen", help="An optional second opinion for uncertain messages through Ollama.", border=True)
    emotion_models = [
        f"{encoder.name} · {LANGUAGE_PAIRS[pair]}"
        for pair in LANGUAGE_PAIRS
        for encoder in ENCODERS
        if emotion_checkpoint(PROJECT_ROOT / "models", pair, encoder.key) is not None
    ]
    st.caption("Installed seven-emotion checkpoints: " + (", ".join(emotion_models) or "none found"))
    st.button("Open seven-emotion prediction and file evaluation", on_click=navigate_to, args=("Emotion lab",), key="overview_to_emotions")
    st.write("")
    first, second = st.columns(2)
    with first, st.container(border=True):
        st.subheader("Try one message")
        st.write("Start with an example. Inspect sentiment, spelling changes and review signals.")
        st.button("Open text analysis", on_click=navigate_to, args=("Analyze text",), width="stretch", type="primary")
    with second, st.container(border=True):
        st.subheader("Check the model's performance")
        st.write("Compare predictions with labelled phrases and inspect the mistakes.")
        st.button("Open model evaluation", on_click=navigate_to, args=("Model effectiveness",), width="stretch")
    with st.container(border=True):
        st.subheader("From message to result")
        st.graphviz_chart(
            r"""digraph { graph [bgcolor=transparent, rankdir=LR, nodesep=0.45, pad=0.2];
            node [shape=box, style="rounded,filled", fillcolor="#F1F2FC", color="#CBCDE9", fontname="Arial", fontcolor="#313651"];
            edge [color="#8A91B5", fontname="Arial", fontsize=10];
            input [label="Your message"]; normalize [label="Clean spelling\nkeep emphasis + negation"];
            classify [label="Predict sentiment"]; route [label="Check uncertainty\nemoji + sarcasm"];
            review [label="Qwen second opinion"]; result [label="Final result"];
            input -> normalize -> classify -> route;
            route -> review [label="Needs review + Qwen on"]; review -> result;
            route -> result [label="No review / Qwen off"]; }""",
            width="stretch",
        )
        st.caption("Qwen is consulted only when a review is triggered and the reviewer is enabled.")
    st.subheader("The five improvements")
    features = [
        ("Hinglish-aware encoders", "mBERT, MuRIL and XLM-R training and checkpoint selection", "Train on the real labelled dataset"),
        ("Spelling and emphasis", "Normalize variants while retaining elongation and negation", "Measure consistency across spelling variants"),
        ("Emoji and sarcasm", "Rule-based conflict and sarcasm signals", "Train contextual emoji and sarcasm components"),
        ("Local Qwen review", "Optional review of uncertain messages", "Measure when the second opinion helps"),
        ("Stronger evaluation", "Live metrics, error analysis and shared training splits", "Evaluate on an independent held-out benchmark"),
    ]
    st.dataframe(pd.DataFrame(features, columns=["Improvement", "Available in the code", "Next research step"]), width="stretch", hide_index=True)
    st.info("The bundled 75 phrases are demo data. Research accuracy requires an independently labelled, held-out dataset.")


def load_dataset(uploaded_file: object | None) -> tuple[pd.DataFrame, str]:
    if uploaded_file is not None:
        return read_dataset(uploaded_file), f"Uploaded dataset: {uploaded_file.name}"
    return read_dataset(SAMPLE_DATA), "Bundled 75-phrase workbook"


def show_invalid_dataset(report) -> None:
    st.error("Fix the dataset before continuing.")
    if report.empty_text_rows:
        st.write(f"• {report.empty_text_rows} row(s) have no message text.")
    if report.invalid_label_rows:
        st.write(f"• {report.invalid_label_rows} row(s) need a positive, neutral or negative label.")
    if not report.rows:
        st.write("• This file contains no data rows.")
    with st.expander("Detailed validation report"):
        st.json(report_dict(report))


def render_dataset() -> None:
    page_header("Dataset explorer", "Inspect the messages, labels and annotations before using them for training or evaluation.")
    with st.container(border=True):
        uploaded = st.file_uploader(
            "Choose a dataset", type=["csv", "xlsx"], key="explorer_dataset",
            help="CSV needs text and label columns. XLSX needs those columns in a phrases sheet.",
        )
        st.caption("Leave this empty to explore the demo workbook. Labels must be positive, neutral or negative.")
    try:
        frame, source = load_dataset(uploaded)
        report = dataframe_report(frame)
    except Exception as exc:
        st.error(f"Could not read the dataset: {exc}")
        return
    if not report.valid:
        show_invalid_dataset(report)
        return
    st.caption(f"Source: {source}")
    if uploaded is None:
        st.info("Demo dataset · 75 illustrative phrases. Use an independent dataset for research results.")
    if report.duplicate_text_rows:
        st.warning(f"{report.duplicate_text_rows} repeated message(s) found. Keep duplicates in the same training split.")
    sarcasm = sarcasm_rate(frame)
    columns = st.columns(4)
    columns[0].metric("Messages", report.rows, border=True)
    columns[1].metric("Positive", report.label_counts["positive"], border=True)
    columns[2].metric("Neutral", report.label_counts["neutral"], border=True)
    columns[3].metric("Negative", report.label_counts["negative"], border=True)
    left, right = st.columns(2)
    with left, st.container(border=True):
        st.markdown("**How are the labels distributed?**")
        balance = pd.DataFrame({"Sentiment": list(report.label_counts), "Messages": list(report.label_counts.values())})
        chart = px.bar(balance, x="Sentiment", y="Messages", color="Sentiment", color_discrete_map=LABEL_COLORS, text="Messages")
        chart.update_layout(showlegend=False)
        chart.update_xaxes(title=None)
        st.plotly_chart(style_chart(chart), width="stretch", config={"displayModeBar": False})
        st.caption("A balanced dataset represents all three sentiments. Real data may be uneven.")
    with right, st.container(border=True):
        st.markdown("**How long are the messages?**")
        lengths = frame["text"].fillna("").astype(str).str.len()
        chart = px.histogram(pd.DataFrame({"Characters per message": lengths}), x="Characters per message", nbins=min(20, max(5, len(frame))))
        chart.update_traces(marker_color="#5655B8")
        chart.update_yaxes(title="Messages")
        st.plotly_chart(style_chart(chart), width="stretch", config={"displayModeBar": False})
        st.caption("Longer messages may need a larger token limit during training.")
    with st.container(border=True):
        st.subheader("Browse the messages")
        search_col, filter_col = st.columns([2, 1])
        query = search_col.text_input("Search messages", placeholder="Search any word, label or annotation")
        labels = filter_col.multiselect("Sentiment filter", ["positive", "neutral", "negative"], default=["positive", "neutral", "negative"])
        visible = frame[frame["label"].astype(str).str.strip().str.lower().isin(labels)].copy()
        if query.strip():
            searchable = visible.fillna("").astype(str).apply(lambda row: row.str.contains(query, case=False, regex=False).any(), axis=1)
            visible = visible[searchable]
        st.caption(f"Showing {len(visible)} of {len(frame)} messages")
        st.dataframe(visible, width="stretch", hide_index=True)
        st.caption("Sarcasm-labelled share: " + ("not annotated" if sarcasm is None else f"{sarcasm:.0%}") + " · Additional fields: " + (", ".join(report.optional_columns) or "none"))
        if uploaded is None:
            st.download_button("Download demo workbook", data=SAMPLE_DATA.read_bytes(), file_name=SAMPLE_DATA.name, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


def evaluation_models(model_path: str, use_reviewer: bool, ollama_model: str) -> tuple[list[tuple[str, EmotionPipeline]], list[str]]:
    models: list[tuple[str, EmotionPipeline]] = [
        ("Rule-based baseline", EmotionPipeline(classifier=RuleBasedClassifier()))
    ]
    notices: list[str] = []
    transformer = None
    transformer_name = ""
    if model_path.strip():
        path = Path(model_path.strip())
        if not checkpoint_ready(path):
            notices.append("The selected encoder has no trained checkpoint. Only ready models will be evaluated.")
        else:
            transformer = local_classifier(str(path), (path / "config.json").stat().st_mtime_ns)
            transformer_name = classifier_name(model_path)
            models.append((transformer_name, EmotionPipeline(classifier=transformer)))
    if use_reviewer:
        reviewer = OllamaReviewer(model=ollama_model)
        ready, message = reviewer.availability()
        if ready:
            models.append((
                f"{transformer_name or 'Rule baseline'} + Qwen review",
                EmotionPipeline(classifier=transformer or RuleBasedClassifier(), reviewer=reviewer),
            ))
        else:
            notices.append(f"Qwen review is unavailable: {message}")
    return models, notices


def render_model_panel(evaluation: ModelEvaluation) -> None:
    summary = evaluation.summary
    columns = st.columns(3)
    columns[0].metric("Accuracy", f"{float(summary['accuracy']):.1%}", help="The share of messages with a correct final sentiment.", border=True)
    columns[1].metric("Macro-F1", f"{float(summary['macro_f1']):.1%}", help="Averages the F1 score of each sentiment equally. Higher is better, especially for uneven class sizes.", border=True)
    columns[2].metric("Messages evaluated", int(summary["rows"]), border=True)
    left, right = st.columns(2)
    with left, st.container(border=True):
        st.markdown("**Which sentiments does it handle well?**")
        class_frame = evaluation.per_class.melt(
            id_vars=["sentiment", "support"], value_vars=["precision", "recall", "f1"],
            var_name="Metric", value_name="Score",
        )
        class_frame["Metric"] = class_frame["Metric"].str.title()
        chart = px.bar(
            class_frame, x="sentiment", y="Score", color="Metric", barmode="group", range_y=[0, 1],
            color_discrete_sequence=["#5655B8", "#087F83", "#A7A3D2"],
            labels={"sentiment": "Sentiment"},
        )
        chart.update_yaxes(tickformat=".0%")
        st.plotly_chart(style_chart(chart), width="stretch", config={"displayModeBar": False})
        st.caption("Precision: how often a predicted label is right. Recall: how many real examples it finds. F1 balances both.")
    with right, st.container(border=True):
        st.markdown("**Where does it confuse the labels?**")
        chart = px.imshow(
            evaluation.confusion, text_auto=True, aspect="auto", color_continuous_scale="Purples",
            labels={"x": "Predicted sentiment", "y": "Actual sentiment", "color": "Messages"},
        )
        st.plotly_chart(style_chart(chart), width="stretch", config={"displayModeBar": False})
        st.caption("Rows are the true labels; columns are predictions. Numbers on the diagonal are correct.")
    with st.expander("Confidence reliability and review activity"):
        columns = st.columns(3)
        columns[0].metric("Calibration error", f"{float(summary['calibration_error']):.3f}", help="Distance between confidence and observed correctness. Lower is better.")
        columns[1].metric("Avg confidence", f"{float(summary['average_confidence']):.1%}")
        columns[2].metric("Review routed", f"{float(summary['review_route_rate']):.1%}", help="Messages flagged for a second look, whether or not Qwen was enabled.")
        st.caption("Confidence comes from the primary classifier. For a Qwen variant, correctness uses the final reviewed label.")
        calibration = go.Figure()
        calibration.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Ideal confidence", line={"dash": "dash", "color": "#9CA3B8"}))
        if not evaluation.calibration.empty:
            calibration.add_trace(go.Scatter(
                x=evaluation.calibration["mean_confidence"], y=evaluation.calibration["empirical_accuracy"],
                mode="lines+markers", name=evaluation.model, line={"color": "#5655B8"},
                customdata=evaluation.calibration["count"],
                hovertemplate="Confidence: %{x:.0%}<br>Correct: %{y:.0%}<br>Messages: %{customdata}<extra></extra>",
            ))
        calibration.update_layout(
            xaxis={"title": "Classifier confidence", "range": [0, 1], "tickformat": ".0%"},
            yaxis={"title": "Actual correctness", "range": [0, 1], "tickformat": ".0%"},
        )
        st.plotly_chart(style_chart(calibration), width="stretch")
        st.caption("A reliable classifier follows the dashed line: 80% confidence should mean about 80% correct.")
    errors = evaluation.predictions[~evaluation.predictions["correct"]]
    with st.expander(f"Inspect incorrect predictions · {len(errors)} messages"):
        if errors.empty:
            st.success("All labels matched on this dataset.")
        else:
            st.dataframe(errors, width="stretch", hide_index=True)
    st.download_button(
        "Download these predictions", data=evaluation.predictions.to_csv(index=False).encode("utf-8"),
        file_name=f"{evaluation.model.lower().replace(' ', '_')}_predictions.csv",
        mime="text/csv", key=f"predictions_{evaluation.model}",
    )


def render_evaluation(model_path: str, use_reviewer: bool, ollama_model: str) -> None:
    page_header("Model effectiveness", "Check predictions against labelled messages, then see which labels the model gets right and where it struggles.")
    st.caption("This view evaluates three-class sentiment. For seven-emotion XLM-R accuracy on a CSV/XLSX file, open Emotion lab → Predict and evaluate a file.")
    with st.container(border=True):
        st.markdown("**1. Choose the labelled messages**")
        uploaded = st.file_uploader("Evaluation dataset (CSV or XLSX)", type=["csv", "xlsx"], key="evaluation_dataset", help="Leave empty to use the demo workbook.")
        st.caption("**2. Choose a classifier** in the sidebar. The rule baseline is always included as a comparison.")
        selected = st.session_state.get("encoder_choice", "Rule-based baseline")
        st.caption(f"Selected: {selected} · Qwen second opinion: {'on' if use_reviewer else 'off'}")
    try:
        frame, source = load_dataset(uploaded)
        report = dataframe_report(frame)
    except Exception as exc:
        st.error(f"Could not read this evaluation dataset: {exc}")
        return
    if not report.valid:
        show_invalid_dataset(report)
        return
    st.caption(f"{source} · {report.rows} labelled messages")
    if uploaded is None:
        st.info("Demo evaluation · these 75 phrases check the implementation. They do not establish research accuracy.")
    digest = hashlib.sha256(frame.to_csv(index=False).encode("utf-8")).hexdigest()
    signature = (source, digest, selected, model_path, use_reviewer, ollama_model)
    if st.button("Run model evaluation", type="primary"):
        try:
            models, notices = evaluation_models(model_path, use_reviewer, ollama_model)
            for notice in notices:
                st.warning(notice)
            progress = st.progress(0, text="Preparing evaluation…")
            evaluations = []
            for index, (name, pipeline) in enumerate(models, start=1):
                progress.progress((index - 1) / len(models), text=f"Checking messages with {name}…")
                evaluations.append(evaluate_pipeline(frame, pipeline, name))
            progress.empty()
            st.session_state.model_evaluations = evaluations
            st.session_state.model_evaluation_signature = signature
        except Exception as exc:
            st.error(f"Evaluation could not finish: {exc}")
            return
    evaluations = st.session_state.get("model_evaluations", [])
    if not evaluations:
        with st.container(border=True):
            st.subheader("Your comparison will appear here")
            st.write("Select **Run model evaluation** to measure accuracy and inspect actual mistakes.")
        return
    if st.session_state.get("model_evaluation_signature") != signature:
        st.warning("These results use previous data or settings. Run evaluation again to refresh them.")
    st.divider()
    st.subheader("Results on the selected messages")
    summary = pd.DataFrame([item.summary for item in evaluations])
    if len(evaluations) == 1:
        st.caption("Only the rule-based baseline was evaluated. Select a trained encoder to compare models.")
    else:
        comparison = summary.melt(id_vars=["model"], value_vars=["accuracy", "macro_f1"], var_name="Metric", value_name="Score")
        comparison["Metric"] = comparison["Metric"].map({"accuracy": "Accuracy", "macro_f1": "Macro-F1"})
        chart = px.bar(comparison, x="model", y="Score", color="Metric", barmode="group", text_auto=".1%", range_y=[0, 1], color_discrete_sequence=["#5655B8", "#087F83"])
        chart.update_xaxes(title=None)
        chart.update_yaxes(tickformat=".0%")
        st.plotly_chart(style_chart(chart), width="stretch", config={"displayModeBar": False})
    tabs = st.tabs([item.model for item in evaluations])
    for tab, evaluation in zip(tabs, evaluations):
        with tab:
            render_model_panel(evaluation)


def main() -> None:
    st.set_page_config(page_title="Hinglish Lab", page_icon="🧪", layout="wide", initial_sidebar_state="expanded")
    apply_theme()
    if "history" not in st.session_state:
        st.session_state.history = []
    page, model_path, use_reviewer, ollama_model = render_sidebar()
    if page == "Analyze text":
        render_analysis(model_path, use_reviewer, ollama_model)
    elif page == "Emotion lab":
        from dashboard.emotions import render_emotion_lab
        render_emotion_lab(page_header, style_chart)
    elif page == "Model disagreement":
        from dashboard.disagreement import render_disagreement_lab
        render_disagreement_lab(page_header, style_chart)
    elif page == "Model effectiveness":
        render_evaluation(model_path, use_reviewer, ollama_model)
    elif page == "Dataset explorer":
        render_dataset()
    else:
        render_overview()


if __name__ == "__main__":
    main()
