from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from hinglish_emotion.emotions import EmotionDetector, LocalEmotionClassifier
from hinglish_emotion.emotion_data import read_emotion_csv, validate_emotion_rows, evaluate_emotions
from hinglish_emotion.model_catalog import ENCODERS, checkpoint_path, checkpoint_ready
from hinglish_emotion.tasks import EMOTION_LABELS, EMOTION_NAMES, LANGUAGE_PAIRS
from hinglish_emotion.checkpoint_metadata import emotion_checkpoint_metadata


ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "data" / "mixed_emotions_demo.csv"
ICONS = {"joy": "😊", "sadness": "😢", "anger": "😡", "fear": "😨", "surprise": "😲", "disgust": "🤢", "neutral": "😐"}
EXAMPLES = {
    "Choose an example": ("", "hi-en"),
    "Hindi–English · happy": ("Aaj main bahut khush hoon because I passed my exam 😊", "hi-en"),
    "French–English · happy": ("Je suis très heureux because I passed my exam 😊", "fr-en"),
    "French–English · sad": ("Je suis triste because my friend moved away", "fr-en"),
    "Hindi–English · fear": ("Kal exam hai and mujhe darr lag raha hai", "hi-en"),
}


def example_changed():
    text, pair = EXAMPLES[st.session_state.emotion_example]
    st.session_state.emotion_message = text
    st.session_state.emotion_language = pair


@st.cache_resource(max_entries=1, show_spinner=False)
def trained_classifier(path: str, version: int):
    return LocalEmotionClassifier(path)


def detector_for(path: str) -> EmotionDetector:
    if not path:
        return EmotionDetector()
    return EmotionDetector(trained_classifier(path, (Path(path) / "config.json").stat().st_mtime_ns))


def render_emotion_lab(header, chart_style):
    header("Emotion lab", "Detect the dominant feeling in a phrase, then compare performance across Hindi–English and French–English.")
    st.caption("SEVEN LABELS  ·  " + " · ".join(EMOTION_NAMES.values()))
    with st.expander("Train a real model · readiness and cloud notebook"):
        audit_path = ROOT / "data" / "audits" / "masac.json"
        if audit_path.is_file():
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            st.write("**Public dataset reuse:** " + ("evidence recorded" if audit.get("reuse_conditions_verified") else "pending publisher terms or permission"))
            st.caption(f"Source audit quarantined {audit.get('quarantined_cross_split_duplicate_rows', 0)} overlapping phrase rows. Episodes still recur across the source partitions.")
        review_path = ROOT / "data" / "audits" / "pilot_annotations.json"
        if review_path.is_file():
            review = json.loads(review_path.read_text(encoding="utf-8"))
            st.write("**Independent labels:** " + ("reconciliation completed" if review.get("ready") else "review or adjudication pending"))
        else:
            st.write("**Independent labels:** collection and two-reviewer annotation are pending.")
        st.caption("Training happens in the cloud notebook. This dashboard loads a saved model and evaluates phrases.")
        notebook_path = ROOT / "notebooks" / "train_emotion_pilot.ipynb"
        if notebook_path.is_file():
            st.download_button("Download training notebook", notebook_path.read_bytes(), file_name=notebook_path.name, mime="application/x-ipynb+json")
        bundle_path = ROOT / "runs" / "emotion_cloud_source.zip"
        if bundle_path.is_file():
            st.download_button("Download current source bundle", bundle_path.read_bytes(), file_name=bundle_path.name, mime="application/zip")
    with st.container(border=True):
        default_index = 1 if checkpoint_ready(ROOT / "models" / "xlmr_emotion") else 0
        choice = st.selectbox("Emotion classifier", ["Demo lexical rules", "XLM-R", "mBERT", "MuRIL", "Custom emotion checkpoint"], index=default_index)
        path = ""
        if choice == "Custom emotion checkpoint":
            custom = st.text_input("Seven-emotion checkpoint folder", placeholder="models/xlmr_emotion")
            if custom.strip():
                candidate = Path(custom.strip())
                path = str(candidate if candidate.is_absolute() else ROOT / candidate)
        elif choice != "Demo lexical rules":
            encoder = next(item for item in ENCODERS if item.name == choice)
            path = str(checkpoint_path(encoder, ROOT / "models", "emotion"))
        ready = choice == "Demo lexical rules" or bool(path and checkpoint_ready(path))
        metadata = None
        if ready and path:
            try:
                metadata = emotion_checkpoint_metadata(path)
            except ValueError as exc:
                st.error(str(exc))
                ready = False
        if choice == "Demo lexical rules":
            st.info("Demo mode · uses words and emoji rules in three languages. These are illustrative scores, not trained-model accuracy.")
        elif not ready:
            st.warning("Train a seven-emotion checkpoint first, or choose Demo lexical rules to try the workflow.")
        else:
            st.success("Seven-emotion checkpoint found.")
            st.caption(f"Text preparation: {metadata['text_mode']} · token limit: {metadata['max_length']}")
            scope = ", ".join(LANGUAGE_PAIRS[pair] for pair in metadata.get("validated_language_pairs", []))
            st.caption("Verified labelled test scope: " + (scope or "not established"))
            if metadata.get("training_status") != "complete":
                st.warning("Training or provenance is incomplete. Treat this checkpoint as experimental.")
        if choice == "MuRIL":
            st.warning("MuRIL targets Indian languages. Treat it as a Hindi-focused comparison; XLM-R and mBERT are the multilingual choices for this experiment.")
        st.caption("One dominant emotion per phrase. Mixed feelings, sarcasm and missing context can make that choice uncertain.")

    phrase_tab, dataset_tab = st.tabs(["Try a phrase", "Evaluate mixed dataset"])
    with phrase_tab:
        left, right = st.columns([1, 1.1], gap="large")
        with left, st.container(border=True):
            st.subheader("Your phrase")
            st.selectbox("Emotion example", list(EXAMPLES), key="emotion_example", on_change=example_changed)
            if "emotion_message" not in st.session_state:
                st.session_state.emotion_message = ""
            pair = st.selectbox("Language mix", list(LANGUAGE_PAIRS), format_func=LANGUAGE_PAIRS.get, key="emotion_language")
            if pair == "fr-en" and (metadata is None or "fr-en" not in metadata.get("validated_language_pairs", [])):
                st.warning("French–English is experimental. A bilingual reviewer and independent test data are still required.")
            text = st.text_area("Phrase to analyze", key="emotion_message", height=145, placeholder="Je suis triste because my friend moved away")
            checkpoint_version = (Path(path) / "config.json").stat().st_mtime_ns if path and checkpoint_ready(path) else None
            signature = (choice, path, checkpoint_version, pair, text)
            if choice == "MuRIL" and pair == "fr-en":
                st.info("Use XLM-R for French–English; this MuRIL experiment is Hindi–English only.")
            if st.button("Detect emotion", type="primary", width="stretch", disabled=not ready or choice == "MuRIL" and pair == "fr-en"):
                try:
                    with st.spinner("Reading the phrase…"):
                        result = detector_for(path).predict(text, pair)
                    st.session_state.emotion_result = {
                        "result": result, "signature": signature, "text": text, "choice": choice,
                    }
                except (ValueError, RuntimeError, OSError) as exc:
                    st.error(f"Emotion detection could not finish: {exc}")
        with right, st.container(border=True):
            saved = st.session_state.get("emotion_result")
            if not saved:
                st.subheader("Which feeling does the phrase express?")
                st.write("Choose an example and select **Detect emotion** to see all seven scores.")
            else:
                if saved["signature"] != signature:
                    st.info("Showing the previous phrase. Detect emotion again to update the result.")
                result = saved["result"]
                st.subheader(f"{ICONS[result.emotion]} {EMOTION_NAMES[result.emotion]}")
                st.write(saved["text"])
                st.metric("Demo score" if saved["choice"] == "Demo lexical rules" else "Model confidence", f"{result.confidence:.0%}")
                if result.needs_context:
                    st.warning("The emotion is uncertain. More context or a human label may be needed.")
                for warning in result.warnings:
                    st.warning(warning)
                st.caption("Model scores are not calibrated probabilities of being correct.")
                probabilities = pd.DataFrame({
                    "Emotion": [EMOTION_NAMES[label] for label in EMOTION_LABELS],
                    "Score": [result.probabilities[label] for label in EMOTION_LABELS],
                })
                chart = px.bar(probabilities, y="Emotion", x="Score", orientation="h", text_auto=".0%", range_x=[0, 1])
                chart.update_traces(marker_color="#6264CF")
                chart.update_yaxes(autorange="reversed")
                chart.update_xaxes(tickformat=".0%")
                st.plotly_chart(chart_style(chart, 300), width="stretch", config={"displayModeBar": False})
                if result.evidence:
                    st.caption("Matched cues: " + ", ".join(result.evidence))
                elif saved["choice"] == "Demo lexical rules":
                    st.caption("No emotional keyword matched. Neutral is a fallback, not proof of a neutral meaning.")

    with dataset_tab:
        st.subheader("Compare the two language mixes")
        st.write("Every row needs a phrase, its expected emotion, and a language mix. With declared splits, only test rows are scored.")
        uploaded = st.file_uploader("Mixed emotion dataset (CSV)", type=["csv"], key="mixed_emotion_dataset")
        try:
            rows = validate_emotion_rows(list(csv.DictReader(io.StringIO(uploaded.getvalue().decode("utf-8-sig"))))) if uploaded else read_emotion_csv(DEMO)
        except (ValueError, UnicodeError, OSError) as exc:
            st.error(f"Could not use this emotion dataset: {exc}")
            st.code("text,emotion,language_pair\nJe suis triste today,sadness,fr-en", language=None)
            return
        frame = pd.DataFrame(rows)
        counts = st.columns(3)
        counts[0].metric("Phrases in file", len(frame), border=True)
        counts[1].metric("Hindi–English", int(frame["language_pair"].eq("hi-en").sum()), border=True)
        counts[2].metric("French–English", int(frame["language_pair"].eq("fr-en").sum()), border=True)
        if uploaded is None or frame.get("source", pd.Series(dtype=str)).eq("illustrative").any():
            st.info("Illustrative dataset · 84 drafted examples with suggested labels. It includes indirect phrases that demo rules can miss. Validate the labels independently before research use.")
        with st.expander("Browse labels and phrases"):
            st.dataframe(pd.crosstab(frame["language_pair"], frame["emotion"]).reindex(columns=EMOTION_LABELS, fill_value=0), width="stretch")
            st.dataframe(frame, width="stretch", hide_index=True)
            if uploaded is None:
                st.download_button("Download mixed emotion demo CSV", DEMO.read_bytes(), file_name=DEMO.name, mime="text/csv")
        available_pairs = ["hi-en"] if choice == "MuRIL" else list(LANGUAGE_PAIRS)
        selected_pairs = st.multiselect("Evaluate language mixes", available_pairs, default=available_pairs, format_func=LANGUAGE_PAIRS.get)
        selected_rows = [row for row in rows if row["language_pair"] in selected_pairs]
        evaluation_signature = (choice, path, checkpoint_version, hashlib.sha256(json.dumps(selected_rows, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest())
        if st.button("Evaluate seven emotions", type="primary", disabled=not ready or not selected_rows):
            try:
                with st.spinner("Comparing each prediction with its expected emotion…"):
                    report = evaluate_emotions(selected_rows, detector_for(path))
                st.session_state.emotion_evaluation = {"report": report, "signature": evaluation_signature, "choice": choice}
            except (ValueError, RuntimeError, OSError) as exc:
                st.error(f"Emotion evaluation could not finish: {exc}")
        saved = st.session_state.get("emotion_evaluation")
        if not saved:
            st.caption("Run evaluation to generate overall, per-language and per-emotion results.")
            return
        if saved["signature"] != evaluation_signature:
            st.warning("These scores use previous data or model settings. Run evaluation again to refresh them.")
        report = saved["report"]
        if report.get("synthetic_dataset"):
            st.warning("This dataset uses generated labels. Its scores describe synthetic examples, not independent real-world accuracy.")
        st.caption(f"Evaluated with {saved['choice']} · scope: {report['scope']}")
        if saved["choice"] == "Demo lexical rules":
            st.warning("These scores measure the lexical demo. They are not mBERT, MuRIL or XLM-R results.")
        overall = report["overall"]
        if report.get("training_overlap_rows"):
            st.warning(f"{report['training_overlap_rows']} phrase(s) overlap this checkpoint's training or validation data. These are not independent test scores.")
        for pair in report.get("experimental_language_pairs", []):
            st.warning(f"{LANGUAGE_PAIRS[pair]} scores are experimental until reviewed seven-class coverage is established.")
        columns = st.columns(3)
        columns[0].metric("Emotion accuracy", f"{overall['accuracy']:.1%}", border=True, help="Share of phrases with the expected dominant emotion.")
        columns[1].metric("Seven-class macro-F1", f"{overall['macro_f1']:.1%}", border=True, help="Each of the seven emotion classes contributes equally.")
        columns[2].metric("Phrases evaluated", overall["rows"], border=True)
        language_scores = []
        for pair, scores in report["by_language"].items():
            language_scores.append({"Language mix": LANGUAGE_PAIRS[pair], "Accuracy": scores["accuracy"], "Macro-F1": scores["macro_f1"], "Phrases": scores["rows"]})
            if scores["missing_emotions"]:
                st.warning(f"{LANGUAGE_PAIRS[pair]} has no test examples for: {', '.join(scores['missing_emotions'])}. This is incomplete seven-class coverage.")
        st.markdown("**Accuracy by language mix**")
        st.dataframe(pd.DataFrame(language_scores), width="stretch", hide_index=True, column_config={"Accuracy": st.column_config.NumberColumn(format="percent"), "Macro-F1": st.column_config.NumberColumn(format="percent")})
        left, right = st.columns(2)
        with left:
            matrix = pd.DataFrame(overall["confusion_matrix"]).T.loc[list(EMOTION_LABELS), list(EMOTION_LABELS)]
            chart = px.imshow(matrix, text_auto=True, aspect="auto", color_continuous_scale="Purples", labels={"x": "Predicted emotion", "y": "Expected emotion"})
            st.plotly_chart(chart_style(chart, 390), width="stretch")
            st.caption("Correct predictions lie on the diagonal.")
        with right:
            per_class = pd.DataFrame(overall["per_class"]).T.reset_index(names="Emotion")
            st.markdown("**Scores for each emotion**")
            st.dataframe(per_class, width="stretch", hide_index=True, column_config={name: st.column_config.NumberColumn(format="%.3f") for name in ("precision", "recall", "f1")})
        predictions = pd.DataFrame(report["predictions"])
        with st.expander("Inspect incorrect emotion predictions"):
            st.dataframe(predictions[~predictions["correct"]], width="stretch", hide_index=True)
        st.download_button("Download emotion predictions", predictions.to_csv(index=False).encode("utf-8"), file_name="mixed_emotion_predictions.csv", mime="text/csv")
