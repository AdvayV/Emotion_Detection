"""Streamlit view for sequential model comparison and human-reviewed candidates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from hinglish_emotion.checkpoints import checkpoint_version, emotion_checkpoint
from hinglish_emotion.disagreement import MAX_PHRASES, add_reviewed_candidate, compare_phrases, queue_csv
from hinglish_emotion.emotions import LocalEmotionClassifier
from hinglish_emotion.model_catalog import ENCODERS, checkpoint_ready
from hinglish_emotion.tasks import EMOTION_LABELS, EMOTION_NAMES, LANGUAGE_PAIRS
from hinglish_emotion.checkpoint_metadata import emotion_checkpoint_metadata


def available_checkpoints(root: Path, language_pair: str):
    found = []
    for encoder in ENCODERS:
        path = emotion_checkpoint(
            root / "models", language_pair, encoder.key,
            ready=checkpoint_ready, read_metadata=emotion_checkpoint_metadata,
        )
        if path is not None:
            found.append((encoder.name, path, checkpoint_version(path)))
    return found


def render_disagreement_lab(page_header, style_chart, root: Path | None = None) -> None:
    root = Path(root) if root else Path(__file__).resolve().parents[1]
    page_header(
        "Model disagreement → human learning queue",
        "Compare installed seven-emotion checkpoints one at a time, then add only corrections you explicitly review.",
    )
    st.info(
        "Reviewed corrections become candidate data only. Export leaves split unspecified; assign a new versioned training dataset after review and keep frozen evaluation examples out. "
        "Adding candidates does not change current weights or accuracy."
    )
    top_left, top_right = st.columns(2)
    with top_left:
        pair = st.selectbox("Language mix", list(LANGUAGE_PAIRS), format_func=LANGUAGE_PAIRS.get, key="disagreement_language")
    installed = available_checkpoints(root, pair)
    names = [item[0] for item in installed]
    if "MuRIL" not in names:
        st.caption("MuRIL is available only for Hindi–English and requires a complete compatible seven-emotion checkpoint.")
    if "mBERT" not in names:
        st.caption("mBERT is not installed as a complete compatible seven-emotion checkpoint yet.")
    if len(installed) < 2:
        st.warning("At least two compatible trained seven-emotion checkpoints must be installed for this language mix. Demo rules do not provide model votes.")
    with top_right:
        selected_names = st.multiselect(
            "Installed checkpoints to compare", names,
            default=names[:min(len(names), 3)], key=f"disagreement_models_{pair}",
            help="Choose at least two. Each checkpoint is loaded and released before the next one runs.",
        )
    left, right = st.columns(2)
    with left:
        pasted = st.text_area("Phrases (one per line)", key="disagreement_phrases", height=130, placeholder="Aaj main bahut khush hoon\nKal exam hai and mujhe darr lag raha hai")
    with right:
        upload = st.file_uploader("Optional CSV with a text column", type=["csv"], key="disagreement_upload")
        st.caption(f"The first {MAX_PHRASES} non-empty phrases are accepted per run.")
    phrases = [line.strip() for line in pasted.splitlines() if line.strip()]
    source = "entered phrase"
    if upload is not None:
        try:
            frame = pd.read_csv(upload)
            text_column = next((column for column in frame.columns if str(column).strip().lower() in {"text", "phrase", "message", "sentence"}), None)
            if text_column is None:
                st.error("CSV needs a text, phrase, message, or sentence column.")
                return
            phrases = [str(value).strip() for value in frame[text_column].dropna() if str(value).strip()]
            source = f"uploaded CSV: {upload.name}"
        except Exception as exc:
            st.error(f"Could not read this CSV: {exc}")
            return
    if len(phrases) > MAX_PHRASES:
        st.warning(f"Only the first {MAX_PHRASES} phrases will be used.")
        phrases = phrases[:MAX_PHRASES]
    selected = [item for item in installed if item[0] in selected_names]
    text_digest = hashlib.sha256("\n".join(phrases).encode("utf-8")).hexdigest()
    signature = (pair, text_digest, tuple((name, str(path), version) for name, path, version in selected))
    signature_key = hashlib.sha256(repr(signature).encode("utf-8")).hexdigest()[:12]
    if st.session_state.get("disagreement_signature") not in (None, signature):
        st.session_state.pop("disagreement_rows", None)
        st.session_state.pop("disagreement_signature", None)
    if st.button("Compare checkpoints", type="primary", disabled=len(selected) < 2 or not phrases):
        st.session_state.pop("disagreement_rows", None)
        st.session_state.pop("disagreement_signature", None)
        try:
            # This is the shared dashboard classifier's supported cache API. Clear
            # its references before loading any model for the sequential run.
            from dashboard.emotions import trained_classifier
            st.session_state.disagreement_rows = compare_phrases(
                phrases, pair, selected,
                classifier_factory=LocalEmotionClassifier,
                clear_shared_cache=trained_classifier.clear,
            )
            st.session_state.disagreement_signature = signature
            st.session_state.disagreement_source = source
        except (ValueError, RuntimeError, OSError, TypeError) as exc:
            st.session_state.pop("disagreement_rows", None)
            st.session_state.pop("disagreement_signature", None)
            st.error(f"Comparison could not finish: {exc}")
    rows = st.session_state.get("disagreement_rows", [])
    if not rows or st.session_state.get("disagreement_signature") != signature:
        if rows:
            st.info("Phrase, language, checkpoint, or checkpoint version changed. Run comparison again to refresh results.")
        else:
            st.caption("Base probabilities stay attached to their model and source. Uncertainty and disagreement are shown separately; neither score is calibrated.")
        queue = st.session_state.get("disagreement_queue", [])
        if queue:
            st.subheader(f"Human-reviewed session queue · {len(queue)}")
            st.dataframe(pd.DataFrame(queue), hide_index=True, width="stretch")
            st.download_button("Download candidate corrections (CSV)", queue_csv(queue).encode("utf-8-sig"), file_name="human_review_candidates.csv", mime="text/csv")
            if st.button("Clear session review queue", key="clear_disagreement_queue_stale"):
                st.session_state.pop("disagreement_queue", None)
                st.rerun()
        return

    st.caption(f"Compared {len(rows)} phrase(s) from {st.session_state.get('disagreement_source', 'session input')}. Each score below is the checkpoint's base probability; no consensus label is computed.")
    queue = list(st.session_state.get("disagreement_queue", []))
    for index, row in enumerate(rows):
        st.divider()
        st.subheader(row["text"])
        c1, c2 = st.columns(2)
        c1.metric("Model disagreement", "Yes" if row["disagreement"] else "No")
        c2.metric("Score uncertainty", "At least one model" if row["uncertainty"] else "No model flagged")
        chart_rows = []
        for prediction in row["predictions"]:
            st.markdown(f"**{prediction['model']}** · {prediction['source']} · top: {EMOTION_NAMES[prediction['emotion']]} ({prediction['score']:.1%})")
            if prediction["truncated"]:
                st.warning(f"{prediction['model']} truncated this phrase at its saved token limit; treat this prediction as uncertain.")
            if prediction["experimental"]:
                st.warning(f"{prediction['model']} has experimental or unverified validation scope for this phrase.")
            for warning in prediction["warnings"]:
                st.caption(f"{prediction['model']}: {warning}")
            chart_rows.extend({"Model": prediction["model"], "Emotion": EMOTION_NAMES[label], "Base probability": prediction["probabilities"][label]} for label in EMOTION_LABELS)
        st.plotly_chart(style_chart(px.bar(pd.DataFrame(chart_rows), x="Emotion", y="Base probability", color="Model", barmode="group", range_y=[0, 1]), 330), width="stretch", key=f"disagreement_chart_{signature_key}_{index}")
        if row["pairwise_js"]:
            js_rows = [{"Models": " ↔ ".join(item["models"]), "JS divergence (base 2)": item["divergence"], "JS distance": item["distance"]} for item in row["pairwise_js"]]
            st.dataframe(pd.DataFrame(js_rows), hide_index=True, width="stretch")
        suffix = f"{signature_key}_{index}"
        emotion = st.selectbox("Your reviewed seven-emotion label", list(EMOTION_LABELS), format_func=EMOTION_NAMES.get, key=f"disagreement_label_{suffix}")
        notes = st.text_input("Review note (optional)", key=f"disagreement_notes_{suffix}")
        origin = st.selectbox("Phrase origin", ["Unknown", "Original human-written", "Generated / synthetic"], key=f"disagreement_origin_{suffix}")
        confirmed = st.checkbox("I reviewed this phrase and confirm the correction", key=f"disagreement_confirm_{suffix}")
        if st.button("Add reviewed correction", key=f"disagreement_add_{suffix}", disabled=not confirmed):
            group_id = hashlib.sha256(f"{pair}\0{row['text']}".encode("utf-8")).hexdigest()[:16]
            candidate = {
                "text": row["text"], "emotion": emotion, "language_pair": pair,
                "group_id": group_id, "source": st.session_state.get("disagreement_source", "model_disagreement_review"),
                "review_notes": notes,
                "provenance": json.dumps([{
                    "model": p["model"], "checkpoint": p["checkpoint"], "version": p["version"],
                    "predicted_emotion": p["emotion"], "score": p["score"],
                    "warnings": p["warnings"], "truncated": p["truncated"],
                } for p in row["predictions"]], ensure_ascii=False),
                "synthetic": None if origin == "Unknown" else origin == "Generated / synthetic",
            }
            queue = add_reviewed_candidate(queue, candidate)
            st.session_state.disagreement_queue = queue
            st.success("Human-reviewed correction added to this session queue.")
    st.subheader(f"Human-reviewed session queue · {len(queue)}")
    if queue:
        st.dataframe(pd.DataFrame(queue), hide_index=True, width="stretch")
        st.download_button("Download candidate corrections (CSV)", queue_csv(queue).encode("utf-8-sig"), file_name="human_review_candidates.csv", mime="text/csv")
        if st.button("Clear session review queue", key="clear_disagreement_queue"):
            st.session_state.pop("disagreement_queue", None)
            st.rerun()
    else:
        st.caption("No corrections are queued. Predictions alone are never added to the export.")
