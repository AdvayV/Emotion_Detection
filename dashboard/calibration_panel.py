"""Streamlit controls for fitting an experimental emotion calibration profile."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path


def render_calibration_panel(root, style_chart):
    try:
        import pandas as pd
        import plotly.express as px
        import streamlit as st
    except ImportError:
        return

    from hinglish_emotion.calibration import fit_temperature, frozen_phrase_hashes
    from hinglish_emotion.checkpoints import emotion_checkpoint, checkpoint_version
    from hinglish_emotion.emotions import LocalEmotionClassifier
    from hinglish_emotion.evaluation_guard import phrase_hash
    from hinglish_emotion.tasks import EMOTION_LABELS, LANGUAGE_PAIRS

    root = Path(root)
    st.subheader("Experimental confidence calibration")
    st.caption("Temperature scaling adjusts confidence on supplied held-out labels. It does not fix a wrong predicted emotion. Profiles are experimental and unreviewed.")
    pair = st.selectbox("Language mix", list(LANGUAGE_PAIRS), format_func=LANGUAGE_PAIRS.get, key="cal_pair")
    encoders = ["xlmr", "mbert", "muril"]
    models = {key: emotion_checkpoint(root / "models", pair, key) for key in encoders}
    available = {key: path for key, path in models.items() if path}
    if not available:
        st.info("No installed emotion checkpoint is available for this language mix.")
        return
    model_key = st.selectbox("Installed model", list(available), key="cal_model")
    upload = st.file_uploader("CSV with text and gold emotion labels", type=["csv"], key="calibration_csv")
    if upload is None:
        st.info("Upload a labelled held-out validation file to start.")
        return
    try:
        frame = pd.read_csv(upload)
    except Exception as exc:
        st.error(f"Could not read CSV: {exc}")
        return
    if "text" not in frame or not ("emotion" in frame or "label" in frame):
        st.error("This CSV needs text and emotion (or label) columns.")
        return
    split_choice = st.radio("Data declaration", ["Validation split column", "This upload is a dedicated calibration file"], key="cal_split_mode")
    split = "validation" if split_choice.startswith("Validation") else "calibration"
    if split == "validation" and "split" not in frame:
        st.error("Validation mode requires a split column declaring validation rows.")
        return
    if split == "validation":
        frame = frame[frame["split"].astype(str).str.strip().str.lower().eq("validation")].copy()
    elif "split" in frame and frame["split"].astype(str).str.strip().str.lower().isin({"train", "test"}).any():
        st.error("Declared train or test rows cannot be used for calibration.")
        return
    if "language_pair" in frame:
        from hinglish_emotion.batch import language_pair
        try:
            pairs = frame["language_pair"].map(language_pair)
        except ValueError as exc:
            st.error(f"Invalid language column: {exc}")
            return
        frame = frame[pairs.eq(pair)].copy()
        st.caption(f"Selected {LANGUAGE_PAIRS[pair]}: {len(frame)} matching rows.")
        if frame.empty:
            st.error("No calibration rows match the selected language mix.")
            return
    limit = st.select_slider("Maximum phrases processed", options=[500, 2000], value=500, key="cal_limit")
    if len(frame) > limit:
        st.warning(f"Only the first {limit} declared calibration rows will be used.")
        frame = frame.head(limit).copy()
    frame = frame.dropna(subset=["text", "emotion" if "emotion" in frame else "label"])
    label_col = "emotion" if "emotion" in frame else "label"
    if not len(frame):
        st.error("No rows with both text and gold labels are available.")
        return
    synthetic = any("synthetic" in str(value).lower() for col in ("source", "annotation_status") if col in frame for value in frame[col].dropna().unique())
    allow_synthetic = st.checkbox("Allow synthetic or unreviewed labels (profile stays experimental)", value=False, key="cal_allow_synthetic")
    if synthetic and not allow_synthetic:
        st.warning("This file declares synthetic labels. Enable the explicit experimental option to continue.")
    hashes = frozen_phrase_hashes(root)
    overlap_count = sum(phrase_hash(str(text)) in hashes for text in frame["text"])
    if overlap_count:
        st.error(f"{overlap_count} phrase(s) match frozen test/evaluation data. This file cannot be used for calibration.")
        return
    reviewed = bool("human_reviewed" in frame and frame["human_reviewed"].astype(str).str.strip().str.lower().isin({"true", "yes", "1"}).all())
    if not reviewed and not allow_synthetic:
        st.warning("The file does not establish that labels were human reviewed. Enable the explicit experimental option to continue.")
    model_path = available[model_key]
    model_version = checkpoint_version(model_path)
    label_values = frame[label_col].astype(str).tolist()
    selected_hash = hashlib.sha256(frame["text"].astype(str).str.cat(sep="\n").encode("utf-8") + "\0".encode() + "\n".join(label_values).encode("utf-8")).hexdigest()
    signature = (hashlib.sha256(upload.getvalue()).hexdigest(), selected_hash, str(model_path), model_version,
                 split, pair, limit, bool(allow_synthetic))
    run = st.button("Fit calibration", type="primary", disabled=((synthetic or not reviewed) and not allow_synthetic), key="fit_calibration")
    if run:
        st.session_state.pop("calibration_profile", None)
        try:
            with st.spinner(f"Fitting confidence on {len(frame)} phrases; this can take time on CPU..."):
                checkpoint = model_path
                classifier = LocalEmotionClassifier(checkpoint, device="cpu")
                texts = frame["text"].astype(str).tolist()
                results = classifier.predict_batch(texts, batch_size=16)
                probabilities = [result.probabilities for result in results]
                profile = fit_temperature(probabilities, frame[label_col].astype(str).tolist(),
                    model_identity={"model_path": str(checkpoint), "labels": list(EMOTION_LABELS),
                        "checkpoint_version": checkpoint_version(checkpoint), "text_mode": classifier.metadata.get("text_mode"),
                        "preparation_version": classifier.metadata.get("preparation_version"), "scope": pair},
                    split=split, scope=pair, synthetic=synthetic, allow_synthetic=allow_synthetic,
                    human_reviewed=reviewed, allow_unreviewed=allow_synthetic)
                st.session_state["calibration_profile"] = {"profile": profile, "probabilities": probabilities,
                    "labels": label_values, "signature": signature}
        except (ValueError, RuntimeError, OSError) as exc:
            st.error(f"Calibration could not be fitted: {exc}")
    saved = st.session_state.get("calibration_profile")
    if not saved:
        return
    if saved.get("signature") != signature:
        st.info("The upload or calibration settings changed. Fit calibration again to create a matching profile.")
        return
    profile = saved["profile"]
    st.caption(f"Status: {profile['status']} · scope: {profile['scope']} · {profile['rows']} rows")
    before, after = st.columns(2)
    for column, title, metrics in ((before, "Before scaling", profile["metrics_before"]), (after, "After scaling", profile["metrics_after"])):
        column.markdown(f"**{title}**")
        column.metric("NLL", f"{metrics['nll']:.4f}")
        column.metric("Brier score", f"{metrics['brier']:.4f}")
        column.metric("ECE", f"{metrics['ece']:.4f}")
    if profile["metrics_after"]["nll"] > profile["metrics_before"]["nll"] + 1e-10:
        st.warning("The fitted temperature did not improve NLL on this data. It may not transfer beyond these rows.")
    metrics = pd.DataFrame([{"Metric": label, "Before": profile["metrics_before"][key], "After": profile["metrics_after"][key]} for key, label in (("nll", "NLL"), ("brier", "Brier"), ("ece", "ECE"))])
    st.plotly_chart(style_chart(px.bar(metrics, x="Metric", y=["Before", "After"], barmode="group", title="Observed calibration metrics"), 320), width="stretch")
    st.download_button("Download experimental profile JSON", json.dumps(profile, indent=2, ensure_ascii=False).encode("utf-8"), file_name="emotion_calibration_profile.json", mime="application/json", key="download_calibration")
