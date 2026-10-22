"""Interactive, bounded experiment runner for emotion preprocessing and review."""
from __future__ import annotations
import hashlib
from io import BytesIO
import json
from pathlib import Path
import pandas as pd
import streamlit as st
from hinglish_emotion.emotions import EmotionDetector, LocalEmotionClassifier
from hinglish_emotion.experiments import VARIANTS, run_experiments
from hinglish_emotion.model_catalog import ENCODERS, checkpoint_path, checkpoint_ready
from hinglish_emotion.checkpoint_metadata import emotion_checkpoint_metadata
from hinglish_emotion.checkpoints import checkpoint_version, emotion_checkpoint
from hinglish_emotion.ollama_emotion_reviewer import OllamaEmotionReviewer


def render_experiments_panel(root: Path, style_chart=None):
    root = Path(root)
    st.subheader("Ablation and reviewer study")
    st.caption("Run paired local predictions and inspect stability, latency, coverage and declared-label metrics. Draft examples do not establish benchmark accuracy.")
    pairs = ["hi-en", "fr-en"]
    language = st.selectbox("Language mix", pairs, key="experiment_language")
    available = []
    for encoder in ENCODERS:
        path = emotion_checkpoint(root / "models", language, encoder.key)
        if path is not None and checkpoint_ready(path):
            try:
                meta = emotion_checkpoint_metadata(path)
                if language in meta.get("trained_language_pairs", []): available.append((encoder.name, path))
            except (ValueError, OSError, KeyError): pass
    options = ["Demo lexical rules", *[name for name, _ in available]]
    model_name = st.selectbox("One local model", options, key="experiment_model")
    upload = st.file_uploader("CSV with text and optional emotion, split, context and language_pair columns", type=["csv"], key="experiment_data")
    bundled = [root / "evaluation_set/hindi_english_100.csv", root / "evaluation_set/english_french_100.csv"]
    source = upload.name if upload else "Bundled 200 assistant-draft examples" if all(path.is_file() for path in bundled) else "No bundled CSV"
    if upload:
        content = upload.getvalue()
    elif all(path.is_file() for path in bundled):
        try: content = pd.concat([pd.read_csv(path) for path in bundled], ignore_index=True).to_csv(index=False).encode("utf-8")
        except Exception as exc: st.error(f"Could not load bundled drafts: {exc}"); return
    else:
        st.info("Upload a CSV to run the study."); return
    digest = hashlib.sha256(content).hexdigest()
    try: frame = pd.read_csv(BytesIO(content)).fillna("")
    except Exception as exc: st.error(f"Could not read CSV: {exc}"); return
    if "language_pair" in frame: frame=frame[frame["language_pair"].eq(language)].copy()
    if "text" not in frame:
        st.error("CSV must include a text column."); return
    if "language_pair" not in frame: frame["language_pair"] = language
    st.caption(f"Source: {source} · {len(frame)} rows · sha256 {digest[:12]}")
    limit = st.number_input("Maximum rows", min_value=1, max_value=200, value=10, step=1, key="experiment_limit")
    variants = st.multiselect("Input variants", list(VARIANTS), default=["baseline", "normalized", "emoji_removed"], key="experiment_variants")
    context_enabled = st.checkbox("Include context vs no context", value=False, key="experiment_context")
    if context_enabled:
        variants = list(dict.fromkeys([*variants, "context", "no_context"]))
    reviewer_mode = st.selectbox("Optional local Qwen comparison", ["none", "selective", "all"], key="experiment_reviewer")
    review_limit = min(5, int(limit))
    if reviewer_mode != "none": st.caption(f"Reviewer calls are capped at {review_limit} per run. Reviewer output stays a suggestion; primary probabilities remain unchanged.")
    selected_checkpoint = dict(available).get(model_name)
    selected_version = checkpoint_version(selected_checkpoint) if selected_checkpoint else "demo-rules"
    signature = (digest, source, model_name, str(selected_checkpoint or ""), selected_version,
                 language, int(limit), tuple(variants), reviewer_mode, "qwen3:0.6b", min(5, int(limit)))
    if st.button("Run experiments", type="primary", disabled=not variants, key="run_experiments"):
        try:
            if model_name == "Demo lexical rules": predictor = EmotionDetector()
            else:
                path = selected_checkpoint
                from dashboard.emotions import trained_classifier
                trained_classifier.clear()
                predictor = EmotionDetector(LocalEmotionClassifier(path,device="cpu"))
            reviewer = OllamaEmotionReviewer() if reviewer_mode != "none" else None
            result = run_experiments(frame.to_dict("records"), predictor, variants, review_mode=reviewer_mode,
                                     reviewer=reviewer, row_limit=int(limit), require_test_split=True)
            st.session_state.experiment_result = {"result": result, "signature": signature}
        except Exception as exc: st.error(f"Experiment failed: {exc}")
    saved = st.session_state.get("experiment_result")
    if not saved: return
    if saved["signature"] != signature:
        st.info("Settings changed. Run the experiment again to refresh these results."); return
    result = saved["result"]
    st.warning("The bundled 200 phrases are assistant-authored drafts, not reviewed benchmark data. Their labels are illustrative; do not claim benchmark improvements.")
    summary = pd.DataFrame([{"Variant": key, **value} for key, value in result["summaries"].items()])
    st.dataframe(summary, hide_index=True, width="stretch")
    st.metric("Reviewer coverage", f"{result['review_summary']['coverage']:.1%}", help="Share of evaluated variant rows that received a reviewer attempt.")
    st.caption(f"Reviewer suggestions: {result['review_summary']['suggestions']} · errors: {result['review_summary']['errors']} · changed suggestions: {result['review_summary']['changed_labels']}")
    st.dataframe(pd.DataFrame(result["rows"]), hide_index=True, width="stretch")
    st.download_button("Download experiment rows (CSV)", pd.DataFrame(result["rows"]).to_csv(index=False).encode("utf-8-sig"), file_name="emotion_experiments.csv", mime="text/csv", key="experiment_csv")
    st.download_button("Download experiment report (JSON)", json.dumps(result, ensure_ascii=False, indent=2).encode("utf-8"), file_name="emotion_experiments.json", mime="application/json", key="experiment_json")
