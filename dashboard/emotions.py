"""Multilingual emotion UI with explicit prediction sources and file scoring."""
from pathlib import Path
import json
import pandas as pd
import plotly.express as px
import streamlit as st
from hinglish_emotion.emotions import EmotionDetector, LocalEmotionClassifier
from hinglish_emotion.model_catalog import ENCODERS, checkpoint_ready
from hinglish_emotion.tasks import EMOTION_LABELS, EMOTION_NAMES, LANGUAGE_PAIRS
from hinglish_emotion.checkpoint_metadata import emotion_checkpoint_metadata
from hinglish_emotion.checkpoints import checkpoint_version, emotion_checkpoint

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "data/demo/emotions/mixed_emotions_demo.csv"
ACTIVE_LANGUAGE_PAIRS = LANGUAGE_PAIRS
ICONS = {"joy":"😊", "sadness":"😢", "anger":"😡", "fear":"😨", "surprise":"😲", "disgust":"🤢", "neutral":"😐"}
EXAMPLES = {
    "Choose an example": ("", "hi-en"),
    "Hindi–English · happy": ("Aaj main bahut khush hoon because I passed my exam 😊", "hi-en"),
    "Hindi–English · fear": ("Kal exam hai and mujhe darr lag raha hai", "hi-en"),
    "Hindi–English · sad": ("Mera dost doosre shehar chala gaya, uski bahut yaad aa rahi hai.", "hi-en"),
    "Hindi–English · neutral": ("Meeting kal subah das baje hai.", "hi-en"),
    "English–French · happy": ("Je suis très heureux because I passed my exam 😊", "fr-en"),
    "English–French · sad": ("Je suis triste because my friend moved away", "fr-en"),
    "English–French · fear": ("J'ai peur de demain, the interview is stressing me out", "fr-en"),
    "English–French · neutral": ("La réunion is tomorrow at ten", "fr-en"),
}


def clear_result():
    st.session_state.pop("emotion_result", None)


def example_changed():
    text, pair = EXAMPLES[st.session_state.emotion_example]
    st.session_state.emotion_message = text
    st.session_state.emotion_language = pair
    clear_result()


@st.cache_resource(max_entries=1, show_spinner=False)
def trained_classifier(path: str, version: str):
    return LocalEmotionClassifier(path)


@st.cache_resource(max_entries=2, show_spinner=False)
def trained_sarcasm_classifier(path: str, version: str):
    """Cache an optional classifier against both its location and file version."""
    from hinglish_emotion.sarcasm import LocalSarcasmClassifier
    return LocalSarcasmClassifier(path)


def detector_for(path: str):
    return EmotionDetector(trained_classifier(path, checkpoint_version(path))) if path else EmotionDetector()


def resolve_model(choice, pair, custom=""):
    if choice == "Demo lexical rules": return None, None, ""
    if choice in {item.name for item in ENCODERS}:
        encoder = next(item for item in ENCODERS if item.name == choice)
        path = emotion_checkpoint(ROOT / "models", pair, encoder.key, ready=checkpoint_ready, read_metadata=emotion_checkpoint_metadata)
        if path is None:
            return None, None, f"No complete {choice} checkpoint trained for {LANGUAGE_PAIRS[pair]} is installed. Train with the Colab package or explicitly choose Demo lexical rules."
    elif choice == "Custom emotion checkpoint":
        if not custom.strip(): return None, None, "Enter a local checkpoint folder."
        path = Path(custom.strip())
        if not path.is_absolute(): path = ROOT / path
    if not checkpoint_ready(path): return None, None, "This checkpoint is not trained or installed yet."
    try: metadata = emotion_checkpoint_metadata(path)
    except ValueError as exc: return None, None, str(exc)
    trained = metadata.get("trained_language_pairs", [])
    if trained and pair not in trained: return None, metadata, f"This checkpoint was trained for {', '.join(trained)}, not {pair}."
    return path, metadata, ""


def use_rules():
    st.session_state.emotion_classifier_choice = "Demo lexical rules"
    clear_result()


def render_training_status(path, metadata):
    if not path:
        st.info("Choose an installed checkpoint to inspect its saved training run. Demo rules have no trained weights.")
        return
    st.write(f"**Checkpoint:** {path}")
    st.write("**Trained language mixes:** " + ", ".join(LANGUAGE_PAIRS.get(pair,pair) for pair in metadata.get("trained_language_pairs", [])))
    st.caption(f"Training status: {metadata.get('training_status')} · input: {metadata['text_mode']} · maximum tokens: {metadata['max_length']}")
    file = Path(path) / "metrics.json"
    if not file.is_file():
        st.info("No saved training metrics are bundled with this checkpoint.")
        return
    try:
        metrics = json.loads(file.read_text(encoding="utf-8"))
        if any(metrics.get(key) != metadata.get(key) for key in ("dataset_sha256", "split_sha256")):
            raise ValueError("Saved metrics do not match this checkpoint dataset and split")
        scope = metrics.get("evaluation_scope", "unverified")
        st.subheader("Recorded Colab run")
        st.caption(f"Scope: {scope}. These are saved run results, not the accuracy of your current typed phrase.")
        c=st.columns(3)
        c[0].metric("Saved test accuracy", f"{metrics['test']['accuracy']:.1%}")
        c[1].metric("Saved test macro-F1", f"{metrics['test']['macro_f1']:.4f}")
        c[2].metric("Test phrases", metrics["split_sizes"]["test"])
        if "synthetic" in scope: st.warning("Synthetic test results do not establish real-world accuracy. Confident predictions can still be wrong on ordinary phrases.")
        sizes = metrics.get("split_sizes", {})
        st.caption(f"Training phrases: {sizes.get('train', 'unknown'):,} · validation phrases: {sizes.get('validation', 'unknown'):,}" if all(isinstance(sizes.get(key), int) for key in ("train", "validation")) else "Training split sizes were not recorded.")
        language_scores = metrics.get("test_by_language", {})
        if language_scores:
            st.markdown("**Saved test results by language mix**")
            rows = [{"Language mix": LANGUAGE_PAIRS.get(pair, pair), "Accuracy": scores["accuracy"], "Macro-F1": scores["macro_f1"], "Test phrases": scores["rows"]} for pair, scores in language_scores.items()]
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={"Accuracy": st.column_config.NumberColumn(format="percent"), "Macro-F1": st.column_config.NumberColumn(format="%.4f")})
        history = pd.DataFrame(metrics.get("history", []))
        if not history.empty:
            best = history.loc[history["validation_macro_f1"].idxmax()]
            st.caption(f"Best validation checkpoint: epoch {int(best['epoch'])}. Later epochs are not automatically better.")
            st.dataframe(history,hide_index=True,width="stretch")
        st.dataframe(pd.DataFrame(metrics["test"]["per_class"]).T.reset_index(names="Emotion"),hide_index=True,width="stretch")
    except (ValueError, KeyError, TypeError, OSError) as exc: st.warning(f"Could not verify saved metrics: {exc}")


def render_emotion_lab(header, chart_style):
    header("Emotion lab", "Predict seven emotions from Hindi–English and English–French phrases, then measure results against your own dataset.")
    st.caption("Local predictions · Seven emotions · CSV and XLSX evaluation")
    with st.container(border=True):
        c1,c2=st.columns([1,1.3])
        with c1:
            if st.session_state.get("emotion_language") not in LANGUAGE_PAIRS: st.session_state.emotion_language="hi-en"
            pair=st.selectbox("Language mix",list(LANGUAGE_PAIRS),format_func=LANGUAGE_PAIRS.get,key="emotion_language",on_change=clear_result)
        with c2:
            default=1 if emotion_checkpoint(ROOT/"models",pair,"xlmr",ready=checkpoint_ready,read_metadata=emotion_checkpoint_metadata) else 0
            choice=st.selectbox("Emotion classifier",["Demo lexical rules","XLM-R","mBERT","MuRIL","Custom emotion checkpoint"],index=default,key="emotion_classifier_choice",on_change=clear_result)
        custom=st.text_input("Seven-emotion checkpoint folder",placeholder="models/xlmr_multilingual_emotion",key="custom_emotion_path",on_change=clear_result) if choice=="Custom emotion checkpoint" else ""
        path,metadata,issue=resolve_model(choice,pair,custom)
        if issue:
            st.warning(issue)
            st.button("Use demo rules instead",on_click=use_rules)
        elif metadata:
            st.success("Seven-emotion checkpoint found. Predictions will use the local trained weights.")
            st.caption(f"Input preparation: {metadata['text_mode']} · token limit: {metadata['max_length']} · model folder: {path.name}")
            if metadata.get("evaluation_scope") in {"synthetic_only","mixed_test"}: st.warning("This checkpoint was evaluated on generated phrases. Real-world accuracy is unverified; a high prediction score can still be wrong.")
        else: st.info("Demo lexical rules are active. They use words and emoji cues and are not XLM-R predictions.")
    phrase_tab,file_tab,training_tab=st.tabs(["Try a phrase","Predict and evaluate a file","Training and Colab"])
    with phrase_tab:
        left,right=st.columns([1,1.1],gap="large")
        with left,st.container(border=True):
            st.subheader("Your phrase")
            st.selectbox("Emotion example",list(EXAMPLES),key="emotion_example",on_change=example_changed)
            text=st.text_area("Phrase to analyze",key="emotion_message",height=155,placeholder="Aaj main bahut khush hoon because I passed my exam 😊",on_change=clear_result)
            version=checkpoint_version(path) if path else None
            context=st.text_area("Optional conversation context",key="emotion_context",height=80,placeholder="What happened before this message?",on_change=clear_result)
            qwen_enabled=st.checkbox("Use optional local Qwen review when needed",value=False,key="emotion_qwen_enabled",on_change=clear_result)
            sarcasm_enabled=st.checkbox("Use trained sarcasm model",value=False,key="emotion_sarcasm_enabled",on_change=clear_result)
            sarcasm_path_text=st.text_input("Sarcasm checkpoint folder",value="models/xlmr_sarcasm",key="sarcasm_checkpoint_path",on_change=clear_result) if sarcasm_enabled else "models/xlmr_sarcasm"
            sarcasm_path=Path(sarcasm_path_text.strip()) if sarcasm_path_text.strip() else ROOT/"models"/"xlmr_sarcasm"
            if not sarcasm_path.is_absolute(): sarcasm_path=ROOT/sarcasm_path
            sarcasm_ready=(sarcasm_path/"config.json").is_file()
            try: sarcasm_version=checkpoint_version(sarcasm_path) if sarcasm_ready else "missing"
            except OSError: sarcasm_version="unreadable"
            if sarcasm_enabled and not sarcasm_ready:
                st.caption("No trained sarcasm weights were found here. The workbench will use its rule-based cue score.")
            signature=(choice,str(path),version,pair,text,context,qwen_enabled,sarcasm_enabled,str(sarcasm_path),sarcasm_version)
            if st.button("Detect emotion",type="primary",width="stretch",disabled=bool(issue)):
                clear_result()
                try:
                    from hinglish_emotion.emotion_workbench import EmotionWorkbench
                    reviewer=None
                    if qwen_enabled:
                        from hinglish_emotion.ollama_emotion_reviewer import OllamaEmotionReviewer
                        reviewer=OllamaEmotionReviewer(model="qwen3:0.6b")
                    sarcasm_classifier=None
                    if sarcasm_enabled:
                        try:
                            if sarcasm_ready: sarcasm_classifier=trained_sarcasm_classifier(str(sarcasm_path),sarcasm_version)
                        except (ImportError,ValueError,RuntimeError,OSError,TypeError) as exc:
                            st.warning(f"The trained sarcasm model could not load; continuing with available cues. {exc}")
                    workbench=EmotionWorkbench(detector_for(str(path) if path else ""),reviewer=reviewer,sarcasm_classifier=sarcasm_classifier)
                    with st.spinner("Running local prediction…"): result=workbench.analyze(text,pair,context)
                    st.session_state.emotion_result={"result":result.primary,"workbench":result,"signature":signature,"text":text,"choice":choice,"context":context}
                    history=list(st.session_state.get("emotion_history",[])); history.append(result.to_dict()); st.session_state.emotion_history=history[-100:]
                except (ValueError,RuntimeError,OSError) as exc: st.error(f"The selected model could not predict: {exc}. You can explicitly switch to demo rules.")
            st.caption("Editing the phrase, language or model clears the previous result. Predictions do not train the model.")
        with right,st.container(border=True):
            saved=st.session_state.get("emotion_result")
            if not saved or saved["signature"]!=signature:
                st.subheader("Your result will appear here")
                st.write("Choose a phrase and select Detect emotion. The result shows which model actually ran.")
            else:
                result=saved["workbench"]
                primary=result.primary
                st.subheader(f"{ICONS[result.final_emotion]} {EMOTION_NAMES[result.final_emotion]}")
                if result.final_emotion != primary.emotion:
                    st.caption(f"Base prediction: {EMOTION_NAMES[primary.emotion]} · final after review/cues: {EMOTION_NAMES[result.final_emotion]}")
                st.write(saved["text"])
                st.metric("Base demo score" if primary.source=="lexical_demo" else "Base prediction score",f"{primary.confidence:.2%}")
                st.caption(f"Actual source: {primary.model_name} · {primary.source}. Scores are the base classifier probabilities, not a reviewed score or dataset accuracy.")
                if primary.needs_context: st.warning("The scores are close, low, or the input was truncated. More context may help.")
                for warning in primary.warnings: st.warning(warning)
                probabilities=pd.DataFrame({"Emotion":[EMOTION_NAMES[label] for label in EMOTION_LABELS],"Score":[primary.probabilities[label] for label in EMOTION_LABELS]})
                chart=px.bar(probabilities,y="Emotion",x="Score",orientation="h",text_auto=".1%",range_x=[0,1],color_discrete_sequence=["#6264CF"])
                chart.update_yaxes(autorange="reversed");chart.update_xaxes(tickformat=".0%")
                st.plotly_chart(chart_style(chart,300),width="stretch")
                st.caption("Scores are uncalibrated. Experimental-data notices describe validation scope, not whether this score distribution is ambiguous.")
                with st.expander("Prediction details"):
                    normalization=result.normalization
                    changes=[f"{item['raw']} → {item['normalized']}" for item in normalization.get("tokens",[]) if item.get("raw") != item.get("normalized")]
                    st.markdown("**Text preparation**")
                    st.write("Normalization changes: " + (", ".join(changes) if changes else "none"))
                    st.write("Normalized cue text: " + (normalization.get("normalized_text") or "(empty)"))
                    st.caption("These normalization details explain the rule cues. The encoder follows its saved input preparation setting; raw checkpoints keep your original phrase.")
                    cues=result.cues
                    st.markdown("**Signals and routing**")
                    relation_labels={"none":"No emoji cue", "agreement":"Emoji agrees with word cues", "conflict":"Emoji conflicts with word cues", "uncertain":"Emoji relation is uncertain"}
                    st.write(relation_labels.get(cues.get("emoji_relation"),cues.get("emoji_relation","unknown")))
                    source_label="trained binary classifier" if cues.get("sarcasm_source")=="supervised" else "heuristic rule cue (not learned)"
                    st.write(f"Sarcasm cue: {cues.get('sarcasm_score',0):.2f} · {source_label}. {cues.get('sarcasm_message','')}")
                    intensity=cues.get("intensity",{})
                    st.write(f"Intensity estimate (heuristic): valence {intensity.get('valence',0):.2f}, arousal {intensity.get('arousal',0):.2f}, strength {intensity.get('strength',0):.2f}; cues: {', '.join(intensity.get('cues',[])) or 'none'}")
                    st.write("Lexical evidence: " + (", ".join(cues.get("evidence",[])) or "none detected"))
                    review=result.review
                    st.write(f"Qwen review: {review.get('status','unknown')} · {review.get('message','')}")
                    if review.get("suggestion"):
                        st.write(f"Suggested emotion: {EMOTION_NAMES.get(review['suggestion'],review['suggestion'])} · evidence: {', '.join(review.get('evidence',[])) or 'none'}")
                    st.json({"primary":primary.to_dict(),"final_emotion":result.final_emotion,"normalization":result.normalization,"cues":result.cues,"review":result.review,"context":result.context})
        history=st.session_state.get("emotion_history",[])
        if history:
            st.download_button("Download session history (CSV)",pd.json_normalize(history).to_csv(index=False),file_name="emotion_session_history.csv",mime="text/csv")
            if st.button("Clear session history"):
                st.session_state.pop("emotion_history",None); st.rerun()
    with file_tab:
        from dashboard.emotion_batch import render_file_predictions
        render_file_predictions(ROOT,choice,pair,lambda selected,language:resolve_model(selected,language,custom),chart_style)
    with training_tab:
        render_training_status(path,metadata)
        st.subheader("Separate Colab training files")
        st.write("Start with the project training notebook and full package below. Choose XLM-R or mBERT for both language mixes, MuRIL for Hindi–English, or the separate reviewed-data sarcasm job. The guide covers Drive recovery, export and installation. New checkpoints are discovered after installation.")
        project_training=ROOT/"colab"/"project_training"
        if project_training.is_dir():
            training_downloads=[("PROJECT_TRAINING.ipynb","Download project training notebook","application/x-ipynb+json"),("project_training.zip","Download full Colab training package","application/zip"),("START_HERE.md","Download project training guide","text/markdown")]
            for name,label,mime in training_downloads:
                file=project_training/name
                if file.is_file(): st.download_button(label,file.read_bytes(),file_name=name,mime=mime)
        with st.expander("Alternative XLM-R-only training workflow"):
            folder=ROOT/"colab/xlmr_multilingual"
            for name,label,mime in [("TRAIN_XLMR.ipynb","Download multilingual Colab notebook","application/x-ipynb+json"),("xlmr_multilingual.zip","Download multilingual training package","application/zip")]:
                file=folder/name
                if file.is_file(): st.download_button(label,file.read_bytes(),file_name=name,mime=mime)
        st.caption("The bootstrap dataset is synthetic, not research validation. Replace it with reviewed labels for final claims. The selected checkpoint’s trained language mixes and saved test results are shown above.")
        with st.expander("How this extends Team 5 G2"):
            st.write("The base paper describes BERT/RoBERTa sentiment classification. This extension adds seven emotions, language-aware checkpoints, labelled-file evaluation, leakage checks and reproducible Colab training. A supervised sarcasm training workflow is available, but trained sarcasm weights are not provided. Emotion/emoji fusion is not implemented; emoji cues are shown separately.")
            paper=ROOT/"Team 5 - G2.docx"
            if paper.is_file(): st.download_button("Download base paper",paper.read_bytes(),file_name=paper.name,mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
