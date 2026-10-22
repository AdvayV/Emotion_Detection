"""Research workflows with explicit boundaries between software and trained assets."""
from pathlib import Path
import hashlib,json
import pandas as pd
import plotly.express as px
import streamlit as st


def render_conversation_panel(root,style_chart):
    from dashboard.emotions import resolve_model,detector_for
    from hinglish_emotion.checkpoints import checkpoint_version
    from hinglish_emotion.conversation_lab import parse_turns,analyze_turns
    from hinglish_emotion.emotion_workbench import EmotionWorkbench
    from hinglish_emotion.tasks import LANGUAGE_PAIRS
    st.subheader("Conversation emotion trace")
    st.caption("Track seven-emotion predictions by turn and speaker. Earlier turns reach Qwen when enabled. The encoder uses context only with a checkpoint explicitly trained for it.")
    c=st.columns(2)
    pair=c[0].selectbox("Conversation language",list(LANGUAGE_PAIRS),format_func=LANGUAGE_PAIRS.get,key="conversation_pair")
    choice=c[1].selectbox("Conversation classifier",["Demo lexical rules","XLM-R","mBERT","MuRIL","XLM-R learned fusion"],index=1,key="conversation_model")
    path,metadata,issue=resolve_model(choice,pair)
    if issue: st.warning(issue)
    if metadata and not metadata.get("trained_with_context"):
        st.info("This checkpoint predicts individual phrases. Earlier turns do not change its base prediction; optional Qwen can review the supplied history.")
    text=st.text_area("Conversation turns (Speaker: phrase)",height=160,key="conversation_text",placeholder="A: Aaj main bahut khush hoon!\nB: Mujhe kal ke interview se darr lag raha hai.")
    review=st.checkbox("Use local Qwen for conversation review",value=False,key="conversation_qwen")
    history=int(st.slider("Earlier turns available as context",0,8,4,key="conversation_window"))
    version=checkpoint_version(path) if path else "demo"
    signature=(pair,choice,str(path),version,text,review,history)
    if st.button("Analyze conversation",disabled=bool(issue),type="primary",key="run_conversation"):
        st.session_state.pop("conversation_result",None)
        try:
            turns=parse_turns(text)
            if review and len(turns)>5: raise ValueError("Use at most five turns when Qwen review is enabled")
            reviewer=None
            if review:
                from hinglish_emotion.ollama_emotion_reviewer import OllamaEmotionReviewer
                reviewer=OllamaEmotionReviewer()
            workbench=EmotionWorkbench(detector_for(str(path) if path else ""),reviewer=reviewer)
            with st.spinner("Analyzing conversation locally…"):
                rows=analyze_turns(turns,workbench,pair,history)
            st.session_state.conversation_result={"rows":rows,"signature":signature}
        except (ValueError,RuntimeError,OSError) as exc: st.error(str(exc))
    saved=st.session_state.get("conversation_result")
    if not saved or saved["signature"]!=signature: return
    rows=saved["rows"];frame=pd.DataFrame(rows)
    st.dataframe(frame[["turn","speaker","text","base_emotion","final_emotion","base_score","speaker_emotion_changed","review_status"]],hide_index=True,width="stretch")
    chart_rows=[{"Turn":row["turn"],"Emotion":label,"Base score":score} for row in rows for label,score in row["probabilities"].items()]
    st.plotly_chart(style_chart(px.line(pd.DataFrame(chart_rows),x="Turn",y="Base score",color="Emotion",markers=True),350),width="stretch")
    st.caption("Score curves describe model outputs across turns; they are not measured emotional intensity or a validated mental-state trajectory.")
    st.download_button("Download conversation analysis",json.dumps(rows,ensure_ascii=False,indent=2).encode("utf-8"),"conversation_analysis.json","application/json",key="conversation_download")


def render_research_workbench(page_header,style_chart):
    root=Path(__file__).resolve().parents[1]
    page_header("Research workbench","Review data, test changes, inspect confidence and prepare the next GPU experiment.")
    section=st.radio("Research workflow",["Experiments","Confidence calibration","Human review","Conversation","GPU training"],horizontal=True,key="research_workflow")
    if section=="Experiments":
        from dashboard.experiments_panel import render_experiments_panel
        render_experiments_panel(root,style_chart)
    elif section=="Confidence calibration":
        from dashboard.calibration_panel import render_calibration_panel
        render_calibration_panel(root,style_chart)
    elif section=="Human review":
        from dashboard.review_panel import render_review_panel
        render_review_panel(root,style_chart)
    elif section=="Conversation": render_conversation_panel(root,style_chart)
    else:
        st.subheader("Learned emoji fusion and optional supervised heads")
        st.write("The GPU workflow trains an XLM-R text encoder with a learned emoji embedding and gate. Context-conditioned emotion and an auxiliary sarcasm head are enabled only when suitable reviewed labels are supplied.")
        st.warning("The existing dataset has no reviewed sarcasm or conversation labels. Its default run trains emotion and emoji fusion only. No trained advanced checkpoint is installed yet.")
        ready=(root/"models/xlmr_fusion_emotion/model.safetensors").is_file()
        st.write("Advanced checkpoint: "+("installed; choose XLM-R learned fusion in Emotion lab" if ready else "awaiting the GPU training/export run"))
        folder=root/"colab/advanced_training"
        for filename,label,mime in [("TRAIN_ADVANCED.ipynb","Download advanced Colab notebook","application/x-ipynb+json"),("TRAIN_ADVANCED_KAGGLE.ipynb","Download advanced Kaggle notebook","application/x-ipynb+json"),("xlmr_advanced_training.zip","Download advanced training package","application/zip"),("START_HERE.md","Download advanced training instructions","text/markdown")]:
            file=folder/filename
            if file.is_file(): st.download_button(label,file.read_bytes(),filename,mime,key="advanced_"+filename)
        st.code('python -m training.install_checkpoint "C:/path/to/xlmr_fusion_emotion.zip"',language="powershell")
        st.caption("Training stays on the GPU notebook. Local inference uses the installed exported weights. Existing XLM-R, mBERT and MuRIL checkpoints are retained.")
