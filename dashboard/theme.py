"""Shared visual language for CodeMood's local research dashboard.

HTML is limited to presentation; Streamlit owns every interactive control.
"""
from html import escape

import streamlit as st


PALETTE = {
    "ink": "#19243B", "muted": "#59677D", "teal": "#087F83",
    "indigo": "#574CE8", "line": "#E1E6EF", "canvas": "#F5F6FA",
    "surface": "#FFFFFF",
}
EMOTION_COLORS = {
    "joy": "#B8740A", "sadness": "#456BD2", "anger": "#C44953",
    "fear": "#8059BA", "surprise": "#BD6232", "disgust": "#418469",
    "neutral": "#718096",
}
WORKSPACE_META = {
    "Emotion lab": ("PREDICT & EXPLORE", "One phrase. Seven possible emotions."),
    "Model comparison": ("MEASURE & COMPARE", "The same phrases. A fairer comparison."),
    "Model disagreement": ("INSPECT & REVIEW", "See where the models part ways."),
    "Ambiguity lab": ("UNCERTAINTY & COVERAGE", "Make room for more than one emotion."),
    "Analyze text": ("SENTIMENT STUDIO", "Explore the tone behind the words."),
    "Model effectiveness": ("SENTIMENT EVALUATION", "Measure predictions against labels."),
    "Dataset explorer": ("DATA COLLECTION", "Get to know the phrases behind the results."),
    "Research workbench": ("RESEARCH TOOLS", "Inspect, review and prepare the next experiment."),
    "How the project works": ("THE CODEMOOD FRAMEWORK", "From mixed-language phrases to evidence."),
}


def page_intro(title: str, description: str) -> None:
    """Render an escaped, responsive page introduction, without fake live status."""
    eyebrow, tagline = WORKSPACE_META.get(title, ("CODEMOOD WORKSPACE", "Explore language. Understand emotion."))
    st.markdown(
        f'<section class="cm-hero" aria-label="{escape(title, quote=True)}">'
        f'<div class="cm-eyebrow">CODEMOOD <span>/</span> {escape(eyebrow)}</div>'
        f'<h1>{escape(title)}</h1><p class="cm-tagline">{escape(tagline)}</p>'
        f'<p class="cm-description">{escape(description)}</p>'
        '<div class="cm-orbit" aria-hidden="true"><i></i><i></i><i></i></div></section>',
        unsafe_allow_html=True,
    )


def sidebar_brand() -> None:
    st.markdown(
        '<div class="cm-brand"><div class="cm-monogram" aria-hidden="true">cm<span>•</span></div>'
        '<div><strong>CodeMood</strong><p>LANGUAGE × EMOTION</p></div></div>'
        '<div class="cm-local"><span aria-hidden="true"></span> Local research workspace</div>',
        unsafe_allow_html=True,
    )


def section_label(number: str, title: str, description: str = "") -> None:
    st.markdown(
        f'<div class="cm-section-label"><span>{escape(number)}</span>'
        f'<div><strong>{escape(title)}</strong>'
        + (f'<p>{escape(description)}</p>' if description else '') + '</div></div>',
        unsafe_allow_html=True,
    )


def status_strip(items: list[tuple[str, str]]) -> None:
    st.markdown(
        '<div class="cm-status-strip">' + ''.join(
            f'<div><span>{escape(label)}</span><strong>{escape(value)}</strong></div>'
            for label, value in items
        ) + '</div>', unsafe_allow_html=True,
    )


def empty_prediction() -> None:
    st.markdown(
        '<div class="cm-empty"><div class="cm-spectrum" aria-hidden="true">'
        + ''.join(f'<i style="--emotion:{color}"></i>' for color in EMOTION_COLORS.values())
        + '</div><h3>Your result will appear here</h3>'
        '<p>Write a phrase or choose an example, then select <strong>Detect emotion</strong>.</p>'
        '<div class="cm-empty-steps"><span>01 &nbsp; Choose a model</span>'
        '<span>02 &nbsp; Add your phrase</span><span>03 &nbsp; Explore the scores</span></div>'
        '<small>Seven emotions · Base scores · Review signals</small></div>',
        unsafe_allow_html=True,
    )


def prediction_banner(label: str, icon: str, *, demo: bool = False) -> None:
    color = EMOTION_COLORS.get(label, PALETTE["indigo"])
    names = {"joy": "Happy / Joy", "sadness": "Sadness", "anger": "Anger", "fear": "Fear", "surprise": "Surprise", "disgust": "Disgust", "neutral": "Neutral"}
    st.markdown(
        f'<div class="cm-prediction" style="--emotion:{color}">'
        f'<span class="cm-emotion-icon" aria-hidden="true">{escape(icon)}</span>'
        f'<div><div class="cm-eyebrow">{"DEMO RULE RESULT" if demo else "DETECTED EMOTION"}</div>'
        f'<h3>{escape(names.get(label, label.title()))}</h3></div></div>',
        unsafe_allow_html=True,
    )


def apply_theme() -> None:
    """Style public controls without replacing widgets or hiding navigation."""
    st.markdown("""
    <style>
    :root { --lab-ink:#19243B; --lab-muted:#59677D; --lab-line:#E1E6EF;
      --lab-indigo:#574CE8; --lab-teal:#087F83; --lab-surface:#FFFFFF; }
    .stApp { background:#F5F6FA; color:var(--lab-ink); }
    [data-testid="stHeader"] { background:rgba(245,246,250,.96); }
    [data-testid="stMainBlockContainer"],
    [data-testid="stAppViewContainer"] > .main .block-container {
      max-width:1480px; padding:2rem clamp(1.25rem,3.5vw,3.5rem) 4rem; }
    h1,h2,h3 { color:var(--lab-ink); letter-spacing:-.025em; }
    h1 { font-weight:750; } h2 { font-size:1.6rem; } h3 { font-size:1.2rem; }
    [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color:var(--lab-muted); line-height:1.6; }
    [data-testid="stSidebar"] { background:#FFFFFF; border-right:1px solid var(--lab-line); }
    [data-testid="stSidebarUserContent"] { padding:1.5rem 1.25rem 2rem; }
    .cm-brand { display:flex; align-items:center; gap:.8rem; margin:.3rem 0 1.3rem; }
    .cm-monogram { width:45px; height:45px; border-radius:14px; background:#242746;
      color:#FFF; font-size:1.35rem; font-weight:750; display:flex;
      justify-content:center; align-items:center; letter-spacing:-.08em; }
    .cm-monogram span { color:#B7B0FF; margin-left:2px; }
    .cm-brand strong { font-size:1.4rem; letter-spacing:-.04em; color:#19243B; }
    .cm-brand p { font-size:.59rem; letter-spacing:.16em; color:#59677D; margin:0; }
    .cm-local { display:flex; align-items:center; gap:.45rem; background:#F0F5F4;
      color:#335C54; border:1px solid #DCE9E4; border-radius:8px; padding:.45rem .65rem;
      font-size:.73rem; margin-bottom:1.2rem; }
    .cm-local > span { width:6px; height:6px; border-radius:50%; background:#387F6A; }
    [data-testid="stSidebar"] [data-testid="stRadio"] [role="radiogroup"] { gap:.2rem; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
      border-radius:9px; padding:.48rem .65rem; margin:0; transition:background .15s; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover { background:#F3F3FA; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
      background:#EEECFD; color:#4539C0; box-shadow:inset 3px 0 0 #6759DD; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label p { font-size:.87rem; font-weight:550; }
    [data-testid="stSidebar"] h3 { font-size:1rem; }
    .cm-hero { position:relative; isolation:isolate; overflow:hidden; color:#FFF;
      background:linear-gradient(115deg,#20243E 0%,#31325B 62%,#47417C 100%);
      border:1px solid #3F416B; border-radius:21px; padding:1.9rem 2.2rem;
      margin:0 0 .6rem; box-shadow:0 10px 25px rgba(30,34,65,.08); }
    .cm-eyebrow { font-size:.64rem; font-weight:700; letter-spacing:.14em; line-height:1.6; }
    .cm-hero .cm-eyebrow { color:#CBC8EE; }
    .cm-eyebrow > span { margin:0 .65rem; opacity:.6; }
    .cm-hero h1 { color:#FFF; font-size:2.3rem; line-height:1.15; margin:.5rem 0 .4rem;
      padding:0; max-width:85%; letter-spacing:-.045em; }
    .cm-hero .cm-tagline { color:#EEEFFC; font-size:1.05rem; margin:0 0 .45rem; }
    .cm-hero .cm-description { color:#D2D5E8; max-width:760px; font-size:.88rem;
      line-height:1.65; margin:0; position:relative; z-index:1; }
    .cm-orbit { position:absolute; right:-30px; top:-70px; width:300px; height:300px;
      z-index:-1; opacity:.45; pointer-events:none; }
    .cm-orbit i { position:absolute; inset:0; border:1px solid #9C94D5; border-radius:50%; }
    .cm-orbit i:nth-child(2) { inset:42px; } .cm-orbit i:nth-child(3) { inset:84px; background:#6D64A4; opacity:.35; }
    .cm-status-strip { display:flex; flex-wrap:wrap; gap:.6rem 2rem; margin:.2rem 0 .65rem;
      border:1px solid var(--lab-line); border-radius:12px; background:#FFF; padding:.8rem 1.1rem; }
    .cm-status-strip > div { display:flex; gap:.6rem; align-items:center; font-size:.77rem; }
    .cm-status-strip span { color:var(--lab-muted); } .cm-status-strip strong { color:var(--lab-ink); font-weight:600; }
    [data-testid="stMetric"] { background:#FFF; border:1px solid var(--lab-line);
      border-radius:14px; padding:1rem 1.1rem; box-shadow:0 3px 8px rgba(30,34,65,.025); }
    [data-testid="stMetricLabel"] { color:#59677D; }
    [data-testid="stMetricValue"] { color:#202B44; font-variant-numeric:tabular-nums; letter-spacing:-.04em; }
    [data-testid="stVerticalBlockBorderWrapper"], [data-testid="stVerticalBlock"]:has(> [data-testid="stVerticalBlockBorderWrapper"]) {
      border-color:var(--lab-line); }
    div[data-testid="stVerticalBlockBorderWrapper"] > div { border-radius:16px; }
    .st-key-emotion_setup,.st-key-emotion_input,.st-key-emotion_output,.st-key-comparison_setup,
    .st-key-overview_predict,.st-key-overview_compare,.st-key-overview_data {
      background:#FFF; border-radius:16px; box-shadow:0 4px 14px rgba(30,34,65,.025); }
    .cm-section-label { display:flex; align-items:center; gap:.8rem; padding:.15rem 0 .7rem; }
    .cm-section-label > span { color:#5B50C8; background:#F0EEFC; border:1px solid #E1DCF7;
      border-radius:9px; width:34px; height:34px; display:flex; align-items:center;
      justify-content:center; font-size:.73rem; font-weight:750; flex-shrink:0; }
    .cm-section-label strong { font-size:1rem; color:#202B44; }
    .cm-section-label p { margin:.15rem 0 0; font-size:.78rem; color:#59677D; }
    [data-testid="stAlert"] { border-radius:11px; font-size:.88rem; }
    div.stButton > button, div.stDownloadButton > button, [data-testid="stFormSubmitButton"] > button {
      border-radius:9px; min-height:2.65rem; font-weight:600; transition:box-shadow .15s,background .15s; }
    div.stButton > button[kind="primary"], [data-testid="stFormSubmitButton"] > button[kind="primary"] {
      background:#574CE8; border-color:#574CE8; color:#FFF; box-shadow:0 3px 9px #574CE820; }
    div.stButton > button[kind="primary"]:hover { background:#483DD1; border-color:#483DD1; }
    button:focus-visible, input:focus-visible, textarea:focus-visible { outline:3px solid #9991E9 !important; outline-offset:3px; }
    div.stButton > button:disabled { background:#E9ECF2 !important; border-color:#D4DAE5 !important;
      color:#68758B !important; cursor:not-allowed; opacity:.85; box-shadow:none; }
    [data-baseweb="textarea"], [data-baseweb="input"], [data-baseweb="select"] > div {
      border-radius:9px; background:#F7F8FC; }
    textarea { line-height:1.7 !important; }
    [data-baseweb="tab-list"] { gap:.25rem; border-bottom:1px solid #E1E6EF; padding-bottom:.3rem; }
    [data-baseweb="tab"] { border-radius:8px; padding:.55rem .9rem; height:auto; font-size:.85rem; }
    [data-baseweb="tab"][aria-selected="true"] { color:#4B3DC8; background:#EFEDFC; }
    [data-baseweb="tab-highlight"] { background:#6759DD; }
    [data-testid="stExpander"] { border-color:#E1E6EF; border-radius:11px; background:#FFFFFF; }
    [data-testid="stDataFrame"], [data-testid="stPlotlyChart"] { border-radius:11px; overflow:hidden; }
    .cm-empty { text-align:center; padding:3.8rem 1rem 3rem; min-height:405px; }
    .cm-spectrum { display:flex; align-items:center; justify-content:center; gap:7px; height:65px; margin-bottom:1.6rem; }
    .cm-spectrum i { width:9px; height:24px; background:var(--emotion); border-radius:6px; opacity:.8; }
    .cm-spectrum i:nth-child(2),.cm-spectrum i:nth-child(6) { height:38px; }
    .cm-spectrum i:nth-child(3),.cm-spectrum i:nth-child(5) { height:53px; }
    .cm-spectrum i:nth-child(4) { height:65px; }
    .cm-empty h3 { font-size:1.15rem; margin:0 0 .6rem; padding:0; }
    .cm-empty p { font-size:.88rem; line-height:1.7; color:#59677D; max-width:320px; margin:auto; }
    .cm-empty-steps { display:flex; flex-direction:column; gap:.5rem; margin:1.7rem auto; width:fit-content;
      text-align:left; color:#59677D; font-size:.77rem; }
    .cm-empty small { color:#59677D; font-size:.71rem; }
    .cm-prediction { display:flex; align-items:center; gap:1rem; padding:1rem;
      border-left:4px solid var(--emotion); background:#F7F8FC; border-radius:10px; margin-bottom:.5rem; }
    .cm-prediction .cm-eyebrow { color:#59677D; font-size:.61rem; }
    .cm-prediction h3 { font-size:1.65rem; margin:.1rem 0 0; padding:0; }
    .cm-emotion-icon { font-size:2rem; }
    @media (max-width: 760px) {
      [data-testid="stMainBlockContainer"], [data-testid="stAppViewContainer"] > .main .block-container {
        padding:3.5rem 1rem 2.5rem; }
      .cm-hero { padding:1.4rem; border-radius:16px; } .cm-hero h1 { font-size:1.8rem; max-width:100%; }
      .cm-hero .cm-tagline { font-size:.95rem; } .cm-orbit { opacity:.2; }
      .cm-status-strip { gap:.55rem 1rem; } .cm-status-strip > div { flex-wrap:wrap; gap:.3rem; }
      .cm-empty { padding:2rem .5rem; min-height:310px; }
      [data-baseweb="tab-list"] { overflow-x:auto; } [data-baseweb="tab"] { white-space:nowrap; }
    }
    @media (prefers-reduced-motion: reduce) { *,*::before,*::after { transition:none !important; } }
    </style>
    """, unsafe_allow_html=True)
