"""Small, accessible visual theme for the Streamlit workspace."""

import streamlit as st


PALETTE = {
    "ink": "#182638",
    "muted": "#58697C",
    "teal": "#087F83",
    "indigo": "#5655B8",
    "line": "#DCE5E8",
    "canvas": "#F5F8F8",
    "surface": "#FFFFFF",
}


def apply_theme() -> None:
    """Apply restrained styling while keeping Streamlit navigation visible."""
    st.markdown(
        """
        <style>
        :root {
          --lab-ink: #182638; --lab-muted: #58697C; --lab-teal: #087F83;
          --lab-indigo: #5655B8; --lab-line: #DCE5E8; --lab-surface: #FFFFFF;
        }
        .stApp { background: #F5F8F8; color: var(--lab-ink); }
        [data-testid="stHeader"] { background: rgba(245, 248, 248, .96); }
        [data-testid="stAppViewContainer"] > .main .block-container {
          max-width: 1440px; padding: 2.2rem clamp(1rem, 4vw, 3.6rem) 3.5rem;
        }
        [data-testid="stSidebar"] { background: #EEF4F4; border-right: 1px solid var(--lab-line); }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stSidebar"] label { color: var(--lab-ink); }
        h1, h2, h3 { color: var(--lab-ink); letter-spacing: -0.025em; }
        h1 { font-weight: 730; }
        [data-testid="stCaptionContainer"] { color: var(--lab-muted); }
        [data-testid="stMetric"] {
          background: var(--lab-surface); border: 1px solid var(--lab-line);
          padding: 1rem 1.1rem; border-radius: 14px;
        }
        [data-testid="stVerticalBlockBorderWrapper"] {
          border-color: var(--lab-line); border-radius: 16px;
          background: rgba(255, 255, 255, .82);
        }
        [data-testid="stAlert"] { border-radius: 12px; }
        div.stButton > button[kind="primary"] { background: var(--lab-teal); border-color: var(--lab-teal); }
        div.stButton > button { border-radius: 10px; min-height: 2.7rem; }
        input, textarea, [data-baseweb="select"] > div { border-radius: 9px; }
        @media (max-width: 760px) {
          [data-testid="stAppViewContainer"] > .main .block-container { padding: 1.25rem 1rem 2.25rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
