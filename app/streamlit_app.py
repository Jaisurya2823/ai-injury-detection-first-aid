from __future__ import annotations

import mimetypes
import sys
from pathlib import Path

# Ensure the project root is on sys.path so `app.*` imports work regardless of
# how Streamlit is launched (e.g. `streamlit run app/streamlit_app.py` from the
# project root on Windows, where Streamlit does not add the root automatically).
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

from app.config import Settings, _load_dotenv

# ── Auto-create .env from .env.example if .env is missing ────────────────────
_env_file  = _ROOT / ".env"
_example   = _ROOT / ".env.example"
if not _env_file.exists() and _example.exists():
    import shutil
    shutil.copy(_example, _env_file)
    # Re-run dotenv loader so the new file takes effect immediately
    _load_dotenv()

from app.core.factory import build_service

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SAFE-AI · First Aid",
    page_icon="🩹",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}

/* ── Background ──────────────────────────────────────────────────────────── */
.stApp {
    background:
        radial-gradient(ellipse 80% 50% at 15% 15%, rgba(37,99,235,0.07) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 85% 85%, rgba(220,38,38,0.06) 0%, transparent 60%),
        radial-gradient(ellipse 50% 60% at 50% 50%, rgba(79,70,229,0.04) 0%, transparent 70%),
        linear-gradient(160deg, #010a1e 0%, #030d24 30%, #060420 60%, #010a1e 100%) !important;
    min-height: 100vh;
}

/* ── Remove Streamlit chrome ──────────────────────────────────────────────── */
#MainMenu, footer, header { visibility: hidden; display: none !important; }
.stDeployButton { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }

/* ── Sidebar ──────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: rgba(3, 10, 28, 0.92) !important;
    border-right: 1px solid rgba(255,255,255,0.05) !important;
    backdrop-filter: blur(30px) !important;
}
[data-testid="stSidebar"] > div:first-child { padding-top: 1.75rem !important; }

/* ── Scrollbar ───────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 99px; }

/* ── Typography ──────────────────────────────────────────────────────────── */
.stMarkdown p, .stMarkdown li { color: #7a8fa8 !important; }
h1, h2, h3 { color: #dce9ff !important; }

/* ── Metrics ──────────────────────────────────────────────────────────────── */
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 18px !important;
    padding: 18px 20px !important;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.05), 0 4px 16px rgba(0,0,0,0.2) !important;
    backdrop-filter: blur(12px) !important;
}
[data-testid="stMetricLabel"] {
    color: #3d4f66 !important;
    font-size: 0.68rem !important;
    font-weight: 800 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
}
[data-testid="stMetricValue"] {
    color: #dce9ff !important;
    font-size: 1.25rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.01em !important;
}
[data-testid="stMetricDelta"] { display: none !important; }

/* ── Form labels ─────────────────────────────────────────────────────────── */
.stSelectbox label, .stMultiSelect label {
    color: #3d4f66 !important;
    font-size: 0.7rem !important;
    font-weight: 800 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
}
.stRadio > label {
    color: #3d4f66 !important;
    font-size: 0.7rem !important;
    font-weight: 800 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
}

/* ── Selects / dropdowns ─────────────────────────────────────────────────── */
.stSelectbox > div > div,
.stMultiSelect > div > div {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.09) !important;
    border-radius: 12px !important;
    color: #c8daff !important;
    backdrop-filter: blur(12px) !important;
}
.stSelectbox svg, .stMultiSelect svg { color: #3d4f66 !important; }
.stRadio div[role="radiogroup"] label span { color: #8898b0 !important; }
.stRadio div[role="radiogroup"] label p { color: #8898b0 !important; font-size: 0.9rem !important; font-weight: 500 !important; text-transform: none !important; letter-spacing: normal !important; }

/* ── File uploader ───────────────────────────────────────────────────────── */
[data-testid="stFileUploader"] > section {
    background: rgba(255,255,255,0.03) !important;
    border: 1.5px dashed rgba(37,99,235,0.35) !important;
    border-radius: 20px !important;
    transition: all 0.25s ease !important;
}
[data-testid="stFileUploader"] > section:hover {
    border-color: rgba(37,99,235,0.65) !important;
    background: rgba(37,99,235,0.04) !important;
}
[data-testid="stFileUploader"] label { color: #3d4f66 !important; font-size: 0.7rem !important; font-weight: 800 !important; text-transform: uppercase !important; letter-spacing: 0.1em !important; }
[data-testid="stFileUploaderDropzoneInstructions"] { color: #4a5e78 !important; }

/* ── Camera ──────────────────────────────────────────────────────────────── */
[data-testid="stCameraInput"] { border: 1px solid rgba(37,99,235,0.2) !important; border-radius: 20px !important; overflow: hidden !important; }
[data-testid="stCameraInput"] label { color: #3d4f66 !important; font-size: 0.7rem !important; font-weight: 800 !important; text-transform: uppercase !important; letter-spacing: 0.1em !important; }

/* ── Buttons ─────────────────────────────────────────────────────────────── */
.stButton > button {
    border-radius: 12px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #e11d48 0%, #be123c 50%, #9f1239 100%) !important;
    border: none !important;
    color: #fff !important;
    padding: 15px 32px !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.01em !important;
    box-shadow: 0 4px 28px rgba(225,29,72,0.35), 0 0 0 1px rgba(225,29,72,0.2),
                inset 0 1px 0 rgba(255,255,255,0.15) !important;
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 8px 36px rgba(225,29,72,0.55), 0 0 0 1px rgba(225,29,72,0.4),
                inset 0 1px 0 rgba(255,255,255,0.15) !important;
    transform: translateY(-1px) !important;
}
.stButton > button[kind="primary"]:active { transform: translateY(0) !important; }
.stButton > button[kind="primary"]:disabled {
    opacity: 0.35 !important;
    box-shadow: none !important;
    transform: none !important;
    cursor: not-allowed !important;
}
.stButton > button[kind="secondary"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.09) !important;
    color: #6b7e99 !important;
    padding: 11px 20px !important;
    font-size: 0.85rem !important;
}
.stButton > button[kind="secondary"]:hover {
    background: rgba(255,255,255,0.08) !important;
    border-color: rgba(255,255,255,0.15) !important;
    color: #c8daff !important;
}

/* ── Info / Warning / Success / Error ─────────────────────────────────────── */
[data-testid="stAlert"] { border-radius: 14px !important; backdrop-filter: blur(12px) !important; }
.stAlert > div { background: transparent !important; }
div[data-baseweb="notification"] {
    border-radius: 14px !important;
    backdrop-filter: blur(12px) !important;
}

/* ── Spinner ─────────────────────────────────────────────────────────────── */
[data-testid="stSpinner"] > div > div { border-top-color: #e11d48 !important; }

/* ── Expander ────────────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    background: rgba(255,255,255,0.025) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 14px !important;
}
[data-testid="stExpander"] summary { color: #4a5e78 !important; font-size: 0.78rem !important; }

/* ── Divider ─────────────────────────────────────────────────────────────── */
hr { border-color: rgba(255,255,255,0.04) !important; margin: 2rem 0 !important; }

/* ── Dataframe ───────────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 16px !important;
    overflow: hidden !important;
    backdrop-filter: blur(12px) !important;
}

/* ─────────────────────────── CUSTOM HTML ELEMENTS ─────────────────────────── */

/* Glass panels */
.g-panel {
    background: rgba(255,255,255,0.035);
    backdrop-filter: blur(28px) saturate(160%);
    -webkit-backdrop-filter: blur(28px) saturate(160%);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 22px;
    padding: 26px 30px;
    margin-bottom: 16px;
    box-shadow:
        inset 0 1px 0 rgba(255,255,255,0.06),
        0 4px 32px rgba(0,0,0,0.3);
}
.g-panel-sm {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 15px;
    padding: 16px 20px;
    margin-bottom: 10px;
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
}

/* Section labels */
.sec-label {
    font-size: 0.67rem;
    font-weight: 900;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #263347;
    margin-bottom: 3px;
}
.sec-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #c8daff;
    margin-bottom: 18px;
    letter-spacing: -0.01em;
}

/* Key-value rows */
.kv-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 8px 0;
    border-bottom: 1px solid rgba(255,255,255,0.035);
    font-size: 0.82rem;
}
.kv-row:last-child { border-bottom: none; }
.kv-row .kk { color: #3d4f66; font-weight: 500; }
.kv-row .vv { color: #b8cfff; font-weight: 700; font-family: 'SF Mono', ui-monospace, monospace; font-size: 0.78rem; }

/* Hero */
.hero-wrap { padding: 6px 0 28px; }
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    background: rgba(225,29,72,0.12);
    border: 1px solid rgba(225,29,72,0.22);
    border-radius: 99px;
    padding: 4px 14px;
    font-size: 0.7rem;
    font-weight: 800;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #fca5a5;
    margin-bottom: 14px;
}
.hero-title {
    font-size: 2.3rem;
    font-weight: 900;
    letter-spacing: -0.035em;
    background: linear-gradient(135deg, #f0f8ff 0%, #a8c4e8 60%, #7ba3d4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 6px;
    line-height: 1.1;
}
.hero-sub {
    font-size: 0.88rem;
    color: #2e3f56;
    line-height: 1.5;
    max-width: 600px;
}

/* Session chip */
.session-chip {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(37,99,235,0.1);
    border: 1px solid rgba(37,99,235,0.2);
    border-radius: 99px;
    padding: 5px 14px;
    font-size: 0.72rem;
    font-weight: 700;
    color: #7cb2ff;
    font-family: ui-monospace, monospace;
    margin-bottom: 20px;
    letter-spacing: 0.02em;
}
.session-chip span:first-child { color: #2d4a7a; }

/* Risk banners */
.risk-box {
    border-radius: 16px;
    padding: 16px 20px;
    margin-bottom: 14px;
    line-height: 1.55;
}
.risk-emergency { background: rgba(220,38,38,0.11); border: 1px solid rgba(220,38,38,0.24); color: #fca5a5; }
.risk-urgent    { background: rgba(245,158,11,0.09); border: 1px solid rgba(245,158,11,0.22); color: #fcd34d; }
.risk-routine   { background: rgba(16,185,129,0.08); border: 1px solid rgba(16,185,129,0.2);  color: #6ee7b7; }
.risk-info      { background: rgba(100,116,139,0.1); border: 1px solid rgba(100,116,139,0.2); color: #8eacc7; }
.risk-title { font-size: 0.72rem; font-weight: 900; letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 7px; }
.risk-warn  { font-size: 0.86rem; font-weight: 500; opacity: 0.82; margin-bottom: 5px; }
.risk-act   { font-size: 0.9rem; font-weight: 400; }

/* Red flag warning */
.flag-warn {
    background: rgba(220,38,38,0.1);
    border: 1px solid rgba(220,38,38,0.22);
    border-radius: 14px;
    padding: 14px 18px;
    color: #fca5a5;
    font-size: 0.88rem;
    font-weight: 600;
    margin-top: 10px;
    line-height: 1.5;
}

/* Steps */
.step-item {
    display: flex;
    gap: 14px;
    align-items: flex-start;
    padding: 12px 0;
    border-bottom: 1px solid rgba(255,255,255,0.04);
    color: #8aaccc;
    font-size: 0.88rem;
    line-height: 1.55;
}
.step-item:last-child { border-bottom: none; }
.step-num {
    flex: 0 0 26px;
    height: 26px;
    background: rgba(220,38,38,0.14);
    border: 1px solid rgba(220,38,38,0.28);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.67rem;
    font-weight: 900;
    color: #fca5a5;
    margin-top: 2px;
}

/* Explanation box */
.explain-box {
    background: rgba(59,130,246,0.07);
    border: 1px solid rgba(59,130,246,0.18);
    border-radius: 14px;
    padding: 16px 20px;
    color: #93c5fd;
    font-size: 0.88rem;
    line-height: 1.65;
    margin-bottom: 14px;
}

/* Medicine cards */
.med-card {
    background: rgba(16,185,129,0.06);
    border: 1px solid rgba(16,185,129,0.16);
    border-radius: 12px;
    padding: 13px 16px;
    margin-bottom: 8px;
}
.med-name    { font-size: 0.90rem; font-weight: 700; color: #6ee7b7; margin-bottom: 4px; }
.med-purpose { font-size: 0.82rem; color: #a7f3d0; margin-bottom: 3px; }
.med-dose    { font-size: 0.80rem; color: #9ca3af; margin-bottom: 3px; }
.med-warn    { font-size: 0.78rem; color: #fcd34d; font-style: italic; }

/* When to seek medical care */
.seek-box {
    background: rgba(245,158,11,0.07);
    border: 1px solid rgba(245,158,11,0.18);
    border-radius: 14px;
    padding: 14px 18px;
    margin-top: 4px;
}
.seek-item {
    display: flex;
    gap: 10px;
    align-items: flex-start;
    padding: 5px 0;
    color: #fcd34d;
    font-size: 0.84rem;
    line-height: 1.5;
}
.seek-dot {
    flex: 0 0 8px;
    height: 8px;
    border-radius: 50%;
    background: #f59e0b;
    margin-top: 5px;
}

/* Evidence cards */
.ev-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 13px;
    padding: 13px 18px;
    margin-bottom: 9px;
}
.ev-title { font-size: 0.87rem; font-weight: 600; color: #c0d5ff; margin-bottom: 5px; }
.ev-meta  { font-size: 0.72rem; color: #3d4f66; display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
.ev-score {
    background: rgba(16,185,129,0.12);
    border: 1px solid rgba(16,185,129,0.22);
    border-radius: 99px;
    padding: 2px 10px;
    font-size: 0.68rem;
    font-weight: 800;
    color: #6ee7b7;
    letter-spacing: 0.04em;
}

/* Empty state */
.empty-state {
    text-align: center;
    padding: 60px 32px;
    color: #263347;
}
.empty-icon { font-size: 3rem; opacity: 0.35; margin-bottom: 16px; }
.empty-text { font-size: 0.88rem; color: #2e3f56; line-height: 1.65; }

/* Storage note */
.store-note {
    font-size: 0.7rem;
    color: #1e2d3d;
    padding: 8px 12px;
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.04);
    border-radius: 9px;
    margin-top: 12px;
}

/* Sidebar logo area */
.sb-brand {
    padding: 0 0 16px;
    border-bottom: 1px solid rgba(255,255,255,0.04);
    margin-bottom: 20px;
}
.sb-name { font-size: 1.05rem; font-weight: 800; color: #c8daff; letter-spacing: -0.01em; }
.sb-ver  { font-size: 0.68rem; font-weight: 700; color: #1e2d3d; text-transform: uppercase; letter-spacing: 0.1em; }

/* Sidebar section head */
.sb-head { font-size: 0.65rem; font-weight: 900; letter-spacing: 0.15em; text-transform: uppercase; color: #1e2d3d; margin: 18px 0 8px; }

/* Disclaimer */
.disclaimer {
    font-size: 0.71rem;
    color: #1a2535;
    line-height: 1.7;
    padding: 14px;
    background: rgba(255,255,255,0.015);
    border: 1px solid rgba(255,255,255,0.035);
    border-radius: 12px;
    margin-top: 16px;
}

/* Success processing */
.proc-success {
    display: flex;
    align-items: center;
    gap: 10px;
    background: rgba(16,185,129,0.09);
    border: 1px solid rgba(16,185,129,0.2);
    border-radius: 12px;
    padding: 12px 16px;
    font-size: 0.85rem;
    font-weight: 600;
    color: #6ee7b7;
    margin-bottom: 18px;
}
</style>
""", unsafe_allow_html=True)


# ── Service init ─────────────────────────────────────────────────────────────
@st.cache_resource
def get_service():
    import os, logging
    settings = Settings.from_env()
    # Log startup config so PowerShell shows what was loaded
    logging.getLogger(__name__).info(
        "SAFE-AI startup: mode=%s  confidence=%.2f  api_key_set=%s",
        settings.model_mode,
        settings.confidence_threshold,
        bool(os.getenv("ROBOFLOW_API_KEY")),
    )
    svc = build_service(settings)
    logging.getLogger(__name__).info(
        "Detector: %s", svc.pipeline.detector.name
    )
    return settings, svc


try:
    settings, service = get_service()
    _service_error: str | None = None
except Exception as _exc:
    service, settings = None, None
    _service_error = str(_exc)


# ── Hero header ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
  <div class="hero-badge">🩹 &nbsp;Medical AI Assistant</div>
  <div class="hero-title">SAFE-AI First Aid</div>
  <div class="hero-sub">
    Preliminary visual injury screening powered by computer vision &amp; RAG guidance.
    &nbsp;·&nbsp; <strong style="color:#1e2d3d;">Not a medical diagnosis tool.</strong>
    &nbsp;For emergencies, call local emergency services.
  </div>
</div>
""", unsafe_allow_html=True)

if _service_error:
    st.error(f"⚠️ Service failed to initialize — {_service_error}")
    st.stop()

# ── Config health check: warn if key env vars are missing ────────────────────
import os as _os
if settings is not None:
    _mode = settings.model_mode
    _api_key = _os.getenv("ROBOFLOW_API_KEY", "").strip()
    if _mode in ("workflow", "roboflow") and not _api_key:
        st.warning(
            "⚠️ **ROBOFLOW_API_KEY is not set.**  "
            "Open the `.env` file in your project folder and set:\n\n"
            "```\nROBOFLOW_API_KEY=your_actual_key_here\n```\n\n"
            "Get your free API key at **app.roboflow.com → Settings → API Keys**, "
            "then restart the app with `python run_streamlit.py`."
        )
    elif _mode in ("null",) or (service is not None and
                                 getattr(service.pipeline.detector, "name", "") == "null"):
        st.info(
            "ℹ️ **Detector is not configured.**  "
            "Edit `.env` in your project folder and set `FIRST_AID_MODEL_MODE=workflow` "
            "and `ROBOFLOW_API_KEY=your_key_here`, then restart."
        )

# ── Session state ──────────────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = service.new_session()
# Bug fix: persist last result across reruns so it doesn't vanish on interaction
if "last_result" not in st.session_state:
    st.session_state.last_result = None

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    detector_name = getattr(service.pipeline.detector, "name", "unknown")

    st.markdown(f"""
    <div class="sb-brand">
      <div class="sb-name">🩹 SAFE-AI</div>
      <div class="sb-ver">First Aid System · v1.1</div>
    </div>

    <div class="sb-head">System</div>
    <div class="g-panel" style="padding:16px 20px;">
      <div class="kv-row"><span class="kk">Detector</span><span class="vv">{detector_name}</span></div>
      <div class="kv-row"><span class="kk">Mode</span><span class="vv">{settings.model_mode}</span></div>
      <div class="kv-row"><span class="kk">Storage</span><span class="vv">SQLite3</span></div>
      <div class="kv-row"><span class="kk">Session</span><span class="vv" style="font-size:0.63rem;">{st.session_state.session_id[:14]}…</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-head">Storage</div>', unsafe_allow_html=True)
    try:
        _stats = service.storage_stats()
        st.markdown(f"""
        <div class="g-panel" style="padding:16px 20px;">
          <div class="kv-row"><span class="kk">Files</span><span class="vv">{_stats["files"]}</span></div>
          <div class="kv-row"><span class="kk">Compressed</span><span class="vv">{_stats["compressed_bytes"] / 1048576:.2f} MB</span></div>
          <div class="kv-row"><span class="kk">Original</span><span class="vv">{_stats["original_bytes"] / 1048576:.2f} MB</span></div>
          <div class="kv-row"><span class="kk">Ratio</span><span class="vv">{_stats.get("compression_ratio", 0):.1%}</span></div>
        </div>
        """, unsafe_allow_html=True)
    except Exception:
        st.caption("Stats unavailable")

    if st.button("🧹 Run Cleanup", use_container_width=True, type="secondary"):
        try:
            _n = service.cleanup()
            st.success(f"Removed {_n} expired files")
        except Exception as _e:
            st.error(str(_e))

    if st.button("🔄 New Session", use_container_width=True, type="secondary"):
        st.session_state.session_id = service.new_session()
        st.session_state.last_result = None
        st.rerun()

    st.markdown("""
    <div class="disclaimer">
      <strong style="color:#2e3f56;">⚠ Disclaimer</strong><br>
      This tool provides <em>preliminary visual screening only</em>.
      It does not diagnose, prescribe, or replace professional medical care.
      Always consult a qualified medical professional for any injury.
    </div>
    """, unsafe_allow_html=True)


# ── Session chip ───────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="session-chip">
  <span>SESSION</span>
  <span>{st.session_state.session_id}</span>
</div>
""", unsafe_allow_html=True)


# ── Two-column main layout ─────────────────────────────────────────────────────
left_col, right_col = st.columns([11, 9], gap="large")

# ════════════════════════ LEFT COLUMN — INPUTS ════════════════════════════════
with left_col:

    # ── Upload ─────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="sec-label">Step 01</div>
    <div class="sec-title">📷 Upload Injury Media</div>
    """, unsafe_allow_html=True)

    input_mode = st.radio(
        "Input method",
        ["📁  File Upload", "📸  Camera Capture"],
        horizontal=True,
        label_visibility="collapsed",
    )

    uploaded = None
    if input_mode == "📁  File Upload":
        uploaded = st.file_uploader(
            "CHOOSE IMAGE OR VIDEO",
            type=["jpg", "jpeg", "png", "webp", "bmp", "gif", "tiff",
                  "mp4", "mov", "avi", "webm", "mkv"],
            help="Supported: JPG, PNG, WEBP, BMP, GIF, TIFF, MP4, MOV, AVI, WEBM, MKV",
        )
    else:
        uploaded = st.camera_input("CAPTURE INJURY IMAGE")

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    # ── Patient context ────────────────────────────────────────────────────────
    st.markdown("""
    <div class="sec-label" style="margin-top:22px;">Step 02</div>
    <div class="sec-title">📋 Patient Context</div>
    """, unsafe_allow_html=True)

    ctx_c1, ctx_c2 = st.columns(2)
    with ctx_c1:
        mechanism = st.selectbox(
            "Injury Mechanism",
            ["unknown", "fall", "cut/contact", "heat", "impact", "other"],
        )
        pain = st.selectbox(
            "Pain Level",
            ["unknown", "mild", "moderate", "severe"],
        )
        # Bug fix: time_since_injury was in DB schema & API but missing from UI
        time_since = st.selectbox(
            "Time Since Injury",
            ["unknown", "<30 min", "30 min – 2 hrs", ">2 hrs", ">24 hrs"],
        )
    with ctx_c2:
        bleeding = st.selectbox(
            "Bleeding Status",
            ["unknown", "none", "minor", "persistent/heavy"],
        )
        movement = st.selectbox(
            "Movement Limitation",
            ["unknown", "none", "some", "major"],
        )

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    flags = st.multiselect(
        "🚩  Red Flags",
        options=[
            "heavy_bleeding", "continuous_bleeding", "breathing_difficulty",
            "unconscious", "confusion", "major_burn", "chemical_burn",
            "electrical_burn", "serious_eye_injury",
        ],
        format_func=lambda x: x.replace("_", " ").title(),
        help="Select any reported danger signs",
    )

    if flags:
        st.markdown("""
        <div class="flag-warn">
          🚨 <strong>Red flag(s) reported.</strong> This system will route to emergency guidance.
          Do not delay — seek professional medical help immediately.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # ── Analyze button ─────────────────────────────────────────────────────────
    analyze_clicked = st.button(
        "🔬  Analyze Injury",
        type="primary",
        disabled=uploaded is None,
        use_container_width=True,
    )
    if uploaded is None:
        st.markdown(
            "<div style='text-align:center;font-size:0.72rem;color:#1e2d3d;margin-top:8px;'>Upload an image or video to enable analysis</div>",
            unsafe_allow_html=True,
        )


# ════════════════════════ RIGHT COLUMN — RESULTS ══════════════════════════════
with right_col:
    st.markdown("""
    <div class="sec-label">Output</div>
    <div class="sec-title">📊 Analysis Results</div>
    """, unsafe_allow_html=True)

    # ── Run analysis ───────────────────────────────────────────────────────────
    if analyze_clicked and uploaded is not None:
        # Bug fix: detect MIME from extension when browser doesn't provide it
        mime = (uploaded.type or "").strip()
        if not mime or mime == "application/octet-stream":
            guessed, _ = mimetypes.guess_type(getattr(uploaded, "name", "file.jpg") or "file.jpg")
            mime = guessed or "image/jpeg"

        ctx = {
            "mechanism": mechanism,
            "bleeding_status": bleeding,
            "pain_level": pain,
            "movement_limitation": movement,
            # Bug fix: wire up time_since_injury
            "time_since_injury": None if time_since == "unknown" else time_since,
            "red_flags": flags,
        }

        with st.spinner("Running AI pipeline…"):
            try:
                _result = service.process_upload(
                    uploaded.getvalue(),
                    mime,
                    st.session_state.session_id,
                    ctx,
                )
                st.session_state.last_result = _result
            except Exception as _exc:
                st.error(f"Processing failed: {_exc}")
                st.session_state.last_result = None

    # ── Render result (from session state so it survives reruns) ──────────────
    result = st.session_state.last_result

    if result is None:
        st.markdown("""
        <div class="g-panel">
          <div class="empty-state">
            <div class="empty-icon">🩺</div>
            <div class="empty-text">
              Upload an injury image and complete the patient context,<br>
              then click <strong style="color:#3d5a8a;">Analyze Injury</strong> to see results here.
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        a, s, g = result["analysis"], result["safety"], result["guidance"]

        # ── Success notice ─────────────────────────────────────────────────────
        st.markdown("""
        <div class="proc-success">
          ✅ &nbsp;Processing complete — review findings below
        </div>
        """, unsafe_allow_html=True)

        # ── Metrics ────────────────────────────────────────────────────────────
        # Bug fix: use distinct variable names (res_c*) to avoid shadowing ctx_c1/ctx_c2
        res_c1, res_c2, res_c3, res_c4 = st.columns(4)
        labels_str = ", ".join(a["labels"]) if a["labels"] else "Unknown"
        res_c1.metric("Detected", labels_str)
        res_c2.metric("Confidence", f"{a['confidence'] * 100:.1f}%")
        res_c3.metric("Severity", a["severity"].replace("_", " ").title())
        res_c4.metric("Status", a["status"])

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

        # ── Risk banner ────────────────────────────────────────────────────────
        risk = s["risk_level"]
        _risk_cls = {
            "emergency_or_urgent": ("risk-emergency", "🚨"),
            "urgent":              ("risk-urgent",    "⚠️"),
            "routine_first_aid":   ("risk-routine",   "✅"),
            "insufficient_evidence": ("risk-info",    "🔍"),
        }.get(risk, ("risk-info", "ℹ️"))
        _rcls, _ric = _risk_cls

        st.markdown(f"""
        <div class="risk-box {_rcls}">
          <div class="risk-title">{_ric} &nbsp;{risk.replace("_", " ")}</div>
          <div class="risk-warn">{s['warning']}</div>
          <div class="risk-act">{s['action']}</div>
        </div>
        """, unsafe_allow_html=True)

        # ── What is this injury? ───────────────────────────────────────────────
        if g.get("explanation"):
            st.markdown("""
            <div class="sec-label" style="margin-bottom:8px;">🩺 What Is This Injury?</div>
            """, unsafe_allow_html=True)
            st.markdown(f'<div class="explain-box">{g["explanation"]}</div>',
                        unsafe_allow_html=True)

        # ── Medicines ──────────────────────────────────────────────────────────
        if g.get("medicines"):
            st.markdown("""
            <div class="sec-label" style="margin-bottom:8px;margin-top:14px;">💊 First Aid Medicines</div>
            """, unsafe_allow_html=True)
            meds_html = ""
            for med in g["medicines"]:
                meds_html += f"""
                <div class="med-card">
                  <div class="med-name">💊 {med['name']}</div>
                  <div class="med-purpose">Purpose: {med['purpose']}</div>
                  <div class="med-dose">📋 Dose: {med['dose_note']}</div>
                  <div class="med-warn">⚠️ {med['warning']}</div>
                </div>"""
            st.markdown(f'<div style="margin-bottom:4px;">{meds_html}</div>',
                        unsafe_allow_html=True)
            st.markdown("""
            <div style="font-size:0.75rem;color:#6b7280;margin-bottom:12px;padding:0 4px;">
              ⚠️ These are general OTC first-aid medicines. Always follow the package instructions
              and consult a pharmacist or doctor before use, especially for children, pregnant women,
              or people with medical conditions.
            </div>""", unsafe_allow_html=True)

        # ── Step-by-step instructions ──────────────────────────────────────────
        st.markdown("""
        <div class="sec-label" style="margin-bottom:8px;margin-top:14px;">📋 Step-by-Step Instructions</div>
        """, unsafe_allow_html=True)
        steps_html = "".join(
            f'<div class="step-item"><div class="step-num">{i+1}</div><div>{step}</div></div>'
            for i, step in enumerate(g["steps"])
        )
        st.markdown(f'<div class="g-panel" style="padding:20px 24px;">{steps_html}</div>',
                    unsafe_allow_html=True)

        # ── When to seek medical care ──────────────────────────────────────────
        if g.get("when_to_seek"):
            st.markdown("""
            <div class="sec-label" style="margin-bottom:8px;margin-top:14px;">🚨 When to Seek Medical Care</div>
            """, unsafe_allow_html=True)
            seek_items = "".join(
                f'<div class="seek-item"><div class="seek-dot"></div><div>{item}</div></div>'
                for item in g["when_to_seek"]
            )
            st.markdown(f'<div class="seek-box">{seek_items}</div>',
                        unsafe_allow_html=True)

        # ── Evidence ───────────────────────────────────────────────────────────
        if g.get("evidence"):
            st.markdown("""
            <div class="sec-label" style="margin-bottom:8px;margin-top:14px;">📚 Evidence Sources (RAG)</div>
            """, unsafe_allow_html=True)
            for _ev in g["evidence"]:
                st.markdown(f"""
                <div class="ev-card">
                  <div class="ev-title">{_ev['title']}</div>
                  <div class="ev-meta">
                    <span class="ev-score">score {_ev['score']:.3f}</span>
                    <span>{_ev['source']}</span>
                  </div>
                </div>
                """, unsafe_allow_html=True)

        # ── Storage note ───────────────────────────────────────────────────────
        _dedup = " · deduplicated" if result.get("deduplicated") else ""
        _ratio = result["compressed_size"] / max(1, result["original_size"]) * 100
        st.markdown(f"""
        <div class="store-note">
          💾 &nbsp;Stored {result['compressed_size']:,} B
          (from {result['original_size']:,} B · {_ratio:.1f}% of original{_dedup})
        </div>
        """, unsafe_allow_html=True)

        # ── Technical details ──────────────────────────────────────────────────
        with st.expander("🔧 Technical Details (raw JSON)"):
            st.json(result)


# ── Session history ────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div class="sec-label">Session</div>
<div class="sec-title" style="margin-bottom:14px;">🕓 Analysis History</div>
""", unsafe_allow_html=True)

try:
    _history = service.history(st.session_state.session_id, 10)
    if _history:
        _keep = ["created_at", "injury_labels", "confidence", "severity",
                 "analysis_status", "media_type", "compressed_size"]
        _display = [{k: v for k, v in row.items() if k in _keep} for row in _history]
        st.dataframe(_display, use_container_width=True)
    else:
        st.markdown("""
        <div style="text-align:center;padding:24px;color:#1a2535;font-size:0.85rem;">
          No analyses recorded in this session yet.
        </div>
        """, unsafe_allow_html=True)
except Exception as _exc:
    st.caption(f"History unavailable: {_exc}")
