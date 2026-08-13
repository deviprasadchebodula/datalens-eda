import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go

try:
    from google import genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False


# =========================================================
# GEMINI AI CLIENT (free tier - key read from st.secrets, never hardcoded)
# =========================================================

def get_gemini_client(user_provided_key=None):
    """Returns a configured Gemini client. Priority: user-provided key (this
    session only) > developer key from st.secrets. Returns None if neither exists
    or if no secrets.toml file is present at all (safe, never crashes)."""
    if not GENAI_AVAILABLE:
        return None

    api_key = user_provided_key
    if not api_key:
        try:
            api_key = st.secrets.get("GEMINI_API_KEY", None)
        except Exception:
            # No secrets.toml file exists yet, or it's malformed - treat as no key
            api_key = None

    if not api_key:
        return None
    try:
        return genai.Client(api_key=api_key)
    except Exception:
        return None


def build_dataset_context(df, max_cols=25):
    """Builds a compact text summary of the dataframe to send to the AI,
    instead of sending the raw dataset (cheaper, faster, more private)."""
    lines = []
    lines.append(f"Rows: {df.shape[0]:,}, Columns: {df.shape[1]}")
    lines.append(f"Column names and types: {dict(df.dtypes.astype(str))}")
    lines.append(f"Missing values per column: {dict(df.isnull().sum())}")
    lines.append(f"Duplicate rows: {int(df.duplicated().sum())}")

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if numeric_cols:
        desc = df[numeric_cols[:max_cols]].describe().round(2).to_dict()
        lines.append(f"Numeric column statistics: {desc}")
        if len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr(numeric_only=True).round(2).to_dict()
            lines.append(f"Correlation matrix: {corr}")

    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        top_vals = {c: df[c].value_counts().head(3).to_dict() for c in cat_cols[:max_cols]}
        lines.append(f"Top categorical values: {top_vals}")

    return "\n".join(lines)


def ask_gemini(client, dataset_context, question, model="gemini-3.6-flash"):
    """Sends a question + dataset context to Gemini and returns the text reply.
    Uses the Interactions API (current recommended API as of 2026)."""
    prompt = (
        "You are a data analyst assistant embedded in an EDA tool called DataLens. "
        "You are given a statistical summary of a user's uploaded CSV dataset (not the raw data). "
        "Answer the user's question clearly and concisely using only this summary. "
        "If something can't be determined from the summary, say so honestly.\n\n"
        f"DATASET SUMMARY:\n{dataset_context}\n\n"
        f"QUESTION:\n{question}"
    )
    interaction = client.interactions.create(model=model, input=prompt)
    return interaction.output_text


# =========================================================
# PLOTLY THEME (light theme, blue accent - matches mockup)
# =========================================================

ACCENT = "#2f6fed"

_datalens_template = go.layout.Template()
_datalens_template.layout = go.Layout(
    font=dict(family="Inter, -apple-system, sans-serif", color="#3d4759", size=13),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    colorway=["#2f6fed", "#7c5cff", "#16a394", "#e8a33d", "#ef5c7c", "#38b6ff"],
    title=dict(font=dict(size=16, color="#1a2233"), x=0.02, xanchor="left"),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(20,30,50,.08)", borderwidth=1),
    xaxis=dict(gridcolor="rgba(20,30,50,.07)", zerolinecolor="rgba(20,30,50,.12)", linecolor="rgba(20,30,50,.12)"),
    yaxis=dict(gridcolor="rgba(20,30,50,.07)", zerolinecolor="rgba(20,30,50,.12)", linecolor="rgba(20,30,50,.12)"),
    margin=dict(t=52, l=10, r=10, b=10),
)

pio.templates["datalens"] = _datalens_template
pio.templates.default = "plotly_white+datalens"
px.defaults.template = "plotly_white+datalens"
px.defaults.color_continuous_scale = "Blues"


# =========================================================
# INLINE SVG ICONS (no external font/CDN dependency - always renders)
# =========================================================

_ICON_PATHS = {
    "home": '<path d="M4 12 12 4l8 8"/><path d="M6 10v10h12V10"/>',
    "search": '<circle cx="11" cy="11" r="7"/><path d="m21 21-4.3-4.3"/>',
    "bar-chart": '<path d="M4 20V10"/><path d="M12 20V4"/><path d="M20 20v-6"/>',
    "shield": '<path d="M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6z"/>',
    "sparkles": '<path d="M12 3v4M12 17v4M3 12h4M17 12h4M6 6l2 2M16 16l2 2M6 18l2-2M16 8l2-2"/>',
    "download": '<path d="M12 3v12"/><path d="m7 11 5 5 5-5"/><path d="M5 21h14"/>',
    "folder": '<path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>',
    "database": '<ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v14c0 1.7 3.6 3 8 3s8-1.3 8-3V5"/><path d="M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3"/>',
    "columns": '<rect x="3" y="4" width="7" height="16" rx="1"/><rect x="14" y="4" width="7" height="16" rx="1"/>',
    "alert": '<path d="M10.3 3.9 2 19h20L13.7 3.9a2 2 0 0 0-3.4 0z"/><path d="M12 9v4"/><path d="M12 17h.01"/>',
    "copy": '<rect x="9" y="9" width="12" height="12" rx="2"/><path d="M5 15V5a2 2 0 0 1 2-2h10"/>',
    "check": '<path d="M20 6 9 17l-5-5"/>',
    "bulb": '<path d="M9 18h6"/><path d="M10 22h4"/><path d="M12 2a6 6 0 0 0-4 10.5c.5.5 1 1.3 1 2.5h6c0-1.2.5-2 1-2.5A6 6 0 0 0 12 2z"/>',
    "refresh": '<path d="M21 12a9 9 0 1 1-3-6.7"/><path d="M21 3v6h-6"/>',
    "file": '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6"/>',
    "arrow-left": '<path d="M19 12H5"/><path d="m12 19-7-7 7-7"/>',
    "clean": '<path d="M3 21 15 9"/><path d="m17 3 4 4-3.5 3.5-4-4z"/><path d="m14 6 4 4"/>',
    "trend": '<path d="M3 17 9 11l4 4 8-8"/><path d="M17 7h4v4"/>',
}


def icon(name, size=15, color="currentColor", stroke=2):
    d = _ICON_PATHS.get(name, "")
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke}" '
        f'stroke-linecap="round" stroke-linejoin="round" '
        f'style="vertical-align:-3px;display:inline-block">{d}</svg>'
    )


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="DataLens | EDA Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    '<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/tabler-icons/2.44.0/tabler-icons.min.css">',
    unsafe_allow_html=True,
)

st.markdown("""
<style>
.block-container{max-width:1500px;padding-top:2rem;padding-bottom:4rem}
[data-testid="stAppViewContainer"]{background:#f7f8fb}
[data-testid="stHeader"]{background:rgba(247,248,251,0)}
[data-testid="stSidebar"]{background:#ffffff;border-right:1px solid #e9ebf1}
[data-testid="stSidebar"] .block-container{padding-top:1.5rem;padding-left:1.15rem;padding-right:1.15rem}
[data-testid="stSidebar"] .stCaption{color:#8a93a6}
[data-testid="stRadio"] label{border-radius:10px;padding:.55rem .65rem;transition:all .15s ease}
[data-testid="stRadio"] label:hover{background:#f0f4ff}
[data-testid="stRadio"] label[data-checked="true"]{background:#e8eefd!important;border:1px solid #cfdcfb}
[data-testid="stRadio"] label p{font-weight:600!important;color:#1a2233}
[data-testid="stRadio"] > div{gap:.15rem}
[data-testid="stFileUploader"]{border:1.5px dashed #2f6fed55;border-radius:14px;padding:.35rem;background:#f5f8ff}
[data-testid="stMetric"]{background:#ffffff;border:1px solid #e9ebf1;border-radius:14px;padding:1rem 1.1rem;box-shadow:0 2px 8px rgba(20,30,50,.06);position:relative;overflow:hidden}
[data-testid="stMetric"]::before{content:"";position:absolute;top:0;left:0;width:3px;height:100%;background:linear-gradient(180deg,#2f6fed,#7c5cff)}
[data-testid="stMetricLabel"]{color:#5c6577}
[data-testid="stMetricValue"]{font-size:1.85rem;font-weight:750;color:#12182b}
[data-testid="stMetricDelta"]{font-weight:600}
.stButton>button,.stDownloadButton>button{border-radius:10px;min-height:42px;font-weight:650;border:1px solid #dbe2f5;background:#ffffff;color:#1a2233;transition:all .15s ease}
.stButton>button:hover,.stDownloadButton>button:hover{border-color:#2f6fed;box-shadow:0 4px 14px rgba(47,111,237,.14)}
h1{font-size:2.3rem!important;font-weight:780!important;letter-spacing:-1px;color:#12182b}
h2{font-weight:730!important;color:#12182b;font-size:1.5rem!important} h3{font-weight:680!important;color:#12182b;font-size:1.2rem!important}
p, span, div, label{color:#3d4759;font-size:1.02rem}
[data-testid="stMarkdownContainer"] p{font-size:1.02rem}
[data-testid="stDataFrame"]{border:1px solid #e9ebf1;border-radius:14px;overflow:hidden}
[data-testid="stExpander"]{border:1px solid #e9ebf1;border-radius:14px;background:#fff}
[data-baseweb="select"]>div{border-radius:10px;border-color:#dbe2f5!important}
hr{border-color:#e9ebf1;margin:1.35rem 0}
.dl-hero{padding:2.2rem 2.4rem;border-radius:20px;border:1px solid #e3e9fb;background:linear-gradient(135deg,#eaf0ff,#f7f9ff);margin-bottom:1.5rem}
.dl-kicker{color:#2f6fed;font-size:.82rem;font-weight:750;letter-spacing:1.4px;text-transform:uppercase;margin-bottom:.55rem}
.dl-hero-title{font-size:2.6rem;font-weight:800;letter-spacing:-1.5px;margin:0;color:#12182b}
.dl-hero-text{color:#5c6577;font-size:1.05rem;max-width:720px;margin-top:.65rem}
.dl-feature{min-height:150px;padding:1.15rem;border-radius:16px;border:1px solid #e9ebf1;background:#ffffff;box-shadow:0 2px 8px rgba(20,30,50,.05)}
.dl-feature-icon{margin-bottom:.4rem}
.dl-feature-title{font-size:1.1rem;font-weight:720;margin-bottom:.3rem;color:#12182b}
.dl-feature-text{color:#5c6577;font-size:.98rem;line-height:1.55}
.dl-status{display:inline-block;padding:.35rem .7rem;border-radius:999px;background:#e0e9ff;color:#2f6fed;border:1px solid #cfdcfb;font-size:.8rem;font-weight:650}
.dl-card{background:#ffffff;border:1px solid #e9ebf1;border-radius:14px;padding:1rem 1.1rem;box-shadow:0 2px 8px rgba(20,30,50,.05)}
.dl-stat-card{background:#ffffff;border:1px solid #e9ebf1;border-radius:14px;padding:.95rem 1.05rem;box-shadow:0 2px 8px rgba(20,30,50,.05)}
.dl-stat-label{display:flex;align-items:center;gap:6px;color:#5c6577;font-size:.92rem;margin-bottom:8px}
.dl-stat-value{font-size:1.85rem;font-weight:750;color:#12182b;margin-bottom:2px}
.dl-stat-delta{font-size:.78rem;font-weight:600}
.dl-stat-delta.up{color:#1a9e6f}
.dl-stat-delta.down{color:#1a9e6f}
.dl-stat-delta.warn{color:#c17d1b}
.dl-stat-delta.flat{color:#8a93a6}
.dl-insight-card{background:#ffffff;border:1px solid #cfdcfb;border-radius:14px;padding:1.1rem 1.2rem;box-shadow:0 2px 10px rgba(37,99,235,.06)}
.dl-insight-row{display:flex;align-items:flex-start;gap:8px;margin-bottom:8px;font-size:.88rem;color:#3d4759}
.dl-insight-badge{background:#e6efff;color:#2f6fed;padding:1px 6px;border-radius:5px;font-weight:650;font-size:.85rem}
.dl-legend-row{display:flex;align-items:center;justify-content:space-between;font-size:.82rem;margin-bottom:5px;color:#3d4759}
.dl-legend-bar{height:5px;background:#eef1f7;border-radius:3px;overflow:hidden;margin-bottom:10px}
.dl-legend-fill{height:100%;border-radius:3px}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] img{vertical-align:-3px;margin-right:2px}
.stAlert{border-radius:12px!important}
</style>
""", unsafe_allow_html=True)


# =========================================================
# SESSION STATE
# =========================================================

if "df" not in st.session_state:
    st.session_state.df = None

if "cleaned_df" not in st.session_state:
    st.session_state.cleaned_df = None

if "cleaning_summary" not in st.session_state:
    st.session_state.cleaning_summary = None

if "file_id" not in st.session_state:
    st.session_state.file_id = None

if "detected_separator" not in st.session_state:
    st.session_state.detected_separator = None

if "separator_name" not in st.session_state:
    st.session_state.separator_name = None

if "detected_encoding" not in st.session_state:
    st.session_state.detected_encoding = None


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        f'<div style="display:flex;align-items:center;gap:8px">'
        f'{icon("bar-chart", 20, "#2f6fed")}'
        f'<span style="font-size:1.3rem;font-weight:800;letter-spacing:-.5px;">DataLens</span>'
        f'</div>',
        unsafe_allow_html=True
    )
    st.caption("Exploratory data analysis")
    st.divider()

    uploaded_file = st.file_uploader(
        "Drop CSV here",
        type=["csv"],
        label_visibility="visible"
    )

    if uploaded_file is not None:
        st.success(f"✓ {uploaded_file.name}")

    st.divider()
    st.markdown("**NAVIGATION**")

    page = st.radio(
        "Navigation",
        [
            "Overview",
            "Data Explorer",
            "Visualizations",
            "Data Quality",
            "Insights",
            "Export"
        ],
        label_visibility="collapsed"
    )

    st.divider()
    st.caption("Python • Pandas • NumPy • Plotly • Streamlit")


# =========================================================
# SMART FILE IMPORT
# =========================================================

def detect_separator(uploaded_file):

    uploaded_file.seek(0)

    sample = uploaded_file.read(100000)

    uploaded_file.seek(0)

    # Detect encoding
    encoding = "utf-8"

    for test_encoding in [
        "utf-8",
        "utf-8-sig",
        "latin1",
        "cp1252"
    ]:

        try:
            text = sample.decode(test_encoding)
            encoding = test_encoding
            break

        except UnicodeDecodeError:
            continue

    # Common separators
    separators = {
        ",": "Comma (,)",
        ";": "Semicolon (;)",
        "\t": "Tab",
        "|": "Pipe (|)",
        ":": "Colon (:)"
    }

    lines = [
        line
        for line in text.splitlines()
        if line.strip()
    ][:20]

    scores = {}

    for separator in separators:

        counts = [
            line.count(separator)
            for line in lines
        ]

        positive = [
            count
            for count in counts
            if count > 0
        ]

        if positive:

            consistency = (
                len(positive) / len(counts)
            )

            average = (
                sum(positive) / len(positive)
            )

            scores[separator] = (
                consistency * average
            )

        else:

            scores[separator] = 0

    if not scores or max(scores.values()) == 0:

        return ",", "Comma (,)", encoding

    best_separator = max(
        scores,
        key=scores.get
    )

    return (
        best_separator,
        separators[best_separator],
        encoding
    )


# =========================================================
# LOAD DATA
# =========================================================

if uploaded_file is not None:

    try:

        file_id = (
            uploaded_file.name,
            uploaded_file.size
        )

        if st.session_state.get("file_id") != file_id:

            (
                detected_separator,
                separator_name,
                detected_encoding
            ) = detect_separator(uploaded_file)

            uploaded_file.seek(0)

            df = pd.read_csv(
                uploaded_file,
                sep=detected_separator,
                encoding=detected_encoding,
                engine="python"
            )

            st.session_state.df = df.copy()

            st.session_state.cleaned_df = df.copy()

            st.session_state.cleaning_summary = None

            st.session_state.file_id = file_id

            st.session_state.detected_separator = (
                detected_separator
            )

            st.session_state.separator_name = (
                separator_name
            )

            st.session_state.detected_encoding = (
                detected_encoding
            )

    except Exception as e:

        st.error(
            f"Unable to read the file: {e}"
        )

        st.stop()


# =========================================================
# DATA
# =========================================================

original_df = st.session_state.df
df = st.session_state.cleaned_df


# =========================================================
# LANDING PAGE
# =========================================================

if df is None:

    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:14px;margin-bottom:4px">
            {icon('bar-chart', 32, '#2f6fed')}
            <h1 style="margin:0;font-size:2.4rem">DataLens</h1>
        </div>
        <p style="color:#5c6577;font-size:1.2rem;margin:0 0 1.8rem">
            Turn raw data into meaningful insights, automatically.
        </p>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)

    features = [
        ("folder", "Upload", "Auto-detect separators and encoding."),
        ("search", "Explore", "Types, distributions and key stats."),
        ("clean", "Clean", "Missing values, duplicates, outliers."),
        ("bulb", "Insights", "Patterns and automated summaries.")
    ]

    for col, (icon_name, title, text) in zip([col1, col2, col3, col4], features):
        with col:
            st.markdown(
                f"""
                <div class="dl-feature">
                    <div class="dl-feature-icon">{icon(icon_name, 24, '#2f6fed')}</div>
                    <div class="dl-feature-title" style="font-size:1.15rem">{title}</div>
                    <div class="dl-feature-text" style="font-size:1rem">{text}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.write("")
    st.markdown(
        f"""
        <div style="background:#dbe8ff;border-radius:12px;padding:1.1rem 1.3rem;
                    display:flex;align-items:center;gap:10px;color:#1a4fc4;font-weight:600;font-size:1.1rem">
            {icon('arrow-left', 18, '#1a4fc4')}
            Upload a CSV from the sidebar to get started
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")
    st.write("")

    # ---- How it works: 3 steps ----
    st.markdown("<h2 style='font-size:1.6rem;margin-bottom:1rem'>How it works</h2>", unsafe_allow_html=True)

    step1, step2, step3 = st.columns(3)

    steps = [
        ("1", "Upload your CSV", "Drop any CSV file in the sidebar. DataLens automatically detects the separator and encoding, even for messy real-world files."),
        ("2", "Explore automatically", "Get instant stats, charts, missing-value reports, and correlation analysis across every column — no manual coding required."),
        ("3", "Ask the AI anything", "Chat with your dataset in plain English. Ask what stands out, what's correlated, or what needs cleaning."),
    ]

    for col, (num, title, text) in zip([step1, step2, step3], steps):
        with col:
            st.markdown(
                f"""
                <div class="dl-feature" style="min-height:170px">
                    <div style="width:34px;height:34px;border-radius:50%;background:#2f6fed;
                                color:white;display:flex;align-items:center;justify-content:center;
                                font-weight:700;font-size:1.1rem;margin-bottom:.6rem">{num}</div>
                    <div class="dl-feature-title" style="font-size:1.15rem">{title}</div>
                    <div class="dl-feature-text" style="font-size:1rem">{text}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.write("")
    st.write("")

    # ---- Visual preview: what the dashboard looks like ----
    st.markdown("<h2 style='font-size:1.6rem;margin-bottom:1rem'>See it in action</h2>", unsafe_allow_html=True)

    st.markdown(
        """<p style="font-size:1rem;color:#5c6577;margin:0 0 1.1rem">
        A preview of the Overview dashboard you'll see once your CSV is loaded:
        </p>""",
        unsafe_allow_html=True
    )

    preview_svg = """
    <svg viewBox="0 0 760 220" style="width:100%;height:auto" role="img" aria-label="Preview of dashboard with a bar chart and a donut chart">
        <rect x="0" y="0" width="170" height="60" rx="8" fill="#ffffff" stroke="#e9ebf1"></rect>
        <text x="14" y="22" font-size="11" fill="#8a93a6" font-family="Inter, sans-serif">Rows</text>
        <text x="14" y="46" font-size="20" font-weight="700" fill="#12182b" font-family="Inter, sans-serif">12,480</text>
        <rect x="182" y="0" width="170" height="60" rx="8" fill="#ffffff" stroke="#e9ebf1"></rect>
        <text x="196" y="22" font-size="11" fill="#8a93a6" font-family="Inter, sans-serif">Columns</text>
        <text x="196" y="46" font-size="20" font-weight="700" fill="#12182b" font-family="Inter, sans-serif">18</text>
        <rect x="364" y="0" width="170" height="60" rx="8" fill="#ffffff" stroke="#e9ebf1"></rect>
        <text x="378" y="22" font-size="11" fill="#8a93a6" font-family="Inter, sans-serif">Missing</text>
        <text x="378" y="46" font-size="20" font-weight="700" fill="#12182b" font-family="Inter, sans-serif">2.3%</text>
        <rect x="546" y="0" width="170" height="60" rx="8" fill="#ffffff" stroke="#e9ebf1"></rect>
        <text x="560" y="22" font-size="11" fill="#8a93a6" font-family="Inter, sans-serif">Duplicates</text>
        <text x="560" y="46" font-size="20" font-weight="700" fill="#12182b" font-family="Inter, sans-serif">0</text>
        <rect x="0" y="76" width="430" height="140" rx="10" fill="#ffffff" stroke="#e9ebf1"></rect>
        <text x="16" y="98" font-size="12" font-weight="600" fill="#12182b" font-family="Inter, sans-serif">Distribution</text>
        <g fill="#2f6fed">
            <rect x="24" y="150" width="24" height="50" rx="3"></rect>
            <rect x="60" y="130" width="24" height="70" rx="3"></rect>
            <rect x="96" y="140" width="24" height="60" rx="3"></rect>
            <rect x="132" y="115" width="24" height="85" rx="3"></rect>
            <rect x="168" y="150" width="24" height="50" rx="3"></rect>
            <rect x="204" y="105" width="24" height="95" rx="3"></rect>
            <rect x="240" y="125" width="24" height="75" rx="3"></rect>
            <rect x="276" y="95" width="24" height="105" rx="3"></rect>
            <rect x="312" y="135" width="24" height="65" rx="3"></rect>
            <rect x="348" y="110" width="24" height="90" rx="3"></rect>
            <rect x="384" y="145" width="24" height="55" rx="3"></rect>
        </g>
        <line x1="16" y1="200" x2="414" y2="200" stroke="#e9ebf1" stroke-width="1"></line>
        <rect x="442" y="76" width="274" height="140" rx="10" fill="#ffffff" stroke="#e9ebf1"></rect>
        <text x="458" y="98" font-size="12" font-weight="600" fill="#12182b" font-family="Inter, sans-serif">Column types</text>
        <circle cx="510" cy="150" r="34" fill="none" stroke="#e9ebf1" stroke-width="16"></circle>
        <circle cx="510" cy="150" r="34" fill="none" stroke="#2f6fed" stroke-width="16" stroke-dasharray="106 214" stroke-dashoffset="0" transform="rotate(-90 510 150)"></circle>
        <circle cx="510" cy="150" r="34" fill="none" stroke="#7c5cff" stroke-width="16" stroke-dasharray="60 214" stroke-dashoffset="-106" transform="rotate(-90 510 150)"></circle>
        <circle cx="510" cy="150" r="34" fill="none" stroke="#16a394" stroke-width="16" stroke-dasharray="48 214" stroke-dashoffset="-166" transform="rotate(-90 510 150)"></circle>
        <g font-family="Inter, sans-serif" font-size="11" fill="#3d4759">
            <circle cx="570" cy="120" r="4" fill="#2f6fed"></circle>
            <text x="580" y="124">Numerical</text>
            <circle cx="570" cy="144" r="4" fill="#7c5cff"></circle>
            <text x="580" y="148">Categorical</text>
            <circle cx="570" cy="168" r="4" fill="#16a394"></circle>
            <text x="580" y="172">Datetime</text>
        </g>
    </svg>
    """

    st.markdown(
        f"""<div style="background:#fafbfe;border:1px solid #e9ebf1;border-radius:12px;padding:1.3rem">{preview_svg}</div>""",
        unsafe_allow_html=True
    )

    st.write("")
    st.write("")

    # ---- AI chatbot showcase ----
    st.markdown(
        f"""
        <div class="dl-feature" style="min-height:auto;padding:1.6rem;display:flex;align-items:center;gap:1.5rem;border:1px solid #cfdcfb;background:linear-gradient(135deg,#f5f8ff,#ffffff)">
            <div style="flex-shrink:0">{icon('sparkles', 40, '#2f6fed')}</div>
            <div>
                <div style="font-size:1.3rem;font-weight:750;color:#12182b;margin-bottom:.35rem">Chat with your data</div>
                <div style="font-size:1.05rem;color:#5c6577;line-height:1.5">
                    Once your dataset is loaded, head to the <b>Insights</b> page and ask questions in plain English —
                    powered by Google Gemini. Get an instant AI-written summary of your dataset, or ask follow-up
                    questions like "what's driving the outliers in column X?"
                </div>
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")
    st.caption("DataLens • Built with Python, Pandas, NumPy, Plotly, Streamlit & Google Gemini")
    st.stop()



# =========================================================
# IMPORT INFORMATION
# =========================================================

if st.session_state.separator_name:

    st.sidebar.divider()

    st.sidebar.subheader(
        "File Information"
    )

    st.sidebar.write(
        f"**File:** {uploaded_file.name if uploaded_file else 'Loaded dataset'}"
    )

    st.sidebar.success(
        f"Separator: "
        f"{st.session_state.separator_name}"
    )

    st.sidebar.caption(
        f"Encoding: "
        f"{st.session_state.detected_encoding}"
    )

    st.sidebar.caption(
        f"Rows: {len(df):,}"
    )

    st.sidebar.caption(
        f"Columns: {df.shape[1]:,}"
    )

    if df.shape[1] == 1:

        st.sidebar.warning(
            "Only 1 column detected. "
            "Check the file separator."
        )


# =========================================================
# COLUMN TYPES
# =========================================================

numeric_columns = df.select_dtypes(
    include=np.number
).columns.tolist()

categorical_columns = df.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()


# =========================================================
# OVERVIEW
# =========================================================

if page == "Overview":

    # ---- Header row: title + re-analyze button ----
    hcol1, hcol2 = st.columns([5, 1.3])
    with hcol1:
        st.markdown(
            f"""<h1 style="margin-bottom:2px">Overview</h1>
            <p style="color:#5c6577;font-size:.95rem;margin:0">
            {uploaded_file.name if uploaded_file else 'Dataset'} &middot; last analyzed just now
            </p>""",
            unsafe_allow_html=True,
        )
    with hcol2:
        st.markdown("<div style='margin-top:14px'></div>", unsafe_allow_html=True)
        st.button("Re-analyze", key="reanalyze_btn", use_container_width=True)

    st.write("")

    # ---- Compute real stats ----
    n_rows, n_cols = df.shape
    n_missing = int(df.isnull().sum().sum())
    missing_pct = round((n_missing / (n_rows * n_cols)) * 100, 1) if n_rows and n_cols else 0
    n_dupes = int(df.duplicated().sum())
    dupe_pct = round((n_dupes / n_rows) * 100, 1) if n_rows else 0

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    datetime_cols = df.select_dtypes(include="datetime").columns.tolist()
    # try to catch date-like object columns too
    for c in df.select_dtypes(include="object").columns:
        if c not in datetime_cols:
            try:
                pd.to_datetime(df[c].dropna().head(20), errors="raise")
                datetime_cols.append(c)
            except Exception:
                pass
    categorical_cols = [
        c for c in df.select_dtypes(include=["object", "category", "bool"]).columns
        if c not in datetime_cols
    ]

    type_counts = [
        ("Numerical", len(numeric_cols), "#2f6fed"),
        ("Categorical", len(categorical_cols), "#7c5cff"),
        ("Datetime", len(datetime_cols), "#16a394"),
    ]
    total_typed = sum(c for _, c, _ in type_counts) or 1

    # ---- Stat cards ----
    s1, s2, s3, s4 = st.columns(4)

    with s1:
        st.markdown(f"""
        <div class="dl-stat-card">
            <div class="dl-stat-label">{icon('database', 14, '#2f6fed')} Rows</div>
            <div class="dl-stat-value">{n_rows:,}</div>
            <div class="dl-stat-delta flat">{n_cols} columns total</div>
        </div>""", unsafe_allow_html=True)

    with s2:
        st.markdown(f"""
        <div class="dl-stat-card">
            <div class="dl-stat-label">{icon('columns', 14, '#2f6fed')} Columns</div>
            <div class="dl-stat-value">{n_cols}</div>
            <div class="dl-stat-delta flat">{len(numeric_cols)} numeric &middot; {len(categorical_cols)} categorical</div>
        </div>""", unsafe_allow_html=True)

    with s3:
        delta_class = "warn" if missing_pct > 0 else "up"
        delta_text = f"{n_missing:,} missing cells" if missing_pct > 0 else "No missing values"
        st.markdown(f"""
        <div class="dl-stat-card">
            <div class="dl-stat-label">{icon('alert', 14, '#c17d1b' if missing_pct > 0 else '#2f6fed')} Missing values</div>
            <div class="dl-stat-value">{missing_pct}%</div>
            <div class="dl-stat-delta {delta_class}">{delta_text}</div>
        </div>""", unsafe_allow_html=True)

    with s4:
        delta_class = "warn" if n_dupes > 0 else "up"
        delta_text = f"{n_dupes:,} rows ({dupe_pct}%)" if n_dupes > 0 else "Clean"
        st.markdown(f"""
        <div class="dl-stat-card">
            <div class="dl-stat-label">{icon('copy', 14, '#c17d1b' if n_dupes > 0 else '#2f6fed')} Duplicates</div>
            <div class="dl-stat-value">{n_dupes:,}</div>
            <div class="dl-stat-delta {delta_class}">{delta_text}</div>
        </div>""", unsafe_allow_html=True)

    st.write("")

    # ---- Chart + column type breakdown ----
    c1, c2 = st.columns([1.4, 1])

    with c1:
        with st.container(border=True):
            if numeric_cols:
                chart_col = numeric_cols[0]
                st.markdown(f"**Distribution &middot; {chart_col}**")
                fig = px.histogram(df, x=chart_col, nbins=30)
                fig.update_traces(marker_color="#2f6fed", marker_line_width=0)
                fig.update_layout(height=260, showlegend=False, yaxis_title="Count", xaxis_title=chart_col)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            else:
                st.markdown("**Distribution**")
                st.info("No numeric columns available to chart.")

    with c2:
        with st.container(border=True):
            st.markdown("**Column types**")
            st.write("")
            for label, count, color in type_counts:
                pct = round((count / total_typed) * 100)
                st.markdown(f"""
                <div class="dl-legend-row">
                    <span><span style="display:inline-block;width:8px;height:8px;border-radius:2px;background:{color};margin-right:6px"></span>{label}</span>
                    <span style="font-weight:650">{count}</span>
                </div>
                <div class="dl-legend-bar"><div class="dl-legend-fill" style="width:{pct}%;background:{color}"></div></div>
                """, unsafe_allow_html=True)

    st.write("")

    # ---- Auto-generated insights (computed from real data) ----
    insight_rows = []

    if missing_pct > 0:
        insight_rows.append(f"Dataset has <span class='dl-insight-badge'>{missing_pct}%</span> missing values across {n_cols} columns.")
    else:
        insight_rows.append("No missing values detected across the dataset.")

    if len(numeric_cols) >= 2:
        corr_raw = df[numeric_cols].corr(numeric_only=True).abs()
        corr_arr = np.array(corr_raw.values, copy=True)
        np.fill_diagonal(corr_arr, 0)
        corr = pd.DataFrame(corr_arr, index=corr_raw.index, columns=corr_raw.columns)
        if corr.max().max() > 0:
            max_pair = corr.stack().idxmax()
            max_val = corr.loc[max_pair]
            insight_rows.append(
                f"<b>{max_pair[0]}</b> and <b>{max_pair[1]}</b> are strongly correlated "
                f"<span class='dl-insight-badge'>r = {max_val:.2f}</span>."
            )

    skewed = []
    for c in numeric_cols:
        s = df[c].dropna()
        if len(s) > 5:
            sk = s.skew()
            if abs(sk) > 1:
                skewed.append(c)
    if skewed:
        cols_str = ", ".join(f"<b>{c}</b>" for c in skewed[:3])
        insight_rows.append(f"{cols_str} {'is' if len(skewed) == 1 else 'are'} right-skewed and may benefit from a log transform.")

    if n_dupes > 0:
        insight_rows.append(f"<span class='dl-insight-badge'>{n_dupes:,}</span> duplicate rows detected &mdash; consider removing before modeling.")

    rows_html = "".join(
        f'<div class="dl-insight-row">{icon("check", 14, "#1a9e6f")}<span>{r}</span></div>'
        for r in insight_rows
    )

    st.markdown(f"""
    <div class="dl-insight-card">
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:12px;font-weight:700;color:#12182b">
            {icon('sparkles', 16, '#2f6fed')} Auto-generated insights
        </div>
        {rows_html}
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    st.divider()

    st.subheader("Dataset Preview")
    st.dataframe(df.head(100), use_container_width=True, height=400)

    st.divider()

    st.subheader("Column Information")
    column_info = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values,
        "Missing": df.isnull().sum().values,
        "Unique Values": df.nunique().values,
        "Missing %": (df.isnull().sum().values / len(df) * 100).round(2)
    })
    st.dataframe(column_info, use_container_width=True)


# =========================================================
# DATA EXPLORER
# =========================================================

elif page == "Data Explorer":

    st.title("Data Explorer")

    st.write(
        "Explore individual columns in detail."
    )

    selected_column = st.selectbox(
        "Select a column",
        df.columns
    )

    series = df[selected_column]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Data Type",
            str(series.dtype)
        )

    with col2:
        st.metric(
            "Unique Values",
            series.nunique()
        )

    with col3:
        st.metric(
            "Missing Values",
            series.isnull().sum()
        )

    st.divider()

    if pd.api.types.is_numeric_dtype(series):

        st.subheader("Numerical Analysis")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Mean",
                f"{series.mean():.2f}"
            )

        with col2:
            st.metric(
                "Median",
                f"{series.median():.2f}"
            )

        with col3:
            st.metric(
                "Minimum",
                f"{series.min():.2f}"
            )

        with col4:
            st.metric(
                "Maximum",
                f"{series.max():.2f}"
            )

        fig = px.histogram(
            df,
            x=selected_column,
            title=f"Distribution of {selected_column}",
            marginal="box"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:

        st.subheader("Categorical Analysis")

        value_counts = (
            series
            .value_counts()
            .head(20)
            .reset_index()
        )

        value_counts.columns = [
            selected_column,
            "Count"
        ]

        fig = px.bar(
            value_counts,
            x=selected_column,
            y="Count",
            title=f"Top Values in {selected_column}"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.dataframe(
            value_counts,
            use_container_width=True
        )


# =========================================================
# VISUALIZATIONS
# =========================================================

elif page == "Visualizations":

    st.title("Interactive Visualizations")

    chart_type = st.selectbox(
        "Choose visualization",
        [
            "Histogram",
            "Scatter Plot",
            "Box Plot",
            "Bar Chart",
            "Line Chart",
            "Correlation Heatmap"
        ]
    )

    st.divider()

    if chart_type == "Histogram":

        if numeric_columns:

            column = st.selectbox(
                "Select column",
                numeric_columns
            )

            fig = px.histogram(
                df,
                x=column,
                title=f"Distribution of {column}",
                marginal="box"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:

            st.warning(
                "No numerical columns available."
            )


    elif chart_type == "Scatter Plot":

        if len(numeric_columns) >= 2:

            col1, col2 = st.columns(2)

            with col1:

                x_column = st.selectbox(
                    "X-axis",
                    numeric_columns
                )

            with col2:

                y_column = st.selectbox(
                    "Y-axis",
                    numeric_columns,
                    index=min(
                        1,
                        len(numeric_columns) - 1
                    )
                )

            fig = px.scatter(
                df,
                x=x_column,
                y=y_column,
                title=f"{x_column} vs {y_column}"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:

            st.warning(
                "At least two numerical columns are required."
            )


    elif chart_type == "Box Plot":

        if numeric_columns:

            column = st.selectbox(
                "Select column",
                numeric_columns
            )

            fig = px.box(
                df,
                y=column,
                title=f"Box Plot — {column}"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:

            st.warning(
                "No numerical columns available."
            )


    elif chart_type == "Bar Chart":

        if categorical_columns:

            column = st.selectbox(
                "Select category",
                categorical_columns
            )

            counts = (
                df[column]
                .value_counts()
                .head(20)
                .reset_index()
            )

            counts.columns = [
                column,
                "Count"
            ]

            fig = px.bar(
                counts,
                x=column,
                y="Count",
                title=f"Top Categories — {column}"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:

            st.warning(
                "No categorical columns available."
            )


    elif chart_type == "Line Chart":

        if numeric_columns:

            column = st.selectbox(
                "Select column",
                numeric_columns
            )

            fig = px.line(
                df,
                y=column,
                title=f"{column} — Trend"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:

            st.warning(
                "No numerical columns available."
            )


    elif chart_type == "Correlation Heatmap":

        if len(numeric_columns) >= 2:

            correlation = df[numeric_columns].corr()

            fig = px.imshow(
                correlation,
                text_auto=".2f",
                aspect="auto",
                title="Feature Correlation Matrix"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:

            st.warning(
                "At least two numerical columns are required."
            )


# =========================================================
# DATA QUALITY & CLEANING
# =========================================================

elif page == "Data Quality":

    st.title("Data Quality & Cleaning")

    st.write(
        "Identify and fix common data-quality problems."
    )

    st.divider()

    # -----------------------------------------------------
    # CURRENT STATUS
    # -----------------------------------------------------

    st.subheader("Current Data Status")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Rows",
            f"{len(df):,}"
        )

    with col2:
        st.metric(
            "Columns",
            df.shape[1]
        )

    with col3:
        st.metric(
            "Missing Values",
            f"{df.isnull().sum().sum():,}"
        )

    with col4:
        st.metric(
            "Duplicates",
            f"{df.duplicated().sum():,}"
        )

    st.divider()

    # -----------------------------------------------------
    # MISSING VALUES
    # -----------------------------------------------------

    st.subheader("Missing Values")

    missing_columns = [
        column
        for column in df.columns
        if df[column].isnull().sum() > 0
    ]

    cleaning_options = {}

    if not missing_columns:

        st.success(
            "No missing values found."
        )

    else:

        st.write(
            "Choose how you want to handle missing values."
        )

        for column in missing_columns:

            missing_count = df[column].isnull().sum()

            col1, col2 = st.columns([2, 2])

            with col1:

                st.write(
                    f"**{column}** — "
                    f"{missing_count:,} missing"
                )

            with col2:

                if pd.api.types.is_numeric_dtype(
                    df[column]
                ):

                    options = [
                        "Do Nothing",
                        "Fill with Mean",
                        "Fill with Median",
                        "Drop Rows"
                    ]

                else:

                    options = [
                        "Do Nothing",
                        "Fill with Mode",
                        "Drop Rows"
                    ]

                cleaning_options[column] = st.selectbox(
                    f"Action for {column}",
                    options,
                    key=f"missing_{column}"
                )

    st.divider()

    # -----------------------------------------------------
    # DUPLICATES
    # -----------------------------------------------------

    st.subheader("Duplicate Rows")

    duplicate_count = df.duplicated().sum()

    if duplicate_count == 0:

        st.success(
            "No duplicate rows found."
        )

        remove_duplicates = False

    else:

        st.warning(
            f"{duplicate_count:,} duplicate rows detected."
        )

        remove_duplicates = st.checkbox(
            "Remove duplicate rows"
        )

    st.divider()

    # -----------------------------------------------------
    # OUTLIERS
    # -----------------------------------------------------

    st.subheader("Outlier Detection")

    outlier_counts = {}

    if numeric_columns:

        for column in numeric_columns:

            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)

            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = (
                (df[column] < lower_bound) |
                (df[column] > upper_bound)
            ).sum()

            outlier_counts[column] = int(outliers)

        outlier_df = pd.DataFrame({
            "Column": list(outlier_counts.keys()),
            "Potential Outliers": list(
                outlier_counts.values()
            )
        })

        st.dataframe(
            outlier_df,
            use_container_width=True,
            hide_index=True
        )

        remove_outliers = st.checkbox(
            "Remove rows containing numerical outliers"
        )

    else:

        st.info(
            "No numerical columns available."
        )

        remove_outliers = False

    st.divider()

    # -----------------------------------------------------
    # APPLY CLEANING
    # -----------------------------------------------------

    st.subheader("Apply Cleaning")

    if st.button(
        "Apply Cleaning",
        type="primary",
        use_container_width=True
    ):

        cleaned = df.copy()

        before_rows = len(cleaned)
        before_missing = cleaned.isnull().sum().sum()
        before_duplicates = cleaned.duplicated().sum()

        # HANDLE MISSING VALUES

        for column, action in cleaning_options.items():

            if action == "Fill with Mean":

                cleaned[column] = cleaned[column].fillna(
                    cleaned[column].mean()
                )

            elif action == "Fill with Median":

                cleaned[column] = cleaned[column].fillna(
                    cleaned[column].median()
                )

            elif action == "Fill with Mode":

                mode = cleaned[column].mode()

                if not mode.empty:

                    cleaned[column] = cleaned[column].fillna(
                        mode.iloc[0]
                    )

            elif action == "Drop Rows":

                cleaned = cleaned.dropna(
                    subset=[column]
                )

        # REMOVE DUPLICATES

        if remove_duplicates:

            cleaned = cleaned.drop_duplicates()

        # REMOVE OUTLIERS

        if remove_outliers and numeric_columns:

            mask = pd.Series(
                True,
                index=cleaned.index
            )

            for column in numeric_columns:

                Q1 = cleaned[column].quantile(0.25)
                Q3 = cleaned[column].quantile(0.75)

                IQR = Q3 - Q1

                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                mask &= (
                    (cleaned[column] >= lower_bound) &
                    (cleaned[column] <= upper_bound)
                )

            cleaned = cleaned[mask]

        # SAVE CLEANED DATA

        st.session_state.cleaned_df = cleaned

        after_rows = len(cleaned)
        after_missing = cleaned.isnull().sum().sum()
        after_duplicates = cleaned.duplicated().sum()

        rows_removed = before_rows - after_rows
        missing_fixed = before_missing - after_missing

        duplicates_removed = (
            before_duplicates -
            after_duplicates
        )

        st.session_state.cleaning_summary = {
            "before_rows": before_rows,
            "after_rows": after_rows,
            "rows_removed": rows_removed,
            "before_missing": before_missing,
            "after_missing": after_missing,
            "missing_fixed": missing_fixed,
            "before_duplicates": before_duplicates,
            "after_duplicates": after_duplicates,
            "duplicates_removed": duplicates_removed
        }

        st.success(
            "Data cleaning completed successfully!"
        )

        # QUICK RESULTS

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Original Rows",
                f"{before_rows:,}"
            )

        with col2:

            st.metric(
                "Cleaned Rows",
                f"{after_rows:,}"
            )

        with col3:

            st.metric(
                "Rows Removed",
                f"{rows_removed:,}"
            )

        st.divider()

        # BEFORE VS AFTER

        st.subheader(
            "Before vs After Cleaning"
        )

        comparison = pd.DataFrame({
            "Metric": [
                "Rows",
                "Missing Values",
                "Duplicate Rows"
            ],
            "Before Cleaning": [
                before_rows,
                before_missing,
                before_duplicates
            ],
            "After Cleaning": [
                after_rows,
                after_missing,
                after_duplicates
            ]
        })

        st.dataframe(
            comparison,
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        # VISUAL COMPARISON

        st.subheader(
            "Cleaning Impact"
        )

        chart_data = pd.DataFrame({
            "Metric": [
                "Rows",
                "Missing Values",
                "Duplicate Rows"
            ],
            "Before": [
                before_rows,
                before_missing,
                before_duplicates
            ],
            "After": [
                after_rows,
                after_missing,
                after_duplicates
            ]
        })

        chart_data = chart_data.melt(
            id_vars="Metric",
            var_name="Stage",
            value_name="Count"
        )

        fig = px.bar(
            chart_data,
            x="Metric",
            y="Count",
            color="Stage",
            barmode="group",
            title="Data Quality Before vs After Cleaning"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.divider()

        # CLEANING RESULTS

        st.subheader(
            "Cleaning Results"
        )

        if rows_removed > 0:

            st.success(
                f"🗑️ {rows_removed:,} rows removed."
            )

        else:

            st.info(
                "No rows were removed."
            )

        if missing_fixed > 0:

            st.success(
                f"🩹 {missing_fixed:,} missing values handled."
            )

        elif after_missing == 0:

            st.success(
                "No missing values remain."
            )

        if duplicates_removed > 0:

            st.success(
                f"{duplicates_removed:,} duplicate rows removed."
            )

        elif after_duplicates == 0:

            st.success(
                "No duplicate rows remain."
            )

        st.divider()

        # CLEANED DATASET

        st.subheader(
            "Cleaned Dataset"
        )

        st.dataframe(
            cleaned.head(100),
            use_container_width=True,
            height=400
        )


# =========================================================
# SMART INSIGHTS
# =========================================================

elif page == "Insights":

    st.title("Smart Insights")

    st.write(
        "DataLens automatically analyzes your dataset "
        "and highlights important patterns."
    )

    st.divider()

    insights = []

    # Dataset size

    insights.append(
        f"**Dataset Size:** "
        f"The dataset contains **{len(df):,} rows** "
        f"across **{df.shape[1]} columns**."
    )

    # Missing values

    total_missing = df.isnull().sum().sum()

    if total_missing > 0:

        percentage = (
            total_missing /
            (df.shape[0] * df.shape[1])
        ) * 100

        insights.append(
            f"**Data Quality:** "
            f"The dataset contains **{total_missing:,} "
            f"missing values** ({percentage:.2f}% of cells)."
        )

    else:

        insights.append(
            "**Data Quality:** "
            "No missing values remain."
        )

    # Duplicates

    duplicates = df.duplicated().sum()

    if duplicates > 0:

        insights.append(
            f"**Duplicates:** "
            f"{duplicates:,} duplicate rows detected."
        )

    else:

        insights.append(
            "**Duplicates:** "
            "No duplicate rows detected."
        )

    # Numerical analysis

    if numeric_columns:

        insights.append(
            f"🔢 **Numerical Features:** "
            f"{len(numeric_columns)} numerical columns found."
        )

        for column in numeric_columns:

            series = df[column].dropna()

            if len(series) > 2:

                skewness = series.skew()

                if abs(skewness) > 1:

                    direction = (
                        "right-skewed"
                        if skewness > 0
                        else "left-skewed"
                    )

                    insights.append(
                        f"**Distribution:** "
                        f"`{column}` is strongly "
                        f"**{direction}** "
                        f"(skewness: {skewness:.2f})."
                    )

        # Correlation

        if len(numeric_columns) >= 2:

            correlation = df[numeric_columns].corr()

            pairs = []

            for i in range(len(correlation.columns)):

                for j in range(
                    i + 1,
                    len(correlation.columns)
                ):

                    value = correlation.iloc[i, j]

                    if not pd.isna(value):

                        pairs.append(
                            (
                                correlation.columns[i],
                                correlation.columns[j],
                                value
                            )
                        )

            if pairs:

                strongest = max(
                    pairs,
                    key=lambda x: abs(x[2])
                )

                insights.append(
                    f"🔗 **Strongest Relationship:** "
                    f"`{strongest[0]}` and "
                    f"`{strongest[1]}` have a correlation "
                    f"of **{strongest[2]:.2f}**."
                )

    # Categorical analysis

    if categorical_columns:

        insights.append(
            f"**Categorical Features:** "
            f"{len(categorical_columns)} categorical columns found."
        )

    # Display

    st.subheader("Key Findings")

    for insight in insights:

        st.info(insight)

    st.divider()

    st.subheader("📋 Analysis Summary")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Numerical Columns",
            len(numeric_columns)
        )

    with col2:

        st.metric(
            "Categorical Columns",
            len(categorical_columns)
        )

    with col3:

        st.metric(
            "Total Insights",
            len(insights)
        )

    st.divider()

    # =====================================================
    # AI-POWERED INSIGHTS & CHAT (Gemini)
    # =====================================================

    st.subheader("AI Insights")

    gemini_client = get_gemini_client()

    if not GENAI_AVAILABLE:
        st.warning(
            "The `google-genai` package isn't installed. Run "
            "`pip install google-genai` to enable AI features."
        )

    elif gemini_client is None:
        st.info(
            "AI features are disabled. Add your free Gemini API key to enable them: "
            "get one at [aistudio.google.com/apikey](https://aistudio.google.com/apikey), "
            "then create a file at `.streamlit/secrets.toml` in this app's folder containing:\n\n"
            "```toml\nGEMINI_API_KEY = \"your-key-here\"\n```\n\n"
            "Restart the app after saving that file."
        )

    if gemini_client is not None:
        dataset_context = build_dataset_context(df)

        # ---- Auto-generated AI summary ----
        if "ai_summary" not in st.session_state:
            st.session_state.ai_summary = None

        gen_col1, gen_col2 = st.columns([1, 4])
        with gen_col1:
            generate_clicked = st.button("Generate AI Summary", use_container_width=True)

        if generate_clicked:
            with st.spinner("Analyzing your dataset..."):
                try:
                    st.session_state.ai_summary = ask_gemini(
                        gemini_client,
                        dataset_context,
                        "Write a short, plain-English summary (3-5 sentences) of the most "
                        "important patterns, quality issues, and notable relationships in "
                        "this dataset. Be specific, referencing actual column names."
                    )
                except Exception as e:
                    st.error(f"AI request failed: {e}")

        if st.session_state.ai_summary:
            st.markdown(
                f"""<div class="dl-insight-card">{st.session_state.ai_summary}</div>""",
                unsafe_allow_html=True
            )

        st.write("")

        # ---- Chat with your data ----
        st.markdown("**Chat with your data**")

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        user_question = st.chat_input("Ask a question about this dataset...")

        if user_question:
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            with st.chat_message("user"):
                st.markdown(user_question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        answer = ask_gemini(gemini_client, dataset_context, user_question)
                        st.markdown(answer)
                        st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    except Exception as e:
                        error_msg = f"AI request failed: {e}"
                        st.error(error_msg)
                        st.session_state.chat_history.append({"role": "assistant", "content": error_msg})

        if st.session_state.chat_history:
            if st.button("Clear chat"):
                st.session_state.chat_history = []
                st.rerun()


# =========================================================
# EXPORT & EDA REPORT
# =========================================================

elif page == "Export":

    st.title("Export & EDA Report")

    st.write(
        "Download your cleaned dataset and generate a complete "
        "automated EDA report."
    )

    st.divider()

    # ---------------------------------------------------------
    # DOWNLOAD CLEANED DATASET
    # ---------------------------------------------------------

    st.subheader("Download Cleaned Dataset")

    csv_data = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇️ Download Cleaned CSV",
        data=csv_data,
        file_name="datalens_cleaned.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.success("Your cleaned dataset is ready to download.")

    st.divider()

    # ---------------------------------------------------------
    # AUTOMATED EDA REPORT
    # ---------------------------------------------------------

    st.subheader("Automated EDA Report")

    st.write(
        "The report contains dataset information, data quality, "
        "column statistics, numerical analysis, categorical analysis, "
        "correlations, outliers, cleaning results and smart insights."
    )

    report = []

    def add(line=""):
        report.append(str(line))

    # ---------------------------------------------------------
    # 1. DATASET INFORMATION
    # ---------------------------------------------------------

    add("=" * 70)
    add("DATALENS - AUTOMATED EDA REPORT")
    add("=" * 70)
    add()

    add("1. DATASET INFORMATION")
    add("-" * 70)
    add(f"Rows: {len(df):,}")
    add(f"Columns: {df.shape[1]:,}")
    add(f"Numerical Columns: {len(numeric_columns)}")
    add(f"Categorical Columns: {len(categorical_columns)}")

    if uploaded_file is not None:
        add(f"File Name: {uploaded_file.name}")

    if st.session_state.get("separator_name"):
        add(
            f"Detected Separator: "
            f"{st.session_state.separator_name}"
        )

    if st.session_state.get("detected_encoding"):
        add(
            f"Detected Encoding: "
            f"{st.session_state.detected_encoding}"
        )

    add()

    # ---------------------------------------------------------
    # 2. DATA QUALITY
    # ---------------------------------------------------------

    total_missing = int(df.isnull().sum().sum())
    total_duplicates = int(df.duplicated().sum())
    total_cells = len(df) * df.shape[1]

    missing_percentage = (
        (total_missing / total_cells) * 100
        if total_cells else 0
    )

    duplicate_percentage = (
        (total_duplicates / len(df)) * 100
        if len(df) else 0
    )

    add("2. DATA QUALITY")
    add("-" * 70)
    add(f"Missing Values: {total_missing:,}")
    add(f"Missing Percentage: {missing_percentage:.2f}%")
    add(f"Duplicate Rows: {total_duplicates:,}")
    add(f"Duplicate Percentage: {duplicate_percentage:.2f}%")
    add()

    add("Missing Values by Column:")

    missing_by_column = df.isnull().sum()

    found_missing = False

    for column, count in missing_by_column.items():
        if count > 0:
            found_missing = True
            add(f"  {column}: {int(count):,}")

    if not found_missing:
        add("  None")

    add()

    # ---------------------------------------------------------
    # 3. COLUMN INFORMATION
    # ---------------------------------------------------------

    add("3. COLUMN INFORMATION")
    add("-" * 70)

    for column in df.columns:
        add(f"Column: {column}")
        add(f"  Data Type: {df[column].dtype}")
        add(f"  Missing: {int(df[column].isnull().sum()):,}")
        add(
            f"  Unique Values: "
            f"{int(df[column].nunique(dropna=True)):,}"
        )
        add()

    # ---------------------------------------------------------
    # 4. NUMERICAL ANALYSIS
    # ---------------------------------------------------------

    add("4. NUMERICAL ANALYSIS")
    add("-" * 70)

    if numeric_columns:
        for column in numeric_columns:
            series = pd.to_numeric(
                df[column],
                errors="coerce"
            ).dropna()

            if series.empty:
                continue

            add(f"Column: {column}")
            add(f"  Mean: {series.mean():.4f}")
            add(f"  Median: {series.median():.4f}")
            add(f"  Minimum: {series.min():.4f}")
            add(f"  Maximum: {series.max():.4f}")
            add(f"  Standard Deviation: {series.std():.4f}")
            add(f"  Skewness: {series.skew():.4f}")
            add()
    else:
        add("No numerical columns detected.")
        add()

    # ---------------------------------------------------------
    # 5. CATEGORICAL ANALYSIS
    # ---------------------------------------------------------

    add("5. CATEGORICAL ANALYSIS")
    add("-" * 70)

    if categorical_columns:
        for column in categorical_columns:
            add(f"Column: {column}")
            add(
                f"  Unique Values: "
                f"{int(df[column].nunique(dropna=True)):,}"
            )

            top_values = (
                df[column]
                .value_counts(dropna=True)
                .head(5)
            )

            add("  Top Values:")

            if top_values.empty:
                add("    None")
            else:
                for value, count in top_values.items():
                    add(f"    {value}: {int(count):,}")

            add()
    else:
        add("No categorical columns detected.")
        add()

    # ---------------------------------------------------------
    # 6. CORRELATION ANALYSIS
    # ---------------------------------------------------------

    add("6. CORRELATION ANALYSIS")
    add("-" * 70)

    if len(numeric_columns) >= 2:
        correlation = df[numeric_columns].corr()
        pairs = []

        for i in range(len(correlation.columns)):
            for j in range(i + 1, len(correlation.columns)):
                value = correlation.iloc[i, j]

                if pd.notna(value):
                    pairs.append(
                        (
                            correlation.columns[i],
                            correlation.columns[j],
                            float(value)
                        )
                    )

        if pairs:
            strongest = max(
                pairs,
                key=lambda item: abs(item[2])
            )

            add(
                f"Strongest Relationship: "
                f"{strongest[0]} vs {strongest[1]}"
            )
            add(f"Correlation: {strongest[2]:.4f}")
            add()
            add("Correlation Matrix:")
            add(correlation.round(3).to_string())
        else:
            add("No valid correlations found.")
    else:
        add(
            "At least two numerical columns are required "
            "for correlation analysis."
        )

    add()

    # ---------------------------------------------------------
    # 7. OUTLIER ANALYSIS
    # ---------------------------------------------------------

    add("7. OUTLIER ANALYSIS")
    add("-" * 70)

    found_outliers = False

    for column in numeric_columns:
        series = pd.to_numeric(
            df[column],
            errors="coerce"
        ).dropna()

        if len(series) < 4:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        if iqr == 0:
            continue

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        count = int(
            ((series < lower) | (series > upper)).sum()
        )

        if count > 0:
            found_outliers = True
            add(
                f"{column}: {count:,} potential outliers"
            )

    if not found_outliers:
        add("No obvious IQR-based outliers detected.")

    add()

    # ---------------------------------------------------------
    # 8. CLEANING SUMMARY
    # ---------------------------------------------------------

    add("8. CLEANING SUMMARY")
    add("-" * 70)

    cleaning_summary = st.session_state.get(
        "cleaning_summary"
    )

    if cleaning_summary:
        add(
            f"Original Rows: "
            f"{cleaning_summary.get('before_rows', len(df)):,}"
        )
        add(
            f"Cleaned Rows: "
            f"{cleaning_summary.get('after_rows', len(df)):,}"
        )
        add(
            f"Rows Removed: "
            f"{cleaning_summary.get('rows_removed', 0):,}"
        )
        add(
            f"Missing Values Before: "
            f"{cleaning_summary.get('before_missing', 0):,}"
        )
        add(
            f"Missing Values After: "
            f"{cleaning_summary.get('after_missing', 0):,}"
        )
        add(
            f"Missing Values Fixed: "
            f"{cleaning_summary.get('missing_fixed', 0):,}"
        )
        add(
            f"Duplicates Removed: "
            f"{cleaning_summary.get('duplicates_removed', 0):,}"
        )
    else:
        add(
            "No cleaning operation has been recorded "
            "in this session."
        )

    add()

    # ---------------------------------------------------------
    # 9. SMART INSIGHTS
    # ---------------------------------------------------------

    add("9. SMART INSIGHTS")
    add("-" * 70)

    add(
        f"Dataset contains {len(df):,} rows "
        f"and {df.shape[1]:,} columns."
    )

    if total_missing == 0:
        add("No missing values remain in the current dataset.")
    else:
        add(
            f"{total_missing:,} missing values remain "
            f"({missing_percentage:.2f}% of all cells)."
        )

    if total_duplicates == 0:
        add("No duplicate rows were detected.")
    else:
        add(f"{total_duplicates:,} duplicate rows remain.")

    if numeric_columns:
        add(
            f"{len(numeric_columns)} numerical columns detected."
        )

    if categorical_columns:
        add(
            f"{len(categorical_columns)} categorical columns detected."
        )

    skewed_columns = []

    for column in numeric_columns:
        series = pd.to_numeric(
            df[column],
            errors="coerce"
        ).dropna()

        if len(series) > 2:
            skew = series.skew()

            if pd.notna(skew) and abs(skew) > 1:
                direction = (
                    "right-skewed"
                    if skew > 0
                    else "left-skewed"
                )

                skewed_columns.append(
                    f"{column} ({direction}, skew={skew:.2f})"
                )

    if skewed_columns:
        add(
            "Strongly skewed numerical columns: "
            + ", ".join(skewed_columns)
        )

    if len(numeric_columns) >= 2:
        correlation = df[numeric_columns].corr()
        pairs = []

        for i in range(len(correlation.columns)):
            for j in range(i + 1, len(correlation.columns)):
                value = correlation.iloc[i, j]

                if pd.notna(value):
                    pairs.append(
                        (
                            correlation.columns[i],
                            correlation.columns[j],
                            float(value)
                        )
                    )

        if pairs:
            strongest = max(
                pairs,
                key=lambda item: abs(item[2])
            )

            add(
                f"Strongest numerical correlation: "
                f"{strongest[0]} and {strongest[1]} "
                f"({strongest[2]:.2f})."
            )

    add()
    add("=" * 70)
    add("Generated by DataLens")
    add("Intelligent Exploratory Data Analysis Platform")
    add("=" * 70)

    report_text = "\n".join(report)

    st.success("EDA Report generated successfully!")

    st.subheader("Report Preview")

    st.text_area(
        "Generated EDA Report",
        report_text,
        height=500
    )

    st.download_button(
        label="Download EDA Report",
        data=report_text.encode("utf-8"),
        file_name="datalens_eda_report.txt",
        mime="text/plain",
        use_container_width=True
    )
