"""
dashboard/theme.py
------------------
Centralised design-system for the Loan Default Risk Analytics dashboard.

Import and call  apply_theme()  at the top of every page (after
st.set_page_config) to inject the shared CSS.

Palette
-------
  Primary navy   : #0a2342
  Accent blue    : #1a56a0
  Light blue     : #e8f0fb
  White          : #ffffff
  Surface grey   : #f5f7fa
  Border         : #dce3ed
  Text primary   : #0d1b2e
  Text muted     : #5a6a80
  Success green  : #1a7a4a
  Danger red     : #b91c1c
  Warning amber  : #b45309
"""

import streamlit as st


_CSS = """
/* ── Reset & global ──────────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: -apple-system, "Segoe UI", system-ui, sans-serif !important;
    color: #0d1b2e;
}

/* ── Page background ─────────────────────────────────────────────────────── */
.stApp {
    background-color: #f5f7fa !important;
}

/* ── Sidebar ─────────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background-color: #0a2342 !important;
    border-right: 1px solid #0e2d54;
}
section[data-testid="stSidebar"] * {
    color: #c9d8ec !important;
}
section[data-testid="stSidebar"] a,
section[data-testid="stSidebar"] .stRadio label,
section[data-testid="stSidebar"] .stSelectbox label {
    color: #c9d8ec !important;
}
/* Nav links */
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a {
    color: #c9d8ec !important;
    border-radius: 6px;
    padding: 6px 12px;
    margin: 2px 0;
    font-size: 0.92rem;
    font-weight: 500;
    transition: background 0.15s;
}
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a:hover {
    background-color: rgba(255,255,255,0.08) !important;
    color: #ffffff !important;
}
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a[aria-current="page"] {
    background-color: #1a56a0 !important;
    color: #ffffff !important;
    font-weight: 700;
}
section[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.15) !important;
}
/* Hide Streamlit's auto-generated app/page title in the sidebar header */
[data-testid="stSidebarHeader"] {
    display: none !important;
}

/* ── Main content container ──────────────────────────────────────────────── */
.block-container {
    padding: 2rem 2.5rem 3rem 2.5rem !important;
    max-width: 1200px !important;
}

/* ── Page title (h1) ─────────────────────────────────────────────────────── */
h1 {
    color: #0a2342 !important;
    font-size: 1.75rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em;
    margin-bottom: 0.15rem !important;
}

/* ── Section headings (h2 / h3 / subheader) ──────────────────────────────── */
h2, h3 {
    color: #0a2342 !important;
    font-weight: 600 !important;
}

/* ── Divider ─────────────────────────────────────────────────────────────── */
hr {
    border: none !important;
    border-top: 1px solid #dce3ed !important;
    margin: 1.4rem 0 !important;
}

/* ── Metric cards ────────────────────────────────────────────────────────── */
/* Equal-height cards: column containers grow to match the tallest sibling   */
[data-testid="stHorizontalBlock"] > div {
    display: flex !important;
    flex-direction: column !important;
}
[data-testid="stMetric"] {
    background: #ffffff !important;
    border: 1px solid #dce3ed !important;
    border-radius: 8px !important;
    padding: 1rem 1.2rem 1rem 1.2rem !important;
    box-shadow: 0 1px 3px rgba(10,35,66,0.07) !important;
    min-height: 6rem !important;
    flex: 1 1 auto !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: flex-start !important;
}
[data-testid="stMetricLabel"] {
    color: #5a6a80 !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    min-height: 1.1rem;
}
[data-testid="stMetricValue"] {
    color: #0a2342 !important;
    font-size: 1.45rem !important;
    font-weight: 700 !important;
    line-height: 1.3 !important;
}
[data-testid="stMetricDelta"] {
    font-size: 0.8rem !important;
    min-height: 1.2rem;
}

/* ── Info / success / warning / error boxes ──────────────────────────────── */
[data-testid="stAlert"] {
    border-radius: 8px !important;
    border-left-width: 4px !important;
}

/* ── DataFrames / tables ─────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border: 1px solid #dce3ed !important;
    border-radius: 8px !important;
    overflow: hidden;
}

/* ── Primary buttons ─────────────────────────────────────────────────────── */
[data-testid="baseButton-primary"],
button[kind="primary"] {
    background-color: #1a56a0 !important;
    color: #ffffff !important;
    border-radius: 6px !important;
    border: none !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em;
}
[data-testid="baseButton-primary"]:hover,
button[kind="primary"]:hover {
    background-color: #154380 !important;
}

/* ── Secondary buttons ───────────────────────────────────────────────────── */
button[kind="secondary"] {
    border: 1px solid #dce3ed !important;
    border-radius: 6px !important;
    color: #0a2342 !important;
    background: #ffffff !important;
}

/* ── Selectbox / slider / radio labels ───────────────────────────────────── */
label[data-baseweb="label"] {
    color: #0d1b2e !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
}

/* ── Expanders ───────────────────────────────────────────────────────────── */
details[data-testid="stExpander"] {
    border: 1px solid #dce3ed !important;
    border-radius: 8px !important;
    background: #ffffff !important;
}
details[data-testid="stExpander"] summary {
    font-weight: 600 !important;
    color: #0a2342 !important;
    padding: 0.75rem 1rem !important;
}

/* ── Tab strip ───────────────────────────────────────────────────────────── */
[data-testid="stTabs"] [data-baseweb="tab"] {
    color: #5a6a80 !important;
    font-weight: 500;
}
[data-testid="stTabs"] [aria-selected="true"] {
    color: #1a56a0 !important;
    border-bottom-color: #1a56a0 !important;
}

/* ── Matplotlib figure container ─────────────────────────────────────────── */
[data-testid="stImage"] img,
[data-testid="stPyplotChart"] {
    border-radius: 8px;
}

/* ── Card helper class ───────────────────────────────────────────────────── */
.ldr-card {
    background: #ffffff;
    border: 1px solid #dce3ed;
    border-radius: 8px;
    padding: 1.25rem 1.4rem;
    box-shadow: 0 1px 3px rgba(10,35,66,0.06);
    margin-bottom: 1rem;
}
.ldr-hero {
    background: linear-gradient(135deg, #0a2342 0%, #1a56a0 100%);
    border-radius: 10px;
    padding: 2.2rem 2.4rem;
    margin-bottom: 1.5rem;
}
.ldr-hero h1 { color: #ffffff !important; }
.ldr-hero p  { color: #c9d8ec !important; margin: 0; font-size: 1rem; }
.ldr-tag {
    display: inline-block;
    background: rgba(255,255,255,0.15);
    color: #c9d8ec !important;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    padding: 3px 10px;
    border-radius: 20px;
    margin-bottom: 0.7rem;
}
.ldr-section-title {
    color: #0a2342;
    font-size: 1.05rem;
    font-weight: 700;
    margin-bottom: 0.6rem;
    padding-bottom: 0.35rem;
    border-bottom: 2px solid #1a56a0;
    display: inline-block;
}
/* ── KPI cards (home page — HTML grid, always same height) ───────────────── */
.ldr-kpi-card {
    background: #ffffff;
    border: 1px solid #dce3ed;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    box-shadow: 0 1px 3px rgba(10,35,66,0.07);
    display: flex;
    flex-direction: column;
    gap: 0.18rem;
}
.ldr-kpi-label {
    color: #5a6a80;
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.ldr-kpi-value {
    color: #0a2342;
    font-size: 1.45rem;
    font-weight: 700;
    line-height: 1.2;
}
.ldr-kpi-sub {
    color: #5a6a80;
    font-size: 0.78rem;
    min-height: 1.1rem;
}
.ldr-kpi-danger { color: #b91c1c !important; }

/* ── Module cards (home page — HTML grid, always same height) ────────────── */
.ldr-module-card {
    background: #ffffff;
    border: 1px solid #dce3ed;
    border-top: 3px solid #1a56a0;
    border-radius: 8px;
    padding: 1.1rem 1.2rem;
    box-shadow: 0 1px 4px rgba(10,35,66,0.06);
    display: flex;
    flex-direction: column;
}
.ldr-module-card h4 {
    color: #0a2342;
    font-size: 0.95rem;
    font-weight: 700;
    margin: 0 0 0.35rem 0;
}
.ldr-module-card p {
    color: #5a6a80;
    font-size: 0.83rem;
    margin: 0;
    line-height: 1.5;
    flex: 1;
}
.ldr-stat-row {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
    margin-bottom: 0.5rem;
}
.ldr-stat-chip {
    background: #e8f0fb;
    border: 1px solid #bfd0ed;
    border-radius: 20px;
    color: #1a56a0;
    font-size: 0.78rem;
    font-weight: 600;
    padding: 3px 12px;
}
.sidebar-app-name {
    color: #ffffff !important;
    font-size: 1.05rem;
    font-weight: 700;
    letter-spacing: -0.01em;
}
.sidebar-subtitle {
    color: #8fb3d5 !important;
    font-size: 0.78rem;
    margin-top: -2px;
}
.sidebar-badge {
    display: inline-block;
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 4px;
    color: #a8c4e0 !important;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    padding: 2px 8px;
    margin-top: 0.4rem;
}

/* ── HTML table (pyarrow-free fallback) ──────────────────────────────────── */
.ldr-table-wrap {
    overflow-x: auto;
    border: 1px solid #dce3ed;
    border-radius: 8px;
    margin-bottom: 0.8rem;
}
.ldr-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.82rem;
    color: #0d1b2e;
    background: #ffffff;
}
.ldr-table thead tr {
    background: #f0f4fa;
    border-bottom: 2px solid #dce3ed;
}
.ldr-table thead th {
    padding: 0.55rem 0.85rem;
    text-align: left;
    font-weight: 700;
    color: #0a2342;
    white-space: nowrap;
    font-size: 0.78rem;
    letter-spacing: 0.02em;
}
.ldr-table tbody tr {
    border-bottom: 1px solid #f0f4fa;
}
.ldr-table tbody tr:last-child { border-bottom: none; }
.ldr-table tbody tr:hover { background: #f7f9fd; }
.ldr-table tbody td {
    padding: 0.48rem 0.85rem;
    white-space: nowrap;
}
"""


def apply_theme() -> None:
    """Inject the shared CSS into the current Streamlit page."""
    st.markdown(f"<style>{_CSS}</style>", unsafe_allow_html=True)


def sidebar_header() -> None:
    """Render the branded sidebar header (call once per page)."""
    # ── Logo block ────────────────────────────────────────────────────────────
    st.sidebar.markdown(
        """
        <div style="padding: 0.6rem 0 1rem 0; border-bottom: 1px solid rgba(255,255,255,0.12); margin-bottom: 0.5rem;">
          <div style="font-size:2rem;line-height:1;margin-bottom:0.35rem;">🏦</div>
          <div class="sidebar-app-name">Loan Default Risk<br>Analysis and Prediction</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    # ── Navigation label ─────────────────────────────────────────────────────
    st.sidebar.markdown(
        "<div style='color:#8fb3d5;font-size:0.68rem;font-weight:700;"
        "letter-spacing:0.07em;padding:0.3rem 0 0.4rem 0;'>NAVIGATION</div>",
        unsafe_allow_html=True,
    )
    # ── Project status block ──────────────────────────────────────────────────
    st.sidebar.markdown(
        """
        <div style="background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.12);
                    border-radius:7px;padding:0.7rem 0.9rem;font-size:0.79rem;
                    line-height:2.0;margin:0.6rem 0 0.8rem 0;">
          <div style="color:#8fb3d5;font-weight:700;font-size:0.68rem;letter-spacing:0.07em;margin-bottom:0.3rem;">PROJECT STATUS</div>
          <div style="color:#6ee7b7;">&#9679;&nbsp; Dataset loaded</div>
          <div style="color:#6ee7b7;">&#10003;&nbsp; Model deployed</div>
          <div style="color:#6ee7b7;">&#10003;&nbsp; 255,347 records</div>
          <div style="color:#6ee7b7;">&#10003;&nbsp; 16 features &nbsp;&bull;&nbsp; 1 target variable</div>
          <div style="color:#6ee7b7;">&#10003;&nbsp; Application ready</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str = "") -> None:
    """Render a consistent page title + optional subtitle."""
    apply_theme()
    st.markdown(f"<h1>{title}</h1>", unsafe_allow_html=True)
    if subtitle:
        st.markdown(
            f"<p style='color:#5a6a80;font-size:0.9rem;margin-top:-0.3rem;"
            f"margin-bottom:0.8rem;'>{subtitle}</p>",
            unsafe_allow_html=True,
        )
    st.markdown(
        "<hr style='border-top:1px solid #dce3ed;margin:0.6rem 0 1.2rem 0;'>",
        unsafe_allow_html=True,
    )


def section_title(text: str) -> None:
    """Render a consistent section heading with blue underline accent."""
    st.markdown(
        f"<div class='ldr-section-title'>{text}</div>",
        unsafe_allow_html=True,
    )


def show_table(df, fmt: dict = None) -> None:
    """
    Render a pandas DataFrame as a styled HTML table.

    Uses zero pyarrow / Arrow serialisation — safe on machines where
    pyarrow's DLL is blocked by Application Control policies.

    Parameters
    ----------
    df  : pd.DataFrame  — the data to display (pass the raw frame, not a Styler)
    fmt : dict          — optional {column_name: format_string} mapping, e.g.
                          {"rate": "{:.2f}%", "amount": "${:,.0f}"}
    """
    import pandas as pd

    display = df.copy()

    if fmt:
        for col, pattern in fmt.items():
            if col in display.columns:
                display[col] = display[col].apply(
                    lambda v: pattern.format(v) if pd.notna(v) else ""
                )

    # Build the HTML table manually — no Styler, no Arrow
    header = "".join(f"<th>{c}</th>" for c in display.columns)
    rows = ""
    for _, row in display.iterrows():
        cells = "".join(f"<td>{v}</td>" for v in row)
        rows += f"<tr>{cells}</tr>"

    html = (
        "<div class='ldr-table-wrap'>"
        f"<table class='ldr-table'>"
        f"<thead><tr>{header}</tr></thead>"
        f"<tbody>{rows}</tbody>"
        "</table></div>"
    )
    st.markdown(html, unsafe_allow_html=True)
