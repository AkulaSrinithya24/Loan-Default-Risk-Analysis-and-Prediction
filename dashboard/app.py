"""
dashboard/app.py
----------------
Main entry point for the Loan Default Risk Analytics Streamlit dashboard.

Run with:
    streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

st.set_page_config(
    page_title="Loan Default Risk Analysis and Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

from dashboard.theme import apply_theme

apply_theme()

# ── Extra home-page CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Sidebar: bigger logo area ───────────────────────────────────────── */
.sidebar-logo-wrap {
    padding: 1rem 0 1.2rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.12);
    margin-bottom: 0.5rem;
}
.sidebar-logo-icon { font-size: 2.2rem; line-height: 1; margin-bottom: 0.4rem; }
.sidebar-logo-name {
    color: #ffffff !important;
    font-size: 1.1rem; font-weight: 700; line-height: 1.3; letter-spacing: -0.01em;
}
.sidebar-logo-sub  { color: #8fb3d5 !important; font-size: 0.78rem; margin-top: 2px; }
.sidebar-logo-badge {
    display: inline-block;
    background: rgba(255,255,255,0.12); border: 1px solid rgba(255,255,255,0.2);
    border-radius: 4px; color: #a8c4e0 !important;
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.07em;
    padding: 2px 8px; margin-top: 0.5rem;
}

/* ── Project status block ─────────────────────────────────────────────── */
.ps-block {
    background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.12);
    border-radius: 7px; padding: 0.75rem 0.9rem;
    font-size: 0.79rem; line-height: 2.0; margin-top: 1rem;
}
.ps-label { color: #8fb3d5 !important; font-weight: 700; font-size: 0.68rem; letter-spacing: 0.07em; margin-bottom: 0.3rem; }
.ps-item-ok { color: #6ee7b7 !important; }

/* ── Hero banner — split layout ──────────────────────────────────────── */
.home-hero {
    display: grid;
    grid-template-columns: 1fr 420px;
    border-radius: 14px;
    margin-bottom: 1.6rem;
    min-height: 300px;
    overflow: hidden;
    box-shadow: 0 4px 24px rgba(10,35,66,0.18);
}
/* LEFT: solid dark navy — content lives here */
.home-hero-left {
    background: linear-gradient(150deg, #061228 0%, #0a2342 60%, #0f3460 100%);
    padding: 2.6rem 2.8rem;
    display: flex; flex-direction: column; justify-content: center;
    position: relative;
}
/* subtle blue shimmer line on the right edge of left panel */
.home-hero-left::after {
    content: "";
    position: absolute; right: 0; top: 10%; bottom: 10%;
    width: 1px;
    background: linear-gradient(to bottom, transparent, rgba(96,165,250,0.4), transparent);
}
/* RIGHT: image only — no text overlay */
.home-hero-right {
    background-image: url("https://images.unsplash.com/photo-1486406146926-c627a92ad1ab?w=900&q=85&auto=format&fit=crop");
    background-size: cover;
    background-position: center center;
}
.home-hero h1 {
    color: #ffffff !important; font-size: 2.1rem !important; font-weight: 800 !important;
    line-height: 1.2; margin: 0 0 0.3rem 0 !important;
}
.home-hero h1 span { color: #60a5fa; }
.home-hero-sub {
    color: #d6e4f5; font-size: 0.92rem; line-height: 1.6;
    max-width: 520px; margin: 0 0 1.4rem 0;
}
/* floating stat chips in the hero */
.hero-stats {
    display: flex; gap: 0.85rem; flex-wrap: wrap; margin-bottom: 1.4rem;
}
.hero-stat-chip {
    background: rgba(255,255,255,0.10);
    border: 1px solid rgba(255,255,255,0.20);
    border-radius: 8px; padding: 0.45rem 1rem;
    display: flex; align-items: center; gap: 0.45rem;
}
.hero-stat-chip-val {
    color: #ffffff !important; font-size: 1.05rem; font-weight: 800; line-height: 1;
}
.hero-stat-chip-lbl {
    color: #a8c4e0 !important; font-size: 0.7rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.05em;
}
.hero-btns { display: flex; gap: 0.75rem; flex-wrap: wrap; }
.hero-btn-primary {
    background: #1a56a0; color: #ffffff !important; border: none;
    border-radius: 8px; padding: 0.6rem 1.4rem;
    font-size: 0.9rem; font-weight: 700; text-decoration: none !important; white-space: nowrap;
    box-shadow: 0 2px 10px rgba(26,86,160,0.5);
}
.hero-btn-secondary {
    background: rgba(255,255,255,0.12); color: #ffffff !important;
    border: 1.5px solid rgba(255,255,255,0.3); border-radius: 8px;
    padding: 0.6rem 1.4rem; font-size: 0.9rem; font-weight: 600;
    text-decoration: none !important; white-space: nowrap;
}
.hero-pills { margin-top: 1.3rem; display: flex; gap: 1.4rem; flex-wrap: wrap; }
.hero-pill  { color: #a8c4e0; font-size: 0.78rem; font-weight: 600; letter-spacing: 0.02em; }

/* ── Section header with icon ─────────────────────────────────────────── */
.section-hdr {
    display: flex; align-items: center; gap: 0.55rem;
    margin-bottom: 0.85rem; margin-top: 0.2rem;
}
.section-hdr-icon  { font-size: 1.25rem; }
.section-hdr-title { color: #0a2342; font-size: 1.1rem; font-weight: 700; letter-spacing: -0.01em; }
.section-hdr-sub   { margin-left: auto; color: #1a56a0; font-size: 0.82rem; font-weight: 600; text-decoration: none; }

/* ── KPI cards ────────────────────────────────────────────────────────── */
.kpi-grid {
    display: grid; grid-template-columns: repeat(5, 1fr);
    gap: 0.75rem; margin-bottom: 1.5rem;
}
.kpi-card {
    background: #ffffff; border: 1px solid #dce3ed; border-radius: 10px;
    padding: 1rem 1.15rem; box-shadow: 0 1px 3px rgba(10,35,66,0.06);
    position: relative; overflow: hidden;
}
.kpi-card-icon  { font-size: 1.6rem; margin-bottom: 0.4rem; display: block; opacity: 0.85; }
.kpi-card-label { color: #5a6a80; font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.07em; margin-bottom: 0.2rem; }
.kpi-card-value { color: #0a2342; font-size: 1.45rem; font-weight: 800; line-height: 1.15; }
.kpi-card-delta { font-size: 0.78rem; margin-top: 0.2rem; min-height: 1.1rem; }
.kpi-accent-bar { position: absolute; left: 0; top: 0; bottom: 0; width: 4px; border-radius: 10px 0 0 10px; }

/* ── Module cards ────────────────────────────────────────────────────── */
.module-grid {
    display: grid; grid-template-columns: repeat(5, 1fr);
    gap: 0.75rem; margin-bottom: 1.4rem;
}
.module-card {
    background: #ffffff; border: 1px solid #dce3ed; border-radius: 10px;
    padding: 1.15rem 1.15rem 1rem 1.15rem; box-shadow: 0 1px 3px rgba(10,35,66,0.06);
    display: flex; flex-direction: column; position: relative;
}
.module-card-num {
    position: absolute; top: 0.9rem; right: 1rem;
    font-size: 0.72rem; font-weight: 800; letter-spacing: 0.02em; opacity: 0.7;
}
.module-card-icon-wrap {
    width: 38px; height: 38px; border-radius: 9px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.25rem; margin-bottom: 0.6rem;
}
.module-card h4 { color: #0a2342; font-size: 0.92rem; font-weight: 700; margin: 0 0 0.3rem 0; line-height: 1.3; }
.module-card p  { color: #5a6a80; font-size: 0.8rem; line-height: 1.5; flex: 1; margin: 0 0 0.75rem 0; }
.module-card-explore {
    display: flex; align-items: center; gap: 0.4rem;
    font-size: 0.8rem; font-weight: 700; text-decoration: none !important; margin-top: auto;
}
.explore-circle {
    width: 24px; height: 24px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.75rem; flex-shrink: 0;
}

/* ── How It Works strip ───────────────────────────────────────────────── */
.hiw-strip {
    background: #ffffff; border: 1px solid #dce3ed; border-radius: 10px;
    padding: 1.1rem 1.5rem; margin-bottom: 1.4rem;
    display: flex; align-items: center; overflow-x: auto;
}
.hiw-step       { display: flex; align-items: center; gap: 0.5rem; white-space: nowrap; flex-shrink: 0; }
.hiw-step-icon  { font-size: 1.1rem; }
.hiw-step-title { color: #0a2342; font-size: 0.8rem; font-weight: 700; }
.hiw-step-sub   { color: #5a6a80; font-size: 0.72rem; }
.hiw-arrow      { color: #bfd0ed; font-size: 1.1rem; margin: 0 0.6rem; flex-shrink: 0; }
.hiw-label      { color: #5a6a80; font-size: 0.75rem; font-weight: 700; letter-spacing: 0.05em; text-transform: uppercase; margin-right: 1rem; flex-shrink: 0; }

/* ── Bottom DS + CTA grid ─────────────────────────────────────────────── */
.ds-cta-grid {
    display: grid; grid-template-columns: 2fr 1fr;
    gap: 0.75rem; margin-bottom: 1rem;
}
.ds-card {
    background: #ffffff; border: 1px solid #dce3ed; border-radius: 10px;
    padding: 1.2rem 1.4rem; box-shadow: 0 1px 3px rgba(10,35,66,0.06);
}
.cta-card {
    background: linear-gradient(135deg, #0a2342 0%, #1a56a0 100%);
    border-radius: 10px; padding: 1.2rem 1.4rem;
    display: flex; flex-direction: column; justify-content: space-between;
}
.cta-card p { color: #c9d8ec; font-size: 0.84rem; margin: 0 0 1rem 0; line-height: 1.6; }
.cta-btn {
    display: inline-block; background: #ffffff; color: #0a2342 !important;
    font-weight: 700; font-size: 0.85rem; border-radius: 7px;
    padding: 0.5rem 1.2rem; text-decoration: none !important; width: fit-content;
}
.tech-stack {
    margin-top: 0.85rem; background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15); border-radius: 7px;
    padding: 0.7rem 0.85rem; font-size: 0.78rem; color: #c9d8ec; line-height: 1.7;
}

/* ── Footer ──────────────────────────────────────────────────────────── */
.home-footer {
    text-align: center; color: #8fa3bc; font-size: 0.76rem;
    padding: 0.6rem 0 0.2rem 0; border-top: 1px solid #dce3ed; margin-top: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
# ── Logo block ────────────────────────────────────────────────────────────────
st.sidebar.markdown(
    """
    <div class="sidebar-logo-wrap">
      <div class="sidebar-logo-icon">🏦</div>
      <div class="sidebar-logo-name">Loan Default Risk<br>Analysis and Prediction</div>
    </div>
    """,
    unsafe_allow_html=True,
)
# ── Navigation label ──────────────────────────────────────────────────────────
st.sidebar.markdown(
    "<div style='color:#8fb3d5;font-size:0.68rem;font-weight:700;"
    "letter-spacing:0.07em;padding:0.3rem 0 0.4rem 0;'>NAVIGATION</div>",
    unsafe_allow_html=True,
)
# ── Project status block ──────────────────────────────────────────────────────
st.sidebar.markdown(
    """
    <div class="ps-block">
      <div class="ps-label">PROJECT STATUS</div>
      <div class="ps-item-ok">&#9679;&nbsp; Dataset loaded</div>
      <div class="ps-item-ok">&#10003;&nbsp; Model deployed</div>
      <div class="ps-item-ok">&#10003;&nbsp; 255,347 records</div>
      <div class="ps-item-ok">&#10003;&nbsp; 16 features &nbsp;&bull;&nbsp; 1 target variable</div>
      <div class="ps-item-ok">&#10003;&nbsp; Application ready</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── KPI data ───────────────────────────────────────────────────────────────────
@st.cache_data
def _get_kpis():
    from src.data_loader import load_data
    from src.data_cleaner import clean_data
    from src.kpi import compute_portfolio_kpis
    from src.config import RAW_DATA_PATH
    df = clean_data(load_data(RAW_DATA_PATH))
    return compute_portfolio_kpis(df)

kpis = _get_kpis()

# ── Hero banner ────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div class="home-hero">
      <!-- LEFT: content on solid dark navy background -->
      <div class="home-hero-left">
        <h1>Loan Default Risk<br><span>Analysis and Prediction</span></h1>
        <p class="home-hero-sub">
          Data-driven credit risk insights — from portfolio analysis and risk segmentation
          to machine learning prediction, built on
          <strong style="color:#ffffff;">255,347</strong> real-world loan records.
        </p>
        <div class="hero-stats">
          <div class="hero-stat-chip">
            <span style="font-size:1.3rem;">🗄️</span>
            <div>
              <div class="hero-stat-chip-val">255,347</div>
              <div class="hero-stat-chip-lbl">Loan Records</div>
            </div>
          </div>
          <div class="hero-stat-chip">
            <span style="font-size:1.3rem;">📋</span>
            <div>
              <div class="hero-stat-chip-val">16</div>
              <div class="hero-stat-chip-lbl">Features</div>
            </div>
          </div>
          <div class="hero-stat-chip">
            <span style="font-size:1.3rem;">⚠️</span>
            <div>
              <div class="hero-stat-chip-val" style="color:#f87171 !important;">{kpis['default_rate_pct']}%</div>
              <div class="hero-stat-chip-lbl">Default Rate</div>
            </div>
          </div>
          <div class="hero-stat-chip">
            <span style="font-size:1.3rem;">🎯</span>
            <div>
              <div class="hero-stat-chip-val">Binary</div>
              <div class="hero-stat-chip-lbl">Target (0 / 1)</div>
            </div>
          </div>
        </div>
        <div class="hero-btns">
          <a href="/overview" target="_self" class="hero-btn-primary">📊 Explore the Dashboard →</a>
          <a href="/prediction" target="_self" class="hero-btn-secondary">🎯 Predict Default Risk →</a>
        </div>
        <div class="hero-pills">
          <span class="hero-pill">📈 Portfolio Analytics</span>
          <span class="hero-pill">🛡️ Risk Intelligence</span>
          <span class="hero-pill">🤖 Machine Learning</span>
          <span class="hero-pill">📊 Interactive Dashboard</span>
        </div>
      </div>
      <!-- RIGHT: finance image, no content on top -->
      <div class="home-hero-right"></div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Portfolio At a Glance ──────────────────────────────────────────────────────
st.markdown(
    """
    <div class="section-hdr">
      <span class="section-hdr-icon">📊</span>
      <span class="section-hdr-title">Portfolio At a Glance</span>
      <span style="margin-left:auto;color:#5a6a80;font-size:0.8rem;">
        🗄️ Dataset: 255,347 loan records | 16 features
      </span>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    f"""
    <div class="kpi-grid">
      <div class="kpi-card">
        <div class="kpi-accent-bar" style="background:#1a56a0;"></div>
        <span class="kpi-card-icon">🗄️</span>
        <div class="kpi-card-label">TOTAL LOANS</div>
        <div class="kpi-card-value">{kpis['total_loans']:,}</div>
        <div class="kpi-card-delta" style="color:#5a6a80;">Loan records in the dataset</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-accent-bar" style="background:#b91c1c;"></div>
        <span class="kpi-card-icon">⚠️</span>
        <div class="kpi-card-label">DEFAULT RATE</div>
        <div class="kpi-card-value" style="color:#b91c1c;">{kpis['default_rate_pct']}%</div>
        <div class="kpi-card-delta" style="color:#b91c1c;">↑ {kpis['total_defaulted']:,} defaults</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-accent-bar" style="background:#1a7a4a;"></div>
        <span class="kpi-card-icon">💰</span>
        <div class="kpi-card-label">TOTAL PORTFOLIO</div>
        <div class="kpi-card-value">${kpis['total_loan_value']/1e9:.2f}B</div>
        <div class="kpi-card-delta" style="color:#5a6a80;">Total value of all loans</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-accent-bar" style="background:#7c5cd8;"></div>
        <span class="kpi-card-icon">🎯</span>
        <div class="kpi-card-label">AVG CREDIT SCORE</div>
        <div class="kpi-card-value">{kpis['avg_credit_score']:.0f}</div>
        <div class="kpi-card-delta" style="color:#5a6a80;">Average borrower credit score</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-accent-bar" style="background:#b45309;"></div>
        <span class="kpi-card-icon">📈</span>
        <div class="kpi-card-label">AVG INTEREST RATE</div>
        <div class="kpi-card-value">{kpis['avg_interest_rate']}%</div>
        <div class="kpi-card-delta" style="color:#5a6a80;">Average loan interest rate</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Platform Modules ───────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="section-hdr">
      <span class="section-hdr-icon">⚙️</span>
      <span class="section-hdr-title">Platform Modules</span>
      <a href="/overview" target="_self" class="section-hdr-sub">Navigate to explore each module →</a>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="module-grid">
      <div class="module-card">
        <span class="module-card-num" style="color:#1a56a0;">01</span>
        <div class="module-card-icon-wrap" style="background:#e8f0fb;">📊</div>
        <h4>Overview &amp; KPIs</h4>
        <p>Portfolio-level KPIs, default distribution, segment analysis and key trends.</p>
        <a href="/overview" target="_self" class="module-card-explore" style="color:#1a56a0;">
          Explore <span class="explore-circle" style="background:#1a56a0;color:#fff;">→</span>
        </a>
      </div>
      <div class="module-card">
        <span class="module-card-num" style="color:#059669;">02</span>
        <div class="module-card-icon-wrap" style="background:#d1fae5;">🔍</div>
        <h4>Exploratory Analysis</h4>
        <p>Numeric distributions, correlation heatmap, categorical default rates and cross-feature views.</p>
        <a href="/eda" target="_self" class="module-card-explore" style="color:#059669;">
          Explore <span class="explore-circle" style="background:#059669;color:#fff;">→</span>
        </a>
      </div>
      <div class="module-card">
        <span class="module-card-num" style="color:#b91c1c;">03</span>
        <div class="module-card-icon-wrap" style="background:#fee2e2;">🛡️</div>
        <h4>Risk Analysis</h4>
        <p>Heuristic risk scoring, tier segmentation, key default drivers and business insights.</p>
        <a href="/risk_analysis" target="_self" class="module-card-explore" style="color:#b91c1c;">
          Explore <span class="explore-circle" style="background:#b91c1c;color:#fff;">→</span>
        </a>
      </div>
      <div class="module-card">
        <span class="module-card-num" style="color:#7c5cd8;">04</span>
        <div class="module-card-icon-wrap" style="background:#ede9fe;">📈</div>
        <h4>Model Performance</h4>
        <p>ROC/PR curves, confusion matrix, feature importances and threshold analysis.</p>
        <a href="/model_performance" target="_self" class="module-card-explore" style="color:#7c5cd8;">
          Explore <span class="explore-circle" style="background:#7c5cd8;color:#fff;">→</span>
        </a>
      </div>
      <div class="module-card">
        <span class="module-card-num" style="color:#b45309;">05</span>
        <div class="module-card-icon-wrap" style="background:#fef3c7;">🎯</div>
        <h4>Prediction</h4>
        <p>Enter a new loan application and get an instant default probability and risk tier.</p>
        <a href="/prediction" target="_self" class="module-card-explore" style="color:#b45309;">
          Explore <span class="explore-circle" style="background:#b45309;color:#fff;">→</span>
        </a>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── How It Works pipeline strip ────────────────────────────────────────────────
st.markdown(
    """
    <div class="hiw-strip">
      <span class="hiw-label">⚙️ How It Works</span>
      <div class="hiw-step">
        <span class="hiw-step-icon">🗄️</span>
        <div><div class="hiw-step-title">Dataset</div><div class="hiw-step-sub">Loan records &amp; features</div></div>
      </div>
      <span class="hiw-arrow">→</span>
      <div class="hiw-step">
        <span class="hiw-step-icon">🔍</span>
        <div><div class="hiw-step-title">Analysis</div><div class="hiw-step-sub">Exploratory data analysis</div></div>
      </div>
      <span class="hiw-arrow">→</span>
      <div class="hiw-step">
        <span class="hiw-step-icon">💡</span>
        <div><div class="hiw-step-title">Risk Insights</div><div class="hiw-step-sub">Segment &amp; identify drivers</div></div>
      </div>
      <span class="hiw-arrow">→</span>
      <div class="hiw-step">
        <span class="hiw-step-icon">📊</span>
        <div><div class="hiw-step-title">Model Evaluation</div><div class="hiw-step-sub">Validate performance</div></div>
      </div>
      <span class="hiw-arrow">→</span>
      <div class="hiw-step">
        <span class="hiw-step-icon">🎯</span>
        <div><div class="hiw-step-title">Prediction</div><div class="hiw-step-sub">Predict default probability</div></div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Dataset Summary + Get Started ─────────────────────────────────────────────
st.markdown(
    f"""
    <div class="ds-cta-grid">
      <div class="ds-card">
        <div style="color:#0a2342;font-size:1rem;font-weight:700;
                    margin-bottom:0.7rem;border-bottom:2px solid #1a56a0;
                    display:inline-block;padding-bottom:0.3rem;">Dataset Summary</div>
        <div style="display:flex;gap:0.5rem;flex-wrap:wrap;margin-bottom:0.7rem;">
          <span class="ldr-stat-chip">255,347 loan records</span>
          <span class="ldr-stat-chip">16 features</span>
          <span class="ldr-stat-chip">Binary target (Default 0/1)</span>
          <span class="ldr-stat-chip">80/20 train-test split</span>
        </div>
        <p style="color:#5a6a80;font-size:0.84rem;margin:0;line-height:1.7;">
          The dataset contains
          <strong style="color:#0a2342;">9 numeric features</strong>
          (age, income, loan amount, credit score, months employed, number of credit lines,
          interest rate, loan term, DTI ratio),
          <strong style="color:#0a2342;">4 categorical features</strong>
          (education, employment type, marital status, loan purpose) and
          <strong style="color:#0a2342;">3 binary flags</strong>
          (mortgage, dependents, co-signer) —
          <strong style="color:#0a2342;">16 feature columns</strong>
          plus 1 binary target (Default) = 17 modelling columns, plus 1 ID column (LoanID) = 18 total columns in the raw file.<br><br>
          The predictive model is a
          <strong style="color:#0a2342;">Random Forest classifier</strong>
          (200 trees) trained with a decision threshold of
          <strong style="color:#0a2342;">0.40</strong> to balance
          precision and recall on this imbalanced dataset.
        </p>
      </div>
      <div class="cta-card">
        <div>
          <div style="color:#ffffff;font-size:1rem;font-weight:700;margin-bottom:0.5rem;">Get Started</div>
          <p>Ready to assess a loan application? Use the Prediction module to get
             an instant default probability and risk tier from the trained model.</p>
        </div>
        <div>
          <a href="/prediction" target="_self" class="cta-btn">🎯 Go to Prediction →</a>
          <div class="tech-stack">
            <strong style="color:#ffffff;">Tech stack</strong><br>
            pandas &nbsp;·&nbsp; scikit-learn<br>
            matplotlib &nbsp;·&nbsp; seaborn<br>
            Streamlit &nbsp;·&nbsp; joblib
          </div>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="home-footer">
      Loan Default Risk Analysis and Prediction &nbsp;·&nbsp; 2024
    </div>
    """,
    unsafe_allow_html=True,
)
