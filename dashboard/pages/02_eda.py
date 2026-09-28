"""
dashboard/pages/02_eda.py
--------------------------
Exploratory Data Analysis page.
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_loader import load_data
from src.data_cleaner import clean_data
from src.eda import (
    plot_numeric_by_default,
    plot_all_numeric_by_default,
    get_default_rates_by_category,
)
from src.config import (
    RAW_DATA_PATH, TARGET_COLUMN,
    NUMERIC_FEATURES, CATEGORICAL_FEATURES, BINARY_FEATURES,
)

st.set_page_config(page_title="EDA", page_icon="🔍", layout="wide")

from dashboard.theme import page_header, section_title, show_table, apply_theme

apply_theme()

# ── Extra page-level CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
/* Ensure pyplot figures never overflow horizontally */
[data-testid="stPyplotChart"] { width: 100% !important; overflow: hidden; }
[data-testid="stPyplotChart"] img { width: 100% !important; height: auto !important; }

/* Dataset summary card */
.ds-summary-card {
    background: linear-gradient(135deg, #0a2342 0%, #1a56a0 100%);
    border-radius: 12px;
    padding: 1.6rem 2rem;
    margin-bottom: 1.4rem;
    display: flex;
    align-items: center;
    gap: 2rem;
    flex-wrap: wrap;
}
.ds-summary-title {
    color: #ffffff;
    font-size: 1.25rem;
    font-weight: 700;
    margin: 0 0 0.25rem 0;
    letter-spacing: -0.01em;
}
.ds-summary-sub {
    color: #c9d8ec;
    font-size: 0.82rem;
    margin: 0;
}
.ds-stat-pill {
    background: rgba(255,255,255,0.13);
    border: 1px solid rgba(255,255,255,0.22);
    border-radius: 8px;
    padding: 0.55rem 1.1rem;
    text-align: center;
    min-width: 90px;
}
.ds-stat-pill .val {
    color: #ffffff;
    font-size: 1.3rem;
    font-weight: 700;
    display: block;
    line-height: 1.2;
}
.ds-stat-pill .lbl {
    color: #a8c4e0;
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Action buttons row */
.eda-action-row {
    display: flex;
    gap: 0.75rem;
    margin-bottom: 1.2rem;
    flex-wrap: wrap;
}
.eda-btn-primary {
    background: #1a56a0;
    color: #ffffff !important;
    border: none;
    border-radius: 7px;
    padding: 0.55rem 1.3rem;
    font-size: 0.88rem;
    font-weight: 600;
    cursor: pointer;
    text-decoration: none !important;
    letter-spacing: 0.01em;
    display: inline-block;
}
.eda-btn-secondary {
    background: #ffffff;
    color: #0a2342 !important;
    border: 1.5px solid #dce3ed;
    border-radius: 7px;
    padding: 0.55rem 1.3rem;
    font-size: 0.88rem;
    font-weight: 600;
    cursor: pointer;
    text-decoration: none !important;
    display: inline-block;
}

/* Section card wrapper */
.eda-section-card {
    background: #ffffff;
    border: 1px solid #dce3ed;
    border-radius: 10px;
    padding: 1.3rem 1.5rem 1.1rem 1.5rem;
    margin-bottom: 1.3rem;
    box-shadow: 0 1px 4px rgba(10,35,66,0.06);
}
</style>
""", unsafe_allow_html=True)

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown("<h1>Exploratory Data Analysis</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='color:#5a6a80;font-size:0.9rem;margin-top:-0.3rem;margin-bottom:0.6rem;'>"
    "Distributions, correlations, categorical default rates and cross-feature patterns"
    "</p>",
    unsafe_allow_html=True,
)
st.markdown(
    "<hr style='border-top:1px solid #dce3ed;margin:0.4rem 0 1rem 0;'>",
    unsafe_allow_html=True,
)


@st.cache_data
def get_data():
    return clean_data(load_data(RAW_DATA_PATH))

df = get_data()

# ── Dataset Summary Card ───────────────────────────────────────────────────────
# 9 numeric + 4 categorical + 3 binary = 16 input features (LoanID and Default excluded)
total_loans  = f"{len(df):,}"
default_rate = f"{df[TARGET_COLUMN].mean()*100:.2f}%"
n_features   = str(len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES) + len(BINARY_FEATURES))  # = 16
n_numeric    = str(len(NUMERIC_FEATURES))

st.markdown(f"""
<div class="ds-summary-card">
  <div style="flex:1;min-width:180px;">
    <p class="ds-summary-title">📊 Dataset Summary</p>
    <p class="ds-summary-sub">Loan Default Risk Analysis and Prediction</p>
  </div>
  <div class="ds-stat-pill">
    <span class="val">{total_loans}</span>
    <span class="lbl">Loan Records</span>
  </div>
  <div class="ds-stat-pill">
    <span class="val">{n_features}</span>
    <span class="lbl">Features</span>
  </div>
  <div class="ds-stat-pill">
    <span class="val" style="color:#f87171;">{default_rate}</span>
    <span class="lbl">Default Rate</span>
  </div>
  <div class="ds-stat-pill">
    <span class="val">{n_numeric}</span>
    <span class="lbl">Numeric Cols</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Quick action buttons ───────────────────────────────────────────────────────
st.markdown("""
<div class="eda-action-row">
  <a href="/prediction" target="_self" class="eda-btn-primary">🎯 Go to Prediction →</a>
  <a href="/overview" target="_self" class="eda-btn-secondary">📈 Overview &amp; KPIs</a>
  <a href="/risk_analysis" target="_self" class="eda-btn-secondary">🛡️ Risk Analysis</a>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Section 1: Numeric distributions
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<div class='eda-section-card'>", unsafe_allow_html=True)
section_title("1. Numeric Feature Distributions by Default Status")
st.markdown(
    "<p style='color:#5a6a80;font-size:0.84rem;margin:0.3rem 0 0.8rem 0;'>"
    "Select a feature for a detailed KDE + box plot comparison.</p>",
    unsafe_allow_html=True,
)
feat = st.selectbox("Select numeric feature:", NUMERIC_FEATURES, index=0, key="feat_sel_1")
fig = plot_numeric_by_default(df, feat)
fig.set_size_inches(10, 3.8)
fig.tight_layout()
st.pyplot(fig, use_container_width=True)
plt.close(fig)

with st.expander("Show all numeric features at once"):
    fig2 = plot_all_numeric_by_default(df)
    fig2.set_size_inches(14, fig2.get_size_inches()[1])
    fig2.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)
st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Section 2: Correlation Heatmap
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<div class='eda-section-card'>", unsafe_allow_html=True)
section_title("2. Correlation Heatmap")
st.markdown(
    "<p style='color:#5a6a80;font-size:0.84rem;margin:0.3rem 0 0.8rem 0;'>"
    "Lower-triangle Pearson correlation matrix across all numeric features and the Default target.</p>",
    unsafe_allow_html=True,
)

num_cols_corr = NUMERIC_FEATURES + [TARGET_COLUMN]
corr = df[num_cols_corr].corr().round(2)
n = len(num_cols_corr)

# Compact, screen-fit figure: ~0.72 per cell, capped
cell_size = 0.72
fig_w = min(n * cell_size + 1.5, 11)
fig_h = min(n * cell_size + 0.8, 9)

fig3, ax3 = plt.subplots(figsize=(fig_w, fig_h), facecolor="white")
ax3.set_facecolor("white")
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(
    corr, mask=mask, annot=True, fmt=".2f",
    cmap="RdBu_r", center=0, linewidths=0.5,
    annot_kws={"size": 7.5}, ax=ax3,
    vmin=-1, vmax=1,
    cbar_kws={"shrink": 0.75, "aspect": 20},
)
ax3.set_title("Correlation Matrix — Numeric Features + Default",
              fontweight="bold", fontsize=11, pad=8, color="#0a2342")
ax3.tick_params(axis="both", labelsize=8.5, colors="#0a2342")
plt.setp(ax3.get_xticklabels(), rotation=35, ha="right", fontsize=8.5)
plt.setp(ax3.get_yticklabels(), rotation=0, fontsize=8.5)
fig3.tight_layout(pad=0.6)
st.pyplot(fig3, use_container_width=True)
plt.close(fig3)

st.markdown(
    "<p style='color:#0a2342;font-weight:600;font-size:0.88rem;"
    "margin:0.9rem 0 0.4rem 0;'>Correlation with Default (ranked):</p>",
    unsafe_allow_html=True,
)
corr_target = (
    df[NUMERIC_FEATURES + [TARGET_COLUMN]]
    .corr()[TARGET_COLUMN]
    .drop(TARGET_COLUMN)
    .abs()
    .sort_values(ascending=False)
    .round(4)
    .reset_index()
)
corr_target.columns = ["Feature", "|Pearson r| with Default"]
show_table(corr_target)
st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Section 3: Categorical default rates
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<div class='eda-section-card'>", unsafe_allow_html=True)
section_title("3. Default Rate by Categorical Feature")
st.markdown("<div style='margin-bottom:0.6rem;'></div>", unsafe_allow_html=True)
cat_choice = st.selectbox("Select categorical feature:", CATEGORICAL_FEATURES,
                           index=0, key="cat_sel_3")
rates = get_default_rates_by_category(df, cat_choice)
n_cats = len(rates)

fig4, ax4 = plt.subplots(figsize=(7, max(2.5, n_cats * 0.5 + 0.8)), facecolor="white")
ax4.set_facecolor("white")
bars = ax4.barh(
    rates[cat_choice].astype(str),
    rates["default_rate_pct"],
    color="#b91c1c", edgecolor="white", linewidth=0.8,
)
ax4.bar_label(bars, fmt="%.1f%%", padding=4, fontsize=8, color="#0a2342")
ax4.set_xlabel("Default Rate (%)", fontsize=9, color="#5a6a80")
ax4.set_title(f"Default Rate by {cat_choice}", fontweight="bold",
              color="#0a2342", fontsize=11)
ax4.set_xlim(0, rates["default_rate_pct"].max() * 1.25)
ax4.invert_yaxis()
ax4.tick_params(labelsize=9, labelcolor="#0a2342")
ax4.spines[["top", "right"]].set_visible(False)
ax4.spines[["left", "bottom"]].set_color("#dce3ed")
fig4.tight_layout(pad=0.5)
st.pyplot(fig4, use_container_width=True)
plt.close(fig4)

show_table(rates, fmt={
    "default_rate_pct": "{:.2f}%",
    "total": "{:,}",
    "defaults": "{:,}",
})
st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Section 4: Binary features
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<div class='eda-section-card'>", unsafe_allow_html=True)
section_title("4. Default Rate by Binary Features")
st.markdown("<div style='margin-bottom:0.6rem;'></div>", unsafe_allow_html=True)

binary_int = [c for c in BINARY_FEATURES if c in df.columns]
label_map  = {0: "No", 1: "Yes"}
n_bins     = len(binary_int)
n_cols_b   = min(n_bins, 3)
n_rows_b   = int(np.ceil(n_bins / n_cols_b))

fig5, axes5 = plt.subplots(n_rows_b, n_cols_b,
                            figsize=(4.5 * n_cols_b, 3.2 * n_rows_b),
                            facecolor="white")
axes5 = np.array(axes5).flatten()

for i, col in enumerate(binary_int):
    tmp = df.copy()
    tmp["_label"] = tmp[col].map(label_map)
    r = get_default_rates_by_category(tmp, "_label")
    bars = axes5[i].bar(
        r["_label"], r["default_rate_pct"],
        color=["#4a90d9", "#b91c1c"], edgecolor="white", linewidth=0.8, width=0.45,
    )
    axes5[i].bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9, color="#0a2342")
    axes5[i].set_title(f"Default Rate by {col}", fontweight="bold",
                       color="#0a2342", fontsize=10)
    axes5[i].set_ylabel("Default Rate (%)", fontsize=8, color="#5a6a80")
    axes5[i].set_ylim(0, r["default_rate_pct"].max() * 1.35)
    axes5[i].tick_params(labelsize=9, labelcolor="#0a2342")
    axes5[i].spines[["top", "right"]].set_visible(False)
    axes5[i].spines[["left", "bottom"]].set_color("#dce3ed")
    axes5[i].set_facecolor("white")

for j in range(n_bins, len(axes5)):
    axes5[j].set_visible(False)

fig5.suptitle("Default Rate by Binary Features", fontsize=12,
              fontweight="bold", color="#0a2342", y=1.01)
fig5.tight_layout(pad=0.6)
st.pyplot(fig5, use_container_width=True)
plt.close(fig5)
st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Section 5: Cross-Feature Default Rate Heatmap
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<div class='eda-section-card'>", unsafe_allow_html=True)
section_title("5. Cross-Feature Default Rate Heatmap")
st.markdown(
    "<p style='color:#5a6a80;font-size:0.84rem;margin:0.3rem 0 0.8rem 0;'>"
    "Select two categorical features to view default rate patterns across their combinations.</p>",
    unsafe_allow_html=True,
)

all_cats = CATEGORICAL_FEATURES
col_a, col_b = st.columns(2)
with col_a:
    cat1 = st.selectbox("Row feature:", all_cats, index=1, key="cat1_sel")
with col_b:
    remaining = [c for c in all_cats if c != cat1]
    cat2 = st.selectbox("Column feature:", remaining, index=0, key="cat2_sel")

pivot = (
    df.groupby([cat1, cat2])[TARGET_COLUMN]
    .mean()
    .mul(100)
    .round(1)
    .unstack(cat2)
)

n_rows_h = pivot.shape[0]
n_cols_h = pivot.shape[1]
# Compact sizing that fits the container
hw = min(max(5.5, n_cols_h * 1.1 + 1.5), 10)
hh = min(max(3.0, n_rows_h * 0.85 + 1.2), 7)

fig6, ax6 = plt.subplots(figsize=(hw, hh), facecolor="white")
ax6.set_facecolor("white")
sns.heatmap(
    pivot, annot=True, fmt=".1f", cmap="YlOrRd",
    linewidths=0.5, linecolor="white",
    cbar_kws={"label": "Default Rate (%)", "shrink": 0.8},
    annot_kws={"size": 9},
    ax=ax6,
)
ax6.set_title(f"Default Rate (%) — {cat1} vs {cat2}",
              fontweight="bold", fontsize=11, color="#0a2342", pad=8)
ax6.set_xlabel(cat2, fontsize=9, color="#5a6a80")
ax6.set_ylabel(cat1, fontsize=9, color="#5a6a80")
ax6.tick_params(axis="both", labelsize=8.5, colors="#0a2342")
plt.setp(ax6.get_xticklabels(), rotation=30, ha="right", fontsize=8.5)
plt.setp(ax6.get_yticklabels(), rotation=0, fontsize=8.5)
fig6.tight_layout(pad=0.6)
st.pyplot(fig6, use_container_width=True)
plt.close(fig6)
st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Section 6: Age Band & Income Quartile
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<div class='eda-section-card'>", unsafe_allow_html=True)
section_title("6. Default Rate by Age Band & Income Quartile")
st.markdown("<div style='margin-bottom:0.6rem;'></div>", unsafe_allow_html=True)
c_left, c_right = st.columns(2)

with c_left:
    tmp = df.copy()
    tmp["AgeBand"] = pd.cut(tmp["Age"], bins=[17, 25, 35, 45, 55, 70],
                            labels=["18-25", "26-35", "36-45", "46-55", "56-69"])
    age_rates = get_default_rates_by_category(tmp, "AgeBand")
    fig7, ax7 = plt.subplots(figsize=(5, 3.2), facecolor="white")
    ax7.set_facecolor("white")
    bars = ax7.bar(age_rates["AgeBand"].astype(str),
                   age_rates["default_rate_pct"],
                   color="#b91c1c", edgecolor="white", width=0.55)
    ax7.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=8, color="#0a2342")
    ax7.set_title("Default Rate by Age Band", fontweight="bold",
                  color="#0a2342", fontsize=11)
    ax7.set_ylabel("Default Rate (%)", color="#5a6a80", fontsize=9)
    ax7.set_ylim(0, age_rates["default_rate_pct"].max() * 1.3)
    ax7.tick_params(labelcolor="#0a2342", labelsize=9)
    ax7.spines[["top", "right"]].set_visible(False)
    ax7.spines[["left", "bottom"]].set_color("#dce3ed")
    fig7.tight_layout(pad=0.5)
    st.pyplot(fig7, use_container_width=True)
    plt.close(fig7)

with c_right:
    tmp2 = df.copy()
    tmp2["IncomeQ"] = pd.qcut(tmp2["Income"], q=4,
                               labels=["Q1 (Low)", "Q2", "Q3", "Q4 (High)"])
    inc_rates = get_default_rates_by_category(tmp2, "IncomeQ")
    inc_rates = inc_rates.sort_values("IncomeQ")
    fig8, ax8 = plt.subplots(figsize=(5, 3.2), facecolor="white")
    ax8.set_facecolor("white")
    bars = ax8.bar(inc_rates["IncomeQ"].astype(str),
                   inc_rates["default_rate_pct"],
                   color="#7c5cd8", edgecolor="white", width=0.55)
    ax8.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=8, color="#0a2342")
    ax8.set_title("Default Rate by Income Quartile", fontweight="bold",
                  color="#0a2342", fontsize=11)
    ax8.set_ylabel("Default Rate (%)", color="#5a6a80", fontsize=9)
    ax8.set_ylim(0, inc_rates["default_rate_pct"].max() * 1.3)
    ax8.tick_params(labelcolor="#0a2342", labelsize=9)
    ax8.spines[["top", "right"]].set_visible(False)
    ax8.spines[["left", "bottom"]].set_color("#dce3ed")
    fig8.tight_layout(pad=0.5)
    st.pyplot(fig8, use_container_width=True)
    plt.close(fig8)

st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Footer CTA — Get Started / Prediction navigation
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(135deg,#0a2342 0%,#1a56a0 100%);
            border-radius:10px;padding:1.5rem 2rem;margin-top:0.5rem;
            display:flex;align-items:center;justify-content:space-between;
            flex-wrap:wrap;gap:1rem;">
  <div>
    <p style="color:#ffffff;font-size:1.05rem;font-weight:700;margin:0 0 0.25rem 0;">
      Ready to predict loan default risk?
    </p>
    <p style="color:#c9d8ec;font-size:0.84rem;margin:0;">
      Use the trained ML model to get instant default probability on new loan applications.
    </p>
  </div>
  <div style="display:flex;gap:0.75rem;flex-wrap:wrap;">
    <a href="/prediction" target="_self"
       style="background:#ffffff;color:#0a2342!important;border-radius:7px;
              padding:0.55rem 1.4rem;font-size:0.88rem;font-weight:700;
              text-decoration:none!important;white-space:nowrap;">
      🎯 Go to Prediction →
    </a>
    <a href="/overview" target="_self"
       style="background:rgba(255,255,255,0.15);color:#ffffff!important;
              border:1.5px solid rgba(255,255,255,0.3);border-radius:7px;
              padding:0.55rem 1.4rem;font-size:0.88rem;font-weight:600;
              text-decoration:none!important;white-space:nowrap;">
      📊 Get Started
    </a>
  </div>
</div>
""", unsafe_allow_html=True)
