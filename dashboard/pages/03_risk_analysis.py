"""
dashboard/pages/03_risk_analysis.py
-------------------------------------
Risk Analysis page — risk scores, tiers, key drivers, insights.
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from src.data_loader import load_data
from src.data_cleaner import clean_data
from src.risk_analysis import (
    assign_risk_tier,
    get_risk_tier_summary,
    get_key_drivers,
    plot_risk_tier_distribution,
    plot_risk_score_by_default,
    plot_key_drivers,
    plot_risk_profile_heatmap,
)
from src.insights import generate_insights
from src.config import RAW_DATA_PATH

st.set_page_config(page_title="Risk Analysis", page_icon="⚠️", layout="wide")

from dashboard.theme import page_header, section_title, show_table

page_header("Risk Analysis",
            "Risk scoring, tier segmentation, key default drivers and business insights")


@st.cache_data
def get_data():
    return clean_data(load_data(RAW_DATA_PATH))

@st.cache_data
def get_tiered_data():
    df = get_data()
    return assign_risk_tier(df)

df = get_data()
df_tiers = get_tiered_data()

# ── Section 1: Risk tier summary ───────────────────────────────────────────────
section_title("1. Risk Tier Summary")
st.markdown("<div style='margin-bottom:0.6rem;'></div>", unsafe_allow_html=True)

tier_summary = get_risk_tier_summary(df_tiers)

col1, col2, col3, col4 = st.columns(4)
tier_colors = {"Low": "green", "Medium": "orange", "High": "red", "Very High": "red"}
for col, (_, row) in zip([col1, col2, col3, col4], tier_summary.iterrows()):
    col.metric(
        f"{row['RiskTier']} Risk",
        f"{row['default_rate_pct']}%",
        delta=f"{row['total_loans']:,} loans",
        delta_color="off",
    )

st.markdown("<div style='margin-top:0.7rem;'></div>", unsafe_allow_html=True)
show_table(tier_summary, fmt={
    "default_rate_pct": "{:.2f}%",
    "avg_risk_score": "{:.1f}",
    "avg_credit_score": "{:.1f}",
    "avg_loan_amount": "${:,.0f}",
    "avg_interest_rate": "{:.2f}%",
    "avg_dti_ratio": "{:.3f}",
    "total_loan_value": "${:,.0f}",
    "default_value": "${:,.0f}",
    "default_value_pct": "{:.2f}%",
})

st.markdown("<hr>", unsafe_allow_html=True)

# ── Section 2: Risk tier distribution chart ────────────────────────────────────
section_title("2. Risk Tier Distribution")
st.markdown("<div style='margin-bottom:0.6rem;'></div>", unsafe_allow_html=True)
fig1 = plot_risk_tier_distribution(df_tiers)
st.pyplot(fig1)
plt.close(fig1)

st.markdown("<hr>", unsafe_allow_html=True)

# ── Section 3: Risk score vs actual default ────────────────────────────────────
section_title("3. Heuristic Risk Score vs Actual Default")
st.markdown("<div style='margin-bottom:0.6rem;'></div>", unsafe_allow_html=True)
fig2 = plot_risk_score_by_default(df_tiers)
st.pyplot(fig2)
plt.close(fig2)

st.markdown("<hr>", unsafe_allow_html=True)

# ── Section 4: Key drivers ─────────────────────────────────────────────────────
section_title("4. Key Numeric Drivers of Default")
st.markdown(
    "<p style='color:#5a6a80;font-size:0.85rem;margin:0.3rem 0 0.7rem 0;'>"
    "The chart below shows the <strong>relative difference</strong> (%) in the mean "
    "value of each feature between defaulters and non-defaulters. "
    "Larger bars indicate stronger drivers.</p>",
    unsafe_allow_html=True,
)
fig3 = plot_key_drivers(df)
st.pyplot(fig3)
plt.close(fig3)

drivers = get_key_drivers(df)
show_table(drivers, fmt={
    "Mean (Default)": "{:.3f}",
    "Mean (No Default)": "{:.3f}",
    "Abs Difference": "{:.3f}",
    "Rel Diff (%)": "{:.2f}%",
})

st.markdown("<hr>", unsafe_allow_html=True)

# ── Section 5: Risk profile heatmap ───────────────────────────────────────────
section_title("5. Default Rate by Risk Tier x Employment Type")
st.markdown("<div style='margin-bottom:0.6rem;'></div>", unsafe_allow_html=True)
fig4 = plot_risk_profile_heatmap(df_tiers)
st.pyplot(fig4)
plt.close(fig4)

st.markdown("<hr>", unsafe_allow_html=True)

# ── Section 6: Business insights ──────────────────────────────────────────────
section_title("6. Business Insights & Recommendations")
st.markdown("<div style='margin-bottom:0.6rem;'></div>", unsafe_allow_html=True)

insights = generate_insights(df)
priority_filter = st.multiselect(
    "Filter by priority:",
    options=["High", "Medium"],
    default=["High", "Medium"],
)

filtered = [i for i in insights if i["priority"] in priority_filter]
for ins in filtered:
    color = "#b91c1c" if ins["priority"] == "High" else "#b45309"
    with st.expander(f"[{ins['priority']}] {ins['category']} — {ins['finding'][:80]}..."):
        st.markdown(f"**Finding:** {ins['finding']}")
        st.markdown(f"**Recommendation:** {ins['recommendation']}")
        st.code(ins["evidence"], language=None)
