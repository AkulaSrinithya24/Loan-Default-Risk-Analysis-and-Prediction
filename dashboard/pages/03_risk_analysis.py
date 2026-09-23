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
st.title("⚠️ Risk Analysis")
st.markdown("---")


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
st.subheader("1. Risk Tier Summary")
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

st.markdown("---")
st.dataframe(
    tier_summary.style.format({
        "default_rate_pct": "{:.2f}%",
        "avg_risk_score": "{:.1f}",
        "avg_credit_score": "{:.1f}",
        "avg_loan_amount": "${:,.0f}",
        "avg_interest_rate": "{:.2f}%",
        "avg_dti_ratio": "{:.3f}",
        "total_loan_value": "${:,.0f}",
        "default_value": "${:,.0f}",
        "default_value_pct": "{:.2f}%",
    }),
    use_container_width=True,
)

st.markdown("---")

# ── Section 2: Risk tier distribution chart ────────────────────────────────────
st.subheader("2. Risk Tier Distribution")
fig1 = plot_risk_tier_distribution(df_tiers)
st.pyplot(fig1)
plt.close(fig1)

st.markdown("---")

# ── Section 3: Risk score vs actual default ────────────────────────────────────
st.subheader("3. Heuristic Risk Score vs Actual Default")
fig2 = plot_risk_score_by_default(df_tiers)
st.pyplot(fig2)
plt.close(fig2)

st.markdown("---")

# ── Section 4: Key drivers ─────────────────────────────────────────────────────
st.subheader("4. Key Numeric Drivers of Default")
st.markdown(
    "The chart below shows the **relative difference** (%) in the mean value of each "
    "feature between defaulters and non-defaulters. Larger bars = stronger driver."
)
fig3 = plot_key_drivers(df)
st.pyplot(fig3)
plt.close(fig3)

drivers = get_key_drivers(df)
st.dataframe(
    drivers.style.format({
        "Mean (Default)": "{:.3f}",
        "Mean (No Default)": "{:.3f}",
        "Abs Difference": "{:.3f}",
        "Rel Diff (%)": "{:.2f}%",
    }),
    use_container_width=True,
)

st.markdown("---")

# ── Section 5: Risk profile heatmap ───────────────────────────────────────────
st.subheader("5. Default Rate by Risk Tier x Employment Type")
fig4 = plot_risk_profile_heatmap(df_tiers)
st.pyplot(fig4)
plt.close(fig4)

st.markdown("---")

# ── Section 6: Business insights ──────────────────────────────────────────────
st.subheader("6. Business Insights & Recommendations")

insights = generate_insights(df)
priority_filter = st.multiselect(
    "Filter by priority:",
    options=["High", "Medium"],
    default=["High", "Medium"],
)

filtered = [i for i in insights if i["priority"] in priority_filter]
for ins in filtered:
    color = "#e05c5c" if ins["priority"] == "High" else "#f39c12"
    with st.expander(f"[{ins['priority']}] {ins['category']} — {ins['finding'][:80]}..."):
        st.markdown(f"**Finding:** {ins['finding']}")
        st.markdown(f"**Recommendation:** {ins['recommendation']}")
        st.code(ins["evidence"], language=None)
