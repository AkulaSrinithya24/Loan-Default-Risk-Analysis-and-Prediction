"""
dashboard/pages/07_scenario_analysis.py
-----------------------------------------
Portfolio Scenario Analysis page.

PURPOSE
-------
A business-oriented "what-if" analysis tool that lets the user build a
hypothetical portfolio by adjusting the *mix* of employment types, loan
purposes, credit-score range, DTI range, and average loan amount, then
estimates how the portfolio's default rate and financial exposure would
change under that scenario.

WHAT THIS PAGE DOES NOT DUPLICATE
----------------------------------
- Portfolio-level KPI tiles                       -> 01_overview.py
- Default count / class-balance charts            -> 01_overview.py
- Feature distribution KDE / box plots            -> 02_eda.py
- Default-rate-by-segment bar charts              -> 01_overview.py / 02_eda.py
- Risk tier distribution / heuristic risk scores  -> 03_risk_analysis.py
- ROC / PR curves / confusion matrix              -> 04_model_performance.py
- Single-applicant prediction form                -> 05_prediction.py
- Dollar exposure treemap / cross-segment tables  -> 06_portfolio_exposure.py

UNIQUE CONTRIBUTIONS
--------------------
1. Baseline vs Scenario comparison table (rate, value, count, exposure)
2. Baseline vs Scenario default-rate comparison bar chart
3. Baseline vs Scenario defaulted loan value comparison bar chart
4. Dynamically generated scenario interpretation text

SCENARIO METHODOLOGY
--------------------
Defensible resampling approach — NO arbitrary multipliers:

Step 1  Load the cleaned historical dataset.
Step 2  Build a scenario population by resampling rows from the cleaned
        dataset subject to the selected filters:
          - EmploymentType  : rows whose EmploymentType share matches the
                              user-specified mix (stratified resample).
          - LoanPurpose     : same stratified resample approach.
          - CreditScore     : filter to the user-selected band.
          - DTIRatio        : filter to the user-selected band.
          - LoanAmount      : scale each row's LoanAmount so the portfolio
                              mean matches the user-specified target mean
                              (preserves relative loan-size distribution,
                              just shifts the level; capped at NUMERIC_BOUNDS).
Step 3  Pass the scenario population through the EXISTING trained model
        pipeline (pipeline.predict_proba) to obtain per-row default
        probabilities.  No retraining occurs.
Step 4  Apply the EXISTING DECISION_THRESHOLD (0.40) to derive predicted
        default flags.
Step 5  Compute scenario metrics from the predicted flags and the
        (scaled) LoanAmount values.

Why this is defensible
----------------------
- All records come from the real dataset, so all feature correlations
  (e.g. income vs credit score) are preserved.
- Only the *composition* of the portfolio changes, not invented values.
- The existing model + threshold are unchanged.
- Scenario estimates are clearly labelled throughout.

LIMITATIONS
-----------
- Scenarios are estimates; real outcomes depend on macro conditions the
  model was not trained on.
- The model was trained on a snapshot dataset; it does not account for
  concept drift.
- Loan-amount scaling assumes a linear shift in the distribution; it
  does not model non-linear interactions.
- If a filter combination yields very few records the estimate will have
  high variance.  The page warns when the scenario population is small.

FILES CREATED   : dashboard/pages/07_scenario_analysis.py
FILES MODIFIED  : none
"""

# ── Standard path setup (identical to every other page) ──────────────────────
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Project imports ───────────────────────────────────────────────────────────
from src.data_loader import load_data
from src.data_cleaner import clean_data
from src.kpi import compute_portfolio_kpis
from src.config import (
    RAW_DATA_PATH,
    TARGET_COLUMN,
    MODEL_PATH,
    DEPLOYMENT_MODEL_PATH,
    DECISION_THRESHOLD,
    ALL_FEATURES,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    BINARY_FEATURES,
)

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Scenario Analysis",
    page_icon="🔮",
    layout="wide",
)

from dashboard.theme import page_header, section_title, show_table, apply_theme

apply_theme()

page_header(
    "Portfolio Scenario Analysis",
    "Explore how hypothetical changes in portfolio composition could affect "
    "estimated default risk and financial exposure.",
)

# ── Design tokens (consistent with project palette) ──────────────────────────
NAVY       = "#0a2342"
BLUE       = "#1a56a0"
LIGHT_BLUE = "#e8f0fb"
RED        = "#b91c1c"
AMBER      = "#b45309"
GREEN      = "#1a7a4a"
MUTED      = "#5a6a80"
BORDER     = "#dce3ed"

# ═════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_data
def _load_clean_data() -> pd.DataFrame:
    """Load and clean the dataset — same pattern used by every existing page."""
    return clean_data(load_data(RAW_DATA_PATH))


df_full = _load_clean_data()

# ═════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ═════════════════════════════════════════════════════════════════════════════

# Resolve model path — same fallback logic as 05_prediction.py and
# 04_model_performance.py.  The model is NOT retrained or modified.
if MODEL_PATH.exists():
    _model_path = MODEL_PATH
elif DEPLOYMENT_MODEL_PATH.exists():
    _model_path = DEPLOYMENT_MODEL_PATH
    st.info(
        "Using the deployment model (smaller variant). "
        "The scenario analysis methodology is identical; "
        "estimated probabilities may differ slightly from the full model."
    )
else:
    st.error(
        "No trained model file found. Run `python run_pipeline.py` or "
        "`python scripts/create_deployment_model.py` to generate a model."
    )
    st.stop()


@st.cache_resource
def _load_pipeline():
    """Load the trained pipeline — cached so it is loaded only once."""
    import joblib
    return joblib.load(_model_path)


pipeline = _load_pipeline()

# ═════════════════════════════════════════════════════════════════════════════
# PRE-COMPUTE BASELINE METRICS FROM THE HISTORICAL DATASET
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_data
def _compute_baseline(df: pd.DataFrame) -> dict:
    """
    Compute historical baseline metrics directly from the dataset.
    These values come 100% from the real data — no model involved.
    They should match the Overview and Portfolio Exposure pages exactly.
    """
    kpis = compute_portfolio_kpis(df)
    return {
        "total_loans":          kpis["total_loans"],
        "default_rate_pct":     kpis["default_rate_pct"],
        "total_loan_value":     kpis["total_loan_value"],
        "defaulted_loan_value": kpis["defaulted_loan_value"],
        "default_count":        kpis["total_defaulted"],
        "exposure_rate_pct":    kpis["default_value_rate_pct"],
        "avg_loan_amount":      kpis["avg_loan_amount"],
        "avg_credit_score":     kpis["avg_credit_score"],
        "avg_dti_ratio":        kpis["avg_dti_ratio"],
    }


baseline = _compute_baseline(df_full)

# ═════════════════════════════════════════════════════════════════════════════
# VALID CATEGORY VALUES (from data_cleaner.VALID_VALUES)
# ═════════════════════════════════════════════════════════════════════════════

EMPLOYMENT_TYPES = ["Full-time", "Part-time", "Self-employed", "Unemployed"]
LOAN_PURPOSES    = ["Auto", "Business", "Education", "Home", "Other"]

# Credit score range bounds (from NUMERIC_BOUNDS in data_cleaner)
CS_MIN, CS_MAX = 300, 850
# DTI range bounds
DTI_MIN, DTI_MAX = 0.0, 1.5   # practical max from the dataset (bounded at 5.0 but realistic upper ~ 1.5)
# Loan amount practical range
LA_MIN  = int(df_full["LoanAmount"].quantile(0.05))   # 5th percentile
LA_MAX  = int(df_full["LoanAmount"].quantile(0.95))   # 95th percentile
LA_MEAN = int(df_full["LoanAmount"].mean())

# ═════════════════════════════════════════════════════════════════════════════
# SCENARIO GENERATION
# ═════════════════════════════════════════════════════════════════════════════

def _build_scenario_population(
    df: pd.DataFrame,
    emp_weights: dict,       # {EmploymentType: float}  — must sum to ~1.0
    purpose_weights: dict,   # {LoanPurpose:    float}  — must sum to ~1.0
    cs_range: tuple,         # (min_cs, max_cs) — credit score filter
    dti_range: tuple,        # (min_dti, max_dti) — DTI filter
    target_loan_mean: float, # target average loan amount
    target_n: int = 5000,    # size of scenario portfolio (rows resampled)
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Build a hypothetical portfolio population by resampling historical records.

    Methodology
    -----------
    1. Apply credit-score and DTI filters to create an eligible pool.
    2. From the eligible pool, stratified-resample rows so that the
       EmploymentType distribution matches emp_weights.
    3. Re-stratify by LoanPurpose to match purpose_weights.
    4. Scale LoanAmount so the population mean equals target_loan_mean.
    5. Return the scenario DataFrame ready for model inference.

    The returned DataFrame has the same 16 feature columns expected by
    the pipeline (no target column, no LoanID).
    """
    rng = np.random.default_rng(random_state)

    # ── Step 1: filter by credit score and DTI ────────────────────────────
    pool = df[
        (df["CreditScore"] >= cs_range[0]) &
        (df["CreditScore"] <= cs_range[1]) &
        (df["DTIRatio"]    >= dti_range[0]) &
        (df["DTIRatio"]    <= dti_range[1])
    ].copy()

    if len(pool) < 50:
        # Pool too small — return None to signal insufficient data
        return None

    # ── Step 2: stratified resample by EmploymentType ────────────────────
    # For each employment type, sample a fraction proportional to emp_weight.
    emp_frames = []
    for emp_type, weight in emp_weights.items():
        n_target = max(1, int(round(target_n * weight)))
        sub = pool[pool["EmploymentType"] == emp_type]
        if len(sub) == 0:
            continue
        sampled = sub.sample(
            n=min(n_target, len(sub) * 5),   # allow replacement up to 5x
            replace=(n_target > len(sub)),
            random_state=rng.integers(0, 100_000),
        )
        emp_frames.append(sampled)

    if not emp_frames:
        return None

    emp_pop = pd.concat(emp_frames, ignore_index=True)

    # ── Step 3: stratified resample by LoanPurpose ────────────────────────
    purpose_frames = []
    for purpose, weight in purpose_weights.items():
        n_target = max(1, int(round(target_n * weight)))
        sub = emp_pop[emp_pop["LoanPurpose"] == purpose]
        if len(sub) == 0:
            # Fall back to pool for this purpose
            sub = pool[pool["LoanPurpose"] == purpose]
        if len(sub) == 0:
            continue
        sampled = sub.sample(
            n=min(n_target, len(sub) * 5),
            replace=(n_target > len(sub)),
            random_state=rng.integers(0, 100_000),
        )
        purpose_frames.append(sampled)

    if not purpose_frames:
        return None

    scenario_pop = pd.concat(purpose_frames, ignore_index=True)

    # Trim/pad to exactly target_n rows
    if len(scenario_pop) > target_n:
        scenario_pop = scenario_pop.sample(n=target_n, random_state=random_state)
    elif len(scenario_pop) < target_n:
        top_up = scenario_pop.sample(
            n=target_n - len(scenario_pop),
            replace=True,
            random_state=random_state,
        )
        scenario_pop = pd.concat([scenario_pop, top_up], ignore_index=True)

    # ── Step 4: scale LoanAmount to target mean ────────────────────────────
    # Only scale if the current mean is non-zero and differs from target.
    current_mean = scenario_pop["LoanAmount"].mean()
    if current_mean > 0 and abs(current_mean - target_loan_mean) > 1:
        scale_factor = target_loan_mean / current_mean
        scenario_pop["LoanAmount"] = (
            scenario_pop["LoanAmount"] * scale_factor
        ).clip(lower=100, upper=10_000_000).round(2)

    # ── Step 5: retain only the 16 feature columns the pipeline expects ────
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES
    # All columns must be present — they are because pool came from clean_data()
    return scenario_pop[feature_cols].reset_index(drop=True)


def _run_scenario_inference(
    scenario_df: pd.DataFrame,
    pipeline,
    threshold: float,
) -> dict:
    """
    Pass the scenario population through the existing pipeline and compute
    scenario metrics.

    Returns
    -------
    dict with keys:
        total_loans, default_count, default_rate_pct,
        total_loan_value, defaulted_loan_value, exposure_rate_pct,
        avg_loan_amount, avg_default_prob
    """
    probs = pipeline.predict_proba(scenario_df)[:, 1]   # P(default)
    preds = (probs >= threshold).astype(int)

    loan_vals  = scenario_df["LoanAmount"].values
    def_mask   = preds == 1

    total_val  = float(loan_vals.sum())
    def_val    = float(loan_vals[def_mask].sum())

    return {
        "total_loans":          len(scenario_df),
        "default_count":        int(preds.sum()),
        "default_rate_pct":     round(float(preds.mean()) * 100, 2),
        "total_loan_value":     round(total_val, 2),
        "defaulted_loan_value": round(def_val, 2),
        "exposure_rate_pct":    round(def_val / total_val * 100, 2) if total_val > 0 else 0.0,
        "avg_loan_amount":      round(float(loan_vals.mean()), 2),
        "avg_default_prob":     round(float(probs.mean()) * 100, 2),
    }

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — HISTORICAL PORTFOLIO BASELINE
# ═════════════════════════════════════════════════════════════════════════════

section_title("1. Historical Portfolio Baseline")
st.markdown(
    "<p style='color:#5a6a80; font-size:0.85rem; margin-bottom:0.4rem;'>"
    "Baseline values are calculated directly from the historical dataset "
    "used in this project. They match the values shown on the Overview "
    "and Portfolio Exposure pages.</p>",
    unsafe_allow_html=True,
)

b_c1, b_c2, b_c3, b_c4 = st.columns(4)
b_c1.metric("Total Loans",         f"{baseline['total_loans']:,}")
b_c2.metric("Historical Default Rate", f"{baseline['default_rate_pct']}%",
            delta=f"{baseline['default_count']:,} defaults", delta_color="inverse")
b_c3.metric("Total Loan Value",    f"${baseline['total_loan_value']/1e9:.2f}B")
b_c4.metric("Defaulted Loan Value",f"${baseline['defaulted_loan_value']/1e9:.2f}B",
            delta=f"{baseline['exposure_rate_pct']}% exposure rate", delta_color="inverse")

st.markdown("<hr>", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — SCENARIO CONTROLS
# ═════════════════════════════════════════════════════════════════════════════

section_title("2. Build a What-If Scenario")
st.markdown(
    "<p style='color:#5a6a80; font-size:0.85rem; margin-bottom:0.8rem;'>"
    "Adjust the controls below to define a hypothetical portfolio. "
    "The scenario engine resamples historical loan records to match your "
    "chosen composition, then passes them through the existing trained model "
    "to estimate outcomes. <strong>No model retraining occurs.</strong></p>",
    unsafe_allow_html=True,
)

# ── Sliders render in two columns to keep the layout compact ────────────────
ctrl_left, ctrl_right = st.columns(2)

# ─── Left column: Employment & Loan Purpose mix ──────────────────────────────
with ctrl_left:

    st.markdown(
        f"<div style='color:{NAVY};font-weight:700;font-size:0.92rem;"
        f"margin-bottom:0.4rem;'>A. Employment Type Mix</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Set the approximate % of loans from each employment type. "
        "Percentages are normalised automatically."
    )

    emp_ft   = st.slider("Full-time (%)",    0, 100, 55, 5, key="emp_ft")
    emp_pt   = st.slider("Part-time (%)",    0, 100, 20, 5, key="emp_pt")
    emp_se   = st.slider("Self-employed (%)",0, 100, 15, 5, key="emp_se")
    emp_un   = st.slider("Unemployed (%)",   0, 100, 10, 5, key="emp_un")

    emp_raw_total = emp_ft + emp_pt + emp_se + emp_un
    if emp_raw_total == 0:
        emp_raw_total = 1  # guard against divide-by-zero
    emp_weights = {
        "Full-time":    emp_ft   / emp_raw_total,
        "Part-time":    emp_pt   / emp_raw_total,
        "Self-employed":emp_se   / emp_raw_total,
        "Unemployed":   emp_un   / emp_raw_total,
    }

    # Show normalised percentages so user sees what the model actually uses
    st.markdown(
        "<div style='font-size:0.78rem;color:#5a6a80;margin-top:0.1rem;"
        "margin-bottom:1rem;'>"
        f"Normalised: Full-time {emp_weights['Full-time']*100:.1f}% | "
        f"Part-time {emp_weights['Part-time']*100:.1f}% | "
        f"Self-emp {emp_weights['Self-employed']*100:.1f}% | "
        f"Unemployed {emp_weights['Unemployed']*100:.1f}%"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"<div style='color:{NAVY};font-weight:700;font-size:0.92rem;"
        f"margin-bottom:0.4rem;'>B. Loan Purpose Mix</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Set the approximate % of loans for each purpose. "
        "Percentages are normalised automatically."
    )

    pur_home  = st.slider("Home (%)",      0, 100, 30, 5, key="pur_home")
    pur_auto  = st.slider("Auto (%)",      0, 100, 20, 5, key="pur_auto")
    pur_educ  = st.slider("Education (%)", 0, 100, 20, 5, key="pur_educ")
    pur_biz   = st.slider("Business (%)",  0, 100, 20, 5, key="pur_biz")
    pur_other = st.slider("Other (%)",     0, 100, 10, 5, key="pur_other")

    pur_raw_total = pur_home + pur_auto + pur_educ + pur_biz + pur_other
    if pur_raw_total == 0:
        pur_raw_total = 1
    purpose_weights = {
        "Home":      pur_home  / pur_raw_total,
        "Auto":      pur_auto  / pur_raw_total,
        "Education": pur_educ  / pur_raw_total,
        "Business":  pur_biz   / pur_raw_total,
        "Other":     pur_other / pur_raw_total,
    }

    st.markdown(
        "<div style='font-size:0.78rem;color:#5a6a80;margin-top:0.1rem;'>"
        f"Normalised: Home {purpose_weights['Home']*100:.1f}% | "
        f"Auto {purpose_weights['Auto']*100:.1f}% | "
        f"Edu {purpose_weights['Education']*100:.1f}% | "
        f"Biz {purpose_weights['Business']*100:.1f}% | "
        f"Other {purpose_weights['Other']*100:.1f}%"
        "</div>",
        unsafe_allow_html=True,
    )

# ─── Right column: Credit score, DTI, Loan amount ───────────────────────────
with ctrl_right:

    st.markdown(
        f"<div style='color:{NAVY};font-weight:700;font-size:0.92rem;"
        f"margin-bottom:0.4rem;'>C. Credit Score Band</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Filter the scenario pool to applicants within this credit score "
        "range. Records outside the range are excluded from the scenario."
    )

    cs_range = st.slider(
        "Credit Score Range",
        min_value=CS_MIN,
        max_value=CS_MAX,
        value=(CS_MIN, CS_MAX),
        step=10,
        key="cs_range",
    )

    st.markdown(
        f"<div style='color:{NAVY};font-weight:700;font-size:0.92rem;"
        f"margin-top:1rem;margin-bottom:0.4rem;'>D. DTI Ratio Band</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Filter the scenario pool to applicants within this Debt-to-Income "
        "ratio range. Higher DTI bands historically correlate with elevated "
        "default rates."
    )

    dti_range = st.slider(
        "DTI Ratio Range",
        min_value=0.0,
        max_value=DTI_MAX,
        value=(0.0, DTI_MAX),
        step=0.05,
        key="dti_range",
        format="%.2f",
    )

    st.markdown(
        f"<div style='color:{NAVY};font-weight:700;font-size:0.92rem;"
        f"margin-top:1rem;margin-bottom:0.4rem;'>E. Average Loan Amount</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Set a target average loan amount for the scenario portfolio. "
        "Each row's LoanAmount is scaled proportionally so the portfolio "
        "mean matches this value. Feature correlations are preserved."
    )

    target_loan_mean = st.slider(
        "Target Average Loan Amount ($)",
        min_value=LA_MIN,
        max_value=LA_MAX,
        value=LA_MEAN,
        step=1000,
        key="loan_mean",
        format="$%d",
    )

    # ── Scenario size note ─────────────────────────────────────────────────
    st.markdown(
        "<div style='margin-top:1.2rem; padding:0.75rem 1rem; "
        "background:#e8f0fb; border-radius:8px; border:1px solid #bfd0ed;'>"
        "<span style='color:#0a2342;font-weight:700;font-size:0.82rem;'>"
        "How the scenario works</span><br>"
        "<span style='color:#5a6a80;font-size:0.78rem;'>"
        "The engine builds a synthetic portfolio of 5,000 records by "
        "resampling rows from the historical dataset that pass the credit "
        "score and DTI filters, weighted to match the selected employment "
        "and loan-purpose mix. The existing trained model then scores each "
        "record. Results are scenario <em>estimates</em>, not predictions "
        "of actual future outcomes."
        "</span></div>",
        unsafe_allow_html=True,
    )

st.markdown("<hr>", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — SCENARIO COMPUTATION
# ═════════════════════════════════════════════════════════════════════════════

section_title("3. Scenario Impact")
st.markdown("<div style='margin-bottom:0.5rem;'></div>", unsafe_allow_html=True)

# Build the scenario population using the controls above
with st.spinner("Building scenario population and running model inference..."):
    scenario_df = _build_scenario_population(
        df=df_full,
        emp_weights=emp_weights,
        purpose_weights=purpose_weights,
        cs_range=cs_range,
        dti_range=dti_range,
        target_loan_mean=target_loan_mean,
        target_n=5000,
        random_state=42,
    )

if scenario_df is None:
    st.warning(
        "The selected credit score and DTI filters returned fewer than 50 "
        "records from the historical dataset. This is not enough to build a "
        "reliable scenario estimate. Try widening the credit score or DTI "
        "range and re-running."
    )
    st.stop()

# Warn if the pool is small (but still usable)
if len(scenario_df) < 500:
    st.warning(
        f"The scenario population contains only {len(scenario_df):,} records "
        "after filtering and resampling. Estimates may be less stable. "
        "Consider widening the credit score or DTI range."
    )

# Run model inference on the scenario population
with st.spinner("Scoring scenario records..."):
    scenario = _run_scenario_inference(scenario_df, pipeline, DECISION_THRESHOLD)

# ── Comparison table ─────────────────────────────────────────────────────────
st.markdown(
    "<p style='color:#5a6a80;font-size:0.85rem;margin-bottom:0.6rem;'>"
    "The table below compares historical baseline values (from the real "
    "dataset) with scenario estimates (from the trained model applied to the "
    "resampled hypothetical portfolio). <strong>Scenario values are estimates."
    "</strong></p>",
    unsafe_allow_html=True,
)

# ── Metric: how the scenario scale factor maps to the baseline portfolio
# The scenario uses target_n=5000 rows; to compare dollar values fairly
# we scale the scenario dollar totals to the baseline portfolio size.
# However, for rates (default_rate_pct, exposure_rate_pct) no scaling needed.
_scale = baseline["total_loans"] / scenario["total_loans"]
scaled_def_value  = scenario["defaulted_loan_value"]   * _scale
scaled_total_val  = scenario["total_loan_value"]        * _scale
# Re-compute the exposure rate on the scaled values (should be identical
# to unscaled since both numerator and denominator scale by the same factor).
scaled_exposure   = round(scaled_def_value / scaled_total_val * 100, 2) if scaled_total_val > 0 else 0.0
scaled_def_count  = int(round(scenario["default_count"] * _scale))

# Build comparison as a DataFrame for show_table()
comparison_rows = [
    {
        "Metric":              "Default Rate (%)",
        "Historical Baseline": f"{baseline['default_rate_pct']:.2f}%",
        "Scenario Estimate":   f"{scenario['default_rate_pct']:.2f}%",
        "Absolute Change":     f"{scenario['default_rate_pct'] - baseline['default_rate_pct']:+.2f} pp",
        "Relative Change":     f"{(scenario['default_rate_pct'] - baseline['default_rate_pct']) / baseline['default_rate_pct'] * 100:+.1f}%",
    },
    {
        "Metric":              "Defaulted Loan Value (scaled)",
        "Historical Baseline": f"${baseline['defaulted_loan_value']/1e9:.3f}B",
        "Scenario Estimate":   f"${scaled_def_value/1e9:.3f}B",
        "Absolute Change":     f"${(scaled_def_value - baseline['defaulted_loan_value'])/1e6:+.1f}M",
        "Relative Change":     f"{(scaled_def_value - baseline['defaulted_loan_value']) / baseline['defaulted_loan_value'] * 100:+.1f}%",
    },
    {
        "Metric":              "Exposure Rate (defaulted $ / total $)",
        "Historical Baseline": f"{baseline['exposure_rate_pct']:.2f}%",
        "Scenario Estimate":   f"{scaled_exposure:.2f}%",
        "Absolute Change":     f"{scaled_exposure - baseline['exposure_rate_pct']:+.2f} pp",
        "Relative Change":     f"{(scaled_exposure - baseline['exposure_rate_pct']) / baseline['exposure_rate_pct'] * 100:+.1f}%",
    },
    {
        "Metric":              "Expected Default Count (scaled)",
        "Historical Baseline": f"{baseline['default_count']:,}",
        "Scenario Estimate":   f"{scaled_def_count:,}",
        "Absolute Change":     f"{scaled_def_count - baseline['default_count']:+,}",
        "Relative Change":     f"{(scaled_def_count - baseline['default_count']) / baseline['default_count'] * 100:+.1f}%",
    },
    {
        "Metric":              "Avg Loan Amount (scenario)",
        "Historical Baseline": f"${baseline['avg_loan_amount']:,.0f}",
        "Scenario Estimate":   f"${scenario['avg_loan_amount']:,.0f}",
        "Absolute Change":     f"${scenario['avg_loan_amount'] - baseline['avg_loan_amount']:+,.0f}",
        "Relative Change":     f"{(scenario['avg_loan_amount'] - baseline['avg_loan_amount']) / baseline['avg_loan_amount'] * 100:+.1f}%",
    },
]

comparison_df = pd.DataFrame(comparison_rows)
show_table(comparison_df)

st.markdown("<div style='margin-top:0.6rem;'></div>", unsafe_allow_html=True)
st.caption(
    "pp = percentage points. 'Scaled' metrics normalise the scenario "
    f"({scenario['total_loans']:,} synthetic records) to the baseline "
    f"portfolio size ({baseline['total_loans']:,} loans) for fair comparison. "
    "Rates are not scaled."
)

st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)

# ── Comparison charts ────────────────────────────────────────────────────────
chart_left, chart_right = st.columns(2)

# Chart colours — intentionally minimal (no repeated palettes from other pages)
CHART_BASELINE = NAVY
CHART_SCENARIO = AMBER

# ── Chart 1: Default Rate comparison ────────────────────────────────────────
with chart_left:
    fig1, ax1 = plt.subplots(figsize=(5.5, 3.6), facecolor="white")
    ax1.set_facecolor("white")

    bars1 = ax1.bar(
        ["Historical\nBaseline", "Scenario\nEstimate"],
        [baseline["default_rate_pct"], scenario["default_rate_pct"]],
        color=[CHART_BASELINE, CHART_SCENARIO],
        width=0.45,
        edgecolor="white",
        linewidth=0.8,
    )

    # Value labels
    for bar in bars1:
        h = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.15,
            f"{h:.2f}%",
            ha="center", va="bottom",
            fontsize=10, fontweight="700", color=NAVY,
        )

    ax1.set_ylabel("Default Rate (%)", fontsize=9, color=MUTED)
    ax1.set_title("Default Rate — Baseline vs Scenario",
                  fontsize=10.5, fontweight="700", color=NAVY, pad=8)
    ax1.set_ylim(0, max(baseline["default_rate_pct"],
                        scenario["default_rate_pct"]) * 1.30 + 1)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.spines[["left", "bottom"]].set_color(BORDER)
    ax1.tick_params(colors=MUTED, labelsize=9)

    # Annotation: delta
    delta_rate = scenario["default_rate_pct"] - baseline["default_rate_pct"]
    arrow_color = RED if delta_rate > 0 else GREEN
    ax1.annotate(
        f"{delta_rate:+.2f} pp",
        xy=(1, scenario["default_rate_pct"]),
        xytext=(1.28, (baseline["default_rate_pct"] + scenario["default_rate_pct"]) / 2),
        fontsize=9, fontweight="700", color=arrow_color,
        arrowprops=dict(arrowstyle="-", color=arrow_color, lw=1.2),
    )

    fig1.tight_layout()
    st.pyplot(fig1)
    plt.close(fig1)

# ── Chart 2: Defaulted Loan Value comparison ─────────────────────────────────
with chart_right:
    fig2, ax2 = plt.subplots(figsize=(5.5, 3.6), facecolor="white")
    ax2.set_facecolor("white")

    val_base_b = baseline["defaulted_loan_value"] / 1e9
    val_scen_b = scaled_def_value / 1e9

    bars2 = ax2.bar(
        ["Historical\nBaseline", "Scenario\nEstimate"],
        [val_base_b, val_scen_b],
        color=[CHART_BASELINE, CHART_SCENARIO],
        width=0.45,
        edgecolor="white",
        linewidth=0.8,
    )

    for bar in bars2:
        h = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            h + val_base_b * 0.01,
            f"${h:.3f}B",
            ha="center", va="bottom",
            fontsize=10, fontweight="700", color=NAVY,
        )

    ax2.set_ylabel("Defaulted Loan Value ($B)", fontsize=9, color=MUTED)
    ax2.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"${x:.2f}B")
    )
    ax2.set_title("Defaulted Loan Value — Baseline vs Scenario",
                  fontsize=10.5, fontweight="700", color=NAVY, pad=8)
    ax2.set_ylim(0, max(val_base_b, val_scen_b) * 1.30 + 0.05)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.spines[["left", "bottom"]].set_color(BORDER)
    ax2.tick_params(colors=MUTED, labelsize=9)

    delta_val_pct = (scaled_def_value - baseline["defaulted_loan_value"]) / baseline["defaulted_loan_value"] * 100
    arrow_color2  = RED if delta_val_pct > 0 else GREEN
    ax2.annotate(
        f"{delta_val_pct:+.1f}%",
        xy=(1, val_scen_b),
        xytext=(1.28, (val_base_b + val_scen_b) / 2),
        fontsize=9, fontweight="700", color=arrow_color2,
        arrowprops=dict(arrowstyle="-", color=arrow_color2, lw=1.2),
    )

    fig2.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

st.markdown("<hr>", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — SCENARIO INTERPRETATION
# ═════════════════════════════════════════════════════════════════════════════

section_title("4. Scenario Interpretation")
st.markdown("<div style='margin-bottom:0.4rem;'></div>", unsafe_allow_html=True)

# Generate a short, fact-based interpretation from the computed numbers only.
# No recommendations — only description of what the numbers show.
_dr_delta    = scenario["default_rate_pct"] - baseline["default_rate_pct"]
_val_delta   = scaled_def_value - baseline["defaulted_loan_value"]
_exp_delta   = scaled_exposure - baseline["exposure_rate_pct"]

_dr_dir   = "increases" if _dr_delta   > 0 else ("decreases" if _dr_delta < 0   else "remains unchanged")
_val_dir  = "increases" if _val_delta  > 0 else ("decreases" if _val_delta < 0  else "remains unchanged")
_exp_dir  = "increases" if _exp_delta  > 0 else ("decreases" if _exp_delta < 0  else "remains unchanged")

_interp_lines = [
    f"Under the selected scenario, the estimated default rate "
    f"<strong>{_dr_dir}</strong> compared with the historical baseline "
    f"({baseline['default_rate_pct']:.2f}% baseline vs "
    f"{scenario['default_rate_pct']:.2f}% scenario, "
    f"a change of {_dr_delta:+.2f} percentage points).",

    f"Estimated defaulted loan value "
    f"<strong>{_val_dir}</strong> "
    f"(baseline ${baseline['defaulted_loan_value']/1e9:.3f}B vs "
    f"scaled scenario ${scaled_def_value/1e9:.3f}B, "
    f"a change of ${_val_delta/1e6:+.1f}M).",

    f"The portfolio exposure rate "
    f"<strong>{_exp_dir}</strong> "
    f"({baseline['exposure_rate_pct']:.2f}% baseline vs "
    f"{scaled_exposure:.2f}% scenario, "
    f"{_exp_delta:+.2f} percentage points).",
]

# Add a note about which controls drove the change
_driving_factors = []
if emp_weights.get("Unemployed", 0) > 0.15:
    _driving_factors.append(
        "a high proportion of Unemployed borrowers (above 15%), "
        "which historically carries elevated default rates in this dataset"
    )
if cs_range[1] < 550:
    _driving_factors.append(
        f"a restricted credit score ceiling of {cs_range[1]}, "
        "which filters to higher-risk borrowers"
    )
if dti_range[0] > 0.5:
    _driving_factors.append(
        f"a minimum DTI of {dti_range[0]:.2f}, "
        "selecting borrowers with higher debt burden"
    )
if target_loan_mean > baseline["avg_loan_amount"] * 1.20:
    _driving_factors.append(
        f"an elevated target loan mean (${target_loan_mean:,}) above the "
        f"historical average (${baseline['avg_loan_amount']:,.0f})"
    )

if _driving_factors:
    _interp_lines.append(
        "Notable scenario characteristics include: "
        + "; ".join(_driving_factors) + "."
    )

_interp_html = "".join(
    f"<p style='margin:0 0 0.55rem 0; color:#0d1b2e; font-size:0.88rem;'>"
    f"{line}</p>"
    for line in _interp_lines
)

st.markdown(
    f"<div style='background:#ffffff; border:1px solid {BORDER}; "
    f"border-left:4px solid {BLUE}; border-radius:6px; "
    f"padding:1rem 1.25rem; margin-bottom:0.8rem;'>"
    f"{_interp_html}"
    f"</div>",
    unsafe_allow_html=True,
)

st.caption(
    "Interpretation is generated automatically from the computed scenario "
    "metrics. It describes calculated results only. It does not constitute "
    "a recommendation or prediction of actual future outcomes."
)

st.markdown("<hr>", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5 — METHODOLOGY NOTE
# ═════════════════════════════════════════════════════════════════════════════

section_title("5. Methodology")
st.markdown("<div style='margin-bottom:0.4rem;'></div>", unsafe_allow_html=True)

with st.expander("Read the scenario methodology", expanded=False):
    st.markdown(
        """
**Historical baseline**

All baseline values are computed directly from the cleaned historical dataset
(`Loan_default.csv`, 255,347 records) using the same `compute_portfolio_kpis()`
function used by the Overview page. They are not estimated — they are observed
facts from the data.

**Scenario population construction**

1. Records from the cleaned dataset are filtered to those whose `CreditScore`
   and `DTIRatio` fall within the selected ranges.
2. From this eligible pool, records are resampled (with replacement when
   needed) so that `EmploymentType` proportions match the user-specified mix.
3. A second pass resamples by `LoanPurpose` to match the selected purpose mix.
4. If either resample yields fewer than 50 records, the scenario is aborted and
   the user is asked to widen the filters.
5. The portfolio is trimmed or padded to exactly 5,000 records.
6. `LoanAmount` values are scaled by a constant factor so the population mean
   equals the user-selected target. All other features are unchanged.

**Model scoring**

The scenario population (5,000 records, 16 features) is passed through the
*existing* trained scikit-learn pipeline (`pipeline.predict_proba()`). The
pipeline applies the same `StandardScaler` and `OrdinalEncoder` that were
fitted during original model training. The `DECISION_THRESHOLD = 0.40` is
unchanged.

No model retraining occurs. No model hyperparameters are changed.
No model artifact files are written.

**Scaling for fair comparison**

Because the scenario uses 5,000 records while the historical portfolio has
~255,000 loans, dollar totals (defaulted loan value, total loan value,
default count) are scaled by `baseline_size / scenario_size` before
being compared. Rates (default rate %, exposure rate %) are not scaled —
they are directly comparable.

**Limitations**

- All scenario records come from the real dataset, so unusual borrower
  profiles not present in the data cannot be simulated.
- The model was trained on the existing data distribution; performance may
  degrade if the scenario composition is far from the training distribution.
- `LoanAmount` scaling is a proportional shift; it does not model the
  downstream effect of loan size on a borrower's ability to repay.
- Scenario estimates should not be interpreted as predictions of actual
  future outcomes.
        """,
        unsafe_allow_html=False,
    )
