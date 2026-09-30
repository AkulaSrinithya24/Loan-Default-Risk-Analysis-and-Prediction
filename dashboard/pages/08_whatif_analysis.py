"""
dashboard/pages/08_whatif_analysis.py
---------------------------------------
What-If Scenario Analysis page.

PURPOSE
-------
Let the user enter a BASELINE loan application and a SCENARIO loan
application side-by-side, then compare the model's predicted default
probability for each.  This is individual-applicant what-if analysis,
not portfolio-level analysis.

WHAT THIS PAGE DOES NOT DUPLICATE
----------------------------------
- 05_prediction.py  — single prediction form + verdict + risk checks
  (this page is two side-by-side forms with a diff table; no
   data-quality checks, no risk indicators, no application review
   section — those are prediction-page concerns)
- 06_portfolio_exposure.py — portfolio dollar-exposure analysis
- 07_scenario_analysis.py  — portfolio-level resampling scenario

MODEL USAGE
-----------
Identical to 05_prediction.py:
  - Loads the same model file (MODEL_PATH, fallback DEPLOYMENT_MODEL_PATH)
  - Builds a 16-column DataFrame in the same format
  - Calls pipeline.predict_proba(df)[0, 1]
  - Uses the same DECISION_THRESHOLD = 0.40
  - Uses the same risk-tier thresholds (0.20 / 0.40 / 0.60)

No model is retrained.  No model file is written.  No sklearn or scipy
imports are made directly -- only joblib.load() via the pipeline.

FILES CREATED  : dashboard/pages/08_whatif_analysis.py
FILES MODIFIED : none
"""

# ── Path setup (identical pattern used by every page) ─────────────────────────
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Project imports ────────────────────────────────────────────────────────────
from src.config import (
    MODEL_PATH,
    DEPLOYMENT_MODEL_PATH,
    DECISION_THRESHOLD,
)
from dashboard.theme import page_header, section_title, apply_theme

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="What-If Analysis",
    page_icon="🔄",
    layout="wide",
)

page_header(
    "What-If Scenario Analysis",
    "Enter a baseline loan application and a modified scenario. "
    "Compare the model's predicted default probability for each.",
)

# ── Palette (project-consistent) ──────────────────────────────────────────────
NAVY   = "#0a2342"
BLUE   = "#1a56a0"
RED    = "#b91c1c"
GREEN  = "#1a7a4a"
AMBER  = "#b45309"
MUTED  = "#5a6a80"
BORDER = "#dce3ed"

# ── Model loading — identical to 05_prediction.py ─────────────────────────────
if MODEL_PATH.exists():
    _model_path = MODEL_PATH
elif DEPLOYMENT_MODEL_PATH.exists():
    _model_path = DEPLOYMENT_MODEL_PATH
    st.info(
        "The deployment model is currently being used. "
        "Prediction functionality remains fully available."
    )
else:
    st.error(
        "No trained model file found. "
        "Run `python run_pipeline.py` to generate one."
    )
    st.stop()


@st.cache_resource
def _load_pipeline():
    """Load the trained pipeline — cached, read-only, never modified."""
    import joblib
    return joblib.load(_model_path)


pipeline = _load_pipeline()


# ── Helper: build the same input DataFrame used by 05_prediction.py ───────────
def _make_input_df(
    age, income, loan_amount, credit_score, months_employed,
    num_credit_lines, interest_rate, loan_term, dti_ratio,
    education, employment_type, marital_status, loan_purpose,
    has_mortgage, has_dependents, has_cosigner,
) -> pd.DataFrame:
    """
    Return a single-row DataFrame in the exact format expected by the
    trained pipeline.  Column names, types, and binary encoding are
    identical to 05_prediction.py.
    """
    return pd.DataFrame([{
        "Age":            age,
        "Income":         income,
        "LoanAmount":     loan_amount,
        "CreditScore":    credit_score,
        "MonthsEmployed": months_employed,
        "NumCreditLines": num_credit_lines,
        "InterestRate":   interest_rate,
        "LoanTerm":       loan_term,
        "DTIRatio":       dti_ratio,
        "Education":      education,
        "EmploymentType": employment_type,
        "MaritalStatus":  marital_status,
        "HasMortgage":    1 if has_mortgage  == "Yes" else 0,
        "HasDependents":  1 if has_dependents == "Yes" else 0,
        "LoanPurpose":    loan_purpose,
        "HasCoSigner":    1 if has_cosigner  == "Yes" else 0,
    }])


# ── Helper: risk tier label — identical to 05_prediction.py ───────────────────
def _risk_tier(prob: float) -> str:
    if prob >= 0.60:
        return "Very High"
    if prob >= 0.40:
        return "High"
    if prob >= 0.20:
        return "Medium"
    return "Low"


# ── Helper: verdict label ──────────────────────────────────────────────────────
def _verdict(prob: float) -> str:
    return "DEFAULT" if prob >= DECISION_THRESHOLD else "NO DEFAULT"


# ── Helper: tier colour for display ───────────────────────────────────────────
def _tier_color(tier: str) -> str:
    return {
        "Very High": RED,
        "High":      AMBER,
        "Medium":    "#ca8a04",
        "Low":       GREEN,
    }.get(tier, NAVY)


# =============================================================================
# DISCLAIMER BANNER
# =============================================================================

st.markdown(
    f"""
    <div style="background:{BLUE}10; border:1px solid {BLUE}40;
                border-left:4px solid {BLUE}; border-radius:6px;
                padding:0.75rem 1rem; margin-bottom:1rem;
                color:{NAVY}; font-size:0.84rem;">
      <strong>Model-based scenario analysis.</strong>
      Both results are generated by the existing trained machine-learning model.
      Changing an input value does not imply that the variable <em>causes</em>
      the default probability to change in the real world. This tool
      illustrates the model's response to different input combinations.
    </div>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# INPUT FORMS — SIDE BY SIDE
# =============================================================================

section_title("Loan Application Inputs")
st.markdown(
    "<p style='color:#5a6a80;font-size:0.85rem;margin-bottom:0.7rem;'>"
    "Fill in the <strong>Baseline</strong> (left) and the "
    "<strong>Scenario</strong> (right). Modify only the variables you want "
    "to change. Then click <em>Run Comparison</em>.</p>",
    unsafe_allow_html=True,
)

col_base, col_divider, col_scen = st.columns([10, 1, 10])

# ── Column headers ─────────────────────────────────────────────────────────────
with col_base:
    st.markdown(
        f"<div style='background:{NAVY};color:white;font-weight:700;"
        f"font-size:0.9rem;padding:0.5rem 0.9rem;border-radius:6px;"
        f"margin-bottom:0.7rem;text-align:center;'>BASELINE</div>",
        unsafe_allow_html=True,
    )

with col_divider:
    st.markdown(
        "<div style='border-left:1px solid #dce3ed;height:900px;"
        "margin:0 auto;width:1px;'></div>",
        unsafe_allow_html=True,
    )

with col_scen:
    st.markdown(
        f"<div style='background:{AMBER};color:white;font-weight:700;"
        f"font-size:0.9rem;padding:0.5rem 0.9rem;border-radius:6px;"
        f"margin-bottom:0.7rem;text-align:center;'>SCENARIO</div>",
        unsafe_allow_html=True,
    )

# ── Borrower profile ───────────────────────────────────────────────────────────
with col_base:
    st.markdown("**Borrower Profile**")
    b_age       = st.slider("Age",             18, 69,   35,   key="b_age")
    b_income    = st.number_input("Annual Income ($)", 15000, 150000, 65000, 1000, key="b_income")
    b_cs        = st.slider("Credit Score",    300, 849, 600,  key="b_cs")
    b_me        = st.slider("Months Employed", 0,  119,  48,   key="b_me")
    b_edu       = st.selectbox("Education",
                    ["High School", "Bachelor's", "Master's", "PhD"], key="b_edu")
    b_emp       = st.selectbox("Employment Type",
                    ["Full-time", "Part-time", "Self-employed", "Unemployed"], key="b_emp")
    b_mar       = st.selectbox("Marital Status",
                    ["Single", "Married", "Divorced"], key="b_mar")

with col_scen:
    st.markdown("**Borrower Profile**")
    s_age       = st.slider("Age",             18, 69,   b_age,  key="s_age")
    s_income    = st.number_input("Annual Income ($)", 15000, 150000, b_income, 1000, key="s_income")
    s_cs        = st.slider("Credit Score",    300, 849, b_cs,   key="s_cs")
    s_me        = st.slider("Months Employed", 0,  119,  b_me,   key="s_me")
    s_edu       = st.selectbox("Education",
                    ["High School", "Bachelor's", "Master's", "PhD"],
                    index=["High School", "Bachelor's", "Master's", "PhD"].index(b_edu),
                    key="s_edu")
    s_emp       = st.selectbox("Employment Type",
                    ["Full-time", "Part-time", "Self-employed", "Unemployed"],
                    index=["Full-time", "Part-time", "Self-employed", "Unemployed"].index(b_emp),
                    key="s_emp")
    s_mar       = st.selectbox("Marital Status",
                    ["Single", "Married", "Divorced"],
                    index=["Single", "Married", "Divorced"].index(b_mar),
                    key="s_mar")

# ── Loan details ───────────────────────────────────────────────────────────────
with col_base:
    st.markdown("**Loan Details**")
    b_la        = st.number_input("Loan Amount ($)", 5000, 250000, 100000, 1000, key="b_la")
    b_ir        = st.slider("Interest Rate (%)", 2.0, 25.0, 10.0, 0.1, key="b_ir")
    b_lt        = st.selectbox("Loan Term (months)", [12, 24, 36, 48, 60], index=2, key="b_lt")
    b_dti       = st.slider("DTI Ratio",         0.10, 0.90, 0.40, 0.01, key="b_dti")
    b_ncl       = st.slider("Num Credit Lines",  1, 4, 2, key="b_ncl")
    b_pur       = st.selectbox("Loan Purpose",
                    ["Auto", "Business", "Education", "Home", "Other"], key="b_pur")

with col_scen:
    st.markdown("**Loan Details**")
    s_la        = st.number_input("Loan Amount ($)", 5000, 250000, b_la,  1000, key="s_la")
    s_ir        = st.slider("Interest Rate (%)", 2.0, 25.0, b_ir,  0.1,  key="s_ir")
    _lt_opts    = [12, 24, 36, 48, 60]
    s_lt        = st.selectbox("Loan Term (months)", _lt_opts,
                    index=_lt_opts.index(b_lt), key="s_lt")
    s_dti       = st.slider("DTI Ratio",         0.10, 0.90, b_dti, 0.01, key="s_dti")
    s_ncl       = st.slider("Num Credit Lines",  1, 4, b_ncl,       key="s_ncl")
    _pur_opts   = ["Auto", "Business", "Education", "Home", "Other"]
    s_pur       = st.selectbox("Loan Purpose", _pur_opts,
                    index=_pur_opts.index(b_pur), key="s_pur")

# ── Additional flags ───────────────────────────────────────────────────────────
with col_base:
    st.markdown("**Additional Information**")
    st.markdown("**Existing Mortgage**")
    b_mort = st.radio("Mortgage?", ["Yes", "No"], index=1,
                      label_visibility="collapsed", key="b_mort")
    st.markdown("**Dependents**")
    b_dep  = st.radio("Dependents?", ["Yes", "No"], index=1,
                      label_visibility="collapsed", key="b_dep")
    st.markdown("**Co-Signer**")
    b_cos  = st.radio("Co-Signer?", ["Yes", "No"], index=1,
                      label_visibility="collapsed", key="b_cos")

with col_scen:
    st.markdown("**Additional Information**")
    st.markdown("**Existing Mortgage**")
    s_mort = st.radio("Mortgage?", ["Yes", "No"],
                      index=["Yes", "No"].index(b_mort),
                      label_visibility="collapsed", key="s_mort")
    st.markdown("**Dependents**")
    s_dep  = st.radio("Dependents?", ["Yes", "No"],
                      index=["Yes", "No"].index(b_dep),
                      label_visibility="collapsed", key="s_dep")
    st.markdown("**Co-Signer**")
    s_cos  = st.radio("Co-Signer?", ["Yes", "No"],
                      index=["Yes", "No"].index(b_cos),
                      label_visibility="collapsed", key="s_cos")

# =============================================================================
# RUN BUTTON
# =============================================================================

st.markdown("")
run = st.button(
    "Run Comparison",
    type="primary",
    use_container_width=True,
    key="run_comparison",
)

# =============================================================================
# RESULTS
# =============================================================================

if run:

    st.markdown("<hr>", unsafe_allow_html=True)
    section_title("Comparison Results")
    st.caption(
        "Results are generated by the trained machine-learning model. "
        "They represent the model's output for the given inputs, not a "
        "real-world prediction of default."
    )

    # ── Build input DataFrames ─────────────────────────────────────────────
    df_base = _make_input_df(
        b_age, b_income, b_la, b_cs, b_me, b_ncl, b_ir, b_lt, b_dti,
        b_edu, b_emp, b_mar, b_pur, b_mort, b_dep, b_cos,
    )
    df_scen = _make_input_df(
        s_age, s_income, s_la, s_cs, s_me, s_ncl, s_ir, s_lt, s_dti,
        s_edu, s_emp, s_mar, s_pur, s_mort, s_dep, s_cos,
    )

    # ── Run predictions ────────────────────────────────────────────────────
    prob_base = float(pipeline.predict_proba(df_base)[0, 1])
    prob_scen = float(pipeline.predict_proba(df_scen)[0, 1])

    tier_base    = _risk_tier(prob_base)
    tier_scen    = _risk_tier(prob_scen)
    verdict_base = _verdict(prob_base)
    verdict_scen = _verdict(prob_scen)
    delta_pp     = prob_scen - prob_base          # in probability units (0–1)
    delta_pct_pt = delta_pp * 100                 # in percentage points

    # ── Top metric tiles ───────────────────────────────────────────────────
    m1, m2, m3 = st.columns(3)

    m1.metric(
        "Baseline Default Probability",
        f"{prob_base * 100:.1f}%",
        delta=f"{verdict_base} | {tier_base}",
        delta_color="off",
    )
    m2.metric(
        "Scenario Default Probability",
        f"{prob_scen * 100:.1f}%",
        delta=f"{verdict_scen} | {tier_scen}",
        delta_color="off",
    )
    delta_sign = "+" if delta_pct_pt >= 0 else ""
    m3.metric(
        "Probability Change",
        f"{delta_sign}{delta_pct_pt:.1f} pp",
        delta=("Increased" if delta_pp > 0.001
               else "Decreased" if delta_pp < -0.001
               else "Unchanged"),
        delta_color="inverse",
    )

    st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)

    # ── Detailed comparison table ──────────────────────────────────────────
    section_title("Side-by-Side Comparison")

    # Colour-coded HTML for verdict cells
    def _verdict_html(v):
        c = RED if v == "DEFAULT" else GREEN
        return f"<span style='color:{c};font-weight:700;'>{v}</span>"

    def _tier_html(t):
        c = _tier_color(t)
        return f"<span style='color:{c};font-weight:700;'>{t}</span>"

    comparison_rows = [
        ("Default Probability",
         f"{prob_base * 100:.2f}%",
         f"{prob_scen * 100:.2f}%",
         f"{delta_sign}{delta_pct_pt:.2f} pp"),
        ("Model Verdict",
         verdict_base,
         verdict_scen,
         "Changed" if verdict_base != verdict_scen else "No change"),
        ("Risk Tier",
         tier_base,
         tier_scen,
         "Changed" if tier_base != tier_scen else "No change"),
    ]

    # Build HTML table directly (same approach as show_table in theme.py)
    rows_html = ""
    for metric, bval, sval, chg in comparison_rows:
        rows_html += (
            f"<tr>"
            f"<td style='padding:0.55rem 0.9rem;font-weight:600;"
            f"color:{NAVY};border-bottom:1px solid {BORDER};'>{metric}</td>"
            f"<td style='padding:0.55rem 0.9rem;border-bottom:1px solid {BORDER};"
            f"color:{NAVY};'>{bval}</td>"
            f"<td style='padding:0.55rem 0.9rem;border-bottom:1px solid {BORDER};"
            f"color:{AMBER};'>{sval}</td>"
            f"<td style='padding:0.55rem 0.9rem;border-bottom:1px solid {BORDER};"
            f"color:{MUTED};font-style:italic;'>{chg}</td>"
            f"</tr>"
        )

    table_html = (
        "<div style='overflow-x:auto;border:1px solid #dce3ed;"
        "border-radius:8px;margin-bottom:0.9rem;'>"
        "<table style='width:100%;border-collapse:collapse;"
        "font-size:0.87rem;background:#fff;'>"
        "<thead><tr>"
        f"<th style='padding:0.55rem 0.9rem;background:{NAVY};color:white;"
        f"font-weight:700;text-align:left;'>Metric</th>"
        f"<th style='padding:0.55rem 0.9rem;background:{NAVY};color:white;"
        f"font-weight:700;text-align:left;'>Baseline</th>"
        f"<th style='padding:0.55rem 0.9rem;background:{AMBER};color:white;"
        f"font-weight:700;text-align:left;'>Scenario</th>"
        f"<th style='padding:0.55rem 0.9rem;background:{NAVY};color:white;"
        f"font-weight:700;text-align:left;'>Change</th>"
        "</tr></thead>"
        f"<tbody>{rows_html}</tbody>"
        "</table></div>"
    )
    st.markdown(table_html, unsafe_allow_html=True)

    # ── Input differences table ────────────────────────────────────────────
    # Show only variables that actually changed — makes the scenario legible
    all_inputs = [
        ("Age",               b_age,    s_age),
        ("Annual Income ($)", b_income, s_income),
        ("Loan Amount ($)",   b_la,     s_la),
        ("Credit Score",      b_cs,     s_cs),
        ("Months Employed",   b_me,     s_me),
        ("Num Credit Lines",  b_ncl,    s_ncl),
        ("Interest Rate (%)", b_ir,     s_ir),
        ("Loan Term (mo)",    b_lt,     s_lt),
        ("DTI Ratio",         b_dti,    s_dti),
        ("Education",         b_edu,    s_edu),
        ("Employment Type",   b_emp,    s_emp),
        ("Marital Status",    b_mar,    s_mar),
        ("Loan Purpose",      b_pur,    s_pur),
        ("Has Mortgage",      b_mort,   s_mort),
        ("Has Dependents",    b_dep,    s_dep),
        ("Has Co-Signer",     b_cos,    s_cos),
    ]
    changed = [(name, bv, sv) for name, bv, sv in all_inputs if bv != sv]

    if changed:
        with st.expander("Variables changed in this scenario", expanded=True):
            diff_rows_html = ""
            for name, bv, sv in changed:
                diff_rows_html += (
                    f"<tr>"
                    f"<td style='padding:0.45rem 0.8rem;border-bottom:1px solid {BORDER};"
                    f"color:{NAVY};font-weight:600;'>{name}</td>"
                    f"<td style='padding:0.45rem 0.8rem;border-bottom:1px solid {BORDER};"
                    f"color:{NAVY};'>{bv}</td>"
                    f"<td style='padding:0.45rem 0.8rem;border-bottom:1px solid {BORDER};"
                    f"color:{AMBER};'>{sv}</td>"
                    f"</tr>"
                )
            diff_html = (
                "<div style='overflow-x:auto;border:1px solid #dce3ed;"
                "border-radius:6px;'>"
                "<table style='width:100%;border-collapse:collapse;"
                "font-size:0.84rem;background:#fff;'>"
                "<thead><tr>"
                f"<th style='padding:0.45rem 0.8rem;background:{NAVY};color:white;'>Variable</th>"
                f"<th style='padding:0.45rem 0.8rem;background:{NAVY};color:white;'>Baseline</th>"
                f"<th style='padding:0.45rem 0.8rem;background:{AMBER};color:white;'>Scenario</th>"
                "</tr></thead>"
                f"<tbody>{diff_rows_html}</tbody>"
                "</table></div>"
            )
            st.markdown(diff_html, unsafe_allow_html=True)
    else:
        st.info(
            "No inputs were changed between baseline and scenario. "
            "Modify at least one variable in the Scenario column to see a difference."
        )

    st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)

    # ── Comparison bar chart ───────────────────────────────────────────────
    section_title("Probability Comparison Chart")

    fig, ax = plt.subplots(figsize=(5, 3.2), facecolor="white")
    ax.set_facecolor("white")

    bars = ax.bar(
        ["Baseline", "Scenario"],
        [prob_base * 100, prob_scen * 100],
        color=[NAVY, AMBER],
        width=0.40,
        edgecolor="white",
        linewidth=0.8,
    )

    # Value labels above each bar
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.5,
            f"{h:.1f}%",
            ha="center", va="bottom",
            fontsize=11, fontweight="700",
            color=NAVY,
        )

    # Threshold line
    ax.axhline(
        y=DECISION_THRESHOLD * 100,
        color=RED, linewidth=1.2, linestyle="--", alpha=0.7,
    )
    ax.text(
        1.45, DECISION_THRESHOLD * 100 + 0.5,
        f"Threshold {DECISION_THRESHOLD * 100:.0f}%",
        fontsize=8, color=RED, va="bottom",
    )

    ax.set_ylim(0, max(prob_base * 100, prob_scen * 100, DECISION_THRESHOLD * 100) * 1.35 + 5)
    ax.set_ylabel("Default Probability (%)", fontsize=9, color=MUTED)
    ax.set_title("Baseline vs Scenario — Default Probability",
                 fontsize=10.5, fontweight="700", color=NAVY, pad=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(BORDER)
    ax.tick_params(colors=MUTED, labelsize=9)

    # Legend patches
    ax.legend(
        handles=[
            mpatches.Patch(color=NAVY,  label="Baseline"),
            mpatches.Patch(color=AMBER, label="Scenario"),
        ],
        fontsize=8, framealpha=0,
    )

    fig.tight_layout()

    # Centre the chart — don't stretch it full-width
    _, chart_col, _ = st.columns([1, 2, 1])
    with chart_col:
        st.pyplot(fig)
    plt.close(fig)

    # ── Interpretation ─────────────────────────────────────────────────────
    st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
    section_title("Interpretation")

    if abs(delta_pct_pt) < 0.05:
        direction_text = (
            "The scenario produces the same predicted default probability "
            "as the baseline."
        )
    elif delta_pp > 0:
        direction_text = (
            f"The scenario produces a <strong>higher</strong> predicted "
            f"default probability than the baseline "
            f"({prob_base*100:.1f}% baseline vs {prob_scen*100:.1f}% scenario, "
            f"a change of {delta_sign}{delta_pct_pt:.2f} percentage points)."
        )
    else:
        direction_text = (
            f"The scenario produces a <strong>lower</strong> predicted "
            f"default probability than the baseline "
            f"({prob_base*100:.1f}% baseline vs {prob_scen*100:.1f}% scenario, "
            f"a change of {delta_pct_pt:.2f} percentage points)."
        )

    # Verdict change note
    if verdict_base != verdict_scen:
        verdict_note = (
            f" The model verdict changes from "
            f"<strong>{verdict_base}</strong> to "
            f"<strong>{verdict_scen}</strong>."
        )
    else:
        verdict_note = (
            f" The model verdict remains <strong>{verdict_base}</strong> "
            "in both cases."
        )

    # Risk tier change note
    if tier_base != tier_scen:
        tier_note = (
            f" The risk tier changes from "
            f"<strong>{tier_base}</strong> to "
            f"<strong>{tier_scen}</strong>."
        )
    else:
        tier_note = (
            f" The risk tier remains <strong>{tier_base}</strong> "
            "in both cases."
        )

    st.markdown(
        f"<div style='background:#fff;border:1px solid {BORDER};"
        f"border-left:4px solid {BLUE};border-radius:6px;"
        f"padding:1rem 1.2rem;color:#0d1b2e;font-size:0.88rem;'>"
        f"{direction_text}{verdict_note}{tier_note}"
        f"<br><br>"
        f"<span style='color:{MUTED};font-size:0.80rem;'>"
        f"This interpretation describes the model's computed output for "
        f"the given inputs. It does not imply causation or constitute a "
        f"loan decision recommendation."
        f"</span>"
        f"</div>",
        unsafe_allow_html=True,
    )
