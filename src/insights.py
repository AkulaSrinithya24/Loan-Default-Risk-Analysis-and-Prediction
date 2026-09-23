"""
insights.py
-----------
Generates structured business insights and actionable recommendations
based on KPI and risk analysis results.

Public API
----------
generate_insights(df)         -> list[dict]
print_insights(df)            -> None
get_opportunities(df)         -> list[dict]
"""

import pandas as pd
from src.config import TARGET_COLUMN
from src.kpi import compute_portfolio_kpis, compute_segment_kpis
from src.risk_analysis import get_key_drivers, get_risk_tier_summary, assign_risk_tier


def generate_insights(df: pd.DataFrame) -> list:
    """Generate a list of data-driven insights from the clean dataset.

    Each insight is a dict with keys:
        category, finding, evidence, recommendation, priority
    """
    kpis    = compute_portfolio_kpis(df)
    drivers = get_key_drivers(df)
    df_t    = assign_risk_tier(df)
    tiers   = get_risk_tier_summary(df_t)

    emp_rates  = compute_segment_kpis(df, "EmploymentType")
    purp_rates = compute_segment_kpis(df, "LoanPurpose")
    edu_rates  = compute_segment_kpis(df, "Education")

    top_driver     = drivers.iloc[0]["Feature"]
    top_driver_pct = drivers.iloc[0]["Rel Diff (%)"]

    high_emp  = emp_rates.iloc[0]["EmploymentType"]
    high_emp_rate = emp_rates.iloc[0]["default_rate_pct"]
    low_emp   = emp_rates.iloc[-1]["EmploymentType"]
    low_emp_rate  = emp_rates.iloc[-1]["default_rate_pct"]

    high_purp = purp_rates.iloc[0]["LoanPurpose"]
    high_purp_rate = purp_rates.iloc[0]["default_rate_pct"]

    vh_tier = tiers[tiers["RiskTier"] == "Very High"].iloc[0] if "Very High" in tiers["RiskTier"].values else None
    low_tier = tiers[tiers["RiskTier"] == "Low"].iloc[0] if "Low" in tiers["RiskTier"].values else None

    insights = [
        {
            "category":       "Portfolio Risk",
            "finding":        f"Overall portfolio default rate is {kpis['default_rate_pct']}%.",
            "evidence":       f"{kpis['total_defaulted']:,} out of {kpis['total_loans']:,} loans defaulted. "
                              f"Defaulted loans represent {kpis['default_value_rate_pct']}% of total loan value.",
            "recommendation": "Tighten underwriting criteria to reduce portfolio default rate below 8%.",
            "priority":       "High",
        },
        {
            "category":       "Key Numeric Driver",
            "finding":        f"{top_driver} is the strongest numeric driver of default "
                              f"({top_driver_pct:.1f}% relative difference between defaulters and non-defaulters).",
            "evidence":       drivers[["Feature", "Mean (Default)", "Mean (No Default)", "Rel Diff (%)"]].head(3).to_string(index=False),
            "recommendation": f"Incorporate {top_driver} as a primary screening variable in loan approval workflows.",
            "priority":       "High",
        },
        {
            "category":       "Employment Risk",
            "finding":        f"{high_emp} borrowers have the highest default rate ({high_emp_rate}%), "
                              f"compared to {low_emp} borrowers at {low_emp_rate}%.",
            "evidence":       emp_rates[["EmploymentType", "default_rate_pct", "total_loans"]].to_string(index=False),
            "recommendation": f"Apply stricter income verification and lower LTV ratios for {high_emp} applicants.",
            "priority":       "High",
        },
        {
            "category":       "Loan Purpose Risk",
            "finding":        f"{high_purp} loans show the highest default rate ({high_purp_rate}%).",
            "evidence":       purp_rates[["LoanPurpose", "default_rate_pct", "total_loans"]].to_string(index=False),
            "recommendation": f"Review collateral requirements and repayment terms for {high_purp} loans.",
            "priority":       "Medium",
        },
        {
            "category":       "Risk Tier Concentration",
            "finding":        (
                f"Very High-risk tier accounts for "
                f"{vh_tier['total_loans']:,} loans ({vh_tier['default_rate_pct']}% default rate) "
                if vh_tier is not None else "Very High tier data unavailable."
            ),
            "evidence":       tiers[["RiskTier", "total_loans", "default_rate_pct", "default_value_pct"]].to_string(index=False),
            "recommendation": "Consider loan caps or mandatory co-signer requirements for Very High-risk tier borrowers.",
            "priority":       "High",
        },
        {
            "category":       "Credit Score Opportunity",
            "finding":        f"Average credit score of defaulters vs non-defaulters shows separation.",
            "evidence":       f"Defaulters avg score: {df[df[TARGET_COLUMN]==1]['CreditScore'].mean():.0f} | "
                              f"Non-defaulters avg score: {df[df[TARGET_COLUMN]==0]['CreditScore'].mean():.0f}",
            "recommendation": "Set minimum credit score thresholds differentiated by loan purpose and employment type.",
            "priority":       "Medium",
        },
        {
            "category":       "Interest Rate Policy",
            "finding":        "Higher interest rates correlate with higher default - not purely causal but indicative of risk pricing.",
            "evidence":       f"Avg rate (defaulters): {df[df[TARGET_COLUMN]==1]['InterestRate'].mean():.2f}% | "
                              f"Avg rate (non-defaulters): {df[df[TARGET_COLUMN]==0]['InterestRate'].mean():.2f}%",
            "recommendation": "Ensure interest rate bands reflect risk tiers. Avoid rate escalation for already high-risk borrowers.",
            "priority":       "Medium",
        },
        {
            "category":       "Predictive Modelling",
            "finding":        "Machine learning can predict default risk before loan disbursement.",
            "evidence":       "Random Forest model trained on all 16 features enables probability scoring per applicant.",
            "recommendation": "Deploy the ML model as a pre-approval scoring tool; flag applications above 40% predicted default probability for manual review.",
            "priority":       "High",
        },
    ]
    return insights


def get_opportunities(df: pd.DataFrame) -> list:
    """Return a list of actionable business opportunity items."""
    return [
        ins for ins in generate_insights(df)
        if ins["priority"] == "High"
    ]


def print_insights(df: pd.DataFrame) -> None:
    """Print all insights to stdout."""
    insights = generate_insights(df)
    print("=" * 70)
    print("  BUSINESS INSIGHTS & RECOMMENDATIONS")
    print("=" * 70)
    for i, ins in enumerate(insights, 1):
        print(f"\n[{i}] [{ins['priority']}] {ins['category']}")
        print(f"    Finding       : {ins['finding']}")
        print(f"    Evidence      : {ins['evidence'][:120]}...")
        print(f"    Recommendation: {ins['recommendation']}")
    print("=" * 70)
