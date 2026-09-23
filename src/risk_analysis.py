"""
risk_analysis.py
----------------
Risk scoring, segmentation, and key-driver analysis for the loan portfolio.

Public API
----------
compute_risk_score(df)                    -> pd.Series   (0-100 score per row)
assign_risk_tier(df)                      -> pd.DataFrame (with RiskScore & RiskTier)
get_risk_tier_summary(df_with_tiers)      -> pd.DataFrame
get_key_drivers(df)                       -> pd.DataFrame
plot_risk_tier_distribution(df_tiers)     -> Figure
plot_risk_score_by_default(df_tiers)      -> Figure
plot_key_drivers(df)                      -> Figure
plot_risk_profile_heatmap(df_tiers)       -> Figure
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from src.config import (
    TARGET_COLUMN,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)

COLOR_LOW    = "#27ae60"
COLOR_MEDIUM = "#f39c12"
COLOR_HIGH   = "#e05c5c"
COLOR_VERY_HIGH = "#8e1414"

TIER_COLORS = {
    "Low":       COLOR_LOW,
    "Medium":    COLOR_MEDIUM,
    "High":      COLOR_HIGH,
    "Very High": COLOR_VERY_HIGH,
}


# ── Risk scoring ───────────────────────────────────────────────────────────────

def compute_risk_score(df: pd.DataFrame) -> pd.Series:
    """Compute a simple heuristic risk score (0–100) per loan.

    Higher score = higher default risk.

    Scoring components (weighted):
    --------------------------------
    - CreditScore      (inverse, weight 0.25): lower score → higher risk
    - DTIRatio         (weight 0.20): higher ratio → higher risk
    - InterestRate     (weight 0.20): higher rate → higher risk
    - Income           (inverse, weight 0.15): lower income → higher risk
    - MonthsEmployed   (inverse, weight 0.10): less employed → higher risk
    - LoanAmount       (weight 0.10): larger loan → higher risk
    """
    d = df.copy()

    def minmax(s: pd.Series) -> pd.Series:
        lo, hi = s.min(), s.max()
        return (s - lo) / (hi - lo) if hi > lo else pd.Series(0.5, index=s.index)

    risk = (
          0.25 * (1 - minmax(d["CreditScore"]))   # lower score → more risk
        + 0.20 * minmax(d["DTIRatio"])
        + 0.20 * minmax(d["InterestRate"])
        + 0.15 * (1 - minmax(d["Income"]))        # lower income → more risk
        + 0.10 * (1 - minmax(d["MonthsEmployed"])) # less employed → more risk
        + 0.10 * minmax(d["LoanAmount"])
    )
    return (risk * 100).round(1)


def assign_risk_tier(df: pd.DataFrame) -> pd.DataFrame:
    """Add RiskScore and RiskTier columns to the DataFrame.

    Tiers (based on score percentiles):
        Low       : score < 25th percentile
        Medium    : 25th <= score < 60th
        High      : 60th <= score < 85th
        Very High : score >= 85th
    """
    df = df.copy()
    df["RiskScore"] = compute_risk_score(df)

    p25 = df["RiskScore"].quantile(0.25)
    p60 = df["RiskScore"].quantile(0.60)
    p85 = df["RiskScore"].quantile(0.85)

    def _tier(s):
        if s < p25:
            return "Low"
        elif s < p60:
            return "Medium"
        elif s < p85:
            return "High"
        else:
            return "Very High"

    df["RiskTier"] = df["RiskScore"].apply(_tier)
    return df


def get_risk_tier_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise default rates and loan metrics per risk tier.

    Expects df to have RiskScore and RiskTier columns (output of assign_risk_tier).
    """
    tier_order = ["Low", "Medium", "High", "Very High"]
    rows = []
    for tier in tier_order:
        grp = df[df["RiskTier"] == tier]
        if len(grp) == 0:
            continue
        total_v = grp["LoanAmount"].sum()
        def_v   = grp.loc[grp[TARGET_COLUMN] == 1, "LoanAmount"].sum()
        rows.append({
            "RiskTier":          tier,
            "total_loans":       len(grp),
            "defaults":          int(grp[TARGET_COLUMN].sum()),
            "default_rate_pct":  round(grp[TARGET_COLUMN].mean() * 100, 2),
            "avg_risk_score":    round(grp["RiskScore"].mean(), 1),
            "avg_credit_score":  round(grp["CreditScore"].mean(), 1),
            "avg_loan_amount":   round(grp["LoanAmount"].mean(), 0),
            "avg_interest_rate": round(grp["InterestRate"].mean(), 2),
            "avg_dti_ratio":     round(grp["DTIRatio"].mean(), 3),
            "total_loan_value":  round(total_v, 0),
            "default_value":     round(def_v, 0),
            "default_value_pct": round(def_v / total_v * 100, 2) if total_v > 0 else 0,
        })
    return pd.DataFrame(rows)


# ── Key drivers ────────────────────────────────────────────────────────────────

def get_key_drivers(df: pd.DataFrame) -> pd.DataFrame:
    """Identify key numeric drivers of default using mean difference analysis.

    For each numeric feature, compute:
    - mean value for defaulters
    - mean value for non-defaulters
    - absolute difference
    - relative difference (%)
    - direction of risk (higher or lower value → more risk)

    Returns a DataFrame sorted by |relative_diff_pct| descending.
    """
    rows = []
    for col in NUMERIC_FEATURES:
        mean_def   = df.loc[df[TARGET_COLUMN] == 1, col].mean()
        mean_nodef = df.loc[df[TARGET_COLUMN] == 0, col].mean()
        abs_diff   = abs(mean_def - mean_nodef)
        rel_diff   = abs_diff / abs(mean_nodef) * 100 if mean_nodef != 0 else 0
        direction  = "Higher -> more risk" if mean_def > mean_nodef else "Lower -> more risk"
        rows.append({
            "Feature":            col,
            "Mean (Default)":     round(mean_def, 3),
            "Mean (No Default)":  round(mean_nodef, 3),
            "Abs Difference":     round(abs_diff, 3),
            "Rel Diff (%)":       round(rel_diff, 2),
            "Risk Direction":     direction,
        })
    return (
        pd.DataFrame(rows)
        .sort_values("Rel Diff (%)", ascending=False)
        .reset_index(drop=True)
    )


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_risk_tier_distribution(df: pd.DataFrame) -> plt.Figure:
    """Bar chart of loan count per risk tier."""
    tier_order = ["Low", "Medium", "High", "Very High"]
    counts = df["RiskTier"].value_counts().reindex(tier_order).fillna(0)

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(
        counts.index, counts.values,
        color=[TIER_COLORS[t] for t in counts.index],
        edgecolor="white", linewidth=0.8,
    )
    ax.bar_label(bars, fmt=lambda v: f"{int(v):,}", padding=4, fontsize=9)
    ax.set_title("Loan Count by Risk Tier", fontweight="bold")
    ax.set_ylabel("Number of Loans")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    fig.tight_layout()
    return fig


def plot_risk_score_by_default(df: pd.DataFrame) -> plt.Figure:
    """KDE + box plot of RiskScore split by Default status."""
    def_0 = df.loc[df[TARGET_COLUMN] == 0, "RiskScore"]
    def_1 = df.loc[df[TARGET_COLUMN] == 1, "RiskScore"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(def_0, bins=40, alpha=0.55, color="#4a90d9",
                 label="No Default", density=True, edgecolor="none")
    axes[0].hist(def_1, bins=40, alpha=0.55, color="#e05c5c",
                 label="Default", density=True, edgecolor="none")
    axes[0].set_title("Risk Score Distribution by Default Status", fontweight="bold")
    axes[0].set_xlabel("Risk Score")
    axes[0].set_ylabel("Density")
    axes[0].legend()

    bp = axes[1].boxplot(
        [def_0.values, def_1.values],
        tick_labels=["No Default", "Default"],
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
        flierprops=dict(marker="o", markersize=2, alpha=0.3),
    )
    for patch, color in zip(bp["boxes"], ["#4a90d9", "#e05c5c"]):
        patch.set_facecolor(color); patch.set_alpha(0.6)
    axes[1].set_title("Risk Score Box Plot by Default Status", fontweight="bold")
    axes[1].set_ylabel("Risk Score")

    fig.suptitle("Heuristic Risk Score vs Actual Default", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_key_drivers(df: pd.DataFrame) -> plt.Figure:
    """Horizontal bar chart of relative difference (%) for each numeric feature."""
    drivers = get_key_drivers(df)

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#e05c5c" if "Higher" in d else "#4a90d9"
              for d in drivers["Risk Direction"]]
    bars = ax.barh(
        drivers["Feature"], drivers["Rel Diff (%)"],
        color=colors, edgecolor="white", linewidth=0.8,
    )
    ax.bar_label(bars, fmt="%.1f%%", padding=4, fontsize=9)
    ax.set_xlabel("Relative Difference between Defaulters and Non-Defaulters (%)")
    ax.set_title("Key Numeric Drivers of Loan Default\n(Red = higher value -> more risk  |  Blue = lower value -> more risk)",
                 fontweight="bold")
    ax.invert_yaxis()
    fig.tight_layout()
    return fig


def plot_risk_profile_heatmap(df: pd.DataFrame) -> plt.Figure:
    """Heatmap: default rate by RiskTier × EmploymentType."""
    tier_order = ["Low", "Medium", "High", "Very High"]
    pivot = (
        df.groupby(["RiskTier", "EmploymentType"])[TARGET_COLUMN]
        .mean()
        .mul(100)
        .round(1)
        .unstack("EmploymentType")
        .reindex(tier_order)
    )

    fig, ax = plt.subplots(figsize=(9, 4))
    sns.heatmap(
        pivot, annot=True, fmt=".1f", cmap="YlOrRd",
        linewidths=0.5, linecolor="white",
        cbar_kws={"label": "Default Rate (%)"},
        ax=ax,
    )
    ax.set_title("Default Rate (%) by Risk Tier × Employment Type", fontweight="bold")
    ax.set_ylabel("Risk Tier")
    ax.set_xlabel("Employment Type")
    fig.tight_layout()
    return fig
