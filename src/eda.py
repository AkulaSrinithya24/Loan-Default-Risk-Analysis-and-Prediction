"""
eda.py
------
Reusable EDA helper functions for the Loan Default Risk Analytics project.
All functions return a matplotlib Figure so they can be used both in
notebooks and in the Streamlit dashboard.

Public API
----------
plot_default_rate_by_category(df, col)        -> Figure
plot_numeric_by_default(df, col)              -> Figure
plot_all_numeric_by_default(df)               -> Figure
plot_correlation_heatmap(df)                  -> Figure
plot_default_rate_heatmap(df, col1, col2)     -> Figure
plot_feature_boxplots(df)                     -> Figure
get_default_rates_by_category(df, col)        -> pd.DataFrame
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
    BINARY_FEATURES,
)

# Consistent colour palette
COLOR_NO_DEFAULT = "#4a90d9"
COLOR_DEFAULT    = "#e05c5c"
PALETTE          = {"No Default": COLOR_NO_DEFAULT, "Default": COLOR_DEFAULT}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _label_default(df: pd.DataFrame) -> pd.DataFrame:
    """Add a string 'DefaultLabel' column for readable plot legends."""
    df = df.copy()
    df["DefaultLabel"] = df[TARGET_COLUMN].map({0: "No Default", 1: "Default"})
    return df


def get_default_rates_by_category(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Return a DataFrame with default rate per category value.

    Returns columns: [col, 'total', 'defaults', 'default_rate_pct']
    """
    grp = df.groupby(col)[TARGET_COLUMN].agg(
        total="count", defaults="sum"
    ).reset_index()
    grp["default_rate_pct"] = (grp["defaults"] / grp["total"] * 100).round(2)
    return grp.sort_values("default_rate_pct", ascending=False)


# ── Plot functions ─────────────────────────────────────────────────────────────

def plot_default_rate_by_category(df: pd.DataFrame, col: str) -> plt.Figure:
    """Horizontal bar chart showing default rate (%) per category value."""
    rates = get_default_rates_by_category(df, col)

    fig, ax = plt.subplots(figsize=(8, max(3, len(rates) * 0.55 + 1)))
    bars = ax.barh(
        rates[col].astype(str),
        rates["default_rate_pct"],
        color=COLOR_DEFAULT,
        edgecolor="white",
        linewidth=0.8,
    )
    ax.bar_label(bars, fmt="%.1f%%", padding=4, fontsize=9)
    ax.set_xlabel("Default Rate (%)")
    ax.set_title(f"Default Rate by {col}", fontweight="bold")
    ax.set_xlim(0, rates["default_rate_pct"].max() * 1.25)
    ax.invert_yaxis()
    fig.tight_layout()
    return fig


def plot_numeric_by_default(df: pd.DataFrame, col: str) -> plt.Figure:
    """Overlapping KDE + rug plot for a numeric feature split by Default."""
    df_l = _label_default(df)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # KDE
    for label, color in PALETTE.items():
        subset = df_l[df_l["DefaultLabel"] == label][col].dropna()
        axes[0].hist(subset, bins=40, alpha=0.55, color=color,
                     label=label, density=True, edgecolor="none")
        subset.plot.kde(ax=axes[0], color=color, linewidth=2)
    axes[0].set_title(f"{col} — Distribution by Default Status", fontweight="bold")
    axes[0].set_xlabel(col)
    axes[0].set_ylabel("Density")
    axes[0].legend()

    # Box plot
    groups = [
        df_l[df_l["DefaultLabel"] == lbl][col].dropna().values
        for lbl in ["No Default", "Default"]
    ]
    bp = axes[1].boxplot(
        groups,
        tick_labels=["No Default", "Default"],
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
        flierprops=dict(marker="o", markersize=2, alpha=0.3),
    )
    for patch, color in zip(bp["boxes"], [COLOR_NO_DEFAULT, COLOR_DEFAULT]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    axes[1].set_title(f"{col} — Box Plot by Default Status", fontweight="bold")
    axes[1].set_ylabel(col)

    fig.suptitle(col, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_all_numeric_by_default(df: pd.DataFrame) -> plt.Figure:
    """Grid of KDE histograms for all numeric features, split by Default."""
    df_l = _label_default(df)
    n_cols = 3
    n_rows = int(np.ceil(len(NUMERIC_FEATURES) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 3.5))
    axes = axes.flatten()

    for i, col in enumerate(NUMERIC_FEATURES):
        for label, color in PALETTE.items():
            subset = df_l[df_l["DefaultLabel"] == label][col].dropna()
            axes[i].hist(subset, bins=35, alpha=0.5, color=color,
                         label=label, density=True, edgecolor="none")
        axes[i].set_title(col, fontweight="bold")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Density")
        if i == 0:
            axes[i].legend(fontsize=8)

    for j in range(len(NUMERIC_FEATURES), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Numeric Feature Distributions by Default Status",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame) -> plt.Figure:
    """Lower-triangle correlation heatmap for all numeric features + target."""
    num_cols = NUMERIC_FEATURES + [TARGET_COLUMN]
    corr = df[num_cols].corr().round(2)

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="RdBu_r", center=0, linewidths=0.5,
        annot_kws={"size": 8}, ax=ax,
        vmin=-1, vmax=1,
    )
    ax.set_title("Correlation Matrix — Numeric Features + Default",
                 fontweight="bold")
    fig.tight_layout()
    return fig


def plot_default_rate_heatmap(
    df: pd.DataFrame, col1: str, col2: str
) -> plt.Figure:
    """2-D heatmap of default rate for two categorical features."""
    pivot = (
        df.groupby([col1, col2])[TARGET_COLUMN]
        .mean()
        .mul(100)
        .round(1)
        .unstack(col2)
    )

    fig, ax = plt.subplots(figsize=(max(6, pivot.shape[1] * 1.2),
                                    max(4, pivot.shape[0] * 0.8)))
    sns.heatmap(
        pivot, annot=True, fmt=".1f", cmap="YlOrRd",
        linewidths=0.5, linecolor="white",
        cbar_kws={"label": "Default Rate (%)"},
        ax=ax,
    )
    ax.set_title(f"Default Rate (%) — {col1} vs {col2}", fontweight="bold")
    ax.set_xlabel(col2)
    ax.set_ylabel(col1)
    fig.tight_layout()
    return fig


def plot_feature_boxplots(df: pd.DataFrame) -> plt.Figure:
    """Box plots of all numeric features for a compact outlier overview."""
    n_cols = 3
    n_rows = int(np.ceil(len(NUMERIC_FEATURES) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 3.5))
    axes = axes.flatten()

    for i, col in enumerate(NUMERIC_FEATURES):
        axes[i].boxplot(
            df[col].dropna(),
            vert=True,
            patch_artist=True,
            boxprops=dict(facecolor=COLOR_NO_DEFAULT, alpha=0.6),
            medianprops=dict(color="#c0392b", linewidth=2),
            flierprops=dict(marker="o", markersize=2, alpha=0.3,
                            color=COLOR_DEFAULT),
        )
        axes[i].set_title(col, fontweight="bold")
        axes[i].set_ylabel(col)

    for j in range(len(NUMERIC_FEATURES), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Numeric Feature Box Plots (Clean Data)",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


def plot_categorical_default_rates(df: pd.DataFrame) -> plt.Figure:
    """Grid of bar charts: default rate per value for each categorical column."""
    all_cats = CATEGORICAL_FEATURES + [
        c for c in BINARY_FEATURES if c in df.columns and df[c].dtype == object
    ]
    # Include binary cols encoded as int too — map them back for display
    binary_int = [c for c in BINARY_FEATURES if c in df.columns
                  and pd.api.types.is_integer_dtype(df[c])]

    n_cols = 2
    all_cols = CATEGORICAL_FEATURES  # binary ints shown separately
    n_rows = int(np.ceil(len(all_cols) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(14, n_rows * 3.8))
    axes = axes.flatten()

    for i, col in enumerate(all_cols):
        rates = get_default_rates_by_category(df, col)
        bars = axes[i].bar(
            rates[col].astype(str),
            rates["default_rate_pct"],
            color=COLOR_DEFAULT, edgecolor="white", linewidth=0.8,
        )
        axes[i].bar_label(bars, fmt="%.1f%%", padding=3, fontsize=8)
        axes[i].set_title(f"Default Rate by {col}", fontweight="bold")
        axes[i].set_ylabel("Default Rate (%)")
        axes[i].set_ylim(0, rates["default_rate_pct"].max() * 1.3)
        axes[i].tick_params(axis="x", rotation=25)

    for j in range(len(all_cols), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Default Rate by Categorical Features",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


def plot_binary_default_rates(df: pd.DataFrame) -> plt.Figure:
    """Grouped bar chart for all binary (0/1) features vs default rate."""
    binary_int = [c for c in BINARY_FEATURES if c in df.columns]
    label_map = {0: "No", 1: "Yes"}

    n_cols = 3
    n_rows = int(np.ceil(len(binary_int) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(13, n_rows * 3.5))
    axes = axes.flatten()

    for i, col in enumerate(binary_int):
        tmp = df.copy()
        tmp["_label"] = tmp[col].map(label_map)
        rates = get_default_rates_by_category(tmp, "_label")
        bars = axes[i].bar(
            rates["_label"],
            rates["default_rate_pct"],
            color=[COLOR_NO_DEFAULT, COLOR_DEFAULT],
            edgecolor="white", linewidth=0.8,
        )
        axes[i].bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
        axes[i].set_title(f"Default Rate by {col}", fontweight="bold")
        axes[i].set_ylabel("Default Rate (%)")
        axes[i].set_ylim(0, rates["default_rate_pct"].max() * 1.3)

    for j in range(len(binary_int), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Default Rate by Binary Features",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig
