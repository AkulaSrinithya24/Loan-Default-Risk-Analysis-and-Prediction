"""
dashboard/pages/06_portfolio_exposure.py
-----------------------------------------
Portfolio Exposure Analysis page.

Purpose
-------
Identify *where* financial default exposure is concentrated — by loan purpose,
employment type, education, income band, and loan-size band — using dollar-value
exposure rather than simple default rates.

What this page does NOT duplicate
----------------------------------
- Portfolio-level KPI metrics                   → 01_overview.py
- Default counts / class-balance charts         → 01_overview.py
- Feature distribution KDE / box plots          → 02_eda.py
- Default-rate bar charts by segment            → 01_overview.py, 02_eda.py
- Risk tier distribution chart                  → 03_risk_analysis.py
- Risk tier × employment heatmap                → 03_risk_analysis.py
- Binary feature default rates                  → 02_eda.py

Unique contributions
--------------------
1.  Focused exposure-value summary cards
2.  Defaulted-$ by Loan Purpose (sorted bar)
3.  Defaulted-$ by Employment Type (sorted bar)
4.  Defaulted-$ by Education Level (sorted bar)
5.  Exposure Treemap — area ∝ defaulted loan value, selectable dimension
6.  Cross-segment exposure table — selectable pair of dimensions
7.  Exposure by Loan-Size Band
8.  Exposure by Income Quartile
"""

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
import matplotlib.patches as mpatches

from src.data_loader import load_data
from src.data_cleaner import clean_data
from src.config import RAW_DATA_PATH, TARGET_COLUMN

st.set_page_config(
    page_title="Portfolio Exposure Analysis",
    page_icon="💰",
    layout="wide",
)

from dashboard.theme import page_header, section_title, show_table, apply_theme

apply_theme()

page_header(
    "Portfolio Exposure Analysis",
    "Where is defaulted loan value concentrated? Segment-level dollar exposure across "
    "loan purpose, employment, education, income, and loan size.",
)

# ── Colour palette (consistent with project theme) ──────────────────────────
NAVY       = "#0a2342"
BLUE       = "#1a56a0"
LIGHT_BLUE = "#e8f0fb"
RED        = "#b91c1c"
AMBER      = "#b45309"
GREEN      = "#1a7a4a"
MUTED      = "#5a6a80"
BORDER     = "#dce3ed"
BG         = "#f5f7fa"

# ── Segment colour palettes for charts ──────────────────────────────────────
PURPOSE_COLORS    = ["#0a2342", "#1a56a0", "#2d6fbb", "#5090cc", "#80b4df"]
EMPLOYMENT_COLORS = ["#b91c1c", "#d44040", "#e87070", "#f0a0a0"]
EDUCATION_COLORS  = ["#1a7a4a", "#28a060", "#50bb80", "#88d4a8"]
LOAN_SIZE_COLORS  = ["#0a2342", "#1a56a0", "#2d6fbb", "#5090cc", "#80b4df", "#aac8e8"]
INCOME_COLORS     = ["#b45309", "#cc7020", "#e09040", "#f0b870"]

# ── Data loading ─────────────────────────────────────────────────────────────
@st.cache_data
def get_data() -> pd.DataFrame:
    """Load and clean the dataset — identical pattern used by every page."""
    return clean_data(load_data(RAW_DATA_PATH))


df = get_data()

# Pre-compute portfolio totals needed throughout the page
_total_loan_value     = float(df["LoanAmount"].sum())
_defaulted_df         = df[df[TARGET_COLUMN] == 1].copy()
_total_default_value  = float(_defaulted_df["LoanAmount"].sum())
_total_loans          = len(df)
_total_defaults       = int(df[TARGET_COLUMN].sum())
_exposure_pct         = round(_total_default_value / _total_loan_value * 100, 2)

# ── Helper: build per-segment exposure table ─────────────────────────────────
def _exposure_by(col: str, df_defaulted: pd.DataFrame, df_full: pd.DataFrame) -> pd.DataFrame:
    """
    For each unique value in `col` compute:
      - defaulted_loan_value : total $ value of defaulted loans in segment
      - total_loan_value     : total $ value of ALL loans in segment
      - exposure_pct         : defaulted / total (within segment) × 100
      - portfolio_share_pct  : segment defaulted $ / portfolio total defaulted $ × 100
      - default_count        : number of defaulted loans
      - total_count          : number of all loans
      - default_rate_pct     : count-based default rate within segment
    """
    grp_def   = df_defaulted.groupby(col)["LoanAmount"].agg(
        defaulted_loan_value="sum", default_count="count"
    ).reset_index()
    grp_total = df_full.groupby(col)["LoanAmount"].agg(
        total_loan_value="sum", total_count="count"
    ).reset_index()

    out = grp_total.merge(grp_def, on=col, how="left").fillna(0)
    out["defaulted_loan_value"] = out["defaulted_loan_value"].astype(float)
    out["default_count"]        = out["default_count"].astype(int)

    out["exposure_pct"]         = (out["defaulted_loan_value"] / out["total_loan_value"] * 100).round(2)
    out["portfolio_share_pct"]  = (out["defaulted_loan_value"] / _total_default_value * 100).round(2)
    out["default_rate_pct"]     = (out["default_count"] / out["total_count"] * 100).round(2)

    return out.sort_values("defaulted_loan_value", ascending=False).reset_index(drop=True)


# ── Helper: horizontal exposure bar chart ────────────────────────────────────
def _hbar_exposure(
    labels: list,
    values_m: list,          # values already divided by 1e6
    share_pcts: list,
    colors: list,
    title: str,
    xlabel: str = "Defaulted Loan Value ($ Millions)",
) -> plt.Figure:
    """
    Horizontal bar chart for defaulted $ by segment, with portfolio-share
    annotations on the right side of each bar.
    """
    n = len(labels)
    fig_h = max(2.8, n * 0.62)
    fig, ax = plt.subplots(figsize=(8, fig_h), facecolor="white")
    ax.set_facecolor("white")

    bars = ax.barh(
        range(n), values_m,
        color=colors[:n] if len(colors) >= n else (colors * n)[:n],
        edgecolor="white", linewidth=0.6, height=0.6,
    )

    # Value labels inside or beside each bar
    for i, (bar, val, pct) in enumerate(zip(bars, values_m, share_pcts)):
        w = bar.get_width()
        ax.text(
            w + max(values_m) * 0.01, i,
            f"${val:.1f}M  ({pct:.1f}%)",
            va="center", ha="left",
            fontsize=8.5, color=NAVY, fontweight="600",
        )

    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=9, color=NAVY)
    ax.set_xlabel(xlabel, fontsize=8.5, color=MUTED)
    ax.set_title(title, fontsize=10.5, fontweight="700", color=NAVY, pad=8)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}M"))
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(BORDER)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    ax.set_xlim(0, max(values_m) * 1.30)
    ax.invert_yaxis()
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Exposure Summary Cards
# ─────────────────────────────────────────────────────────────────────────────
section_title("1. Portfolio Default Exposure Summary")
st.markdown("<div style='margin-bottom:0.5rem;'></div>", unsafe_allow_html=True)

st.markdown(
    f"""
    <div style="
        background: linear-gradient(135deg, {NAVY} 0%, {BLUE} 100%);
        border-radius: 12px; padding: 1.6rem 2rem; margin-bottom: 1.4rem;
        display: flex; flex-wrap: wrap; gap: 1.5rem; align-items: center;">
      <div style="flex:1; min-width:200px;">
        <div style="color:#c9d8ec; font-size:0.75rem; font-weight:700;
                    letter-spacing:0.07em; text-transform:uppercase;">Total Defaulted Value</div>
        <div style="color:#ffffff; font-size:2rem; font-weight:800; line-height:1.2;">
          ${_total_default_value/1e9:.2f}B</div>
        <div style="color:#a8c4e0; font-size:0.82rem; margin-top:0.2rem;">
          across {_total_defaults:,} defaulted loans</div>
      </div>
      <div style="flex:1; min-width:160px; border-left:1px solid rgba(255,255,255,0.15);
                  padding-left:1.5rem;">
        <div style="color:#c9d8ec; font-size:0.75rem; font-weight:700;
                    letter-spacing:0.07em; text-transform:uppercase;">Exposure Rate</div>
        <div style="color:#fca5a5; font-size:2rem; font-weight:800; line-height:1.2;">
          {_exposure_pct}%</div>
        <div style="color:#a8c4e0; font-size:0.82rem; margin-top:0.2rem;">
          of ${_total_loan_value/1e9:.2f}B total portfolio</div>
      </div>
      <div style="flex:1; min-width:160px; border-left:1px solid rgba(255,255,255,0.15);
                  padding-left:1.5rem;">
        <div style="color:#c9d8ec; font-size:0.75rem; font-weight:700;
                    letter-spacing:0.07em; text-transform:uppercase;">Non-Defaulted Value</div>
        <div style="color:#6ee7b7; font-size:2rem; font-weight:800; line-height:1.2;">
          ${(_total_loan_value - _total_default_value)/1e9:.2f}B</div>
        <div style="color:#a8c4e0; font-size:0.82rem; margin-top:0.2rem;">
          {100 - _exposure_pct:.2f}% performing</div>
      </div>
      <div style="flex:1; min-width:160px; border-left:1px solid rgba(255,255,255,0.15);
                  padding-left:1.5rem;">
        <div style="color:#c9d8ec; font-size:0.75rem; font-weight:700;
                    letter-spacing:0.07em; text-transform:uppercase;">Avg Defaulted Loan</div>
        <div style="color:#ffffff; font-size:2rem; font-weight:800; line-height:1.2;">
          ${_total_default_value/_total_defaults:,.0f}</div>
        <div style="color:#a8c4e0; font-size:0.82rem; margin-top:0.2rem;">
          vs ${_total_loan_value/_total_loans:,.0f} portfolio avg</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<hr>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Exposure by Loan Purpose & Employment Type (side by side)
# ─────────────────────────────────────────────────────────────────────────────
section_title("2. Exposure by Loan Purpose & Employment Type")
st.markdown("<div style='margin-bottom:0.6rem;'></div>", unsafe_allow_html=True)

purpose_exp    = _exposure_by("LoanPurpose",    _defaulted_df, df)
employment_exp = _exposure_by("EmploymentType", _defaulted_df, df)

col_left, col_right = st.columns(2)

with col_left:
    fig_purpose = _hbar_exposure(
        labels    = purpose_exp["LoanPurpose"].tolist(),
        values_m  = (purpose_exp["defaulted_loan_value"] / 1e6).tolist(),
        share_pcts= purpose_exp["portfolio_share_pct"].tolist(),
        colors    = PURPOSE_COLORS,
        title     = "Defaulted Value by Loan Purpose",
    )
    st.pyplot(fig_purpose)
    plt.close(fig_purpose)

with col_right:
    fig_emp = _hbar_exposure(
        labels    = employment_exp["EmploymentType"].tolist(),
        values_m  = (employment_exp["defaulted_loan_value"] / 1e6).tolist(),
        share_pcts= employment_exp["portfolio_share_pct"].tolist(),
        colors    = EMPLOYMENT_COLORS,
        title     = "Defaulted Value by Employment Type",
    )
    st.pyplot(fig_emp)
    plt.close(fig_emp)

st.markdown("<div style='margin-top:0.8rem;'></div>", unsafe_allow_html=True)

# Detail tables (collapsible)
with st.expander("📋 Loan Purpose — Exposure Detail Table", expanded=False):
    show_table(
        purpose_exp[["LoanPurpose", "defaulted_loan_value", "total_loan_value",
                      "exposure_pct", "portfolio_share_pct",
                      "default_count", "total_count", "default_rate_pct"]],
        fmt={
            "defaulted_loan_value": "${:,.0f}",
            "total_loan_value":     "${:,.0f}",
            "exposure_pct":         "{:.2f}%",
            "portfolio_share_pct":  "{:.2f}%",
            "default_rate_pct":     "{:.2f}%",
        },
    )

with st.expander("📋 Employment Type — Exposure Detail Table", expanded=False):
    show_table(
        employment_exp[["EmploymentType", "defaulted_loan_value", "total_loan_value",
                         "exposure_pct", "portfolio_share_pct",
                         "default_count", "total_count", "default_rate_pct"]],
        fmt={
            "defaulted_loan_value": "${:,.0f}",
            "total_loan_value":     "${:,.0f}",
            "exposure_pct":         "{:.2f}%",
            "portfolio_share_pct":  "{:.2f}%",
            "default_rate_pct":     "{:.2f}%",
        },
    )

st.markdown("<hr>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Exposure by Education Level
# ─────────────────────────────────────────────────────────────────────────────
section_title("3. Exposure by Education Level")
st.markdown("<div style='margin-bottom:0.6rem;'></div>", unsafe_allow_html=True)

education_exp = _exposure_by("Education", _defaulted_df, df)

col_edu_chart, col_edu_info = st.columns([2, 1])

with col_edu_chart:
    fig_edu = _hbar_exposure(
        labels    = education_exp["Education"].tolist(),
        values_m  = (education_exp["defaulted_loan_value"] / 1e6).tolist(),
        share_pcts= education_exp["portfolio_share_pct"].tolist(),
        colors    = EDUCATION_COLORS,
        title     = "Defaulted Value by Education Level",
    )
    st.pyplot(fig_edu)
    plt.close(fig_edu)

with col_edu_info:
    st.markdown(
        "<div style='margin-top:0.5rem;'></div>",
        unsafe_allow_html=True,
    )
    for _, row in education_exp.iterrows():
        bar_w = row["portfolio_share_pct"] / education_exp["portfolio_share_pct"].max() * 100
        st.markdown(
            f"""
            <div style="background:#ffffff; border:1px solid {BORDER};
                        border-radius:8px; padding:0.75rem 1rem;
                        margin-bottom:0.5rem; box-shadow:0 1px 3px rgba(10,35,66,0.05);">
              <div style="display:flex; justify-content:space-between; margin-bottom:0.3rem;">
                <span style="color:{NAVY}; font-weight:700; font-size:0.88rem;">
                  {row["Education"]}</span>
                <span style="color:{RED}; font-weight:700; font-size:0.88rem;">
                  {row["portfolio_share_pct"]:.1f}% of exposure</span>
              </div>
              <div style="background:{BORDER}; border-radius:4px; height:6px;">
                <div style="background:{BLUE}; width:{bar_w:.1f}%;
                            height:6px; border-radius:4px;"></div>
              </div>
              <div style="color:{MUTED}; font-size:0.75rem; margin-top:0.3rem;">
                ${row["defaulted_loan_value"]/1e6:.1f}M defaulted
                &nbsp;·&nbsp; {row["exposure_pct"]:.2f}% segment exposure rate</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("<hr>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Interactive Exposure Treemap
# ─────────────────────────────────────────────────────────────────────────────
section_title("4. Exposure Treemap — Defaulted Value by Segment")
st.markdown(
    "<p style='color:#5a6a80; font-size:0.85rem; margin-bottom:0.8rem;'>"
    "Each rectangle is proportional to the defaulted loan value in that segment. "
    "Select a dimension to explore different concentrations.</p>",
    unsafe_allow_html=True,
)

treemap_dim = st.selectbox(
    "Segment dimension",
    options=["LoanPurpose", "EmploymentType", "Education", "MaritalStatus"],
    index=0,
    key="treemap_dim",
)

@st.cache_data
def _build_treemap_data(col: str) -> pd.DataFrame:
    return _exposure_by(col, _defaulted_df, df)

treemap_data = _build_treemap_data(treemap_dim)


def _draw_treemap(data: pd.DataFrame, label_col: str) -> plt.Figure:
    """
    Pure-matplotlib squarified treemap.
    Each cell area ∝ defaulted_loan_value.
    Uses a simple row-layout approximation (sorted descending, rows filled left→right).
    """
    values = data["defaulted_loan_value"].values.astype(float)
    labels = data[label_col].values
    shares = data["portfolio_share_pct"].values
    total  = values.sum()

    if total == 0:
        fig, ax = plt.subplots(figsize=(9, 4), facecolor="white")
        ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=12)
        return fig

    # Normalise to [0, 1]
    norm = values / total

    # Treemap layout via squarify algorithm (manual row-strip)
    # We implement a simple one-level strip layout that looks clean
    PALETTE = [
        "#0a2342", "#1a56a0", "#2d6fbb", "#4a84cc", "#6ea8df",
        "#b91c1c", "#1a7a4a", "#b45309", "#6b21a8", "#0e7490",
    ]

    fig, ax = plt.subplots(figsize=(9, 4.5), facecolor="white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor("white")

    # ── Strip layout ──────────────────────────────────────────────────────
    # Sort descending so largest cells fill first
    order     = np.argsort(norm)[::-1]
    norm_ord  = norm[order]
    labels_ord = labels[order]
    shares_ord = shares[order]
    values_ord = values[order]

    # Build rows: greedily fill until width would exceed aspect ratio ideal
    ASPECT   = 9 / 4.5   # width / height of figure
    rects    = []          # (x, y, w, h, idx)
    y_cursor = 1.0

    remaining      = list(zip(norm_ord, range(len(norm_ord))))
    remaining_vals = list(norm_ord)

    # Simple strip layout: each strip has height = sum of strip norms / row_width
    # We use Bruls squarify approximation: rows built to minimise worst aspect ratio
    def _worst_ratio(row, width):
        if not row or width == 0:
            return float("inf")
        total_r = sum(row)
        # heights in this strip
        heights = [v / total_r * total_r / width for v in row]
        widths  = [v / total_r * width for v in row]
        ratios  = []
        for w, h in zip(widths, heights):
            if h > 0 and w > 0:
                ratios.append(max(w / h, h / w))
        return max(ratios) if ratios else float("inf")

    def _layout_row(row_vals, row_idxs, x0, y0, strip_w, strip_h):
        """Fill one horizontal strip with cells from row_vals."""
        total_r = sum(row_vals)
        x_cursor = x0
        for v, idx in zip(row_vals, row_idxs):
            cell_w = v / total_r * strip_w
            rects.append((x_cursor, y0 - strip_h, cell_w, strip_h, idx))
            x_cursor += cell_w

    current_row      = []
    current_idxs     = []
    x0, y0           = 0.0, 1.0
    strip_w          = 1.0

    for i, v in enumerate(norm_ord):
        test_row = current_row + [v]
        strip_h  = sum(test_row) / strip_w
        if current_row and _worst_ratio(test_row, strip_w) > _worst_ratio(current_row, strip_w):
            # Flush current row
            strip_h_cur = sum(current_row) / strip_w
            _layout_row(current_row, current_idxs, x0, y0, strip_w, strip_h_cur)
            y0 -= strip_h_cur
            current_row  = [v]
            current_idxs = [i]
        else:
            current_row.append(v)
            current_idxs.append(i)

    if current_row:
        strip_h_cur = sum(current_row) / strip_w
        _layout_row(current_row, current_idxs, x0, y0, strip_w, strip_h_cur)

    # ── Draw rectangles ───────────────────────────────────────────────────
    for (rx, ry, rw, rh, idx) in rects:
        color = PALETTE[idx % len(PALETTE)]
        rect  = mpatches.FancyBboxPatch(
            (rx + 0.003, ry + 0.003),
            rw - 0.006, rh - 0.006,
            boxstyle="round,pad=0.005",
            facecolor=color, edgecolor="white", linewidth=1.5,
        )
        ax.add_patch(rect)

        # Label if cell is large enough
        cx = rx + rw / 2
        cy = ry + rh / 2
        if rw > 0.08 and rh > 0.06:
            ax.text(cx, cy + rh * 0.10, labels_ord[idx],
                    ha="center", va="center",
                    fontsize=min(9.5, rw * 55), fontweight="700",
                    color="white", clip_on=True)
            ax.text(cx, cy - rh * 0.14,
                    f"${values_ord[idx]/1e6:.1f}M",
                    ha="center", va="center",
                    fontsize=min(8.0, rw * 45),
                    color="rgba(255,255,255,0.85)" if False else "#d6e4f5",
                    clip_on=True)
            ax.text(cx, cy - rh * 0.34,
                    f"({shares_ord[idx]:.1f}%)",
                    ha="center", va="center",
                    fontsize=min(7.0, rw * 38),
                    color="#a8c4e0", clip_on=True)
        elif rw > 0.04 and rh > 0.035:
            ax.text(cx, cy, labels_ord[idx],
                    ha="center", va="center",
                    fontsize=min(7.5, rw * 48), fontweight="700",
                    color="white", clip_on=True)

    ax.set_title(
        f"Default Exposure Treemap — by {label_col}  "
        f"(total defaulted: ${total/1e6:.1f}M)",
        fontsize=10, fontweight="700", color=NAVY, pad=10,
    )
    fig.tight_layout(pad=0.5)
    return fig


fig_treemap = _draw_treemap(treemap_data, treemap_dim)
st.pyplot(fig_treemap)
plt.close(fig_treemap)

st.markdown("<hr>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — Cross-Segment Exposure Table (interactive)
# ─────────────────────────────────────────────────────────────────────────────
section_title("5. Cross-Segment Borrower Exposure Analysis")
st.markdown(
    "<p style='color:#5a6a80; font-size:0.85rem; margin-bottom:0.8rem;'>"
    "Select two dimensions to see how default exposure is distributed across "
    "borrower segment combinations — sorted by defaulted loan value.</p>",
    unsafe_allow_html=True,
)

SEGMENT_COLS = ["LoanPurpose", "EmploymentType", "Education", "MaritalStatus"]

cs1, cs2, cs3 = st.columns([1, 1, 1])
with cs1:
    seg_row = st.selectbox(
        "Primary segment", options=SEGMENT_COLS, index=0, key="cross_row"
    )
with cs2:
    remaining_cols = [c for c in SEGMENT_COLS if c != seg_row]
    seg_col = st.selectbox(
        "Secondary segment", options=remaining_cols, index=0, key="cross_col"
    )
with cs3:
    top_n = st.slider("Show top N segments", min_value=5, max_value=30, value=15, key="cross_n")


@st.cache_data
def _cross_segment_exposure(col_a: str, col_b: str) -> pd.DataFrame:
    """Compute defaulted value, exposure %, and portfolio share for every (col_a, col_b) pair."""
    grp_def = (
        _defaulted_df.groupby([col_a, col_b])["LoanAmount"]
        .agg(defaulted_loan_value="sum", default_count="count")
        .reset_index()
    )
    grp_total = (
        df.groupby([col_a, col_b])["LoanAmount"]
        .agg(total_loan_value="sum", total_count="count")
        .reset_index()
    )
    out = grp_total.merge(grp_def, on=[col_a, col_b], how="left").fillna(0)
    out["defaulted_loan_value"] = out["defaulted_loan_value"].astype(float)
    out["default_count"]        = out["default_count"].astype(int)
    out["exposure_pct"]         = (out["defaulted_loan_value"] / out["total_loan_value"] * 100).round(2)
    out["portfolio_share_pct"]  = (out["defaulted_loan_value"] / _total_default_value * 100).round(2)
    out["default_rate_pct"]     = (out["default_count"] / out["total_count"] * 100).round(2)
    return out.sort_values("defaulted_loan_value", ascending=False).reset_index(drop=True)


cross_df = _cross_segment_exposure(seg_row, seg_col).head(top_n)

# Visual bar chart for top cross-segment combinations
cross_df["segment_label"] = cross_df[seg_row] + " / " + cross_df[seg_col]
cross_vals_m = (cross_df["defaulted_loan_value"] / 1e6).tolist()
cross_labels  = cross_df["segment_label"].tolist()
cross_shares  = cross_df["portfolio_share_pct"].tolist()

n_bars = len(cross_labels)
fig_cross, ax_cross = plt.subplots(
    figsize=(9, max(3.5, n_bars * 0.50)), facecolor="white"
)
ax_cross.set_facecolor("white")

bar_colors = [NAVY if i % 2 == 0 else BLUE for i in range(n_bars)]
bars_c = ax_cross.barh(
    range(n_bars), cross_vals_m,
    color=bar_colors, edgecolor="white", linewidth=0.5, height=0.65,
)

for i, (bar, val, pct) in enumerate(zip(bars_c, cross_vals_m, cross_shares)):
    w = bar.get_width()
    ax_cross.text(
        w + max(cross_vals_m) * 0.01, i,
        f"${val:.1f}M  ({pct:.1f}%)",
        va="center", ha="left", fontsize=8, color=NAVY, fontweight="600",
    )

ax_cross.set_yticks(range(n_bars))
ax_cross.set_yticklabels(cross_labels, fontsize=8.2, color=NAVY)
ax_cross.set_xlabel("Defaulted Loan Value ($ Millions)", fontsize=8.5, color=MUTED)
ax_cross.set_title(
    f"Top {n_bars} Cross-Segment Combinations — {seg_row} × {seg_col}",
    fontsize=10.5, fontweight="700", color=NAVY, pad=8,
)
ax_cross.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}M"))
ax_cross.spines[["top", "right", "left"]].set_visible(False)
ax_cross.spines["bottom"].set_color(BORDER)
ax_cross.tick_params(colors=MUTED, labelsize=8)
ax_cross.set_xlim(0, max(cross_vals_m) * 1.28)
ax_cross.invert_yaxis()
fig_cross.tight_layout()
st.pyplot(fig_cross)
plt.close(fig_cross)

st.markdown("<div style='margin-top:0.7rem;'></div>", unsafe_allow_html=True)

with st.expander("📋 Full Cross-Segment Exposure Table", expanded=False):
    show_table(
        cross_df[[seg_row, seg_col, "defaulted_loan_value", "total_loan_value",
                   "exposure_pct", "portfolio_share_pct",
                   "default_count", "total_count", "default_rate_pct"]],
        fmt={
            "defaulted_loan_value": "${:,.0f}",
            "total_loan_value":     "${:,.0f}",
            "exposure_pct":         "{:.2f}%",
            "portfolio_share_pct":  "{:.2f}%",
            "default_rate_pct":     "{:.2f}%",
        },
    )

st.markdown("<hr>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — Exposure by Loan-Size Band
# ─────────────────────────────────────────────────────────────────────────────
section_title("6. Exposure by Loan-Size Band")
st.markdown(
    "<p style='color:#5a6a80; font-size:0.85rem; margin-bottom:0.8rem;'>"
    "Examines whether default dollar losses are concentrated in small or large loans.</p>",
    unsafe_allow_html=True,
)

LOAN_BINS   = [0, 50_000, 100_000, 150_000, 200_000, 300_000, float("inf")]
LOAN_LABELS = ["<$50K", "$50K–100K", "$100K–150K", "$150K–200K", "$200K–300K", ">$300K"]


@st.cache_data
def _exposure_by_loan_size() -> pd.DataFrame:
    tmp = df.copy()
    tmp["LoanSizeBand"] = pd.cut(
        tmp["LoanAmount"], bins=LOAN_BINS, labels=LOAN_LABELS, right=False
    )
    def_tmp = tmp[tmp[TARGET_COLUMN] == 1].copy()

    grp_def   = def_tmp.groupby("LoanSizeBand", observed=True)["LoanAmount"].agg(
        defaulted_loan_value="sum", default_count="count"
    ).reset_index()
    grp_total = tmp.groupby("LoanSizeBand", observed=True)["LoanAmount"].agg(
        total_loan_value="sum", total_count="count"
    ).reset_index()

    out = grp_total.merge(grp_def, on="LoanSizeBand", how="left").fillna(0)
    out["defaulted_loan_value"] = out["defaulted_loan_value"].astype(float)
    out["default_count"]        = out["default_count"].astype(int)
    out["exposure_pct"]         = (out["defaulted_loan_value"] / out["total_loan_value"] * 100).round(2)
    out["portfolio_share_pct"]  = (out["defaulted_loan_value"] / _total_default_value * 100).round(2)
    out["default_rate_pct"]     = (out["default_count"] / out["total_count"] * 100).round(2)
    return out


loan_size_exp = _exposure_by_loan_size()

col_ls_chart, col_ls_chart2 = st.columns(2)

with col_ls_chart:
    # Grouped bar: total vs defaulted value
    x      = np.arange(len(loan_size_exp))
    width  = 0.38
    total_m   = loan_size_exp["total_loan_value"].values / 1e6
    default_m = loan_size_exp["defaulted_loan_value"].values / 1e6

    fig_ls, ax_ls = plt.subplots(figsize=(7, 4), facecolor="white")
    ax_ls.set_facecolor("white")
    ax_ls.bar(x - width / 2, total_m,    width, label="Total Value",    color=BLUE,  alpha=0.75, edgecolor="white")
    ax_ls.bar(x + width / 2, default_m,  width, label="Defaulted Value",color=RED,   alpha=0.90, edgecolor="white")

    ax_ls.set_xticks(x)
    ax_ls.set_xticklabels(loan_size_exp["LoanSizeBand"].astype(str).tolist(),
                           fontsize=8.5, color=NAVY, rotation=20, ha="right")
    ax_ls.set_ylabel("Loan Value ($ Millions)", fontsize=8.5, color=MUTED)
    ax_ls.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}M"))
    ax_ls.set_title("Total vs Defaulted Value by Loan-Size Band",
                    fontsize=10, fontweight="700", color=NAVY, pad=8)
    ax_ls.legend(fontsize=8, framealpha=0)
    ax_ls.spines[["top", "right"]].set_visible(False)
    ax_ls.spines[["left", "bottom"]].set_color(BORDER)
    ax_ls.tick_params(colors=MUTED, labelsize=8)
    fig_ls.tight_layout()
    st.pyplot(fig_ls)
    plt.close(fig_ls)

with col_ls_chart2:
    # Exposure rate (defaulted / total) + portfolio share as dual-axis
    bands  = loan_size_exp["LoanSizeBand"].astype(str).tolist()
    exp_r  = loan_size_exp["exposure_pct"].values
    port_s = loan_size_exp["portfolio_share_pct"].values

    fig_ls2, ax1 = plt.subplots(figsize=(7, 4), facecolor="white")
    ax1.set_facecolor("white")
    ax2 = ax1.twinx()

    ax1.bar(bands, exp_r,  color=AMBER, alpha=0.8, edgecolor="white", label="Exposure Rate (%)")
    ax2.plot(bands, port_s, color=NAVY,  linewidth=2.2, marker="o",
             markersize=5, label="Portfolio Share (%)")

    ax1.set_ylabel("Exposure Rate (%)", fontsize=8.5, color=AMBER)
    ax2.set_ylabel("Portfolio Share (%)", fontsize=8.5, color=NAVY)
    ax1.set_title("Exposure Rate & Portfolio Share by Loan-Size Band",
                  fontsize=10, fontweight="700", color=NAVY, pad=8)
    ax1.tick_params(axis="x", rotation=20, labelsize=8)
    ax1.tick_params(axis="y", colors=AMBER, labelsize=8)
    ax2.tick_params(axis="y", colors=NAVY, labelsize=8)
    ax1.spines[["top"]].set_visible(False)
    ax1.spines[["left", "bottom"]].set_color(BORDER)
    ax2.spines[["top"]].set_visible(False)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, framealpha=0)
    fig_ls2.tight_layout()
    st.pyplot(fig_ls2)
    plt.close(fig_ls2)

with st.expander("📋 Loan-Size Band — Exposure Detail Table", expanded=False):
    show_table(
        loan_size_exp[["LoanSizeBand", "defaulted_loan_value", "total_loan_value",
                        "exposure_pct", "portfolio_share_pct",
                        "default_count", "total_count", "default_rate_pct"]],
        fmt={
            "defaulted_loan_value": "${:,.0f}",
            "total_loan_value":     "${:,.0f}",
            "exposure_pct":         "{:.2f}%",
            "portfolio_share_pct":  "{:.2f}%",
            "default_rate_pct":     "{:.2f}%",
        },
    )

st.markdown("<hr>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — Exposure by Income Quartile
# ─────────────────────────────────────────────────────────────────────────────
section_title("7. Exposure by Income Quartile")
st.markdown(
    "<p style='color:#5a6a80; font-size:0.85rem; margin-bottom:0.8rem;'>"
    "Identifies which income bracket carries the heaviest share of defaulted loan value.</p>",
    unsafe_allow_html=True,
)


@st.cache_data
def _exposure_by_income_quartile() -> pd.DataFrame:
    tmp = df.copy()
    q1, q2, q3 = tmp["Income"].quantile([0.25, 0.50, 0.75]).values
    tmp["IncomeQuartile"] = pd.cut(
        tmp["Income"],
        bins=[-np.inf, q1, q2, q3, np.inf],
        labels=[
            f"Q1 ≤${q1:,.0f}",
            f"Q2 ${q1:,.0f}–${q2:,.0f}",
            f"Q3 ${q2:,.0f}–${q3:,.0f}",
            f"Q4 >${q3:,.0f}",
        ],
    )
    def_tmp = tmp[tmp[TARGET_COLUMN] == 1].copy()

    grp_def   = def_tmp.groupby("IncomeQuartile", observed=True)["LoanAmount"].agg(
        defaulted_loan_value="sum", default_count="count"
    ).reset_index()
    grp_total = tmp.groupby("IncomeQuartile", observed=True)["LoanAmount"].agg(
        total_loan_value="sum", total_count="count"
    ).reset_index()

    out = grp_total.merge(grp_def, on="IncomeQuartile", how="left").fillna(0)
    out["defaulted_loan_value"] = out["defaulted_loan_value"].astype(float)
    out["default_count"]        = out["default_count"].astype(int)
    out["exposure_pct"]         = (out["defaulted_loan_value"] / out["total_loan_value"] * 100).round(2)
    out["portfolio_share_pct"]  = (out["defaulted_loan_value"] / _total_default_value * 100).round(2)
    out["default_rate_pct"]     = (out["default_count"] / out["total_count"] * 100).round(2)
    return out


income_exp = _exposure_by_income_quartile()

col_iq_left, col_iq_right = st.columns(2)

with col_iq_left:
    quartile_labels = income_exp["IncomeQuartile"].astype(str).tolist()
    default_vals_m  = (income_exp["defaulted_loan_value"] / 1e6).tolist()
    port_shares     = income_exp["portfolio_share_pct"].tolist()

    fig_iq = _hbar_exposure(
        labels    = quartile_labels,
        values_m  = default_vals_m,
        share_pcts= port_shares,
        colors    = INCOME_COLORS,
        title     = "Defaulted Value by Income Quartile",
    )
    st.pyplot(fig_iq)
    plt.close(fig_iq)

with col_iq_right:
    # Stacked proportion: defaulted vs performing within each quartile
    total_m_iq   = income_exp["total_loan_value"].values / 1e6
    default_m_iq = income_exp["defaulted_loan_value"].values / 1e6
    perform_m_iq = total_m_iq - default_m_iq

    fig_iq2, ax_iq = plt.subplots(figsize=(6, 3.8), facecolor="white")
    ax_iq.set_facecolor("white")
    x_iq = np.arange(len(quartile_labels))

    ax_iq.bar(x_iq, perform_m_iq, label="Performing",  color=BLUE,  alpha=0.80, edgecolor="white")
    ax_iq.bar(x_iq, default_m_iq, bottom=perform_m_iq,
              label="Defaulted", color=RED,   alpha=0.90, edgecolor="white")

    ax_iq.set_xticks(x_iq)
    ax_iq.set_xticklabels(quartile_labels, fontsize=8, color=NAVY, rotation=15, ha="right")
    ax_iq.set_ylabel("Loan Value ($ Millions)", fontsize=8.5, color=MUTED)
    ax_iq.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}M"))
    ax_iq.set_title("Performing vs Defaulted Value — Income Quartile",
                    fontsize=10, fontweight="700", color=NAVY, pad=8)
    ax_iq.legend(fontsize=8, framealpha=0)
    ax_iq.spines[["top", "right"]].set_visible(False)
    ax_iq.spines[["left", "bottom"]].set_color(BORDER)
    ax_iq.tick_params(colors=MUTED, labelsize=8)
    fig_iq2.tight_layout()
    st.pyplot(fig_iq2)
    plt.close(fig_iq2)

with st.expander("📋 Income Quartile — Exposure Detail Table", expanded=False):
    show_table(
        income_exp[["IncomeQuartile", "defaulted_loan_value", "total_loan_value",
                     "exposure_pct", "portfolio_share_pct",
                     "default_count", "total_count", "default_rate_pct"]],
        fmt={
            "defaulted_loan_value": "${:,.0f}",
            "total_loan_value":     "${:,.0f}",
            "exposure_pct":         "{:.2f}%",
            "portfolio_share_pct":  "{:.2f}%",
            "default_rate_pct":     "{:.2f}%",
        },
    )

st.markdown("<hr>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — Exposure Summary: Top Concentration Findings
# ─────────────────────────────────────────────────────────────────────────────
section_title("8. Key Exposure Concentration Findings")
st.markdown("<div style='margin-bottom:0.6rem;'></div>", unsafe_allow_html=True)


@st.cache_data
def _top_concentration_findings() -> list:
    """
    Return a list of finding dicts:
    {dimension, segment, defaulted_value, portfolio_share_pct, exposure_pct}
    representing the single highest-exposure segment in each dimension.
    """
    dims = {
        "Loan Purpose":     ("LoanPurpose",    purpose_exp),
        "Employment Type":  ("EmploymentType", employment_exp),
        "Education":        ("Education",      education_exp),
        "MaritalStatus":    ("MaritalStatus",  _exposure_by("MaritalStatus", _defaulted_df, df)),
    }
    findings = []
    for label, (col, exp_df) in dims.items():
        top = exp_df.iloc[0]
        findings.append({
            "dimension":          label,
            "top_segment":        top[col],
            "defaulted_value":    top["defaulted_loan_value"],
            "portfolio_share":    top["portfolio_share_pct"],
            "segment_exposure":   top["exposure_pct"],
        })
    return findings


findings = _top_concentration_findings()

cols_f = st.columns(len(findings))
for col_f, f in zip(cols_f, findings):
    with col_f:
        st.markdown(
            f"""
            <div style="background:#ffffff; border:1px solid {BORDER};
                        border-top:3px solid {RED}; border-radius:8px;
                        padding:1rem 1.1rem;
                        box-shadow:0 1px 4px rgba(10,35,66,0.06);
                        margin-bottom:0.5rem;">
              <div style="color:{MUTED}; font-size:0.70rem; font-weight:700;
                          text-transform:uppercase; letter-spacing:0.07em;
                          margin-bottom:0.25rem;">{f["dimension"]}</div>
              <div style="color:{NAVY}; font-size:1.05rem; font-weight:800;
                          margin-bottom:0.15rem;">{f["top_segment"]}</div>
              <div style="color:{RED}; font-size:1.4rem; font-weight:800;
                          line-height:1.1;">${f["defaulted_value"]/1e6:.1f}M</div>
              <div style="color:{MUTED}; font-size:0.78rem; margin-top:0.2rem;">
                {f["portfolio_share"]:.1f}% of total exposure<br>
                {f["segment_exposure"]:.2f}% segment exposure rate</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown(
    "<p style='color:#5a6a80; font-size:0.8rem; margin-top:0.6rem;'>"
    "<em>Each card shows the single segment with the highest defaulted loan value "
    "within its dimension. Portfolio share = segment's defaulted $ ÷ total portfolio defaulted $. "
    "Segment exposure rate = defaulted $ ÷ total $ within that segment.</em></p>",
    unsafe_allow_html=True,
)
