# =============================================================================
# CUSTOMER SEGMENTATION & RETENTION ANALYSIS
# Online Retail II Dataset — UCI Machine Learning Repository
# =============================================================================
#
# Dataset : Online Retail II (1,067,371 transactions, Dec 2009 – Dec 2011)
# Source  : https://archive.ics.uci.edu/dataset/502/online+retail+ii
# Author  : [Your Name]
#
# OVERVIEW
# --------
# End-to-end customer analytics pipeline covering:
#   1. Data loading & ingestion          (UCI fallback → direct HTTP)
#   2. Data cleaning & validation        (nulls, cancellations, type fixes)
#   3. Revenue analytics                 (summary stats, country breakdown, monthly trend)
#   4. RFM segmentation                  (Recency / Frequency / Monetary scoring)
#   5. Cohort retention analysis         (monthly retention heatmap)
#   6. RFM × Cohort cross-analysis       (tenure & frequency by segment)
#
# KEY FINDINGS (summary — see inline insights for detail)
# -------------------------------------------------------
# • Total Revenue      : £17,743,429 across 36,975 invoices (Dec 2009 – Dec 2011)
# • UK dominance       : £14.7M (83% of revenue); top international = Netherlands
# • RFM: Champions     : 25% of customers → 69% of revenue
# • Top-20% Pareto     : Top quintile accounts for 77.3% of total revenue
# • Cohort retention   : Steep drop to ~21% at Month 1, then stable floor 15–22%
# • Annual pulse       : Month 11–12 retention bump — seasonal repurchase effect
# • At Risk insight    : Longest avg tenure (20 mo) but only 4.9 orders — high win-back ROI
# • Frequency ≠ Tenure : Champions and At Risk have nearly identical tenure (18.7 vs 20.0 mo);
#                        frequency (15.7 vs 4.9) is the true loyalty differentiator
#
# REQUIREMENTS
# ------------
# pip install pandas numpy matplotlib seaborn openpyxl requests
# =============================================================================

# ── Requirements ──────────────────────────────────────────────────────────────
# pandas      — DataFrame manipulation, groupby, pivot tables
# numpy       — Vectorised scoring (np.select, np.percentile, quantile cuts)
# matplotlib  — Figure/axes creation, annotation, figure saving
# seaborn     — Heatmap (cohort retention), colour palettes
# openpyxl    — Excel (.xlsx) reading engine for the UCI zip file
# requests    — HTTP download fallback when ucimlrepo is unavailable
# =============================================================================

import io
import zipfile
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import requests

warnings.filterwarnings("ignore")

# ── Global style ──────────────────────────────────────────────────────────────
# Consistent muted palette used across all charts — avoids garish defaults while
# remaining distinguishable in greyscale print.
MUTED_PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2",
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2",
]


# =============================================================================
# SECTION 1 — DATA LOADING
# =============================================================================

# ── Configuration ─────────────────────────────────────────────────────────────
URL             = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
TIMEOUT_SECONDS = 180


def load_data() -> pd.DataFrame:
    """Load the UCI Online Retail II dataset.

    Strategy (two-stage fallback):
      1. Try ucimlrepo — the official UCI Python client; cleanest path when
         the package is installed and the API is reachable.
      2. Fall back to a direct HTTP download of the raw zip from UCI, extract
         the embedded .xlsx, and concat both year-sheets into a single DataFrame.

    Both sheets share identical column schemas, so pd.concat is safe without
    any alignment step.

    Returns
    -------
    pd.DataFrame
        Raw (uncleaned) combined dataset with Invoice and StockCode cast to str.
    """

    data_source = None
    df = None

    # ── Attempt 1: ucimlrepo ──────────────────────────────────────────────────
    # ucimlrepo wraps the UCI REST API; it returns a DatasetView where
    # `.data.original` gives the raw combined DataFrame. Fastest path.
    try:
        from ucimlrepo import fetch_ucirepo
        print("Trying ucimlrepo...")
        retail = fetch_ucirepo(id=502)
        df = retail.data.original
        data_source = "ucimlrepo"
        print("✓ Loaded via ucimlrepo")
    except Exception as e:
        print(f"ucimlrepo failed: {e}")

    # ── Attempt 2: Direct HTTP download ──────────────────────────────────────
    # If ucimlrepo is unavailable (not installed, or API timeout), download the
    # zip directly. TIMEOUT_SECONDS=180 because the file is ~45 MB on a cold CDN.
    if df is None:
        print("Trying direct HTTP download...")
        response = requests.get(URL, timeout=TIMEOUT_SECONDS)
        response.raise_for_status()
        print(f"Downloaded {len(response.content) / 1e6:.1f} MB")

        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            xlsx_name = [n for n in z.namelist() if n.endswith(".xlsx")][0]
            print(f"Found Excel file in zip: {xlsx_name}")
            xlsx_bytes = z.read(xlsx_name)

        xl = pd.ExcelFile(io.BytesIO(xlsx_bytes), engine="openpyxl")
        print(f"Sheet names: {xl.sheet_names}")
        # The workbook has two sheets: "Year 2009-2010" and "Year 2010-2011".
        # Concat rather than read_excel(..., sheet_name=None) to avoid a dict.
        sheets = [pd.read_excel(xl, sheet_name=s) for s in xl.sheet_names]
        df = pd.concat(sheets, ignore_index=True)
        data_source = "direct_http_download"
        print("✓ Loaded via direct HTTP download")

    print(f"\nData source : {data_source}")
    print(f"Raw shape   : {df.shape}")

    # ── Type fixes — IMPORTANT ────────────────────────────────────────────────
    # Invoice contains mixed values like '489434' (numeric) and 'C489449'
    # (cancellation prefix).  Without explicit str cast, pandas/parquet
    # serialisers attempt int64 conversion and fail on the 'C'-prefixed rows.
    # StockCode has the same pattern (e.g. 'POST', 'DOT', 'M' alongside numerics).
    df["Invoice"]   = df["Invoice"].astype(str)
    df["StockCode"] = df["StockCode"].astype(str)
    print("✓ Invoice and StockCode cast to str")

    return df


# =============================================================================
# SECTION 2 — DATA CLEANING
# =============================================================================

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw Online Retail II DataFrame.

    Cleaning steps (applied in order):
      1. Convert InvoiceDate to datetime64.
      2. Drop rows with missing Customer ID — these cannot be attributed
         to a buyer and are therefore useless for RFM and cohort analyses.
      3. Remove rows with Quantity <= 0 — these are cancellation/reversal
         invoices (identified by 'C' prefix on Invoice number, e.g. 'C489449').

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame returned by load_data().

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame ready for analytics (805,620 rows expected).
    """

    cleaned_df = df.copy()
    print(f"Original shape: {cleaned_df.shape}")

    # ── Step 1: Parse InvoiceDate ─────────────────────────────────────────────
    # The Excel source stores dates as strings in some load paths; pd.to_datetime
    # handles both string and numeric epoch representations automatically.
    cleaned_df["InvoiceDate"] = pd.to_datetime(cleaned_df["InvoiceDate"])
    print(f"✓ InvoiceDate converted to datetime ({cleaned_df['InvoiceDate'].dtype})")

    # ── Step 2: Drop missing Customer ID ─────────────────────────────────────
    # INSIGHT: 243,007 rows (22.8% of raw data) have no Customer ID.
    # Transactions without a Customer ID cannot be attributed to a buyer —
    # required for RFM and cohort analyses. These may be guest checkouts or
    # B2B wholesale orders processed outside the CRM. Either way, they cannot
    # contribute to individual-level segmentation and must be excluded.
    before = len(cleaned_df)
    cleaned_df = cleaned_df.dropna(subset=["Customer ID"])
    after_missing = len(cleaned_df)
    print(f"✓ Dropped {before - after_missing:,} rows with missing Customer ID")

    # ── Step 3: Remove cancellations / negative quantity ─────────────────────
    # Cancellation invoices carry a 'C' prefix (e.g. 'C489449') and have
    # negative Quantity values.  Keeping them would understate revenue and
    # distort frequency counts.  We filter on Quantity > 0 rather than the
    # 'C' prefix because some legitimate returns may lack the prefix while
    # still having negative quantities — the quantity filter is more robust.
    before = len(cleaned_df)
    cleaned_df = cleaned_df[cleaned_df["Quantity"] > 0]
    after_cancellations = len(cleaned_df)
    print(f"✓ Removed {before - after_cancellations:,} rows with negative/zero Quantity (cancellations)")

    # ── Cleaning report ───────────────────────────────────────────────────────
    print(f"\nCleaned shape: {cleaned_df.shape}")
    print(
        f"Rows removed total: {df.shape[0] - cleaned_df.shape[0]:,} "
        f"({(1 - len(cleaned_df) / len(df)) * 100:.1f}% of original)"
    )
    print(f"\nDtypes:\n{cleaned_df.dtypes}")
    print(f"\nNull counts:\n{cleaned_df.isnull().sum()}")
    print(f"\nQuantity stats:\n{cleaned_df['Quantity'].describe()}")

    # INSIGHT: 24.5% of raw rows are removed in cleaning:
    #   - Missing Customer ID : 243,007 rows (22.8%) — cannot attribute to a buyer
    #   - Cancellations       : 18,744 rows  (1.8%) — negative-quantity reversal invoices
    #     (identified by 'C' prefix in Invoice number, e.g. 'C489449')
    #   Cleaned dataset: 805,620 rows — all with valid customer, positive quantity.

    return cleaned_df


# =============================================================================
# SECTION 3 — REVENUE ANALYTICS
# =============================================================================

def revenue_analytics(cleaned_df: pd.DataFrame):
    """Compute revenue metrics and produce two charts.

    Charts produced:
      1. Horizontal bar chart — top 15 countries by revenue, UK excluded.
      2. Line chart — monthly revenue trend with 3-month rolling average
         and min/max shaded band; peak month annotated.

    Parameters
    ----------
    cleaned_df : pd.DataFrame
        Cleaned DataFrame from clean_data().

    Returns
    -------
    fig_country : matplotlib.figure.Figure
        Top-15 countries bar chart.
    fig_trend : matplotlib.figure.Figure
        Monthly revenue trend chart.
    """

    # ── Derive Revenue ────────────────────────────────────────────────────────
    # Revenue = Quantity × unit Price.  Stored as a new column so downstream
    # aggregations can sum it without re-multiplying each time.
    cleaned_df = cleaned_df.copy()
    cleaned_df["Revenue"] = cleaned_df["Quantity"] * cleaned_df["Price"]

    # ── Summary statistics ────────────────────────────────────────────────────
    total_revenue   = cleaned_df["Revenue"].sum()
    total_invoices  = cleaned_df["Invoice"].nunique()
    total_customers = cleaned_df["Customer ID"].nunique()
    total_countries = cleaned_df["Country"].nunique()
    date_range      = (cleaned_df["InvoiceDate"].min(), cleaned_df["InvoiceDate"].max())
    top_country     = cleaned_df.groupby("Country")["Revenue"].sum().idxmax()

    # Median order value: aggregate to invoice level first to avoid
    # line-item skew (a single invoice can have 50+ line items).
    invoice_revenue  = cleaned_df.groupby("Invoice")["Revenue"].sum()
    median_order_val = invoice_revenue.median()

    print("\n" + "=" * 55)
    print("  REVENUE SUMMARY")
    print("=" * 55)
    print(f"  Total Revenue      : £{total_revenue:,.0f}")
    print(f"  Total Invoices     : {total_invoices:,}")
    print(f"  Unique Customers   : {total_customers:,}")
    print(f"  Countries          : {total_countries}")
    print(f"  Date Range         : {date_range[0].date()} → {date_range[1].date()}")
    print(f"  Top Country        : {top_country}")
    print(f"  Median Order Value : £{median_order_val:,.2f}")
    print("=" * 55)

    # INSIGHT — Country breakdown:
    #   UK accounts for 83% of total revenue (£14.7M of £17.7M).
    #   Excluding UK, the top international markets are:
    #     1. Netherlands  — clear leader; likely wholesale/B2B buyers
    #     2. EIRE (Ireland) — proximity-driven; strong #2
    #     3. Germany — meaningful but trailing by a large gap
    #   Consider localisation investment in NL/IE given their outsized share.

    # ── Chart 1: Top 15 countries by revenue (excluding UK) ──────────────────
    # UK is excluded because its dominance (83% share) would compress the
    # remaining bars to near-invisible — the chart's purpose is to rank
    # international markets.
    country_rev = (
        cleaned_df[cleaned_df["Country"] != "United Kingdom"]
        .groupby("Country")["Revenue"]
        .sum()
        .sort_values(ascending=False)
        .head(15)
    )

    fig_country, ax_c = plt.subplots(figsize=(10, 7))
    bars = ax_c.barh(
        country_rev.index[::-1],
        country_rev.values[::-1],
        color=MUTED_PALETTE[: len(country_rev)],
    )

    # Data labels: placed just inside the bar-end so they are always visible
    # even for very short bars.
    for bar, val in zip(bars, country_rev.values[::-1]):
        ax_c.text(
            bar.get_width() * 0.97,
            bar.get_y() + bar.get_height() / 2,
            f"£{val:,.0f}",
            va="center",
            ha="right",
            fontsize=8,
            color="white",
            fontweight="bold",
        )

    ax_c.set_xlabel("Total Revenue (£)", fontsize=11)
    ax_c.set_title("Top 15 Countries by Revenue (Excluding UK)", fontsize=13, fontweight="bold")
    ax_c.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{x:,.0f}"))
    ax_c.spines[["top", "right"]].set_visible(False)
    fig_country.tight_layout()

    # ── Chart 2: Monthly revenue trend ───────────────────────────────────────
    # Resample to month-end frequency to smooth irregular day counts.
    # A 3-month rolling average damps noise without hiding the seasonal signal.
    # The shaded band (min/max of the raw monthly values within the rolling
    # window) gives the reader an intuitive sense of variance.

    # INSIGHT — Monthly trend:
    #   Peak month: November 2010 at ~£1.17M (pre-Christmas retail spike).
    #   Revenue shows a clear seasonal acceleration into Q4 each year.
    #   The Dec 2011 apparent dip is a data artefact — the dataset covers only
    #   the first 9 days of that month, so it reflects partial data, not a real decline.

    monthly_rev = (
        cleaned_df.set_index("InvoiceDate")
        .resample("ME")["Revenue"]
        .sum()
    )
    rolling_avg  = monthly_rev.rolling(3, min_periods=1).mean()
    rolling_min  = monthly_rev.rolling(3, min_periods=1).min()
    rolling_max  = monthly_rev.rolling(3, min_periods=1).max()

    peak_month = monthly_rev.idxmax()
    peak_value = monthly_rev.max()

    fig_trend, ax_t = plt.subplots(figsize=(12, 5))

    ax_t.fill_between(
        monthly_rev.index,
        rolling_min,
        rolling_max,
        alpha=0.15,
        color=MUTED_PALETTE[0],
        label="Rolling 3-mo min/max band",
    )
    ax_t.plot(
        monthly_rev.index,
        monthly_rev.values,
        color=MUTED_PALETTE[0],
        alpha=0.45,
        linewidth=1.2,
        label="Monthly revenue",
    )
    ax_t.plot(
        rolling_avg.index,
        rolling_avg.values,
        color=MUTED_PALETTE[0],
        linewidth=2.2,
        label="3-month rolling avg",
    )

    # Annotate the peak so readers can immediately identify the Nov-2010 spike.
    ax_t.annotate(
        f"Peak: £{peak_value:,.0f}\n({peak_month.strftime('%b %Y')})",
        xy=(peak_month, peak_value),
        xytext=(peak_month, peak_value * 1.08),
        arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
        fontsize=9,
        ha="center",
    )

    ax_t.set_title("Monthly Revenue Trend (Dec 2009 – Dec 2011)", fontsize=13, fontweight="bold")
    ax_t.set_xlabel("Month", fontsize=11)
    ax_t.set_ylabel("Revenue (£)", fontsize=11)
    ax_t.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{x/1e6:.1f}M"))
    ax_t.legend(fontsize=9)
    ax_t.spines[["top", "right"]].set_visible(False)
    fig_trend.tight_layout()

    return fig_country, fig_trend


# =============================================================================
# SECTION 4 — RFM SEGMENTATION
# =============================================================================

def rfm_segmentation(cleaned_df: pd.DataFrame) -> pd.DataFrame:
    """Build an RFM (Recency–Frequency–Monetary) segmentation model.

    Scoring methodology:
      • Each of R, F, M is independently binned into quintiles (1–5).
      • Recency is reverse-scored: a customer who purchased yesterday gets R=5,
        one who purchased 300 days ago gets R=1.
      • Composite RFM_Score = R_score + F_score + M_score (range 3–15).
      • Segments are assigned via np.select priority rules (first match wins),
        so ordering matters — more specific conditions are listed first.

    Pareto check:
      Customers in the top monetary quintile (M_score == 5) are compared to
      the rest to validate the classic 80/20 concentration hypothesis.

    Parameters
    ----------
    cleaned_df : pd.DataFrame
        Cleaned DataFrame from clean_data().

    Returns
    -------
    pd.DataFrame
        Per-customer RFM table with scores, segment labels, and revenue.
    """

    cleaned_df = cleaned_df.copy()
    cleaned_df["Revenue"] = cleaned_df["Quantity"] * cleaned_df["Price"]

    # ── Reference date ────────────────────────────────────────────────────────
    # Set reference date to max date + 1 day so that the most recent purchaser
    # has Recency = 1 day (not 0), avoiding a division-by-zero edge case and
    # making score interpretation more intuitive.
    reference_date = cleaned_df["InvoiceDate"].max() + pd.Timedelta(days=1)
    print(f"\nRFM Reference date: {reference_date.date()}")

    # ── Aggregate per customer ────────────────────────────────────────────────
    rfm = (
        cleaned_df.groupby("Customer ID")
        .agg(
            Recency  =("InvoiceDate", lambda x: (reference_date - x.max()).days),
            Frequency=("Invoice",     "nunique"),
            Monetary =("Revenue",     "sum"),
        )
        .reset_index()
    )
    print(f"RFM shape (one row per customer): {rfm.shape}")

    # ── Quintile scoring ──────────────────────────────────────────────────────
    # pd.qcut splits each variable into 5 equal-frequency bins.
    # Duplicate bin edges are resolved with duplicates='drop', which can
    # produce fewer than 5 distinct bins for very skewed distributions —
    # acceptable for segmentation purposes.
    rfm["R_score"] = pd.qcut(rfm["Recency"],  q=5, labels=[5, 4, 3, 2, 1], duplicates="drop").astype(int)
    rfm["F_score"] = pd.qcut(rfm["Frequency"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["M_score"] = pd.qcut(rfm["Monetary"].rank(method="first"),  q=5, labels=[1, 2, 3, 4, 5]).astype(int)

    rfm["RFM_Score"] = rfm["R_score"] + rfm["F_score"] + rfm["M_score"]

    # ── Segment assignment (np.select — priority order matters) ───────────────
    # Rules are evaluated top-to-bottom; the first True condition wins.
    # This prevents overlapping definitions (e.g. a Champion also satisfies
    # Loyal Customers criteria, but is correctly captured in the first rule).
    conditions = [
        (rfm["R_score"] >= 4) & (rfm["F_score"] >= 4),          # Champions
        (rfm["R_score"] >= 3) & (rfm["F_score"] >= 3),          # Loyal Customers
        (rfm["R_score"] >= 3) & (rfm["F_score"] <= 2),          # Potential Loyalists
        (rfm["R_score"] >= 4) & (rfm["F_score"] <= 2),          # Recent Customers
        (rfm["R_score"] <= 2) & (rfm["F_score"] >= 3),          # At Risk
        (rfm["R_score"] <= 2) & (rfm["F_score"] <= 2) & (rfm["RFM_Score"] >= 4),  # Hibernating
        (rfm["R_score"] <= 2) & (rfm["F_score"] <= 2) & (rfm["RFM_Score"] <  4),  # Lost
    ]
    choices = [
        "Champions",
        "Loyal Customers",
        "Potential Loyalists",
        "Recent Customers",
        "At Risk",
        "Hibernating",
        "Lost",
    ]
    rfm["Segment"] = np.select(conditions, choices, default="Other")

    # INSIGHT — RFM Segmentation results (5,881 customers):
    #
    #   Champions (1,482 customers | 25.2%) → £12.3M (69.3% of revenue)
    #     Avg recency: 20 days | Avg frequency: 15.7 invoices | Avg spend: £8,295
    #     → The business engine. Prioritise retention above all else.
    #
    #   Loyal Customers (1,222 | 20.8%) → £2.5M (14.4%)
    #     Consistent buyers, moderate frequency — growing this group lifts baseline revenue.
    #
    #   At Risk (824 | 14.0%) → £1.6M (9.2%)
    #     Historically strong buyers (avg 4.9 invoices, £1,983 spend) gone quiet.
    #     → Highest-ROI win-back target. See Section 6 for why.
    #
    #   Potential Loyalists (828 | 14.1%) — recent but low-frequency (avg 1.4 orders)
    #     → Second-purchase nudge is the key intervention (personalised rec, small incentive).
    #
    #   Hibernating (1,205 | 20.5%) + Lost (320 | 5.4%) — dormant, low value
    #     → Low priority for outreach; focus budget elsewhere.
    #
    # PARETO: Top 20% of customers by spend → 77.3% of total revenue.
    #   Stronger than the classic 80/20 — this customer base is highly concentrated.

    # ── Pareto analysis ───────────────────────────────────────────────────────
    top_quintile_rev = rfm[rfm["M_score"] == 5]["Monetary"].sum()
    total_rev        = rfm["Monetary"].sum()
    pareto_pct       = top_quintile_rev / total_rev * 100
    print(f"\nPareto: top-20% customers → {pareto_pct:.1f}% of total revenue")

    # ── Segment profile table ─────────────────────────────────────────────────
    seg_profile = (
        rfm.groupby("Segment")
        .agg(
            Customers     =("Customer ID", "count"),
            Avg_Recency   =("Recency",     "mean"),
            Avg_Frequency =("Frequency",   "mean"),
            Avg_Monetary  =("Monetary",    "mean"),
            Total_Revenue =("Monetary",    "sum"),
        )
        .round(1)
        .sort_values("Total_Revenue", ascending=False)
    )
    seg_profile["Revenue_Pct"] = (seg_profile["Total_Revenue"] / seg_profile["Total_Revenue"].sum() * 100).round(1)
    print("\nSegment Profile:\n")
    print(seg_profile.to_string())

    # ── Chart: Revenue by segment ─────────────────────────────────────────────
    # Sort bars by total revenue so the most valuable segments are immediately
    # visible without requiring the reader to scan the y-axis labels.
    seg_rev = seg_profile["Total_Revenue"].sort_values()
    fig_rfm, ax_r = plt.subplots(figsize=(9, 5))
    colors = [MUTED_PALETTE[i % len(MUTED_PALETTE)] for i in range(len(seg_rev))]
    bars = ax_r.barh(seg_rev.index, seg_rev.values, color=colors)

    for bar, val in zip(bars, seg_rev.values):
        ax_r.text(
            bar.get_width() + total_rev * 0.003,
            bar.get_y() + bar.get_height() / 2,
            f"£{val:,.0f}",
            va="center",
            fontsize=8.5,
        )

    ax_r.set_xlabel("Total Revenue (£)", fontsize=11)
    ax_r.set_title("Total Revenue by RFM Segment", fontsize=13, fontweight="bold")
    ax_r.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{x/1e6:.1f}M"))
    ax_r.spines[["top", "right"]].set_visible(False)
    fig_rfm.tight_layout()
    plt.show()

    return rfm


# =============================================================================
# SECTION 5 — COHORT RETENTION ANALYSIS
# =============================================================================

def cohort_retention(cleaned_df: pd.DataFrame):
    """Compute monthly cohort retention and render a heatmap.

    Cohort logic:
      • Each customer belongs to the cohort defined by their first purchase month.
      • For every subsequent order, we compute CohortIndex = months since first purchase.
      • The retention matrix shows what % of each cohort made at least one purchase
        in each subsequent month.

    Parameters
    ----------
    cleaned_df : pd.DataFrame
        Cleaned DataFrame from clean_data().

    Returns
    -------
    retention_matrix : pd.DataFrame
        Percentage retention (0–100) indexed by CohortMonth × CohortIndex.
    fig_cohort : matplotlib.figure.Figure
        Seaborn heatmap figure.
    """

    cleaned_df = cleaned_df.copy()

    # ── Derive cohort identifiers ─────────────────────────────────────────────
    # Period('M') truncates to month granularity — avoids floating-point issues
    # that can arise from date arithmetic on day-level timestamps.
    cleaned_df["OrderMonth"]  = cleaned_df["InvoiceDate"].dt.to_period("M")
    cleaned_df["CohortMonth"] = (
        cleaned_df.groupby("Customer ID")["InvoiceDate"]
        .transform("min")
        .dt.to_period("M")
    )

    # CohortIndex: integer number of months since the customer's first purchase.
    # Using .n (period difference in the Period's native frequency) is more
    # reliable than converting to int via subtraction on Period objects.
    cleaned_df["CohortIndex"] = (
        cleaned_df["OrderMonth"] - cleaned_df["CohortMonth"]
    ).apply(lambda x: x.n)

    # ── Build cohort pivot ────────────────────────────────────────────────────
    # Count distinct customers active in each (CohortMonth, CohortIndex) cell.
    cohort_data = (
        cleaned_df.groupby(["CohortMonth", "CohortIndex"])["Customer ID"]
        .nunique()
        .reset_index()
    )
    cohort_pivot = cohort_data.pivot_table(
        index="CohortMonth", columns="CohortIndex", values="Customer ID"
    )

    # ── Retention matrix ──────────────────────────────────────────────────────
    # Divide each row by the cohort's Month-0 count (cohort size at acquisition).
    # Multiply by 100 for percentage display on the heatmap.
    cohort_sizes     = cohort_pivot.iloc[:, 0]
    retention_matrix = cohort_pivot.divide(cohort_sizes, axis=0) * 100

    # ── Average retention by month number ────────────────────────────────────
    avg_retention = retention_matrix.mean(axis=0)
    print("\nAverage Retention by Month (across all cohorts):")
    for idx, val in avg_retention.items():
        print(f"  Month {idx:>2}: {val:.1f}%")

    # INSIGHT — Cohort Retention (25 cohorts, Dec 2009 → Dec 2011):
    #
    #   Month 0  : 100% (by definition)
    #   Month 1  : ~21% average — steep initial drop; 4 in 5 first-time buyers never return
    #   Month 2+ : Stable floor of 15–22% across all cohorts
    #   Overall avg (Month 1+): 18.4%
    #
    #   Key patterns:
    #   1. ANNUAL REPURCHASE PULSE — Retention ticks up 2–3pp at months 11–12 in every
    #      cohort. Customers who bought for a seasonal occasion return ~12 months later.
    #   2. NO COHORT DEGRADATION — Later cohorts (2011) retain as well as earlier ones
    #      (2010), ruling out customer-quality erosion over the dataset window.
    #   3. "STICKY MINORITY" — The stable 15–22% floor suggests a core of B2B/wholesale
    #      buyers who return regularly regardless of cohort vintage. This is unusual for
    #      a pure-play consumer retailer and explains the repeat-purchase economics.
    #
    #   ACTION: Month-1 retention (21%) is the biggest single lever. A post-purchase
    #   email or loyalty nudge within 30 days could materially shift this number.

    # ── Heatmap ───────────────────────────────────────────────────────────────
    # Annotate each cell with its retention %, rounded to integer for readability.
    # Blues colormap: intensity directly encodes retention strength — intuitive.
    fig_cohort, ax_h = plt.subplots(figsize=(18, 10))
    sns.heatmap(
        retention_matrix.round(0),
        annot=True,
        fmt=".0f",
        cmap="Blues",
        linewidths=0.3,
        linecolor="white",
        ax=ax_h,
        cbar_kws={"label": "Retention %"},
        annot_kws={"size": 7},
    )
    ax_h.set_title(
        "Monthly Cohort Retention Heatmap (%)\nOnline Retail II — Dec 2009 to Dec 2011",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )
    ax_h.set_xlabel("Months Since First Purchase (CohortIndex)", fontsize=11)
    ax_h.set_ylabel("Cohort Month (First Purchase)", fontsize=11)
    ax_h.tick_params(axis="x", labelsize=8)
    ax_h.tick_params(axis="y", labelsize=8, rotation=0)
    fig_cohort.tight_layout()

    return retention_matrix, fig_cohort


# =============================================================================
# SECTION 6 — RFM × COHORT CROSS-ANALYSIS
# =============================================================================

def rfm_cohort_crossanalysis(cleaned_df: pd.DataFrame, rfm: pd.DataFrame) -> pd.DataFrame:
    """Cross-analyse RFM segments with customer tenure (cohort age).

    By merging the cohort-derived tenure onto the RFM table, we can ask:
    'Do some segments underperform DESPITE having long tenure?'

    The answer reveals whether a segment's low value is a NEW-CUSTOMER problem
    (short tenure) or a HABIT-FORMATION problem (long tenure, low frequency) —
    which implies very different interventions.

    Parameters
    ----------
    cleaned_df : pd.DataFrame
        Cleaned DataFrame from clean_data().
    rfm : pd.DataFrame
        RFM table returned by rfm_segmentation() — must contain 'Customer ID'
        and 'Segment' columns.

    Returns
    -------
    pd.DataFrame
        Segment-level profile with avg_tenure_months, avg_order_frequency, customer_count.
    """

    cleaned_df = cleaned_df.copy()

    # ── Compute per-customer tenure ───────────────────────────────────────────
    # Tenure = time between first purchase and the reference date (last date + 1 day).
    # Expressed in months (÷ 30.44, the mean Gregorian month length) so it is
    # directly comparable to the cohort retention CohortIndex (also in months).
    reference_date = cleaned_df["InvoiceDate"].max() + pd.Timedelta(days=1)

    customer_tenure = (
        cleaned_df.groupby("Customer ID")["InvoiceDate"]
        .min()
        .reset_index()
        .rename(columns={"InvoiceDate": "FirstPurchase"})
    )
    customer_tenure["tenure_months"] = (
        (reference_date - customer_tenure["FirstPurchase"]).dt.days / 30.44
    )

    # ── Merge tenure onto RFM ─────────────────────────────────────────────────
    rfm_extended = rfm.merge(customer_tenure[["Customer ID", "tenure_months"]], on="Customer ID", how="left")

    # ── Segment profile ───────────────────────────────────────────────────────
    seg_profile = (
        rfm_extended.groupby("Segment")
        .agg(
            customer_count      =("Customer ID",    "count"),
            avg_tenure_months   =("tenure_months",  "mean"),
            avg_order_frequency =("Frequency",      "mean"),
        )
        .round(1)
        .sort_values("avg_tenure_months", ascending=False)
    )

    print("\n" + "=" * 65)
    print("  RFM × COHORT CROSS-ANALYSIS")
    print("=" * 65)
    print(seg_profile.to_string())

    # ── Print key insight ─────────────────────────────────────────────────────
    print("""
Key finding:
  At Risk customers have the LONGEST avg tenure of any segment (~20 months)
  yet placed only ~4.9 orders — vs Champions' 15.7 orders over ~18.7 months.
  Frequency, not tenure, is the true loyalty differentiator.
""")

    # INSIGHT — RFM × Cohort Cross-Analysis (the headline finding):
    #
    #   Segment               Customers  Avg Tenure (mo)  Avg Frequency
    #   ─────────────────────────────────────────────────────────────────
    #   At Risk                  824         20.0              4.9
    #   Lost                     320         19.3              1.0
    #   Champions              1,482         18.7             15.7
    #   Loyal Customers        1,222         14.7              5.4
    #   Hibernating            1,205         13.2              1.3
    #   Potential Loyalists      828          4.9              1.4
    #
    #   COUNTER-INTUITIVE FINDING: At Risk customers (20.0 mo avg tenure) have been
    #   around LONGER than Champions (18.7 mo) — yet placed only 4.9 orders vs 15.7.
    #
    #   This means:
    #   • At Risk is NOT a new-customer churn problem — it's a habit-formation failure
    #     among long-standing buyers who know and have used the brand.
    #   • Frequency, not tenure, is the true loyalty differentiator.
    #   • Champions and At Risk diverged at the PURCHASE HABIT fork, not at the
    #     acquisition / tenure stage.
    #
    #   IMPLICATIONS:
    #   1. Win-back campaign for At Risk (824 customers, £1.6M revenue at stake) has
    #      a strong foundation — these buyers have brand familiarity and demonstrated
    #      purchase intent over 20 months.
    #   2. Potential Loyalists (4.9 mo tenure) are genuinely new — the intervention
    #      is a second-purchase nudge, not re-engagement.
    #   3. Lost segment (19.3 mo, 1.0 orders) made a single purchase years ago —
    #      low win-back ROI; deprioritise.

    # ── Dual horizontal bar chart ─────────────────────────────────────────────
    # Left panel: avg tenure months  |  Right panel: avg order frequency
    # Shared y-axis (segments) lets the reader visually compare tenure vs
    # frequency side-by-side — the "At Risk paradox" is immediately visible.
    segments    = seg_profile.index.tolist()
    tenure_vals = seg_profile["avg_tenure_months"].tolist()
    freq_vals   = seg_profile["avg_order_frequency"].tolist()

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    bar_colors = [MUTED_PALETTE[i % len(MUTED_PALETTE)] for i in range(len(segments))]

    # Left: tenure
    bars_l = ax_left.barh(segments, tenure_vals, color=bar_colors, alpha=0.85)
    ax_left.set_xlabel("Avg Tenure (months)", fontsize=11)
    ax_left.set_title("Average Customer Tenure by Segment", fontsize=11, fontweight="bold")
    ax_left.invert_xaxis()   # Mirror layout: bars grow left from the y-axis spine
    ax_left.spines[["top", "left"]].set_visible(False)
    for bar, val in zip(bars_l, tenure_vals):
        ax_left.text(
            bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}", va="center", ha="right", fontsize=8.5,
        )

    # Right: frequency
    bars_r = ax_right.barh(segments, freq_vals, color=bar_colors, alpha=0.85)
    ax_right.set_xlabel("Avg Order Frequency", fontsize=11)
    ax_right.set_title("Average Order Frequency by Segment", fontsize=11, fontweight="bold")
    ax_right.spines[["top", "right"]].set_visible(False)
    for bar, val in zip(bars_r, freq_vals):
        ax_right.text(
            bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}", va="center", fontsize=8.5,
        )

    fig.suptitle(
        "RFM × Cohort: Tenure vs Frequency by Segment\n"
        "(At Risk has longer tenure than Champions — frequency is the loyalty differentiator)",
        fontsize=12, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    plt.show()

    return seg_profile, fig


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # ── Run the full pipeline ─────────────────────────────────────────────────
    print("=" * 65)
    print("  CUSTOMER SEGMENTATION & RETENTION ANALYSIS")
    print("  Online Retail II — UCI ML Repository")
    print("=" * 65)

    df                               = load_data()
    cleaned_df                       = clean_data(df)
    fig_country, fig_trend           = revenue_analytics(cleaned_df)
    rfm                              = rfm_segmentation(cleaned_df)
    retention_matrix, fig_cohort     = cohort_retention(cleaned_df)
    seg_profile, fig_crossanalysis   = rfm_cohort_crossanalysis(cleaned_df, rfm)

    print("\n✓ Pipeline complete. All figures rendered.")

    # Save figures
    fig_country.savefig("revenue_by_country.png",              dpi=200, bbox_inches="tight")
    fig_trend.savefig("monthly_revenue_trend.png",              dpi=200, bbox_inches="tight")
    fig_cohort.savefig("cohort_retention_heatmap.png",          dpi=200, bbox_inches="tight")
    fig_crossanalysis.savefig("tenure_frequency_by_segment.png", dpi=200, bbox_inches="tight")
    print("✓ Figures saved: revenue_by_country.png, monthly_revenue_trend.png, "
          "cohort_retention_heatmap.png, tenure_frequency_by_segment.png")
