"""
analytics/engine.py — Analytics Engine
────────────────────────────────────────
Computes all business KPIs, trend analyses, and growth/decline signals
from the processed master sales DataFrame.

All public functions accept a pd.DataFrame produced by data/pipeline.py
and return DataFrames or dicts that the dashboard and recommendation
engine can consume directly.
"""
import os
import pandas as pd
import numpy as np
from scipy import stats


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pct_change(a: float, b: float) -> float | None:
    """Safe percentage change from b → a."""
    if b is None or b == 0:
        return None
    return round((a - b) / abs(b), 4)


def _load_master() -> pd.DataFrame:
    """Load the processed master parquet (fallback for standalone use)."""
    path = os.path.join(os.path.dirname(__file__), "..", "data", "processed",
                        "master_sales.parquet")
    return pd.read_parquet(path)


# ── 1. Overall KPI Summary ────────────────────────────────────────────────────

def overall_kpis(df: pd.DataFrame) -> dict:
    """
    Returns high-level summary KPIs for the entire dataset.

    Keys: total_revenue, total_orders, total_units, avg_order_value,
          gross_profit, profit_margin, unique_customers, unique_products.
    """
    rev    = df["revenue"].sum()
    profit = df["gross_profit"].sum() if "gross_profit" in df.columns else 0
    return {
        "total_revenue":    round(float(rev), 2),
        "total_orders":     int(len(df)),
        "total_units":      int(df["quantity"].sum()),
        "avg_order_value":  round(float(df["revenue"].mean()), 2),
        "gross_profit":     round(float(profit), 2),
        "profit_margin":    round(float(profit / rev) if rev else 0, 4),
        "unique_customers": int(df["customer_id"].nunique()),
        "unique_products":  int(df["product_id"].nunique()),
    }


# ── 2. Monthly Revenue Trend ──────────────────────────────────────────────────

def monthly_revenue_trend(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
        year_month, total_revenue, total_orders, total_units,
        avg_order_value, mom_growth, yoy_growth
    sorted ascending by year_month.
    """
    monthly = (
        df.groupby("year_month")
        .agg(
            total_revenue=("revenue",   "sum"),
            total_orders= ("sale_id",   "count"),
            total_units=  ("quantity",  "sum"),
        )
        .reset_index()
        .sort_values("year_month")
    )
    monthly["avg_order_value"] = (
        monthly["total_revenue"] / monthly["total_orders"]
    ).round(2)
    monthly["mom_growth"] = monthly["total_revenue"].pct_change().round(4)
    monthly["yoy_growth"] = monthly["total_revenue"].pct_change(12).round(4)
    return monthly


# ── 3. Category Performance ───────────────────────────────────────────────────

def category_performance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Revenue, profit margin, order count and trend per product category.
    """
    perf = (
        df.groupby("category_name")
        .agg(
            total_revenue= ("revenue",      "sum"),
            total_orders=  ("sale_id",      "count"),
            total_units=   ("quantity",     "sum"),
            gross_profit=  ("gross_profit", "sum"),
        )
        .reset_index()
    )
    perf["profit_margin"] = (perf["gross_profit"] / perf["total_revenue"]).round(4)
    perf["revenue_share"] = (perf["total_revenue"] / perf["total_revenue"].sum()).round(4)
    perf = perf.sort_values("total_revenue", ascending=False).reset_index(drop=True)
    return perf


# ── 4. Regional Performance ───────────────────────────────────────────────────

def regional_performance(df: pd.DataFrame) -> pd.DataFrame:
    """Revenue and orders broken down by region."""
    reg = (
        df.groupby("region")
        .agg(
            total_revenue=   ("revenue",     "sum"),
            total_orders=    ("sale_id",     "count"),
            unique_customers=("customer_id", "nunique"),
            total_units=     ("quantity",    "sum"),
        )
        .reset_index()
    )
    reg["avg_order_value"] = (reg["total_revenue"] / reg["total_orders"]).round(2)
    reg["revenue_share"]   = (reg["total_revenue"] / reg["total_revenue"].sum()).round(4)
    reg = reg.sort_values("total_revenue", ascending=False).reset_index(drop=True)
    return reg


# ── 5. Product Performance ────────────────────────────────────────────────────

def product_performance(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Top N products by revenue with profit margin."""
    prod = (
        df.groupby(["product_id", "product_name", "category_name"])
        .agg(
            total_revenue= ("revenue",      "sum"),
            total_orders=  ("sale_id",      "count"),
            total_units=   ("quantity",     "sum"),
            gross_profit=  ("gross_profit", "sum"),
        )
        .reset_index()
    )
    prod["profit_margin"] = (prod["gross_profit"] / prod["total_revenue"]).round(4)
    prod = prod.sort_values("total_revenue", ascending=False).head(top_n)
    return prod.reset_index(drop=True)


# ── 6. Customer Segment Analysis ─────────────────────────────────────────────

def customer_segment_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Revenue and order metrics broken down by customer segment."""
    seg = (
        df.groupby("segment")
        .agg(
            total_revenue=   ("revenue",     "sum"),
            total_orders=    ("sale_id",     "count"),
            unique_customers=("customer_id", "nunique"),
            avg_order_value= ("revenue",     "mean"),
        )
        .reset_index()
    )
    seg["revenue_share"] = (seg["total_revenue"] / seg["total_revenue"].sum()).round(4)
    seg["avg_order_value"] = seg["avg_order_value"].round(2)
    return seg.sort_values("total_revenue", ascending=False).reset_index(drop=True)


# ── 7. Growth / Decline Signal per Product ───────────────────────────────────

def product_growth_signals(
    df: pd.DataFrame,
    window_months: int = 3,
) -> pd.DataFrame:
    """
    Compare the last <window_months> months vs the previous <window_months>
    months for each product and label each as:
        Growing | Declining | Stable | New
    """
    df = df.copy()
    df["order_date"] = pd.to_datetime(df["order_date"])
    cutoff  = df["order_date"].max()
    recent_start  = cutoff - pd.DateOffset(months=window_months)
    prev_start    = cutoff - pd.DateOffset(months=window_months * 2)

    recent = (
        df[df["order_date"] > recent_start]
        .groupby("product_id")["revenue"].sum()
        .rename("rev_recent")
    )
    prev = (
        df[(df["order_date"] > prev_start) & (df["order_date"] <= recent_start)]
        .groupby("product_id")["revenue"].sum()
        .rename("rev_prev")
    )

    signals = (
        pd.concat([recent, prev], axis=1)
        .fillna(0)
        .reset_index()
    )
    # Merge product names
    prod_names = df[["product_id", "product_name", "category_name"]].drop_duplicates("product_id")
    signals = signals.merge(prod_names, on="product_id", how="left")

    signals["growth_rate"] = signals.apply(
        lambda r: _pct_change(r["rev_recent"], r["rev_prev"]), axis=1
    )
    signals["trend"] = signals.apply(
        lambda r: "New"      if r["rev_prev"] == 0 and r["rev_recent"] > 0
             else "Growing"  if r.get("growth_rate") is not None and r["growth_rate"] > 0.05
             else "Declining" if r.get("growth_rate") is not None and r["growth_rate"] < -0.05
             else "Stable",
        axis=1
    )
    return signals.sort_values("growth_rate", ascending=False, na_position="last")


# ── 8. Seasonal Analysis ─────────────────────────────────────────────────────

def seasonal_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Average monthly revenue indexed by calendar month to detect seasonality."""
    monthly = (
        df.groupby(["year", "month"])["revenue"]
        .sum()
        .reset_index()
    )
    season = (
        monthly.groupby("month")["revenue"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
        .rename(columns={"mean": "avg_revenue", "std": "std_revenue",
                         "min": "min_revenue", "max": "max_revenue"})
    )
    season["month_name"] = pd.to_datetime(season["month"], format="%m").dt.strftime("%B")
    return season


# ── 9. Sales Velocity (rolling 30-day) ────────────────────────────────────────

def sales_velocity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Daily revenue aggregated with a 30-day rolling average — useful for
    spotting acceleration / deceleration in sales pace.
    """
    daily = (
        df.groupby("order_date")["revenue"]
        .sum()
        .reset_index()
        .sort_values("order_date")
    )
    daily["rolling_30d_revenue"] = (
        daily["revenue"].rolling(window=30, min_periods=1).mean().round(2)
    )
    daily["rolling_7d_revenue"] = (
        daily["revenue"].rolling(window=7, min_periods=1).mean().round(2)
    )
    return daily


# ── 10. Linear Trend Slope ────────────────────────────────────────────────────

def compute_trend_slope(df: pd.DataFrame, group_col: str = "year_month") -> dict:
    """
    Fits a linear regression on monthly revenue and returns the slope
    (revenue change per month), R², and trend direction label.
    """
    monthly = (
        df.groupby(group_col)["revenue"]
        .sum()
        .reset_index()
        .sort_values(group_col)
    )
    x = np.arange(len(monthly))
    y = monthly["revenue"].values

    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    return {
        "slope":     round(float(slope),   2),
        "r_squared": round(float(r_value ** 2), 4),
        "p_value":   round(float(p_value), 6),
        "direction": "Upward" if slope > 0 else "Downward",
        "strength":  "Strong" if r_value ** 2 > 0.7 else
                     "Moderate" if r_value ** 2 > 0.4 else "Weak",
    }
