"""
recommendations/engine.py — Recommendation & Growth Opportunity Engine
────────────────────────────────────────────────────────────────────────
Analyses the enriched sales data and forecasts to surface:

  1. Growth Opportunities  — products / regions / segments on an upswing
  2. Decline Alerts        — assets losing momentum that need intervention
  3. Cross-sell Signals    — product pairs frequently bought together
  4. Customer Value Tiers  — RFM-based segmentation for targeting
  5. Inventory / Restock   — products with rising demand trajectories
  6. Strategic Insights    — concise, human-readable action items

All public functions accept a pd.DataFrame (master_sales) and return
DataFrames or lists of dict objects ready for the dashboard.
"""
import pandas as pd
import numpy as np
from datetime import timedelta


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_pct(new: float, old: float) -> float | None:
    if old == 0:
        return None
    return round((new - old) / abs(old) * 100, 2)


def _window_revenue(df: pd.DataFrame, days: int, end_offset: int = 0) -> pd.DataFrame:
    """Sum revenue per product_id for a trailing `days`-day window."""
    cutoff = df["order_date"].max() - timedelta(days=end_offset)
    start  = cutoff - timedelta(days=days)
    mask   = (df["order_date"] > start) & (df["order_date"] <= cutoff)
    return (
        df[mask]
        .groupby("product_id")["revenue"]
        .sum()
        .reset_index()
    )


# ── 1. Growth Opportunities ───────────────────────────────────────────────────

def growth_opportunities(
    df: pd.DataFrame,
    window_days: int = 90,
    min_growth_pct: float = 10.0,
    min_revenue: float = 500.0,
) -> pd.DataFrame:
    """
    Returns products with strong revenue growth in the latest window,
    scored and ranked by growth rate and absolute revenue.

    Columns: product_id, product_name, category_name,
             rev_recent, rev_prev, growth_pct, opportunity_score
    """
    recent = _window_revenue(df, window_days)
    prev   = _window_revenue(df, window_days, end_offset=window_days)
    prev   = prev.rename(columns={"revenue": "rev_prev"})
    recent = recent.rename(columns={"revenue": "rev_recent"})

    merged = recent.merge(prev, on="product_id", how="left").fillna(0)
    merged["growth_pct"] = merged.apply(
        lambda r: _safe_pct(r["rev_recent"], r["rev_prev"]), axis=1
    )

    # Attach names
    meta = df[["product_id", "product_name", "category_name"]].drop_duplicates("product_id")
    merged = merged.merge(meta, on="product_id", how="left")

    # Filter & score
    opps = merged[
        (merged["rev_recent"] >= min_revenue) &
        (merged["growth_pct"].fillna(-999) >= min_growth_pct)
    ].copy()

    # Score = normalised combination of growth % and revenue size
    if len(opps) > 0:
        max_growth = opps["growth_pct"].max() or 1
        max_rev    = opps["rev_recent"].max() or 1
        opps["opportunity_score"] = (
            0.6 * (opps["growth_pct"] / max_growth) +
            0.4 * (opps["rev_recent"] / max_rev)
        ).round(4)
        opps = opps.sort_values("opportunity_score", ascending=False)

    return opps.reset_index(drop=True)[[
        "product_id", "product_name", "category_name",
        "rev_recent", "rev_prev", "growth_pct", "opportunity_score"
    ]]


# ── 2. Decline Alerts ─────────────────────────────────────────────────────────

def decline_alerts(
    df: pd.DataFrame,
    window_days: int = 90,
    decline_threshold_pct: float = -10.0,
    min_prev_revenue: float = 300.0,
) -> pd.DataFrame:
    """
    Returns products with significant revenue decline that require
    management attention (e.g. promotion, repricing, discontinuation).

    Columns: product_id, product_name, category_name,
             rev_recent, rev_prev, decline_pct, severity
    """
    recent = _window_revenue(df, window_days).rename(columns={"revenue": "rev_recent"})
    prev   = _window_revenue(df, window_days, end_offset=window_days).rename(
        columns={"revenue": "rev_prev"}
    )

    merged = prev.merge(recent, on="product_id", how="left").fillna(0)
    merged["decline_pct"] = merged.apply(
        lambda r: _safe_pct(r["rev_recent"], r["rev_prev"]), axis=1
    )

    meta = df[["product_id", "product_name", "category_name"]].drop_duplicates("product_id")
    merged = merged.merge(meta, on="product_id", how="left")

    alerts = merged[
        (merged["rev_prev"] >= min_prev_revenue) &
        (merged["decline_pct"].fillna(0) <= decline_threshold_pct)
    ].copy()

    alerts["severity"] = alerts["decline_pct"].apply(
        lambda p: "Critical" if p is not None and p < -30
             else "High"     if p is not None and p < -20
             else "Medium"
    )
    return alerts.sort_values("decline_pct").reset_index(drop=True)[[
        "product_id", "product_name", "category_name",
        "rev_recent", "rev_prev", "decline_pct", "severity"
    ]]


# ── 3. RFM Customer Segmentation ─────────────────────────────────────────────

def rfm_segmentation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recency-Frequency-Monetary analysis to identify:
        Champions, Loyal Customers, At Risk, Lost, New Customers.

    Returns a DataFrame indexed by customer_id with RFM scores and labels.
    """
    snapshot_date = df["order_date"].max() + timedelta(days=1)

    rfm = (
        df.groupby("customer_id")
        .agg(
            recency=   ("order_date", lambda x: (snapshot_date - x.max()).days),
            frequency= ("sale_id",    "count"),
            monetary=  ("revenue",    "sum"),
        )
        .reset_index()
    )

    # Score each dimension 1–5 (5 = best)
    rfm["r_score"] = pd.qcut(rfm["recency"],   q=5, labels=[5, 4, 3, 2, 1]).astype(int)
    rfm["f_score"] = pd.qcut(rfm["frequency"].rank(method="first"), q=5,
                              labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["m_score"] = pd.qcut(rfm["monetary"].rank(method="first"),  q=5,
                              labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["rfm_score"] = rfm["r_score"] + rfm["f_score"] + rfm["m_score"]

    # Segment labels
    def _segment(row: pd.Series) -> str:
        r, f, m = row["r_score"], row["f_score"], row["m_score"]
        if r >= 4 and f >= 4 and m >= 4:
            return "Champion"
        if r >= 3 and f >= 3:
            return "Loyal Customer"
        if r >= 4 and f <= 2:
            return "New Customer"
        if r <= 2 and f >= 3:
            return "At Risk"
        if r == 1 and f == 1:
            return "Lost"
        return "Potential Loyalist"

    rfm["segment"] = rfm.apply(_segment, axis=1)

    # Merge customer names
    names = df[["customer_id", "customer_name"]].drop_duplicates("customer_id")
    rfm   = rfm.merge(names, on="customer_id", how="left")

    return rfm.sort_values("rfm_score", ascending=False).reset_index(drop=True)


# ── 4. Cross-Sell Product Pairs ───────────────────────────────────────────────

def cross_sell_pairs(
    df: pd.DataFrame,
    min_support: int = 5,
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Identifies product pairs frequently purchased by the same customer,
    ranked by co-occurrence count.  Uses a simple co-occurrence matrix
    (full Apriori is overkill for this dataset scale).
    """
    # Get (customer_id, product_id) pairs
    customer_products = (
        df.groupby("customer_id")["product_id"]
        .apply(list)
        .reset_index()
    )

    from itertools import combinations
    pair_counts: dict[tuple, int] = {}
    for _, row in customer_products.iterrows():
        unique_prods = list(set(row["product_id"]))
        for a, b in combinations(sorted(unique_prods), 2):
            key = (a, b)
            pair_counts[key] = pair_counts.get(key, 0) + 1

    pairs_df = pd.DataFrame([
        {"product_a": k[0], "product_b": k[1], "co_occurrence": v}
        for k, v in pair_counts.items()
        if v >= min_support
    ])

    if pairs_df.empty:
        return pairs_df

    # Attach names
    names = df[["product_id", "product_name"]].drop_duplicates("product_id")
    pairs_df = (
        pairs_df
        .merge(names.rename(columns={"product_id": "product_a",
                                      "product_name": "name_a"}), on="product_a", how="left")
        .merge(names.rename(columns={"product_id": "product_b",
                                      "product_name": "name_b"}), on="product_b", how="left")
    )
    return (
        pairs_df
        .sort_values("co_occurrence", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


# ── 5. Region Growth Opportunities ───────────────────────────────────────────

def region_growth_opportunities(
    df: pd.DataFrame,
    window_days: int = 90,
) -> pd.DataFrame:
    """Rank regions by revenue growth rate for geographic expansion signals."""
    recent_r = (
        df[df["order_date"] > df["order_date"].max() - timedelta(days=window_days)]
        .groupby("region")["revenue"].sum()
        .rename("rev_recent")
    )
    prev_r = (
        df[
            (df["order_date"] > df["order_date"].max() - timedelta(days=window_days * 2)) &
            (df["order_date"] <= df["order_date"].max() - timedelta(days=window_days))
        ]
        .groupby("region")["revenue"].sum()
        .rename("rev_prev")
    )
    merged = pd.concat([recent_r, prev_r], axis=1).fillna(0).reset_index()
    merged["growth_pct"] = merged.apply(
        lambda r: _safe_pct(r["rev_recent"], r["rev_prev"]), axis=1
    )
    merged["trend"] = merged["growth_pct"].apply(
        lambda p: "Growing" if p is not None and p > 5
             else "Declining" if p is not None and p < -5
             else "Stable"
    )
    return merged.sort_values("growth_pct", ascending=False, na_position="last")


# ── 6. Strategic Action Items ─────────────────────────────────────────────────

def strategic_insights(
    df: pd.DataFrame,
    forecast_df: pd.DataFrame | None = None,
) -> list[dict]:
    """
    Returns a list of concise, prioritised action items combining
    the signals from growth, decline, RFM, and (optionally) forecasts.

    Each item: { priority, category, insight, action }
    """
    insights = []

    # — Revenue trend direction
    monthly = df.groupby("year_month")["revenue"].sum().sort_index()
    if len(monthly) >= 3:
        recent_trend = monthly.iloc[-3:].pct_change().mean()
        if recent_trend > 0.03:
            insights.append({
                "priority": 1,
                "category": "Revenue Trend",
                "insight":  f"Revenue growing at {recent_trend*100:.1f}% MoM on average.",
                "action":   "Scale marketing spend and inventory to capitalise on momentum.",
            })
        elif recent_trend < -0.03:
            insights.append({
                "priority": 1,
                "category": "Revenue Trend",
                "insight":  f"Revenue declining at {abs(recent_trend)*100:.1f}% MoM on average.",
                "action":   "Investigate churn drivers and launch retention campaigns immediately.",
            })

    # — Top performing category
    cat_rev = df.groupby("category_name")["revenue"].sum()
    top_cat = cat_rev.idxmax()
    insights.append({
        "priority": 2,
        "category": "Category Performance",
        "insight":  f"'{top_cat}' is the highest-revenue category.",
        "action":   f"Expand product range and cross-sell opportunities in {top_cat}.",
    })

    # — Best growth opportunities
    opps = growth_opportunities(df)
    if not opps.empty:
        top_opp = opps.iloc[0]
        insights.append({
            "priority": 2,
            "category": "Growth Opportunity",
            "insight":  f"{top_opp['product_name']} grew {top_opp['growth_pct']:.1f}% in the last 90 days.",
            "action":   "Increase stock levels, run targeted promotions, and feature in campaigns.",
        })

    # — Decline alerts
    declines = decline_alerts(df)
    if not declines.empty:
        top_dec = declines.iloc[0]
        insights.append({
            "priority": 1,
            "category": "Decline Alert",
            "insight":  f"{top_dec['product_name']} declined {abs(top_dec['decline_pct']):.1f}% in the last 90 days.",
            "action":   "Review pricing, consider bundling, or plan for product retirement.",
        })

    # — Customer segment health
    rfm = rfm_segmentation(df)
    lost_pct = (rfm["segment"] == "Lost").mean() * 100
    if lost_pct > 15:
        insights.append({
            "priority": 2,
            "category": "Customer Retention",
            "insight":  f"{lost_pct:.1f}% of customers are classified as 'Lost'.",
            "action":   "Launch win-back email series with personalised discount offers.",
        })

    # — Forecast outlook
    if forecast_df is not None and not forecast_df.empty:
        last_actual = df.groupby("year_month")["revenue"].sum().iloc[-1]
        next_month  = float(forecast_df.iloc[0]["yhat"])
        fct_pct     = _safe_pct(next_month, last_actual) or 0
        insights.append({
            "priority": 2,
            "category": "Forecast",
            "insight":  f"Next-month revenue forecast: ${next_month:,.0f} "
                        f"({'+' if fct_pct >= 0 else ''}{fct_pct:.1f}% vs last month).",
            "action":   "Align procurement and staffing plans to forecast trajectory.",
        })

    return sorted(insights, key=lambda x: x["priority"])
