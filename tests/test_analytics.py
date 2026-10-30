"""
tests/test_analytics.py — Unit tests for the analytics engine.
"""
import sys
import os
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from analytics.engine import (
    overall_kpis, monthly_revenue_trend, category_performance,
    regional_performance, product_performance, product_growth_signals,
    seasonal_analysis, compute_trend_slope,
)


# ── Shared fixture ────────────────────────────────────────────────────────────

@pytest.fixture
def master_df() -> pd.DataFrame:
    """Minimal master DataFrame sufficient to exercise analytics."""
    np.random.seed(0)
    n = 200
    dates = pd.date_range("2022-01-01", periods=n, freq="3D")
    return pd.DataFrame({
        "sale_id":      range(1, n + 1),
        "order_date":   dates,
        "customer_id":  np.random.randint(1, 30, n),
        "product_id":   np.random.randint(1, 10, n),
        "product_name": [f"P{i}" for i in np.random.randint(1, 10, n)],
        "category_name":np.random.choice(["Electronics", "Clothing", "Sports"], n),
        "region":       np.random.choice(["North", "South", "East"], n),
        "segment":      np.random.choice(["Consumer", "Corporate"], n),
        "quantity":     np.random.randint(1, 10, n),
        "revenue":      np.random.uniform(50, 500, n).round(2),
        "gross_profit": np.random.uniform(10, 200, n).round(2),
        "year":         dates.year,
        "month":        dates.month,
        "year_month":   dates.to_period("M").astype(str),
    })


# ── overall_kpis ──────────────────────────────────────────────────────────────

class TestOverallKpis:
    def test_keys_present(self, master_df):
        kpis = overall_kpis(master_df)
        for key in ["total_revenue", "total_orders", "unique_customers",
                    "profit_margin", "gross_profit"]:
            assert key in kpis

    def test_total_orders_matches_len(self, master_df):
        kpis = overall_kpis(master_df)
        assert kpis["total_orders"] == len(master_df)

    def test_revenue_positive(self, master_df):
        kpis = overall_kpis(master_df)
        assert kpis["total_revenue"] > 0


# ── monthly_revenue_trend ─────────────────────────────────────────────────────

class TestMonthlyTrend:
    def test_sorted_ascending(self, master_df):
        trend = monthly_revenue_trend(master_df)
        assert list(trend["year_month"]) == sorted(trend["year_month"].tolist())

    def test_no_missing_revenue(self, master_df):
        trend = monthly_revenue_trend(master_df)
        assert trend["total_revenue"].isna().sum() == 0

    def test_revenue_sum_matches(self, master_df):
        trend  = monthly_revenue_trend(master_df)
        assert abs(trend["total_revenue"].sum() - master_df["revenue"].sum()) < 1e-1


# ── category_performance ──────────────────────────────────────────────────────

class TestCategoryPerformance:
    def test_all_categories_present(self, master_df):
        perf = category_performance(master_df)
        assert set(perf["category_name"]) == set(master_df["category_name"].unique())

    def test_revenue_share_sums_to_one(self, master_df):
        perf = category_performance(master_df)
        assert abs(perf["revenue_share"].sum() - 1.0) < 1e-3


# ── regional_performance ──────────────────────────────────────────────────────

class TestRegionalPerformance:
    def test_all_regions_present(self, master_df):
        reg = regional_performance(master_df)
        assert set(reg["region"]) == set(master_df["region"].unique())

    def test_unique_customers_positive(self, master_df):
        reg = regional_performance(master_df)
        assert (reg["unique_customers"] > 0).all()


# ── product_growth_signals ────────────────────────────────────────────────────

class TestGrowthSignals:
    def test_trend_labels_valid(self, master_df):
        signals = product_growth_signals(master_df)
        valid   = {"Growing", "Declining", "Stable", "New"}
        assert set(signals["trend"]).issubset(valid)

    def test_no_duplicate_products(self, master_df):
        signals = product_growth_signals(master_df)
        assert signals["product_id"].duplicated().sum() == 0


# ── compute_trend_slope ───────────────────────────────────────────────────────

class TestTrendSlope:
    def test_direction_valid(self, master_df):
        result = compute_trend_slope(master_df)
        assert result["direction"] in ("Upward", "Downward")

    def test_r_squared_in_range(self, master_df):
        result = compute_trend_slope(master_df)
        assert 0.0 <= result["r_squared"] <= 1.0
