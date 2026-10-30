"""
tests/test_recommendations.py — Unit tests for the recommendation engine.
"""
import sys
import os
import pytest
import pandas as pd
import numpy as np
from datetime import timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from recommendations.engine import (
    growth_opportunities,
    decline_alerts,
    rfm_segmentation,
    cross_sell_pairs,
    region_growth_opportunities,
    strategic_insights,
)


# ── Fixture ───────────────────────────────────────────────────────────────────

@pytest.fixture
def master_df() -> pd.DataFrame:
    """Synthetic master DF spanning 12 months with 10 products across 3 regions."""
    np.random.seed(7)
    n = 600
    max_date = pd.Timestamp("2024-06-30")
    dates = pd.to_datetime(
        [max_date - timedelta(days=int(x)) for x in np.random.randint(0, 365, n)]
    )
    product_ids = np.random.randint(1, 11, n)
    return pd.DataFrame({
        "sale_id":      range(1, n + 1),
        "order_date":   dates,
        "customer_id":  np.random.randint(1, 51, n),
        "customer_name":[f"Cust_{i}" for i in np.random.randint(1, 51, n)],
        "product_id":   product_ids,
        "product_name": [f"Prod_{p}" for p in product_ids],
        "category_name":np.random.choice(["Electronics", "Clothing", "Sports"], n),
        "region":       np.random.choice(["North", "South", "East"], n),
        "segment":      np.random.choice(["Consumer", "Corporate", "SMB"], n),
        "quantity":     np.random.randint(1, 8, n),
        "revenue":      np.random.uniform(50, 800, n).round(2),
        "gross_profit": np.random.uniform(10, 300, n).round(2),
        "year_month":   dates.to_period("M").astype(str),
    })


# ── growth_opportunities ──────────────────────────────────────────────────────

class TestGrowthOpportunities:
    def test_returns_dataframe(self, master_df):
        result = growth_opportunities(master_df)
        assert isinstance(result, pd.DataFrame)

    def test_all_growth_positive(self, master_df):
        result = growth_opportunities(master_df, min_growth_pct=5.0, min_revenue=100)
        if not result.empty:
            assert (result["growth_pct"] >= 5.0).all()

    def test_score_between_0_and_1(self, master_df):
        result = growth_opportunities(master_df, min_revenue=100)
        if not result.empty:
            assert result["opportunity_score"].between(0, 1).all()


# ── decline_alerts ────────────────────────────────────────────────────────────

class TestDeclineAlerts:
    def test_returns_dataframe(self, master_df):
        result = decline_alerts(master_df)
        assert isinstance(result, pd.DataFrame)

    def test_severity_values_valid(self, master_df):
        result = decline_alerts(master_df)
        if not result.empty:
            assert set(result["severity"]).issubset({"Critical", "High", "Medium"})


# ── rfm_segmentation ──────────────────────────────────────────────────────────

class TestRFMSegmentation:
    def test_one_row_per_customer(self, master_df):
        rfm = rfm_segmentation(master_df)
        assert rfm["customer_id"].duplicated().sum() == 0

    def test_segment_labels_valid(self, master_df):
        rfm = rfm_segmentation(master_df)
        valid = {"Champion", "Loyal Customer", "New Customer",
                 "At Risk", "Lost", "Potential Loyalist"}
        assert set(rfm["segment"]).issubset(valid)

    def test_rfm_scores_range(self, master_df):
        rfm = rfm_segmentation(master_df)
        assert rfm["rfm_score"].between(3, 15).all()


# ── cross_sell_pairs ──────────────────────────────────────────────────────────

class TestCrossSellPairs:
    def test_returns_dataframe(self, master_df):
        result = cross_sell_pairs(master_df, min_support=2)
        assert isinstance(result, pd.DataFrame)

    def test_no_self_pairs(self, master_df):
        result = cross_sell_pairs(master_df, min_support=2)
        if not result.empty:
            assert (result["product_a"] != result["product_b"]).all()

    def test_co_occurrence_above_min(self, master_df):
        min_s  = 2
        result = cross_sell_pairs(master_df, min_support=min_s)
        if not result.empty:
            assert (result["co_occurrence"] >= min_s).all()


# ── region_growth_opportunities ───────────────────────────────────────────────

class TestRegionGrowth:
    def test_all_regions_present(self, master_df):
        result = region_growth_opportunities(master_df)
        assert set(result["region"]) == set(master_df["region"].unique())

    def test_trend_labels_valid(self, master_df):
        result = region_growth_opportunities(master_df)
        assert set(result["trend"]).issubset({"Growing", "Declining", "Stable"})


# ── strategic_insights ────────────────────────────────────────────────────────

class TestStrategicInsights:
    def test_returns_list(self, master_df):
        result = strategic_insights(master_df)
        assert isinstance(result, list)

    def test_each_insight_has_required_keys(self, master_df):
        for ins in strategic_insights(master_df):
            for key in ["priority", "category", "insight", "action"]:
                assert key in ins, f"Missing key: {key}"

    def test_sorted_by_priority(self, master_df):
        result = strategic_insights(master_df)
        priorities = [r["priority"] for r in result]
        assert priorities == sorted(priorities)
