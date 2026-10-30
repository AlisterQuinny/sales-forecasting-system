"""
tests/test_pipeline.py — Unit tests for the data processing pipeline.
"""
import sys
import os
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.pipeline import clean_sales, clean_products, engineer_features, build_monthly_kpis


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_sales() -> pd.DataFrame:
    return pd.DataFrame({
        "sale_id":     [1, 2, 3, 4, 4],          # row 4 is a duplicate
        "order_date":  pd.to_datetime(["2023-01-10", "2023-02-15", "2023-03-20",
                                       "2023-04-05", "2023-04-05"]),
        "customer_id": [1, 2, 3, 4, 4],
        "product_id":  [10, 20, 30, 40, 40],
        "quantity":    [2, 1, 5, 0, 0],           # quantity=0 should be dropped
        "unit_price":  [100.0, 50.0, 30.0, 20.0, 20.0],
        "discount":    [0.0, 0.1, 0.0, 0.05, 0.05],
        "revenue":     [200.0, 45.0, 150.0, 0.0, 0.0],
        "region":      ["North", "South", "East", "West", "West"],
        "salesperson": ["S1", "S2", "S3", "S4", "S4"],
    })


@pytest.fixture
def sample_products() -> pd.DataFrame:
    return pd.DataFrame({
        "product_id":   [10, 20, 30, 40],
        "product_name": ["Prod A", "Prod B", "Prod C", "Prod D"],
        "category_id":  [1, 2, 1, 3],
        "unit_cost":    [60.0, 25.0, 15.0, -5.0],   # negative cost → should be clipped
        "unit_price":   [100.0, 50.0, 30.0, 20.0],
        "sku":          ["SKU1", "SKU2", "SKU3", "SKU4"],
    })


@pytest.fixture
def sample_customers() -> pd.DataFrame:
    return pd.DataFrame({
        "customer_id":   [1, 2, 3, 4],
        "customer_name": ["Alice", "Bob", "Carol", "Dave"],
        "region":        ["North", "South", "East", "West"],
        "segment":       ["Consumer", "Corporate", "SMB", "Consumer"],
    })


@pytest.fixture
def sample_categories() -> pd.DataFrame:
    return pd.DataFrame({
        "category_id":   [1, 2, 3],
        "category_name": ["Electronics", "Clothing", "Sports"],
    })


# ── Tests: clean_sales ────────────────────────────────────────────────────────

class TestCleanSales:
    def test_removes_duplicates(self, sample_sales):
        cleaned = clean_sales(sample_sales)
        assert cleaned["sale_id"].duplicated().sum() == 0

    def test_removes_zero_quantity(self, sample_sales):
        cleaned = clean_sales(sample_sales)
        assert (cleaned["quantity"] <= 0).sum() == 0

    def test_removes_zero_revenue(self, sample_sales):
        cleaned = clean_sales(sample_sales)
        assert (cleaned["revenue"] <= 0).sum() == 0

    def test_discount_clipped(self, sample_sales):
        df = sample_sales.copy()
        df.loc[0, "discount"] = 1.5   # out-of-range
        cleaned = clean_sales(df)
        assert cleaned["discount"].max() <= 1.0

    def test_output_dtypes(self, sample_sales):
        cleaned = clean_sales(sample_sales)
        assert cleaned["quantity"].dtype == int
        assert pd.api.types.is_datetime64_any_dtype(cleaned["order_date"])


# ── Tests: clean_products ─────────────────────────────────────────────────────

class TestCleanProducts:
    def test_removes_duplicates(self, sample_products):
        df = pd.concat([sample_products, sample_products.iloc[:1]])
        cleaned = clean_products(df)
        assert cleaned["product_id"].duplicated().sum() == 0

    def test_clips_negative_cost(self, sample_products):
        cleaned = clean_products(sample_products)
        assert (cleaned["unit_cost"] < 0).sum() == 0


# ── Tests: engineer_features ──────────────────────────────────────────────────

class TestEngineerFeatures:
    def test_columns_present(self, sample_sales, sample_products,
                             sample_customers, sample_categories):
        sales = clean_sales(sample_sales)
        prods = clean_products(sample_products)
        master = engineer_features(sales, prods, sample_customers, sample_categories)
        for col in ["year", "month", "quarter", "season", "year_month",
                    "gross_profit", "profit_margin"]:
            assert col in master.columns, f"Missing column: {col}"

    def test_gross_profit_non_negative(self, sample_sales, sample_products,
                                       sample_customers, sample_categories):
        sales = clean_sales(sample_sales)
        prods = clean_products(sample_products)
        master = engineer_features(sales, prods, sample_customers, sample_categories)
        # After clipping costs to >= 0, gross profit can be <= revenue
        assert (master["gross_profit"] <= master["revenue"] + 1e-6).all()

    def test_no_nulls_in_core_cols(self, sample_sales, sample_products,
                                   sample_customers, sample_categories):
        sales = clean_sales(sample_sales)
        prods = clean_products(sample_products)
        master = engineer_features(sales, prods, sample_customers, sample_categories)
        for col in ["year", "month", "revenue", "year_month"]:
            assert master[col].isna().sum() == 0, f"Nulls found in {col}"


# ── Tests: build_monthly_kpis ─────────────────────────────────────────────────

class TestBuildMonthlyKpis:
    def test_has_expected_columns(self, sample_sales, sample_products,
                                  sample_customers, sample_categories):
        sales  = clean_sales(sample_sales)
        prods  = clean_products(sample_products)
        master = engineer_features(sales, prods, sample_customers, sample_categories)
        kpis   = build_monthly_kpis(master)
        for col in ["year_month", "total_revenue", "total_orders", "mom_growth"]:
            assert col in kpis.columns

    def test_revenue_sums_match(self, sample_sales, sample_products,
                                sample_customers, sample_categories):
        sales  = clean_sales(sample_sales)
        prods  = clean_products(sample_products)
        master = engineer_features(sales, prods, sample_customers, sample_categories)
        kpis   = build_monthly_kpis(master)
        assert abs(kpis["total_revenue"].sum() - master["revenue"].sum()) < 1e-2
