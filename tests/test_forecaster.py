"""
tests/test_forecaster.py — Unit tests for the forecasting models.
"""
import sys
import os
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.forecaster import (
    LinearTrendForecaster,
    RandomForestForecaster,
    GBForecaster,
    _build_monthly_series,
    evaluate_model,
)


# ── Fixture ───────────────────────────────────────────────────────────────────

@pytest.fixture
def master_df() -> pd.DataFrame:
    """24 months of synthetic monthly sales data."""
    np.random.seed(42)
    n = 500
    dates = pd.date_range("2022-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "sale_id":    range(1, n + 1),
        "order_date": dates,
        "revenue":    np.random.uniform(100, 1000, n).round(2),
        "quantity":   np.random.randint(1, 10, n),
        "year_month": dates.to_period("M").astype(str),
    })


# ── _build_monthly_series ─────────────────────────────────────────────────────

class TestBuildMonthlySeries:
    def test_returns_dataframe(self, master_df):
        monthly = _build_monthly_series(master_df)
        assert isinstance(monthly, pd.DataFrame)

    def test_has_ds_and_y(self, master_df):
        monthly = _build_monthly_series(master_df)
        assert "ds" in monthly.columns and "y" in monthly.columns

    def test_sorted(self, master_df):
        monthly = _build_monthly_series(master_df)
        assert monthly["ds"].is_monotonic_increasing


# ── evaluate_model ────────────────────────────────────────────────────────────

class TestEvaluateModel:
    def test_perfect_prediction(self):
        y = np.array([100.0, 200.0, 300.0])
        metrics = evaluate_model(y, y)
        assert metrics["MAE"]  == 0.0
        assert metrics["RMSE"] == 0.0
        assert metrics["R2"]   == 1.0

    def test_keys_present(self):
        y    = np.array([100.0, 200.0, 300.0])
        yhat = np.array([110.0, 190.0, 310.0])
        m    = evaluate_model(y, yhat)
        for key in ["MAE", "RMSE", "R2", "MAPE"]:
            assert key in m


# ── LinearTrendForecaster ─────────────────────────────────────────────────────

class TestLinearTrendForecaster:
    def test_fit_and_predict(self, master_df):
        fc = LinearTrendForecaster()
        fc.fit(master_df)
        out = fc.predict(6)
        assert len(out) == 6
        assert set(out.columns) == {"ds", "yhat", "yhat_lower", "yhat_upper"}

    def test_forecast_non_negative(self, master_df):
        fc = LinearTrendForecaster()
        fc.fit(master_df)
        out = fc.predict(12)
        assert (out["yhat"] >= 0).all()
        assert (out["yhat_lower"] >= 0).all()

    def test_upper_ge_lower(self, master_df):
        fc = LinearTrendForecaster()
        fc.fit(master_df)
        out = fc.predict(12)
        assert (out["yhat_upper"] >= out["yhat_lower"]).all()

    def test_metrics_populated(self, master_df):
        fc = LinearTrendForecaster()
        fc.fit(master_df)
        assert fc.metrics_["MAE"] >= 0
        assert 0 <= fc.metrics_["R2"] <= 1.01  # allow tiny float tolerance


# ── RandomForestForecaster ────────────────────────────────────────────────────

class TestRandomForestForecaster:
    def test_fit_and_predict(self, master_df):
        fc = RandomForestForecaster(n_estimators=20)
        fc.fit(master_df)
        out = fc.predict(6)
        assert len(out) == 6
        assert "yhat" in out.columns

    def test_forecast_positive(self, master_df):
        fc = RandomForestForecaster(n_estimators=20)
        fc.fit(master_df)
        out = fc.predict(6)
        assert (out["yhat"] >= 0).all()

    def test_confidence_interval_order(self, master_df):
        fc = RandomForestForecaster(n_estimators=20)
        fc.fit(master_df)
        out = fc.predict(6)
        assert (out["yhat_upper"] >= out["yhat_lower"]).all()


# ── GBForecaster ──────────────────────────────────────────────────────────────

class TestGBForecaster:
    def test_fit_and_predict(self, master_df):
        fc = GBForecaster(n_estimators=20)
        fc.fit(master_df)
        out = fc.predict(3)
        assert len(out) == 3

    def test_metrics_present(self, master_df):
        fc = GBForecaster(n_estimators=20)
        fc.fit(master_df)
        assert "MAE" in fc.metrics_
