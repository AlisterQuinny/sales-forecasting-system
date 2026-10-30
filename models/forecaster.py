"""
models/forecaster.py — Sales Forecasting Model
────────────────────────────────────────────────
Implements three complementary forecasting approaches:

  1. LinearTrendForecaster  — fast baseline, good for trend extrapolation
  2. RandomForestForecaster — captures non-linear seasonality patterns
  3. ProphetForecaster      — Facebook Prophet (optional, best accuracy)

All forecasters expose a common interface:
    fit(df)  → self
    predict(horizon_months) → pd.DataFrame with columns:
        ds, yhat, yhat_lower, yhat_upper

Usage:
    from models.forecaster import RandomForestForecaster
    fc = RandomForestForecaster()
    fc.fit(master_df)
    forecast = fc.predict(12)
"""
import os
import joblib
import warnings
import numpy as np
import pandas as pd

from sklearn.ensemble         import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model     import Ridge
from sklearn.preprocessing    import StandardScaler
from sklearn.pipeline         import Pipeline
from sklearn.model_selection  import TimeSeriesSplit, cross_val_score
from sklearn.metrics          import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

MODEL_DIR = os.path.join(os.path.dirname(__file__))


# ── Shared helpers ────────────────────────────────────────────────────────────

def _build_monthly_series(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate master_df to a clean monthly time series."""
    monthly = (
        df.groupby("year_month")["revenue"]
        .sum()
        .reset_index()
        .sort_values("year_month")
        .rename(columns={"year_month": "ds", "revenue": "y"})
    )
    monthly["ds"] = pd.to_datetime(monthly["ds"])
    return monthly.reset_index(drop=True)


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add numeric time-based features for regression models."""
    df = df.copy()
    df["t"]          = np.arange(len(df))           # ordinal time index
    df["month"]      = df["ds"].dt.month
    df["quarter"]    = df["ds"].dt.quarter
    df["sin_month"]  = np.sin(2 * np.pi * df["month"] / 12)
    df["cos_month"]  = np.cos(2 * np.pi * df["month"] / 12)
    df["year"]       = df["ds"].dt.year - df["ds"].dt.year.min()
    return df


def _future_dates(last_date: pd.Timestamp, horizon: int) -> pd.DataFrame:
    """Generate a DataFrame of future monthly timestamps."""
    future_dates = pd.date_range(
        start=last_date + pd.offsets.MonthBegin(1),
        periods=horizon,
        freq="MS",
    )
    return pd.DataFrame({"ds": future_dates})


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1, y_true))) * 100
    return {
        "MAE":  round(mae,  2),
        "RMSE": round(rmse, 2),
        "R2":   round(r2,   4),
        "MAPE": round(mape, 2),
    }


# ── 1. Linear Trend Forecaster ────────────────────────────────────────────────

class LinearTrendForecaster:
    """
    Ridge regression on ordinal time index + Fourier seasonal features.
    Fast and interpretable; suitable as a baseline.
    """

    def __init__(self):
        self.model_    = None
        self.monthly_  = None
        self.metrics_  = {}

    def fit(self, df: pd.DataFrame) -> "LinearTrendForecaster":
        self.monthly_ = _build_monthly_series(df)
        feat = _add_time_features(self.monthly_)
        X = feat[["t", "sin_month", "cos_month", "year"]].values
        y = feat["y"].values

        self.model_ = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge",  Ridge(alpha=1.0)),
        ])
        self.model_.fit(X, y)

        y_pred = self.model_.predict(X)
        self.metrics_ = evaluate_model(y, y_pred)
        return self

    def predict(self, horizon: int = 12) -> pd.DataFrame:
        last_date  = self.monthly_["ds"].iloc[-1]
        future     = _future_dates(last_date, horizon)
        n_hist     = len(self.monthly_)
        future["t"]         = np.arange(n_hist, n_hist + horizon)
        future["month"]     = future["ds"].dt.month
        future["quarter"]   = future["ds"].dt.quarter
        future["sin_month"] = np.sin(2 * np.pi * future["month"] / 12)
        future["cos_month"] = np.cos(2 * np.pi * future["month"] / 12)
        future["year"]      = future["ds"].dt.year - self.monthly_["ds"].dt.year.min()

        X = future[["t", "sin_month", "cos_month", "year"]].values
        yhat = self.model_.predict(X).clip(min=0)

        # Simple prediction interval: ±1 MAE
        margin = self.metrics_.get("MAE", 0) * 1.5
        future["yhat"]       = np.round(yhat, 2)
        future["yhat_lower"] = np.round((yhat - margin).clip(min=0), 2)
        future["yhat_upper"] = np.round(yhat + margin, 2)
        return future[["ds", "yhat", "yhat_lower", "yhat_upper"]]

    def save(self, path: str | None = None) -> str:
        path = path or os.path.join(MODEL_DIR, "linear_forecaster.pkl")
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: str) -> "LinearTrendForecaster":
        return joblib.load(path)


# ── 2. Random Forest Forecaster ───────────────────────────────────────────────

class RandomForestForecaster:
    """
    Random Forest that learns from lagged revenue features + calendar.
    Best for datasets with at least 24 months of history.
    """

    N_LAGS = 6

    def __init__(self, n_estimators: int = 200, random_state: int = 42):
        self.n_estimators_  = n_estimators
        self.random_state_  = random_state
        self.model_         = None
        self.monthly_       = None
        self.feature_cols_  = []
        self.metrics_       = {}

    def _make_lag_features(self, series: pd.Series) -> pd.DataFrame:
        feat = pd.DataFrame({"y": series.values})
        for lag in range(1, self.N_LAGS + 1):
            feat[f"lag_{lag}"] = feat["y"].shift(lag)
        # Rolling stats
        feat["roll_mean_3"] = feat["y"].shift(1).rolling(3).mean()
        feat["roll_std_3"]  = feat["y"].shift(1).rolling(3).std()
        feat["roll_mean_6"] = feat["y"].shift(1).rolling(6).mean()
        return feat

    def fit(self, df: pd.DataFrame) -> "RandomForestForecaster":
        self.monthly_ = _build_monthly_series(df)
        feat  = _add_time_features(self.monthly_)
        lags  = self._make_lag_features(self.monthly_["y"])
        combined = pd.concat([feat.reset_index(drop=True), lags.drop("y", axis=1)], axis=1)
        combined = combined.dropna()

        self.feature_cols_ = (
            ["t", "month", "quarter", "sin_month", "cos_month", "year"]
            + [c for c in combined.columns if c.startswith("lag_") or c.startswith("roll_")]
        )
        X = combined[self.feature_cols_].values
        y = combined["y"].values

        self.model_ = Pipeline([
            ("scaler", StandardScaler()),
            ("rf",     RandomForestRegressor(
                n_estimators=self.n_estimators_,
                random_state=self.random_state_,
                n_jobs=-1,
            )),
        ])

        # Time-series cross-validation
        tscv   = TimeSeriesSplit(n_splits=3)
        cv_mae = -cross_val_score(self.model_, X, y, cv=tscv,
                                  scoring="neg_mean_absolute_error").mean()

        self.model_.fit(X, y)
        y_pred = self.model_.predict(X)
        self.metrics_ = evaluate_model(y, y_pred)
        self.metrics_["CV_MAE"] = round(float(cv_mae), 2)
        return self

    def predict(self, horizon: int = 12) -> pd.DataFrame:
        """Iterative (walk-forward) prediction."""
        history = list(self.monthly_["y"].values)
        results = []
        last_date = self.monthly_["ds"].iloc[-1]

        for step in range(horizon):
            forecast_date = last_date + pd.offsets.MonthBegin(step + 1)
            lags_vals = {f"lag_{i}": history[-(i)] for i in range(1, self.N_LAGS + 1)}
            roll_3    = np.mean(history[-3:])
            roll_std3 = np.std(history[-3:])
            roll_6    = np.mean(history[-6:])

            row = {
                "t":          len(self.monthly_) + step,
                "month":      forecast_date.month,
                "quarter":    forecast_date.quarter,
                "sin_month":  np.sin(2 * np.pi * forecast_date.month / 12),
                "cos_month":  np.cos(2 * np.pi * forecast_date.month / 12),
                "year":       forecast_date.year - self.monthly_["ds"].dt.year.min(),
                "roll_mean_3": roll_3,
                "roll_std_3":  roll_std3,
                "roll_mean_6": roll_6,
                **lags_vals,
            }
            X_row = np.array([[row[c] for c in self.feature_cols_]])
            yhat  = float(self.model_.predict(X_row)[0])
            yhat  = max(0, yhat)

            # Confidence interval via individual tree predictions
            # Works for both RandomForest ("rf") and GradientBoosting ("gb")
            estimator_key = "rf" if "rf" in self.model_.named_steps else "gb"
            base_model = self.model_.named_steps[estimator_key]
            scaler     = self.model_.named_steps["scaler"]
            X_scaled   = scaler.transform(X_row)
            # RF: flat list of trees; GB: 2D array (n_stages, n_outputs)
            raw_estimators = base_model.estimators_
            if hasattr(raw_estimators[0], "__iter__"):
                flat_estimators = [e for stage in raw_estimators for e in stage]
            else:
                flat_estimators = list(raw_estimators)
            tree_preds = np.array([e.predict(X_scaled)[0] for e in flat_estimators])
            lower = max(0, float(np.percentile(tree_preds, 10)))
            upper = float(np.percentile(tree_preds, 90))

            results.append({
                "ds": forecast_date,
                "yhat": round(yhat, 2),
                "yhat_lower": round(lower, 2),
                "yhat_upper": round(upper, 2),
            })
            history.append(yhat)

        return pd.DataFrame(results)

    def save(self, path: str | None = None) -> str:
        path = path or os.path.join(MODEL_DIR, "rf_forecaster.pkl")
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: str) -> "RandomForestForecaster":
        return joblib.load(path)


# ── 3. Gradient Boosting Forecaster ──────────────────────────────────────────

class GBForecaster(RandomForestForecaster):
    """Gradient Boosting variant — often more accurate on smaller datasets."""

    def __init__(self, n_estimators: int = 200, random_state: int = 42):
        super().__init__(n_estimators, random_state)

    def fit(self, df: pd.DataFrame) -> "GBForecaster":
        # Override only the estimator; re-use lag feature logic from parent
        self.monthly_ = _build_monthly_series(df)
        feat  = _add_time_features(self.monthly_)
        lags  = self._make_lag_features(self.monthly_["y"])
        combined = pd.concat([feat.reset_index(drop=True), lags.drop("y", axis=1)], axis=1)
        combined = combined.dropna()

        self.feature_cols_ = (
            ["t", "month", "quarter", "sin_month", "cos_month", "year"]
            + [c for c in combined.columns if c.startswith("lag_") or c.startswith("roll_")]
        )
        X = combined[self.feature_cols_].values
        y = combined["y"].values

        self.model_ = Pipeline([
            ("scaler", StandardScaler()),
            ("gb",     GradientBoostingRegressor(
                n_estimators=self.n_estimators_,
                random_state=self.random_state_,
                learning_rate=0.05,
            )),
        ])
        self.model_.fit(X, y)
        y_pred = self.model_.predict(X)
        self.metrics_ = evaluate_model(y, y_pred)
        return self

    def save(self, path: str | None = None) -> str:
        path = path or os.path.join(MODEL_DIR, "gb_forecaster.pkl")
        joblib.dump(self, path)
        return path


# ── 4. Optional Prophet Forecaster ───────────────────────────────────────────

class ProphetForecaster:
    """
    Wrapper around Facebook Prophet.  Requires `prophet` to be installed.
    Automatically adds monthly seasonality and a US holiday calendar.
    """

    def __init__(self):
        self.model_   = None
        self.monthly_ = None
        self.metrics_ = {}

    def fit(self, df: pd.DataFrame) -> "ProphetForecaster":
        try:
            from prophet import Prophet
        except ImportError:
            raise ImportError("Install prophet: pip install prophet")

        self.monthly_ = _build_monthly_series(df)
        prophet_df = self.monthly_.rename(columns={"ds": "ds", "y": "y"})

        self.model_ = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode="multiplicative",
            interval_width=0.80,
        )
        self.model_.fit(prophet_df)

        in_sample = self.model_.predict(prophet_df)
        self.metrics_ = evaluate_model(
            prophet_df["y"].values, in_sample["yhat"].values
        )
        return self

    def predict(self, horizon: int = 12) -> pd.DataFrame:
        last_date = self.monthly_["ds"].iloc[-1]
        future    = pd.date_range(
            start=last_date + pd.offsets.MonthBegin(1),
            periods=horizon, freq="MS"
        )
        future_df = pd.DataFrame({"ds": future})
        forecast  = self.model_.predict(future_df)
        out = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
        out["yhat"]       = out["yhat"].clip(lower=0).round(2)
        out["yhat_lower"] = out["yhat_lower"].clip(lower=0).round(2)
        out["yhat_upper"] = out["yhat_upper"].round(2)
        return out

    def save(self, path: str | None = None) -> str:
        path = path or os.path.join(MODEL_DIR, "prophet_forecaster.pkl")
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: str) -> "ProphetForecaster":
        return joblib.load(path)


# ── Convenience: train all available models ───────────────────────────────────

def train_all(df: pd.DataFrame, save: bool = True) -> dict:
    """
    Train LinearTrend, RandomForest, and GradientBoosting forecasters.
    Returns a dict of {name: forecaster} and optionally saves each model.
    """
    models = {
        "LinearTrend":   LinearTrendForecaster(),
        "RandomForest":  RandomForestForecaster(),
        "GradientBoost": GBForecaster(),
    }
    results = {}
    for name, fc in models.items():
        fc.fit(df)
        if save:
            fc.save()
        results[name] = fc
        print(f"  [{name}] metrics: {fc.metrics_}")
    return results
