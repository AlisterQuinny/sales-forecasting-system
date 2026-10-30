"""
data/pipeline.py — Data Processing Pipeline
─────────────────────────────────────────────
Responsibilities:
  1. Load raw CSV files (or DB) into Pandas DataFrames.
  2. Clean and validate data (types, nulls, duplicates, outliers).
  3. Feature-engineer columns used by analytics and ML.
  4. Produce processed Parquet files in data/processed/.
  5. Optionally write back to MySQL via SQLAlchemy.

Usage:
    python data/pipeline.py                     # CSV mode (default)
    python data/pipeline.py --source db         # MySQL mode
    python data/pipeline.py --source db --write # also write processed to DB
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np

# Allow running as a script from any CWD
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.logger import get_logger

logger = get_logger("pipeline")

RAW_DIR  = os.path.join(os.path.dirname(__file__), "raw")
PROC_DIR = os.path.join(os.path.dirname(__file__), "processed")
os.makedirs(PROC_DIR, exist_ok=True)


# ── Load ──────────────────────────────────────────────────────────────────────

def load_csv() -> dict[str, pd.DataFrame]:
    """Load all raw CSV files."""
    logger.info("Loading raw CSV files from %s", RAW_DIR)
    return {
        "sales":      pd.read_csv(os.path.join(RAW_DIR, "sales.csv"),      parse_dates=["order_date"]),
        "products":   pd.read_csv(os.path.join(RAW_DIR, "products.csv")),
        "customers":  pd.read_csv(os.path.join(RAW_DIR, "customers.csv")),
        "categories": pd.read_csv(os.path.join(RAW_DIR, "categories.csv")),
    }


def load_db() -> dict[str, pd.DataFrame]:
    """Load tables directly from MySQL."""
    from utils.db import get_engine
    engine = get_engine()
    logger.info("Loading tables from MySQL database")
    tables = ["sales", "products", "customers", "categories"]
    data = {}
    for t in tables:
        data[t] = pd.read_sql_table(t, engine)
    data["sales"]["order_date"] = pd.to_datetime(data["sales"]["order_date"])
    return data


# ── Clean ─────────────────────────────────────────────────────────────────────

def clean_sales(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Cleaning sales table  [rows=%d]", len(df))
    df = df.drop_duplicates(subset=["sale_id"])
    df = df.dropna(subset=["order_date", "product_id", "customer_id", "revenue"])

    # Remove nonsensical values
    df = df[df["quantity"]  > 0]
    df = df[df["unit_price"] > 0]
    df = df[df["revenue"]    > 0]
    df["discount"] = df["discount"].clip(0, 1)

    # Type coercion
    df["order_date"]  = pd.to_datetime(df["order_date"])
    df["sale_id"]     = df["sale_id"].astype(int)
    df["product_id"]  = df["product_id"].astype(int)
    df["customer_id"] = df["customer_id"].astype(int)
    df["quantity"]    = df["quantity"].astype(int)

    # Remove extreme revenue outliers (> 3 IQR fences)
    q1, q3 = df["revenue"].quantile([0.25, 0.75])
    iqr    = q3 - q1
    df = df[df["revenue"] <= q3 + 3 * iqr]

    logger.info("After cleaning: %d rows", len(df))
    return df.reset_index(drop=True)


def clean_products(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates(subset=["product_id"])
    df["unit_cost"]  = df["unit_cost"].clip(lower=0)
    df["unit_price"] = df["unit_price"].clip(lower=0)
    return df.reset_index(drop=True)


# ── Feature Engineering ───────────────────────────────────────────────────────

def engineer_features(
    sales: pd.DataFrame,
    products: pd.DataFrame,
    customers: pd.DataFrame,
    categories: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return a single enriched 'master' DataFrame used by the analytics
    engine and the forecasting model.
    """
    logger.info("Engineering features …")

    # Join all dimensions onto the fact table
    df = sales.copy()
    df = df.merge(
        products[["product_id", "product_name", "category_id", "unit_cost"]],
        on="product_id", how="left"
    )
    df = df.merge(
        categories[["category_id", "category_name"]],
        on="category_id", how="left"
    )
    df = df.merge(
        customers[["customer_id", "customer_name", "segment"]],
        on="customer_id", how="left"
    )

    # Gross profit
    df["cost"]         = df["quantity"] * df["unit_cost"]
    df["gross_profit"] = df["revenue"] - df["cost"]
    df["profit_margin"] = np.where(
        df["revenue"] > 0, df["gross_profit"] / df["revenue"], 0
    )

    # Date-based features
    df["year"]       = df["order_date"].dt.year
    df["month"]      = df["order_date"].dt.month
    df["quarter"]    = df["order_date"].dt.quarter
    df["week"]       = df["order_date"].dt.isocalendar().week.astype(int)
    df["day_of_week"] = df["order_date"].dt.day_of_week      # 0=Mon
    df["year_month"] = df["order_date"].dt.to_period("M").astype(str)

    # Season label
    df["season"] = df["month"].map(
        lambda m: "Q1" if m <= 3 else "Q2" if m <= 6 else "Q3" if m <= 9 else "Q4"
    )

    # Is weekend
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    logger.info("Feature engineering complete  [cols=%d, rows=%d]",
                len(df.columns), len(df))
    return df


# ── Monthly KPI aggregation ───────────────────────────────────────────────────

def build_monthly_kpis(master: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to monthly level with MoM & YoY growth rates."""
    logger.info("Building monthly KPIs …")
    monthly = (
        master.groupby("year_month")
        .agg(
            total_revenue=("revenue",      "sum"),
            total_orders= ("sale_id",      "count"),
            total_units=  ("quantity",     "sum"),
            total_cost=   ("cost",         "sum"),
        )
        .reset_index()
        .sort_values("year_month")
    )
    monthly["avg_order_value"] = monthly["total_revenue"] / monthly["total_orders"]
    monthly["gross_profit"]    = monthly["total_revenue"] - monthly["total_cost"]
    monthly["profit_margin"]   = monthly["gross_profit"]  / monthly["total_revenue"]

    # MoM growth
    monthly["mom_growth"] = monthly["total_revenue"].pct_change()
    # YoY growth (shift 12 months)
    monthly["yoy_growth"] = monthly["total_revenue"].pct_change(12)

    return monthly


# ── Save processed files ──────────────────────────────────────────────────────

def save_processed(master: pd.DataFrame, kpis: pd.DataFrame) -> None:
    master_path = os.path.join(PROC_DIR, "master_sales.parquet")
    kpis_path   = os.path.join(PROC_DIR, "monthly_kpis.parquet")
    master.to_parquet(master_path, index=False)
    kpis.to_parquet(kpis_path,     index=False)
    logger.info("Saved: %s", master_path)
    logger.info("Saved: %s", kpis_path)


# ── Optional DB writeback ─────────────────────────────────────────────────────

def write_kpis_to_db(kpis: pd.DataFrame) -> None:
    from utils.db import get_engine
    engine = get_engine()
    kpis_db = kpis[[
        "year_month", "total_revenue", "total_orders", "total_units",
        "avg_order_value", "gross_profit", "profit_margin",
        "mom_growth", "yoy_growth"
    ]].copy()
    kpis_db.to_sql("monthly_kpis", engine, if_exists="replace", index=False)
    logger.info("monthly_kpis written to MySQL")


# ── Entry point ───────────────────────────────────────────────────────────────

def run_pipeline(source: str = "csv", write_db: bool = False) -> pd.DataFrame:
    """
    Full pipeline execution.

    Returns
    -------
    master : pd.DataFrame
        Enriched master sales DataFrame (also saved to Parquet).
    """
    raw = load_csv() if source == "csv" else load_db()

    raw["sales"]    = clean_sales(raw["sales"])
    raw["products"] = clean_products(raw["products"])

    master = engineer_features(
        raw["sales"], raw["products"], raw["customers"], raw["categories"]
    )
    kpis = build_monthly_kpis(master)
    save_processed(master, kpis)

    if write_db:
        write_kpis_to_db(kpis)

    logger.info("Pipeline finished successfully.")
    return master


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sales Data Processing Pipeline")
    parser.add_argument("--source", choices=["csv", "db"], default="csv")
    parser.add_argument("--write",  action="store_true",
                        help="Write processed KPIs back to MySQL")
    args = parser.parse_args()
    run_pipeline(source=args.source, write_db=args.write)
