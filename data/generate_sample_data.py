"""
data/generate_sample_data.py
────────────────────────────
Generates realistic synthetic sales data and saves it as CSV files
inside data/raw/.  Run once to bootstrap the project without a live DB.

Usage:
    python data/generate_sample_data.py
"""
import os
import random
import numpy as np
import pandas as pd
from datetime import date, timedelta

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── output paths ──────────────────────────────────────────────────────────────
RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")
os.makedirs(RAW_DIR, exist_ok=True)

# ── configuration ─────────────────────────────────────────────────────────────
N_CUSTOMERS    = 500
N_PRODUCTS     = 80
N_SALES        = 12_000
START_DATE     = date(2021, 1, 1)
END_DATE       = date(2024, 12, 31)
REGIONS        = ["North", "South", "East", "West", "Central"]
SEGMENTS       = ["Consumer", "Corporate", "SMB"]
CATEGORIES     = {
    1: ("Electronics",    40, 800),
    2: ("Clothing",       10, 150),
    3: ("Home & Kitchen", 15, 300),
    4: ("Sports",         20, 250),
    5: ("Books",           5,  60),
}


# ── helpers ───────────────────────────────────────────────────────────────────
def random_date(start: date, end: date) -> date:
    delta = (end - start).days
    return start + timedelta(days=random.randint(0, delta))


# ── customers ─────────────────────────────────────────────────────────────────
customers = pd.DataFrame({
    "customer_id":   range(1, N_CUSTOMERS + 1),
    "customer_name": [f"Customer_{i}" for i in range(1, N_CUSTOMERS + 1)],
    "email":         [f"customer{i}@example.com" for i in range(1, N_CUSTOMERS + 1)],
    "region":        [random.choice(REGIONS)   for _ in range(N_CUSTOMERS)],
    "country":       ["USA"] * N_CUSTOMERS,
    "city":          [f"City_{random.randint(1,50)}" for _ in range(N_CUSTOMERS)],
    "segment":       [random.choice(SEGMENTS)  for _ in range(N_CUSTOMERS)],
    "created_at":    [random_date(START_DATE, END_DATE).isoformat()
                      for _ in range(N_CUSTOMERS)],
})

# ── categories ────────────────────────────────────────────────────────────────
categories = pd.DataFrame([
    {"category_id": k, "category_name": v[0]}
    for k, v in CATEGORIES.items()
])

# ── products ──────────────────────────────────────────────────────────────────
product_rows = []
for pid in range(1, N_PRODUCTS + 1):
    cat_id = random.choice(list(CATEGORIES.keys()))
    _, low_price, high_price = CATEGORIES[cat_id]
    cost  = round(random.uniform(low_price * 0.4, low_price * 0.8), 2)
    price = round(cost * random.uniform(1.3, 2.5), 2)
    product_rows.append({
        "product_id":   pid,
        "product_name": f"Product_{pid}",
        "category_id":  cat_id,
        "unit_cost":    cost,
        "unit_price":   price,
        "sku":          f"SKU-{pid:04d}",
    })
products = pd.DataFrame(product_rows)

# ── sales (with seasonal trend) ───────────────────────────────────────────────
date_range = pd.date_range(START_DATE, END_DATE, freq="D")

sale_rows = []
for sid in range(1, N_SALES + 1):
    # Bias toward recent dates (growth trend)
    weights   = np.linspace(0.5, 1.5, len(date_range))
    weights  /= weights.sum()
    sale_date = np.random.choice(date_range, p=weights)

    # Seasonal uplift: Q4 boost
    sale_date_ts = pd.Timestamp(sale_date)
    month = sale_date_ts.month
    seasonal = 1.3 if month in [11, 12] else (1.1 if month in [6, 7] else 1.0)

    cust    = customers.sample(1).iloc[0]
    product = products.sample(1).iloc[0]
    qty     = max(1, int(np.random.exponential(2) * seasonal))
    discount = round(random.choice([0, 0, 0, 0.05, 0.10, 0.15, 0.20]), 2)

    sale_rows.append({
        "sale_id":    sid,
        "order_date": sale_date_ts.date().isoformat(),
        "customer_id": cust["customer_id"],
        "product_id":  product["product_id"],
        "quantity":    qty,
        "unit_price":  product["unit_price"],
        "discount":    discount,
        "revenue":     round(qty * product["unit_price"] * (1 - discount), 2),
        "region":      cust["region"],
        "salesperson": f"Sales_{random.randint(1, 20)}",
    })

sales = pd.DataFrame(sale_rows)

# ── write CSV files ───────────────────────────────────────────────────────────
customers.to_csv(os.path.join(RAW_DIR, "customers.csv"),  index=False)
categories.to_csv(os.path.join(RAW_DIR, "categories.csv"), index=False)
products.to_csv(os.path.join(RAW_DIR, "products.csv"),    index=False)
sales.to_csv(os.path.join(RAW_DIR, "sales.csv"),          index=False)

print(f"[OK] Generated {len(customers)} customers, {len(products)} products, "
      f"{len(sales)} sales records -> {RAW_DIR}")
