# Sales Forecasting & Business Growth Intelligence System

## Overview
A complete decision-support system that helps businesses forecast sales, understand performance trends, identify growth opportunities, and make data-driven decisions.

---

## Project Structure

```
sales_forecasting/
├── data/
│   ├── raw/                        # Raw CSV inputs
│   ├── processed/                  # Processed Parquet files
│   ├── generate_sample_data.py     # Synthetic data generator
│   └── pipeline.py                 # Data processing pipeline
├── sql/
│   ├── schema.sql                  # MySQL database schema
│   └── queries.sql                 # Analytical SQL queries
├── analytics/
│   └── engine.py                   # Analytics & KPI engine
├── models/
│   └── forecaster.py               # ML forecasting models
├── recommendations/
│   └── engine.py                   # Growth & recommendation engine
├── dashboard/
│   └── app.py                      # Streamlit interactive dashboard
├── utils/
│   ├── db.py                       # Database connection helper
│   └── logger.py                   # Centralised logging
├── tests/
│   ├── test_pipeline.py
│   ├── test_analytics.py
│   ├── test_forecaster.py
│   └── test_recommendations.py
├── requirements.txt
├── .env.example
└── README.md
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure environment (optional, for MySQL)
```bash
cp .env.example .env
# Edit .env with your MySQL credentials
```

### 3. Set up the database (optional)
```sql
-- Run in MySQL Workbench or CLI:
source sql/schema.sql
```

### 4. Generate sample data
```bash
python data/generate_sample_data.py
```
This creates `data/raw/customers.csv`, `products.csv`, `categories.csv`, and `sales.csv`.

### 5. Run the data pipeline
```bash
# CSV mode (no database required)
python data/pipeline.py

# MySQL mode
python data/pipeline.py --source db

# MySQL mode + write KPIs back to DB
python data/pipeline.py --source db --write
```

### 6. Launch the dashboard
```bash
streamlit run dashboard/app.py
```

---

## System Architecture

```
Data Input (CSV / MySQL)
        │
        ▼
Data Processing Pipeline (pipeline.py)
  • Clean & validate
  • Feature engineering
  • Monthly KPI aggregation
        │
        ├──────────────────────────────────┐
        ▼                                  ▼
Analytics Engine (analytics/engine.py)   Forecasting Models (models/forecaster.py)
  • KPI summaries                          • LinearTrendForecaster (Ridge + Fourier)
  • Revenue trends                         • RandomForestForecaster (lag features)
  • Category/region/product analysis       • GBForecaster (Gradient Boosting)
  • Growth signals                         • ProphetForecaster (optional)
  • Seasonal patterns                      • Walk-forward prediction
        │                                  │
        └──────────────┬───────────────────┘
                       ▼
         Recommendation Engine (recommendations/engine.py)
           • Growth opportunities scoring
           • Decline alerts with severity
           • RFM customer segmentation
           • Cross-sell product pairs
           • Strategic action items
                       │
                       ▼
         Streamlit Dashboard (dashboard/app.py)
           🏠 Overview    — KPI cards, revenue trend, velocity
           📊 Analytics   — Category, region, product, seasonality
           🔮 Forecast    — Multi-model forecast with confidence bands
           🎯 Recommendations — Insights, growth, declines, RFM
           📋 Raw Data    — Searchable, downloadable data explorer
```

---

## Modules

### `data/pipeline.py`
- Loads CSV or MySQL data
- Cleans and validates all tables (dedup, null handling, outlier removal)
- Engineers 15+ features: date parts, seasonality, gross profit, margins
- Produces `data/processed/master_sales.parquet` and `monthly_kpis.parquet`

### `analytics/engine.py`
- `overall_kpis()` — high-level business summary
- `monthly_revenue_trend()` — MoM/YoY growth rates
- `category_performance()` — revenue and margin by category
- `regional_performance()` — geographic sales breakdown
- `product_performance()` — top N products
- `customer_segment_analysis()` — Consumer / Corporate / SMB
- `product_growth_signals()` — Growing / Declining / Stable per product
- `seasonal_analysis()` — average revenue by calendar month
- `sales_velocity()` — 7-day and 30-day rolling averages
- `compute_trend_slope()` — linear regression on monthly revenue

### `models/forecaster.py`
| Model | Description | Best For |
|-------|-------------|----------|
| `LinearTrendForecaster` | Ridge regression + Fourier features | Quick baseline |
| `RandomForestForecaster` | Random Forest with lag/rolling features | General use |
| `GBForecaster` | Gradient Boosting variant | Higher accuracy |
| `ProphetForecaster` | Facebook Prophet (optional) | Seasonal decomposition |

All models share a common API: `.fit(df)` → `.predict(horizon)` → DataFrame.

### `recommendations/engine.py`
- `growth_opportunities()` — products with strong momentum, scored 0–1
- `decline_alerts()` — Critical / High / Medium severity alerts
- `rfm_segmentation()` — 6-tier customer value segmentation
- `cross_sell_pairs()` — co-purchase frequency matrix
- `region_growth_opportunities()` — geographic expansion signals
- `strategic_insights()` — prioritised, human-readable action items

### `dashboard/app.py`
Five-page Streamlit app with interactive Plotly charts, sidebar filters
(year, region, category), forecast horizon slider, and CSV export.

---

## Running Tests

```bash
# From the sales_forecasting/ directory
pytest tests/ -v
```

---

## Technologies

| Layer | Technology |
|-------|------------|
| Language | Python 3.11+ |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn, Prophet (optional) |
| Database | MySQL / SQLAlchemy / PyMySQL |
| Dashboard | Streamlit + Plotly |
| Statistical Analysis | SciPy |
| Serialisation | Joblib, Parquet |

---

## Future Scope

- **Real-time data integration** via Kafka or webhook connectors
- **Automated business alerts** with email/Slack notifications
- **Scenario planning** — revenue simulation under different assumptions
- **Advanced customer segmentation** using clustering (K-Means, DBSCAN)
- **REST API** (FastAPI) to expose forecasts to other systems
- **Cloud deployment** on AWS / GCP / Azure with scheduled pipeline runs
