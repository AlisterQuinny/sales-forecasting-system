"""
dashboard/app.py — Streamlit Interactive Dashboard
────────────────────────────────────────────────────
Sales Forecasting & Business Growth Intelligence System

Sections:
  🏠 Overview         — KPI cards + revenue trend
  📊 Analytics        — category, region, product deep-dive
  🔮 Forecast         — multi-model revenue forecast
  🎯 Recommendations  — growth opps, decline alerts, RFM, insights
  📋 Raw Data         — filterable data explorer

Run:
    streamlit run dashboard/app.py
"""
import os
import sys
import warnings
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from data.pipeline              import run_pipeline
from analytics.engine           import (
    overall_kpis, monthly_revenue_trend, category_performance,
    regional_performance, product_performance, customer_segment_analysis,
    product_growth_signals, seasonal_analysis, sales_velocity, compute_trend_slope,
)
from models.forecaster          import RandomForestForecaster, LinearTrendForecaster
from recommendations.engine     import (
    growth_opportunities, decline_alerts, rfm_segmentation,
    cross_sell_pairs, region_growth_opportunities, strategic_insights,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sales Intelligence System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #f7f8fa;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 16px 20px;
        text-align: center;
    }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #1f2328; }
    .metric-label { font-size: 0.85rem; color: #57606a; margin-top: 4px; }
    .insight-card {
        background: #fff;
        border-left: 4px solid #3b82d4;
        border-radius: 4px;
        padding: 12px 16px;
        margin-bottom: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,.06);
    }
    .insight-card.priority-1 { border-left-color: #ef4444; }
    .insight-card.priority-2 { border-left-color: #f59e0b; }
    .badge-growing  { background:#dcfce7; color:#16a34a; padding:2px 8px; border-radius:12px; font-size:.8rem; }
    .badge-declining{ background:#fee2e2; color:#dc2626; padding:2px 8px; border-radius:12px; font-size:.8rem; }
    .badge-stable   { background:#f1f5f9; color:#475569; padding:2px 8px; border-radius:12px; font-size:.8rem; }
    .badge-new      { background:#dbeafe; color:#1d4ed8; padding:2px 8px; border-radius:12px; font-size:.8rem; }
</style>
""", unsafe_allow_html=True)

# ── Data loading (cached) ─────────────────────────────────────────────────────

@st.cache_data(show_spinner="⚙️  Loading and processing data …")
def load_data() -> pd.DataFrame:
    """Load from Parquet if available, otherwise run the full pipeline."""
    parquet_path = os.path.join(ROOT, "data", "processed", "master_sales.parquet")
    if os.path.exists(parquet_path):
        df = pd.read_parquet(parquet_path)
    else:
        df = run_pipeline(source="csv")
    df["order_date"] = pd.to_datetime(df["order_date"])
    return df


@st.cache_resource(show_spinner="🤖  Training forecast models …")
def load_forecaster(df: pd.DataFrame):
    """Train (or load saved) RandomForest forecaster."""
    model_path = os.path.join(ROOT, "models", "rf_forecaster.pkl")
    if os.path.exists(model_path):
        return RandomForestForecaster.load(model_path)
    fc = RandomForestForecaster()
    fc.fit(df)
    fc.save(model_path)
    return fc


# ── Sidebar controls ──────────────────────────────────────────────────────────

def sidebar(df: pd.DataFrame):
    st.sidebar.image(
        "https://img.icons8.com/color/96/combo-chart--v2.png", width=60
    )
    st.sidebar.title("📊 Sales Intelligence")
    st.sidebar.markdown("---")

    years = sorted(df["year"].unique())
    selected_years = st.sidebar.multiselect(
        "📅 Filter by Year", years, default=years, key="years"
    )
    regions = sorted(df["region"].unique())
    selected_regions = st.sidebar.multiselect(
        "🌍 Filter by Region", regions, default=regions, key="regions"
    )
    categories = sorted(df["category_name"].dropna().unique())
    selected_cats = st.sidebar.multiselect(
        "🏷️ Filter by Category", categories, default=categories, key="cats"
    )

    st.sidebar.markdown("---")
    horizon = st.sidebar.slider(
        "🔮 Forecast Horizon (months)", min_value=3, max_value=24, value=12
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("Sales Forecasting & Business Growth Intelligence System v1.0")

    # Apply filters
    mask = (
        df["year"].isin(selected_years) &
        df["region"].isin(selected_regions) &
        df["category_name"].isin(selected_cats)
    )
    return df[mask].copy(), horizon


# ── KPI Cards ─────────────────────────────────────────────────────────────────

def render_kpi_cards(kpis: dict):
    cols = st.columns(4)
    items = [
        ("💰 Total Revenue",     f"${kpis['total_revenue']:,.0f}"),
        ("📦 Total Orders",      f"{kpis['total_orders']:,}"),
        ("👥 Unique Customers",  f"{kpis['unique_customers']:,}"),
        ("📈 Profit Margin",     f"{kpis['profit_margin']*100:.1f}%"),
    ]
    for col, (label, value) in zip(cols, items):
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    cols2 = st.columns(4)
    items2 = [
        ("🛒 Avg Order Value",  f"${kpis['avg_order_value']:,.2f}"),
        ("📦 Total Units Sold", f"{kpis['total_units']:,}"),
        ("🏷️ Products Active",  f"{kpis['unique_products']}"),
        ("💵 Gross Profit",     f"${kpis['gross_profit']:,.0f}"),
    ]
    for col, (label, value) in zip(cols2, items2):
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
        </div>""", unsafe_allow_html=True)


# ── Overview Page ─────────────────────────────────────────────────────────────

def page_overview(df: pd.DataFrame):
    st.header("🏠 Business Overview")
    kpis = overall_kpis(df)
    render_kpi_cards(kpis)

    st.markdown("---")

    # Revenue trend
    monthly = monthly_revenue_trend(df)
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Monthly Revenue Trend")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=monthly["year_month"], y=monthly["total_revenue"],
            mode="lines+markers", name="Revenue",
            line=dict(color="#3b82d4", width=2),
            fill="tozeroy", fillcolor="rgba(59,130,212,0.08)"
        ))
        fig.update_layout(
            height=320, margin=dict(l=0, r=0, t=30, b=0),
            xaxis_title="Month", yaxis_title="Revenue ($)",
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("MoM Growth Rate")
        fig2 = go.Figure(go.Bar(
            x=monthly["year_month"],
            y=(monthly["mom_growth"] * 100).round(1),
            marker_color=["#16a34a" if v >= 0 else "#dc2626"
                          for v in monthly["mom_growth"].fillna(0)],
        ))
        fig2.update_layout(
            height=320, margin=dict(l=0, r=0, t=30, b=0),
            xaxis_title="Month", yaxis_title="MoM Growth (%)",
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Trend slope summary
    trend = compute_trend_slope(df)
    st.info(
        f"📐 **Trend Direction:** {trend['direction']}  |  "
        f"**Strength:** {trend['strength']}  |  "
        f"**R²:** {trend['r_squared']}  |  "
        f"**Monthly Δ:** ${trend['slope']:,.0f}"
    )

    # Sales velocity
    st.subheader("Sales Velocity (7-day & 30-day Rolling Average)")
    velocity = sales_velocity(df)
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=velocity["order_date"], y=velocity["rolling_7d_revenue"],
        mode="lines", name="7-day Avg",
        line=dict(color="#f59e0b", width=1.5)
    ))
    fig3.add_trace(go.Scatter(
        x=velocity["order_date"], y=velocity["rolling_30d_revenue"],
        mode="lines", name="30-day Avg",
        line=dict(color="#3b82d4", width=2.5)
    ))
    fig3.update_layout(
        height=280, margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title="Date", yaxis_title="Revenue ($)",
        hovermode="x unified", legend=dict(orientation="h"),
    )
    st.plotly_chart(fig3, use_container_width=True)


# ── Analytics Page ────────────────────────────────────────────────────────────

def page_analytics(df: pd.DataFrame):
    st.header("📊 Business Analytics")

    tab1, tab2, tab3, tab4 = st.tabs([
        "🏷️ Categories", "🌍 Regions", "📦 Products", "👥 Customers"
    ])

    # ── Categories ──
    with tab1:
        cat_df = category_performance(df)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Revenue by Category")
            fig = px.bar(
                cat_df, x="category_name", y="total_revenue",
                color="category_name", text_auto=".2s",
                color_discrete_sequence=px.colors.qualitative.Safe,
            )
            fig.update_layout(showlegend=False, height=350,
                              margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.subheader("Profit Margin by Category")
            fig2 = px.bar(
                cat_df, x="category_name",
                y=(cat_df["profit_margin"] * 100).round(1),
                color="category_name", text_auto=".1f",
                color_discrete_sequence=px.colors.qualitative.Pastel,
            )
            fig2.update_layout(showlegend=False, height=350,
                               yaxis_title="Margin (%)",
                               margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Revenue Share")
        fig3 = px.pie(cat_df, names="category_name", values="total_revenue",
                      hole=0.45,
                      color_discrete_sequence=px.colors.qualitative.Set2)
        fig3.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig3, use_container_width=True)

    # ── Regions ──
    with tab2:
        reg_df = regional_performance(df)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Revenue by Region")
            fig = px.bar(
                reg_df, x="region", y="total_revenue",
                color="region", text_auto=".2s",
                color_discrete_sequence=px.colors.qualitative.Bold,
            )
            fig.update_layout(showlegend=False, height=350,
                              margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.subheader("Customers by Region")
            fig2 = px.pie(
                reg_df, names="region", values="unique_customers",
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Bold,
            )
            fig2.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Regional Growth Signals")
        rg = region_growth_opportunities(df)
        styled = rg.copy()
        styled["growth_pct"] = styled["growth_pct"].apply(
            lambda x: f"{x:+.1f}%" if x is not None else "N/A"
        )
        st.dataframe(styled, use_container_width=True, hide_index=True)

    # ── Products ──
    with tab3:
        prod_df = product_performance(df, top_n=15)
        st.subheader("Top 15 Products by Revenue")
        fig = px.bar(
            prod_df, x="total_revenue", y="product_name",
            orientation="h", color="category_name", text_auto=".2s",
            color_discrete_sequence=px.colors.qualitative.Set3,
        )
        fig.update_layout(height=500, margin=dict(l=0, r=0, t=10, b=0),
                          yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Product Growth Signals (last 3 months vs prior 3 months)")
        signals = product_growth_signals(df)
        for _, row in signals.head(20).iterrows():
            trend_cls = row["trend"].lower()
            gr_txt = f"{row['growth_rate']*100:.1f}%" if row["growth_rate"] is not None else "N/A"
            st.markdown(
                f"**{row['product_name']}** ({row['category_name']})  "
                f"— Recent: **${row['rev_recent']:,.0f}** | Prev: **${row['rev_prev']:,.0f}** | "
                f"Growth: <span class='badge-{trend_cls}'>{gr_txt}  {row['trend']}</span>",
                unsafe_allow_html=True,
            )

    # ── Customers ──
    with tab4:
        seg_df = customer_segment_analysis(df)
        st.subheader("Revenue by Customer Segment")
        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(seg_df, x="segment", y="total_revenue",
                         color="segment", text_auto=".2s",
                         color_discrete_sequence=px.colors.qualitative.Prism)
            fig.update_layout(showlegend=False, height=320,
                              margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.subheader("Avg Order Value by Segment")
            fig2 = px.bar(seg_df, x="segment", y="avg_order_value",
                          color="segment", text_auto=".2f",
                          color_discrete_sequence=px.colors.qualitative.Prism)
            fig2.update_layout(showlegend=False, height=320,
                               yaxis_title="Avg Order Value ($)",
                               margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Seasonal Revenue Pattern")
        season_df = seasonal_analysis(df)
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            x=season_df["month_name"], y=season_df["avg_revenue"],
            name="Avg Revenue", marker_color="#3b82d4",
        ))
        fig3.add_trace(go.Scatter(
            x=season_df["month_name"], y=season_df["avg_revenue"],
            mode="lines+markers", name="Trend",
            line=dict(color="#f59e0b", width=2.5),
        ))
        fig3.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0),
                           hovermode="x unified")
        st.plotly_chart(fig3, use_container_width=True)


# ── Forecast Page ─────────────────────────────────────────────────────────────

def page_forecast(df: pd.DataFrame, horizon: int):
    st.header("🔮 Sales Forecast")

    fc = load_forecaster(df)

    col_info, col_metrics = st.columns([2, 1])
    with col_info:
        st.subheader(f"Random Forest — {horizon}-Month Revenue Forecast")
        st.caption(
            "Walk-forward iterative prediction using lag features + "
            "calendar seasonality signals."
        )
    with col_metrics:
        m = fc.metrics_
        st.metric("MAE",  f"${m.get('MAE', 'N/A'):,}")
        st.metric("MAPE", f"{m.get('MAPE', 'N/A')}%")
        st.metric("R²",   f"{m.get('R2',   'N/A')}")

    forecast = fc.predict(horizon)

    # Historical + Forecast chart
    monthly = monthly_revenue_trend(df)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly["year_month"], y=monthly["total_revenue"],
        mode="lines+markers", name="Historical",
        line=dict(color="#3b82d4", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=forecast["ds"].dt.strftime("%Y-%m"),
        y=forecast["yhat"],
        mode="lines+markers", name="Forecast",
        line=dict(color="#f59e0b", width=2, dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x=pd.concat([forecast["ds"], forecast["ds"][::-1]]).dt.strftime("%Y-%m"),
        y=pd.concat([forecast["yhat_upper"], forecast["yhat_lower"][::-1]]),
        fill="toself", fillcolor="rgba(245,158,11,0.12)",
        line=dict(color="rgba(255,255,255,0)"),
        name="80% Confidence Interval",
    ))
    fig.update_layout(
        height=380, margin=dict(l=0, r=0, t=20, b=0),
        xaxis_title="Month", yaxis_title="Revenue ($)",
        hovermode="x unified", legend=dict(orientation="h"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Forecast table
    st.subheader("Forecast Data Table")
    forecast_display = forecast.copy()
    forecast_display["ds"] = forecast_display["ds"].dt.strftime("%Y-%m")
    forecast_display.columns = ["Month", "Forecast ($)", "Lower Bound ($)", "Upper Bound ($)"]
    forecast_display = forecast_display.applymap(
        lambda x: f"${x:,.0f}" if isinstance(x, (int, float)) else x
    )
    st.dataframe(forecast_display, use_container_width=True, hide_index=True)

    # Compare baseline
    st.subheader("Model Comparison — Linear Trend Baseline")
    lin_fc = LinearTrendForecaster()
    lin_fc.fit(df)
    lin_forecast = lin_fc.predict(horizon)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=forecast["ds"].dt.strftime("%Y-%m"),
        y=forecast["yhat"], mode="lines+markers",
        name="Random Forest", line=dict(color="#f59e0b", width=2),
    ))
    fig2.add_trace(go.Scatter(
        x=lin_forecast["ds"].dt.strftime("%Y-%m"),
        y=lin_forecast["yhat"], mode="lines",
        name="Linear Trend", line=dict(color="#7c5cd8", width=2, dash="dash"),
    ))
    fig2.update_layout(
        height=300, margin=dict(l=0, r=0, t=20, b=0),
        xaxis_title="Month", yaxis_title="Revenue ($)",
        hovermode="x unified", legend=dict(orientation="h"),
    )
    st.plotly_chart(fig2, use_container_width=True)

    return forecast


# ── Recommendations Page ──────────────────────────────────────────────────────

def page_recommendations(df: pd.DataFrame, forecast_df: pd.DataFrame | None = None):
    st.header("🎯 Growth Recommendations")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "💡 Strategic Insights",
        "🚀 Growth Opportunities",
        "⚠️ Decline Alerts",
        "👥 Customer RFM",
        "🔗 Cross-Sell",
    ])

    # ── Strategic Insights ──
    with tab1:
        insights = strategic_insights(df, forecast_df)
        st.subheader(f"{len(insights)} Actionable Insights")
        for ins in insights:
            p = ins["priority"]
            cls = f"priority-{p}"
            icon = "🔴" if p == 1 else "🟡"
            st.markdown(f"""
            <div class="insight-card {cls}">
                <strong>{icon} {ins['category']}</strong><br>
                📌 {ins['insight']}<br>
                ✅ <em>{ins['action']}</em>
            </div>""", unsafe_allow_html=True)

    # ── Growth Opportunities ──
    with tab2:
        opps = growth_opportunities(df)
        st.subheader(f"{len(opps)} Growth Opportunities Identified")
        if opps.empty:
            st.info("No strong growth signals in the current filter window.")
        else:
            fig = px.scatter(
                opps, x="growth_pct", y="rev_recent",
                size="opportunity_score", color="category_name",
                hover_name="product_name", text="product_name",
                labels={"growth_pct": "Growth %", "rev_recent": "Recent Revenue ($)"},
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig.update_traces(textposition="top center")
            fig.update_layout(height=400, margin=dict(l=0, r=0, t=20, b=0))
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(opps, use_container_width=True, hide_index=True)

    # ── Decline Alerts ──
    with tab3:
        declines = decline_alerts(df)
        st.subheader(f"{len(declines)} Products Under-Performing")
        if declines.empty:
            st.success("No significant declines detected.")
        else:
            severity_colors = {"Critical": "#ef4444", "High": "#f97316", "Medium": "#f59e0b"}
            fig = px.bar(
                declines.head(15),
                x="decline_pct", y="product_name",
                orientation="h",
                color="severity",
                color_discrete_map=severity_colors,
                labels={"decline_pct": "Decline %", "product_name": "Product"},
            )
            fig.update_layout(height=420, margin=dict(l=0, r=0, t=20, b=0),
                              yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(declines, use_container_width=True, hide_index=True)

    # ── RFM ──
    with tab4:
        rfm = rfm_segmentation(df)
        st.subheader("Customer RFM Segmentation")
        seg_counts = rfm["segment"].value_counts().reset_index()
        seg_counts.columns = ["Segment", "Count"]
        col1, col2 = st.columns(2)
        with col1:
            fig = px.pie(seg_counts, names="Segment", values="Count", hole=0.45,
                         color_discrete_sequence=px.colors.qualitative.Safe)
            fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig2 = px.scatter(
                rfm, x="recency", y="monetary",
                color="segment", size="frequency",
                hover_name="customer_name",
                labels={"recency": "Recency (days)", "monetary": "Lifetime Value ($)"},
                color_discrete_sequence=px.colors.qualitative.Prism,
            )
            fig2.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Top 20 Champions by Lifetime Value")
        champs = rfm[rfm["segment"] == "Champion"].sort_values(
            "monetary", ascending=False
        ).head(20)[[
            "customer_name", "recency", "frequency", "monetary", "rfm_score", "segment"
        ]]
        st.dataframe(champs, use_container_width=True, hide_index=True)

    # ── Cross-Sell ──
    with tab5:
        pairs = cross_sell_pairs(df)
        st.subheader("Frequently Co-Purchased Product Pairs")
        if pairs.empty:
            st.info("Not enough co-purchase data for current filter.")
        else:
            fig = px.bar(
                pairs.head(15),
                x="co_occurrence", y=pairs.head(15).apply(
                    lambda r: f"{r['name_a']} + {r['name_b']}", axis=1
                ),
                orientation="h",
                color="co_occurrence",
                color_continuous_scale="Blues",
                labels={"co_occurrence": "Co-purchase Count", "y": "Product Pair"},
            )
            fig.update_layout(height=450, margin=dict(l=0, r=0, t=10, b=0),
                              yaxis=dict(autorange="reversed"),
                              coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)


# ── Raw Data Page ─────────────────────────────────────────────────────────────

def page_raw_data(df: pd.DataFrame):
    st.header("📋 Data Explorer")
    st.caption(f"Showing {len(df):,} rows matching current filters.")

    search = st.text_input("🔍 Search product or customer name …")
    display_df = df.copy()
    if search:
        mask = (
            display_df["product_name"].str.contains(search, case=False, na=False) |
            display_df["customer_name"].str.contains(search, case=False, na=False)
        )
        display_df = display_df[mask]

    cols_to_show = [
        "order_date", "customer_name", "product_name", "category_name",
        "region", "quantity", "unit_price", "discount", "revenue", "gross_profit",
    ]
    available_cols = [c for c in cols_to_show if c in display_df.columns]
    st.dataframe(
        display_df[available_cols].sort_values("order_date", ascending=False),
        use_container_width=True,
        height=500,
    )

    # Download
    csv = display_df[available_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download as CSV",
        data=csv,
        file_name="sales_data_export.csv",
        mime="text/csv",
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    master_df = load_data()
    filtered_df, horizon = sidebar(master_df)

    if filtered_df.empty:
        st.warning("No data matches the current filters. Please adjust your selections.")
        return

    page = st.sidebar.radio(
        "Navigate", [
            "🏠 Overview",
            "📊 Analytics",
            "🔮 Forecast",
            "🎯 Recommendations",
            "📋 Raw Data",
        ],
        label_visibility="collapsed",
    )

    if page == "🏠 Overview":
        page_overview(filtered_df)

    elif page == "📊 Analytics":
        page_analytics(filtered_df)

    elif page == "🔮 Forecast":
        forecast_df = page_forecast(filtered_df, horizon)
        st.session_state["forecast_df"] = forecast_df

    elif page == "🎯 Recommendations":
        fdf = st.session_state.get("forecast_df", None)
        page_recommendations(filtered_df, fdf)

    elif page == "📋 Raw Data":
        page_raw_data(filtered_df)


if __name__ == "__main__":
    main()
