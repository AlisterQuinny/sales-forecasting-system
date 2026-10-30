-- ============================================================
-- Analytical Queries — Sales Forecasting System
-- ============================================================

USE sales_db;

-- 1. Monthly Revenue Trend
SELECT
    DATE_FORMAT(s.order_date, '%Y-%m')   AS year_month,
    SUM(s.revenue)                        AS total_revenue,
    COUNT(DISTINCT s.sale_id)             AS total_orders,
    SUM(s.quantity)                       AS total_units,
    AVG(s.revenue)                        AS avg_order_value
FROM sales s
GROUP BY year_month
ORDER BY year_month;

-- 2. Revenue by Product Category (with Profit Margin)
SELECT
    c.category_name,
    SUM(s.revenue)                                         AS total_revenue,
    SUM(s.quantity * p.unit_cost)                          AS total_cost,
    SUM(s.revenue) - SUM(s.quantity * p.unit_cost)         AS gross_profit,
    ROUND(
        (SUM(s.revenue) - SUM(s.quantity * p.unit_cost))
        / NULLIF(SUM(s.revenue), 0), 4
    )                                                       AS profit_margin
FROM sales s
JOIN products  p ON s.product_id   = p.product_id
JOIN categories c ON p.category_id = c.category_id
GROUP BY c.category_id, c.category_name
ORDER BY total_revenue DESC;

-- 3. Top 10 Products by Revenue
SELECT
    p.product_name,
    c.category_name,
    SUM(s.revenue)        AS total_revenue,
    SUM(s.quantity)       AS total_units_sold,
    COUNT(s.sale_id)      AS order_count
FROM sales s
JOIN products   p ON s.product_id   = p.product_id
JOIN categories c ON p.category_id  = c.category_id
GROUP BY p.product_id, p.product_name, c.category_name
ORDER BY total_revenue DESC
LIMIT 10;

-- 4. Regional Performance
SELECT
    s.region,
    SUM(s.revenue)                AS total_revenue,
    COUNT(DISTINCT s.customer_id) AS unique_customers,
    COUNT(s.sale_id)              AS total_orders,
    AVG(s.revenue)                AS avg_order_value
FROM sales s
GROUP BY s.region
ORDER BY total_revenue DESC;

-- 5. Month-over-Month Growth
WITH monthly AS (
    SELECT
        DATE_FORMAT(order_date, '%Y-%m') AS ym,
        SUM(revenue)                      AS rev
    FROM sales
    GROUP BY ym
)
SELECT
    ym,
    rev,
    LAG(rev) OVER (ORDER BY ym)                            AS prev_month_rev,
    ROUND((rev - LAG(rev) OVER (ORDER BY ym))
          / NULLIF(LAG(rev) OVER (ORDER BY ym), 0), 4)    AS mom_growth
FROM monthly
ORDER BY ym;

-- 6. Year-over-Year Growth
WITH yearly AS (
    SELECT
        YEAR(order_date) AS yr,
        SUM(revenue)      AS rev
    FROM sales
    GROUP BY yr
)
SELECT
    yr,
    rev,
    LAG(rev) OVER (ORDER BY yr)                           AS prev_year_rev,
    ROUND((rev - LAG(rev) OVER (ORDER BY yr))
          / NULLIF(LAG(rev) OVER (ORDER BY yr), 0), 4)   AS yoy_growth
FROM yearly
ORDER BY yr;

-- 7. Customer Purchase Frequency & Value Segments
SELECT
    cust.customer_id,
    cust.customer_name,
    cust.segment,
    cust.region,
    COUNT(s.sale_id)  AS purchase_count,
    SUM(s.revenue)    AS lifetime_value,
    AVG(s.revenue)    AS avg_order_value,
    MAX(s.order_date) AS last_purchase_date
FROM customers cust
LEFT JOIN sales s ON cust.customer_id = s.customer_id
GROUP BY cust.customer_id, cust.customer_name, cust.segment, cust.region
ORDER BY lifetime_value DESC;

-- 8. Declining vs. Growing Products (last 3 months vs previous 3 months)
WITH recent AS (
    SELECT product_id, SUM(revenue) AS rev_recent
    FROM   sales
    WHERE  order_date >= DATE_SUB(CURDATE(), INTERVAL 3 MONTH)
    GROUP BY product_id
),
previous AS (
    SELECT product_id, SUM(revenue) AS rev_prev
    FROM   sales
    WHERE  order_date BETWEEN DATE_SUB(CURDATE(), INTERVAL 6 MONTH)
                          AND DATE_SUB(CURDATE(), INTERVAL 3 MONTH)
    GROUP BY product_id
)
SELECT
    p.product_name,
    c.category_name,
    COALESCE(r.rev_recent, 0)  AS rev_recent_3m,
    COALESCE(pr.rev_prev,  0)  AS rev_prev_3m,
    ROUND(
        (COALESCE(r.rev_recent, 0) - COALESCE(pr.rev_prev, 0))
        / NULLIF(COALESCE(pr.rev_prev, 0), 0), 4
    )                          AS growth_rate,
    CASE
        WHEN COALESCE(pr.rev_prev, 0) = 0 THEN 'New'
        WHEN r.rev_recent > pr.rev_prev   THEN 'Growing'
        WHEN r.rev_recent < pr.rev_prev   THEN 'Declining'
        ELSE 'Stable'
    END                        AS trend_label
FROM products p
JOIN categories c ON p.category_id = c.category_id
LEFT JOIN recent   r  ON p.product_id = r.product_id
LEFT JOIN previous pr ON p.product_id = pr.product_id
ORDER BY growth_rate DESC;
