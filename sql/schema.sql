-- ============================================================
-- Sales Forecasting & Business Growth Intelligence System
-- Database Schema
-- ============================================================

CREATE DATABASE IF NOT EXISTS sales_db
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

USE sales_db;

-- ------------------------------------------------------------
-- Dimension: Customers
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS customers (
    customer_id   INT          NOT NULL AUTO_INCREMENT,
    customer_name VARCHAR(255) NOT NULL,
    email         VARCHAR(255),
    phone         VARCHAR(50),
    region        VARCHAR(100) NOT NULL,
    country       VARCHAR(100) NOT NULL,
    city          VARCHAR(100),
    segment       ENUM('Consumer', 'Corporate', 'SMB') NOT NULL DEFAULT 'Consumer',
    created_at    DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (customer_id)
) ENGINE=InnoDB;

-- ------------------------------------------------------------
-- Dimension: Categories
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS categories (
    category_id   INT          NOT NULL AUTO_INCREMENT,
    category_name VARCHAR(100) NOT NULL,
    description   TEXT,
    PRIMARY KEY (category_id)
) ENGINE=InnoDB;

-- ------------------------------------------------------------
-- Dimension: Products
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS products (
    product_id    INT            NOT NULL AUTO_INCREMENT,
    product_name  VARCHAR(255)   NOT NULL,
    category_id   INT            NOT NULL,
    unit_cost     DECIMAL(12, 2) NOT NULL DEFAULT 0.00,
    unit_price    DECIMAL(12, 2) NOT NULL DEFAULT 0.00,
    sku           VARCHAR(100),
    is_active     TINYINT(1)     NOT NULL DEFAULT 1,
    created_at    DATETIME       NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (product_id),
    CONSTRAINT fk_product_category FOREIGN KEY (category_id)
        REFERENCES categories (category_id) ON DELETE RESTRICT
) ENGINE=InnoDB;

-- ------------------------------------------------------------
-- Fact: Sales Orders
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS sales (
    sale_id       INT            NOT NULL AUTO_INCREMENT,
    order_date    DATE           NOT NULL,
    customer_id   INT            NOT NULL,
    product_id    INT            NOT NULL,
    quantity      INT            NOT NULL DEFAULT 1,
    unit_price    DECIMAL(12, 2) NOT NULL,
    discount      DECIMAL(5, 4)  NOT NULL DEFAULT 0.0000,
    revenue       DECIMAL(14, 2) GENERATED ALWAYS AS
                      (quantity * unit_price * (1 - discount)) STORED,
    region        VARCHAR(100)   NOT NULL,
    salesperson   VARCHAR(255),
    PRIMARY KEY (sale_id),
    CONSTRAINT fk_sale_customer FOREIGN KEY (customer_id)
        REFERENCES customers (customer_id) ON DELETE RESTRICT,
    CONSTRAINT fk_sale_product  FOREIGN KEY (product_id)
        REFERENCES products (product_id)  ON DELETE RESTRICT,
    INDEX idx_order_date (order_date),
    INDEX idx_region     (region),
    INDEX idx_customer   (customer_id),
    INDEX idx_product    (product_id)
) ENGINE=InnoDB;

-- ------------------------------------------------------------
-- Fact: Sales Forecasts (written back by the ML engine)
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS sales_forecasts (
    forecast_id   INT            NOT NULL AUTO_INCREMENT,
    forecast_date DATE           NOT NULL,
    product_id    INT,
    region        VARCHAR(100),
    predicted_revenue DECIMAL(14, 2) NOT NULL,
    lower_bound   DECIMAL(14, 2),
    upper_bound   DECIMAL(14, 2),
    model_name    VARCHAR(100)   NOT NULL,
    created_at    DATETIME       NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (forecast_id),
    INDEX idx_forecast_date (forecast_date)
) ENGINE=InnoDB;

-- ------------------------------------------------------------
-- Analytics: Monthly KPI Snapshots (materialized by pipeline)
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS monthly_kpis (
    kpi_id        INT            NOT NULL AUTO_INCREMENT,
    year_month    CHAR(7)        NOT NULL COMMENT 'YYYY-MM',
    region        VARCHAR(100),
    category_id   INT,
    total_revenue DECIMAL(16, 2) NOT NULL DEFAULT 0.00,
    total_orders  INT            NOT NULL DEFAULT 0,
    total_units   INT            NOT NULL DEFAULT 0,
    avg_order_value DECIMAL(12, 2),
    gross_profit  DECIMAL(16, 2),
    profit_margin DECIMAL(6, 4),
    mom_growth    DECIMAL(6, 4) COMMENT 'Month-over-Month growth rate',
    yoy_growth    DECIMAL(6, 4) COMMENT 'Year-over-Year growth rate',
    updated_at    DATETIME       NOT NULL DEFAULT CURRENT_TIMESTAMP
                                 ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (kpi_id),
    UNIQUE KEY uq_kpi_month_region_cat (year_month, region, category_id)
) ENGINE=InnoDB;

-- ------------------------------------------------------------
-- Seeded Reference Data
-- ------------------------------------------------------------
INSERT IGNORE INTO categories (category_id, category_name, description) VALUES
    (1, 'Electronics',    'Consumer electronics and accessories'),
    (2, 'Clothing',       'Apparel and fashion items'),
    (3, 'Home & Kitchen', 'Household goods and kitchen appliances'),
    (4, 'Sports',         'Sporting goods and fitness equipment'),
    (5, 'Books',          'Books, e-books, and stationery');
