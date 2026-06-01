-- MySQL version: refreshable pre-aggregation physical tables.
-- Run after importing the 9 Olist base tables.

DROP TABLE IF EXISTS mv_monthly_sales;
CREATE TABLE mv_monthly_sales AS
WITH payment_agg AS (
    SELECT order_id, SUM(payment_value) AS payment_value
    FROM payments GROUP BY order_id
), item_agg AS (
    SELECT order_id, SUM(freight_value) AS total_freight
    FROM order_items GROUP BY order_id
)
SELECT
    DATE_FORMAT(o.order_purchase_timestamp, '%Y-%m') AS year_month,
    ROUND(SUM(COALESCE(pa.payment_value, 0)), 2) AS total_gmv,
    COUNT(DISTINCT o.order_id) AS total_orders,
    ROUND(SUM(COALESCE(pa.payment_value, 0)) / NULLIF(COUNT(DISTINCT o.order_id), 0), 2) AS avg_basket,
    ROUND(SUM(COALESCE(ia.total_freight, 0)), 2) AS total_freight
FROM orders o
LEFT JOIN payment_agg pa ON o.order_id = pa.order_id
LEFT JOIN item_agg ia ON o.order_id = ia.order_id
WHERE o.order_purchase_timestamp IS NOT NULL
GROUP BY DATE_FORMAT(o.order_purchase_timestamp, '%Y-%m');
CREATE INDEX idx_mv_monthly_sales_ym ON mv_monthly_sales(year_month);

DROP TABLE IF EXISTS mv_state_sales;
CREATE TABLE mv_state_sales AS
WITH payment_agg AS (
    SELECT order_id, SUM(payment_value) AS payment_value
    FROM payments GROUP BY order_id
)
SELECT
    DATE_FORMAT(o.order_purchase_timestamp, '%Y-%m') AS year_month,
    c.customer_state,
    ROUND(SUM(COALESCE(pa.payment_value, 0)), 2) AS total_gmv,
    COUNT(DISTINCT o.order_id) AS total_orders,
    COUNT(DISTINCT c.customer_unique_id) AS unique_customers
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
LEFT JOIN payment_agg pa ON o.order_id = pa.order_id
WHERE o.order_purchase_timestamp IS NOT NULL
GROUP BY DATE_FORMAT(o.order_purchase_timestamp, '%Y-%m'), c.customer_state;
CREATE INDEX idx_mv_state_sales_ym_state ON mv_state_sales(year_month, customer_state);

DROP TABLE IF EXISTS mv_category_sales;
CREATE TABLE mv_category_sales AS
SELECT
    DATE_FORMAT(o.order_purchase_timestamp, '%Y-%m') AS year_month,
    COALESCE(t.product_category_name_english, p.product_category_name, 'unknown') AS product_category_english,
    ROUND(SUM(oi.price + oi.freight_value), 2) AS total_gmv,
    COUNT(DISTINCT oi.order_id) AS total_orders,
    ROUND(AVG(oi.price), 2) AS avg_price
FROM order_items oi
JOIN orders o ON oi.order_id = o.order_id
LEFT JOIN products p ON oi.product_id = p.product_id
LEFT JOIN product_category_name_translation t ON p.product_category_name = t.product_category_name
WHERE o.order_purchase_timestamp IS NOT NULL
GROUP BY DATE_FORMAT(o.order_purchase_timestamp, '%Y-%m'), COALESCE(t.product_category_name_english, p.product_category_name, 'unknown');
CREATE INDEX idx_mv_category_sales_ym_cat ON mv_category_sales(year_month, product_category_english);

DROP TABLE IF EXISTS mv_delivery_perf;
CREATE TABLE mv_delivery_perf AS
SELECT
    DATE_FORMAT(o.order_purchase_timestamp, '%Y-%m') AS year_month,
    c.customer_state,
    ROUND(AVG(CASE WHEN o.order_delivered_customer_date IS NOT NULL THEN DATEDIFF(o.order_delivered_customer_date, o.order_purchase_timestamp) END), 2) AS avg_delivery_days,
    ROUND(AVG(CASE WHEN o.order_delivered_customer_date <= o.order_estimated_delivery_date THEN 1.0 ELSE 0.0 END), 4) AS on_time_rate,
    SUM(CASE WHEN o.order_delivered_customer_date > o.order_estimated_delivery_date THEN 1 ELSE 0 END) AS delayed_orders,
    COUNT(DISTINCT o.order_id) AS total_orders
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
WHERE o.order_purchase_timestamp IS NOT NULL
GROUP BY DATE_FORMAT(o.order_purchase_timestamp, '%Y-%m'), c.customer_state;
CREATE INDEX idx_mv_delivery_perf_ym_state ON mv_delivery_perf(year_month, customer_state);

DROP TABLE IF EXISTS mv_seller_perf;
CREATE TABLE mv_seller_perf AS
SELECT
    DATE_FORMAT(o.order_purchase_timestamp, '%Y-%m') AS year_month,
    s.seller_id,
    s.seller_state,
    ROUND(SUM(oi.price + oi.freight_value), 2) AS total_gmv,
    COUNT(DISTINCT oi.order_id) AS total_orders,
    ROUND(AVG(r.review_score), 2) AS avg_review_score
FROM order_items oi
JOIN orders o ON oi.order_id = o.order_id
LEFT JOIN sellers s ON oi.seller_id = s.seller_id
LEFT JOIN order_reviews r ON oi.order_id = r.order_id
WHERE o.order_purchase_timestamp IS NOT NULL
GROUP BY DATE_FORMAT(o.order_purchase_timestamp, '%Y-%m'), s.seller_id, s.seller_state;
CREATE INDEX idx_mv_seller_perf_ym_seller ON mv_seller_perf(year_month, seller_id);

DROP TABLE IF EXISTS mv_payment_dist;
CREATE TABLE mv_payment_dist AS
SELECT
    DATE_FORMAT(o.order_purchase_timestamp, '%Y-%m') AS year_month,
    p.payment_type,
    COUNT(*) AS total_transactions,
    ROUND(AVG(p.payment_installments), 2) AS avg_installments,
    ROUND(SUM(p.payment_value), 2) AS total_value
FROM payments p
JOIN orders o ON p.order_id = o.order_id
WHERE o.order_purchase_timestamp IS NOT NULL
GROUP BY DATE_FORMAT(o.order_purchase_timestamp, '%Y-%m'), p.payment_type;
CREATE INDEX idx_mv_payment_dist_ym_type ON mv_payment_dist(year_month, payment_type);
