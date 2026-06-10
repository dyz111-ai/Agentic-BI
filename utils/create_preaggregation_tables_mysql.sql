-- =============================================================================
-- Olist Agentic BI — MySQL 预聚合表创建脚本
-- =============================================================================
-- 用途：基于 9 张原始业务表一次性构建 6 张 mv_* 预聚合表，供 DataAgent 优先查询。
-- 与 utils/preaggregation.py 中 refresh_preaggregations() 逻辑一致。
--
-- 使用前请确保已导入 Olist CSV 到 MySQL（见 utils/load_olist_csvs.py）。
--
-- 执行方式（示例）：
--   mysql -u root -p olist_agentic_bi < utils/create_preaggregation_tables_mysql.sql
--
-- 或在 MySQL 客户端中：
--   USE olist_agentic_bi;
--   SOURCE utils/create_preaggregation_tables_mysql.sql;
-- =============================================================================

SET NAMES utf8mb4;

-- ---------------------------------------------------------------------------
-- 1. 删除旧表（刷新时先 DROP 再 CREATE）
-- ---------------------------------------------------------------------------
DROP TABLE IF EXISTS mv_payment_dist;
DROP TABLE IF EXISTS mv_seller_perf;
DROP TABLE IF EXISTS mv_delivery_perf;
DROP TABLE IF EXISTS mv_category_sales;
DROP TABLE IF EXISTS mv_state_sales;
DROP TABLE IF EXISTS mv_monthly_sales;

-- ---------------------------------------------------------------------------
-- 2. mv_monthly_sales — 粒度：年-月
--    字段：year_month, total_gmv, total_orders, avg_basket, total_freight
-- ---------------------------------------------------------------------------
CREATE TABLE mv_monthly_sales AS
SELECT
    DATE_FORMAT(o.order_purchase_timestamp, '%Y-%m') AS `year_month`,
    ROUND(SUM(COALESCE(pa.payment_value, 0)), 2) AS total_gmv,
    COUNT(DISTINCT o.order_id) AS total_orders,
    ROUND(SUM(COALESCE(pa.payment_value, 0)) / NULLIF(COUNT(DISTINCT o.order_id), 0), 2) AS avg_basket,
    ROUND(SUM(COALESCE(ia.total_freight, 0)), 2) AS total_freight
FROM orders o
LEFT JOIN (
    SELECT order_id, SUM(payment_value) AS payment_value
    FROM payments
    GROUP BY order_id
) pa ON o.order_id = pa.order_id
LEFT JOIN (
    SELECT order_id, SUM(freight_value) AS total_freight
    FROM order_items
    GROUP BY order_id
) ia ON o.order_id = ia.order_id
WHERE o.order_purchase_timestamp IS NOT NULL
GROUP BY DATE_FORMAT(o.order_purchase_timestamp, '%Y-%m')
ORDER BY `year_month`;

-- ---------------------------------------------------------------------------
-- 3. mv_state_sales — 粒度：年-月-州
--    字段：year_month, customer_state, total_gmv, total_orders, unique_customers
-- ---------------------------------------------------------------------------
CREATE TABLE mv_state_sales AS
SELECT
    DATE_FORMAT(o.order_purchase_timestamp, '%Y-%m') AS `year_month`,
    c.customer_state,
    ROUND(SUM(COALESCE(pa.payment_value, 0)), 2) AS total_gmv,
    COUNT(DISTINCT o.order_id) AS total_orders,
    COUNT(DISTINCT c.customer_unique_id) AS unique_customers
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
LEFT JOIN (
    SELECT order_id, SUM(payment_value) AS payment_value
    FROM payments
    GROUP BY order_id
) pa ON o.order_id = pa.order_id
WHERE o.order_purchase_timestamp IS NOT NULL
GROUP BY DATE_FORMAT(o.order_purchase_timestamp, '%Y-%m'), c.customer_state
ORDER BY `year_month`, total_gmv DESC;

-- ---------------------------------------------------------------------------
-- 4. mv_category_sales — 粒度：年-月-品类
--    字段：year_month, product_category_english, total_gmv, total_orders, avg_price
-- ---------------------------------------------------------------------------
CREATE TABLE mv_category_sales AS
SELECT
    DATE_FORMAT(o.order_purchase_timestamp, '%Y-%m') AS `year_month`,
    COALESCE(t.product_category_name_english, p.product_category_name, 'unknown') AS product_category_english,
    ROUND(SUM(oi.price + oi.freight_value), 2) AS total_gmv,
    COUNT(DISTINCT oi.order_id) AS total_orders,
    ROUND(AVG(oi.price), 2) AS avg_price
FROM order_items oi
JOIN orders o ON oi.order_id = o.order_id
LEFT JOIN products p ON oi.product_id = p.product_id
LEFT JOIN product_category_name_translation t ON p.product_category_name = t.product_category_name
WHERE o.order_purchase_timestamp IS NOT NULL
GROUP BY DATE_FORMAT(o.order_purchase_timestamp, '%Y-%m'),
         COALESCE(t.product_category_name_english, p.product_category_name, 'unknown')
ORDER BY `year_month`, total_gmv DESC;

-- ---------------------------------------------------------------------------
-- 5. mv_delivery_perf — 粒度：年-月-州
--    字段：year_month, customer_state, avg_delivery_days, on_time_rate,
--          delayed_orders, total_orders
-- ---------------------------------------------------------------------------
CREATE TABLE mv_delivery_perf AS
SELECT
    DATE_FORMAT(o.order_purchase_timestamp, '%Y-%m') AS `year_month`,
    c.customer_state,
    ROUND(AVG(CASE WHEN o.order_delivered_customer_date IS NOT NULL
              THEN DATEDIFF(o.order_delivered_customer_date, o.order_purchase_timestamp) END), 2) AS avg_delivery_days,
    ROUND(AVG(
        CASE
            WHEN o.order_delivered_customer_date IS NOT NULL
                 AND o.order_delivered_customer_date <= o.order_estimated_delivery_date
            THEN 1.0
            ELSE 0.0
        END
    ), 4) AS on_time_rate,
    SUM(
        CASE
            WHEN o.order_delivered_customer_date > o.order_estimated_delivery_date
            THEN 1
            ELSE 0
        END
    ) AS delayed_orders,
    COUNT(DISTINCT o.order_id) AS total_orders
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
WHERE o.order_purchase_timestamp IS NOT NULL
GROUP BY DATE_FORMAT(o.order_purchase_timestamp, '%Y-%m'), c.customer_state
ORDER BY `year_month`, avg_delivery_days DESC;

-- ---------------------------------------------------------------------------
-- 6. mv_seller_perf — 粒度：年-月-卖家
--    字段：year_month, seller_id, seller_state, total_gmv, total_orders, avg_review_score
-- ---------------------------------------------------------------------------
CREATE TABLE mv_seller_perf AS
SELECT
    DATE_FORMAT(o.order_purchase_timestamp, '%Y-%m') AS `year_month`,
    s.seller_id,
    s.seller_state AS seller_state,
    ROUND(SUM(oi.price + oi.freight_value), 2) AS total_gmv,
    COUNT(DISTINCT oi.order_id) AS total_orders,
    ROUND(AVG(r.review_score), 2) AS avg_review_score
FROM order_items oi
JOIN orders o ON oi.order_id = o.order_id
LEFT JOIN sellers s ON oi.seller_id = s.seller_id
LEFT JOIN order_reviews r ON oi.order_id = r.order_id
WHERE o.order_purchase_timestamp IS NOT NULL
GROUP BY DATE_FORMAT(o.order_purchase_timestamp, '%Y-%m'), s.seller_id, s.seller_state
ORDER BY `year_month`, avg_review_score ASC;

-- ---------------------------------------------------------------------------
-- 7. mv_payment_dist — 粒度：年-月-支付类型
--    字段：year_month, payment_type, total_transactions, avg_installments, total_value
-- ---------------------------------------------------------------------------
CREATE TABLE mv_payment_dist AS
SELECT
    DATE_FORMAT(o.order_purchase_timestamp, '%Y-%m') AS `year_month`,
    p.payment_type,
    COUNT(*) AS total_transactions,
    ROUND(AVG(p.payment_installments), 2) AS avg_installments,
    ROUND(SUM(p.payment_value), 2) AS total_value
FROM payments p
JOIN orders o ON p.order_id = o.order_id
WHERE o.order_purchase_timestamp IS NOT NULL
GROUP BY DATE_FORMAT(o.order_purchase_timestamp, '%Y-%m'), p.payment_type
ORDER BY `year_month`, total_value DESC;

-- ---------------------------------------------------------------------------
-- 8. 索引（加速 Agent 按 year_month / 维度过滤）
-- ---------------------------------------------------------------------------
CREATE INDEX idx_mv_monthly_sales_ym ON mv_monthly_sales (`year_month`);
CREATE INDEX idx_mv_state_sales_ym_state ON mv_state_sales (`year_month`, customer_state);
CREATE INDEX idx_mv_category_sales_ym_cat ON mv_category_sales (`year_month`, product_category_english(64));
CREATE INDEX idx_mv_delivery_perf_ym_state ON mv_delivery_perf (`year_month`, customer_state);
CREATE INDEX idx_mv_seller_perf_ym_seller ON mv_seller_perf (`year_month`, seller_id);
CREATE INDEX idx_mv_payment_dist_ym_type ON mv_payment_dist (`year_month`, payment_type);
