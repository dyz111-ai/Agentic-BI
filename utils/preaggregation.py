from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.engine import Engine
from utils.db import get_engine


def _date_expr(dialect: str, col: str) -> str:
    if dialect == "sqlite":
        return f"strftime('%Y-%m', {col})"
    return f"DATE_FORMAT({col}, '%Y-%m')"


def _datediff_expr(dialect: str, end_col: str, start_col: str) -> str:
    if dialect == "sqlite":
        return f"(julianday({end_col}) - julianday({start_col}))"
    return f"DATEDIFF({end_col}, {start_col})"


def refresh_preaggregations(engine: Engine | None = None) -> None:
    """Create/refresh pre-aggregated physical tables named mv_*.

    MySQL has no native materialized views, so this project uses refreshable
    precomputed tables with mv_ prefix. SQLite demo uses the same approach.
    """
    engine = engine or get_engine()
    dialect = engine.dialect.name
    ym = _date_expr(dialect, "o.order_purchase_timestamp")
    delivery_days = _datediff_expr(dialect, "o.order_delivered_customer_date", "o.order_purchase_timestamp")

    common_drop = [
        "DROP TABLE IF EXISTS mv_monthly_sales",
        "DROP TABLE IF EXISTS mv_state_sales",
        "DROP TABLE IF EXISTS mv_category_sales",
        "DROP TABLE IF EXISTS mv_delivery_perf",
        "DROP TABLE IF EXISTS mv_seller_perf",
        "DROP TABLE IF EXISTS mv_payment_dist",
    ]

    if dialect == "sqlite":
        create_index_prefix = "CREATE INDEX IF NOT EXISTS"
    else:
        create_index_prefix = "CREATE INDEX"

    statements = common_drop + [
        f"""
        CREATE TABLE mv_monthly_sales AS
        WITH payment_agg AS (
            SELECT order_id, SUM(payment_value) AS payment_value
            FROM payments GROUP BY order_id
        ), item_agg AS (
            SELECT order_id, SUM(freight_value) AS total_freight
            FROM order_items GROUP BY order_id
        )
        SELECT
            {ym} AS year_month,
            ROUND(SUM(COALESCE(pa.payment_value, 0)), 2) AS total_gmv,
            COUNT(DISTINCT o.order_id) AS total_orders,
            ROUND(SUM(COALESCE(pa.payment_value, 0)) / NULLIF(COUNT(DISTINCT o.order_id), 0), 2) AS avg_basket,
            ROUND(SUM(COALESCE(ia.total_freight, 0)), 2) AS total_freight
        FROM orders o
        LEFT JOIN payment_agg pa ON o.order_id = pa.order_id
        LEFT JOIN item_agg ia ON o.order_id = ia.order_id
        WHERE o.order_purchase_timestamp IS NOT NULL
        GROUP BY {ym}
        ORDER BY year_month
        """,
        f"""
        CREATE TABLE mv_state_sales AS
        WITH payment_agg AS (
            SELECT order_id, SUM(payment_value) AS payment_value
            FROM payments GROUP BY order_id
        )
        SELECT
            {ym} AS year_month,
            c.customer_state,
            ROUND(SUM(COALESCE(pa.payment_value, 0)), 2) AS total_gmv,
            COUNT(DISTINCT o.order_id) AS total_orders,
            COUNT(DISTINCT c.customer_unique_id) AS unique_customers
        FROM orders o
        JOIN customers c ON o.customer_id = c.customer_id
        LEFT JOIN payment_agg pa ON o.order_id = pa.order_id
        WHERE o.order_purchase_timestamp IS NOT NULL
        GROUP BY {ym}, c.customer_state
        ORDER BY year_month, total_gmv DESC
        """,
        f"""
        CREATE TABLE mv_category_sales AS
        SELECT
            {ym} AS year_month,
            COALESCE(t.product_category_name_english, p.product_category_name, 'unknown') AS product_category_english,
            ROUND(SUM(oi.price + oi.freight_value), 2) AS total_gmv,
            COUNT(DISTINCT oi.order_id) AS total_orders,
            ROUND(AVG(oi.price), 2) AS avg_price
        FROM order_items oi
        JOIN orders o ON oi.order_id = o.order_id
        LEFT JOIN products p ON oi.product_id = p.product_id
        LEFT JOIN product_category_name_translation t ON p.product_category_name = t.product_category_name
        WHERE o.order_purchase_timestamp IS NOT NULL
        GROUP BY {ym}, COALESCE(t.product_category_name_english, p.product_category_name, 'unknown')
        ORDER BY year_month, total_gmv DESC
        """,
        f"""
        CREATE TABLE mv_delivery_perf AS
        SELECT
            {ym} AS year_month,
            c.customer_state,
            ROUND(AVG(CASE WHEN o.order_delivered_customer_date IS NOT NULL THEN {delivery_days} END), 2) AS avg_delivery_days,
            ROUND(AVG(CASE WHEN o.order_delivered_customer_date <= o.order_estimated_delivery_date THEN 1.0 ELSE 0.0 END), 4) AS on_time_rate,
            SUM(CASE WHEN o.order_delivered_customer_date > o.order_estimated_delivery_date THEN 1 ELSE 0 END) AS delayed_orders,
            COUNT(DISTINCT o.order_id) AS total_orders
        FROM orders o
        JOIN customers c ON o.customer_id = c.customer_id
        WHERE o.order_purchase_timestamp IS NOT NULL
        GROUP BY {ym}, c.customer_state
        ORDER BY year_month, avg_delivery_days DESC
        """,
        f"""
        CREATE TABLE mv_seller_perf AS
        SELECT
            {ym} AS year_month,
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
        GROUP BY {ym}, s.seller_id, s.seller_state
        ORDER BY year_month, avg_review_score ASC
        """,
        f"""
        CREATE TABLE mv_payment_dist AS
        SELECT
            {ym} AS year_month,
            p.payment_type,
            COUNT(*) AS total_transactions,
            ROUND(AVG(p.payment_installments), 2) AS avg_installments,
            ROUND(SUM(p.payment_value), 2) AS total_value
        FROM payments p
        JOIN orders o ON p.order_id = o.order_id
        WHERE o.order_purchase_timestamp IS NOT NULL
        GROUP BY {ym}, p.payment_type
        ORDER BY year_month, total_value DESC
        """,
    ]

    # Index creation. SQLite tolerates IF NOT EXISTS; MySQL can fail if rerun, so tables are dropped first.
    index_statements = [
        f"{create_index_prefix} idx_mv_monthly_sales_ym ON mv_monthly_sales(year_month)",
        f"{create_index_prefix} idx_mv_state_sales_ym_state ON mv_state_sales(year_month, customer_state)",
        f"{create_index_prefix} idx_mv_category_sales_ym_cat ON mv_category_sales(year_month, product_category_english)",
        f"{create_index_prefix} idx_mv_delivery_perf_ym_state ON mv_delivery_perf(year_month, customer_state)",
        f"{create_index_prefix} idx_mv_seller_perf_ym_seller ON mv_seller_perf(year_month, seller_id)",
        f"{create_index_prefix} idx_mv_payment_dist_ym_type ON mv_payment_dist(year_month, payment_type)",
    ]

    with engine.begin() as conn:
        for stmt in statements + index_statements:
            conn.execute(text(stmt))


if __name__ == "__main__":
    refresh_preaggregations()
    print("Pre-aggregation tables refreshed.")
