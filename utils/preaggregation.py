from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from sqlalchemy import text
from sqlalchemy.engine import Engine

from utils.db import (
    get_engine,
    get_dialect,
    execute_statement,
    create_index_sql,
    drop_table_if_exists,
)


PREAGG_TABLES = [
    "mv_monthly_sales",
    "mv_state_sales",
    "mv_category_sales",
    "mv_delivery_perf",
    "mv_seller_perf",
    "mv_payment_dist",
]


BASE_DIR = Path(__file__).resolve().parents[1]


def _ensure_project_root() -> None:
    """Ensure project root is in sys.path for direct script execution."""
    root = str(BASE_DIR)
    if root not in sys.path:
        sys.path.insert(0, root)


def _get_year_month_expr(dialect: str, col: str) -> str:
    """Generate year-month extraction expression.
    
    SQLite: strftime('%Y-%m', col)
    MySQL:  DATE_FORMAT(col, '%Y-%m')
    
    NOTE: 使用 DATE_FORMAT 配合反引号别名
    """
    if dialect == "sqlite":
        return f"strftime('%Y-%m', {col})"
    return f"DATE_FORMAT({col}, '%%Y-%%m')"


def _get_date_diff_expr(dialect: str, end_col: str, start_col: str) -> str:
    """Generate date difference expression in days.
    
    SQLite: julianday(end_col) - julianday(start_col)
    MySQL: DATEDIFF(end_col, start_col)
    """
    if dialect == "sqlite":
        return f"(julianday({end_col}) - julianday({start_col}))"
    return f"DATEDIFF({end_col}, {start_col})"


def _drop_existing_tables(engine: Engine, dialect: str) -> None:
    """Drop all pre-aggregation tables if they exist."""
    for table_name in PREAGG_TABLES:
        drop_table_if_exists(table_name, engine)


def _build_mv_monthly_sales(dialect: str) -> str:
    """Build mv_monthly_sales table creation SQL.
    
    Columns: year_month, total_gmv, total_orders, avg_basket, total_freight
    
    NOTE: GROUP BY 使用完整日期表达式而非列别名，兼容 MySQL ONLY_FULL_GROUP_BY 模式
    """
    ym = _get_year_month_expr(dialect, "o.order_purchase_timestamp")
    
    sql = f"""
CREATE TABLE mv_monthly_sales AS
WITH payment_agg AS (
    SELECT 
        order_id, 
        SUM(payment_value) AS payment_value
    FROM payments 
    GROUP BY order_id
), 
item_agg AS (
    SELECT 
        order_id, 
        SUM(freight_value) AS total_freight
    FROM order_items 
    GROUP BY order_id
)
SELECT
    {ym} AS `year_month`,
    ROUND(SUM(COALESCE(pa.payment_value, 0)), 2) AS total_gmv,
    COUNT(DISTINCT o.order_id) AS total_orders,
    ROUND(SUM(COALESCE(pa.payment_value, 0)) / NULLIF(COUNT(DISTINCT o.order_id), 0), 2) AS avg_basket,
    ROUND(SUM(COALESCE(ia.total_freight, 0)), 2) AS total_freight
FROM orders o
LEFT JOIN payment_agg pa ON o.order_id = pa.order_id
LEFT JOIN item_agg ia ON o.order_id = ia.order_id
WHERE o.order_purchase_timestamp IS NOT NULL
GROUP BY {ym}
ORDER BY `year_month`
    """
    return sql


def _build_mv_state_sales(dialect: str) -> str:
    """Build mv_state_sales table creation SQL.
    
    Columns: year_month, customer_state, total_gmv, total_orders, unique_customers
    
    NOTE: GROUP BY 使用完整日期表达式而非列别名，兼容 MySQL ONLY_FULL_GROUP_BY 模式
    """
    ym = _get_year_month_expr(dialect, "o.order_purchase_timestamp")
    
    sql = f"""
CREATE TABLE mv_state_sales AS
WITH payment_agg AS (
    SELECT 
        order_id, 
        SUM(payment_value) AS payment_value
    FROM payments 
    GROUP BY order_id
)
SELECT
    {ym} AS `year_month`,
    c.customer_state,
    ROUND(SUM(COALESCE(pa.payment_value, 0)), 2) AS total_gmv,
    COUNT(DISTINCT o.order_id) AS total_orders,
    COUNT(DISTINCT c.customer_unique_id) AS unique_customers
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
LEFT JOIN payment_agg pa ON o.order_id = pa.order_id
WHERE o.order_purchase_timestamp IS NOT NULL
GROUP BY {ym}, c.customer_state
ORDER BY `year_month`, total_gmv DESC
    """
    return sql


def _build_mv_category_sales(dialect: str) -> str:
    """Build mv_category_sales table creation SQL.
    
    Columns: year_month, product_category_english, total_gmv, total_orders, avg_price
    
    NOTE: 
    - GROUP BY 使用完整日期表达式而非列别名，兼容 MySQL ONLY_FULL_GROUP_BY 模式
    - GROUP BY 中的 category_expr 直接使用表达式（MySQL 允许在 GROUP BY 中使用复杂表达式）
    """
    ym = _get_year_month_expr(dialect, "o.order_purchase_timestamp")
    category_expr = "COALESCE(t.product_category_name_english, p.product_category_name, 'unknown')"
    
    sql = f"""
CREATE TABLE mv_category_sales AS
SELECT
    {ym} AS `year_month`,
    {category_expr} AS product_category_english,
    ROUND(SUM(oi.price + oi.freight_value), 2) AS total_gmv,
    COUNT(DISTINCT oi.order_id) AS total_orders,
    ROUND(AVG(oi.price), 2) AS avg_price
FROM order_items oi
JOIN orders o ON oi.order_id = o.order_id
LEFT JOIN products p ON oi.product_id = p.product_id
LEFT JOIN product_category_name_translation t ON p.product_category_name = t.product_category_name
WHERE o.order_purchase_timestamp IS NOT NULL
GROUP BY {ym}, {category_expr}
ORDER BY `year_month`, total_gmv DESC
    """
    return sql


def _build_mv_delivery_perf(dialect: str) -> str:
    """Build mv_delivery_perf table creation SQL.
    
    Columns: year_month, customer_state, avg_delivery_days, on_time_rate, 
             delayed_orders, total_orders
    
    NOTE: GROUP BY 使用完整日期表达式而非列别名，兼容 MySQL ONLY_FULL_GROUP_BY 模式
    """
    ym = _get_year_month_expr(dialect, "o.order_purchase_timestamp")
    date_diff = _get_date_diff_expr(
        dialect, 
        "o.order_delivered_customer_date", 
        "o.order_purchase_timestamp"
    )
    
    on_time_case = """
        CASE 
            WHEN o.order_delivered_customer_date IS NOT NULL 
                 AND o.order_delivered_customer_date <= o.order_estimated_delivery_date 
            THEN 1.0 
            ELSE 0.0 
        END
    """
    delayed_case = """
        CASE 
            WHEN o.order_delivered_customer_date > o.order_estimated_delivery_date 
            THEN 1 
            ELSE 0 
        END
    """
    
    sql = f"""
CREATE TABLE mv_delivery_perf AS
SELECT
    {ym} AS `year_month`,
    c.customer_state,
    ROUND(AVG(CASE WHEN o.order_delivered_customer_date IS NOT NULL THEN {date_diff} END), 2) AS avg_delivery_days,
    ROUND(AVG({on_time_case}), 4) AS on_time_rate,
    SUM({delayed_case}) AS delayed_orders,
    COUNT(DISTINCT o.order_id) AS total_orders
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
WHERE o.order_purchase_timestamp IS NOT NULL
GROUP BY {ym}, c.customer_state
ORDER BY `year_month`, avg_delivery_days DESC
    """
    return sql


def _build_mv_seller_perf(dialect: str) -> str:
    """Build mv_seller_perf table creation SQL.
    
    Columns: year_month, seller_id, seller_state, total_gmv, total_orders, avg_review_score
    
    NOTE: GROUP BY 使用完整日期表达式而非列别名，兼容 MySQL ONLY_FULL_GROUP_BY 模式
    """
    ym = _get_year_month_expr(dialect, "o.order_purchase_timestamp")
    seller_state_expr = "s.seller_state"
    
    sql = f"""
CREATE TABLE mv_seller_perf AS
SELECT
    {ym} AS `year_month`,
    s.seller_id,
    {seller_state_expr} AS seller_state,
    ROUND(SUM(oi.price + oi.freight_value), 2) AS total_gmv,
    COUNT(DISTINCT oi.order_id) AS total_orders,
    ROUND(AVG(r.review_score), 2) AS avg_review_score
FROM order_items oi
JOIN orders o ON oi.order_id = o.order_id
LEFT JOIN sellers s ON oi.seller_id = s.seller_id
LEFT JOIN order_reviews r ON oi.order_id = r.order_id
WHERE o.order_purchase_timestamp IS NOT NULL
GROUP BY {ym}, s.seller_id, {seller_state_expr}
ORDER BY `year_month`, avg_review_score ASC
    """
    return sql


def _build_mv_payment_dist(dialect: str) -> str:
    """Build mv_payment_dist table creation SQL.
    
    Columns: year_month, payment_type, total_transactions, avg_installments, total_value
    
    NOTE: GROUP BY 使用完整日期表达式而非列别名，兼容 MySQL ONLY_FULL_GROUP_BY 模式
    """
    ym = _get_year_month_expr(dialect, "o.order_purchase_timestamp")
    
    sql = f"""
CREATE TABLE mv_payment_dist AS
SELECT
    {ym} AS `year_month`,
    p.payment_type,
    COUNT(*) AS total_transactions,
    ROUND(AVG(p.payment_installments), 2) AS avg_installments,
    ROUND(SUM(p.payment_value), 2) AS total_value
FROM payments p
JOIN orders o ON p.order_id = o.order_id
WHERE o.order_purchase_timestamp IS NOT NULL
GROUP BY {ym}, p.payment_type
ORDER BY `year_month`, total_value DESC
    """
    return sql


def _create_indexes(engine: Engine, dialect: str) -> None:
    """Create indexes on pre-aggregation tables."""
    index_defs = [
        ("mv_monthly_sales", "idx_mv_monthly_sales_ym", ["year_month"]),
        ("mv_state_sales", "idx_mv_state_sales_ym_state", ["year_month", "customer_state"]),
        ("mv_category_sales", "idx_mv_category_sales_ym_cat", ["year_month", "product_category_english"]),
        ("mv_delivery_perf", "idx_mv_delivery_perf_ym_state", ["year_month", "customer_state"]),
        ("mv_seller_perf", "idx_mv_seller_perf_ym_seller", ["year_month", "seller_id"]),
        ("mv_payment_dist", "idx_mv_payment_dist_ym_type", ["year_month", "payment_type"]),
    ]
    
    for table_name, index_name, columns in index_defs:
        sql = create_index_sql(table_name, index_name, columns, engine)
        execute_statement(sql, engine)


def refresh_preaggregations(engine: Optional[Engine] = None) -> None:
    """Create or refresh all pre-aggregation tables.
    
    This function creates physical tables (mv_*) that store pre-computed
    aggregations for faster query performance. Tables are dropped and
    recreated to ensure fresh data.
    
    The function is cross-database compatible:
    - SQLite: Uses strftime for date extraction, julianday for date diff
    - MySQL: Uses DATE_FORMAT for date extraction, DATEDIFF for date diff
    - GROUP BY uses full expressions (not column aliases) to satisfy MySQL ONLY_FULL_GROUP_BY mode
    
    Args:
        engine: SQLAlchemy engine. Uses default engine if None.
    """
    _ensure_project_root()
    engine = engine or get_engine()
    dialect = get_dialect(engine)
    
    _drop_existing_tables(engine, dialect)
    
    create_statements = [
        _build_mv_monthly_sales(dialect),
        _build_mv_state_sales(dialect),
        _build_mv_category_sales(dialect),
        _build_mv_delivery_perf(dialect),
        _build_mv_seller_perf(dialect),
        _build_mv_payment_dist(dialect),
    ]
    
    for stmt in create_statements:
        print("\n" + "=" * 80)
        print("DEBUG: Executing SQL:")
        print("=" * 80)
        print(stmt.strip())
        print("=" * 80)
        execute_statement(stmt.strip(), engine)
    
    _create_indexes(engine, dialect)


def get_preagg_table_sql(engine: Optional[Engine] = None) -> dict[str, str]:
    """Get the SQL statements for creating all pre-aggregation tables.
    
    This is useful for documentation or for generating SQL files
    that can be run manually.
    
    Args:
        engine: SQLAlchemy engine to determine dialect.
        
    Returns:
        Dictionary mapping table names to their CREATE TABLE SQL statements.
    """
    _ensure_project_root()
    engine = engine or get_engine()
    dialect = get_dialect(engine)
    
    return {
        "mv_monthly_sales": _build_mv_monthly_sales(dialect),
        "mv_state_sales": _build_mv_state_sales(dialect),
        "mv_category_sales": _build_mv_category_sales(dialect),
        "mv_delivery_perf": _build_mv_delivery_perf(dialect),
        "mv_seller_perf": _build_mv_seller_perf(dialect),
        "mv_payment_dist": _build_mv_payment_dist(dialect),
    }


if __name__ == "__main__":
    from utils.db import get_engine
    engine = get_engine()
    print(f"Using database: {engine.url}")
    refresh_preaggregations(engine)
    print("Pre-aggregation tables refreshed successfully.")
