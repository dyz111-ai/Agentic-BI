from __future__ import annotations

"""Compare raw JOIN query vs pre-aggregation query.

Usage:
    python utils/performance_compare.py --db-url sqlite:///data/sample_olist.db
    python utils/performance_compare.py --db-url mysql+pymysql://root:pwd@localhost:3306/olist_agentic_bi

Take screenshots of the output for the report.
"""

import argparse
import time
import pandas as pd
from sqlalchemy import create_engine, text


def is_sqlite(engine):
    return engine.dialect.name == "sqlite"


def year_month_expr(engine, col):
    return f"strftime('%Y-%m', {col})" if is_sqlite(engine) else f"DATE_FORMAT({col}, '%Y-%m')"


def run(engine, sql):
    start = time.perf_counter()
    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn)
    return df, time.perf_counter() - start


def main(db_url: str):
    engine = create_engine(db_url, future=True)
    ym = year_month_expr(engine, "o.order_purchase_timestamp")

    raw_sql = f"""
        SELECT
            {ym} AS year_month,
            c.customer_state,
            ROUND(SUM(p.payment_value), 2) AS total_gmv,
            COUNT(DISTINCT o.order_id) AS total_orders
        FROM orders o
        JOIN customers c ON o.customer_id = c.customer_id
        JOIN payments p ON o.order_id = p.order_id
        WHERE o.order_purchase_timestamp IS NOT NULL
        GROUP BY {ym}, c.customer_state
        ORDER BY total_gmv DESC
        LIMIT 20
    """

    mv_sql = """
        SELECT year_month, customer_state, total_gmv, total_orders
        FROM mv_state_sales
        ORDER BY total_gmv DESC
        LIMIT 20
    """

    print("Running raw JOIN query...")
    raw_df, raw_time = run(engine, raw_sql)
    print(raw_df.head())
    print(f"Raw JOIN elapsed: {raw_time:.6f}s")

    print("\nRunning pre-aggregation query...")
    mv_df, mv_time = run(engine, mv_sql)
    print(mv_df.head())
    print(f"Pre-aggregation elapsed: {mv_time:.6f}s")

    if mv_time > 0:
        print(f"\nSpeedup: {raw_time / mv_time:.2f}x")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-url", default="sqlite:///data/sample_olist.db")
    args = parser.parse_args()
    main(args.db_url)
