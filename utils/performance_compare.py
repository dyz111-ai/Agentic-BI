from __future__ import annotations

import sys
import argparse
import time
from pathlib import Path
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

BASE_DIR = Path(__file__).resolve().parents[1]


def _ensure_project_root() -> None:
    """Ensure project root is in sys.path for direct script execution."""
    root = str(BASE_DIR)
    if root not in sys.path:
        sys.path.insert(0, root)


def _get_engine(db_url: str) -> Engine:
    """Create SQLAlchemy engine."""
    return create_engine(db_url, pool_pre_ping=True, future=True)


def _get_dialect(engine: Engine) -> str:
    """Get database dialect name."""
    return engine.dialect.name


def _build_year_month_expr(dialect: str, col: str) -> str:
    """Build year-month extraction expression."""
    if dialect == "sqlite":
        return f"strftime('%Y-%m', {col})"
    return f"DATE_FORMAT({col}, '%Y-%m')"


def _run_query(engine: Engine, sql: str) -> tuple[pd.DataFrame, float]:
    """Execute query and return DataFrame with elapsed time."""
    start = time.perf_counter()
    with engine.connect() as conn:
        df = pd.read_sql(sql.strip(), conn)
    elapsed = time.perf_counter() - start
    return df, elapsed


def _build_raw_state_sales_query(dialect: str) -> str:
    """各州 GMV：orders + customers + payments 实时 JOIN。"""
    ym = _build_year_month_expr(dialect, "o.order_purchase_timestamp")
    return f"""
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


def _build_raw_monthly_sales_query(dialect: str) -> str:
    """月度 GMV：orders + payments 实时 JOIN。"""
    ym = _build_year_month_expr(dialect, "o.order_purchase_timestamp")
    return f"""
        SELECT
            {ym} AS year_month,
            ROUND(SUM(p.payment_value), 2) AS total_gmv,
            COUNT(DISTINCT o.order_id) AS total_orders
        FROM orders o
        JOIN payments p ON o.order_id = p.order_id
        WHERE o.order_purchase_timestamp IS NOT NULL
        GROUP BY {ym}
        ORDER BY year_month
    """


def _build_raw_payment_dist_query(dialect: str) -> str:
    """支付方式分布：payments + orders 实时 JOIN。"""
    ym = _build_year_month_expr(dialect, "o.order_purchase_timestamp")
    return f"""
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
        ORDER BY total_value DESC
        LIMIT 20
    """


PREAGG_QUERIES = {
    "mv_monthly_sales": """
        SELECT year_month, total_gmv, total_orders, avg_basket, total_freight
        FROM mv_monthly_sales
        ORDER BY year_month
    """,
    "mv_state_sales": """
        SELECT year_month, customer_state, total_gmv, total_orders, unique_customers
        FROM mv_state_sales
        ORDER BY total_gmv DESC
        LIMIT 20
    """,
    "mv_category_sales": """
        SELECT year_month, product_category_english, total_gmv, total_orders, avg_price
        FROM mv_category_sales
        ORDER BY total_gmv DESC
        LIMIT 20
    """,
    "mv_delivery_perf": """
        SELECT year_month, customer_state, avg_delivery_days, on_time_rate, delayed_orders, total_orders
        FROM mv_delivery_perf
        ORDER BY avg_delivery_days DESC
        LIMIT 20
    """,
    "mv_payment_dist": """
        SELECT year_month, payment_type, total_transactions, avg_installments, total_value
        FROM mv_payment_dist
        ORDER BY total_value DESC
        LIMIT 20
    """,
}


# 三组 Raw vs Pre-agg 对比场景（报告截图建议截这一块）
COMPARISON_SCENARIOS = [
    {
        "id": "state_sales",
        "title": "各州 GMV 排名",
        "raw_builder": _build_raw_state_sales_query,
        "preagg_table": "mv_state_sales",
        "preagg_query": PREAGG_QUERIES["mv_state_sales"],
    },
    {
        "id": "monthly_sales",
        "title": "月度 GMV 趋势",
        "raw_builder": _build_raw_monthly_sales_query,
        "preagg_table": "mv_monthly_sales",
        "preagg_query": PREAGG_QUERIES["mv_monthly_sales"],
    },
    {
        "id": "payment_dist",
        "title": "支付方式分布",
        "raw_builder": _build_raw_payment_dist_query,
        "preagg_table": "mv_payment_dist",
        "preagg_query": PREAGG_QUERIES["mv_payment_dist"],
    },
]


def _build_raw_query(dialect: str) -> str:
    """Backward-compatible alias for the state-sales raw query."""
    return _build_raw_state_sales_query(dialect)


def _compare_one_scenario(
    engine: Engine,
    scenario: dict,
    iterations: int,
    verbose: bool,
) -> dict:
    """Run one raw vs pre-agg pair and return timing stats."""
    dialect = _get_dialect(engine)
    raw_query = scenario["raw_builder"](dialect).strip()
    preagg_query = scenario["preagg_query"].strip()

    raw_times: list[float] = []
    preagg_times: list[float] = []

    if verbose:
        print(f"\n【对比组】{scenario['title']}  (Raw JOIN  vs  {scenario['preagg_table']})")
        print("-" * 60)

    for i in range(iterations):
        _, raw_time = _run_query(engine, raw_query)
        _, preagg_time = _run_query(engine, preagg_query)
        raw_times.append(raw_time)
        preagg_times.append(preagg_time)
        if verbose:
            print(f"  Iteration {i + 1}:  Raw {raw_time * 1000:8.2f} ms  |  Pre-agg {preagg_time * 1000:8.2f} ms")

    avg_raw = sum(raw_times) / len(raw_times)
    avg_preagg = sum(preagg_times) / len(preagg_times)
    speedup = avg_raw / avg_preagg if avg_preagg > 0 else 0.0

    if verbose:
        print(f"  → 平均: Raw {avg_raw * 1000:.2f} ms  |  Pre-agg {avg_preagg * 1000:.2f} ms  |  加速 {speedup:.2f}x")

    return {
        "id": scenario["id"],
        "title": scenario["title"],
        "preagg_table": scenario["preagg_table"],
        "iterations": iterations,
        "raw_query_avg_ms": avg_raw * 1000,
        "preagg_query_avg_ms": avg_preagg * 1000,
        "speedup": speedup,
        "raw_query": raw_query,
        "preagg_query": preagg_query,
    }


def compare_raw_vs_preagg(
    engine: Engine,
    iterations: int = 3,
    verbose: bool = True,
) -> dict:
    """Compare raw JOIN vs pre-aggregation for all configured scenarios."""
    dialect = _get_dialect(engine)

    if verbose:
        print(f"Database dialect: {dialect}")
        print(f"Iterations per query: {iterations}")
        print(f"Comparison groups: {len(COMPARISON_SCENARIOS)}")
        print("=" * 60)

    scenarios: list[dict] = []
    for scenario in COMPARISON_SCENARIOS:
        scenarios.append(_compare_one_scenario(engine, scenario, iterations, verbose))

    if verbose:
        print("\n" + "=" * 60)
        print("三组对比汇总（报告可截此表）")
        print("=" * 60)
        print(f"{'场景':<16} {'Raw JOIN (ms)':>14} {'Pre-agg (ms)':>14} {'加速倍数':>10}")
        print("-" * 60)
        for row in scenarios:
            print(
                f"{row['title']:<16} "
                f"{row['raw_query_avg_ms']:>14.2f} "
                f"{row['preagg_query_avg_ms']:>14.2f} "
                f"{row['speedup']:>9.2f}x"
            )
        print("=" * 60)

    # 保持旧字段：第一组（各州 GMV）供兼容
    first = scenarios[0] if scenarios else {}
    return {
        "dialect": dialect,
        "iterations": iterations,
        "scenarios": scenarios,
        "raw_query_avg_ms": first.get("raw_query_avg_ms", 0),
        "preagg_query_avg_ms": first.get("preagg_query_avg_ms", 0),
        "speedup": first.get("speedup", 0),
        "raw_query": first.get("raw_query", ""),
        "preagg_query": first.get("preagg_query", ""),
    }


def benchmark_preagg_tables(
    engine: Engine,
    iterations: int = 3,
    verbose: bool = True
) -> dict:
    """Benchmark all pre-aggregation tables.
    
    Args:
        engine: SQLAlchemy engine.
        iterations: Number of iterations per table.
        verbose: Whether to print detailed results.
        
    Returns:
        Dictionary with benchmark results for each table.
    """
    dialect = _get_dialect(engine)
    results = {}
    
    if verbose:
        print(f"\nBenchmarking pre-aggregation tables...")
        print("-" * 60)
    
    for table, query in PREAGG_QUERIES.items():
        times = []
        for _ in range(iterations):
            _, elapsed = _run_query(engine, query)
            times.append(elapsed)
        
        avg_time = sum(times) / len(times)
        results[table] = {
            "avg_ms": avg_time * 1000,
            "min_ms": min(times) * 1000,
            "max_ms": max(times) * 1000,
        }
        
        if verbose:
            print(f"  {table}: {avg_time * 1000:.2f} ms avg "
                  f"(min: {min(times) * 1000:.2f}, max: {max(times) * 1000:.2f})")
    
    return results


def verify_preagg_correctness(
    engine: Engine,
    verbose: bool = True
) -> dict:
    """Verify pre-aggregation table correctness by comparing with raw queries.
    
    Args:
        engine: SQLAlchemy engine.
        verbose: Whether to print detailed results.
        
    Returns:
        Dictionary with verification results.
    """
    dialect = _get_dialect(engine)
    ym = _build_year_month_expr(dialect, "o.order_purchase_timestamp")
    results = {}
    
    if verbose:
        print(f"\nVerifying pre-aggregation correctness...")
        print("-" * 60)
    
    with engine.connect() as conn:
        raw_sql = f"""
            SELECT 
                {ym} AS year_month,
                SUM(p.payment_value) AS total_gmv
            FROM orders o
            JOIN payments p ON o.order_id = p.order_id
            WHERE o.order_purchase_timestamp IS NOT NULL
            GROUP BY year_month
            ORDER BY year_month
        """
        raw_df = pd.read_sql(text(raw_sql), conn)
        
        preagg_sql = "SELECT year_month, total_gmv FROM mv_monthly_sales ORDER BY year_month"
        preagg_df = pd.read_sql(text(preagg_sql), conn)
    
    raw_df["total_gmv"] = raw_df["total_gmv"].round(2)
    preagg_df["total_gmv"] = preagg_df["total_gmv"].round(2)
    
    merged = raw_df.merge(preagg_df, on="year_month", suffixes=("_raw", "_preagg"))
    
    if merged.empty:
        results["status"] = "no_data"
        results["message"] = "No overlapping data to compare"
    else:
        merged["diff"] = abs(merged["total_gmv_raw"] - merged["total_gmv_preagg"])
        max_diff = merged["diff"].max()
        
        results["status"] = "pass" if max_diff < 0.01 else "warning"
        results["max_difference"] = max_diff
        results["rows_compared"] = len(merged)
        
        if verbose:
            if max_diff < 0.01:
                print(f"  ✓ Pre-aggregation matches raw query (max diff: {max_diff:.4f})")
            else:
                print(f"  ⚠ Differences found (max diff: {max_diff:.4f})")
            print(f"  Rows compared: {len(merged)}")
    
    return results


def print_sql_examples(dialect: str) -> None:
    """Print example SQL queries for documentation."""
    print("\n" + "=" * 60)
    print(f"Example SQL for {dialect.upper()}:")
    print("=" * 60)
    
    print("\n1. 各州 GMV — Raw JOIN (ad-hoc):")
    print(_build_raw_state_sales_query(dialect).strip() + ";")

    print("\n2. 各州 GMV — Pre-aggregation:")
    print(PREAGG_QUERIES["mv_state_sales"].strip() + ";")

    print("\n3. 月度 GMV — Raw JOIN:")
    print(_build_raw_monthly_sales_query(dialect).strip() + ";")

    print("\n4. 月度 GMV — Pre-aggregation:")
    print(PREAGG_QUERIES["mv_monthly_sales"].strip() + ";")

    print("\n5. 支付方式 — Raw JOIN:")
    print(_build_raw_payment_dist_query(dialect).strip() + ";")

    print("\n6. 支付方式 — Pre-aggregation:")
    print(PREAGG_QUERIES["mv_payment_dist"].strip() + ";")


def main(db_url: str, iterations: int = 3, verbose: bool = True) -> dict:
    """Main function to run all performance comparisons.
    
    Args:
        db_url: Database URL.
        iterations: Number of iterations per benchmark.
        verbose: Whether to print detailed results.
        
    Returns:
        Dictionary with all results.
    """
    _ensure_project_root()
    
    engine = _get_engine(db_url)
    dialect = _get_dialect(engine)
    
    if verbose:
        print("=" * 60)
        print("Agentic BI - Pre-aggregation Performance Comparison")
        print("=" * 60)
        print(f"Database: {db_url}")
        print(f"Dialect:  {dialect}")
    
    comparison = compare_raw_vs_preagg(engine, iterations, verbose)
    benchmarks = benchmark_preagg_tables(engine, iterations, verbose)
    verification = verify_preagg_correctness(engine, verbose)
    
    if verbose:
        print_sql_examples(dialect)
        
        print("\n" + "=" * 60)
        print("Take screenshots of this output for your report.")
        print("=" * 60)
    
    return {
        "comparison": comparison,
        "benchmarks": benchmarks,
        "verification": verification,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare raw JOIN query vs pre-aggregation query performance."
    )
    parser.add_argument(
        "--db-url",
        default="sqlite:///data/sample_olist.db",
        help="SQLAlchemy DB URL (default: sqlite:///data/sample_olist.db)"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Number of iterations per benchmark (default: 3)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print summary, not detailed results"
    )
    parser.add_argument(
        "--examples",
        action="store_true",
        help="Print SQL examples and exit"
    )
    
    args = parser.parse_args()
    
    if args.examples:
        print("SQLite examples:")
        print_sql_examples("sqlite")
        print("\n" + "-" * 60)
        print("MySQL examples:")
        print_sql_examples("mysql")
    else:
        main(args.db_url, args.iterations, verbose=not args.quiet)

'''
python utils/performance_compare.py --db-url sqlite:///data/olist.db
'''