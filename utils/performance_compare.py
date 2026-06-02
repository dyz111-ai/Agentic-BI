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
        df = pd.read_sql(text(sql), conn)
    elapsed = time.perf_counter() - start
    return df, elapsed


def _build_raw_query(dialect: str) -> str:
    """Build raw JOIN query for performance comparison.
    
    This simulates a typical ad-hoc query that joins multiple tables.
    """
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


def compare_raw_vs_preagg(
    engine: Engine,
    iterations: int = 3,
    verbose: bool = True
) -> dict:
    """Compare raw JOIN query vs pre-aggregation query performance.
    
    Args:
        engine: SQLAlchemy engine.
        iterations: Number of iterations to run for each query.
        verbose: Whether to print detailed results.
        
    Returns:
        Dictionary with comparison results.
    """
    dialect = _get_dialect(engine)
    
    if verbose:
        print(f"Database dialect: {dialect}")
        print(f"Iterations per query: {iterations}")
        print("=" * 60)
    
    raw_query = _build_raw_query(dialect)
    preagg_query = PREAGG_QUERIES["mv_state_sales"]
    
    raw_times = []
    preagg_times = []
    
    for i in range(iterations):
        _, raw_time = _run_query(engine, raw_query)
        _, preagg_time = _run_query(engine, preagg_query)
        raw_times.append(raw_time)
        preagg_times.append(preagg_time)
        if verbose:
            print(f"Iteration {i + 1}:")
            print(f"  Raw JOIN:     {raw_time * 1000:.2f} ms")
            print(f"  Pre-agg:     {preagg_time * 1000:.2f} ms")
    
    avg_raw = sum(raw_times) / len(raw_times)
    avg_preagg = sum(preagg_times) / len(preagg_times)
    
    results = {
        "dialect": dialect,
        "iterations": iterations,
        "raw_query_avg_ms": avg_raw * 1000,
        "preagg_query_avg_ms": avg_preagg * 1000,
        "speedup": avg_raw / avg_preagg if avg_preagg > 0 else 0,
        "raw_query": raw_query,
        "preagg_query": preagg_query,
    }
    
    if verbose:
        print("=" * 60)
        print("Summary:")
        print(f"  Average raw JOIN time:   {avg_raw * 1000:.2f} ms")
        print(f"  Average pre-agg time:    {avg_preagg * 1000:.2f} ms")
        if avg_preagg > 0:
            print(f"  Speedup:                 {results['speedup']:.2f}x")
        print("=" * 60)
    
    return results


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
    """Print example SQL queries for documentation.
    
    Args:
        dialect: Database dialect ('sqlite' or 'mysql').
    """
    ym = _build_year_month_expr(dialect, "timestamp")
    
    print("\n" + "=" * 60)
    print(f"Example SQL for {dialect.upper()}:")
    print("=" * 60)
    
    print("\n1. Raw JOIN query (ad-hoc):")
    print(f"""
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
LIMIT 20;
""")
    
    print("\n2. Using pre-aggregation table:")
    print("""
SELECT year_month, customer_state, total_gmv, total_orders
FROM mv_state_sales
ORDER BY total_gmv DESC
LIMIT 20;
""")


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
