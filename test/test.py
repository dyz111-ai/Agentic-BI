"""Inspect data/olist.db: base tables vs mv_monthly_sales consistency."""

from __future__ import annotations

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "data" / "olist.db"


def q(conn: sqlite3.Connection, sql: str) -> list[tuple]:
    cur = conn.execute(sql)
    cols = [d[0] for d in cur.description] if cur.description else []
    rows = cur.fetchall()
    return cols, rows


def print_table(title: str, cols: list[str], rows: list[tuple], limit: int | None = None) -> None:
    print(f"\n{'=' * 60}")
    print(title)
    print("=" * 60)
    if not rows:
        print("(no rows)")
        return
    show = rows if limit is None else rows[:limit]
    widths = [max(len(str(c)), *(len(str(r[i])) for r in show)) for i, c in enumerate(cols)]
    fmt = "  ".join(f"{{:{w}}}" for w in widths)
    print(fmt.format(*cols))
    print("  ".join("-" * w for w in widths))
    for row in show:
        print(fmt.format(*[str(v) if v is not None else "NULL" for v in row]))
    if limit and len(rows) > limit:
        print(f"... ({len(rows) - limit} more rows)")


def main() -> None:
    if not DB_PATH.exists():
        print(f"Database not found: {DB_PATH}")
        return

    print(f"Database: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)

    # --- row counts ---
    tables = [
        "orders", "payments", "order_items", "customers", "sellers",
        "mv_monthly_sales", "mv_state_sales",
    ]
    print("\n--- Table row counts ---")
    for t in tables:
        try:
            n = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            print(f"  {t:22s} {n:>10,}")
        except sqlite3.OperationalError:
            print(f"  {t:22s} (missing)")

    # --- mv_monthly_sales full ---
    cols, rows = q(
        conn,
        """
        SELECT year_month, total_gmv, total_orders, avg_basket, total_freight
        FROM mv_monthly_sales
        ORDER BY year_month
        """,
    )
    print_table("mv_monthly_sales (all months)", cols, rows)

    # --- suspicious low GMV ---
    cols, rows = q(
        conn,
        """
        SELECT year_month, total_gmv, total_orders, avg_basket
        FROM mv_monthly_sales
        ORDER BY total_gmv ASC
        LIMIT 10
        """,
    )
    print_table("mv_monthly_sales — lowest GMV months", cols, rows)

    # --- orders by month (base table) ---
    cols, rows = q(
        conn,
        """
        SELECT strftime('%Y-%m', order_purchase_timestamp) AS ym,
               COUNT(DISTINCT order_id) AS orders,
               MIN(order_purchase_timestamp) AS first_ts,
               MAX(order_purchase_timestamp) AS last_ts
        FROM orders
        WHERE order_purchase_timestamp IS NOT NULL
        GROUP BY ym
        ORDER BY ym
        """,
    )
    print_table("orders — count by month (base table)", cols, rows)

    # --- GMV from payments joined to orders by month ---
    cols, rows = q(
        conn,
        """
        SELECT strftime('%Y-%m', o.order_purchase_timestamp) AS ym,
               ROUND(SUM(p.payment_value), 2) AS gmv_from_payments,
               COUNT(DISTINCT o.order_id) AS orders_with_payment_rows,
               COUNT(*) AS payment_rows
        FROM orders o
        JOIN payments p ON o.order_id = p.order_id
        WHERE o.order_purchase_timestamp IS NOT NULL
        GROUP BY ym
        ORDER BY ym
        """,
    )
    print_table("payments — GMV by order month (base table)", cols, rows)

    # --- orders without any payment ---
    cols, rows = q(
        conn,
        """
        SELECT strftime('%Y-%m', o.order_purchase_timestamp) AS ym,
               COUNT(DISTINCT o.order_id) AS orders_no_payment
        FROM orders o
        LEFT JOIN payments p ON o.order_id = p.order_id
        WHERE o.order_purchase_timestamp IS NOT NULL
          AND p.order_id IS NULL
        GROUP BY ym
        ORDER BY ym
        """,
    )
    print_table("orders with NO payment record — by month", cols, rows)

    # --- side-by-side compare 2016 ---
    cols, rows = q(
        conn,
        """
        WITH mv AS (
            SELECT year_month AS ym, total_gmv AS mv_gmv, total_orders AS mv_orders
            FROM mv_monthly_sales
        ),
        base AS (
            SELECT strftime('%Y-%m', o.order_purchase_timestamp) AS ym,
                   ROUND(SUM(COALESCE(p.pay_sum, 0)), 2) AS base_gmv,
                   COUNT(DISTINCT o.order_id) AS base_orders
            FROM orders o
            LEFT JOIN (
                SELECT order_id, SUM(payment_value) AS pay_sum
                FROM payments GROUP BY order_id
            ) p ON o.order_id = p.order_id
            WHERE o.order_purchase_timestamp IS NOT NULL
            GROUP BY ym
        )
        SELECT b.ym,
               b.base_orders,
               b.base_gmv,
               m.mv_orders,
               m.mv_gmv,
               ROUND(m.mv_gmv - b.base_gmv, 2) AS gmv_diff
        FROM base b
        LEFT JOIN mv m ON b.ym = m.ym
        WHERE b.ym LIKE '2016-%'
        ORDER BY b.ym
        """,
    )
    print_table("2016: base vs mv_monthly_sales", cols, rows)

    # --- timestamp sample for suspicious months ---
    for ym in ("2016-09", "2016-11", "2016-12"):
        cols, rows = q(
            conn,
            f"""
            SELECT order_id, order_purchase_timestamp, order_status
            FROM orders
            WHERE strftime('%Y-%m', order_purchase_timestamp) = '{ym}'
            ORDER BY order_purchase_timestamp
            LIMIT 5
            """,
        )
        print_table(f"orders sample — {ym} (first 5)", cols, rows)

    # --- null / invalid timestamps ---
    cols, rows = q(
        conn,
        """
        SELECT
            SUM(CASE WHEN order_purchase_timestamp IS NULL THEN 1 ELSE 0 END) AS null_ts,
            COUNT(*) AS total_orders
        FROM orders
        """,
    )
    print_table("orders — null purchase timestamps", cols, rows)

    conn.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
