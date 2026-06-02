from __future__ import annotations

import sys
import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

BASE_DIR = Path(__file__).resolve().parents[1]


TABLE_FILES = {
    "orders": "olist_orders_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "products": "olist_products_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "order_reviews": "olist_order_reviews_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "product_category_name_translation": "product_category_name_translation.csv",
}

DATE_COLUMNS = {
    "orders": [
        "order_purchase_timestamp", "order_approved_at", 
        "order_delivered_carrier_date", "order_delivered_customer_date", 
        "order_estimated_delivery_date"
    ],
    "order_items": ["shipping_limit_date"],
    "order_reviews": ["review_creation_date", "review_answer_timestamp"],
}


def _ensure_project_root() -> None:
    """Ensure project root is in sys.path for direct script execution."""
    root = str(BASE_DIR)
    if root not in sys.path:
        sys.path.insert(0, root)


def _get_engine(db_url: str) -> Engine:
    """Create SQLAlchemy engine.
    
    Args:
        db_url: Database URL string.
        
    Returns:
        SQLAlchemy Engine instance.
    """
    return create_engine(db_url, future=True, pool_pre_ping=True)


def _parse_date_column(df: pd.DataFrame, table_name: str) -> pd.DataFrame:
    """Parse date columns for a table.
    
    Args:
        df: DataFrame to process.
        table_name: Name of the source table.
        
    Returns:
        DataFrame with parsed datetime columns.
    """
    date_cols = DATE_COLUMNS.get(table_name, [])
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def _sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Sanitize DataFrame for database insertion.
    
    Handles:
    - NaN values replaced with None
    - Data type consistency
    
    Args:
        df: DataFrame to sanitize.
        
    Returns:
        Sanitized DataFrame.
    """
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].where(pd.notnull(df[col]), None)
    return df


def load_olist_csvs(
    db_url: str,
    data_dir: str | Path,
    create_preagg: bool = True,
    chunksize: int = 5000
) -> dict[str, int]:
    """Load Olist CSV files into database.
    
    Args:
        db_url: SQLAlchemy database URL.
        data_dir: Directory containing Olist CSV files.
        create_preagg: Whether to create pre-aggregation tables.
        chunksize: Number of rows to insert per batch.
        
    Returns:
        Dictionary mapping table names to row counts.
        
    Raises:
        FileNotFoundError: If a required CSV file is missing.
    """
    _ensure_project_root()
    data_dir = Path(data_dir)
    engine = _get_engine(db_url)
    
    row_counts = {}
    
    for table, filename in TABLE_FILES.items():
        path = data_dir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required file: {path}\n"
                f"Please download from Kaggle: Brazilian E-Commerce Public Dataset by Olist"
            )
        
        print(f"Loading {path} -> {table}")
        df = pd.read_csv(path)
        df = _parse_date_column(df, table)
        df = _sanitize_dataframe(df)
        
        df.to_sql(
            table,
            engine,
            if_exists="replace",
            index=False,
            chunksize=chunksize
        )
        
        row_counts[table] = len(df)
        print(f"  -> {len(df):,} rows inserted")
    
    if create_preagg:
        print("\nCreating pre-aggregation tables...")
        
        # 强制清除模块缓存并重新加载，确保我们使用最新代码！
        for mod in list(sys.modules.keys()):
            if mod.startswith("utils.") or mod == "utils":
                del sys.modules[mod]
        
        from utils.preaggregation import refresh_preaggregations
        refresh_preaggregations(engine)
        print("Pre-aggregation tables created successfully.")
    
    print("\nDone! All tables are ready.")
    return row_counts


def load_single_table(
    db_url: str,
    table_name: str,
    csv_path: str | Path
) -> int:
    """Load a single CSV file into database.
    
    Args:
        db_url: SQLAlchemy database URL.
        table_name: Target table name.
        csv_path: Path to CSV file.
        
    Returns:
        Number of rows inserted.
    """
    _ensure_project_root()
    engine = _get_engine(db_url)
    csv_path = Path(csv_path)
    
    if not csv_path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")
    
    print(f"Loading {csv_path} -> {table_name}")
    df = pd.read_csv(csv_path)
    
    date_cols = DATE_COLUMNS.get(table_name, [])
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    
    df = _sanitize_dataframe(df)
    df.to_sql(table_name, engine, if_exists="replace", index=False)
    
    print(f"  -> {len(df):,} rows inserted")
    return len(df)


def verify_database(db_url: str) -> dict[str, any]:
    """Verify database contents.
    
    Args:
        db_url: SQLAlchemy database URL.
        
    Returns:
        Dictionary with database verification results.
    """
    _ensure_project_root()
    from utils.db import get_table_row_count, table_exists
    
    engine = _get_engine(db_url)
    dialect = engine.dialect.name
    
    results = {
        "dialect": dialect,
        "tables": {},
        "preaggregation_tables": [
            "mv_monthly_sales",
            "mv_state_sales", 
            "mv_category_sales",
            "mv_delivery_perf",
            "mv_seller_perf",
            "mv_payment_dist",
        ],
        "missing_tables": [],
        "missing_preagg": [],
    }
    
    all_tables = list(TABLE_FILES.keys()) + results["preaggregation_tables"]
    
    for table in all_tables:
        if table_exists(table, engine):
            count = get_table_row_count(table, engine)
            results["tables"][table] = count
        else:
            if table in results["preaggregation_tables"]:
                results["missing_preagg"].append(table)
            else:
                results["missing_tables"].append(table)
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load Olist CSV files into database with cross-platform support."
    )
    parser.add_argument(
        "--db-url",
        required=True,
        help="SQLAlchemy DB URL, e.g.:\n"
             "  SQLite: sqlite:///data/olist.db\n"
             "  MySQL: mysql+pymysql://user:pwd@localhost:3306/olist_agentic_bi"
    )
    parser.add_argument(
        "--data-dir",
        default="data/raw",
        help="Directory containing Olist CSV files (default: data/raw)"
    )
    parser.add_argument(
        "--no-preagg",
        action="store_true",
        help="Skip creating pre-aggregation tables"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Only verify existing database contents"
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=5000,
        help="Number of rows per insert batch (default: 5000)"
    )
    
    args = parser.parse_args()
    
    if args.verify:
        print(f"Verifying database: {args.db_url}")
        results = verify_database(args.db_url)
        print(f"\nDatabase dialect: {results['dialect']}")
        print(f"\nTables loaded ({len(results['tables'])}):")
        for name, count in results["tables"].items():
            print(f"  {name}: {count:,} rows")
        if results["missing_tables"]:
            print(f"\nMissing base tables: {', '.join(results['missing_tables'])}")
        if results["missing_preagg"]:
            print(f"Missing pre-aggregation tables: {', '.join(results['missing_preagg'])}")
    else:
        row_counts = load_olist_csvs(
            args.db_url,
            args.data_dir,
            create_preagg=not args.no_preagg,
            chunksize=args.chunksize
        )
        print("\n" + "="*50)
        print("Summary:")
        print("="*50)
        for table, count in row_counts.items():
            print(f"  {table}: {count:,} rows")
