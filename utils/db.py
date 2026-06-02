from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Union
import time
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, Connection
from sqlalchemy.exc import SQLAlchemyError

BASE_DIR = Path(__file__).resolve().parents[1]

def _get_default_db_url() -> str:
    return f"sqlite:///{BASE_DIR / 'data' / 'sample_olist.db'}"

DB_URL = _get_default_db_url()

def _ensure_project_root() -> None:
    """Ensure project root is in sys.path for direct script execution."""
    root = str(BASE_DIR)
    if root not in sys.path:
        sys.path.insert(0, root)

def ensure_sample_db_if_needed(engine: Engine) -> None:
    """Check if we have a database, and create a sample one if needed.
    
    This function:
    1. Checks if orders table exists
    2. If not, uses init_sample_db.py to create a sample database
    3. Skips if database already has data
    
    Args:
        engine: SQLAlchemy engine
    """
    _ensure_project_root()
    
    try:
        # Check if orders table exists
        if table_exists("orders", engine):
            # Check if it has any data
            with engine.connect() as conn:
                result = conn.execute(text("SELECT COUNT(*) FROM orders"))
                count = result.scalar() or 0
                if count > 0:
                    return
        
        # If we get here, we need to create sample data
        from utils.init_sample_db import build_sample_database
        db_path = BASE_DIR / "data" / "sample_olist.db"
        build_sample_database(db_path=db_path, n_orders=800, seed=42)
        
    except Exception:
        # If anything goes wrong, try to create sample data
        try:
            from utils.init_sample_db import build_sample_database
            db_path = BASE_DIR / "data" / "sample_olist.db"
            build_sample_database(db_path=db_path, n_orders=800, seed=42)
        except Exception:
            # If even that fails, just return and let the app handle it
            pass

def get_engine(db_url: Optional[str] = None) -> Engine:
    """Create SQLAlchemy engine with cross-database compatibility.
    
    Args:
        db_url: Database URL. Defaults to config DB URL.
        
    Returns:
        SQLAlchemy Engine instance.
    """
    _ensure_project_root()
    url = db_url or DB_URL
    if url.startswith("sqlite:///"):
        db_path = url.replace("sqlite:///", "")
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    return create_engine(url, pool_pre_ping=True, future=True)

def is_sqlite(engine: Optional[Engine] = None) -> bool:
    """Check if engine is SQLite.
    
    Args:
        engine: SQLAlchemy engine. Uses default if None.
        
    Returns:
        True if SQLite, False if MySQL.
    """
    if engine is None:
        engine = get_engine()
    return engine.dialect.name == "sqlite"

def is_mysql(engine: Optional[Engine] = None) -> bool:
    """Check if engine is MySQL.
    
    Args:
        engine: SQLAlchemy engine. Uses default if None.
        
    Returns:
        True if MySQL, False if SQLite.
    """
    if engine is None:
        engine = get_engine()
    return engine.dialect.name in ("mysql", "mariadb")

def get_dialect(engine: Optional[Engine] = None) -> str:
    """Get database dialect name.
    
    Args:
        engine: SQLAlchemy engine. Uses default if None.
        
    Returns:
        'sqlite' or 'mysql'.
    """
    if engine is None:
        engine = get_engine()
    return engine.dialect.name

def year_month_expr(engine: Optional[Engine] = None, col: str = "timestamp_col") -> str:
    """Generate year-month extraction expression for current database.
    
    SQLite: strftime('%Y-%m', col)
    MySQL:  DATE_FORMAT(col, '%Y-%m')
    
    NOTE: SQLAlchemy text() 只有在使用 :param 命名参数时才会处理 %
          普通字符串中的 % 不会被处理，所以不需要转义
    
    Args:
        engine: SQLAlchemy engine.
        col: Column name or expression.
        
    Returns:
        SQL expression for year-month extraction.
    """
    dialect = get_dialect(engine)
    if dialect == "sqlite":
        return f"strftime('%Y-%m', {col})"
    return f"DATE_FORMAT({col}, '%Y-%m')"

def date_diff_expr(
    engine: Optional[Engine] = None,
    end_col: str = "end_col",
    start_col: str = "start_col"
) -> str:
    """Generate date difference expression in days.
    
    SQLite: julianday(end_col) - julianday(start_col)
    MySQL: DATEDIFF(end_col, start_col)
    
    Args:
        engine: SQLAlchemy engine.
        end_col: End date column.
        start_col: Start date column.
        
    Returns:
        SQL expression for date difference in days.
    """
    dialect = get_dialect(engine)
    if dialect == "sqlite":
        return f"(julianday({end_col}) - julianday({start_col}))"
    return f"DATEDIFF({end_col}, {start_col})"

def read_df(
    sql: str,
    params: Optional[dict] = None,
    engine: Optional[Engine] = None
) -> pd.DataFrame:
    """Execute SELECT query and return DataFrame.
    
    Args:
        sql: SQL query string.
        params: Query parameters.
        engine: SQLAlchemy engine.
        
    Returns:
        Query results as DataFrame.
    """
    engine = engine or get_engine()
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn, params=params or {})

def execute_sql(
    sql: str,
    engine: Optional[Engine] = None,
    use_raw: bool = False
) -> None:
    """Execute SQL statements (CREATE, DROP, INSERT, etc.).
    
    Args:
        sql: SQL statement string. Supports multiple statements separated by semicolons.
        engine: SQLAlchemy engine.
        use_raw: Use raw_connection instead of connection context manager.
    """
    engine = engine or get_engine()
    statements = [s.strip() for s in sql.split(";") if s.strip()]
    
    if use_raw:
        with engine.raw_connection() as conn:
            cursor = conn.cursor()
            try:
                for stmt in statements:
                    if stmt:
                        cursor.execute(stmt)
                conn.commit()
            finally:
                cursor.close()
    else:
        with engine.begin() as conn:
            for stmt in statements:
                if stmt:
                    conn.execute(text(stmt))

def execute_statement(
    stmt: str,
    engine: Optional[Engine] = None
) -> None:
    """Execute a single SQL statement using raw SQL.
    
    Args:
        stmt: Single SQL statement.
        engine: SQLAlchemy engine.
    """
    engine = engine or get_engine()
    dialect = get_dialect(engine)
    
    if dialect == "sqlite":
        with engine.begin() as conn:
            conn.execute(text(stmt))
    else:
        # For MySQL, use raw pymysql cursor directly
        # The SQL should already have %% escaped for DATE_FORMAT patterns
        raw_conn = engine.raw_connection()
        cursor = raw_conn.cursor()
        try:
            cursor.execute(stmt)
            raw_conn.commit()
        finally:
            cursor.close()
            raw_conn.close()

def timed_read_df(
    sql: str,
    params: Optional[dict] = None,
    engine: Optional[Engine] = None
) -> tuple[pd.DataFrame, float]:
    """Execute query and return DataFrame with execution time.
    
    Args:
        sql: SQL query string.
        params: Query parameters.
        engine: SQLAlchemy engine.
        
    Returns:
        Tuple of (DataFrame, elapsed_time_in_seconds).
    """
    start = time.perf_counter()
    df = read_df(sql, params=params, engine=engine)
    elapsed = time.perf_counter() - start
    return df, elapsed

def table_exists(table_name: str, engine: Optional[Engine] = None) -> bool:
    """Check if table or view exists in database.
    
    Args:
        table_name: Name of table or view.
        engine: SQLAlchemy engine.
        
    Returns:
        True if exists, False otherwise.
    """
    engine = engine or get_engine()
    try:
        dialect = get_dialect(engine)
        with engine.connect() as conn:
            if dialect == "sqlite":
                q = """
                    SELECT name FROM sqlite_master 
                    WHERE type IN ('table','view') AND name = :name
                """
                row = conn.execute(text(q), {"name": table_name}).fetchone()
            else:
                q = """
                    SELECT table_name FROM information_schema.tables 
                    WHERE table_schema = DATABASE() AND table_name = :name
                """
                row = conn.execute(text(q), {"name": table_name}).fetchone()
            return row is not None
    except Exception:
        return False

def create_index_sql(table_name: str, index_name: str, columns: list, engine: Optional[Engine] = None) -> str:
    """Generate SQL to create an index.
    
    Args:
        table_name: Name of the table.
        index_name: Name of the index.
        columns: List of column names to include in the index.
        engine: SQLAlchemy engine (optional).
        
    Returns:
        SQL statement to create the index.
    """
    # Use backticks for column names to handle reserved keywords like year_month
    # For MySQL, TEXT columns need length specification in indexes; SQLite does not support this
    dialect = get_dialect(engine) if engine else "sqlite"
    indexed_columns = []
    for col in columns:
        # Special handling for columns that might be TEXT type in MySQL
        if dialect == "mysql" and col in ["customer_state", "product_category_english", "seller_state", "payment_type", "seller_id"]:
            indexed_columns.append(f"`{col}`(50)")
        else:
            indexed_columns.append(f"`{col}`")
    
    columns_str = ", ".join(indexed_columns)
    return f"CREATE INDEX {index_name} ON {table_name} ({columns_str})"

def drop_table_if_exists(table_name: str, engine: Optional[Engine] = None) -> None:
    """Drop a table if it exists.
    
    Args:
        table_name: Name of the table to drop.
        engine: SQLAlchemy engine (optional).
    """
    engine = engine or get_engine()
    dialect = get_dialect(engine)
    
    with engine.begin() as conn:
        sql = f"DROP TABLE IF EXISTS {table_name}"
        if dialect == "sqlite":
            conn.execute(text(sql))
        else:
            conn.execute(text(sql))
