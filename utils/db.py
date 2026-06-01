from __future__ import annotations

from pathlib import Path
from typing import Optional
import time
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from config.settings import DB_URL, BASE_DIR


def get_engine(db_url: Optional[str] = None) -> Engine:
    """Create SQLAlchemy engine. Defaults to config DB_URL."""
    url = db_url or DB_URL
    if url.startswith("sqlite:///"):
        db_path = url.replace("sqlite:///", "")
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    return create_engine(url, pool_pre_ping=True, future=True)


def read_df(sql: str, params: Optional[dict] = None, engine: Optional[Engine] = None) -> pd.DataFrame:
    engine = engine or get_engine()
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn, params=params or {})


def execute_sql(sql: str, engine: Optional[Engine] = None) -> None:
    engine = engine or get_engine()
    statements = [s.strip() for s in sql.split(";") if s.strip()]
    with engine.begin() as conn:
        for stmt in statements:
            conn.execute(text(stmt))


def timed_read_df(sql: str, params: Optional[dict] = None, engine: Optional[Engine] = None):
    start = time.perf_counter()
    df = read_df(sql, params=params, engine=engine)
    elapsed = time.perf_counter() - start
    return df, elapsed


def table_exists(table_name: str, engine: Optional[Engine] = None) -> bool:
    engine = engine or get_engine()
    try:
        with engine.connect() as conn:
            dialect = engine.dialect.name
            if dialect == "sqlite":
                q = "SELECT name FROM sqlite_master WHERE type IN ('table','view') AND name = :name"
                row = conn.execute(text(q), {"name": table_name}).fetchone()
            else:
                q = "SELECT table_name FROM information_schema.tables WHERE table_schema = DATABASE() AND table_name = :name"
                row = conn.execute(text(q), {"name": table_name}).fetchone()
        return row is not None
    except SQLAlchemyError:
        return False


def ensure_sample_db_if_needed(engine: Optional[Engine] = None) -> None:
    """If default SQLite demo db is missing, create it."""
    engine = engine or get_engine()
    if engine.dialect.name == "sqlite" and not table_exists("orders", engine):
        from utils.init_sample_db import build_sample_database
        build_sample_database(str(BASE_DIR / "data" / "sample_olist.db"))
