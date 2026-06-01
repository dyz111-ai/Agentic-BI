from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine

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
        "order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date",
        "order_delivered_customer_date", "order_estimated_delivery_date"
    ],
    "order_items": ["shipping_limit_date"],
    "order_reviews": ["review_creation_date", "review_answer_timestamp"],
}


def load_olist_csvs(db_url: str, data_dir: str | Path) -> None:
    data_dir = Path(data_dir)
    engine = create_engine(db_url, future=True)

    for table, filename in TABLE_FILES.items():
        path = data_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}")
        print(f"Loading {path} -> {table}")
        df = pd.read_csv(path)
        for col in DATE_COLUMNS.get(table, []):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        df.to_sql(table, engine, if_exists="replace", index=False, chunksize=5000)
        print(f"  rows={len(df)}")

    from utils.preaggregation import refresh_preaggregations
    refresh_preaggregations(engine)
    print("Done. Base tables and pre-aggregation tables are ready.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-url", required=True, help="SQLAlchemy DB URL, e.g. mysql+pymysql://root:pwd@localhost:3306/olist_agentic_bi")
    parser.add_argument("--data-dir", default="data/raw", help="Directory containing Olist CSV files")
    args = parser.parse_args()
    load_olist_csvs(args.db_url, args.data_dir)
