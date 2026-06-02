from __future__ import annotations

import sys
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

BASE_DIR = Path(__file__).resolve().parents[1]


def _ensure_project_root() -> None:
    """Ensure project root is in sys.path for direct script execution."""
    root = str(BASE_DIR)
    if root not in sys.path:
        sys.path.insert(0, root)


BRAZIL_STATES = ["SP", "RJ", "MG", "BA", "PE", "PR", "RS", "SC", "CE", "GO"]

CITIES = {
    "SP": "sao paulo", "RJ": "rio de janeiro", "MG": "belo horizonte", 
    "BA": "salvador", "PE": "recife", "PR": "curitiba", 
    "RS": "porto alegre", "SC": "florianopolis", "CE": "fortaleza", "GO": "goiania"
}

CATEGORIES = [
    ("moveis_decoracao", "furniture_decor"),
    ("beleza_saude", "health_beauty"),
    ("esporte_lazer", "sports_leisure"),
    ("informatica_acessorios", "computers_accessories"),
    ("cama_mesa_banho", "bed_bath_table"),
    ("utilidades_domesticas", "housewares"),
]

PAYMENT_TYPES = ["credit_card", "boleto", "voucher", "debit_card"]

STATE_COORDS = {
    "SP": (-23.55, -46.63), "RJ": (-22.90, -43.20), "MG": (-19.92, -43.94),
    "BA": (-12.97, -38.50), "PE": (-8.05, -34.88), "PR": (-25.43, -49.27),
    "RS": (-30.03, -51.23), "SC": (-27.59, -48.55), "CE": (-3.73, -38.52),
    "GO": (-16.68, -49.25),
}


def _ensure_project_root() -> None:
    """Ensure project root is in sys.path for direct script execution."""
    root = str(BASE_DIR)
    if root not in sys.path:
        sys.path.insert(0, root)


def _generate_customers(rng: random.Random, n: int = 260) -> pd.DataFrame:
    """Generate customer data.
    
    Args:
        rng: Random number generator.
        n: Number of customers to generate.
        
    Returns:
        DataFrame with customer data.
    """
    customers = []
    for i in range(n):
        state = rng.choice(BRAZIL_STATES)
        customers.append({
            "customer_id": f"c_{i:05d}",
            "customer_unique_id": f"cu_{i:05d}",
            "customer_zip_code_prefix": 10000 + i,
            "customer_city": CITIES[state],
            "customer_state": state,
        })
    return pd.DataFrame(customers)


def _generate_sellers(rng: random.Random, n: int = 60) -> pd.DataFrame:
    """Generate seller data.
    
    Args:
        rng: Random number generator.
        n: Number of sellers to generate.
        
    Returns:
        DataFrame with seller data.
    """
    sellers = []
    for i in range(n):
        state = rng.choice(BRAZIL_STATES)
        sellers.append({
            "seller_id": f"s_{i:04d}",
            "seller_zip_code_prefix": 20000 + i,
            "seller_city": CITIES[state],
            "seller_state": state,
        })
    return pd.DataFrame(sellers)


def _generate_products(rng: random.Random, np_random: np.random.Generator, n: int = 120) -> pd.DataFrame:
    """Generate product data.
    
    Args:
        rng: Random number generator.
        np_random: NumPy random generator.
        n: Number of products to generate.
        
    Returns:
        DataFrame with product data.
    """
    products = []
    for i in range(n):
        cat_pt, _ = rng.choice(CATEGORIES)
        weight = max(100, int(np_random.lognormal(mean=7.2, sigma=0.8)))
        products.append({
            "product_id": f"p_{i:05d}",
            "product_category_name": cat_pt,
            "product_name_length": rng.randint(15, 70),
            "product_description_length": rng.randint(100, 2500),
            "product_photos_qty": rng.randint(1, 8),
            "product_weight_g": weight,
            "product_length_cm": rng.randint(10, 80),
            "product_height_cm": rng.randint(2, 60),
            "product_width_cm": rng.randint(10, 60),
        })
    return pd.DataFrame(products)


def _generate_category_translation() -> pd.DataFrame:
    """Generate category translation data.
    
    Returns:
        DataFrame with category translation.
    """
    return pd.DataFrame([
        {"product_category_name": pt, "product_category_name_english": en} 
        for pt, en in CATEGORIES
    ])


def _generate_orders_and_related(
    rng: random.Random,
    np_random: np.random.Generator,
    customers_df: pd.DataFrame,
    sellers_df: pd.DataFrame,
    products_df: pd.DataFrame,
    n_orders: int = 800
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate orders, order items, payments, and reviews.
    
    Args:
        rng: Random number generator.
        np_random: NumPy random generator.
        customers_df: Customers DataFrame.
        sellers_df: Sellers DataFrame.
        products_df: Products DataFrame.
        n_orders: Number of orders to generate.
        
    Returns:
        Tuple of (orders, items, payments, reviews) DataFrames.
    """
    orders, items, payments, reviews = [], [], [], []
    start_date = datetime(2017, 1, 1)
    
    for i in range(n_orders):
        order_id = f"o_{i:06d}"
        customer = customers_df.sample(1, random_state=i).iloc[0]
        
        purchase = start_date + timedelta(days=rng.randint(0, 640), hours=rng.randint(0, 23))
        approved = purchase + timedelta(hours=rng.randint(1, 24))
        carrier = approved + timedelta(days=rng.randint(1, 5))
        
        base_days = rng.randint(4, 22)
        if customer.customer_state in ["BA", "PE", "CE"]:
            base_days += rng.randint(3, 12)
        
        delivered = purchase + timedelta(days=base_days)
        estimated = purchase + timedelta(days=rng.randint(12, 28))
        status = "delivered" if rng.random() > 0.03 else rng.choice(["shipped", "canceled", "invoiced"])
        
        orders.append({
            "order_id": order_id,
            "customer_id": customer.customer_id,
            "order_status": status,
            "order_purchase_timestamp": purchase.isoformat(sep=" "),
            "order_approved_at": approved.isoformat(sep=" "),
            "order_delivered_carrier_date": carrier.isoformat(sep=" "),
            "order_delivered_customer_date": delivered.isoformat(sep=" ") if status == "delivered" else None,
            "order_estimated_delivery_date": estimated.isoformat(sep=" "),
        })
        
        n_items = rng.choice([1, 1, 1, 2, 2, 3])
        order_total = 0.0
        freight_total = 0.0
        chosen_seller = sellers_df.sample(1, random_state=1000 + i).iloc[0]
        
        for j in range(n_items):
            prod = products_df.sample(1, random_state=2000 + i + j).iloc[0]
            price = round(float(np_random.gamma(2.0, 55.0) + 20), 2)
            freight = round(max(6.0, 0.004 * prod.product_weight_g + rng.uniform(5, 35)), 2)
            order_total += price
            freight_total += freight
            
            items.append({
                "order_id": order_id,
                "order_item_id": j + 1,
                "product_id": prod.product_id,
                "seller_id": chosen_seller.seller_id,
                "shipping_limit_date": (purchase + timedelta(days=5)).isoformat(sep=" "),
                "price": price,
                "freight_value": freight,
            })
        
        pay_type = rng.choices(
            PAYMENT_TYPES, 
            weights=[0.70, 0.18, 0.07, 0.05]
        )[0]
        installments = (
            rng.choice([1, 1, 1, 2, 3, 4, 6, 10]) 
            if pay_type == "credit_card" else 1
        )
        
        payments.append({
            "order_id": order_id,
            "payment_sequential": 1,
            "payment_type": pay_type,
            "payment_installments": installments,
            "payment_value": round(order_total + freight_total, 2),
        })
        
        delay = delivered > estimated
        score = rng.choices([1, 2, 3, 4, 5], weights=[0.08, 0.08, 0.14, 0.30, 0.40])[0]
        if delay:
            score = min(score, rng.choice([1, 2, 3]))
        
        neg_comments = [
            "entrega atrasada produto ruim", 
            "demorou muito vendedor ruim", 
            "frete caro embalagem danificada"
        ]
        pos_comments = [
            "produto excelente entrega rapida", 
            "gostei muito qualidade boa", 
            "preco bom chegou antes do prazo"
        ]
        comment = (
            rng.choice(neg_comments) if score <= 2 
            else rng.choice(pos_comments) if score >= 4 
            else "produto ok entrega normal"
        )
        
        reviews.append({
            "review_id": f"r_{i:06d}",
            "order_id": order_id,
            "review_score": score,
            "review_comment_title": comment.split()[0],
            "review_comment_message": comment,
            "review_creation_date": (delivered + timedelta(days=1)).isoformat(sep=" ") if status == "delivered" else None,
            "review_answer_timestamp": (delivered + timedelta(days=2)).isoformat(sep=" ") if status == "delivered" else None,
        })
    
    return (
        pd.DataFrame(orders),
        pd.DataFrame(items),
        pd.DataFrame(payments),
        pd.DataFrame(reviews)
    )


def _generate_geolocation(rng: random.Random) -> pd.DataFrame:
    """Generate geolocation data.
    
    Args:
        rng: Random number generator.
        
    Returns:
        DataFrame with geolocation data.
    """
    geos = []
    for idx, state in enumerate(BRAZIL_STATES):
        lat, lng = STATE_COORDS[state]
        for k in range(10):
            geos.append({
                "geolocation_zip_code_prefix": 10000 + idx * 10 + k,
                "geolocation_lat": lat + rng.uniform(-0.4, 0.4),
                "geolocation_lng": lng + rng.uniform(-0.4, 0.4),
                "geolocation_city": CITIES[state],
                "geolocation_state": state,
            })
    return pd.DataFrame(geos)


def _sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Replace NaN with None for database compatibility."""
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].where(pd.notnull(df[col]), None)
    return df


def build_sample_database(
    db_path: Optional[str | Path] = None,
    n_orders: int = 800,
    seed: int = 42
) -> str:
    """Build sample SQLite database for demo/testing.
    
    Args:
        db_path: Path for the SQLite database file.
        n_orders: Number of orders to generate.
        seed: Random seed for reproducibility.
        
    Returns:
        Path to created database file.
    """
    _ensure_project_root()
    
    if db_path is None:
        db_path = BASE_DIR / "data" / "sample_olist.db"
    else:
        db_path = Path(db_path)
    
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    if db_path.exists():
        db_path.unlink()
    
    rng = random.Random(seed)
    np.random.seed(seed)
    np_random = np.random.default_rng(seed)
    
    print(f"Generating sample data with {n_orders} orders...")
    customers_df = _generate_customers(rng)
    sellers_df = _generate_sellers(rng)
    products_df = _generate_products(rng, np_random)
    translation_df = _generate_category_translation()
    orders_df, items_df, payments_df, reviews_df = _generate_orders_and_related(
        rng, np_random, customers_df, sellers_df, products_df, n_orders
    )
    geolocation_df = _generate_geolocation(rng)
    
    tables = {
        "customers": _sanitize_dataframe(customers_df),
        "sellers": _sanitize_dataframe(sellers_df),
        "products": _sanitize_dataframe(products_df),
        "product_category_name_translation": _sanitize_dataframe(translation_df),
        "orders": _sanitize_dataframe(orders_df),
        "order_items": _sanitize_dataframe(items_df),
        "payments": _sanitize_dataframe(payments_df),
        "order_reviews": _sanitize_dataframe(reviews_df),
        "geolocation": _sanitize_dataframe(geolocation_df),
    }
    
    engine = create_engine(f"sqlite:///{db_path}", future=True)
    
    print("Writing to database...")
    for name, df in tables.items():
        df.to_sql(name, engine, if_exists="replace", index=False)
        print(f"  {name}: {len(df):,} rows")
    
    from utils.preaggregation import refresh_preaggregations
    print("Creating pre-aggregation tables...")
    refresh_preaggregations(engine)
    
    print(f"\nSample database created: {db_path}")
    return str(db_path)


def get_sample_db_stats() -> dict[str, int]:
    """Get statistics about the sample database.
    
    Returns:
        Dictionary with table names and row counts.
    """
    _ensure_project_root()
    db_path = BASE_DIR / "data" / "sample_olist.db"
    
    if not db_path.exists():
        return {}
    
    engine = create_engine(f"sqlite:///{db_path}")
    stats = {}
    
    tables = [
        "customers", "sellers", "products", "orders", 
        "order_items", "payments", "order_reviews", "geolocation",
        "mv_monthly_sales", "mv_state_sales"
    ]
    
    for table in tables:
        try:
            with engine.connect() as conn:
                from sqlalchemy import text
                result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
                stats[table] = result.scalar() or 0
        except Exception:
            pass
    
    return stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build sample SQLite database for Agentic BI demo.")
    parser.add_argument(
        "--db-path",
        default=None,
        help=f"Path for SQLite database (default: {BASE_DIR / 'data' / 'sample_olist.db'})"
    )
    parser.add_argument(
        "--n-orders",
        type=int,
        default=800,
        help="Number of orders to generate (default: 800)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show statistics of existing sample database"
    )
    
    args = parser.parse_args()
    
    if args.stats:
        stats = get_sample_db_stats()
        if stats:
            print("Sample Database Statistics:")
            print("-" * 40)
            for name, count in stats.items():
                print(f"  {name}: {count:,} rows")
        else:
            print("Sample database not found. Run without --stats to create one.")
    else:
        db_path = build_sample_database(args.db_path, args.n_orders, args.seed)
        print("\n" + "=" * 50)
        print("Database created successfully!")
        print("=" * 50)
