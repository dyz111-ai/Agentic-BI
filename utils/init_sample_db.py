from __future__ import annotations

from pathlib import Path
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DB = BASE_DIR / "data" / "sample_olist.db"

BRAZIL_STATES = ["SP", "RJ", "MG", "BA", "PE", "PR", "RS", "SC", "CE", "GO"]
CITIES = {
    "SP": "sao paulo", "RJ": "rio de janeiro", "MG": "belo horizonte", "BA": "salvador",
    "PE": "recife", "PR": "curitiba", "RS": "porto alegre", "SC": "florianopolis",
    "CE": "fortaleza", "GO": "goiania"
}
CATS = [
    ("moveis_decoracao", "furniture_decor"),
    ("beleza_saude", "health_beauty"),
    ("esporte_lazer", "sports_leisure"),
    ("informatica_acessorios", "computers_accessories"),
    ("cama_mesa_banho", "bed_bath_table"),
    ("utilidades_domesticas", "housewares"),
]
PAY_TYPES = ["credit_card", "boleto", "voucher", "debit_card"]


def build_sample_database(db_path: str | Path = DEFAULT_DB, n_orders: int = 800) -> None:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    rng = random.Random(42)
    np.random.seed(42)

    customers = []
    for i in range(260):
        state = rng.choice(BRAZIL_STATES)
        customers.append({
            "customer_id": f"c_{i:05d}",
            "customer_unique_id": f"cu_{i:05d}",
            "customer_zip_code_prefix": 10000 + i,
            "customer_city": CITIES[state],
            "customer_state": state,
        })
    customers_df = pd.DataFrame(customers)

    sellers = []
    for i in range(60):
        state = rng.choice(BRAZIL_STATES)
        sellers.append({
            "seller_id": f"s_{i:04d}",
            "seller_zip_code_prefix": 20000 + i,
            "seller_city": CITIES[state],
            "seller_state": state,
        })
    sellers_df = pd.DataFrame(sellers)

    products = []
    for i in range(120):
        cat_pt, _ = rng.choice(CATS)
        weight = max(100, int(np.random.lognormal(mean=7.2, sigma=0.8)))
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
    products_df = pd.DataFrame(products)

    translation_df = pd.DataFrame([
        {"product_category_name": pt, "product_category_name_english": en} for pt, en in CATS
    ])

    orders, items, payments, reviews = [], [], [], []
    start = datetime(2017, 1, 1)
    for i in range(n_orders):
        order_id = f"o_{i:06d}"
        customer = customers_df.sample(1, random_state=i).iloc[0]
        purchase = start + timedelta(days=rng.randint(0, 640), hours=rng.randint(0, 23))
        approved = purchase + timedelta(hours=rng.randint(1, 24))
        carrier = approved + timedelta(days=rng.randint(1, 5))
        base_days = rng.randint(4, 22)
        # Northeast-like states deliberately a bit slower for diagnostic demo
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
        order_total = 0
        freight_total = 0
        chosen_seller = sellers_df.sample(1, random_state=1000+i).iloc[0]
        for j in range(n_items):
            prod = products_df.sample(1, random_state=2000+i+j).iloc[0]
            price = round(float(np.random.gamma(2.0, 55.0) + 20), 2)
            freight = round(max(6, 0.004 * prod.product_weight_g + rng.uniform(5, 35)), 2)
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
        pay_type = rng.choices(PAY_TYPES, weights=[0.7, 0.18, 0.07, 0.05])[0]
        installments = rng.choice([1, 1, 1, 2, 3, 4, 6, 10]) if pay_type == "credit_card" else 1
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
        neg_comments = ["entrega atrasada produto ruim", "demorou muito vendedor ruim", "frete caro embalagem danificada"]
        pos_comments = ["produto excelente entrega rapida", "gostei muito qualidade boa", "preco bom chegou antes do prazo"]
        comment = rng.choice(neg_comments if score <= 2 else pos_comments if score >= 4 else ["produto ok entrega normal"])
        reviews.append({
            "review_id": f"r_{i:06d}",
            "order_id": order_id,
            "review_score": score,
            "review_comment_title": comment.split()[0],
            "review_comment_message": comment,
            "review_creation_date": (delivered + timedelta(days=1)).isoformat(sep=" ") if status == "delivered" else None,
            "review_answer_timestamp": (delivered + timedelta(days=2)).isoformat(sep=" ") if status == "delivered" else None,
        })

    geos = []
    state_coords = {
        "SP": (-23.55, -46.63), "RJ": (-22.90, -43.20), "MG": (-19.92, -43.94), "BA": (-12.97, -38.50),
        "PE": (-8.05, -34.88), "PR": (-25.43, -49.27), "RS": (-30.03, -51.23), "SC": (-27.59, -48.55),
        "CE": (-3.73, -38.52), "GO": (-16.68, -49.25),
    }
    for idx, st in enumerate(BRAZIL_STATES):
        lat, lng = state_coords[st]
        for k in range(10):
            geos.append({
                "geolocation_zip_code_prefix": 10000 + idx*10 + k,
                "geolocation_lat": lat + rng.uniform(-0.4, 0.4),
                "geolocation_lng": lng + rng.uniform(-0.4, 0.4),
                "geolocation_city": CITIES[st],
                "geolocation_state": st,
            })

    engine = create_engine(f"sqlite:///{db_path}")
    tables = {
        "customers": customers_df,
        "sellers": sellers_df,
        "products": products_df,
        "product_category_name_translation": translation_df,
        "orders": pd.DataFrame(orders),
        "order_items": pd.DataFrame(items),
        "payments": pd.DataFrame(payments),
        "order_reviews": pd.DataFrame(reviews),
        "geolocation": pd.DataFrame(geos),
    }
    for name, df in tables.items():
        df.to_sql(name, engine, if_exists="replace", index=False)

    from utils.preaggregation import refresh_preaggregations
    refresh_preaggregations(engine)
    print(f"Sample database created: {db_path}")


if __name__ == "__main__":
    build_sample_database()
