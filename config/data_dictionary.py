BASE_TABLES = {
    "orders": {
        "description": "订单主表",
        "columns": [
            "order_id (PK)",
            "customer_id (FK)",
            "order_status (delivered/shipped/canceled/invoiced/...)",
            "order_purchase_timestamp",
            "order_approved_at",
            "order_delivered_carrier_date",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ],
        "key_fields": ["order_id", "customer_id", "order_status", "order_purchase_timestamp"],
    },
    "order_items": {
        "description": "订单商品表",
        "columns": [
            "order_id (FK)",
            "order_item_id",
            "product_id (FK)",
            "seller_id (FK)",
            "shipping_limit_date",
            "price",
            "freight_value",
        ],
        "key_fields": ["order_id", "product_id", "seller_id", "price", "freight_value"],
    },
    "products": {
        "description": "商品表",
        "columns": [
            "product_id (PK)",
            "product_category_name (FK)",
            "product_name_length",
            "product_description_length",
            "product_photos_qty",
            "product_weight_g",
            "product_length_cm",
            "product_height_cm",
            "product_width_cm",
        ],
        "key_fields": ["product_id", "product_category_name", "product_weight_g"],
    },
    "customers": {
        "description": "顾客表",
        "columns": [
            "customer_id (PK)",
            "customer_unique_id",
            "customer_zip_code_prefix",
            "customer_city",
            "customer_state",
        ],
        "key_fields": ["customer_id", "customer_unique_id", "customer_state"],
    },
    "sellers": {
        "description": "卖家表",
        "columns": [
            "seller_id (PK)",
            "seller_zip_code_prefix",
            "seller_city",
            "seller_state",
        ],
        "key_fields": ["seller_id", "seller_state"],
    },
    "payments": {
        "description": "支付表",
        "columns": [
            "order_id (FK)",
            "payment_sequential",
            "payment_type (credit_card/boleto/voucher/debit_card)",
            "payment_installments",
            "payment_value",
        ],
        "key_fields": ["order_id", "payment_type", "payment_installments", "payment_value"],
    },
    "order_reviews": {
        "description": "评论表",
        "columns": [
            "review_id (PK)",
            "order_id (FK)",
            "review_score (1-5)",
            "review_comment_title",
            "review_comment_message",
            "review_creation_date",
            "review_answer_timestamp",
        ],
        "key_fields": ["order_id", "review_score", "review_comment_message"],
    },
    "geolocation": {
        "description": "地理位置表",
        "columns": [
            "geolocation_zip_code_prefix",
            "geolocation_lat",
            "geolocation_lng",
            "geolocation_city",
            "geolocation_state",
        ],
        "key_fields": ["geolocation_zip_code_prefix", "geolocation_state"],
    },
    "product_category_name_translation": {
        "description": "品类翻译表",
        "columns": [
            "product_category_name (PK, 葡语)",
            "product_category_name_english (英语)",
        ],
        "key_fields": ["product_category_name", "product_category_name_english"],
        "note": "英语列名是 product_category_name_english，不是 product_category_english",
    },
}


PRE_AGG_TABLES = {
    "mv_monthly_sales": {
        "grain": "year_month",
        "description": "月度销售汇总",
        "fields": [
            "year_month (YYYY-MM格式)",
            "total_gmv (总成交金额)",
            "total_orders (总订单数)",
            "avg_basket (平均客单价)",
            "total_freight (总运费)",
        ],
        "use_cases": [
            "月度销售趋势分析",
            "GMV 环比/同比分析",
            "销售预测建模",
        ],
        "source_tables": ["orders", "payments", "order_items"],
    },
    "mv_state_sales": {
        "grain": "year_month + customer_state",
        "description": "各州销售汇总",
        "fields": [
            "year_month (YYYY-MM格式)",
            "customer_state (州代码，如SP/RJ/MG)",
            "total_gmv (总成交金额)",
            "total_orders (总订单数)",
            "unique_customers (独立客户数)",
        ],
        "use_cases": [
            "各州销售额排名",
            "区域市场对比",
            "地理可视化（地图气泡图）",
            "区域配送分析",
        ],
        "source_tables": ["orders", "customers", "payments"],
    },
    "mv_category_sales": {
        "grain": "year_month + product_category_english",
        "description": "品类销售汇总",
        "fields": [
            "year_month (YYYY-MM格式)",
            "product_category_english (英文品类名)",
            "total_gmv (总成交金额)",
            "total_orders (总订单数)",
            "avg_price (平均商品单价)",
        ],
        "use_cases": [
            "品类表现分析",
            "热门/冷门品类定位",
            "品类销售趋势",
        ],
        "source_tables": ["orders", "order_items", "products", "product_category_name_translation"],
    },
    "mv_delivery_perf": {
        "grain": "year_month + customer_state",
        "description": "配送绩效汇总",
        "fields": [
            "year_month (YYYY-MM格式)",
            "customer_state (州代码)",
            "avg_delivery_days (平均配送天数)",
            "on_time_rate (准时交付率，0-1)",
            "delayed_orders (延迟订单数)",
            "total_orders (总订单数)",
        ],
        "use_cases": [
            "配送延迟诊断",
            "准时率分析",
            "各州配送表现对比",
            "配送问题区域定位",
        ],
        "source_tables": ["orders", "customers"],
        "calculation_notes": [
            "avg_delivery_days = 实际配送天数均值（从下单到送达）",
            "on_time_rate = 准时交付订单 / 总订单数",
            "delayed_orders = 实际送达 > 预计送达的订单数",
        ],
    },
    "mv_seller_perf": {
        "grain": "year_month + seller_id",
        "description": "卖家绩效汇总",
        "fields": [
            "year_month (YYYY-MM格式)",
            "seller_id (卖家ID)",
            "seller_state (卖家所在州)",
            "total_gmv (总成交金额)",
            "total_orders (总订单数)",
            "avg_review_score (平均评分，1-5)",
        ],
        "use_cases": [
            "卖家绩效排名",
            "低评分卖家定位",
            "卖家地域分布分析",
        ],
        "source_tables": ["orders", "order_items", "sellers", "order_reviews"],
    },
    "mv_payment_dist": {
        "grain": "year_month + payment_type",
        "description": "支付方式分布汇总",
        "fields": [
            "year_month (YYYY-MM格式)",
            "payment_type (支付方式)",
            "total_transactions (总交易数)",
            "avg_installments (平均分期数)",
            "total_value (总支付金额)",
        ],
        "use_cases": [
            "支付偏好分析",
            "分期付款使用率",
            "各支付方式占比",
        ],
        "source_tables": ["orders", "payments"],
        "payment_types": [
            "credit_card (信用卡)",
            "boleto (巴西本地支付票据)",
            "voucher (代金券)",
            "debit_card (借记卡)",
        ],
    },
}


TABLE_RELATIONSHIPS = {
    "description": "表关联关系",
    "joins": [
        {"from": "orders", "to": "customers", "on": "orders.customer_id = customers.customer_id"},
        {"from": "orders", "to": "order_items", "on": "orders.order_id = order_items.order_id"},
        {"from": "orders", "to": "payments", "on": "orders.order_id = payments.order_id"},
        {"from": "orders", "to": "order_reviews", "on": "orders.order_id = order_reviews.order_id"},
        {"from": "order_items", "to": "products", "on": "order_items.product_id = products.product_id"},
        {"from": "order_items", "to": "sellers", "on": "order_items.seller_id = sellers.seller_id"},
        {
            "from": "products",
            "to": "product_category_name_translation",
            "on": "products.product_category_name = product_category_name_translation.product_category_name"
        },
    ],
}


DATABASE_NOTES = {
    "sqlite": {
        "date_functions": {
            "year_month": "strftime('%Y-%m', column)",
            "date_diff": "julianday(end_date) - julianday(start_date)",
        },
        "group_by": "标准SQL兼容，无需特殊处理",
        "limit": "LIMIT N 或 LIMIT N OFFSET M",
        "cte": "支持（SQLite 3.8.3+）",
    },
    "mysql": {
        "date_functions": {
            "year_month": "DATE_FORMAT(column, '%Y-%m')",
            "date_diff": "DATEDIFF(end_date, start_date)",
        },
        "group_by": "ONLY_FULL_GROUP_BY 模式下，GROUP BY 子句中的所有非聚合列必须明确列出",
        "limit": "LIMIT N 或 LIMIT N OFFSET M",
        "cte": "支持（MySQL 8.0+）",
    },
}


def get_table_description(table_name: str) -> str:
    """Get description for a table."""
    if table_name in BASE_TABLES:
        return BASE_TABLES[table_name]["description"]
    if table_name in PRE_AGG_TABLES:
        return PRE_AGG_TABLES[table_name]["description"]
    return "Unknown table"


def get_table_fields(table_name: str) -> list[str]:
    """Get field names for a table."""
    if table_name in BASE_TABLES:
        return BASE_TABLES[table_name].get("key_fields", [])
    if table_name in PRE_AGG_TABLES:
        return PRE_AGG_TABLES[table_name]["fields"]
    return []


def format_schema_for_llm() -> str:
    """Format schema information for LLM prompts."""
    lines = []
    
    lines.append("=" * 60)
    lines.append("【预聚合表 - 优先使用】")
    lines.append("=" * 60)
    
    for name, info in PRE_AGG_TABLES.items():
        lines.append(f"\n表名: {name}")
        lines.append(f"描述: {info['description']}")
        lines.append(f"粒度: {info['grain']}")
        lines.append(f"字段: {', '.join(info['fields'])}")
        lines.append(f"用途: {'; '.join(info['use_cases'])}")
    
    lines.append("\n" + "=" * 60)
    lines.append("【基础表 - 回退使用】")
    lines.append("=" * 60)
    
    for name, info in BASE_TABLES.items():
        lines.append(f"\n表名: {name}")
        lines.append(f"描述: {info['description']}")
        lines.append(f"关键字段: {', '.join(info['key_fields'])}")
        if "note" in info:
            lines.append(f"注意: {info['note']}")
    
    lines.append("\n" + "=" * 60)
    lines.append("【表关联】")
    lines.append("=" * 60)
    
    for join in TABLE_RELATIONSHIPS["joins"]:
        lines.append(f"{join['from']}.{join['to']}: {join['on']}")
    
    return "\n".join(lines)
