BASE_TABLES = {
    "orders": "订单主表：order_id, customer_id, order_status, order_purchase_timestamp, order_approved_at, order_delivered_customer_date, order_estimated_delivery_date",
    "order_items": "订单商品表：order_id, product_id, seller_id, price, freight_value, shipping_limit_date",
    "products": "商品表：product_id, product_category_name, product_weight_g, product_length_cm, product_height_cm, product_width_cm",
    "customers": "顾客表：customer_id, customer_unique_id, customer_city, customer_state",
    "sellers": "卖家表：seller_id, seller_city, seller_state",
    "payments": "支付表：order_id, payment_type, payment_installments, payment_value",
    "order_reviews": "评论表：order_id, review_score, review_comment_title, review_comment_message",
    "geolocation": "地理位置表：geolocation_zip_code_prefix, geolocation_lat, geolocation_lng, geolocation_city, geolocation_state",
    "product_category_name_translation": "品类翻译表：product_category_name(葡语), product_category_name_english(英语)。注意：英语列名是 product_category_name_english，不是 product_category_english",
}

PRE_AGG_TABLES = {
    "mv_monthly_sales": {
        "grain": "year_month",
        "fields": "year_month, total_gmv, total_orders, avg_basket, total_freight",
        "use_cases": "月度销售趋势、GMV 环比、预测建模"
    },
    "mv_state_sales": {
        "grain": "year_month + customer_state",
        "fields": "year_month, customer_state, total_gmv, total_orders, unique_customers",
        "use_cases": "各州销售额排名、区域市场对比、地图"
    },
    "mv_category_sales": {
        "grain": "year_month + product_category_english",
        "fields": "year_month, product_category_english, total_gmv, total_orders, avg_price",
        "use_cases": "品类表现分析、下降品类定位"
    },
    "mv_delivery_perf": {
        "grain": "year_month + customer_state",
        "fields": "year_month, customer_state, avg_delivery_days, on_time_rate, delayed_orders, total_orders",
        "use_cases": "配送延迟诊断、准时率分析"
    },
    "mv_seller_perf": {
        "grain": "year_month + seller_id",
        "fields": "year_month, seller_id, seller_state, total_gmv, total_orders, avg_review_score",
        "use_cases": "卖家绩效、低评分卖家定位"
    },
    "mv_payment_dist": {
        "grain": "year_month + payment_type",
        "fields": "year_month, payment_type, total_transactions, avg_installments, total_value",
        "use_cases": "支付偏好、分期率对比"
    },
}
