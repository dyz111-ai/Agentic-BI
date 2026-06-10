"""图表规划提示：供 Visualization Agent LLM 参考，由可视化 Agent 决定 charts 数组。"""

CHART_PLAN_GUIDANCE = """
【你的任务：根据用户问题 + 已查到的数据表，决定画哪些图】
必须输出 charts 数组（至少 1 项，最多 6 项）。只使用下方「可用数据表」中存在的 table 名。

支持的 type：
- line：时间序列（x=year_month/date，y=total_gmv 等）。预测类 intent=forecast 时 title 用「GMV 趋势与预测」，只规划这一张月度折线即可（系统会自动叠加 Prophet 预测线）。
- bar：分类排名（州/品类/卖家/支付等）。
- pie：占比（如 payment_dist）。
- scatter：两数值关系（如 weight_freight 的重量 vs 运费）。
- map：巴西各州气泡图（table 需含 customer_state + 数值列如 total_gmv）。
- heatmap：两分类 × 一数值（如 payment_type × year_month）。
- delivery：各州配送时长+准时率双轴图（table=delivery_by_state，需 avg_delivery_days、on_time_rate）。
- wordcloud：差评/好评关键词词云（table 可留空 ""；variant=negative 或 positive；需 intent=review 且 queries 含 reviews，NLP 分析后渲染）。

【按表名/场景的建议（仅供参考，你来决定）】
| 表名 / 场景 | 建议图表 |
| monthly_sales | line 月度 GMV 趋势 |
| state_sales / state_sales_2017 | bar 各州 GMV 排名；可选 map 各州气泡图 |
| delivery_by_state | delivery 配送时长与准时率 |
| payment_dist | pie 或 bar 支付方式分布 |
| payment_monthly | heatmap 支付方式×月份 |
| top_categories | bar(h) Top 品类 GMV |
| low_score_sellers | bar(h) 低评分卖家 |
| weight_freight | scatter 重量 vs 运费 |
| reviews | bar 差评品类（需 NLP 聚合后由系统处理）；wordcloud 差评/好评关键词各 1 张 |

【注意】
- chart.table 必须对应 queries 里的 name（wordcloud 除外）。
- 只回答配送时不要规划 monthly_sales / state_sales 的销售图。
- 只回答支付时不要规划无关地图。
- 预测题只规划 1 张 monthly_sales 的 line，不要重复规划两张趋势图。
"""
