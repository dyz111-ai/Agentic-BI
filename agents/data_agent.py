from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import json
import re
import pandas as pd
from sqlalchemy.engine import Engine

from utils.db import timed_read_df, fix_mysql_sql
from utils.llm import LLMClient
from utils.conversation_memory import is_follow_up_question, normalize_memory
from config.data_dictionary import BASE_TABLES, PRE_AGG_TABLES


@dataclass
class DataResult:
    intent: str
    sql_blocks: list[dict] = field(default_factory=list)
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    summary: str = ""
    used_preaggregation: bool = False
    elapsed: dict[str, float] = field(default_factory=dict)
    routing_method: str = "rule"      # rule / llm / fallback
    plan: dict[str, Any] = field(default_factory=dict)
    llm_error: str = ""


class DataAnalysisAgent:
    """LLM-powered natural-language-to-SQL agent.

    Priority:
    1) If LLM_API_KEY is configured, call DeepSeek/OpenAI-compatible LLM to plan intent and SQL.
    2) Validate SQL: read-only SELECT/WITH only, no destructive keywords, add LIMIT to risky base-table queries.
    3) Execute the generated SQL.
    4) If LLM fails or returns invalid SQL, fall back to deterministic built-in query templates.

    This keeps the system demonstrable and safe: the Agent is truly LLM-driven when API is available,
    but it still runs offline for course demos.
    """

    ALLOWED_INTENTS = {
        "sales", "forecast", "delivery", "payment", "category",
        "seller", "review", "weight_freight", "overall", "custom_sql"
    }

    FORBIDDEN_SQL = re.compile(
        r"\b(insert|update|delete|drop|alter|truncate|create|replace|attach|detach|pragma|vacuum|grant|revoke)\b",
        re.IGNORECASE,
    )

    def __init__(self, engine: Engine):
        self.engine = engine
        self.llm = LLMClient()

    @property
    def _dialect(self) -> str:
        return self.engine.dialect.name

    def _is_mysql(self) -> bool:
        return self._dialect in ("mysql", "mariadb")

    def _q(self, ident: str) -> str:
        """Quote identifier for MySQL reserved words (e.g. year_month)."""
        return f"`{ident}`" if self._is_mysql() else ident

    def _adapt_sql(self, sql: str) -> str:
        return fix_mysql_sql(sql.strip(), self.engine)

    def analyze(self, question: str, conversation_context: str = "", memory: dict | None = None) -> DataResult:
        memory = normalize_memory(memory)
        # First choice: LLM intent routing + SQL generation.
        if self.llm.enabled:
            try:
                return self._llm_analyze(question, conversation_context=conversation_context)
            except Exception as exc:
                fallback = self._rule_analyze(question, memory=memory)
                fallback.routing_method = "fallback"
                fallback.llm_error = str(exc)
                self._finalize_result_tables(question, fallback)
                return fallback
        result = self._rule_analyze(question, memory=memory)
        self._finalize_result_tables(question, result)
        return result

    def _finalize_result_tables(self, question: str, result: DataResult) -> None:
        self._normalize_table_columns(result)

    # ----------------------- LLM planning path -----------------------
    def _llm_analyze(self, question: str, conversation_context: str = "") -> DataResult:
        plan = self._ask_llm_for_plan(question, conversation_context=conversation_context)
        intent = str(plan.get("intent", "custom_sql")).strip()
        if intent not in self.ALLOWED_INTENTS:
            intent = "custom_sql"
        intent = self._refine_intent(question, intent)
        queries = plan.get("queries") or []
        if not queries:
            raise ValueError("LLM 没有返回 queries")

        result = DataResult(
            intent=intent,
            used_preaggregation=bool(plan.get("used_preaggregation", False)),
            routing_method="llm",
            plan=plan,
        )

        any_base = False
        for i, q in enumerate(queries, start=1):
            name = str(q.get("name") or f"llm_result_{i}").strip()
            source = str(q.get("source") or "llm-generated").strip()
            sql = str(q.get("sql") or "").strip()
            if not sql:
                continue
            safe_sql = self._sanitize_sql(sql, source=source)
            n, df, elapsed = self._run(name, safe_sql)
            result.tables[n] = df
            result.sql_blocks.append({"name": n, "sql": safe_sql, "source": source})
            result.elapsed[n] = elapsed
            if "base" in source.lower() or not source.startswith("pre"):
                any_base = True

        if not result.tables:
            raise ValueError("LLM 返回的 SQL 均为空或无法执行")

        result.used_preaggregation = bool(result.used_preaggregation and not any_base)
        self._normalize_table_columns(result)
        result.summary = self._build_summary(result, plan)
        return result

    def _normalize_table_columns(self, result: DataResult) -> None:
        """Normalize common LLM column aliases so downstream agents can read tables."""
        for name, df in list(result.tables.items()):
            if not isinstance(df, pd.DataFrame) or df.empty:
                continue
            rename: dict[str, str] = {}
            lower = {str(c).lower(): c for c in df.columns}
            if "customer_state" not in lower:
                for alias in ("state", "customer_state_code", "uf"):
                    if alias in lower:
                        rename[lower[alias]] = "customer_state"
                        break
            if "total_gmv" not in lower:
                for alias in ("gmv", "sales", "total_sales", "revenue", "total_value"):
                    if alias in lower:
                        rename[lower[alias]] = "total_gmv"
                        break
            if "year_month" not in lower:
                for alias in ("month", "order_month", "ym"):
                    if alias in lower:
                        rename[lower[alias]] = "year_month"
                        break
            if rename:
                result.tables[name] = df.rename(columns=rename)

    def _ask_llm_for_plan(self, question: str, conversation_context: str = "") -> dict:
        dialect = self.engine.dialect.name
        ym_col = self._q("year_month")
        schema_text = self._schema_prompt()
        if conversation_context:
            question_block = f"{conversation_context.rstrip()}\n{question}"
        else:
            question_block = f"用户问题：{question}"
        prompt = f"""
你是 Agentic BI 系统中的 Data Analysis Agent。你的任务是把中文/英文业务问题转换为可执行 SQL。

数据库方言：{dialect}

必须遵守：
1. 只生成只读 SELECT 查询，不允许 INSERT/UPDATE/DELETE/DROP/ALTER/CREATE 等操作。
2. 优先使用预聚合表 mv_*。只有预聚合表无法覆盖评论文本、商品重量/体积、具体订单明细等问题时，才回退基础表。
3. 直接返回严格 JSON，不要 Markdown，不要解释文字，不要输出思考过程。
4. queries 最多 3 条。每条 SQL 尽量 LIMIT 5000 以内。
5. 如果问题属于常见任务，请使用下列固定 name，便于后续可视化 Agent 识别：
   - 月度销售：monthly_sales
   - 各州销售：state_sales 或 state_sales_2017
   - 配送：delivery_by_state 或 delivery_monthly
   - 支付：payment_dist 或 payment_monthly
   - 品类：top_categories 或 category_monthly
   - 卖家：low_score_sellers
   - 评论差评：reviews
   - 重量运费：weight_freight
6. intent 只能从以下选择：sales, forecast, delivery, payment, category, seller, review, weight_freight, overall, custom_sql。
7. used_preaggregation 表示是否主要命中预聚合表。
8. 对年份过滤，推荐使用 {ym_col} LIKE '2017-%'。
   若需从基础表 orders 按月聚合：
   - MySQL: DATE_FORMAT(order_purchase_timestamp, '%Y-%m')
   - SQLite: strftime('%Y-%m', order_purchase_timestamp)
9. 品类英文名在 product_category_name_translation 表中的列名是 product_category_name_english（不是 product_category_english）。
   预聚合表 mv_category_sales 才有 product_category_english 列。JOIN 翻译表时请写：
   COALESCE(t.product_category_name_english, p.product_category_name, 'unknown') AS product_category_english
10. 评论/差评问题：name 必须为 reviews，SQL 需包含 review_score、review_comment_title、review_comment_message、product_category_english。
11. 若用户问「按月/趋势/月度」：monthly_sales 必须返回 year_month + total_gmv 的时间序列（多行），不要只返回一个 SUM 总数。
12. 若用户问「各州/排名」：state_sales 或 state_sales_2017 需按 customer_state 汇总 total_gmv，可含 year_month 明细或已聚合结果。
13. 若数据库方言为 mysql，列名 year_month 必须写成反引号形式 `year_month`（MySQL 保留字 YEAR 会导致语法错误）。
14. 若提供了【会话上下文】，当前问题可能是追问（如「那个州呢」「该品类差评原因」）。请结合上一轮问题、意图、关键实体与回答，正确理解指代并在 SQL 中使用对应州/品类/卖家等过滤条件。

可用数据字典：
{schema_text}

{question_block}

请返回 JSON 格式：
{{
  "intent": "sales",
  "used_preaggregation": true,
  "reason": "为什么选择这些表",
  "queries": [
    {{"name": "monthly_sales", "source": "pre-aggregation", "sql": "SELECT ..."}}
  ]
}}
"""
        messages = [
            {"role": "system", "content": "你是严谨的数据分析 SQL Agent。只返回合法 JSON，不要思考过程，不要 Markdown。"},
            {"role": "user", "content": prompt},
        ]
        last_error = "LLM 未返回内容"
        for max_tokens in (4096, 8192):
            content = self.llm.chat(messages=messages, temperature=0.0, max_tokens=max_tokens)
            if not content:
                last_error = "LLM 未返回内容"
                continue
            if content.startswith("LLM 调用失败"):
                last_error = content
                continue
            try:
                return self._extract_json(content)
            except (json.JSONDecodeError, ValueError) as exc:
                last_error = f"LLM 返回 JSON 无法解析：{exc}"
                continue
        raise ValueError(last_error)

    def _schema_prompt(self) -> str:
        lines = ["【预聚合表，优先使用】"]
        for name, info in PRE_AGG_TABLES.items():
            lines.append(f"- {name}: 粒度={info['grain']}; 字段={info['fields']}; 用途={info['use_cases']}")
        lines.append("\n【基础表，回退使用】")
        for name, desc in BASE_TABLES.items():
            lines.append(f"- {name}: {desc}")
        lines.append("\n【常用关联】")
        lines.extend([
            "orders.customer_id = customers.customer_id",
            "orders.order_id = order_items.order_id",
            "orders.order_id = payments.order_id",
            "orders.order_id = order_reviews.order_id",
            "order_items.product_id = products.product_id",
            "order_items.seller_id = sellers.seller_id",
            "products.product_category_name = product_category_name_translation.product_category_name",
        ])
        return "\n".join(lines)

    def _extract_json(self, content: str) -> dict:
        text = content.strip()
        # Remove optional markdown fences.
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to recover the first JSON object.
            m = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if not m:
                raise
            return json.loads(m.group(0))

    def _sanitize_sql(self, sql: str, source: str = "") -> str:
        s = sql.strip()
        # Only allow one statement. Remove trailing semicolon; reject internal semicolons.
        s = s.rstrip(";").strip()
        if ";" in s:
            raise ValueError("SQL 包含多个语句，已拒绝执行")
        if self.FORBIDDEN_SQL.search(s):
            raise ValueError("SQL 包含危险关键字，已拒绝执行")
        if not re.match(r"^(select|with)\b", s, flags=re.IGNORECASE):
            raise ValueError("只允许 SELECT/WITH 查询")
        # Avoid huge base-table result sets. Pre-aggregation tables are small, but base fallback can be large.
        source_l = source.lower()
        if "base" in source_l and not re.search(r"\blimit\b", s, flags=re.IGNORECASE):
            s += " LIMIT 5000"
        return self._adapt_sql(s)

    def _preagg_order_by_month(self) -> str:
        return f"ORDER BY {self._q('year_month')}"

    def _build_summary(self, result: DataResult, plan: dict) -> str:
        prefix = "LLM 已完成问题理解、表选择与 SQL 生成。"
        reason = plan.get("reason") or ""
        if result.used_preaggregation:
            strategy = "本次优先命中预聚合表。"
        else:
            strategy = "本次包含基础表回退查询。"

        # Data-aware short summaries for common results.
        details = []
        tables = result.tables
        try:
            if "state_sales_2017" in tables and not tables["state_sales_2017"].empty:
                row = tables["state_sales_2017"].iloc[0]
                details.append(f"2017 年销售额最高的州是 {row.get('customer_state')}，GMV 约 {float(row.get('total_gmv', 0)):.2f}。")
            elif "state_sales" in tables and not tables["state_sales"].empty and "total_gmv" in tables["state_sales"].columns:
                row = tables["state_sales"].sort_values("total_gmv", ascending=False).iloc[0]
                details.append(f"销售额最高的州是 {row.get('customer_state')}，GMV 约 {float(row.get('total_gmv', 0)):.2f}。")
            if "payment_dist" in tables and not tables["payment_dist"].empty:
                row = tables["payment_dist"].iloc[0]
                details.append(f"最受欢迎的支付方式是 {row.get('payment_type')}。")
            if "delivery_by_state" in tables and not tables["delivery_by_state"].empty:
                row = tables["delivery_by_state"].iloc[0]
                if "avg_delivery_days" in row:
                    details.append(f"配送最慢的州是 {row.get('customer_state')}，平均配送约 {float(row.get('avg_delivery_days', 0)):.2f} 天。")
            if "top_categories" in tables and not tables["top_categories"].empty:
                row = tables["top_categories"].iloc[0]
                details.append(f"销售额最高/重点品类是 {row.get('product_category_english')}。")
            if "reviews" in tables:
                details.append("评论文本问题已查询评论与商品品类相关基础表，后续交给评论洞察 Agent 提取差评原因。")
            if "weight_freight" in tables:
                details.append("重量/尺寸/运费问题已查询商品物理属性与运费明细基础表。")
        except Exception:
            pass

        parts = [prefix, strategy]
        if reason:
            parts.append(f"选择原因：{reason}")
        parts.extend(details)
        return " ".join(parts)

    # ----------------------- deterministic fallback path -----------------------
    def _rule_analyze(self, question: str, memory: dict | None = None) -> DataResult:
        memory = normalize_memory(memory)
        q = question.lower()
        if self._has_forecast(q):
            return self._forecast_data()
        if self._has_review(q):
            return self._review_data()
        if self._has_weight_freight(q):
            return self._weight_freight()
        if self._has_delivery(q):
            return self._delivery()
        if self._has_payment(q):
            return self._payment()
        if self._has_category(q):
            return self._category()
        if self._has_seller(q):
            return self._seller()
        if self._is_overall(q):
            return self._overall()
        if is_follow_up_question(question) and memory.get("last_intent"):
            routed = self._route_by_intent(memory["last_intent"])
            if routed is not None:
                return routed
        return self._sales()

    def _route_by_intent(self, intent: str) -> DataResult | None:
        routes = {
            "sales": self._sales,
            "forecast": self._forecast_data,
            "delivery": self._delivery,
            "payment": self._payment,
            "category": self._category,
            "seller": self._seller,
            "review": self._review_data,
            "weight_freight": self._weight_freight,
            "overall": self._overall,
        }
        handler = routes.get(intent)
        if handler is None:
            return None
        return handler()

    def _run(self, name: str, sql: str, params: dict | None = None):
        df, elapsed = timed_read_df(self._adapt_sql(sql), params=params, engine=self.engine)
        return name, df, elapsed

    def _sql_monthly_all(self) -> str:
        ym = self._q("year_month")
        return f"SELECT * FROM mv_monthly_sales {self._preagg_order_by_month()}"

    def _sql_state_2017(self) -> str:
        ym = self._q("year_month")
        return f"""
            SELECT customer_state, ROUND(SUM(total_gmv), 2) AS total_gmv,
                   SUM(total_orders) AS total_orders,
                   SUM(unique_customers) AS unique_customers
            FROM mv_state_sales
            WHERE {ym} LIKE '2017-%'
            GROUP BY customer_state
            ORDER BY total_gmv DESC
            LIMIT 15
        """

    def _sql_order_monthly(self, table: str, extra_order: str = "") -> str:
        ym = self._q("year_month")
        order = f"{ym}, {extra_order}" if extra_order else ym
        return f"SELECT * FROM {table} ORDER BY {order}"

    @staticmethod
    def _is_overall(q: str) -> bool:
        keys = ["整体", "全局", "运营", "三大", "策略", "优化", "overall", "strategy"]
        return any(k in q for k in keys)

    @classmethod
    def _refine_intent(cls, question: str, raw_intent: str) -> str:
        """If LLM returns 'overall' but strong specific keywords exist, downgrade.

        Exception: when what-if / anomaly keywords are present, keep 'overall'
        so the orchestrator can fire the full what-if + anomaly pipeline.
        """
        if raw_intent != "overall":
            return raw_intent
        q = question.lower()
        what_if_signals = {"如果", "假设", "下架", "模拟", "what-if", "移除", "会怎样", "提升多少"}
        anomaly_signals = {"异常", "骤降", "突升", "预警", "报警"}
        if any(s in q for s in what_if_signals | anomaly_signals):
            return "overall"
        for name in ("delivery", "payment", "category", "seller", "review", "forecast", "weight_freight"):
            checker = getattr(cls, f"_has_{name}", None)
            if checker and checker(q):
                return name
        return raw_intent

    @staticmethod
    def _has_forecast(q: str) -> bool:
        return any(k in q for k in ["预测", "未来", "forecast", "6周", "六周", "趋势预测"])

    @staticmethod
    def _has_delivery(q: str) -> bool:
        return any(k in q for k in ["配送", "交付", "准时", "延迟", "物流", "delivery", "on-time"])

    @staticmethod
    def _has_payment(q: str) -> bool:
        return any(k in q for k in ["支付", "分期", "payment", "installment", "付款"])

    @staticmethod
    def _has_category(q: str) -> bool:
        return any(k in q for k in ["品类", "类别", "category", "产品类别"])

    @staticmethod
    def _has_seller(q: str) -> bool:
        return any(k in q for k in ["卖家", "seller", "差评卖家"])

    @staticmethod
    def _has_review(q: str) -> bool:
        return any(k in q for k in ["评论", "差评", "好评", "review", "情感", "原因"])

    @staticmethod
    def _has_weight_freight(q: str) -> bool:
        return any(k in q for k in ["重量", "尺寸", "运费", "weight", "freight", "size"])

    def _sales(self) -> DataResult:
        result = DataResult(
            intent="sales",
            used_preaggregation=True,
        )
        queries = {
            "monthly_sales": self._sql_monthly_all(),
            "state_sales_2017": self._sql_state_2017(),
        }
        for name, sql in queries.items():
            n, df, e = self._run(name, sql)
            result.tables[n] = df
            result.sql_blocks.append({"name": name, "sql": sql, "source": "pre-aggregation"})
            result.elapsed[n] = e
        top = "暂无数据"
        if not result.tables["state_sales_2017"].empty:
            row = result.tables["state_sales_2017"].iloc[0]
            top = f"2017 年销售额最高的州是 {row['customer_state']}，GMV 约 {row['total_gmv']:.2f}。"
        result.summary = f"本地兜底：已优先使用 mv_monthly_sales 与 mv_state_sales。{top}"
        return result

    def _forecast_data(self) -> DataResult:
        result = DataResult(
            intent="forecast",
            used_preaggregation=True,
        )
        sql = self._sql_monthly_all()
        n, df, e = self._run("monthly_sales", sql)
        result.tables[n] = df
        result.sql_blocks.append({"name": n, "sql": sql, "source": "pre-aggregation"})
        result.elapsed[n] = e
        result.summary = "本地兜底：已使用 mv_monthly_sales 获取历史 GMV 序列，用于未来 6 周预测。"
        return result

    def _delivery(self) -> DataResult:
        result = DataResult(
            intent="delivery",
            used_preaggregation=True,
        )
        queries = {
            "delivery_by_state": """
                SELECT customer_state,
                       ROUND(AVG(avg_delivery_days), 2) AS avg_delivery_days,
                       ROUND(AVG(on_time_rate), 4) AS on_time_rate,
                       SUM(delayed_orders) AS delayed_orders,
                       SUM(total_orders) AS total_orders
                FROM mv_delivery_perf
                GROUP BY customer_state
                ORDER BY avg_delivery_days DESC
            """,
        }
        for name, sql in queries.items():
            n, df, e = self._run(name, sql)
            result.tables[n] = df
            result.sql_blocks.append({"name": name, "sql": sql, "source": "pre-aggregation"})
            result.elapsed[n] = e
        if not result.tables["delivery_by_state"].empty:
            row = result.tables["delivery_by_state"].iloc[0]
            result.summary = f"本地兜底：配送最慢的州是 {row['customer_state']}，平均配送约 {row['avg_delivery_days']} 天，准时率 {row['on_time_rate']:.2%}。"
        else:
            result.summary = "未查询到配送数据。"
        return result

    def _payment(self) -> DataResult:
        result = DataResult(
            intent="payment",
            used_preaggregation=True,
        )
        queries = {
            "payment_dist": """
                SELECT payment_type,
                       SUM(total_transactions) AS total_transactions,
                       ROUND(AVG(avg_installments), 2) AS avg_installments,
                       ROUND(SUM(total_value), 2) AS total_value
                FROM mv_payment_dist
                GROUP BY payment_type
                ORDER BY total_transactions DESC
            """,
        }
        for name, sql in queries.items():
            n, df, e = self._run(name, sql)
            result.tables[n] = df
            result.sql_blocks.append({"name": name, "sql": sql, "source": "pre-aggregation"})
            result.elapsed[n] = e
        if not result.tables["payment_dist"].empty:
            row = result.tables["payment_dist"].iloc[0]
            result.summary = f"本地兜底：最受欢迎的支付方式是 {row['payment_type']}，交易数 {int(row['total_transactions'])}，平均分期 {row['avg_installments']}。"
        else:
            result.summary = "未查询到支付数据。"
        return result

    def _category(self) -> DataResult:
        result = DataResult(
            intent="category",
            used_preaggregation=True,
        )
        queries = {
            "top_categories": """
                SELECT product_category_english,
                       ROUND(SUM(total_gmv), 2) AS total_gmv,
                       SUM(total_orders) AS total_orders,
                       ROUND(AVG(avg_price), 2) AS avg_price
                FROM mv_category_sales
                GROUP BY product_category_english
                ORDER BY total_gmv DESC
            LIMIT 15
            """,
        }
        for name, sql in queries.items():
            n, df, e = self._run(name, sql)
            result.tables[n] = df
            result.sql_blocks.append({"name": name, "sql": sql, "source": "pre-aggregation"})
            result.elapsed[n] = e
        if not result.tables["top_categories"].empty:
            row = result.tables["top_categories"].iloc[0]
            result.summary = f"本地兜底：销售额最高的品类是 {row['product_category_english']}，GMV 约 {row['total_gmv']:.2f}。"
        else:
            result.summary = "未查询到品类数据。"
        return result

    def _seller(self) -> DataResult:
        result = DataResult(
            intent="seller",
            used_preaggregation=True,
        )
        sql = """
            SELECT seller_id, seller_state,
                   ROUND(SUM(total_gmv), 2) AS total_gmv,
                   SUM(total_orders) AS total_orders,
                   ROUND(AVG(avg_review_score), 2) AS avg_review_score
            FROM mv_seller_perf
            GROUP BY seller_id, seller_state
            HAVING total_orders >= 2
            ORDER BY avg_review_score ASC, total_orders DESC
            LIMIT 20
        """
        n, df, e = self._run("low_score_sellers", sql)
        result.tables[n] = df
        result.sql_blocks.append({"name": n, "sql": sql, "source": "pre-aggregation"})
        result.elapsed[n] = e
        result.summary = "本地兜底：已使用 mv_seller_perf 定位低评分卖家。"
        return result

    def _weight_freight(self) -> DataResult:
        result = DataResult(
            intent="weight_freight",
            used_preaggregation=False,
        )
        sql = """
            SELECT
                p.product_weight_g,
                (p.product_length_cm * p.product_height_cm * p.product_width_cm) AS product_volume_cm3,
                oi.freight_value,
                oi.price,
                o.order_status,
                CASE WHEN o.order_delivered_customer_date <= o.order_estimated_delivery_date THEN 'on_time' ELSE 'delayed' END AS delivery_status
            FROM order_items oi
            JOIN products p ON oi.product_id = p.product_id
            JOIN orders o ON oi.order_id = o.order_id
            WHERE p.product_weight_g IS NOT NULL AND oi.freight_value IS NOT NULL
            LIMIT 5000
        """
        n, df, e = self._run("weight_freight", sql)
        result.tables[n] = df
        result.sql_blocks.append({"name": n, "sql": sql, "source": "base tables fallback"})
        result.elapsed[n] = e
        result.summary = "本地兜底：该问题需要商品重量/体积与运费字段，已回退查询 order_items + products + orders 原始表。"
        return result

    def _review_data(self) -> DataResult:
        result = DataResult(
            intent="review",
            used_preaggregation=False,
        )
        sql = """
            SELECT
                r.order_id,
                r.review_score,
                r.review_comment_title,
                r.review_comment_message,
                COALESCE(t.product_category_name_english, p.product_category_name, 'unknown') AS product_category_english,
                oi.seller_id
            FROM order_reviews r
            LEFT JOIN order_items oi ON r.order_id = oi.order_id
            LEFT JOIN products p ON oi.product_id = p.product_id
            LEFT JOIN product_category_name_translation t ON p.product_category_name = t.product_category_name
            WHERE r.review_score IS NOT NULL
            LIMIT 8000
        """
        n, df, e = self._run("reviews", sql)
        result.tables[n] = df
        result.sql_blocks.append({"name": n, "sql": sql, "source": "base tables fallback"})
        result.elapsed[n] = e
        result.summary = "本地兜底：评论文本与差评原因不完全在预聚合表中，已回退查询评论与商品品类原始表。"
        return result

    def _overall(self) -> DataResult:
        result = DataResult(
            intent="overall",
            used_preaggregation=True,
        )
        queries = {
            "monthly_sales": self._sql_monthly_all(),
            "state_sales": """
                SELECT customer_state, ROUND(SUM(total_gmv),2) AS total_gmv, SUM(total_orders) AS total_orders
                FROM mv_state_sales GROUP BY customer_state ORDER BY total_gmv DESC LIMIT 15
            """,
            "delivery_by_state": """
                SELECT customer_state, ROUND(AVG(avg_delivery_days),2) AS avg_delivery_days,
                       ROUND(AVG(on_time_rate),4) AS on_time_rate, SUM(delayed_orders) AS delayed_orders
                FROM mv_delivery_perf GROUP BY customer_state ORDER BY avg_delivery_days DESC LIMIT 15
            """,
            "top_categories": """
                SELECT product_category_english, ROUND(SUM(total_gmv),2) AS total_gmv, SUM(total_orders) AS total_orders
                FROM mv_category_sales GROUP BY product_category_english ORDER BY total_gmv DESC LIMIT 15
            """,
            "payment_dist": """
                SELECT payment_type, SUM(total_transactions) AS total_transactions, ROUND(AVG(avg_installments),2) AS avg_installments
                FROM mv_payment_dist GROUP BY payment_type ORDER BY total_transactions DESC
            """,
            "low_score_sellers": """
                SELECT seller_id, seller_state, SUM(total_orders) AS total_orders, ROUND(AVG(avg_review_score),2) AS avg_review_score
                FROM mv_seller_perf GROUP BY seller_id, seller_state HAVING total_orders >= 2
                ORDER BY avg_review_score ASC, total_orders DESC LIMIT 10
            """,
        }
        for name, sql in queries.items():
            n, df, e = self._run(name, sql)
            result.tables[n] = df
            result.sql_blocks.append({"name": name, "sql": sql, "source": "pre-aggregation"})
            result.elapsed[n] = e
        result.summary = "本地兜底：已从销售、区域、配送、品类、支付、卖家六个维度调用预聚合表，生成整体运营分析。"
        return result
