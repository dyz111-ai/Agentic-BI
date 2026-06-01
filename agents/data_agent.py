from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import re
import pandas as pd
from sqlalchemy.engine import Engine

from utils.db import timed_read_df
from utils.llm import LLMClient
from config.data_dictionary import BASE_TABLES, PRE_AGG_TABLES


@dataclass
class DataResult:
    intent: str
    sql_blocks: list[dict] = field(default_factory=list)
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    summary: str = ""
    used_preaggregation: bool = False
    elapsed: dict[str, float] = field(default_factory=dict)
    routing_method: str = "llm"
    plan: dict[str, Any] = field(default_factory=dict)
    llm_error: str = ""


class DataAnalysisAgent:
    """LLM-only natural-language-to-SQL agent. No rule/template fallback."""

    ALLOWED_INTENTS = {
        "sales", "forecast", "delivery", "payment", "category",
        "seller", "review", "weight_freight", "overall", "custom_sql",
    }

    FORBIDDEN_SQL = re.compile(
        r"\b(insert|update|delete|drop|alter|truncate|create|replace|attach|detach|pragma|vacuum|grant|revoke)\b",
        re.IGNORECASE,
    )

    def __init__(self, engine: Engine):
        self.engine = engine
        self.llm = LLMClient()

    def analyze(self, question: str) -> DataResult:
        self.llm.require_enabled()
        return self._llm_analyze(question)

    def _llm_analyze(self, question: str) -> DataResult:
        plan = self._ask_llm_for_plan(question)
        intent = str(plan.get("intent", "custom_sql")).strip()
        if intent not in self.ALLOWED_INTENTS:
            intent = "custom_sql"
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
            name = str(q.get("name") or f"result_{i}").strip()
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
        result.summary = self._build_summary(result, plan)
        return result

    def _ask_llm_for_plan(self, question: str) -> dict:
        dialect = self.engine.dialect.name
        schema_text = self._schema_prompt()
        prompt = f"""
你是 Agentic BI 系统中的 Data Analysis Agent。把业务问题转换为可执行 SQL。

数据库方言：{dialect}

必须遵守：
1. 只生成只读 SELECT/WITH，禁止 INSERT/UPDATE/DELETE/DROP/ALTER/CREATE。
2. 优先使用预聚合表 mv_*；只有预聚合无法覆盖时才查基础表。
3. 直接返回严格 JSON，不要 Markdown，不要思考过程。
4. queries 最多 3 条，每条 SQL 尽量 LIMIT 5000。
5. 每条 query 的 name 用简短英文标识（如 monthly_trend、state_rank），不要重复。
6. intent 从以下选择：sales, forecast, delivery, payment, category, seller, review, weight_freight, overall, custom_sql。
7. used_preaggregation 表示是否主要命中预聚合表。
8. 品类翻译表列名是 product_category_name_english，不是 product_category_english（后者可在 SELECT 里 AS 别名）。
9. 对年份过滤可用 year_month LIKE '2017-%'。

数据字典：
{schema_text}

用户问题：{question}

返回 JSON：
{{
  "intent": "sales",
  "used_preaggregation": true,
  "reason": "为什么选择这些表和字段",
  "queries": [
    {{"name": "monthly_trend", "source": "pre-aggregation", "sql": "SELECT ..."}}
  ]
}}
"""
        messages = [
            {"role": "system", "content": "你是严谨的数据分析 SQL Agent，只返回合法 JSON。"},
            {"role": "user", "content": prompt},
        ]
        last_error = "LLM 未返回内容"
        for max_tokens in (4096, 8192):
            content = self.llm.chat(messages=messages, temperature=0.0, max_tokens=max_tokens)
            try:
                return LLMClient.extract_json(content)
            except ValueError as exc:
                last_error = str(exc)
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

    def _sanitize_sql(self, sql: str, source: str = "") -> str:
        s = sql.strip().rstrip(";").strip()
        if ";" in s:
            raise ValueError("SQL 包含多个语句，已拒绝执行")
        if self.FORBIDDEN_SQL.search(s):
            raise ValueError("SQL 包含危险关键字，已拒绝执行")
        if not re.match(r"^(select|with)\b", s, flags=re.IGNORECASE):
            raise ValueError("只允许 SELECT/WITH 查询")
        if "base" in source.lower() and not re.search(r"\blimit\b", s, flags=re.IGNORECASE):
            s += " LIMIT 5000"
        return s

    def _build_summary(self, result: DataResult, plan: dict) -> str:
        reason = (plan.get("reason") or "").strip()
        strategy = "本次优先命中预聚合表。" if result.used_preaggregation else "本次包含基础表查询。"
        parts = ["LLM 已完成问题理解、表选择与 SQL 生成。", strategy]
        if reason:
            parts.append(f"选择原因：{reason}")
        return " ".join(parts)

    def _run(self, name: str, sql: str, params: dict | None = None):
        df, elapsed = timed_read_df(sql, params=params, engine=self.engine)
        return name, df, elapsed
