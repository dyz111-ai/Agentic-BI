from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import json
import re

import pandas as pd
from sqlalchemy.engine import Engine

from agents.data_agent import DataAnalysisAgent, DataResult
from agents.visualization_agent import VisualizationAgent, VisualizationResult
from agents.nlp_agent import ReviewInsightAgent, NLPResult
from agents.forecasting_agent import ForecastAgent, ForecastResult
from agents.decision_agent import DecisionIntelligenceAgent, DecisionResult
from agents.what_if_agent import WhatIfAgent, WhatIfResult
from agents.anomaly_agent import AnomalyDetectionAgent, AnomalyResult
from utils.llm import LLMClient
from utils.conversation_memory import build_context_prompt, normalize_memory, extract_key_entities


@dataclass
class OrchestratorResult:
    question: str
    data_result: DataResult
    visualization_result: VisualizationResult
    decision_result: DecisionResult
    nlp_result: NLPResult | None = None
    forecast_result: ForecastResult | None = None
    what_if_result: WhatIfResult | None = None
    anomaly_result: AnomalyResult | None = None
    final_answer: str = ""
    direct_answer: list[str] = field(default_factory=list)
    findings: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    technical_details: dict[str, Any] = field(default_factory=dict)
    memory: dict[str, Any] = field(default_factory=dict)
    orchestration_plan: dict[str, Any] = field(default_factory=dict)


class OrchestratorAgent:
    """Coordinator Agent.

    新版本不再只依赖固定 intent 集合调度后续 Agent。

    调度依据按优先级综合判断：
    1. DataAgent 的 LLM plan，例如 needs_review_analysis / needs_forecast / required_agents。
    2. 用户问题语义，例如“差评、评论、预测、未来”。
    3. DataAgent 返回的数据表结构，例如是否含 review_score、review_comment_message、year_month、total_gmv。
    4. intent 仅作为兜底信号，不再作为唯一依据。
    """

    REVIEW_HINTS = (
        "评论", "评价", "差评", "好评", "review", "rating", "score",
        "原因", "complaint", "negative", "positive", "sentiment", "关键词", "主题"
    )
    FORECAST_HINTS = (
        "预测", "未来", "forecast", "predict", "prediction", "next", "后续",
        "6周", "六周", "趋势预测", "sales forecast"
    )

    WHAT_IF_HINTS = (
        "如果", "假设", "下架", "模拟", "what-if", "what if",
        "情景", "scenario", "假如下架", "移除", "会怎样", "提升多少",
    )

    ANOMALY_HINTS = (
        "异常", "骤降", "突升", "预警", "报警", "anomaly",
        "突然下降", "大幅下降", "突然上升", "最近怎么了", "有什么问题",
        "检查", "监控", "告警", "alert",
    )

    REVIEW_COLUMNS = {
        "review_score", "review_comment_title", "review_comment_message",
        "review_creation_date", "review_answer_timestamp", "sentiment_label"
    }
    MONTH_COLUMNS = {"year_month", "month", "date", "ds", "order_month"}
    SALES_COLUMNS = {"total_gmv", "gmv", "sales", "total_sales", "revenue", "total_value", "payment_value"}

    def __init__(self, engine: Engine):
        self.engine = engine
        self.data_agent = DataAnalysisAgent(engine)
        self.viz_agent = VisualizationAgent()
        self.nlp_agent = ReviewInsightAgent()
        self.forecast_agent = ForecastAgent()
        self.decision_agent = DecisionIntelligenceAgent()
        self.what_if_agent = WhatIfAgent(engine)
        self.anomaly_agent = AnomalyDetectionAgent(engine)
        self.llm = LLMClient()

    def handle(self, question: str, memory: dict[str, Any] | None = None) -> OrchestratorResult:
        memory = normalize_memory(memory)
        conversation_context = build_context_prompt(memory)

        data_result = self.data_agent.analyze(
            question,
            conversation_context=conversation_context,
            memory=memory,
        )

        plan = self._build_orchestration_plan(question, data_result)

        nlp_result = None
        if plan["needs_review_analysis"]:
            reviews_df = self._find_review_dataframe(data_result.tables)
            if reviews_df is not None and not reviews_df.empty:
                nlp_result = self.nlp_agent.analyze_reviews(reviews_df)
            else:
                plan["notes"].append("已判断需要评论洞察，但 DataAgent 本次没有返回可分析的评论明细表，因此跳过 NLP Agent。")

        forecast_result = None
        if plan["needs_forecast"]:
            monthly_df = self._find_monthly_sales_dataframe(data_result.tables)
            if monthly_df is not None and not monthly_df.empty:
                forecast_result = self.forecast_agent.forecast_sales(monthly_df)
            else:
                plan["notes"].append("已判断需要预测分析，但没有找到包含 year_month/date 与 GMV/sales 的历史序列表，因此跳过 Forecast Agent。")

        what_if_result = None
        if plan["needs_what_if"]:
            seller_df = self._find_seller_dataframe(data_result.tables)
            what_if_result = self.what_if_agent.simulate_remove_low_score_sellers(
                seller_data=seller_df,
                question=question,
                nlp_result=nlp_result,
            )
            if what_if_result.has_result:
                plan["notes"].append(
                    f"What-If 模拟：移除评分最低 {what_if_result.removed_seller_count} 个卖家，"
                    f"预计评分从 {what_if_result.current_avg_score:.4f} 提升至 {what_if_result.projected_avg_score:.4f}。"
                )
            else:
                plan["notes"].append(f"What-If 模拟未能执行：{what_if_result.summary[:100]}")

        anomaly_result = None
        if plan["needs_anomaly"]:
            anomaly_result = self.anomaly_agent.scan(question=question)
            if anomaly_result.has_alerts:
                plan["notes"].append(
                    f"异常扫描：发现 {anomaly_result.alert_count} 条异常（"
                    f"🚨{anomaly_result.critical_count} ⚠️{anomaly_result.warning_count} ℹ️{anomaly_result.info_count}）"
                )
            else:
                plan["notes"].append("异常扫描已完成，未检测到显著异常。")

        visualization_result = self.viz_agent.visualize(
            data_result=data_result,
            nlp_result=nlp_result,
            forecast_result=forecast_result,
            question=question,
            chart_plan=getattr(data_result, "chart_plan", None) or None,
        )
        decision_result = self.decision_agent.generate(
            question,
            data_result,
            nlp_result,
            forecast_result,
            what_if_result,
            conversation_context=conversation_context,
        )

        final_answer, direct_answer, findings, recommendations, technical_details = self._compose_answer(
            question=question,
            data_result=data_result,
            decision_result=decision_result,
            nlp_result=nlp_result,
            forecast_result=forecast_result,
            what_if_result=what_if_result,
            anomaly_result=anomaly_result,
            visualization_result=visualization_result,
            orchestration_plan=plan,
            conversation_context=conversation_context,
            memory=memory,
        )

        key_entities = extract_key_entities(data_result, nlp_result)
        updated_memory = dict(memory)
        updated_memory["last_question"] = question
        updated_memory["last_intent"] = data_result.intent
        updated_memory["last_tables"] = list(data_result.tables.keys())
        updated_memory["last_orchestration_plan"] = plan
        updated_memory["last_key_entities"] = key_entities

        return OrchestratorResult(
            question=question,
            data_result=data_result,
            visualization_result=visualization_result,
            decision_result=decision_result,
            nlp_result=nlp_result,
            forecast_result=forecast_result,
            what_if_result=what_if_result,
            anomaly_result=anomaly_result,
            final_answer=final_answer,
            direct_answer=direct_answer,
            findings=findings,
            recommendations=recommendations,
            technical_details=technical_details,
            memory=updated_memory,
            orchestration_plan=plan,
        )

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

    def _build_orchestration_plan(self, question: str, data_result: DataResult) -> dict[str, Any]:
        """Build a flexible execution plan from LLM plan + question + returned schema."""
        data_plan = getattr(data_result, "plan", {}) or {}
        required_agents = self._normalize_required_agents(data_plan)
        q = question.lower()

        table_schema = self._table_schema(data_result.tables)

        review_by_plan = self._truthy_any(data_plan, [
            "needs_review_analysis", "needs_nlp", "requires_review_analysis", "requires_nlp",
        ]) or any(a in required_agents for a in {"review", "nlp", "reviewinsightagent", "review_insight"})
        review_by_question = any(h.lower() in q for h in self.REVIEW_HINTS)
        review_by_data = any(self._has_review_columns(df) for df in data_result.tables.values())
        review_by_intent_fallback = getattr(data_result, "intent", "") in {"review", "overall"}

        forecast_by_plan = self._truthy_any(data_plan, [
            "needs_forecast", "requires_forecast", "needs_prediction", "requires_prediction",
        ]) or any(a in required_agents for a in {"forecast", "prediction", "forecastagent"})
        forecast_by_question = any(h.lower() in q for h in self.FORECAST_HINTS)
        forecast_by_data = self._find_monthly_sales_dataframe(data_result.tables) is not None
        forecast_by_intent_fallback = getattr(data_result, "intent", "") in {"forecast"}

        needs_review = bool(review_by_plan or review_by_question or review_by_data or review_by_intent_fallback)
        needs_forecast = bool(forecast_by_plan or forecast_by_question or forecast_by_intent_fallback)

        what_if_by_plan = self._truthy_any(data_plan, [
            "needs_what_if", "requires_what_if", "needs_simulation", "requires_simulation",
        ]) or any(a in required_agents for a in {"what_if", "whatif", "simulation", "whatifagent"})
        what_if_by_question = any(h.lower() in q for h in self.WHAT_IF_HINTS)
        what_if_by_data = self._find_seller_dataframe(data_result.tables) is not None
        what_if_by_intent_fallback = getattr(data_result, "intent", "") in {"seller"}
        needs_what_if = bool(what_if_by_plan or what_if_by_question or (what_if_by_data and what_if_by_intent_fallback))

        anomaly_by_plan = self._truthy_any(data_plan, [
            "needs_anomaly", "requires_anomaly", "needs_alert", "requires_alert",
            "needs_monitoring", "requires_monitoring",
        ]) or any(a in required_agents for a in {"anomaly", "alert", "monitoring", "anomalydetectionagent"})
        anomaly_by_question = any(h.lower() in q for h in self.ANOMALY_HINTS)
        needs_anomaly = bool(anomaly_by_plan or anomaly_by_question)

        # overall 问题通常需要多维综合；如果 DataAgent 已返回历史销售序列，也允许补充预测。
        if getattr(data_result, "intent", "") == "overall" and forecast_by_data:
            needs_forecast = True

        return {
            "required_agents_from_data_plan": sorted(required_agents),
            "needs_review_analysis": needs_review,
            "needs_forecast": needs_forecast,
            "needs_what_if": needs_what_if,
            "needs_anomaly": needs_anomaly,
            "needs_visualization": True,
            "needs_decision": True,
            "review_reason": self._reason_flags(
                plan=review_by_plan, question=review_by_question, data=review_by_data, intent=review_by_intent_fallback
            ),
            "forecast_reason": self._reason_flags(
                plan=forecast_by_plan, question=forecast_by_question, data=forecast_by_data, intent=forecast_by_intent_fallback
            ),
            "what_if_reason": self._reason_flags(
                plan=what_if_by_plan, question=what_if_by_question, data=what_if_by_data, intent=what_if_by_intent_fallback
            ),
            "anomaly_reason": self._reason_flags(
                plan=anomaly_by_plan, question=anomaly_by_question
            ),
            "returned_tables": list(data_result.tables.keys()),
            "returned_schema": table_schema,
            "notes": [],
        }

    def _normalize_required_agents(self, plan: dict[str, Any]) -> set[str]:
        agents = plan.get("required_agents") or plan.get("agents") or plan.get("next_agents") or []
        if isinstance(agents, str):
            agents = re.split(r"[,，/|;；\s]+", agents)
        if not isinstance(agents, (list, tuple, set)):
            return set()
        return {str(a).strip().lower() for a in agents if str(a).strip()}

    def _truthy_any(self, plan: dict[str, Any], keys: list[str]) -> bool:
        for k in keys:
            if k in plan:
                v = plan.get(k)
                if isinstance(v, str):
                    if v.strip().lower() in {"true", "yes", "1", "需要", "是"}:
                        return True
                elif bool(v):
                    return True
        return False

    def _reason_flags(self, **kwargs: bool) -> list[str]:
        return [k for k, v in kwargs.items() if v]

    # ------------------------------------------------------------------
    # DataFrame discovery helpers
    # ------------------------------------------------------------------

    def _table_schema(self, tables: dict[str, pd.DataFrame]) -> dict[str, list[str]]:
        return {name: list(df.columns) for name, df in tables.items() if isinstance(df, pd.DataFrame)}

    def _has_review_columns(self, df: pd.DataFrame) -> bool:
        cols = {str(c).lower() for c in df.columns}
        if cols & self.REVIEW_COLUMNS:
            return True
        return any("review" in c or "comment" in c for c in cols)

    def _find_review_dataframe(self, tables: dict[str, pd.DataFrame]) -> pd.DataFrame | None:
        if "reviews" in tables and isinstance(tables["reviews"], pd.DataFrame):
            return tables["reviews"]
        candidates: list[tuple[int, str, pd.DataFrame]] = []
        for name, df in tables.items():
            if not isinstance(df, pd.DataFrame) or df.empty:
                continue
            cols = {str(c).lower() for c in df.columns}
            score = 0
            if "review_score" in cols:
                score += 3
            if "review_comment_message" in cols:
                score += 3
            if "review_comment_title" in cols:
                score += 2
            if any("review" in c or "comment" in c for c in cols):
                score += 1
            if score > 0:
                candidates.append((score, name, df))
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][2]

    def _find_monthly_sales_dataframe(self, tables: dict[str, pd.DataFrame]) -> pd.DataFrame | None:
        if "monthly_sales" in tables and isinstance(tables["monthly_sales"], pd.DataFrame):
            df = self._normalize_monthly_sales(tables["monthly_sales"])
            if df is not None:
                return df

        best: tuple[int, str, pd.DataFrame] | None = None
        for name, df in tables.items():
            if not isinstance(df, pd.DataFrame) or df.empty:
                continue
            norm = self._normalize_monthly_sales(df)
            if norm is None:
                continue
            cols = {str(c).lower() for c in df.columns}
            score = 0
            if cols & self.MONTH_COLUMNS:
                score += 2
            if cols & self.SALES_COLUMNS:
                score += 3
            if any("month" in c or "date" in c or "timestamp" in c for c in cols):
                score += 1
            if best is None or score > best[0]:
                best = (score, name, norm)
        return best[2] if best else None

    def _find_seller_dataframe(self, tables: dict[str, pd.DataFrame]) -> pd.DataFrame | None:
        if "low_score_sellers" in tables and isinstance(tables["low_score_sellers"], pd.DataFrame):
            return tables["low_score_sellers"]
        if "seller_perf" in tables and isinstance(tables["seller_perf"], pd.DataFrame):
            return tables["seller_perf"]
        candidates: list[tuple[int, str, pd.DataFrame]] = []
        for name, df in tables.items():
            if not isinstance(df, pd.DataFrame) or df.empty:
                continue
            cols = {str(c).lower() for c in df.columns}
            score = 0
            has_score = "avg_review_score" in cols or "review_score" in cols
            has_seller = "seller_id" in cols
            has_orders = "total_orders" in cols or "order_count" in cols
            if not (has_score and has_orders):
                continue
            if has_score:
                score += 3
            if has_seller:
                score += 2
            if has_orders:
                score += 2
            if any("seller" in c or "score" in c for c in cols):
                score += 1
            if score > 4:
                candidates.append((score, name, df))
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][2]

    def _normalize_monthly_sales(self, df: pd.DataFrame) -> pd.DataFrame | None:
        """Return df with year_month and total_gmv columns for ForecastAgent."""
        if df is None or df.empty:
            return None
        work = df.copy()
        lower_map = {str(c).lower(): c for c in work.columns}

        time_col = None
        for c in ["year_month", "month", "date", "ds", "order_month", "order_purchase_timestamp"]:
            if c in lower_map:
                time_col = lower_map[c]
                break
        if time_col is None:
            for c in work.columns:
                cl = str(c).lower()
                if "month" in cl or "date" in cl or "timestamp" in cl:
                    time_col = c
                    break

        value_col = None
        for c in ["total_gmv", "gmv", "sales", "total_sales", "revenue", "total_value", "payment_value"]:
            if c in lower_map:
                value_col = lower_map[c]
                break
        if value_col is None:
            numeric_cols = list(work.select_dtypes(include="number").columns)
            if numeric_cols:
                value_col = numeric_cols[0]

        if time_col is None or value_col is None:
            return None

        work["_date"] = pd.to_datetime(work[time_col].astype(str), errors="coerce")
        # year_month like 2017-01 may parse correctly; if not, append -01.
        if work["_date"].isna().mean() > 0.5:
            work["_date"] = pd.to_datetime(work[time_col].astype(str).str[:7] + "-01", errors="coerce")
        work = work.dropna(subset=["_date"])
        if work.empty:
            return None
        work["year_month"] = work["_date"].dt.strftime("%Y-%m")
        work["total_gmv"] = pd.to_numeric(work[value_col], errors="coerce").fillna(0)
        out = work.groupby("year_month", as_index=False)["total_gmv"].sum().sort_values("year_month")
        return out

    # ------------------------------------------------------------------
    # Final answer
    # ------------------------------------------------------------------

    REASON_LABELS = {
        "plan": "LLM 规划",
        "question": "问题语义",
        "data": "数据结构",
        "intent": "意图兜底",
    }

    BOILERPLATE_PREFIXES = (
        "LLM 已完成问题理解、表选择与 SQL 生成。",
        "本地兜底：",
        "已优先使用",
        "本次优先命中预聚合表。",
        "本次包含基础表回退查询。",
        "评论文本与差评原因不完全在预聚合表中，已回退查询评论与商品品类原始表。",
    )

    ERROR_HINTS = (
        "operationalerror", "sqlalchemy", "no such column", "llm 路由", "生成失败",
        "已切换到本地兜底", "[sql:", "background on this error",
    )

    def _compose_answer(
        self,
        question: str,
        data_result: DataResult,
        decision_result: DecisionResult,
        nlp_result: NLPResult | None,
        forecast_result: ForecastResult | None,
        what_if_result: WhatIfResult | None,
        anomaly_result: AnomalyResult | None,
        visualization_result: VisualizationResult,
        orchestration_plan: dict[str, Any],
        conversation_context: str = "",
        memory: dict[str, Any] | None = None,
    ) -> tuple[str, list[str], list[str], list[str], dict[str, Any]]:
        direct_answer, findings = self._synthesize_narrative(
            question=question,
            data_result=data_result,
            nlp_result=nlp_result,
            forecast_result=forecast_result,
            what_if_result=what_if_result,
            anomaly_result=anomaly_result,
            conversation_context=conversation_context,
        )
        recommendations = [r.strip() for r in decision_result.recommendations if r and r.strip()]
        recommendations = self._filter_user_facing_lines(recommendations)
        if not recommendations:
            recommendations = ["建议结合上方图表与数据明细，制定下一步运营动作。"]

        technical_details = self._build_technical_details(
            data_result=data_result,
            nlp_result=nlp_result,
            forecast_result=forecast_result,
            what_if_result=what_if_result,
            anomaly_result=anomaly_result,
            visualization_result=visualization_result,
            orchestration_plan=orchestration_plan,
            memory=memory,
        )

        lines: list[str] = []
        lines.append("### 直接回答")
        for item in direct_answer:
            lines.append(f"- {item}")
        lines.append("\n### 关键发现")
        for item in findings:
            lines.append(f"- {item}")
        lines.append("\n### 决策建议")
        for rec in recommendations:
            lines.append(f"- {rec}")

        if what_if_result is not None and what_if_result.has_result:
            lines.append("\n### What-If 模拟分析")
            lines.append(what_if_result.summary)
            if what_if_result.llm_analysis:
                lines.append(what_if_result.llm_analysis)

        if anomaly_result is not None and anomaly_result.has_alerts:
            lines.append("\n### 异常预警")
            lines.append(anomaly_result.summary)

        return "\n".join(lines), direct_answer, findings, recommendations, technical_details

    def _synthesize_narrative(
        self,
        question: str,
        data_result: DataResult,
        nlp_result: NLPResult | None,
        forecast_result: ForecastResult | None,
        what_if_result: WhatIfResult | None = None,
        anomaly_result: AnomalyResult | None = None,
        conversation_context: str = "",
    ) -> tuple[list[str], list[str]]:
        """LLM 合成直接回答与关键发现；失败或数据为空时用结构化兜底，禁止模板空话。"""
        if self.llm.enabled:
            try:
                direct, findings = self._llm_synthesize_narrative(
                    question, data_result, nlp_result, forecast_result,
                    what_if_result, anomaly_result,
                    conversation_context=conversation_context,
                )
                if direct or findings:
                    direct, findings = self._merge_and_sanitize_narrative(
                        question, data_result, nlp_result, forecast_result, direct, findings,
                    )
                    return direct, findings
            except Exception:
                pass

        direct = self._synthesize_direct_answer(question, data_result, nlp_result, forecast_result)
        findings = self._collect_insights(question, data_result, nlp_result, forecast_result, direct_answer=direct)
        return direct, findings

    def _build_narrative_evidence(
        self,
        data_result: DataResult,
        nlp_result: NLPResult | None,
        forecast_result: ForecastResult | None,
        what_if_result: WhatIfResult | None = None,
        anomaly_result: AnomalyResult | None = None,
    ) -> dict[str, Any]:
        evidence: dict[str, Any] = {
            "question_intent": getattr(data_result, "intent", ""),
            "used_preaggregation": bool(getattr(data_result, "used_preaggregation", False)),
            "routing_method": getattr(data_result, "routing_method", ""),
            "tables": {},
        }
        tables = data_result.tables or {}
        for name, df in tables.items():
            if isinstance(df, pd.DataFrame):
                preview = df.head(15).copy()
                for col in preview.columns:
                    if pd.api.types.is_float_dtype(preview[col]):
                        preview[col] = preview[col].round(2)
                evidence["tables"][name] = {
                    "row_count": int(len(df)),
                    "columns": list(df.columns),
                    "preview": preview.where(pd.notnull(preview), None).to_dict(orient="records"),
                }
        if nlp_result and getattr(nlp_result, "summary", ""):
            evidence["nlp_summary"] = nlp_result.summary
        if forecast_result and getattr(forecast_result, "summary", ""):
            evidence["forecast_summary"] = forecast_result.summary
        if what_if_result is not None and what_if_result.has_result:
            evidence["what_if_summary"] = what_if_result.summary
        if anomaly_result is not None and anomaly_result.has_alerts:
            evidence["anomaly_summary"] = anomaly_result.summary
        return evidence

    def _has_usable_sales_data(self, data_result: DataResult) -> bool:
        tables = data_result.tables or {}
        monthly = self._find_monthly_sales_dataframe(tables)
        if monthly is not None and not monthly.empty:
            if pd.to_numeric(monthly["total_gmv"], errors="coerce").fillna(0).sum() > 0:
                return True
        state = self._get_state_ranking_df(tables)
        if state is not None and not state.empty:
            if pd.to_numeric(state["total_gmv"], errors="coerce").fillna(0).sum() > 0:
                return True
        for df in tables.values():
            if not isinstance(df, pd.DataFrame) or df.empty:
                continue
            for col in df.columns:
                if str(col).lower() in {"total_gmv", "gmv", "sales", "revenue", "total_value", "payment_value"}:
                    if pd.to_numeric(df[col], errors="coerce").fillna(0).sum() > 0:
                        return True
        return False

    def _has_usable_data(self, data_result: DataResult) -> bool:
        """Check whether data_result contains any non-empty table.

        Replaces narrow GMV-only check with intent-agnostic check so
        weight_freight / review / delivery / payment tables are recognised.
        """
        tables = data_result.tables or {}
        for df in tables.values():
            if not isinstance(df, pd.DataFrame) or df.empty:
                continue
            if len(df) > 0:
                return True
        return False

    def _empty_data_direct_answer(self, question: str, data_result: DataResult) -> list[str]:
        q = question.lower()
        asks_sales = any(k in q for k in ["gmv", "销售", "营收", "2017", "趋势", "州", "排名"])
        if asks_sales:
            return [
                "本次查询未返回有效的 GMV 或销售明细数据。",
                "请确认：① MySQL 是否已导入 Olist 订单 CSV；② 侧边栏是否已点击「刷新预聚合表」；③ 数据库连接是否指向含数据的环境。",
            ]
        if getattr(data_result, "llm_error", "") and not self._has_usable_data(data_result):
            return [f"数据查询未完成：{data_result.llm_error[:200]}"]
        return ["本次查询未返回可用数据，请检查数据库连接与预聚合表是否已刷新。"]

    def _llm_synthesize_narrative(
        self,
        question: str,
        data_result: DataResult,
        nlp_result: NLPResult | None,
        forecast_result: ForecastResult | None,
        what_if_result: WhatIfResult | None = None,
        anomaly_result: AnomalyResult | None = None,
        conversation_context: str = "",
    ) -> tuple[list[str], list[str]]:
        evidence = self._build_narrative_evidence(
            data_result, nlp_result, forecast_result, what_if_result, anomaly_result,
        )
        has_data = self._has_usable_data(data_result)

        if not has_data:
            return self._empty_data_direct_answer(question, data_result), []

        evidence_text = json.dumps(evidence, ensure_ascii=False, default=str, indent=2)
        if conversation_context:
            question_block = f"{conversation_context.rstrip()}\n{question}"
        else:
            question_block = f"用户问题：{question}"
        prompt = f"""
{question_block}

以下是系统真实查询得到的结构化数据（仅供你分析，不要在回答中出现 JSON、evidence、字段名等技术词汇）：

{evidence_text}

请输出严格 JSON，格式如下：
{{
  "direct_answer": ["用 1-3 条完整中文句子直接回答用户问题，必须引用上面的真实数字"],
  "findings": ["用 1-4 条完整中文句子写出关键发现，必须基于真实数据，禁止空泛套话"]
}}

硬性规则：
1. 必须引用查询结果中的具体数字（GMV、州名、月份、评分等）。
2. 禁止输出「已根据查询结果生成图表」「建议结合图表继续分析」等空话。
3. 禁止出现 evidence_json、JSON、SQL、预聚合表名、ETL 等内部术语。
4. 只回答有数据支撑的部分；若某维度不在查询结果中，直接省略该维度，禁止写「未返回」「无法分析」「缺少数据」等表述。
5. 若提供了【会话上下文】，请结合上一轮问题与结论理解当前追问（如「那个州呢」），回答须与上下文连贯。
6. 只返回 JSON，不要 Markdown。
"""
        content = self.llm.chat(
            [
                {"role": "system", "content": "你是电商 BI 分析助手。只返回合法 JSON。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=1200,
        )
        parsed = LLMClient.extract_json(content)
        direct = [str(x).strip() for x in (parsed.get("direct_answer") or []) if str(x).strip()]
        findings = [str(x).strip() for x in (parsed.get("findings") or []) if str(x).strip()]
        direct = self._filter_boilerplate(direct)
        findings = self._filter_boilerplate(findings)
        return self._dedupe_lines(direct)[:4], self._dedupe_lines(findings)[:5]

    def _filter_user_facing_lines(self, lines: list[str]) -> list[str]:
        out: list[str] = []
        for line in lines:
            text = (line or "").strip()
            if not text:
                continue
            if any(p in text for p in self.UNAVAILABLE_PHRASES):
                continue
            out.append(text)
        return out

    def _question_asks_state(self, question: str) -> bool:
        q = question.lower()
        return any(k in q for k in ["各州", "州", "排名", "state", "region", "区域"])

    def _merge_and_sanitize_narrative(
        self,
        question: str,
        data_result: DataResult,
        nlp_result: NLPResult | None,
        forecast_result: ForecastResult | None,
        direct: list[str],
        findings: list[str],
    ) -> tuple[list[str], list[str]]:
        """Strip unavailable-data boilerplate; backfill from rule-based synthesis when data exists."""
        direct = self._filter_user_facing_lines(direct)
        findings = self._filter_user_facing_lines(findings)

        rule_direct = self._synthesize_direct_answer(question, data_result, nlp_result, forecast_result)
        rule_findings = self._collect_insights(
            question, data_result, nlp_result, forecast_result, direct_answer=rule_direct,
        )

        blob = " ".join(direct + findings)
        if self._question_asks_state(question):
            state_df = self._get_state_ranking_df(data_result.tables or {})
            if state_df is not None and not state_df.empty:
                has_state_answer = any(
                    kw in blob for kw in ("位居", "排名第一", "分列", "Top", "最高", "GMV 为")
                ) and "州" in blob
                if not has_state_answer:
                    for line in rule_direct:
                        if "州" in line and line not in direct:
                            direct.append(line)

        if not direct and rule_direct:
            direct = rule_direct[:4]
        if not findings and rule_findings:
            findings = rule_findings[:5]

        return self._dedupe_lines(direct)[:4], self._dedupe_lines(findings)[:5]

    UNAVAILABLE_PHRASES = (
        "当前查询未返回",
        "未返回该",
        "未返回任何",
        "无法提供",
        "无法分析",
        "无法基于",
        "不包含",
        "缺少",
        "请要求数据团队",
        "请确认订单",
        "数据团队补充",
        "无法进行区域",
        "无法制定",
        "在此期间，建议手动",
    )

    BOILERPLATE_PHRASES = (
        "已根据查询结果生成图表",
        "请结合下方明细查看",
        "建议结合图表继续",
        "当前结果以描述性统计为主",
        "evidence_json",
    )

    def _filter_boilerplate(self, lines: list[str]) -> list[str]:
        out: list[str] = []
        for line in lines:
            if any(p in line for p in self.BOILERPLATE_PHRASES):
                continue
            out.append(line)
        return out

    def _synthesize_direct_answer(
        self,
        question: str,
        data_result: DataResult,
        nlp_result: NLPResult | None,
        forecast_result: ForecastResult | None,
    ) -> list[str]:
        q = question.lower()
        lines: list[str] = []
        tables = data_result.tables or {}
        year = "2017" if "2017" in q else None

        monthly = self._find_monthly_sales_dataframe(tables)
        if monthly is not None and not monthly.empty:
            mdf = monthly.copy()
            mdf["year_month"] = mdf["year_month"].astype(str)
            if year:
                mdf = mdf[mdf["year_month"].str.startswith(year)]
            if not mdf.empty:
                asks_gmv = any(k in q for k in ["gmv", "销售额", "销售", "营收", "revenue", "amount", "多少"])
                asks_trend = any(k in q for k in ["按月", "月度", "每月", "趋势", "monthly", "trend", "怎样", "如何"])
                if asks_gmv or year:
                    label = f"{year} 年" if year else "统计期内"
                    lines.append(f"{label}总 GMV 为 {mdf['total_gmv'].sum():,.2f}。")
                if asks_trend or asks_gmv:
                    mdf = mdf.sort_values("year_month")
                    first, last = mdf.iloc[0], mdf.iloc[-1]
                    peak = mdf.loc[mdf["total_gmv"].idxmax()]
                    trough = mdf.loc[mdf["total_gmv"].idxmin()]
                    direction = "上升" if last["total_gmv"] > first["total_gmv"] else ("下降" if last["total_gmv"] < first["total_gmv"] else "平稳")
                    lines.append(
                        f"按月趋势整体{direction}：{first['year_month']} GMV 为 {float(first['total_gmv']):,.2f}，"
                        f"{last['year_month']} 为 {float(last['total_gmv']):,.2f}；"
                        f"最高月份为 {peak['year_month']}（{float(peak['total_gmv']):,.2f}），"
                        f"最低月份为 {trough['year_month']}（{float(trough['total_gmv']):,.2f}）。"
                    )

        state_df = self._get_state_ranking_df(tables)
        if state_df is not None and any(k in q for k in ["州", "排名", "state", "region", "各州", "区域"]):
            top = state_df.head(5)
            if not top.empty:
                r0 = top.iloc[0]
                msg = f"各州排名中，{r0['customer_state']} 位居第一，GMV 为 {float(r0['total_gmv']):,.2f}"
                if len(top) >= 3:
                    msg += f"；{top.iloc[1]['customer_state']} 和 {top.iloc[2]['customer_state']} 分列第二、第三"
                elif len(top) >= 2:
                    msg += f"；{top.iloc[1]['customer_state']} 位居第二"
                msg += "。"
                lines.append(msg)

        if "payment_dist" in tables and not tables["payment_dist"].empty and any(k in q for k in ["支付", "payment", "分期"]):
            row = tables["payment_dist"].sort_values("total_transactions", ascending=False).iloc[0]
            inst = row.get("avg_installments")
            extra = f"，平均分期约 {float(inst):.2f}" if inst is not None and pd.notna(inst) else ""
            lines.append(f"最受欢迎的支付方式是 {row.get('payment_type')}，交易数 {int(row.get('total_transactions', 0))}{extra}。")

        if "delivery_by_state" in tables and not tables["delivery_by_state"].empty and any(k in q for k in ["配送", "交付", "准时", "延迟", "物流"]):
            slow = tables["delivery_by_state"].sort_values("avg_delivery_days", ascending=False).iloc[0]
            rate = slow.get("on_time_rate")
            rate_txt = f"，准时率约 {float(rate):.1%}" if rate is not None and pd.notna(rate) else ""
            lines.append(f"配送最慢的州为 {slow.get('customer_state')}，平均配送约 {float(slow.get('avg_delivery_days', 0)):.2f} 天{rate_txt}。")

        if "top_categories" in tables and not tables["top_categories"].empty and any(k in q for k in ["品类", "类别", "category"]):
            row = tables["top_categories"].sort_values("total_gmv", ascending=False).iloc[0]
            lines.append(f"销售额最高的品类是 {row.get('product_category_english')}，GMV 约 {float(row.get('total_gmv', 0)):,.2f}。")

        if nlp_result and nlp_result.summary and any(k in q for k in self.REVIEW_HINTS):
            lines.append(nlp_result.summary.strip())

        if forecast_result and forecast_result.summary and any(k in q for k in self.FORECAST_HINTS):
            lines.append(forecast_result.summary.strip())

        if not lines:
            if not self._has_usable_data(data_result):
                lines = self._empty_data_direct_answer(question, data_result)
            else:
                fallback = self._fallback_direct_from_summary(data_result.summary)
                lines = fallback if fallback else []
        return self._dedupe_lines(lines)[:6]

    def _collect_insights(
        self,
        question: str,
        data_result: DataResult,
        nlp_result: NLPResult | None,
        forecast_result: ForecastResult | None,
        direct_answer: list[str],
    ) -> list[str]:
        insights: list[str] = []
        tables = data_result.tables or {}
        direct_blob = " ".join(direct_answer)

        monthly = self._find_monthly_sales_dataframe(tables)
        if monthly is not None and not monthly.empty:
            mdf = monthly.copy()
            mdf["year_month"] = mdf["year_month"].astype(str)
            if "2017" in question:
                mdf = mdf[mdf["year_month"].str.startswith("2017")]
            if len(mdf) >= 4:
                mdf = mdf.sort_values("year_month")
                months = mdf["year_month"].str[5:7].astype(int)
                h1 = mdf.loc[months <= 6, "total_gmv"].sum()
                h2 = mdf.loc[months > 6, "total_gmv"].sum()
                if h2 > h1 * 1.15:
                    insights.append("GMV 在下半年增长明显，说明销售在 Q3-Q4 进入高峰。")
                elif h1 > h2 * 1.15:
                    insights.append("GMV 上半年高于下半年，存在明显的季节性回落。")

            state_df = self._get_state_ranking_df(tables)
            if state_df is not None and not state_df.empty:
                total = float(state_df["total_gmv"].sum())
                if total > 0:
                    top3_share = float(state_df.head(3)["total_gmv"].sum()) / total
                    if top3_share >= 0.5:
                        insights.append(f"区域销售集中度较高，Top 3 州合计贡献约 {top3_share:.1%} GMV。")

        if nlp_result is not None:
            top_neg = getattr(nlp_result, "top_negative_categories", pd.DataFrame())
            if isinstance(top_neg, pd.DataFrame) and not top_neg.empty:
                row = top_neg.iloc[0]
                try:
                    if row.get("total_reviews") is not None and float(row.get("total_reviews")) < 20:
                        insights.append("部分差评品类评论样本量偏小，结论更适合作为风险预警而非最终定性。")
                except Exception:
                    pass
            kws = getattr(nlp_result, "negative_keywords", None) or []
            if kws and "关键词" not in direct_blob:
                kw_text = "、".join([str(w) for w, _ in kws[:5]])
                insights.append(f"差评文本高频词包括：{kw_text}，可优先排查对应体验环节。")

        if forecast_result and forecast_result.summary and forecast_result.summary not in direct_blob:
            insights.append(forecast_result.summary)

        if not insights:
            if not self._has_usable_data(data_result):
                return []
            insights.append("可从区域、品类、时间三个维度对比核心 KPI，识别增长或风险点。")
        return self._dedupe_lines(insights)[:5]

    def _get_state_ranking_df(self, tables: dict[str, pd.DataFrame]) -> pd.DataFrame | None:
        for key in ("state_sales_2017", "state_sales"):
            df = tables.get(key)
            if not isinstance(df, pd.DataFrame) or df.empty:
                continue
            if not {"customer_state", "total_gmv"}.issubset(df.columns):
                continue
            work = df.copy()
            if "year_month" in work.columns and work["customer_state"].nunique() < len(work):
                work = work.groupby("customer_state", as_index=False)["total_gmv"].sum()
            return work.sort_values("total_gmv", ascending=False)
        return None

    def _build_technical_details(
        self,
        data_result: DataResult,
        nlp_result: NLPResult | None,
        forecast_result: ForecastResult | None,
        what_if_result: WhatIfResult | None,
        anomaly_result: AnomalyResult | None,
        visualization_result: VisualizationResult,
        orchestration_plan: dict[str, Any],
        memory: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        routing = getattr(data_result, "routing_method", "rule")
        if routing == "llm":
            query_mode = "大模型生成 SQL"
        elif routing == "fallback":
            query_mode = "大模型失败，已切换本地兜底"
        else:
            query_mode = "本地规则兜底（未配置 LLM Key）"

        memory = normalize_memory(memory)
        prior_turns = len(memory.get("turns") or [])

        return {
            "query_mode": query_mode,
            "data_source": "预聚合表 mv_*" if data_result.used_preaggregation else "基础表 / 自定义 SQL",
            "intent": getattr(data_result, "intent", ""),
            "chart_count": len(visualization_result.figures),
            "chart_mode": "LLM 图表规划" if (getattr(data_result, "chart_plan", None) or None) else "规则推断",
            "llm_error": getattr(data_result, "llm_error", "") or "",
            "returned_tables": orchestration_plan.get("returned_tables", []),
            "orchestrator": "OrchestratorAgent",
            "conversation_turns": prior_turns + 1,
            "multi_turn_enabled": True,
            "agents": {
                "评论洞察": {"called": nlp_result is not None, "reason": self._humanize_reasons(orchestration_plan.get("review_reason", []))},
                "销售预测": {"called": forecast_result is not None, "reason": self._humanize_reasons(orchestration_plan.get("forecast_reason", []))},
                "What-If 模拟": {"called": what_if_result is not None and what_if_result.has_result, "reason": self._humanize_reasons(orchestration_plan.get("what_if_reason", []))},
                "异常检测": {"called": anomaly_result is not None and anomaly_result.has_alerts, "reason": self._humanize_reasons(orchestration_plan.get("anomaly_reason", []))},
                "可视化": {"called": True, "reason": "按返回数据自动选图"},
                "决策智能": {"called": True, "reason": "综合各 Agent 输出"},
            },
            "notes": orchestration_plan.get("notes", []),
        }

    def _humanize_reasons(self, flags: list[str]) -> str:
        if not flags:
            return "—"
        return "、".join(self.REASON_LABELS.get(str(f), str(f)) for f in flags)

    @staticmethod
    def _dedupe_lines(lines: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for line in lines:
            cleaned = re.sub(r"\s+", " ", line.strip())
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                out.append(cleaned)
        return out

    def _fallback_direct_from_summary(self, summary: str) -> list[str]:
        if not summary:
            return []
        text = summary.strip()
        for prefix in self.BOILERPLATE_PREFIXES:
            text = text.replace(prefix, "")
        text = re.sub(r"选择原因：[^。]+。", "", text)
        parts = [self._clean_finding(p) for p in re.split(r"(?<=[。！？])\s*", text) if len(p.strip()) >= 8]
        return [p for p in parts if p and not self._looks_like_error(p)][:3]

    @staticmethod
    def _clean_finding(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().strip("-•* "))

    @classmethod
    def _looks_like_error(cls, text: str) -> bool:
        lower = text.lower()
        return any(h in lower for h in cls.ERROR_HINTS)
