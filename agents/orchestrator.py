from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any
import re

import pandas as pd
from sqlalchemy.engine import Engine

from agents.data_agent import DataAnalysisAgent, DataResult
from agents.visualization_agent import VisualizationAgent, VisualizationResult
from agents.nlp_agent import ReviewInsightAgent, NLPResult
from agents.forecasting_agent import ForecastAgent, ForecastResult
from agents.decision_agent import DecisionIntelligenceAgent, DecisionResult
from agents.langgraph_flow import build_bi_graph, invoke_bi_graph
from utils.llm import LLMClient
from utils.conversation_memory import append_turn, build_context_prompt, normalize_memory


@dataclass
class OrchestratorResult:
    question: str
    data_result: DataResult
    visualization_result: VisualizationResult
    decision_result: DecisionResult
    nlp_result: NLPResult | None = None
    forecast_result: ForecastResult | None = None
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
        self.llm = LLMClient()
        self.memory: dict[str, Any] = normalize_memory(None)
        self._graph = build_bi_graph(self)

    def handle(self, question: str, memory: dict[str, Any] | None = None, thread_id: str = "default") -> OrchestratorResult:
        """Run one analysis turn via LangGraph with cross-turn conversation memory."""
        self.memory = normalize_memory(memory if memory is not None else self.memory)
        conversation_context = build_context_prompt(self.memory)

        final_state = invoke_bi_graph(
            self._graph,
            question=question,
            conversation_context=conversation_context,
            memory=self.memory,
            thread_id=thread_id,
        )

        self.memory = final_state.get("memory") or self.memory

        return OrchestratorResult(
            question=question,
            data_result=final_state["data_result"],
            visualization_result=final_state["visualization_result"],
            decision_result=final_state["decision_result"],
            nlp_result=final_state.get("nlp_result"),
            forecast_result=final_state.get("forecast_result"),
            final_answer=final_state.get("final_answer", ""),
            direct_answer=final_state.get("direct_answer") or [],
            findings=final_state.get("findings") or [],
            recommendations=final_state.get("recommendations") or [],
            technical_details=final_state.get("technical_details") or {},
            memory=self.memory.copy(),
            orchestration_plan=final_state.get("orchestration_plan") or {},
        )

    # ------------------------------------------------------------------
    # LangGraph nodes
    # ------------------------------------------------------------------

    def _graph_node_data(self, state: dict[str, Any]) -> dict[str, Any]:
        ctx = state.get("conversation_context") or ""
        data_result = self.data_agent.analyze(state["question"], conversation_context=ctx)
        return {"data_result": data_result}

    def _graph_node_plan(self, state: dict[str, Any]) -> dict[str, Any]:
        plan = self._build_orchestration_plan(state["question"], state["data_result"])
        return {"orchestration_plan": plan}

    def _graph_node_nlp(self, state: dict[str, Any]) -> dict[str, Any]:
        plan = state["orchestration_plan"]
        data_result = state["data_result"]
        nlp_result = None
        if plan.get("needs_review_analysis"):
            reviews_df = self._find_review_dataframe(data_result.tables)
            if reviews_df is not None and not reviews_df.empty:
                nlp_result = self.nlp_agent.analyze_reviews(reviews_df)
            else:
                plan["notes"].append("已判断需要评论洞察，但 DataAgent 本次没有返回可分析的评论明细表，因此跳过 NLP Agent。")
        return {"nlp_result": nlp_result, "orchestration_plan": plan}

    def _graph_node_forecast(self, state: dict[str, Any]) -> dict[str, Any]:
        plan = state["orchestration_plan"]
        data_result = state["data_result"]
        forecast_result = None
        if plan.get("needs_forecast"):
            monthly_df = self._find_monthly_sales_dataframe(data_result.tables)
            if monthly_df is not None and not monthly_df.empty:
                forecast_result = self.forecast_agent.forecast_sales(monthly_df)
            else:
                plan["notes"].append("已判断需要预测分析，但没有找到包含 year_month/date 与 GMV/sales 的历史序列表，因此跳过 Forecast Agent。")
        return {"forecast_result": forecast_result, "orchestration_plan": plan}

    def _graph_node_visualize(self, state: dict[str, Any]) -> dict[str, Any]:
        visualization_result = self.viz_agent.visualize(
            data_result=state["data_result"],
            nlp_result=state.get("nlp_result"),
            forecast_result=state.get("forecast_result"),
            question=state["question"],
            conversation_context=state.get("conversation_context") or "",
        )
        return {"visualization_result": visualization_result}

    def _graph_node_decision(self, state: dict[str, Any]) -> dict[str, Any]:
        decision_result = self.decision_agent.generate(
            state["question"],
            state["data_result"],
            state.get("nlp_result"),
            state.get("forecast_result"),
            conversation_context=state.get("conversation_context") or "",
        )
        return {"decision_result": decision_result}

    def _graph_node_compose(self, state: dict[str, Any]) -> dict[str, Any]:
        final_answer, direct_answer, findings, recommendations, technical_details = self._compose_answer(
            question=state["question"],
            data_result=state["data_result"],
            decision_result=state["decision_result"],
            nlp_result=state.get("nlp_result"),
            forecast_result=state.get("forecast_result"),
            visualization_result=state["visualization_result"],
            orchestration_plan=state["orchestration_plan"],
            conversation_context=state.get("conversation_context") or "",
        )
        memory = append_turn(
            state.get("memory") or {},
            question=state["question"],
            intent=getattr(state["data_result"], "intent", ""),
            tables=list(state["data_result"].tables.keys()),
            direct_answer=direct_answer,
            findings=findings,
            recommendations=recommendations,
            key_entities=self._extract_key_entities(state["data_result"]),
        )
        technical_details["orchestrator"] = "LangGraph StateGraph（会话记忆：conversation_memory）"
        technical_details["thread_id"] = state.get("thread_id", "default")
        technical_details["conversation_turns"] = len(memory.get("turns") or [])
        return {
            "final_answer": final_answer,
            "direct_answer": direct_answer,
            "findings": findings,
            "recommendations": recommendations,
            "technical_details": technical_details,
            "memory": memory,
        }

    @staticmethod
    def _graph_route_after_plan(state: dict[str, Any]) -> str:
        plan = state.get("orchestration_plan") or {}
        if plan.get("needs_review_analysis"):
            return "nlp"
        if plan.get("needs_forecast"):
            return "forecast"
        return "visualize"

    @staticmethod
    def _graph_route_after_nlp(state: dict[str, Any]) -> str:
        plan = state.get("orchestration_plan") or {}
        if plan.get("needs_forecast"):
            return "forecast"
        return "visualize"

    @staticmethod
    def _extract_key_entities(data_result: DataResult) -> dict[str, str]:
        """Best-effort entity extraction from first result rows for follow-up prompts."""
        entities: dict[str, str] = {}
        for df in (data_result.tables or {}).values():
            if not isinstance(df, pd.DataFrame) or df.empty:
                continue
            cols = {str(c).lower(): c for c in df.columns}
            for key, candidates in [
                ("state", ["customer_state", "seller_state", "state"]),
                ("category", ["product_category_english", "product_category_name_english", "product_category_name"]),
                ("payment_type", ["payment_type"]),
            ]:
                if key in entities:
                    continue
                for cand in candidates:
                    if cand in cols:
                        val = df[cols[cand]].dropna().astype(str).iloc[0]
                        if val:
                            entities[key] = val
                        break
        return entities

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

        # overall 问题通常需要多维综合；如果 DataAgent 已返回历史销售序列，也允许补充预测。
        if getattr(data_result, "intent", "") == "overall" and forecast_by_data:
            needs_forecast = True

        return {
            "required_agents_from_data_plan": sorted(required_agents),
            "needs_review_analysis": needs_review,
            "needs_forecast": needs_forecast,
            "needs_visualization": True,
            "needs_decision": True,
            "review_reason": self._reason_flags(
                plan=review_by_plan, question=review_by_question, data=review_by_data, intent=review_by_intent_fallback
            ),
            "forecast_reason": self._reason_flags(
                plan=forecast_by_plan, question=forecast_by_question, data=forecast_by_data, intent=forecast_by_intent_fallback
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

    def _compose_answer(
        self,
        question: str,
        data_result: DataResult,
        decision_result: DecisionResult,
        nlp_result: NLPResult | None,
        forecast_result: ForecastResult | None,
        visualization_result: VisualizationResult,
        orchestration_plan: dict[str, Any],
        conversation_context: str = "",
    ) -> tuple[str, list[str], list[str], list[str], dict[str, Any]]:
        direct_answer, findings = self._llm_synthesize_narrative(
            question, data_result, nlp_result, forecast_result, conversation_context=conversation_context
        )
        recommendations = [r.strip() for r in decision_result.recommendations if r and r.strip()]

        technical_details = self._build_technical_details(
            data_result=data_result,
            nlp_result=nlp_result,
            forecast_result=forecast_result,
            visualization_result=visualization_result,
            orchestration_plan=orchestration_plan,
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

        return "\n".join(lines), direct_answer, findings, recommendations, technical_details

    def _llm_synthesize_narrative(
        self,
        question: str,
        data_result: DataResult,
        nlp_result: NLPResult | None,
        forecast_result: ForecastResult | None,
        conversation_context: str = "",
    ) -> tuple[list[str], list[str]]:
        """LLM-only synthesis for direct answer and key findings."""
        self.llm.require_enabled()
        evidence = self._build_narrative_evidence(data_result, nlp_result, forecast_result)
        evidence_json = json.dumps(evidence, ensure_ascii=False, default=str, indent=2)
        context_block = f"\n{conversation_context}\n" if conversation_context.strip() else ""
        prompt = f"""
你是 Agentic BI 的分析总结 Agent。根据用户问题和真实查询结果，生成「直接回答」与「关键发现」。

{context_block}
用户问题：{question}

evidence_json（只能引用这里的数据，禁止编造）：
{evidence_json}

要求：
1. direct_answer：1-3 条，用一句话直接回答用户问题（含具体数字、排名、支付方式名等）。
2. findings：1-3 条，补充 deeper insight（对比、集中度、异常、样本量提醒等），不要与 direct_answer 重复。
3. 禁止输出「请结合下方明细查看」「描述性统计为主」「建议结合图表」等空泛套话。
4. 若 evidence 不足以回答，写「当前查询结果未提供该指标」，不要猜测。
5. 只返回严格 JSON，不要 Markdown。

返回格式：
{{
  "direct_answer": ["..."],
  "findings": ["..."]
}}
"""
        messages = [
            {"role": "system", "content": "你是严谨的 BI 分析总结 Agent，只返回合法 JSON。"},
            {"role": "user", "content": prompt},
        ]
        last_error = "LLM 未返回内容"
        for max_tokens in (4096, 8192):
            content = self.llm.chat(messages=messages, temperature=0.1, max_tokens=max_tokens)
            try:
                parsed = LLMClient.extract_json(content)
                direct = [str(x).strip() for x in (parsed.get("direct_answer") or []) if str(x).strip()]
                findings = [str(x).strip() for x in (parsed.get("findings") or []) if str(x).strip()]
                direct = self._dedupe_lines(direct)[:3]
                findings = self._dedupe_lines(findings)[:3]
                if not direct and not findings:
                    raise ValueError("LLM 返回的 direct_answer 与 findings 均为空")
                return direct, findings
            except ValueError as exc:
                last_error = str(exc)
        raise ValueError(last_error)

    def _build_narrative_evidence(
        self,
        data_result: DataResult,
        nlp_result: NLPResult | None,
        forecast_result: ForecastResult | None,
    ) -> dict[str, Any]:
        packet: dict[str, Any] = {
            "data_summary": getattr(data_result, "summary", ""),
            "intent": getattr(data_result, "intent", ""),
            "tables": {},
        }
        for name, df in (getattr(data_result, "tables", {}) or {}).items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                work = df.head(15).copy()
                for col in work.columns:
                    if pd.api.types.is_float_dtype(work[col]):
                        work[col] = work[col].round(4)
                work = work.where(pd.notnull(work), None)
                packet["tables"][name] = work.to_dict(orient="records")

        if nlp_result is not None:
            packet["nlp"] = {"summary": getattr(nlp_result, "summary", "")}
            top_neg = getattr(nlp_result, "top_negative_categories", pd.DataFrame())
            if isinstance(top_neg, pd.DataFrame) and not top_neg.empty:
                packet["nlp"]["top_negative_categories"] = top_neg.head(8).to_dict(orient="records")

        if forecast_result is not None:
            packet["forecast"] = {"summary": getattr(forecast_result, "summary", "")}
            fc = getattr(forecast_result, "forecast_df", pd.DataFrame())
            if isinstance(fc, pd.DataFrame) and not fc.empty:
                packet["forecast"]["rows"] = fc.head(8).to_dict(orient="records")

        return packet

    def _build_technical_details(
        self,
        data_result: DataResult,
        nlp_result: NLPResult | None,
        forecast_result: ForecastResult | None,
        visualization_result: VisualizationResult,
        orchestration_plan: dict[str, Any],
    ) -> dict[str, Any]:
        routing = getattr(data_result, "routing_method", "llm")
        query_mode = "大模型生成 SQL" if routing == "llm" else str(routing)

        return {
            "query_mode": query_mode,
            "data_source": "预聚合表 mv_*" if data_result.used_preaggregation else "基础表 / 自定义 SQL",
            "intent": getattr(data_result, "intent", ""),
            "chart_count": len(visualization_result.figures),
            "llm_error": getattr(data_result, "llm_error", "") or "",
            "returned_tables": orchestration_plan.get("returned_tables", []),
            "agents": {
                "评论洞察": {"called": nlp_result is not None, "reason": self._humanize_reasons(orchestration_plan.get("review_reason", []))},
                "销售预测": {"called": forecast_result is not None, "reason": self._humanize_reasons(orchestration_plan.get("forecast_reason", []))},
                "可视化": {"called": True, "reason": "LLM 规划图表（含 wordcloud）"},
                "决策智能": {"called": True, "reason": "LLM 生成建议"},
                "分析总结": {"called": True, "reason": "LLM 生成直接回答与关键发现"},
                "编排框架": {"called": True, "reason": "LangGraph StateGraph"},
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
