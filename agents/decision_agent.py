from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import pandas as pd

from utils.llm import LLMClient
from config.prompts import SYSTEM_PROMPT, DECISION_AGENT_PROMPT


@dataclass
class DecisionResult:
    recommendations: list[str]
    summary: str


class DecisionIntelligenceAgent:
    """Decision Intelligence Agent.

    改进点：
    1. 不再只把 summary 交给 LLM，而是把真实查询表、NLP 结构化结果、预测结果整理成 evidence packet。
    2. Prompt 明确禁止“假设/示例/编造数据”。
    3. 如果 LLM 仍输出疑似编造内容，则自动回退到基于真实字段的规则建议。
    4. 规则兜底也只引用真实表格中存在的字段。
    """

    FORBIDDEN_CLAIMS = (
        "假设", "假定", "示例数据", "假设分析", "假设结果", "可能如下", "以下是假设",
        "床垫", "家电", "家居装饰品",
    )

    META_JARGON = (
        "evidence_json", "evidence packet", "JSON", "ETL", "mv_monthly_sales",
        "mv_state_sales", "预聚合层", "数据血源", "增量+全量", "order_purchase_timestamp",
    )

    UNAVAILABLE_PHRASES = (
        "当前查询未返回",
        "未返回该",
        "未返回任何",
        "无法提供",
        "无法分析",
        "无法基于",
        "请要求数据团队",
        "请确认订单",
        "数据团队补充",
        "无法进行区域",
        "无法制定",
        "缺少按州",
        "查询结果未返回",
    )

    def __init__(self):
        self.llm = LLMClient()

    def generate(
        self,
        question: str,
        data_result: Any,
        nlp_result: Any = None,
        forecast_result: Any = None,
        what_if_result: Any = None,
        conversation_context: str = "",
    ) -> DecisionResult:
        evidence = self._build_evidence_packet(data_result, nlp_result, forecast_result, what_if_result)
        local_summary = self._local_summary_from_evidence(evidence)

        if self.llm.enabled:
            try:
                prompt = self._build_llm_prompt(question, evidence, conversation_context=conversation_context)
                text = self.llm.chat(
                    [
                        {
                            "role": "system",
                            "content": (
                                SYSTEM_PROMPT
                                + "\n"
                                + DECISION_AGENT_PROMPT
                                + "\n你必须严格基于用户提供的查询结果回答，禁止编造数据；回答中禁止出现 evidence_json、JSON、ETL 等内部术语。"
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.1,
                    max_tokens=4096,
                )
                if text and not text.startswith("LLM 调用失败"):
                    recs = self._parse_recommendations(text)
                    recs = self._remove_unsafe_or_empty_recs(recs, evidence)
                    recs = self._filter_user_facing_recs(recs, evidence, question)
                    if recs:
                        return DecisionResult(
                            recommendations=recs,
                            summary="已基于真实查询结果、评论洞察和预测结果生成决策建议。",
                        )
            except Exception:
                pass

        recs = self._rule_based_recommendations_from_evidence(evidence, question)
        return DecisionResult(recommendations=recs, summary=local_summary)

    # ------------------------------------------------------------------
    # LLM prompt and parsing
    # ------------------------------------------------------------------

    def _build_llm_prompt(self, question: str, evidence: dict[str, Any], conversation_context: str = "") -> str:
        evidence_json = json.dumps(evidence, ensure_ascii=False, default=str, indent=2)
        has_tables = bool(evidence.get("tables"))
        empty_hint = ""
        if not has_tables:
            empty_hint = (
                "\n注意：当前查询结果为空。请给出 2-3 条面向业务人员的建议："
                "先确认订单数据是否已导入、预聚合表是否已刷新，再重新分析。"
                "禁止写 ETL、数据血源、evidence_json 等技术用语。"
            )
        if conversation_context:
            question_block = f"{conversation_context.rstrip()}\n{question}"
        else:
            question_block = f"用户问题：{question}"
        return f"""
{question_block}

下面是系统真实查询得到的结构化数据（仅供你阅读，不要在回答里出现 JSON 或字段名）：

{evidence_json}
{empty_hint}

请基于上述查询结果输出 3-5 条具体、可执行的电商运营建议。

硬性规则：
1. 禁止使用“假设”“假定”“示例数据”“可能如下”等表达。
2. 禁止编造查询结果中不存在的品类、州、卖家、评分、差评率、订单量、GMV。
3. 若某项指标缺失，直接跳过该维度，不要写「未返回」「无法分析」或要求用户补数据的表述。
4. 如果某个品类/卖家的 total_reviews 较小，必须提示「样本量偏小，结论仅作为风险预警」。
5. 每条建议用自然中文，格式为「数据依据：…。具体动作：…。」；禁止出现 evidence_json、SQL、表名等技术词。
6. 输出格式：每条建议单独一行，以「- 」开头；不要标题、不要编号、不要 Markdown 小标题。
7. 若提供了【会话上下文】，当前问题可能是追问；请结合上一轮分析结论给出连贯建议，正确理解「那个州/该品类」等指代。
8. 禁止编造预测结果；仅当查询证据中包含 forecast 摘要时，才可给出预测相关建议。
"""

    @staticmethod
    def _parse_recommendations(text: str) -> list[str]:
        """Split LLM output into clean bullet items."""
        lines: list[str] = []
        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue
            line = re.sub(r"^#{1,6}\s+", "", line)
            line = re.sub(r"^[-*•]\s+", "", line)
            line = re.sub(r"^\d+[.)、]\s+", "", line)
            if line and line not in {"决策建议", "运营建议", "建议", "核心结论", "数据依据"}:
                lines.append(line)
        if not lines and text.strip():
            chunks = re.split(r"(?<=[。！？])\s*", text.strip())
            lines = [c.strip() for c in chunks if len(c.strip()) > 4]
        return lines[:8]

    def _sanitize_recommendation(self, rec: str) -> str:
        text = rec.strip()
        text = re.sub(r"evidence_json\s*中\s*", "查询结果中", text, flags=re.IGNORECASE)
        text = re.sub(r"\bevidence_json\b", "查询结果", text, flags=re.IGNORECASE)
        text = re.sub(r"\bgmv_total\b", "GMV汇总", text, flags=re.IGNORECASE)
        return text

    def _remove_unsafe_or_empty_recs(self, recs: list[str], evidence: dict[str, Any]) -> list[str]:
        """过滤明显幻觉/假设式建议及内部术语泄露。"""
        safe: list[str] = []
        evidence_text = json.dumps(evidence, ensure_ascii=False, default=str)
        for rec in recs:
            cleaned = self._sanitize_recommendation(rec)
            if not cleaned or len(cleaned.strip()) < 6:
                continue
            if any(j in cleaned for j in self.META_JARGON):
                continue
            if any(bad in cleaned for bad in self.FORBIDDEN_CLAIMS):
                if not any(bad in evidence_text for bad in self.FORBIDDEN_CLAIMS if bad in cleaned):
                    continue
            safe.append(cleaned.strip())
        return safe[:5]

    # ------------------------------------------------------------------
    # Evidence packet
    # ------------------------------------------------------------------

    def _build_evidence_packet(self, data_result: Any, nlp_result: Any, forecast_result: Any, what_if_result: Any = None) -> dict[str, Any]:
        packet: dict[str, Any] = {
            "data_summary": getattr(data_result, "summary", ""),
            "intent": getattr(data_result, "intent", ""),
            "routing_method": getattr(data_result, "routing_method", ""),
            "used_preaggregation": bool(getattr(data_result, "used_preaggregation", False)),
            "tables": {},
            "nlp": {},
            "forecast": {},
            "what_if": {},
        }

        tables = getattr(data_result, "tables", {}) or {}
        for name, df in tables.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                packet["tables"][name] = self._df_preview(df, max_rows=10)

        if nlp_result is not None:
            packet["nlp"] = self._nlp_evidence(nlp_result)

        if forecast_result is not None:
            packet["forecast"] = self._forecast_evidence(forecast_result)

        if what_if_result is not None and getattr(what_if_result, "has_result", False):
            packet["what_if"] = self._what_if_evidence(what_if_result)

        return packet

    def _nlp_evidence(self, nlp_result: Any) -> dict[str, Any]:
        out: dict[str, Any] = {
            "summary": getattr(nlp_result, "summary", ""),
            "negative_keywords": self._keyword_preview(getattr(nlp_result, "negative_keywords", []), 15),
            "positive_keywords": self._keyword_preview(getattr(nlp_result, "positive_keywords", []), 10),
        }
        top_neg = getattr(nlp_result, "top_negative_categories", pd.DataFrame())
        if isinstance(top_neg, pd.DataFrame) and not top_neg.empty:
            out["top_negative_categories"] = self._df_preview(top_neg, max_rows=10)
        reason_summary = getattr(nlp_result, "negative_reason_summary", "")
        if reason_summary:
            out["negative_reason_summary"] = reason_summary
        sample_reviews = getattr(nlp_result, "sample_negative_reviews", [])
        if sample_reviews:
            out["sample_negative_reviews"] = sample_reviews[:8]
        return out

    def _forecast_evidence(self, forecast_result: Any) -> dict[str, Any]:
        out = {"summary": getattr(forecast_result, "summary", "")}
        fc = getattr(forecast_result, "forecast_df", pd.DataFrame())
        if isinstance(fc, pd.DataFrame) and not fc.empty:
            out["forecast_rows"] = self._df_preview(fc, max_rows=8)
        return out

    def _what_if_evidence(self, what_if_result: Any) -> dict[str, Any]:
        return {
            "scenario": getattr(what_if_result, "scenario", ""),
            "current_avg_score": getattr(what_if_result, "current_avg_score", 0),
            "projected_avg_score": getattr(what_if_result, "projected_avg_score", 0),
            "score_improvement": getattr(what_if_result, "score_improvement", 0),
            "score_improvement_pct": getattr(what_if_result, "score_improvement_pct", 0),
            "removed_seller_count": getattr(what_if_result, "removed_seller_count", 0),
            "removed_order_count": getattr(what_if_result, "removed_order_count", 0),
            "total_order_count": getattr(what_if_result, "total_order_count", 0),
            "summary": getattr(what_if_result, "summary", ""),
        }

    @staticmethod
    def _df_preview(df: pd.DataFrame, max_rows: int = 10) -> list[dict[str, Any]]:
        work = df.head(max_rows).copy()
        # Make floats readable and JSON-friendly.
        for col in work.columns:
            if pd.api.types.is_float_dtype(work[col]):
                work[col] = work[col].round(4)
        work = work.where(pd.notnull(work), None)
        return work.to_dict(orient="records")

    @staticmethod
    def _keyword_preview(keywords: list[tuple[str, int]] | list[Any], max_items: int) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for item in keywords[:max_items]:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                out.append({"keyword": str(item[0]), "count": int(item[1])})
            else:
                out.append({"keyword": str(item), "count": None})
        return out

    def _local_summary_from_evidence(self, evidence: dict[str, Any]) -> str:
        parts = []
        if evidence.get("data_summary"):
            parts.append(str(evidence["data_summary"]))
        if evidence.get("nlp", {}).get("summary"):
            parts.append(str(evidence["nlp"]["summary"]))
        if evidence.get("forecast", {}).get("summary"):
            parts.append(str(evidence["forecast"]["summary"]))
        return "\n".join(parts) if parts else "已完成数据分析，当前可用证据较少。"

    # ------------------------------------------------------------------
    # Rule-based fallback grounded in evidence
    # ------------------------------------------------------------------

    def _rule_based_recommendations(self, data_result: Any, nlp_result: Any, forecast_result: Any) -> list[str]:
        evidence = self._build_evidence_packet(data_result, nlp_result, forecast_result)
        return self._rule_based_recommendations_from_evidence(evidence)

    def _rule_based_recommendations_from_evidence(self, evidence: dict[str, Any], question: str = "") -> list[str]:
        recs: list[str] = []
        tables = evidence.get("tables", {}) or {}
        nlp = evidence.get("nlp", {}) or {}
        forecast = evidence.get("forecast", {}) or {}

        # State sales ranking.
        states = tables.get("state_sales_2017") or tables.get("state_sales")
        if states:
            row = states[0]
            state = row.get("customer_state") or row.get("state")
            gmv = row.get("total_gmv")
            if state:
                msg = f"GMV 最高的州为 {state}"
                if gmv is not None:
                    msg += f"（约 {float(gmv):,.2f}）"
                recs.append(f"{msg}。建议在该州加强库存与物流资源投入，并复制其高转化品类的运营打法到其他区域。")

        # Delivery recommendations.
        delivery = tables.get("delivery_by_state") or tables.get("delivery_monthly")
        if delivery:
            row = delivery[0]
            state = row.get("customer_state") or row.get("state") or row.get("seller_state")
            days = row.get("avg_delivery_days") or row.get("delivery_days")
            rate = row.get("on_time_rate")
            if state:
                detail = f"配送表现最需关注的州为 {state}"
                if days is not None:
                    detail += f"，平均配送约 {days} 天"
                if rate is not None:
                    detail += f"，准时率约 {float(rate):.1%}"
                recs.append(f"{detail}。建议增加本地承运商、优化预计送达日期，并对延迟订单建立周度预警。")

        # Low score sellers.
        sellers = tables.get("low_score_sellers") or tables.get("seller_perf")
        if sellers:
            row = sellers[0]
            seller = row.get("seller_id")
            score = row.get("avg_review_score") or row.get("avg_score")
            if seller:
                msg = f"卖家 {seller} 评分偏低"
                if score is not None:
                    msg += f"，平均评分约 {score}"
                recs.append(f"{msg}。建议设置低评分卖家复核机制，限制问题商品流量，并要求卖家提交售后整改计划。")

        # Payment.
        payments = tables.get("payment_dist") or tables.get("payment_monthly")
        if payments:
            row = payments[0]
            ptype = row.get("payment_type")
            inst = row.get("avg_installments")
            if ptype:
                msg = f"支付方式以 {ptype} 为主"
                if inst is not None:
                    msg += f"，平均分期数约 {inst}"
                recs.append(f"{msg}。建议优先优化该支付链路，并结合分期偏好设计免息或手续费减免活动。")

        # Category sales.
        cats = tables.get("top_categories") or tables.get("category_monthly")
        if cats:
            row = cats[0]
            cat = row.get("product_category_english") or row.get("product_category_name_english")
            gmv = row.get("total_gmv")
            if cat:
                msg = f"重点销售品类为 {cat}"
                if gmv is not None:
                    msg += f"，GMV 约 {gmv}"
                recs.append(f"{msg}。建议围绕该品类做库存、广告投放和物流 SLA 的专项优化。")

        # Negative categories from NLP.
        neg_cats = nlp.get("top_negative_categories") or []
        if neg_cats:
            row = neg_cats[0]
            cat = row.get("product_category_english") or row.get("product_category_name_english")
            total = row.get("total_reviews")
            neg = row.get("negative_reviews")
            rate = row.get("negative_rate")
            avg = row.get("avg_score")
            if cat:
                sample_warning = ""
                try:
                    if total is not None and float(total) < 20:
                        sample_warning = "样本量偏小，结论仅作为风险预警；"
                except Exception:
                    pass
                detail = f"差评风险最高的品类为 {cat}"
                if total is not None and neg is not None:
                    detail += f"，评论数 {total}、差评数 {neg}"
                if rate is not None:
                    detail += f"、差评率 {float(rate):.1%}"
                if avg is not None:
                    detail += f"、平均评分 {float(avg):.2f}"
                recs.append(f"{sample_warning}{detail}。建议先抽检该品类商品质量、详情页描述和售后记录，再决定是否限流或下架问题商品。")

        # Negative keywords.
        kws = nlp.get("negative_keywords") or []
        if kws:
            kw_text = "、".join([str(k.get("keyword")) for k in kws[:5] if k.get("keyword")])
            if kw_text:
                recs.append(f"差评关键词集中在：{kw_text}。建议客服工单按关键词归因到物流、质量、描述不符、售后四类，并为每类设置处理时限。")

        # Forecast.
        fc_summary = forecast.get("summary")
        if fc_summary and forecast.get("forecast_rows"):
            recs.append(f"预测结果显示：{fc_summary} 建议提前规划广告预算、库存补货和物流资源，避免需求变化造成缺货或配送延迟。")

        # What-If simulation.
        what_if = evidence.get("what_if") or {}
        if what_if.get("current_avg_score"):
            recs.append(
                f"What-If 模拟：当前平台加权评分为 {what_if['current_avg_score']:.4f}，"
                f"移除评分最低的 {what_if.get('removed_seller_count', 0)} 个卖家后预计提升至 "
                f"{what_if['projected_avg_score']:.4f}（+{what_if.get('score_improvement_pct', 0):.2f}%）。"
                f"建议先执行分阶段观察（警告 → 限流 → 整改 → 下架），各阶段设置评分/订单数门槛与观察时限。"
            )

        if not recs:
            if not tables:
                recs = [
                    "数据依据：本次分析尚未拿到足够的结构化指标。具体动作：可在侧边栏点击「刷新预聚合表」后重新提问，或换一种更具体的问法（如按月 GMV、各州排名）。",
                ]
            else:
                recs = ["建议结合上方图表与数据明细，从销售、配送、支付、评论四个维度制定下一步运营动作。"]
        return self._filter_user_facing_recs(recs[:6], evidence, question)

    def _filter_user_facing_recs(
        self,
        recs: list[str],
        evidence: dict[str, Any],
        question: str,
    ) -> list[str]:
        q = question.lower()
        asks_forecast = any(k in q for k in ["预测", "未来", "forecast", "6周", "六周"])
        forecast_ev = evidence.get("forecast") or {}
        has_forecast = bool(forecast_ev.get("summary") or forecast_ev.get("forecast_rows"))

        safe: list[str] = []
        for rec in recs:
            if any(p in rec for p in self.UNAVAILABLE_PHRASES):
                continue
            if not asks_forecast and not has_forecast:
                if any(k in rec for k in ("预测模型", "未来6周", "未来 6 周", "Prophet", "yhat")):
                    continue
            safe.append(rec)
        return safe[:5]
