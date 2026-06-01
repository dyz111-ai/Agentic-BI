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
        "床垫", "家电", "家居装饰品"  # 常见幻觉例子；真实数据出现时也不建议让 LLM凭空写。
    )

    def __init__(self):
        self.llm = LLMClient()

    def generate(
        self,
        question: str,
        data_result: Any,
        nlp_result: Any = None,
        forecast_result: Any = None,
    ) -> DecisionResult:
        evidence = self._build_evidence_packet(data_result, nlp_result, forecast_result)
        local_summary = self._local_summary_from_evidence(evidence)

        if self.llm.enabled:
            prompt = self._build_llm_prompt(question, evidence)
            text = self.llm.chat(
                [
                    {
                        "role": "system",
                        "content": (
                            SYSTEM_PROMPT
                            + "\n"
                            + DECISION_AGENT_PROMPT
                            + "\n你必须严格基于用户提供的 evidence_json 回答，禁止编造数据。"
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=1400,
            )
            if text and not text.startswith("LLM 调用失败"):
                recs = self._parse_recommendations(text)
                recs = self._remove_unsafe_or_empty_recs(recs, evidence)
                if recs:
                    return DecisionResult(
                        recommendations=recs,
                        summary="已基于真实查询结果、评论洞察和预测结果生成决策建议。",
                    )

        recs = self._rule_based_recommendations_from_evidence(evidence)
        return DecisionResult(recommendations=recs, summary=local_summary)

    # ------------------------------------------------------------------
    # LLM prompt and parsing
    # ------------------------------------------------------------------

    def _build_llm_prompt(self, question: str, evidence: dict[str, Any]) -> str:
        evidence_json = json.dumps(evidence, ensure_ascii=False, default=str, indent=2)
        return f"""
用户问题：{question}

下面是系统真实查询和分析得到的 evidence_json。你只能引用这里出现的数据、品类、州、卖家、关键词、预测值。

{evidence_json}

请基于 evidence_json 输出 3-5 条具体、可执行的电商运营建议。

硬性规则：
1. 禁止使用“假设”“假定”“示例数据”“可能如下”等表达。
2. 禁止编造 evidence_json 中不存在的品类、州、卖家、评分、差评率、订单量、GMV。
3. 如果 evidence_json 没有提供某项数据，必须写“当前查询结果未提供该指标”，不要补充猜测值。
4. 如果某个品类/卖家的 total_reviews 较小，必须提示“样本量偏小，结论仅作为风险预警”。
5. 每条建议必须包含：数据依据 + 具体动作。
6. 输出格式：每条建议单独一行，以“- ”开头；不要标题、不要编号、不要 Markdown 小标题。
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

    def _remove_unsafe_or_empty_recs(self, recs: list[str], evidence: dict[str, Any]) -> list[str]:
        """过滤明显幻觉/假设式建议。"""
        safe: list[str] = []
        evidence_text = json.dumps(evidence, ensure_ascii=False, default=str)
        for rec in recs:
            if not rec or len(rec.strip()) < 6:
                continue
            if any(bad in rec for bad in self.FORBIDDEN_CLAIMS):
                # 如果这些词并不在 evidence 里，则视为高风险幻觉。
                if not any(bad in evidence_text for bad in self.FORBIDDEN_CLAIMS if bad in rec):
                    continue
            safe.append(rec.strip())
        return safe[:5]

    # ------------------------------------------------------------------
    # Evidence packet
    # ------------------------------------------------------------------

    def _build_evidence_packet(self, data_result: Any, nlp_result: Any, forecast_result: Any) -> dict[str, Any]:
        packet: dict[str, Any] = {
            "data_summary": getattr(data_result, "summary", ""),
            "intent": getattr(data_result, "intent", ""),
            "routing_method": getattr(data_result, "routing_method", ""),
            "used_preaggregation": bool(getattr(data_result, "used_preaggregation", False)),
            "tables": {},
            "nlp": {},
            "forecast": {},
        }

        tables = getattr(data_result, "tables", {}) or {}
        for name, df in tables.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                packet["tables"][name] = self._df_preview(df, max_rows=10)

        if nlp_result is not None:
            packet["nlp"] = self._nlp_evidence(nlp_result)

        if forecast_result is not None:
            packet["forecast"] = self._forecast_evidence(forecast_result)

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

    def _rule_based_recommendations_from_evidence(self, evidence: dict[str, Any]) -> list[str]:
        recs: list[str] = []
        tables = evidence.get("tables", {}) or {}
        nlp = evidence.get("nlp", {}) or {}
        forecast = evidence.get("forecast", {}) or {}

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
        if fc_summary:
            recs.append(f"预测结果显示：{fc_summary} 建议提前规划广告预算、库存补货和物流资源，避免需求变化造成缺货或配送延迟。")

        if not recs:
            recs = ["当前查询结果提供的结构化指标较少。建议先补充销售、配送、支付、评论四类核心 KPI 后，再制定分区域、分品类、分卖家的运营策略。"]
        return recs[:6]
