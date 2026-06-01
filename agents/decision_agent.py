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
    """LLM-only decision agent. No rule/template fallback."""

    FORBIDDEN_CLAIMS = (
        "假设", "假定", "示例数据", "假设分析", "假设结果", "可能如下", "以下是假设",
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
        self.llm.require_enabled()
        evidence = self._build_evidence_packet(data_result, nlp_result, forecast_result)
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
            max_tokens=4096,
        )
        recs = self._parse_recommendations(text)
        recs = self._remove_unsafe_or_empty_recs(recs, evidence)
        if not recs:
            raise RuntimeError("LLM 未返回有效决策建议。")
        return DecisionResult(
            recommendations=recs,
            summary="已基于真实查询结果、评论洞察和预测结果生成决策建议。",
        )

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
        safe: list[str] = []
        evidence_text = json.dumps(evidence, ensure_ascii=False, default=str)
        for rec in recs:
            if not rec or len(rec.strip()) < 6:
                continue
            if any(bad in rec for bad in self.FORBIDDEN_CLAIMS):
                if not any(bad in evidence_text for bad in self.FORBIDDEN_CLAIMS if bad in rec):
                    continue
            safe.append(rec.strip())
        return safe[:5]

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
