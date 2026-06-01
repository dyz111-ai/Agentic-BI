from __future__ import annotations

from dataclasses import dataclass, field
import json
import re

import pandas as pd

from models.forecast import forecast_gmv
from utils.llm import LLMClient


@dataclass
class ForecastResult:
    forecast_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    summary: str = ""
    evidence_json: dict = field(default_factory=dict)


class ForecastAgent:
    """Forecast agent with LLM interpretation and no text-template fallback."""

    def __init__(self):
        self.llm = LLMClient()

    def forecast_sales(self, monthly_sales_df: pd.DataFrame) -> ForecastResult:
        if monthly_sales_df is None or monthly_sales_df.empty:
            raise ValueError("ForecastAgent 需要历史销售时间序列，但本次没有收到可预测数据。")
        if not self.llm.enabled:
            raise RuntimeError("未配置 LLM_API_KEY / DEEPSEEK_API_KEY，ForecastAgent 不生成模板化预测解读。")

        fc = forecast_gmv(monthly_sales_df, periods_weeks=6)
        if fc.empty:
            raise ValueError("历史销售序列不足，无法生成预测。")

        evidence = {
            "forecast_horizon": "6 weeks",
            "history_rows": int(len(monthly_sales_df)),
            "forecast_rows": self._df_records(fc, 12),
        }
        summary = self._interpret_with_llm(evidence)
        return ForecastResult(fc, summary, evidence)

    def _interpret_with_llm(self, evidence: dict) -> str:
        prompt = f"""
你是预测分析 Agent。请只根据下面 JSON 中的预测结果做一句到两句趋势解读。
不要提出运营建议；不要编造 JSON 之外的数据。

JSON：
{json.dumps(evidence, ensure_ascii=False)}

只返回严格 JSON：{{"summary": "..."}}
"""
        content = self.llm.chat(
            messages=[
                {"role": "system", "content": "你只基于输入 JSON 解读预测结果，只返回 JSON。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=500,
        )
        obj = self._extract_json(content)
        summary = str(obj.get("summary") or "").strip()
        if not summary:
            raise ValueError("ForecastAgent 的 LLM 未返回 summary。")
        return summary

    @staticmethod
    def _df_records(df: pd.DataFrame, n: int = 12) -> list[dict]:
        records = []
        for r in df.head(n).to_dict(orient="records"):
            clean = {}
            for k, v in r.items():
                if pd.isna(v):
                    clean[k] = None
                elif hasattr(v, "isoformat"):
                    clean[k] = v.isoformat()
                elif isinstance(v, float):
                    clean[k] = round(v, 4)
                else:
                    clean[k] = v
            records.append(clean)
        return records

    @staticmethod
    def _extract_json(content: str) -> dict:
        text = (content or "").strip()
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if not m:
                raise ValueError(f"LLM 未返回 JSON：{text[:500]}")
            return json.loads(m.group(0))
