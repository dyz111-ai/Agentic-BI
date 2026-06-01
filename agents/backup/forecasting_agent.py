from __future__ import annotations

from dataclasses import dataclass, field
import pandas as pd
from models.forecast import forecast_gmv


@dataclass
class ForecastResult:
    forecast_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    summary: str = ""


class ForecastAgent:
    def forecast_sales(self, monthly_sales_df: pd.DataFrame) -> ForecastResult:
        fc = forecast_gmv(monthly_sales_df, periods_weeks=6)
        if fc.empty:
            return ForecastResult(fc, "历史销售序列不足，无法生成预测。")
        first = fc.iloc[0]["forecast_gmv"]
        last = fc.iloc[-1]["forecast_gmv"]
        direction = "上升" if last > first else "下降" if last < first else "平稳"
        summary = f"未来 6 周 GMV 预测整体呈{direction}趋势，最后一周预测值约 {last:.2f}。"
        return ForecastResult(fc, summary)
