from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from models.forecast import forecast_gmv

PROPHET_SUMMARY_PREFIX = (
    "本模块使用 Prophet，基于 mv_monthly_sales 的历史 GMV 数据预测未来 6 周销售额，并提供 95% 置信区间。"
)


@dataclass
class ForecastResult:
    forecast_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    summary: str = ""


class ForecastAgent:
    def forecast_sales(self, monthly_sales_df: pd.DataFrame) -> ForecastResult:
        fc = forecast_gmv(monthly_sales_df, periods_weeks=6)
        if fc.empty:
            return ForecastResult(fc, "历史销售序列不足，无法使用 Prophet 生成预测。")

        first = float(fc.iloc[0]["yhat"])
        last = float(fc.iloc[-1]["yhat"])
        direction = "上升" if last > first else "下降" if last < first else "平稳"
        trend = f"未来 6 周 GMV 预测整体呈{direction}趋势，首周约 {first:,.2f}，末周约 {last:,.2f}。"
        summary = f"{PROPHET_SUMMARY_PREFIX} {trend}"
        return ForecastResult(fc, summary)
