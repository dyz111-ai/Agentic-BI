from __future__ import annotations

import numpy as np
import pandas as pd


def forecast_gmv(monthly_df: pd.DataFrame, periods_weeks: int = 6) -> pd.DataFrame:
    """Simple deterministic trend forecast for demo/report.

    Input: monthly dataframe with year_month and total_gmv.
    Output: weekly forecast with lower/upper interval.
    For stronger submission, replace this with Prophet/ARIMA/XGBoost.
    """
    if monthly_df.empty or "total_gmv" not in monthly_df.columns:
        return pd.DataFrame(columns=["date", "forecast_gmv", "lower", "upper"])

    df = monthly_df.copy()
    df["date"] = pd.to_datetime(df["year_month"].astype(str) + "-01", errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    y = df["total_gmv"].astype(float).values
    x = np.arange(len(y))

    if len(y) >= 2:
        slope, intercept = np.polyfit(x, y, deg=1)
    else:
        slope, intercept = 0.0, float(y[0])

    residual = y - (slope * x + intercept)
    sigma = float(np.std(residual)) if len(residual) > 1 else max(float(y[-1]) * 0.08, 1.0)
    last_date = df["date"].max()

    rows = []
    for i in range(1, periods_weeks + 1):
        # Convert weekly steps to approximate month index increments.
        future_x = len(y) - 1 + i / 4.345
        pred = max(0.0, slope * future_x + intercept)
        rows.append({
            "date": last_date + pd.Timedelta(weeks=i),
            "forecast_gmv": round(pred, 2),
            "lower": round(max(0.0, pred - 1.96 * sigma), 2),
            "upper": round(pred + 1.96 * sigma, 2),
        })
    return pd.DataFrame(rows)
