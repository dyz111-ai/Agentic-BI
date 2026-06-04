from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def _prepare_prophet_history(monthly_df: pd.DataFrame) -> pd.DataFrame:
    """Convert mv_monthly_sales (year_month, total_gmv) to weekly Prophet series."""
    if monthly_df.empty or "total_gmv" not in monthly_df.columns:
        return pd.DataFrame(columns=["ds", "y"])

    df = monthly_df.copy()
    if "year_month" not in df.columns:
        return pd.DataFrame(columns=["ds", "y"])

    df["ds"] = pd.to_datetime(df["year_month"].astype(str).str[:7] + "-01", errors="coerce")
    df["y"] = pd.to_numeric(df["total_gmv"], errors="coerce")
    df = df.dropna(subset=["ds", "y"]).sort_values("ds")
    monthly = df.groupby("ds", as_index=False)["y"].sum()
    if len(monthly) < 2:
        return pd.DataFrame(columns=["ds", "y"])

    # Monthly GMV -> weekly series (linear interpolation) for Prophet weekly forecast.
    weekly = (
        monthly.set_index("ds")
        .resample("W")
        .interpolate(method="linear")
        .reset_index()
    )
    weekly["y"] = weekly["y"].clip(lower=0)
    return weekly[["ds", "y"]]


def forecast_gmv(monthly_df: pd.DataFrame, periods_weeks: int = 6) -> pd.DataFrame:
    """Prophet-based weekly GMV forecast from mv_monthly_sales history.

    Input: monthly dataframe with year_month and total_gmv (from mv_monthly_sales).
    Output: future weekly forecast with ds, yhat, yhat_lower, yhat_upper (95% interval).
    """
    empty = pd.DataFrame(columns=["ds", "yhat", "yhat_lower", "yhat_upper"])
    history = _prepare_prophet_history(monthly_df)
    if len(history) < 8:
        return empty

    try:
        from prophet import Prophet
    except ImportError as exc:
        raise ImportError("未安装 prophet，请执行 pip install prophet") from exc

    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    logging.getLogger("prophet").setLevel(logging.WARNING)

    try:
        model = Prophet(
            interval_width=0.95,
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=len(history) >= 52,
        )
        model.fit(history)

        last_ds = history["ds"].max()
        future = model.make_future_dataframe(periods=periods_weeks, freq="W")
        pred = model.predict(future)

        fc = pred.loc[pred["ds"] > last_ds, ["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
        fc = fc.head(periods_weeks)
        for col in ("yhat", "yhat_lower", "yhat_upper"):
            fc[col] = fc[col].clip(lower=0).round(2)
        return fc.reset_index(drop=True)
    except Exception as exc:
        logger.warning("Prophet forecast failed: %s", exc)
        return empty
