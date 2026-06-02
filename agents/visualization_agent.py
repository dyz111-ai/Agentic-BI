from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import re

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from config.settings import OUTPUT_DIR


@dataclass
class VisualizationResult:
    figures: dict[str, Any] = field(default_factory=dict)
    saved_files: dict[str, str] = field(default_factory=dict)
    summary: str = ""


STATE_COORDS = {
    "SP": (-23.55, -46.63), "RJ": (-22.90, -43.20), "MG": (-19.92, -43.94), "BA": (-12.97, -38.50),
    "PE": (-8.05, -34.88), "PR": (-25.43, -49.27), "RS": (-30.03, -51.23), "SC": (-27.59, -48.55),
    "CE": (-3.73, -38.52), "GO": (-16.68, -49.25), "ES": (-20.31, -40.31), "DF": (-15.79, -47.88),
    "AM": (-3.10, -60.02), "PA": (-1.45, -48.49), "MA": (-2.53, -44.30), "PB": (-7.12, -34.86),
    "RN": (-5.79, -35.21), "AL": (-9.65, -35.73), "SE": (-10.91, -37.07), "PI": (-5.09, -42.80),
    "MT": (-15.60, -56.10), "MS": (-20.45, -54.62), "RO": (-8.76, -63.90), "AC": (-9.97, -67.82),
    "RR": (2.82, -60.67), "AP": (0.03, -51.05), "TO": (-10.18, -48.33),
}


class VisualizationAgent:
    """Dynamic Visualization Agent.

    旧版主要按固定 table key 作图，例如 monthly_sales、payment_dist。
    新版保留这些专用图，同时增加“字段语义 + 数据类型”的通用推断：
    - 时间列 + 数值列：折线图
    - 州/地区列 + 数值列：柱状图/地理气泡图
    - 品类/支付/卖家等类别列 + 数值列：柱状图或饼图
    - 两个数值列：散点图
    - 两个类别列 + 数值列：热力图

    因此 LLM 生成 SQL 即使用 llm_result_1 这类新表名，只要列结构清楚，也能出图。
    """

    MAX_AUTO_FIGURES = 8

    def visualize(
        self,
        data_result: Any,
        nlp_result: Any = None,
        forecast_result: Any = None,
        question: str | None = None,
    ) -> VisualizationResult:
        res = VisualizationResult()
        tables = getattr(data_result, "tables", {}) or {}
        used_tables: set[str] = set()

        # 1) Keep task-specific charts when known table names exist.
        self._add_known_charts(res, tables, nlp_result, forecast_result, used_tables)

        # 2) Add dynamic charts for unknown/custom LLM query results.
        for table_name, df in tables.items():
            if len(res.figures) >= self.MAX_AUTO_FIGURES:
                break
            if not isinstance(df, pd.DataFrame) or df.empty:
                continue
            # If a known chart already covers the table, skip generic duplicate unless no figures were generated at all.
            if table_name in used_tables and len(res.figures) > 0:
                continue
            for title, fig in self._infer_figures_from_dataframe(table_name, df, question=question):
                if len(res.figures) >= self.MAX_AUTO_FIGURES:
                    break
                if title not in res.figures:
                    res.figures[title] = fig

        # 3) NLP-specific visualizations.
        self._add_nlp_charts(res, nlp_result)

        # 4) Forecast overlay if no monthly figure was created.
        if forecast_result is not None and "GMV 趋势与预测" not in "|".join(res.figures.keys()):
            monthly = self._find_monthly_like_table(tables)
            if monthly is not None and len(res.figures) < self.MAX_AUTO_FIGURES:
                res.figures["GMV 趋势与预测"] = self._monthly_sales_fig(monthly, forecast_result)

        self._save_figures(res)
        res.summary = f"已生成 {len(res.figures)} 个图表。"
        return res

    # ------------------------------------------------------------------
    # Known charts retained for course verification questions
    # ------------------------------------------------------------------

    def _add_known_charts(self, res: VisualizationResult, tables: dict[str, pd.DataFrame], nlp_result: Any, forecast_result: Any, used_tables: set[str]) -> None:
        monthly_df = self._resolve_monthly_series(tables)
        if monthly_df is not None:
            res.figures["月度 GMV 趋势"] = self._monthly_sales_fig(monthly_df, forecast_result)
            used_tables.add("monthly_sales")

        if self._valid(tables.get("state_sales")):
            df = self._aggregate_state_sales(tables["state_sales"])
            if {"customer_state", "total_gmv"}.issubset(df.columns):
                res.figures["各州销售额气泡图"] = self._state_map_fig(df)
                res.figures["各州销售额柱状图"] = px.bar(df.head(15), x="customer_state", y="total_gmv", title="各州 GMV 排名")
                used_tables.add("state_sales")

        if self._valid(tables.get("state_sales_2017")):
            df = self._aggregate_state_sales(tables["state_sales_2017"])
            if {"customer_state", "total_gmv"}.issubset(df.columns):
                res.figures["2017 各州 GMV 排名"] = px.bar(df.head(20), x="customer_state", y="total_gmv", title="2017 各州 GMV 排名")
                used_tables.add("state_sales_2017")

        if self._valid(tables.get("delivery_by_state")):
            df = tables["delivery_by_state"]
            if {"customer_state", "avg_delivery_days"}.issubset(df.columns):
                res.figures["配送准时率/时长"] = self._delivery_fig(df)
                used_tables.add("delivery_by_state")

        if self._valid(tables.get("payment_dist")):
            df = tables["payment_dist"]
            if {"payment_type", "total_transactions"}.issubset(df.columns):
                res.figures["支付方式分布"] = px.pie(df, names="payment_type", values="total_transactions", title="支付方式分布")
                used_tables.add("payment_dist")

        if self._valid(tables.get("payment_monthly")):
            df = tables["payment_monthly"]
            if {"payment_type", "year_month", "total_transactions"}.issubset(df.columns):
                res.figures["支付方式月度热力图"] = self._heatmap(df, index="payment_type", columns="year_month", values="total_transactions", title="支付方式 × 月份交易热力图")
                used_tables.add("payment_monthly")

        if self._valid(tables.get("top_categories")):
            df = tables["top_categories"]
            if {"product_category_english", "total_gmv"}.issubset(df.columns):
                res.figures["Top 品类 GMV"] = px.bar(df.head(20), x="total_gmv", y="product_category_english", orientation="h", title="Top 品类 GMV")
                used_tables.add("top_categories")

        if self._valid(tables.get("weight_freight")):
            df = tables["weight_freight"].copy()
            if {"product_weight_g", "freight_value"}.issubset(df.columns):
                size_col = "price" if "price" in df.columns else None
                kwargs = {}
                if "delivery_status" in df.columns:
                    kwargs["color"] = "delivery_status"
                if size_col:
                    kwargs["size"] = df[size_col].clip(lower=1, upper=df[size_col].quantile(0.95))
                res.figures["商品重量 vs 运费"] = px.scatter(df, x="product_weight_g", y="freight_value", title="商品重量与运费关系", **kwargs)
                used_tables.add("weight_freight")

    def _add_nlp_charts(self, res: VisualizationResult, nlp_result: Any) -> None:
        if nlp_result is not None and getattr(nlp_result, "top_negative_categories", pd.DataFrame()).empty is False:
            df = nlp_result.top_negative_categories
            if {"negative_rate", "product_category_english"}.issubset(df.columns):
                res.figures["差评品类 Top10"] = px.bar(df, x="negative_rate", y="product_category_english", orientation="h", title="Top 10 差评品类")

        if nlp_result is not None and getattr(nlp_result, "negative_keywords", None):
            res.figures["差评关键词"] = self._keyword_bar(nlp_result.negative_keywords, "差评关键词")

        if nlp_result is not None and getattr(nlp_result, "positive_keywords", None):
            if len(res.figures) < self.MAX_AUTO_FIGURES:
                res.figures["好评关键词"] = self._keyword_bar(nlp_result.positive_keywords, "好评关键词")

    # ------------------------------------------------------------------
    # Dynamic chart inference
    # ------------------------------------------------------------------

    def _infer_figures_from_dataframe(self, table_name: str, df: pd.DataFrame, question: str | None = None) -> list[tuple[str, Any]]:
        work = self._clean_df(df)
        if work.empty:
            return []

        cols = list(work.columns)
        numeric_cols = list(work.select_dtypes(include="number").columns)
        categorical_cols = [c for c in cols if c not in numeric_cols and work[c].nunique(dropna=True) <= max(50, min(len(work), 100))]

        figures: list[tuple[str, Any]] = []

        time_col = self._find_time_col(work)
        value_col = self._find_metric_col(work, prefer=["total_gmv", "gmv", "sales", "revenue", "total_value", "payment_value", "total_orders", "orders", "count"])
        state_col = self._find_col(work, ["customer_state", "seller_state", "state"])
        category_col = self._find_col(work, ["product_category_english", "category", "product_category_name"])
        payment_col = self._find_col(work, ["payment_type"])
        seller_col = self._find_col(work, ["seller_id", "seller_state"])
        review_score_col = self._find_col(work, ["review_score", "avg_review_score", "avg_score"])
        freight_col = self._find_col(work, ["freight_value", "total_freight", "avg_freight"])
        weight_col = self._find_col(work, ["product_weight_g", "weight"])

        # Time series
        if time_col and value_col:
            ts = self._prepare_time_series(work, time_col, value_col)
            if not ts.empty:
                figures.append((f"{table_name} 时间趋势", px.line(ts, x="date", y=value_col, markers=True, title=f"{table_name} 时间趋势")))

        # Geographic/state ranking
        if state_col and value_col:
            state_df = work.groupby(state_col, as_index=False)[value_col].sum().sort_values(value_col, ascending=False).head(20)
            figures.append((f"{table_name} 地区排名", px.bar(state_df, x=state_col, y=value_col, title=f"{table_name} 地区排名")))
            if self._looks_like_brazil_state(work[state_col]):
                figures.append((f"{table_name} 地区气泡图", self._state_map_fig(state_df.rename(columns={state_col: "customer_state", value_col: "total_gmv"}))))

        # Category/payment/seller ranking
        for label, cat_col in [("品类", category_col), ("支付方式", payment_col), ("卖家", seller_col)]:
            if cat_col and value_col:
                agg = work.groupby(cat_col, as_index=False)[value_col].sum().sort_values(value_col, ascending=False).head(20)
                orientation = "h" if label in {"品类", "卖家"} else "v"
                if orientation == "h":
                    figures.append((f"{table_name} {label}排名", px.bar(agg, x=value_col, y=cat_col, orientation="h", title=f"{table_name} {label}排名")))
                else:
                    figures.append((f"{table_name} {label}分布", px.bar(agg, x=cat_col, y=value_col, title=f"{table_name} {label}分布")))
                if label == "支付方式" and len(agg) <= 10:
                    figures.append((f"{table_name} 支付方式占比", px.pie(agg, names=cat_col, values=value_col, title=f"{table_name} 支付方式占比")))

        # Delivery/review score style metrics by category/state
        if review_score_col and (category_col or seller_col or state_col):
            group_col = category_col or seller_col or state_col
            agg = work.groupby(group_col, as_index=False)[review_score_col].mean().sort_values(review_score_col).head(15)
            figures.append((f"{table_name} 平均评分对比", px.bar(agg, x=review_score_col, y=group_col, orientation="h", title=f"{table_name} 平均评分对比")))

        # Weight / freight scatter, or generic two numeric scatter
        if weight_col and freight_col:
            sample = work[[weight_col, freight_col] + ([category_col] if category_col else [])].dropna().head(5000)
            kwargs = {"color": category_col} if category_col else {}
            figures.append((f"{table_name} 重量与运费关系", px.scatter(sample, x=weight_col, y=freight_col, title=f"{table_name} 重量与运费关系", **kwargs)))
        elif len(numeric_cols) >= 2 and len(work) >= 2:
            x_col, y_col = self._choose_scatter_cols(work, numeric_cols)
            color_col = category_col or state_col or payment_col
            sample_cols = [x_col, y_col] + ([color_col] if color_col else [])
            sample = work[sample_cols].dropna().head(5000)
            if not sample.empty:
                kwargs = {"color": color_col} if color_col else {}
                figures.append((f"{table_name} 数值关系散点图", px.scatter(sample, x=x_col, y=y_col, title=f"{table_name} 数值关系散点图", **kwargs)))

        # Matrix heatmap: two category columns + one metric.
        if len(categorical_cols) >= 2 and value_col:
            c1, c2 = self._choose_heatmap_categories(work, categorical_cols)
            if c1 and c2 and c1 != c2:
                figures.append((f"{table_name} 交叉热力图", self._heatmap(work, index=c1, columns=c2, values=value_col, title=f"{table_name} 交叉热力图")))

        # Single categorical frequency chart fallback.
        if not figures and categorical_cols:
            c = categorical_cols[0]
            vc = work[c].astype(str).value_counts().head(20).reset_index()
            vc.columns = [c, "count"]
            figures.append((f"{table_name} 频次分布", px.bar(vc, x=c, y="count", title=f"{table_name} 频次分布")))

        # Single numeric histogram fallback.
        if not figures and numeric_cols:
            c = numeric_cols[0]
            figures.append((f"{table_name} {c} 分布", px.histogram(work, x=c, title=f"{table_name} {c} 分布")))

        # Reduce duplicates while preserving order.
        unique: list[tuple[str, Any]] = []
        seen_titles: set[str] = set()
        for title, fig in figures:
            if title not in seen_titles:
                unique.append((title, fig))
                seen_titles.add(title)
        return unique[:3]

    # ------------------------------------------------------------------
    # Specific figure helpers
    # ------------------------------------------------------------------

    def _monthly_sales_fig(self, monthly_df: pd.DataFrame, forecast_result: Any = None):
        df = monthly_df.copy()
        if "year_month" not in df.columns:
            df = self._normalize_monthly_sales(df)
        if df is None or df.empty:
            return go.Figure().update_layout(title="月度 GMV 趋势：无可用数据")

        df["date"] = pd.to_datetime(df["year_month"].astype(str) + "-01", errors="coerce")
        df = df.dropna(subset=["date"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["date"], y=df["total_gmv"], mode="lines+markers", name="历史 GMV"))
        if forecast_result is not None and not getattr(forecast_result, "forecast_df", pd.DataFrame()).empty:
            fc = forecast_result.forecast_df
            fig.add_trace(go.Scatter(x=fc["date"], y=fc["forecast_gmv"], mode="lines+markers", name="未来 6 周预测"))
            if {"upper", "lower"}.issubset(fc.columns):
                fig.add_trace(go.Scatter(x=fc["date"], y=fc["upper"], mode="lines", name="置信区间上界", line=dict(width=0), showlegend=False))
                fig.add_trace(go.Scatter(x=fc["date"], y=fc["lower"], mode="lines", name="置信区间", fill="tonexty", line=dict(width=0)))
        fig.update_layout(title="月度 GMV 趋势与预测", xaxis_title="日期", yaxis_title="GMV")
        return fig

    def _state_map_fig(self, df: pd.DataFrame):
        map_df = df.copy()
        if "customer_state" not in map_df.columns:
            state_col = self._find_col(map_df, ["state", "seller_state", "customer_state"])
            if state_col:
                map_df = map_df.rename(columns={state_col: "customer_state"})
        if "total_gmv" not in map_df.columns:
            metric = self._find_metric_col(map_df)
            if metric:
                map_df = map_df.rename(columns={metric: "total_gmv"})
        if "total_orders" not in map_df.columns:
            map_df["total_orders"] = 1
        map_df["lat"] = map_df["customer_state"].map(lambda s: STATE_COORDS.get(str(s).upper(), (None, None))[0])
        map_df["lon"] = map_df["customer_state"].map(lambda s: STATE_COORDS.get(str(s).upper(), (None, None))[1])
        map_df = map_df.dropna(subset=["lat", "lon"])
        if map_df.empty:
            return go.Figure().update_layout(title="巴西各州气泡图：无可用经纬度")
        fig = px.scatter_geo(
            map_df, lat="lat", lon="lon", size="total_gmv", color="total_orders",
            hover_name="customer_state", scope="south america", title="巴西各州销售额/订单量气泡图"
        )
        fig.update_geos(center=dict(lat=-14, lon=-52), projection_scale=3.2)
        return fig

    def _delivery_fig(self, df: pd.DataFrame):
        state_col = self._find_col(df, ["customer_state", "seller_state", "state"]) or df.columns[0]
        days_col = self._find_col(df, ["avg_delivery_days", "delivery_days", "avg_days"]) or self._find_metric_col(df)
        rate_col = self._find_col(df, ["on_time_rate", "timely_rate", "rate"])
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df[state_col], y=df[days_col], name="平均配送天数"))
        if rate_col:
            fig.add_trace(go.Scatter(x=df[state_col], y=df[rate_col], yaxis="y2", mode="lines+markers", name="准时率"))
            fig.update_layout(yaxis2=dict(title="准时率", overlaying="y", side="right", tickformat=".0%"))
        fig.update_layout(title="配送时长与准时率", yaxis=dict(title="平均配送天数"))
        return fig

    def _heatmap(self, df: pd.DataFrame, index: str, columns: str, values: str, title: str):
        tmp = df[[index, columns, values]].copy().dropna()
        if tmp.empty:
            return go.Figure().update_layout(title=f"{title}：无可用数据")
        # Avoid huge matrix.
        top_i = tmp[index].astype(str).value_counts().head(25).index
        top_c = tmp[columns].astype(str).value_counts().head(25).index
        tmp = tmp[tmp[index].astype(str).isin(top_i) & tmp[columns].astype(str).isin(top_c)]
        pivot = tmp.pivot_table(index=index, columns=columns, values=values, aggfunc="sum", fill_value=0)
        return px.imshow(pivot, aspect="auto", title=title)

    def _keyword_bar(self, keywords, title):
        df = pd.DataFrame(keywords, columns=["keyword", "count"])
        return px.bar(df, x="count", y="keyword", orientation="h", title=title)

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def _valid(self, df: Any) -> bool:
        return isinstance(df, pd.DataFrame) and not df.empty

    def _resolve_monthly_series(self, tables: dict[str, pd.DataFrame]) -> pd.DataFrame | None:
        """Pick or build a monthly GMV series for the trend chart."""
        candidates: list[pd.DataFrame] = []
        if self._valid(tables.get("monthly_sales")):
            candidates.append(tables["monthly_sales"])
        for key in ("state_sales", "state_sales_2017"):
            if self._valid(tables.get(key)):
                df = tables[key]
                if "year_month" in df.columns and "total_gmv" in df.columns:
                    candidates.append(
                        df.groupby("year_month", as_index=False)["total_gmv"].sum().sort_values("year_month")
                    )

        for df in candidates:
            if "year_month" in df.columns and "total_gmv" in df.columns and not df.empty:
                return df.copy()
            norm = self._normalize_monthly_sales(df)
            if norm is not None and not norm.empty:
                return norm
        return self._find_monthly_like_table(tables)

    def _aggregate_state_sales(self, df: pd.DataFrame) -> pd.DataFrame:
        """When state table has month granularity, aggregate to state ranking."""
        if not {"customer_state", "total_gmv"}.issubset(df.columns):
            return df
        if "year_month" in df.columns and df["customer_state"].nunique() < len(df):
            agg = {"total_gmv": "sum"}
            if "total_orders" in df.columns:
                agg["total_orders"] = "sum"
            if "unique_customers" in df.columns:
                agg["unique_customers"] = "sum"
            return df.groupby("customer_state", as_index=False).agg(agg).sort_values("total_gmv", ascending=False)
        return df.sort_values("total_gmv", ascending=False) if "total_gmv" in df.columns else df

    def _clean_df(self, df: pd.DataFrame) -> pd.DataFrame:
        work = df.copy()
        for c in work.columns:
            # Convert numeric-like strings when it is safe.
            if work[c].dtype == object:
                converted = pd.to_numeric(work[c], errors="coerce")
                if converted.notna().mean() > 0.85:
                    work[c] = converted
        return work

    def _find_col(self, df: pd.DataFrame, candidates: list[str]) -> str | None:
        lower = {str(c).lower(): c for c in df.columns}
        for cand in candidates:
            if cand.lower() in lower:
                return lower[cand.lower()]
        for cand in candidates:
            for c in df.columns:
                if cand.lower() in str(c).lower():
                    return c
        return None

    def _find_time_col(self, df: pd.DataFrame) -> str | None:
        for cand in ["year_month", "month", "date", "ds", "order_month", "order_purchase_timestamp", "timestamp"]:
            col = self._find_col(df, [cand])
            if col:
                return col
        for c in df.columns:
            cl = str(c).lower()
            if "date" in cl or "month" in cl or "timestamp" in cl or cl.endswith("_at"):
                return c
        return None

    def _find_metric_col(self, df: pd.DataFrame, prefer: list[str] | None = None) -> str | None:
        prefer = prefer or [
            "total_gmv", "gmv", "total_value", "payment_value", "revenue", "sales",
            "total_orders", "orders", "total_transactions", "unique_customers", "avg_price",
            "avg_delivery_days", "on_time_rate", "negative_rate", "avg_review_score", "avg_score", "count",
            "price", "freight_value",
        ]
        for cand in prefer:
            col = self._find_col(df, [cand])
            if col and pd.api.types.is_numeric_dtype(df[col]):
                return col
        nums = list(df.select_dtypes(include="number").columns)
        return nums[0] if nums else None

    def _prepare_time_series(self, df: pd.DataFrame, time_col: str, value_col: str) -> pd.DataFrame:
        ts = df[[time_col, value_col]].copy()
        if str(time_col).lower() == "year_month":
            ts["date"] = pd.to_datetime(ts[time_col].astype(str).str[:7] + "-01", errors="coerce")
        else:
            ts["date"] = pd.to_datetime(ts[time_col].astype(str), errors="coerce")
            if ts["date"].isna().mean() > 0.5:
                ts["date"] = pd.to_datetime(ts[time_col].astype(str).str[:7] + "-01", errors="coerce")
        ts[value_col] = pd.to_numeric(ts[value_col], errors="coerce")
        ts = ts.dropna(subset=["date", value_col])
        if ts.empty:
            return ts
        # Aggregate if multiple rows per date.
        return ts.groupby("date", as_index=False)[value_col].sum().sort_values("date")

    def _looks_like_brazil_state(self, s: pd.Series) -> bool:
        vals = {str(v).upper() for v in s.dropna().unique()[:50]}
        return bool(vals) and len(vals & set(STATE_COORDS.keys())) >= max(1, min(3, len(vals)))

    def _choose_scatter_cols(self, df: pd.DataFrame, numeric_cols: list[str]) -> tuple[str, str]:
        # Prefer interpretable pairs.
        pairs = [
            ("product_weight_g", "freight_value"),
            ("price", "freight_value"),
            ("avg_delivery_days", "on_time_rate"),
            ("total_orders", "total_gmv"),
            ("total_transactions", "total_value"),
        ]
        lower = {str(c).lower(): c for c in numeric_cols}
        for a, b in pairs:
            if a in lower and b in lower:
                return lower[a], lower[b]
        return numeric_cols[0], numeric_cols[1]

    def _choose_heatmap_categories(self, df: pd.DataFrame, categorical_cols: list[str]) -> tuple[str | None, str | None]:
        preferred = ["payment_type", "payment_installments", "product_category_english", "customer_state", "seller_state", "year_month"]
        chosen = []
        for p in preferred:
            c = self._find_col(df[categorical_cols], [p]) if all(c in df.columns for c in categorical_cols) else None
            if c and c not in chosen and df[c].nunique(dropna=True) <= 40:
                chosen.append(c)
        for c in categorical_cols:
            if c not in chosen and df[c].nunique(dropna=True) <= 40:
                chosen.append(c)
            if len(chosen) >= 2:
                break
        if len(chosen) >= 2:
            return chosen[0], chosen[1]
        return None, None

    def _find_monthly_like_table(self, tables: dict[str, pd.DataFrame]) -> pd.DataFrame | None:
        for df in tables.values():
            norm = self._normalize_monthly_sales(df) if isinstance(df, pd.DataFrame) else None
            if norm is not None and not norm.empty:
                return norm
        return None

    def _normalize_monthly_sales(self, df: pd.DataFrame) -> pd.DataFrame | None:
        if df is None or df.empty:
            return None
        time_col = self._find_time_col(df)
        value_col = self._find_metric_col(df, prefer=["total_gmv", "gmv", "sales", "revenue", "total_value", "payment_value"])
        if not time_col or not value_col:
            return None
        ts = self._prepare_time_series(df, time_col, value_col)
        if ts.empty:
            return None
        ts["year_month"] = ts["date"].dt.strftime("%Y-%m")
        ts = ts.groupby("year_month", as_index=False)[value_col].sum().sort_values("year_month")
        ts = ts.rename(columns={value_col: "total_gmv"})
        return ts[["year_month", "total_gmv"]]

    def _save_figures(self, res: VisualizationResult) -> None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        for name, fig in res.figures.items():
            safe = re.sub(r"[^0-9A-Za-z_\-\u4e00-\u9fff]+", "_", name).strip("_") or "figure"
            html_path = OUTPUT_DIR / f"{safe}.html"
            fig.write_html(str(html_path))
            res.saved_files[name] = str(html_path)
            try:
                png_path = OUTPUT_DIR / f"{safe}.png"
                fig.write_image(str(png_path), scale=2)
                res.saved_files[name + "_png"] = str(png_path)
            except Exception:
                pass
