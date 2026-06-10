from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import json
import re

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import Engine, text
from wordcloud import WordCloud

from config.settings import OUTPUT_DIR
from config.chart_guidance import CHART_PLAN_GUIDANCE
from utils.llm import LLMClient


@dataclass
class VisualizationResult:
    figures: dict[str, Any] = field(default_factory=dict)
    saved_files: dict[str, str] = field(default_factory=dict)
    summary: str = ""
    chart_plan: list[dict] = field(default_factory=list)
    chart_mode: str = ""


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
    """Visualization Agent — 根据问题与 Data Agent 返回的数据，规划并渲染图表。"""

    MAX_AUTO_FIGURES = 8
    VALID_CHART_TYPES = {"line", "bar", "pie", "scatter", "map", "heatmap", "delivery", "wordcloud"}

    def __init__(self, engine: Engine | None = None) -> None:
        self.engine = engine
        self._geo_centroids: dict[str, tuple[float, float]] | None = None
        self.llm = LLMClient()

    def visualize(
        self,
        data_result: Any,
        nlp_result: Any = None,
        forecast_result: Any = None,
        question: str | None = None,
        chart_plan: list[dict] | None = None,
    ) -> VisualizationResult:
        res = VisualizationResult()
        tables = getattr(data_result, "tables", {}) or {}
        if not tables:
            res.summary = "无可用数据，无法生成图表。"
            return res

        if chart_plan:
            plan = chart_plan
            res.chart_mode = "外部传入"
        else:
            plan, res.chart_mode = self._plan_charts(
                question=question or "",
                data_result=data_result,
                nlp_result=nlp_result,
                forecast_result=forecast_result,
            )
        res.chart_plan = plan

        if plan:
            self._execute_chart_plan(
                plan, tables, res,
                forecast_result=forecast_result,
                nlp_result=nlp_result,
            )
        else:
            res.summary = "未能生成图表计划。"

        self._save_figures(res)
        if res.figures:
            res.summary = f"已生成 {len(res.figures)} 个图表（{res.chart_mode}）。"
        elif not res.summary:
            res.summary = "本次未生成图表。"
        return res

    def _plan_charts(
        self,
        question: str,
        data_result: Any,
        nlp_result: Any = None,
        forecast_result: Any = None,
    ) -> tuple[list[dict], str]:
        if self.llm.enabled:
            try:
                plan = self._llm_plan_charts(question, data_result, nlp_result, forecast_result)
                if plan:
                    return plan, "大模型规划"
            except Exception:
                pass
        plan = self._rule_plan_charts(data_result, nlp_result, forecast_result)
        return plan, "规则兜底"

    def _llm_plan_charts(
        self,
        question: str,
        data_result: Any,
        nlp_result: Any = None,
        forecast_result: Any = None,
    ) -> list[dict]:
        tables_ctx = self._build_tables_context(data_result.tables or {})
        intent = getattr(data_result, "intent", "") or ""
        extras = []
        if forecast_result is not None:
            extras.append("已有 Prophet 销售预测结果，月度 line 图会自动叠加预测线，只需规划 1 张月度折线。")
        if nlp_result is not None:
            extras.append("已有 NLP 评论分析结果，可规划 wordcloud 或 reviews 的 bar 图。")
        extra_text = "\n".join(extras) if extras else "无额外 Agent 输出。"

        prompt = f"""
你是 Visualization Agent。Data Agent 已完成 SQL 查询，你只负责决定画哪些图。

用户问题：{question}
分析 intent：{intent}
{extra_text}

{CHART_PLAN_GUIDANCE}

可用数据表（列名 + 样例行）：
{tables_ctx}

请返回严格 JSON：
{{
  "charts": [
    {{"title": "图表标题", "type": "line", "table": "monthly_sales", "x": "year_month", "y": "total_gmv"}}
  ]
}}
只返回 JSON，不要 Markdown。
"""
        content = self.llm.chat(
            [
                {"role": "system", "content": "你是电商 BI 可视化 Agent。只返回合法 JSON。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=2048,
        )
        parsed = LLMClient.extract_json(content)
        return self._parse_chart_plan(parsed.get("charts"))

    @staticmethod
    def _build_tables_context(tables: dict[str, pd.DataFrame]) -> str:
        blocks: list[str] = []
        for name, df in tables.items():
            if not isinstance(df, pd.DataFrame) or df.empty:
                continue
            preview = df.head(3).copy()
            for col in preview.columns:
                if pd.api.types.is_float_dtype(preview[col]):
                    preview[col] = preview[col].round(4)
                else:
                    preview[col] = preview[col].map(
                        lambda v: (str(v)[:80] + "…") if isinstance(v, str) and len(str(v)) > 80 else v
                    )
            blocks.append(
                f"- table={name!r}, rows={len(df)}, columns={list(df.columns)}\n"
                f"  sample={preview.where(pd.notnull(preview), None).to_dict(orient='records')}"
            )
        return "\n".join(blocks) if blocks else "（无表）"

    @staticmethod
    def _parse_chart_plan(charts: Any) -> list[dict]:
        if not isinstance(charts, list):
            return []
        out: list[dict] = []
        for c in charts:
            if not isinstance(c, dict):
                continue
            c_type = str(c.get("type", "")).strip().lower()
            c_table = str(c.get("table", "")).strip()
            c_title = str(c.get("title", "")).strip()
            if c_type not in VisualizationAgent.VALID_CHART_TYPES or not c_title:
                continue
            if c_type != "wordcloud" and not c_table:
                continue
            entry: dict[str, str] = {"title": c_title, "type": c_type, "table": c_table}
            for key in ("x", "y", "orientation", "variant"):
                if c.get(key):
                    entry[key] = str(c[key]).strip()
            out.append(entry)
        return out

    def _rule_plan_charts(
        self,
        data_result: Any,
        nlp_result: Any = None,
        forecast_result: Any = None,
    ) -> list[dict]:
        """无 LLM 时的图表计划兜底（与原 Data Agent chart_plan 对齐）。"""
        tables = data_result.tables or {}
        intent = getattr(data_result, "intent", "") or ""
        plan: list[dict] = []

        def has(name: str) -> bool:
            df = tables.get(name)
            return isinstance(df, pd.DataFrame) and not df.empty

        if has("monthly_sales"):
            title = "GMV 趋势与预测" if (forecast_result or intent == "forecast") else "月度 GMV 趋势"
            plan.append({"title": title, "type": "line", "table": "monthly_sales", "x": "year_month", "y": "total_gmv"})
        if has("state_sales_2017"):
            plan.append({"title": "2017 各州 GMV 排名", "type": "bar", "table": "state_sales_2017", "x": "customer_state", "y": "total_gmv"})
        elif has("state_sales"):
            plan.append({"title": "各州 GMV 排名", "type": "bar", "table": "state_sales", "x": "customer_state", "y": "total_gmv"})
            plan.append({"title": "各州销售额气泡图", "type": "map", "table": "state_sales", "y": "total_gmv"})
        if has("delivery_by_state"):
            plan.append({"title": "各州配送时长与准时率", "type": "delivery", "table": "delivery_by_state"})
        if has("payment_dist"):
            plan.append({"title": "支付方式分布", "type": "pie", "table": "payment_dist", "x": "payment_type", "y": "total_transactions"})
        if has("top_categories"):
            plan.append({"title": "Top 品类 GMV", "type": "bar", "table": "top_categories", "x": "total_gmv", "y": "product_category_english", "orientation": "h"})
        if has("low_score_sellers"):
            plan.append({"title": "低评分卖家 Top 20", "type": "bar", "table": "low_score_sellers", "x": "avg_review_score", "y": "seller_id", "orientation": "h"})
        if has("weight_freight"):
            plan.append({"title": "商品重量与运费关系", "type": "scatter", "table": "weight_freight", "x": "product_weight_g", "y": "freight_value"})
        if has("reviews") or nlp_result is not None:
            plan.append({"title": "差评品类 Top10", "type": "bar", "table": "reviews", "orientation": "h"})
            plan.append({"title": "差评关键词词云", "type": "wordcloud", "table": "", "variant": "negative"})
            if len(plan) < self.MAX_AUTO_FIGURES:
                plan.append({"title": "好评关键词词云", "type": "wordcloud", "table": "", "variant": "positive"})
        return plan[: self.MAX_AUTO_FIGURES]

    # ------------------------------------------------------------------
    # LLM chart plan execution
    # ------------------------------------------------------------------

    def _execute_chart_plan(
        self,
        chart_plan: list[dict],
        tables: dict[str, pd.DataFrame],
        res: VisualizationResult,
        forecast_result: Any = None,
        nlp_result: Any = None,
    ) -> None:
        seen_line_monthly = False
        for chart in chart_plan:
            if len(res.figures) >= self.MAX_AUTO_FIGURES:
                break
            chart_type = str(chart.get("type", "")).strip().lower()
            title = str(chart.get("title", "")).strip()
            table_name = str(chart.get("table", "")).strip()
            if not title or not chart_type:
                continue

            fig = None
            try:
                if chart_type == "wordcloud":
                    fig = self._exec_wordcloud(chart, nlp_result)
                elif chart_type == "delivery":
                    df = tables.get(table_name)
                    if isinstance(df, pd.DataFrame) and not df.empty:
                        fig = self._delivery_fig(df)
                        if title:
                            fig.update_layout(title=title)
                elif chart_type == "line" and (
                    table_name == "monthly_sales" or self._is_monthly_gmv_df(tables.get(table_name, pd.DataFrame()))
                ):
                    if seen_line_monthly:
                        continue
                    df = tables.get(table_name)
                    if isinstance(df, pd.DataFrame) and not df.empty:
                        fig = self._exec_line(chart, df, forecast_result=forecast_result, table_name=table_name)
                        seen_line_monthly = True
                elif chart_type == "bar" and table_name == "reviews":
                    fig = self._exec_review_category_bar(nlp_result, title)
                else:
                    df = tables.get(table_name)
                    if not isinstance(df, pd.DataFrame) or df.empty:
                        continue
                    if chart_type == "line":
                        fig = self._exec_line(chart, df, forecast_result=forecast_result, table_name=table_name)
                    elif chart_type == "bar":
                        fig = self._exec_bar(chart, df)
                    elif chart_type == "pie":
                        fig = self._exec_pie(chart, df)
                    elif chart_type == "scatter":
                        fig = self._exec_scatter(chart, df)
                    elif chart_type == "map":
                        fig = self._exec_map(chart, df)
                    elif chart_type == "heatmap":
                        fig = self._exec_heatmap(chart, df)
            except Exception:
                continue

            if fig is not None and title not in res.figures:
                res.figures[title] = fig

    def _exec_review_category_bar(self, nlp_result: Any, title: str):
        if nlp_result is None:
            return None
        df = getattr(nlp_result, "top_negative_categories", pd.DataFrame())
        if not isinstance(df, pd.DataFrame) or df.empty:
            return None
        if not {"negative_rate", "product_category_english"}.issubset(df.columns):
            return None
        return px.bar(
            df, x="negative_rate", y="product_category_english",
            orientation="h", title=title or "Top 10 差评品类",
        )

    def _exec_wordcloud(self, chart: dict, nlp_result: Any):
        if nlp_result is None:
            return None
        variant = str(chart.get("variant", "")).strip().lower()
        title = str(chart.get("title", "")).strip()
        if variant == "positive" or "好评" in title:
            keywords = getattr(nlp_result, "positive_keywords", None)
            default_title = "好评关键词词云"
            cmap = "Blues"
        else:
            keywords = getattr(nlp_result, "negative_keywords", None)
            default_title = "差评关键词词云"
            cmap = "Reds"
        return self._keyword_wordcloud(keywords, title or default_title, colormap=cmap)

    def _exec_line(
        self,
        chart: dict,
        df: pd.DataFrame,
        forecast_result: Any = None,
        table_name: str = "",
    ):
        title = chart.get("title", "时间趋势")
        if table_name == "monthly_sales" or self._is_monthly_gmv_df(df):
            norm = self._normalize_monthly_sales(df)
            if norm is not None and not norm.empty:
                fig = self._monthly_sales_fig(norm, forecast_result)
                if title:
                    fig.update_layout(title=title)
                return fig
        x = chart.get("x") or self._find_time_col(df)
        y = chart.get("y") or self._find_metric_col(df)
        if not x or not y:
            return None
        ts = self._prepare_time_series(df, x, y)
        if ts.empty:
            return None
        return px.line(ts, x="date", y=y, markers=True, title=title)

    @staticmethod
    def _is_monthly_gmv_df(df: pd.DataFrame) -> bool:
        cols = {str(c).lower() for c in df.columns}
        return "year_month" in cols and bool(cols & {"total_gmv", "gmv", "sales", "revenue"})

    def _exec_bar(self, chart: dict, df: pd.DataFrame):
        title = chart.get("title", "柱状图")
        x = chart.get("x") or self._find_col(df, ["customer_state", "product_category_english", "payment_type", "seller_id"])
        y = chart.get("y") or self._find_metric_col(df)
        orientation = chart.get("orientation", "v")
        if not x or not y:
            return None
        agg = df.groupby(x, as_index=False)[y].sum().sort_values(y, ascending=False).head(20)
        if orientation == "h":
            return px.bar(agg, x=y, y=x, orientation="h", title=title)
        return px.bar(agg, x=x, y=y, title=title)

    def _exec_pie(self, chart: dict, df: pd.DataFrame):
        title = chart.get("title", "饼图")
        names = chart.get("x") or self._find_col(df, ["payment_type", "product_category_english", "customer_state"])
        values = chart.get("y") or self._find_metric_col(df)
        if not names or not values:
            return None
        agg = df.groupby(names, as_index=False)[values].sum().sort_values(values, ascending=False).head(10)
        return px.pie(agg, names=names, values=values, title=title)

    def _exec_scatter(self, chart: dict, df: pd.DataFrame):
        title = chart.get("title", "散点图")
        x = chart.get("x") or self._find_metric_col(df)
        y = chart.get("y") or (self._find_metric_col(df, prefer=["freight_value", "total_gmv", "avg_delivery_days", "on_time_rate", "price", "total_value"]) or self._find_metric_col(df))
        if not x or not y:
            return None
        color_col = self._find_col(df, ["delivery_status", "customer_state", "payment_type", "product_category_english"])
        sample = df[[x, y] + ([color_col] if color_col else [])].dropna().head(5000)
        kwargs = {"color": color_col} if color_col else {}
        return px.scatter(sample, x=x, y=y, title=title, **kwargs)

    def _exec_map(self, chart: dict, df: pd.DataFrame):
        title = chart.get("title", "地理分布")
        state_col = self._find_col(df, ["customer_state", "seller_state", "state"])
        value_col = chart.get("y") or self._find_metric_col(df)
        if not state_col:
            return None
        work = df.copy()
        work = work.rename(columns={state_col: "customer_state"})
        if value_col and value_col != "customer_state":
            work = work.rename(columns={value_col: "total_gmv"})
        if "total_gmv" not in work.columns:
            numeric = [c for c in work.select_dtypes(include="number").columns if c != "customer_state"]
            if numeric:
                work = work.rename(columns={numeric[0]: "total_gmv"})
        if "total_orders" not in work.columns:
            work["total_orders"] = 1
        return self._state_map_fig(work)

    def _exec_heatmap(self, chart: dict, df: pd.DataFrame):
        title = chart.get("title", "热力图")
        x = chart.get("x")
        y = chart.get("y")
        z = chart.get("z") or self._find_metric_col(df)
        if not x:
            cats = [c for c in df.columns if c not in {str(chart.get("y")), str(chart.get("z"))} and df[c].nunique(dropna=True) <= 40]
            if len(cats) >= 2:
                x, y = cats[0], cats[1]
        if not x or not y or not z:
            return None
        return self._heatmap(df, index=x, columns=y, values=z, title=title)

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
            date_col = "ds" if "ds" in fc.columns else "date"
            y_col = "yhat" if "yhat" in fc.columns else "forecast_gmv"
            lower_col = "yhat_lower" if "yhat_lower" in fc.columns else "lower"
            upper_col = "yhat_upper" if "yhat_upper" in fc.columns else "upper"
            fig.add_trace(
                go.Scatter(
                    x=fc[date_col], y=fc[y_col],
                    mode="lines+markers", name="未来 6 周预测（Prophet）",
                )
            )
            if {upper_col, lower_col}.issubset(fc.columns):
                fig.add_trace(
                    go.Scatter(
                        x=fc[date_col], y=fc[upper_col],
                        mode="lines", name="95% 置信区间上界",
                        line=dict(width=0), showlegend=False,
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=fc[date_col], y=fc[lower_col],
                        mode="lines", name="95% 置信区间",
                        fill="tonexty", line=dict(width=0),
                    )
                )
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
        if {"lat", "lon"}.issubset(map_df.columns):
            map_df["lat"] = pd.to_numeric(map_df["lat"], errors="coerce")
            map_df["lon"] = pd.to_numeric(map_df["lon"], errors="coerce")
        else:
            centroids = self._get_geo_centroids()
            map_df["lat"] = map_df["customer_state"].map(
                lambda s: centroids.get(str(s).upper(), (None, None))[0]
            )
            map_df["lon"] = map_df["customer_state"].map(
                lambda s: centroids.get(str(s).upper(), (None, None))[1]
            )
        map_df = map_df.dropna(subset=["lat", "lon"])
        if map_df.empty:
            return go.Figure().update_layout(title="巴西各州气泡图：无可用经纬度")
        fig = px.scatter_geo(
            map_df, lat="lat", lon="lon", size="total_gmv", color="total_orders",
            hover_name="customer_state", scope="south america",
            title="巴西各州销售额/订单量气泡图（geolocation 坐标）",
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

    def _keyword_wordcloud(self, keywords, title: str, colormap: str = "Blues"):
        freq: dict[str, int] = {}
        for item in keywords or []:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                word, count = str(item[0]), int(item[1])
            elif isinstance(item, dict):
                word = str(item.get("word") or item.get("keyword") or "")
                count = int(item.get("count") or 0)
            else:
                continue
            if word and count > 0:
                freq[word] = freq.get(word, 0) + count
        if not freq:
            return None
        try:
            wc = WordCloud(
                width=900,
                height=450,
                background_color="white",
                colormap=colormap,
                max_words=60,
                prefer_horizontal=0.7,
            ).generate_from_frequencies(freq)
            arr = wc.to_array()
            fig = go.Figure(data=go.Image(z=arr))
            h, w = arr.shape[0], arr.shape[1]
            fig.update_layout(
                title=title,
                margin=dict(l=10, r=10, t=50, b=10),
                xaxis=dict(visible=False, range=[0, w]),
                yaxis=dict(visible=False, range=[h, 0], scaleanchor="x", scaleratio=1),
            )
            return fig
        except Exception:
            return self._keyword_bar(list(freq.items()), title)

    def _get_geo_centroids(self) -> dict[str, tuple[float, float]]:
        if self._geo_centroids is not None:
            return self._geo_centroids
        centroids = dict(STATE_COORDS)
        if self.engine is not None:
            try:
                geo_df = pd.read_sql(
                    text(
                        """
                        SELECT geolocation_state AS st,
                               AVG(geolocation_lat) AS lat,
                               AVG(geolocation_lng) AS lon
                        FROM geolocation
                        WHERE geolocation_state IS NOT NULL
                        GROUP BY geolocation_state
                        """
                    ),
                    self.engine,
                )
                for row in geo_df.itertuples(index=False):
                    st = str(row.st).upper()
                    if pd.notna(row.lat) and pd.notna(row.lon):
                        centroids[st] = (float(row.lat), float(row.lon))
            except Exception:
                pass
        self._geo_centroids = centroids
        return self._geo_centroids

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def _valid(self, df: Any) -> bool:
        return isinstance(df, pd.DataFrame) and not df.empty

    def _resolve_monthly_series(
        self, tables: dict[str, pd.DataFrame]
    ) -> tuple[pd.DataFrame | None, set[str]]:
        """Pick or build a monthly GMV series for the trend chart.

        Returns (df, source_keys) where source_keys lists which table names
        were consumed to produce the series, so they can be marked used.
        """
        consumed: set[str] = set()
        candidates: list[pd.DataFrame] = []
        if self._valid(tables.get("monthly_sales")):
            candidates.append(tables["monthly_sales"])
            consumed.add("monthly_sales")
        for key in ("state_sales", "state_sales_2017"):
            if self._valid(tables.get(key)):
                df = tables[key]
                if "year_month" in df.columns and "total_gmv" in df.columns:
                    candidates.append(
                        df.groupby("year_month", as_index=False)["total_gmv"].sum().sort_values("year_month")
                    )
                    consumed.add(key)

        for df in candidates:
            if "year_month" in df.columns and "total_gmv" in df.columns and not df.empty:
                return df.copy(), consumed
            norm = self._normalize_monthly_sales(df)
            if norm is not None and not norm.empty:
                return norm, consumed
        found = self._find_monthly_like_table(tables)
        if found is not None:
            consumed.add("_monthly_like")
        return found, consumed

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
