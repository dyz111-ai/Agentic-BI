from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from config.settings import OUTPUT_DIR
from utils.llm import LLMClient


@dataclass
class VisualizationResult:
    figures: dict[str, Any] = field(default_factory=dict)
    saved_files: dict[str, str] = field(default_factory=dict)
    summary: str = ""
    chart_plan: dict[str, Any] = field(default_factory=dict)


STATE_COORDS = {
    "SP": (-23.55, -46.63), "RJ": (-22.90, -43.20), "MG": (-19.92, -43.94), "BA": (-12.97, -38.50),
    "PE": (-8.05, -34.88), "PR": (-25.43, -49.27), "RS": (-30.03, -51.23), "SC": (-27.59, -48.55),
    "CE": (-3.73, -38.52), "GO": (-16.68, -49.25), "ES": (-20.31, -40.31), "DF": (-15.79, -47.88),
    "AM": (-3.10, -60.02), "PA": (-1.45, -48.49), "MA": (-2.53, -44.30), "PB": (-7.12, -34.86),
    "RN": (-5.79, -35.21), "AL": (-9.65, -35.73), "SE": (-10.91, -37.07), "PI": (-5.09, -42.80),
    "MT": (-15.60, -56.10), "MS": (-20.45, -54.62), "RO": (-8.76, -63.90), "AC": (-9.97, -67.82),
    "RR": (2.82, -60.67), "AP": (0.03, -51.05), "TO": (-10.18, -48.33),
}

ALLOWED_CHART_TYPES = {"line", "bar", "bar_h", "pie", "scatter", "heatmap", "geo_scatter", "keyword_bar"}


class VisualizationAgent:
    """LLM-only chart planner. No rule/template fallback."""

    MAX_CHARTS = 6

    def __init__(self):
        self.llm = LLMClient()

    def visualize(
        self,
        data_result: Any,
        nlp_result: Any = None,
        forecast_result: Any = None,
        question: str | None = None,
    ) -> VisualizationResult:
        self.llm.require_enabled()
        res = VisualizationResult()
        tables = getattr(data_result, "tables", {}) or {}
        if not tables:
            raise ValueError("没有可用于可视化的数据表")

        plan = self._ask_llm_for_chart_plan(
            question=question or "",
            tables=tables,
            nlp_result=nlp_result,
            forecast_result=forecast_result,
        )
        res.chart_plan = plan
        charts = plan.get("charts") or []
        if not charts:
            res.summary = "LLM 未返回任何图表规格，未生成图表。"
            return res

        for spec in charts[: self.MAX_CHARTS]:
            title = str(spec.get("title") or "").strip()
            if not title:
                continue
            fig = self._render_chart(spec, tables, nlp_result, forecast_result)
            if fig is not None:
                res.figures[title] = fig

        self._save_figures(res)
        if res.figures:
            res.summary = f"LLM 已规划 {len(charts)} 个图表，成功渲染 {len(res.figures)} 个。"
        else:
            res.summary = "LLM 返回了图表规格，但均无法根据当前数据渲染。"
        return res

    def _ask_llm_for_chart_plan(
        self,
        question: str,
        tables: dict[str, pd.DataFrame],
        nlp_result: Any,
        forecast_result: Any,
    ) -> dict:
        table_meta = self._tables_metadata(tables, nlp_result)
        forecast_hint = ""
        if forecast_result is not None and not getattr(forecast_result, "forecast_df", pd.DataFrame()).empty:
            forecast_hint = "存在预测结果 forecast_df（含 date, forecast_gmv, upper, lower），可在折线图上设置 overlay_forecast=true。"

        prompt = f"""
你是 Agentic BI 的 Visualization Agent。根据用户问题和查询结果，规划 1-{self.MAX_CHARTS} 个 Plotly 图表。

用户问题：{question}

可用数据表（含列名、类型、样本行）：
{json.dumps(table_meta, ensure_ascii=False, default=str, indent=2)}

{forecast_hint}

规则：
1. 只返回严格 JSON，不要 Markdown。
2. charts 中每个元素必须包含：title, type, table, x, y（pie/heatmap 可省略 y 或改用 names/values/index/columns）。
3. type 只能是：line, bar, bar_h, pie, scatter, heatmap, geo_scatter, keyword_bar。
4. table 必须是上面列出的表名之一（NLP 表以 nlp_ 开头）。
5. x/y/names/values/index/columns 必须是该表真实存在的列名。
6. 时间序列优先 line；州/地区排名可用 bar 或 geo_scatter（x 为州列，y 为数值列）。
7. 支付方式占比可用 pie；两类别交叉可用 heatmap（index, columns, values）。
8. keyword_bar 用于 nlp_negative_keywords / nlp_positive_keywords（x=keyword, y=count）。
9. 若无法从数据得出合理图表，返回空 charts 数组，不要编造列名。
10. 折线图若需叠加预测，设置 overlay_forecast=true。

返回格式：
{{
  "reason": "为何选择这些图表",
  "charts": [
    {{
      "title": "月度 GMV 趋势",
      "type": "line",
      "table": "monthly_trend",
      "x": "year_month",
      "y": "total_gmv",
      "overlay_forecast": false
    }}
  ]
}}
"""
        messages = [
            {"role": "system", "content": "你是严谨的 BI 可视化规划 Agent，只返回合法 JSON。"},
            {"role": "user", "content": prompt},
        ]
        last_error = "LLM 未返回内容"
        for max_tokens in (4096, 8192):
            content = self.llm.chat(messages=messages, temperature=0.0, max_tokens=max_tokens)
            try:
                return LLMClient.extract_json(content)
            except ValueError as exc:
                last_error = str(exc)
        raise ValueError(last_error)

    def _tables_metadata(self, tables: dict[str, pd.DataFrame], nlp_result: Any) -> dict[str, Any]:
        meta: dict[str, Any] = {}
        for name, df in tables.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                meta[name] = {
                    "columns": {str(c): str(df[c].dtype) for c in df.columns},
                    "row_count": len(df),
                    "sample_rows": self._sample_rows(df),
                }
        if nlp_result is not None:
            neg = getattr(nlp_result, "negative_keywords", None) or []
            pos = getattr(nlp_result, "positive_keywords", None) or []
            if neg:
                meta["nlp_negative_keywords"] = {
                    "columns": {"keyword": "object", "count": "int64"},
                    "row_count": len(neg),
                    "sample_rows": [{"keyword": str(k), "count": int(c)} for k, c in neg[:15]],
                }
            if pos:
                meta["nlp_positive_keywords"] = {
                    "columns": {"keyword": "object", "count": "int64"},
                    "row_count": len(pos),
                    "sample_rows": [{"keyword": str(k), "count": int(c)} for k, c in pos[:15]],
                }
            top_neg = getattr(nlp_result, "top_negative_categories", pd.DataFrame())
            if isinstance(top_neg, pd.DataFrame) and not top_neg.empty:
                meta["nlp_top_negative_categories"] = {
                    "columns": {str(c): str(top_neg[c].dtype) for c in top_neg.columns},
                    "row_count": len(top_neg),
                    "sample_rows": self._sample_rows(top_neg),
                }
        return meta

    @staticmethod
    def _sample_rows(df: pd.DataFrame, max_rows: int = 5) -> list[dict[str, Any]]:
        work = df.head(max_rows).copy()
        for col in work.columns:
            if pd.api.types.is_float_dtype(work[col]):
                work[col] = work[col].round(4)
        work = work.where(pd.notnull(work), None)
        return work.to_dict(orient="records")

    def _render_chart(
        self,
        spec: dict,
        tables: dict[str, pd.DataFrame],
        nlp_result: Any,
        forecast_result: Any,
    ) -> go.Figure | None:
        chart_type = str(spec.get("type") or "").strip().lower()
        if chart_type not in ALLOWED_CHART_TYPES:
            return None

        table_name = str(spec.get("table") or "").strip()
        df = self._resolve_table(table_name, tables, nlp_result)
        if df is None or df.empty:
            return None

        x = spec.get("x")
        y = spec.get("y")
        title = str(spec.get("title") or "图表")

        if chart_type == "line":
            return self._render_line(df, x, y, title, spec, forecast_result)
        if chart_type == "bar":
            return self._render_bar(df, x, y, title, horizontal=False)
        if chart_type == "bar_h":
            return self._render_bar(df, x, y, title, horizontal=True)
        if chart_type == "pie":
            names = spec.get("names") or x
            values = spec.get("values") or y
            if names not in df.columns or values not in df.columns:
                return None
            return px.pie(df, names=names, values=values, title=title)
        if chart_type == "scatter":
            color = spec.get("color")
            if x not in df.columns or y not in df.columns:
                return None
            kwargs = {"color": color} if color and color in df.columns else {}
            return px.scatter(df, x=x, y=y, title=title, **kwargs)
        if chart_type == "heatmap":
            index = spec.get("index")
            columns = spec.get("columns")
            values = spec.get("values")
            if not all(c in df.columns for c in [index, columns, values]):
                return None
            return self._heatmap(df, index=index, columns=columns, values=values, title=title)
        if chart_type == "geo_scatter":
            return self._render_geo(df, x, y, title)
        if chart_type == "keyword_bar":
            return self._render_keyword_bar(df, x, y, title)
        return None

    def _resolve_table(self, name: str, tables: dict[str, pd.DataFrame], nlp_result: Any) -> pd.DataFrame | None:
        if name in tables and isinstance(tables[name], pd.DataFrame):
            return tables[name].copy()
        if name == "nlp_negative_keywords" and nlp_result is not None:
            kws = getattr(nlp_result, "negative_keywords", None) or []
            if kws:
                return pd.DataFrame([{"keyword": str(k), "count": int(c)} for k, c in kws])
        if name == "nlp_positive_keywords" and nlp_result is not None:
            kws = getattr(nlp_result, "positive_keywords", None) or []
            if kws:
                return pd.DataFrame([{"keyword": str(k), "count": int(c)} for k, c in kws])
        if name == "nlp_top_negative_categories" and nlp_result is not None:
            df = getattr(nlp_result, "top_negative_categories", pd.DataFrame())
            return df.copy() if isinstance(df, pd.DataFrame) else None
        return None

    def _render_line(
        self,
        df: pd.DataFrame,
        x: str,
        y: str,
        title: str,
        spec: dict,
        forecast_result: Any,
    ) -> go.Figure | None:
        if x not in df.columns or y not in df.columns:
            return None
        work = df[[x, y]].copy()
        work[y] = pd.to_numeric(work[y], errors="coerce")
        if str(x).lower() in {"year_month", "month"}:
            work["date"] = pd.to_datetime(work[x].astype(str).str[:7] + "-01", errors="coerce")
        else:
            work["date"] = pd.to_datetime(work[x], errors="coerce")
        work = work.dropna(subset=["date", y]).sort_values("date")
        if work.empty:
            return None

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=work["date"], y=work[y], mode="lines+markers", name=str(y)))
        if spec.get("overlay_forecast") and forecast_result is not None:
            fc = getattr(forecast_result, "forecast_df", pd.DataFrame())
            if isinstance(fc, pd.DataFrame) and not fc.empty and "date" in fc.columns and "forecast_gmv" in fc.columns:
                fig.add_trace(go.Scatter(x=fc["date"], y=fc["forecast_gmv"], mode="lines+markers", name="预测"))
                if {"upper", "lower"}.issubset(fc.columns):
                    fig.add_trace(go.Scatter(x=fc["date"], y=fc["upper"], mode="lines", line=dict(width=0), showlegend=False))
                    fig.add_trace(go.Scatter(x=fc["date"], y=fc["lower"], mode="lines", fill="tonexty", name="置信区间", line=dict(width=0)))
        fig.update_layout(title=title, xaxis_title=str(x), yaxis_title=str(y))
        return fig

    @staticmethod
    def _render_bar(df: pd.DataFrame, x: str, y: str, title: str, horizontal: bool) -> go.Figure | None:
        if x not in df.columns or y not in df.columns:
            return None
        if horizontal:
            return px.bar(df, x=x, y=y, orientation="h", title=title)
        return px.bar(df, x=x, y=y, title=title)

    def _render_geo(self, df: pd.DataFrame, state_col: str, value_col: str, title: str) -> go.Figure | None:
        if state_col not in df.columns or value_col not in df.columns:
            return None
        map_df = df.copy()
        map_df["lat"] = map_df[state_col].map(lambda s: STATE_COORDS.get(str(s).upper(), (None, None))[0])
        map_df["lon"] = map_df[state_col].map(lambda s: STATE_COORDS.get(str(s).upper(), (None, None))[1])
        map_df = map_df.dropna(subset=["lat", "lon"])
        if map_df.empty:
            return None
        size_col = value_col if pd.api.types.is_numeric_dtype(map_df[value_col]) else None
        fig = px.scatter_geo(
            map_df,
            lat="lat",
            lon="lon",
            size=size_col,
            color=value_col if size_col else state_col,
            hover_name=state_col,
            scope="south america",
            title=title,
        )
        fig.update_geos(center=dict(lat=-14, lon=-52), projection_scale=3.2)
        return fig

    @staticmethod
    def _render_keyword_bar(df: pd.DataFrame, x: str, y: str, title: str) -> go.Figure | None:
        x = x or "keyword"
        y = y or "count"
        if x not in df.columns or y not in df.columns:
            return None
        ordered = df.sort_values(y, ascending=True).tail(20)
        return px.bar(ordered, x=y, y=x, orientation="h", title=title)

    @staticmethod
    def _heatmap(df: pd.DataFrame, index: str, columns: str, values: str, title: str) -> go.Figure:
        tmp = df[[index, columns, values]].copy().dropna()
        pivot = tmp.pivot_table(index=index, columns=columns, values=values, aggfunc="sum", fill_value=0)
        fig = go.Figure(data=go.Heatmap(z=pivot.values, x=list(pivot.columns), y=list(pivot.index), colorscale="Blues"))
        fig.update_layout(title=title, xaxis_title=str(columns), yaxis_title=str(index))
        return fig

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
