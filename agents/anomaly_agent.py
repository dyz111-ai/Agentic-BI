from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import pandas as pd
import numpy as np
from sqlalchemy.engine import Engine

from utils.llm import LLMClient
from config.prompts import ANOMALY_AGENT_PROMPT


@dataclass
class AnomalyAlert:
    dimension: str           # "state_sales" / "delivery" / "review" / "gmv"
    entity: str              # 州名 / 品类名 / "全局"
    metric: str              # "订单量" / "GMV" / "配送天数" / "准时率" / "平均评分"
    current_value: float
    baseline_value: float
    change_pct: float        # 百分比变化，正=上升，负=下降
    severity: str            # "critical" / "warning" / "info"
    description: str
    suggestion: str


@dataclass
class AnomalyResult:
    alerts: list[AnomalyAlert] = field(default_factory=list)
    summary: str = ""
    llm_analysis: str = ""
    alert_count: int = 0
    critical_count: int = 0
    warning_count: int = 0
    info_count: int = 0
    scan_dimensions: list[str] = field(default_factory=list)
    scan_period: str = ""
    baseline_period: str = ""

    @property
    def has_alerts(self) -> bool:
        return self.alert_count > 0


class AnomalyDetectionAgent:
    """Anomaly Detection Agent.

    自动扫描近期数据并预警多维度异常：
    1. 州级：订单量/GMV 环比骤降
    2. 配送：平均配送天数大幅度上升、准时率下降
    3. 评分：平均评分环比下降
    4. 全局：GMV 或总订单量环比异常

    方法：使用 IQR（四分位距）检测离群变化幅度。
    规则兜底：直接输出统计表格和异常定位。
    """

    RECENT_MONTHS = 3          # 近期窗口（月数）
    IQR_MULTIPLIER = 1.5       # IQR 倍数阈值

    SEVERITY_THRESHOLDS = {
        "订单量下降": (0.30, 0.50),    # (critical, warning)，超过即 critical，介于中间为 warning
        "GMV下降": (0.30, 0.50),
        "配送天数上升": (0.25, 0.40),
        "准时率下降": (0.15, 0.25),
        "评分下降": (0.08, 0.15),
    }

    MIN_ORDERS_FOR_STATE = 5   # 州级最近3个月至少这么多单才纳入

    def __init__(self, engine: Engine | None = None):
        self.engine = engine
        self.llm = LLMClient()

    def scan(self, question: str = "") -> AnomalyResult:
        if self.engine is None:
            return AnomalyResult(
                summary="数据库引擎不可用，无法执行异常扫描。",
                scan_dimensions=["state_sales", "delivery", "review"],
            )

        try:
            state_df = self._fetch_state_sales()
            delivery_df = self._fetch_delivery()
            seller_df = self._fetch_seller_review()
        except Exception as exc:
            return AnomalyResult(
                summary=f"数据查询失败：{exc}。请确认预聚合表已刷新。",
                scan_dimensions=["state_sales", "delivery", "review"],
            )

        alerts: list[AnomalyAlert] = []

        if state_df is not None and not state_df.empty:
            alerts.extend(self._detect_state_anomalies(state_df))

        if delivery_df is not None and not delivery_df.empty:
            alerts.extend(self._detect_delivery_anomalies(delivery_df))

        if seller_df is not None and not seller_df.empty:
            alerts.extend(self._detect_review_anomalies(seller_df))

        alerts = self._deduplicate_and_sort(alerts)

        critical = [a for a in alerts if a.severity == "critical"]
        warnings = [a for a in alerts if a.severity == "warning"]
        infos = [a for a in alerts if a.severity == "info"]

        if not alerts:
            summary = "未检测到显著异常，近期各维度核心指标运行平稳。"
        else:
            parts: list[str] = []
            if critical:
                parts.append(f"🚨 **{len(critical)} 条严重预警**")
                for a in critical:
                    parts.append(f"- [{a.entity}] {a.description}")
            if warnings:
                parts.append(f"\n⚠️ **{len(warnings)} 条警告**")
                for a in warnings:
                    parts.append(f"- [{a.entity}] {a.description}")
            if infos:
                parts.append(f"\nℹ️ **{len(infos)} 条提示**")
                for a in infos[:3]:
                    parts.append(f"- [{a.entity}] {a.description}")
            summary = "\n".join(parts)

        llm_analysis = ""
        if self.llm.enabled and alerts:
            llm_analysis = self._generate_llm_analysis(question, alerts, summary)

        return AnomalyResult(
            alerts=alerts,
            summary=summary + ("\n\n" + llm_analysis if llm_analysis else ""),
            llm_analysis=llm_analysis,
            alert_count=len(alerts),
            critical_count=len(critical),
            warning_count=len(warnings),
            info_count=len(infos),
            scan_dimensions=["州级销售", "配送时效", "卖家评分"],
            scan_period=f"最近 {self.RECENT_MONTHS} 个月",
            baseline_period=f"之前各月基线",
        )

    # ------------------------------------------------------------------
    # Data fetching
    # ------------------------------------------------------------------

    def _fetch_state_sales(self) -> pd.DataFrame | None:
        try:
            from utils.db import read_df
            sql = """
                SELECT year_month, customer_state,
                       SUM(total_gmv) AS total_gmv,
                       SUM(total_orders) AS total_orders
                FROM mv_state_sales
                GROUP BY year_month, customer_state
                ORDER BY year_month, total_orders DESC
            """
            return read_df(sql, engine=self.engine)
        except Exception:
            return None

    def _fetch_delivery(self) -> pd.DataFrame | None:
        try:
            from utils.db import read_df
            sql = """
                SELECT year_month, customer_state,
                       ROUND(AVG(avg_delivery_days), 2) AS avg_delivery_days,
                       ROUND(AVG(on_time_rate), 4) AS on_time_rate,
                       SUM(delayed_orders) AS delayed_orders,
                       SUM(total_orders) AS total_orders
                FROM mv_delivery_perf
                GROUP BY year_month, customer_state
                ORDER BY year_month, total_orders DESC
            """
            return read_df(sql, engine=self.engine)
        except Exception:
            return None

    def _fetch_seller_review(self) -> pd.DataFrame | None:
        try:
            from utils.db import read_df
            sql = """
                SELECT year_month, seller_id,
                       ROUND(AVG(avg_review_score), 2) AS avg_review_score,
                       SUM(total_orders) AS total_orders
                FROM mv_seller_perf
                GROUP BY year_month, seller_id
                ORDER BY year_month, total_orders DESC
            """
            return read_df(sql, engine=self.engine)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Detection methods
    # ------------------------------------------------------------------

    def _detect_state_anomalies(self, df: pd.DataFrame) -> list[AnomalyAlert]:
        alerts: list[AnomalyAlert] = []
        df = df.copy()
        df["year_month"] = df["year_month"].astype(str)
        all_months = sorted(df["year_month"].unique())
        if len(all_months) < self.RECENT_MONTHS + 1:
            return alerts

        recent_months = all_months[-self.RECENT_MONTHS:]
        baseline_months = all_months[: -self.RECENT_MONTHS]

        recent = df[df["year_month"].isin(recent_months)]
        baseline = df[df["year_month"].isin(baseline_months)]

        recent_agg = recent.groupby("customer_state").agg(
            recent_orders=("total_orders", "sum"),
            recent_gmv=("total_gmv", "sum"),
        ).reset_index()

        baseline_agg = baseline.groupby("customer_state").agg(
            baseline_orders=("total_orders", "mean"),   # 按月均
            baseline_gmv=("total_gmv", "mean"),
        ).reset_index()

        merged = recent_agg.merge(baseline_agg, on="customer_state", how="inner")
        merged = merged[merged["recent_orders"] >= self.MIN_ORDERS_FOR_STATE]

        if merged.empty:
            return alerts

        merged["orders_change_pct"] = (merged["recent_orders"] - merged["baseline_orders"] * self.RECENT_MONTHS) / np.maximum(merged["baseline_orders"] * self.RECENT_MONTHS, 1)
        merged["gmv_change_pct"] = (merged["recent_gmv"] - merged["baseline_gmv"] * self.RECENT_MONTHS) / np.maximum(merged["baseline_gmv"] * self.RECENT_MONTHS, 1)

        # 用 IQR 检测订单量降幅的异常州
        drops = merged["orders_change_pct"].dropna()
        if len(drops) >= 2:
            q1, q3 = drops.quantile(0.25), drops.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - self.IQR_MULTIPLIER * iqr
            anomaly_states = merged[merged["orders_change_pct"] < lower_bound]
            for _, row in anomaly_states.iterrows():
                pct = float(row["orders_change_pct"])
                state = row["customer_state"]
                severity = self._classify_change(pct, "订单量下降")
                alerts.append(AnomalyAlert(
                    dimension="state_sales",
                    entity=state,
                    metric="订单量",
                    current_value=float(row["recent_orders"]),
                    baseline_value=float(row["baseline_orders"]),
                    change_pct=round(pct * 100, 1),
                    severity=severity,
                    description=f"州 {state} 近期月均订单 {float(row['recent_orders']):.0f} 单，较基线月均 {float(row['baseline_orders']):.0f} 单下降 {abs(pct) * 100:.1f}%",
                    suggestion=f"建议排查 {state} 州配送覆盖、库存及营销活动变化。",
                ))

            # GMV 降幅检测
            gmv_drops = merged["gmv_change_pct"].dropna()
            if len(gmv_drops) >= 2:
                gmv_q1, gmv_q3 = gmv_drops.quantile(0.25), gmv_drops.quantile(0.75)
                gmv_iqr = gmv_q3 - gmv_q1
                gmv_bound = gmv_q1 - self.IQR_MULTIPLIER * gmv_iqr
                gmv_anomalies = merged[
                    (merged["gmv_change_pct"] < gmv_bound) &
                    (~merged["customer_state"].isin(anomaly_states["customer_state"]))
                ]
                for _, row in gmv_anomalies.iterrows():
                    pct = float(row["gmv_change_pct"])
                    state = row["customer_state"]
                    severity = self._classify_change(pct, "GMV下降")
                    alerts.append(AnomalyAlert(
                        dimension="state_sales",
                        entity=state,
                        metric="GMV",
                        current_value=float(row["recent_gmv"]),
                        baseline_value=float(row["baseline_gmv"]),
                        change_pct=round(pct * 100, 1),
                        severity=severity,
                        description=f"州 {state} 近期 GMV {float(row['recent_gmv']):,.0f}，较基线月均 {float(row['baseline_gmv']):,.0f} 下降 {abs(pct) * 100:.1f}%",
                        suggestion=f"建议检查 {state} 州客单价是否下滑，及高价值品类是否流失。",
                    ))

        return alerts

    def _detect_delivery_anomalies(self, df: pd.DataFrame) -> list[AnomalyAlert]:
        alerts: list[AnomalyAlert] = []
        df = df.copy()
        df["year_month"] = df["year_month"].astype(str)
        all_months = sorted(df["year_month"].unique())
        if len(all_months) < self.RECENT_MONTHS + 1:
            return alerts

        recent_months = all_months[-self.RECENT_MONTHS:]
        baseline_months = all_months[: -self.RECENT_MONTHS]

        recent = df[df["year_month"].isin(recent_months)]
        baseline = df[df["year_month"].isin(baseline_months)]

        recent_agg = recent.groupby("customer_state").agg(
            recent_days=("avg_delivery_days", "mean"),
            recent_rate=("on_time_rate", "mean"),
            recent_orders=("total_orders", "sum"),
        ).reset_index()

        baseline_agg = baseline.groupby("customer_state").agg(
            baseline_days=("avg_delivery_days", "mean"),
            baseline_rate=("on_time_rate", "mean"),
        ).reset_index()

        merged = recent_agg.merge(baseline_agg, on="customer_state", how="inner")
        merged = merged[merged["recent_orders"] >= self.MIN_ORDERS_FOR_STATE]
        if merged.empty:
            return alerts

        merged["days_change_pct"] = (merged["recent_days"] - merged["baseline_days"]) / np.maximum(merged["baseline_days"], 0.01)
        merged["rate_change_pct"] = (merged["recent_rate"] - merged["baseline_rate"]) / np.maximum(merged["baseline_rate"], 1e-6)

        # 配送天数 IQR
        days_change = merged["days_change_pct"].dropna()
        if len(days_change) >= 2:
            q1, q3 = days_change.quantile(0.25), days_change.quantile(0.75)
            iqr = q3 - q1
            upper = q3 + self.IQR_MULTIPLIER * iqr
            slow_states = merged[merged["days_change_pct"] > upper]
            for _, row in slow_states.iterrows():
                pct = float(row["days_change_pct"])
                state = row["customer_state"]
                severity = self._classify_change(pct, "配送天数上升")
                alerts.append(AnomalyAlert(
                    dimension="delivery",
                    entity=state,
                    metric="配送天数",
                    current_value=float(row["recent_days"]),
                    baseline_value=float(row["baseline_days"]),
                    change_pct=round(pct * 100, 1),
                    severity=severity,
                    description=f"州 {state} 配送天数从 {float(row['baseline_days']):.1f} 天增至 {float(row['recent_days']):.1f} 天（+{pct * 100:.1f}%）",
                    suggestion=f"建议排查 {state} 州承运商服务、节假日及基建变化。",
                ))

        # 准时率 IQR
        rate_change = merged["rate_change_pct"].dropna()
        if len(rate_change) >= 2:
            q1, q3 = rate_change.quantile(0.25), rate_change.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - self.IQR_MULTIPLIER * iqr
            late_states = merged[
                (merged["rate_change_pct"] < lower) &
                (~merged["customer_state"].isin({a.entity for a in alerts}))
            ]
            for _, row in late_states.iterrows():
                pct = float(row["rate_change_pct"])
                state = row["customer_state"]
                severity = self._classify_change(pct, "准时率下降")
                alerts.append(AnomalyAlert(
                    dimension="delivery",
                    entity=state,
                    metric="准时率",
                    current_value=float(row["recent_rate"]),
                    baseline_value=float(row["baseline_rate"]),
                    change_pct=round(pct * 100, 1),
                    severity=severity,
                    description=f"州 {state} 准时率从 {float(row['baseline_rate']):.1%} 降至 {float(row['recent_rate']):.1%}（{pct * 100:.1f}%）",
                    suggestion=f"建议复查 {state} 州预计送达日期的准确性，以及快递揽收时效。",
                ))

        return alerts

    def _detect_review_anomalies(self, df: pd.DataFrame) -> list[AnomalyAlert]:
        alerts: list[AnomalyAlert] = []
        df = df.copy()
        df["year_month"] = df["year_month"].astype(str)
        all_months = sorted(df["year_month"].unique())
        if len(all_months) < self.RECENT_MONTHS + 1:
            return alerts

        recent_months = all_months[-self.RECENT_MONTHS:]
        baseline_months = all_months[: -self.RECENT_MONTHS]

        recent = df[df["year_month"].isin(recent_months)]
        baseline = df[df["year_month"].isin(baseline_months)]

        recent_agg = recent.groupby("seller_id").agg(
            recent_score=("avg_review_score", "mean"),
            recent_orders=("total_orders", "sum"),
        ).reset_index()

        baseline_agg = baseline.groupby("seller_id").agg(
            baseline_score=("avg_review_score", "mean"),
        ).reset_index()

        merged = recent_agg.merge(baseline_agg, on="seller_id", how="inner")
        merged = merged[merged["recent_orders"] >= 2]
        if merged.empty:
            return alerts

        merged["score_change_pct"] = (merged["recent_score"] - merged["baseline_score"]) / np.maximum(merged["baseline_score"] - 1, 0.01)

        drops = merged["score_change_pct"].dropna()
        if len(drops) >= 3:
            q1, q3 = drops.quantile(0.25), drops.quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                lower = q1 - self.IQR_MULTIPLIER * iqr
                bad_sellers = merged[
                    (merged["score_change_pct"] < lower) &
                    (merged["recent_score"] < merged["baseline_score"]) &
                    (merged["recent_score"] < 3.5)
                ].nlargest(8, "recent_orders")
                for _, row in bad_sellers.iterrows():
                    pct = float(row["score_change_pct"])
                    sid = str(row["seller_id"])[:20]
                    severity = self._classify_change(pct, "评分下降")
                    alerts.append(AnomalyAlert(
                        dimension="review",
                        entity=f"卖家 {sid}",
                        metric="平均评分",
                        current_value=float(row["recent_score"]),
                        baseline_value=float(row["baseline_score"]),
                        change_pct=round(pct * 100, 1),
                        severity=severity,
                        description=f"卖家 {sid} 评分从 {float(row['baseline_score']):.2f} 降至 {float(row['recent_score']):.2f}（{pct * 100:.1f}%），近期 {int(row['recent_orders'])} 单",
                        suggestion=f"建议对卖家 {sid} 进行质量复查，关注商品质量或客服变化。",
                    ))

        return alerts

    def _classify_change(self, change_pct: float, metric_name: str) -> str:
        thresholds = self.SEVERITY_THRESHOLDS.get(metric_name, (0.20, 0.35))
        abs_change = abs(change_pct)
        if abs_change >= thresholds[0]:
            return "critical"
        if abs_change >= thresholds[1]:
            return "warning"
        return "info"

    def _deduplicate_and_sort(self, alerts: list[AnomalyAlert]) -> list[AnomalyAlert]:
        seen: set[tuple[str, str]] = set()
        unique: list[AnomalyAlert] = []
        for a in alerts:
            key = (a.entity, a.metric)
            if key not in seen:
                seen.add(key)
                unique.append(a)
        severity_order = {"critical": 0, "warning": 1, "info": 2}
        unique.sort(key=lambda x: (severity_order.get(x.severity, 3), -abs(x.change_pct)))
        return unique[:12]

    def _generate_llm_analysis(
        self,
        question: str,
        alerts: list[AnomalyAlert],
        summary: str,
    ) -> str:
        alert_lines: list[str] = []
        for a in alerts[:8]:
            sev = {"critical": "🚨严重", "warning": "⚠️警告", "info": "ℹ️提示"}.get(a.severity, "—")
            alert_lines.append(
                f"- [{sev}] {a.description}（当前 {a.current_value}，变化 {a.change_pct:+.1f}%）"
            )

        prompt = f"""
{ANOMALY_AGENT_PROMPT}

用户问题：{question}

扫描到以下异常预警（{len(alerts)} 条）：

{chr(10).join(alert_lines)}

请输出：
1. 业务风险概述：当前最需要关注的异常是什么？（综合严重程度和影响面）
2. 根因推断：结合电商运营经验，哪些因素可能导致上述异常同时出现？
3. 应急处置建议：2-3 条可立即执行的动作，每条包含责任方和检查重点。
4. 后续监控方案：建议设置哪些指标的预警阈值？怎么跟踪改进效果？

硬性规则：
- 自然中文，面向业务管理者
- 引用具体的州名和数字
- 每条建议有明确动作而非泛泛而谈
- 控制在 350 字以内
"""
        try:
            text = self.llm.chat(
                [
                    {"role": "system", "content": "你是电商运营异常监控与风险分析专家。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.15,
                max_tokens=900,
            )
            if text and not text.startswith("LLM 调用失败"):
                return text.strip()
        except Exception:
            pass
        return ""
