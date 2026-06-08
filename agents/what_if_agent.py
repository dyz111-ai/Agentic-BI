from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import pandas as pd
from sqlalchemy.engine import Engine

from utils.llm import LLMClient
from config.prompts import WHAT_IF_AGENT_PROMPT


@dataclass
class WhatIfResult:
    current_avg_score: float = 0.0
    projected_avg_score: float = 0.0
    score_improvement: float = 0.0
    score_improvement_pct: float = 0.0
    removed_seller_count: int = 0
    removed_order_count: int = 0
    remaining_order_count: int = 0
    total_order_count: int = 0
    total_valid_sellers: int = 0
    removed_sellers: pd.DataFrame = field(default_factory=pd.DataFrame)
    summary: str = ""
    llm_analysis: str = ""
    scenario: str = ""

    @property
    def has_result(self) -> bool:
        return self.removed_seller_count > 0


class WhatIfAgent:
    """What-If Simulation Agent.

    模拟业务场景，如：
    - "如果将 Top 20 高差评卖家的商品统一下架，平台整体评分预估提升多少？"
    - "如果限制评分低于 3.0 的卖家流量，对平台整体 GMV 影响多大？"

    核心逻辑：
    1. 从 mv_seller_perf 获取卖家评分与订单数据（加权平均）
    2. 按加权平均评分升序选出 Top N 最低评分卖家
    3. 计算移除后的评分变化
    4. 调用 LLM 生成业务解读与执行建议
    5. LLM 不可用时仍输出计算结果
    """

    DEFAULT_TOP_N = 20
    MIN_ORDERS_THRESHOLD = 2       # 卖家最少订单数（排除单笔交易的噪音）
    MIN_SELLERS_KEEP = 5           # 模拟后至少保留的卖家数

    def __init__(self, engine: Engine | None = None):
        self.engine = engine
        self.llm = LLMClient()

    def simulate_remove_low_score_sellers(
        self,
        seller_data: pd.DataFrame | None = None,
        top_n: int | None = None,
        question: str = "",
        nlp_result: Any = None,
    ) -> WhatIfResult:
        top_n = top_n or self.DEFAULT_TOP_N

        if seller_data is None or seller_data.empty:
            seller_data = self._fetch_seller_perf()

        if seller_data is None or seller_data.empty:
            return WhatIfResult(
                summary="无可用的卖家绩效数据，无法执行 What-if 模拟。请先刷新预聚合表 mv_seller_perf。",
                scenario=f"移除评分最低的 {top_n} 个卖家 — 模拟未执行：无数据。",
            )

        df = self._normalize_seller_data(seller_data)
        if df is None:
            fallback = self._fetch_seller_perf()
            if fallback is not None and not fallback.empty:
                seller_data = fallback
                df = self._normalize_seller_data(seller_data)

        if df is None:
            return WhatIfResult(
                summary="卖家数据缺少必要字段（avg_review_score / total_orders），无法执行模拟。",
                scenario=f"移除评分最低的 {top_n} 个卖家 — 模拟未执行：字段缺失。",
            )

        valid = df[df["total_orders"] >= self.MIN_ORDERS_THRESHOLD].copy()
        if valid.empty or len(valid) < self.MIN_SELLERS_KEEP:
            return WhatIfResult(
                summary=f"符合条件的卖家数量不足（需要至少 {self.MIN_SELLERS_KEEP} 个有 ≥{self.MIN_ORDERS_THRESHOLD} 单的卖家），样本量过小无法可靠模拟。",
                scenario=f"移除评分最低的 {top_n} 个卖家 — 模拟未执行：样本不足。",
            )

        actual_top_n = min(top_n, len(valid) - self.MIN_SELLERS_KEEP)
        total_weighted = (valid["avg_review_score"] * valid["total_orders"]).sum()
        total_orders = valid["total_orders"].sum()
        current_avg = total_weighted / total_orders if total_orders > 0 else 0.0

        worst = valid.nsmallest(actual_top_n, "avg_review_score")
        removed_weighted = (worst["avg_review_score"] * worst["total_orders"]).sum()
        removed_orders = worst["total_orders"].sum()

        remaining_weighted = total_weighted - removed_weighted
        remaining_orders = total_orders - removed_orders
        projected_avg = remaining_weighted / remaining_orders if remaining_orders > 0 else current_avg

        score_improvement = projected_avg - current_avg
        score_improvement_pct = score_improvement / max(current_avg, 0.01) * 100

        scenario = f"移除评分最低的 {actual_top_n} 个卖家"

        summary_parts: list[str] = []
        summary_parts.append(
            f"当前平台加权平均评分为 **{current_avg:.4f}**（基于 {len(valid)} 个有效卖家、{int(total_orders):,} 笔订单）。"
        )
        summary_parts.append(
            f"若移除评分最低的 **{actual_top_n}** 个卖家（共 {int(removed_orders):,} 笔订单，占总量 {removed_orders / max(total_orders, 1) * 100:.1f}%），"
            f"预计平台加权平均评分将提升至 **{projected_avg:.4f}**，"
            f"提升 **{score_improvement:+.4f}**（**{score_improvement_pct:+.2f}%**）。"
        )

        seller_preview = self._format_seller_preview(worst.head(10))
        if seller_preview:
            summary_parts.append(f"\n被移除的低评分卖家（按评分从低到高）：\n{seller_preview}")

        llm_analysis = ""
        if self.llm.enabled:
            llm_analysis = self._generate_llm_analysis(
                question=question,
                current_avg=current_avg,
                projected_avg=projected_avg,
                score_improvement=score_improvement,
                score_improvement_pct=score_improvement_pct,
                actual_top_n=actual_top_n,
                removed_sellers=worst,
                removed_orders=int(removed_orders),
                total_orders=int(total_orders),
                nlp_result=nlp_result,
            )
            if llm_analysis:
                summary_parts.append(llm_analysis)

        return WhatIfResult(
            current_avg_score=round(current_avg, 4),
            projected_avg_score=round(projected_avg, 4),
            score_improvement=round(score_improvement, 4),
            score_improvement_pct=round(score_improvement_pct, 2),
            removed_seller_count=actual_top_n,
            removed_order_count=int(removed_orders),
            remaining_order_count=int(remaining_orders),
            total_order_count=int(total_orders),
            total_valid_sellers=len(valid),
            removed_sellers=worst,
            summary="\n".join(summary_parts),
            llm_analysis=llm_analysis,
            scenario=scenario,
        )

    def _fetch_seller_perf(self) -> pd.DataFrame | None:
        if self.engine is None:
            return None
        try:
            from utils.db import read_df
            sql = """
                SELECT seller_id, seller_state,
                       SUM(total_gmv) AS total_gmv,
                       SUM(total_orders) AS total_orders,
                       ROUND(AVG(avg_review_score), 2) AS avg_review_score
                FROM mv_seller_perf
                GROUP BY seller_id, seller_state
                HAVING total_orders > 0
                ORDER BY avg_review_score ASC
            """
            df = read_df(sql, engine=self.engine)
            if df is not None and not df.empty:
                return df
        except Exception:
            pass

        try:
            from utils.db import read_df
            sql = """
                SELECT oi.seller_id,
                       COALESCE(s.seller_state, 'unknown') AS seller_state,
                       COUNT(DISTINCT oi.order_id) AS total_orders,
                       ROUND(AVG(r.review_score), 2) AS avg_review_score,
                       ROUND(SUM(oi.price + oi.freight_value), 2) AS total_gmv
                FROM order_items oi
                LEFT JOIN sellers s ON oi.seller_id = s.seller_id
                LEFT JOIN order_reviews r ON oi.order_id = r.order_id
                WHERE r.review_score IS NOT NULL
                GROUP BY oi.seller_id, s.seller_state
                HAVING COUNT(DISTINCT oi.order_id) >= 2
                ORDER BY AVG(r.review_score) ASC
                LIMIT 200
            """
            df = read_df(sql, engine=self.engine)
            if df is not None and not df.empty:
                return df
        except Exception:
            pass
        return None

    def _normalize_seller_data(self, df: pd.DataFrame) -> pd.DataFrame | None:
        work = df.copy()

        score_col = None
        for c in ("avg_review_score", "avg_score", "review_score", "score"):
            if c in work.columns:
                score_col = c
                break
        if score_col is None:
            for c in work.columns:
                if "score" in str(c).lower() or "rating" in str(c).lower():
                    score_col = c
                    break

        orders_col = None
        for c in ("total_orders", "orders", "order_count", "count"):
            if c in work.columns:
                orders_col = c
                break
        if orders_col is None:
            for c in work.columns:
                if "order" in str(c).lower():
                    orders_col = c
                    break
            if orders_col is None:
                for c in work.columns:
                    cl = str(c).lower()
                    if "count" in cl and "order" not in cl:
                        orders_col = c
                        break

        if score_col is None or orders_col is None:
            return None

        work = work.rename(columns={score_col: "avg_review_score", orders_col: "total_orders"})
        work["avg_review_score"] = pd.to_numeric(work["avg_review_score"], errors="coerce")
        work["total_orders"] = pd.to_numeric(work["total_orders"], errors="coerce")
        work = work.dropna(subset=["avg_review_score", "total_orders"])
        work["total_orders"] = work["total_orders"].astype(int)
        if work.empty:
            return None
        return work

    @staticmethod
    def _format_seller_preview(removed_sellers: pd.DataFrame) -> str:
        lines: list[str] = []
        for _, row in removed_sellers.iterrows():
            sid = str(row.get("seller_id", row.get("id", "?")))[:32]
            score = float(row.get("avg_review_score", 0))
            orders = int(row.get("total_orders", 0))
            state = str(row.get("seller_state", row.get("customer_state", row.get("state", "未知"))))[:8]
            gmv = row.get("total_gmv")
            extra = f"，GMV {float(gmv):,.0f}" if gmv is not None and pd.notna(gmv) else ""
            lines.append(f"- 卖家 {sid}（{state}）：评分 {score:.2f}，{orders} 单{extra}")
        return "\n".join(lines)

    def _generate_llm_analysis(
        self,
        question: str,
        current_avg: float,
        projected_avg: float,
        score_improvement: float,
        score_improvement_pct: float,
        actual_top_n: int,
        removed_sellers: pd.DataFrame,
        removed_orders: int,
        total_orders: int,
        nlp_result: Any = None,
    ) -> str:
        seller_text = self._format_seller_preview(removed_sellers.head(10))

        nlp_context = ""
        if nlp_result is not None:
            nlp_summary = getattr(nlp_result, "summary", "")
            neg_reason = getattr(nlp_result, "negative_reason_summary", "")
            if nlp_summary:
                nlp_context += f"\n评论洞察：{nlp_summary}"
            if neg_reason:
                nlp_context += f"\n差评原因总结：{neg_reason}"

        prompt = f"""
{WHAT_IF_AGENT_PROMPT}

用户问题：{question}

模拟场景：移除评分最低的 {actual_top_n} 个卖家

模拟结果（基于真实数据计算）：
- 当前平台加权平均评分：{current_avg:.4f}
- 移除后预计加权平均评分：{projected_avg:.4f}
- 评分绝对提升：{score_improvement:+.4f}
- 评分相对提升：{score_improvement_pct:+.2f}%
- 移除卖家数：{actual_top_n}
- 移除订单数：{removed_orders:,}（占总订单 {removed_orders / max(total_orders, 1) * 100:.1f}%）

被移除的低评分卖家（Top 10）：
{seller_text}
{nlp_context}

请输出：
1. 对该模拟结果的业务解读（评分提升幅度是否显著？是否值得考虑执行？）
2. ⚠️ 执行该策略的风险：是否会导致某些区域供给不足？订单量损失占比是否可接受？
3. 📋 建议的分阶段执行方案（先警告观察 → 限流降权 → 协商整改 → 最后下架），每个阶段给出触发条件和时限。

硬性规则：
- 用自然中文，面向业务管理者，不要用技术术语
- 必须引用上面的具体数字
- 每条风险以"⚠️"开头，每条执行建议以"📋"开头
- 控制在 400 字以内
"""
        try:
            text = self.llm.chat(
                [
                    {"role": "system", "content": "你是电商运营策略与 What-If 模拟分析专家。"},
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
