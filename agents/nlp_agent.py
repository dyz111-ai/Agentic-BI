from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import re

import pandas as pd

from models.sentiment import enrich_reviews_with_sentiment, extract_keywords
from utils.llm import LLMClient


@dataclass
class NLPResult:
    enriched_reviews: pd.DataFrame = field(default_factory=pd.DataFrame)
    negative_keywords: list[tuple[str, int]] = field(default_factory=list)
    positive_keywords: list[tuple[str, int]] = field(default_factory=list)
    top_negative_categories: pd.DataFrame = field(default_factory=pd.DataFrame)
    summary: str = ""

    # 新增：供 DecisionAgent 使用的结构化证据
    negative_reason_summary: str = ""
    sample_negative_reviews: list[str] = field(default_factory=list)
    min_reviews_threshold: int = 10


class ReviewInsightAgent:
    """Review / NLP Insight Agent.

    改进点：
    1. 保留原有规则情感分析和关键词统计，保证无 API 也能跑。
    2. 差评品类排序加入最小样本量逻辑，避免 1 条评论 100% 差评率误导。
    3. 输出 total_reviews / negative_reviews / negative_rate / avg_score / sample_warning。
    4. 有 LLM Key 时，对真实低分评论样本做差评原因归纳；无 Key 时使用关键词兜底。
    """

    def __init__(self, min_reviews_threshold: int = 10, sample_size: int = 40):
        self.min_reviews_threshold = min_reviews_threshold
        self.sample_size = sample_size
        self.llm = LLMClient()

    def analyze_reviews(self, reviews_df: pd.DataFrame) -> NLPResult:
        if reviews_df is None or reviews_df.empty:
            return NLPResult(summary="没有可用评论数据。", min_reviews_threshold=self.min_reviews_threshold)

        df = enrich_reviews_with_sentiment(reviews_df.copy())
        df = self._normalize_review_columns(df)

        negative = df[(df["review_score"].astype(float) <= 2) | (df["sentiment_label"].astype(str) == "negative")]
        positive = df[(df["review_score"].astype(float) >= 4) | (df["sentiment_label"].astype(str) == "positive")]

        neg_texts = self._combine_text_columns(negative).tolist()
        pos_texts = self._combine_text_columns(positive).tolist()

        top_neg_cat = self._build_top_negative_categories(df)
        negative_keywords = extract_keywords(neg_texts, 20)
        positive_keywords = extract_keywords(pos_texts, 20)
        sample_reviews = self._sample_negative_reviews(negative)
        reason_summary = self._summarize_negative_reasons(sample_reviews, negative_keywords, top_neg_cat)
        summary = self._build_summary(top_neg_cat, negative_keywords, reason_summary)

        return NLPResult(
            enriched_reviews=df,
            negative_keywords=negative_keywords,
            positive_keywords=positive_keywords,
            top_negative_categories=top_neg_cat,
            summary=summary,
            negative_reason_summary=reason_summary,
            sample_negative_reviews=sample_reviews,
            min_reviews_threshold=self.min_reviews_threshold,
        )

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def _build_top_negative_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        if "product_category_english" not in df.columns:
            return pd.DataFrame()

        work = df.copy()
        work["product_category_english"] = work["product_category_english"].fillna("unknown").astype(str)
        work["review_score"] = pd.to_numeric(work["review_score"], errors="coerce")
        work = work.dropna(subset=["review_score"])
        if work.empty:
            return pd.DataFrame()

        work["is_negative"] = (work["review_score"] <= 2).astype(int)

        top = (
            work.groupby("product_category_english", dropna=False)
            .agg(
                total_reviews=("review_score", "count"),
                negative_reviews=("is_negative", "sum"),
                avg_score=("review_score", "mean"),
            )
            .reset_index()
        )
        top["negative_rate"] = top["negative_reviews"] / top["total_reviews"].clip(lower=1)
        top["sample_warning"] = top["total_reviews"] < self.min_reviews_threshold

        # 排序策略：优先展示样本量足够的高差评率品类；样本不足的仍保留但排在后面。
        top["sample_ok"] = top["total_reviews"] >= self.min_reviews_threshold
        top = top.sort_values(
            ["sample_ok", "negative_rate", "negative_reviews", "total_reviews"],
            ascending=[False, False, False, False],
        ).head(10)

        top["avg_score"] = top["avg_score"].round(3)
        top["negative_rate"] = top["negative_rate"].round(4)
        return top.drop(columns=["sample_ok"])

    def _summarize_negative_reasons(
        self,
        sample_reviews: list[str],
        negative_keywords: list[tuple[str, int]],
        top_neg_cat: pd.DataFrame,
    ) -> str:
        if not sample_reviews:
            return self._keyword_reason_summary(negative_keywords)

        # LLM 主路径：只给真实低分评论样本，不让模型凭空推断。
        if self.llm.enabled:
            cat_rows = []
            if isinstance(top_neg_cat, pd.DataFrame) and not top_neg_cat.empty:
                cat_rows = top_neg_cat.head(5).to_dict(orient="records")
            kw_text = ", ".join([f"{w}({c})" for w, c in negative_keywords[:12]])
            review_text = "\n".join([f"- {r[:300]}" for r in sample_reviews[: self.sample_size]])
            prompt = f"""
你是评论洞察 Agent。请只根据下面真实低分评论样本、关键词和品类统计，总结主要差评原因。

品类统计 Top：{cat_rows}
负面关键词：{kw_text}
低分评论样本：
{review_text}

要求：
1. 禁止编造样本中没有出现的原因。
2. 不要使用“假设”“可能如下”。
3. 输出 3-5 个差评原因，每个原因用一句话，包含证据关键词。
4. 如果样本不足，说明“样本量有限”。
"""
            text = self.llm.chat(
                [
                    {"role": "system", "content": "你是严谨的评论文本分析 Agent，只基于输入样本总结原因。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=900,
            )
            if text and not text.startswith("LLM 调用失败") and "假设" not in text:
                return self._clean_reason_text(text)

        return self._keyword_reason_summary(negative_keywords)

    def _build_summary(
        self,
        top_neg_cat: pd.DataFrame,
        negative_keywords: list[tuple[str, int]],
        reason_summary: str,
    ) -> str:
        parts: list[str] = []
        if isinstance(top_neg_cat, pd.DataFrame) and not top_neg_cat.empty:
            row = top_neg_cat.iloc[0]
            cat = row.get("product_category_english", "unknown")
            total = int(row.get("total_reviews", 0))
            neg = int(row.get("negative_reviews", 0))
            rate = float(row.get("negative_rate", 0))
            avg = float(row.get("avg_score", 0))
            warning = "样本量偏小，结论仅作为风险预警；" if bool(row.get("sample_warning", False)) else ""
            parts.append(
                f"{warning}差评风险最高的品类是 {cat}，评论数 {total}，差评数 {neg}，"
                f"差评率约 {rate:.1%}，平均评分 {avg:.2f}。"
            )
        else:
            parts.append("已完成评论情感与关键词分析，但当前数据中缺少可按品类聚合的字段。")

        if negative_keywords:
            kws = "、".join([w for w, _ in negative_keywords[:5]])
            parts.append(f"高频负面关键词包括：{kws}。")

        if reason_summary:
            parts.append(f"差评原因摘要：{reason_summary}")

        return " ".join(parts)

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def _normalize_review_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        # Some LLM SQL may alias columns differently. Normalize common variants.
        rename_map: dict[str, str] = {}
        lower_to_original = {str(c).lower(): c for c in df.columns}

        aliases = {
            "review_score": ["score", "rating", "avg_score"],
            "review_comment_title": ["comment_title", "title"],
            "review_comment_message": ["comment", "message", "review_text", "comment_message"],
            "product_category_english": ["product_category_name_english", "category", "product_category", "product_category_name"],
        }
        for target, candidates in aliases.items():
            if target in df.columns:
                continue
            for cand in candidates:
                if cand in lower_to_original:
                    rename_map[lower_to_original[cand]] = target
                    break
        if rename_map:
            df = df.rename(columns=rename_map)

        if "review_score" not in df.columns:
            df["review_score"] = 0
        if "review_comment_title" not in df.columns:
            df["review_comment_title"] = ""
        if "review_comment_message" not in df.columns:
            df["review_comment_message"] = ""
        if "product_category_english" not in df.columns:
            df["product_category_english"] = "unknown"

        df["review_score"] = pd.to_numeric(df["review_score"], errors="coerce").fillna(0)
        return df

    @staticmethod
    def _combine_text_columns(df: pd.DataFrame) -> pd.Series:
        if df is None or df.empty:
            return pd.Series([], dtype=str)
        title = df["review_comment_title"] if "review_comment_title" in df.columns else ""
        msg = df["review_comment_message"] if "review_comment_message" in df.columns else ""
        return (title.fillna("").astype(str) + " " + msg.fillna("").astype(str)).str.strip()

    def _sample_negative_reviews(self, negative: pd.DataFrame) -> list[str]:
        texts = self._combine_text_columns(negative)
        texts = texts[texts.str.len() >= 8].drop_duplicates()
        if texts.empty:
            return []
        # Prefer longer comments because they contain more reasons.
        sample = texts.to_frame("text")
        sample["len"] = sample["text"].str.len()
        sample = sample.sort_values("len", ascending=False).head(self.sample_size)
        return sample["text"].tolist()

    @staticmethod
    def _keyword_reason_summary(negative_keywords: list[tuple[str, int]]) -> str:
        if not negative_keywords:
            return "当前评论文本较少，无法稳定提取差评原因。"
        kws = [w for w, _ in negative_keywords[:8]]
        joined = "、".join(kws)
        return f"基于词频统计，主要负面线索集中在：{joined}；需要结合原始评论进一步归因到物流、质量、描述不符或售后问题。"

    @staticmethod
    def _clean_reason_text(text: str) -> str:
        lines = []
        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue
            line = re.sub(r"^[-*•]\s+", "", line)
            line = re.sub(r"^\d+[.)、]\s+", "", line)
            if line:
                lines.append(line)
        return "；".join(lines[:5]) if lines else text.strip()
