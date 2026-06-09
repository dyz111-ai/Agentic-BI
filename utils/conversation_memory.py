from __future__ import annotations

from typing import Any

import pandas as pd

FOLLOW_UP_HINTS = (
    "那个", "这个", "上面", "刚才", "继续", "还有", "它", "该", "此", "呢", "那么",
    "同样", "也", "再", "详细", "展开", "为什么", "怎么回事", "呢？", "它呢",
    "那个州", "该州", "这个品类", "那类", "上述", "前面", "之前",
)


def is_follow_up_question(question: str) -> bool:
    """Heuristic: short question with referential words → likely follow-up."""
    q = (question or "").strip()
    if not q:
        return False
    if any(h in q for h in FOLLOW_UP_HINTS):
        return True
    return len(q) <= 18 and q.endswith(("呢", "吗", "?"))


def normalize_memory(memory: dict[str, Any] | None) -> dict[str, Any]:
    """Ensure memory dict has expected structure."""
    if not memory:
        return {"turns": []}
    out = dict(memory)
    out.setdefault("turns", [])
    return out


def build_context_prompt(memory: dict[str, Any] | None, max_turns: int = 3) -> str:
    """Format recent turns for LLM prompts (follow-up question resolution)."""
    memory = normalize_memory(memory)
    turns = memory.get("turns") or []
    if not turns:
        return ""

    lines = [
        "【会话上下文】当前问题可能是对前几轮结果的追问（如「这个州」「该品类」「上面提到的支付方式」）。",
        "请结合以下历史信息理解指代，并在 SQL / 分析 / 图表中正确使用对应实体。",
    ]
    for i, turn in enumerate(turns[-max_turns:], start=1):
        q = (turn.get("question") or "").strip()
        if not q:
            continue
        lines.append(f"\n--- 第 {len(turns) - len(turns[-max_turns:]) + i} 轮 ---")
        lines.append(f"上一轮问题：{q}")
        intent = turn.get("intent")
        if intent:
            lines.append(f"分析意图：{intent}")
        tables = turn.get("tables") or []
        if tables:
            lines.append(f"返回数据表：{', '.join(tables)}")
        direct = turn.get("direct_answer") or []
        if direct:
            lines.append(f"上一轮直接回答：{'；'.join(direct)}")
        findings = turn.get("findings") or []
        if findings:
            lines.append(f"上一轮关键发现：{'；'.join(findings)}")
        recs = turn.get("recommendations") or []
        if recs:
            lines.append(f"上一轮决策建议：{'；'.join(recs[:2])}")
        entities = turn.get("key_entities") or {}
        if entities:
            parts = [f"{k}={v}" for k, v in entities.items() if v]
            if parts:
                lines.append(f"关键实体：{', '.join(parts)}")

    lines.append("\n当前问题：")
    return "\n".join(lines)


def append_turn(
    memory: dict[str, Any],
    *,
    question: str,
    intent: str = "",
    tables: list[str] | None = None,
    direct_answer: list[str] | None = None,
    findings: list[str] | None = None,
    recommendations: list[str] | None = None,
    key_entities: dict[str, str] | None = None,
    max_turns: int = 5,
) -> dict[str, Any]:
    """Append a completed turn and trim history."""
    memory = normalize_memory(memory)
    turn = {
        "question": question,
        "intent": intent,
        "tables": tables or [],
        "direct_answer": direct_answer or [],
        "findings": findings or [],
        "recommendations": recommendations or [],
        "key_entities": key_entities or {},
    }
    memory["turns"] = (memory.get("turns") or []) + [turn]
    memory["turns"] = memory["turns"][-max_turns:]
    memory["last_question"] = question
    memory["last_intent"] = intent
    memory["last_tables"] = tables or []
    return memory


def extract_key_entities(
    data_result: Any = None,
    nlp_result: Any = None,
) -> dict[str, str]:
    """Pull top state/category/seller labels from query results for follow-up resolution."""
    entities: dict[str, str] = {}
    tables = getattr(data_result, "tables", {}) or {} if data_result else {}

    for df in tables.values():
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        cols = {str(c).lower(): c for c in df.columns}

        if "customer_state" in cols and "top_state" not in entities:
            state_col = cols["customer_state"]
            work = df.copy()
            gmv_col = None
            for cand in ("total_gmv", "gmv", "sales", "total_sales"):
                if cand in cols:
                    gmv_col = cols[cand]
                    break
            if gmv_col is not None:
                work[gmv_col] = pd.to_numeric(work[gmv_col], errors="coerce").fillna(0)
                row = work.nlargest(1, gmv_col)
            else:
                row = work.head(1)
            if not row.empty:
                val = str(row[state_col].iloc[0]).strip()
                if val:
                    entities["top_state"] = val

        if "product_category_english" in cols and "top_category" not in entities:
            cat_col = cols["product_category_english"]
            work = df.copy()
            neg_col = cols.get("negative_rate") or cols.get("negative_reviews")
            if neg_col is not None:
                work[neg_col] = pd.to_numeric(work[neg_col], errors="coerce").fillna(0)
                row = work.nlargest(1, neg_col)
            else:
                row = work.head(1)
            if not row.empty:
                val = str(row[cat_col].iloc[0]).strip()
                if val and val.lower() != "unknown":
                    entities["top_category"] = val

        if "seller_id" in cols and "top_seller" not in entities:
            seller_col = cols["seller_id"]
            score_col = cols.get("avg_review_score") or cols.get("review_score")
            work = df.copy()
            if score_col is not None:
                work[score_col] = pd.to_numeric(work[score_col], errors="coerce")
                row = work.nsmallest(1, score_col)
            else:
                row = work.head(1)
            if not row.empty:
                val = str(row[seller_col].iloc[0]).strip()
                if val:
                    entities["top_seller"] = val

    if nlp_result is not None:
        top_neg = getattr(nlp_result, "top_negative_categories", pd.DataFrame())
        if isinstance(top_neg, pd.DataFrame) and not top_neg.empty:
            cat = str(top_neg.iloc[0].get("product_category_english", "")).strip()
            if cat and cat.lower() != "unknown":
                entities.setdefault("top_negative_category", cat)

    return {k: v for k, v in entities.items() if v}
