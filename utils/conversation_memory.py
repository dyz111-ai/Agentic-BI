from __future__ import annotations

from typing import Any


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
