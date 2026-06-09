from __future__ import annotations

import streamlit as st
from sqlalchemy.exc import SQLAlchemyError

from config.settings import DB_URL, AUTO_REFRESH_MV
from utils.db import get_engine, ensure_sample_db_if_needed, table_exists
from utils.preaggregation import refresh_preaggregations
from agents.orchestrator import OrchestratorAgent
from dashboard.layout import render_chat_assistant, render_sidebar_history
from utils.conversation_memory import normalize_memory, append_turn, extract_key_entities

st.set_page_config(page_title="Agentic BI - Olist", page_icon="📊", layout="wide")

st.title("📊 Agentic BI：多表电商运营分析与决策智能系统")
st.caption("Olist 电商数据 · 多 Agent 协作 · 预聚合视图加速 · 自然语言多轮对话分析")

EXAMPLES = [
    "2017 年 GMV 是多少？按月和各州排名的趋势怎样？",
    "平台整体准时交付率是多少？哪些州延迟最严重？",
    "哪种支付方式最受欢迎？平均分期数是多少？",
    "产品的重量、尺寸与运费之间有什么关系？",
    "Top 10 差评品类及其主要差评原因是什么？",
    "根据历史订单趋势，预测未来 6 周的销售额，并给出趋势解读。",
    "基于全部分析结果，给出平台 3 个月内的三大优先改进策略。",
    "如果将 Top 20 高差评卖家的商品统一下架，平台整体评分预估提升多少？",
    "自动扫描近期运营异常，检测是否有州订单骤降、配送恶化或评分突降",
    "帮我全面检查平台目前存在哪些异常问题？需要优先关注的风险有哪些？",
]


@st.cache_resource(show_spinner=False)
def init_engine(db_url: str):
    engine = get_engine(db_url)
    ensure_sample_db_if_needed(engine)
    if AUTO_REFRESH_MV:
        try:
            if table_exists("orders", engine):
                refresh_preaggregations(engine)
        except Exception as exc:
            st.warning(f"预聚合表刷新失败：{exc}")
    return engine


def init_orchestrator(db_url: str) -> OrchestratorAgent:
    engine = init_engine(db_url)
    return OrchestratorAgent(engine)


def _init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_results" not in st.session_state:
        st.session_state.chat_results = []
    if "history" not in st.session_state:
        st.session_state.history = []
    if "conversation_memory" not in st.session_state:
        st.session_state.conversation_memory = normalize_memory(None)


def _clear_conversation() -> None:
    st.session_state.messages = []
    st.session_state.chat_results = []
    st.session_state.history = []
    st.session_state.conversation_memory = normalize_memory(None)


def _append_history(question: str, result) -> None:
    st.session_state.history.append({
        "question": question,
        "answer": result.final_answer,
        "direct_answer": getattr(result, "direct_answer", []),
        "findings": getattr(result, "findings", []),
        "recommendations": (
            getattr(result, "recommendations", None)
            or getattr(getattr(result, "decision_result", None), "recommendations", None)
            or []
        ),
    })


def _process_question(question: str) -> None:
    question = question.strip()
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})

    try:
        orch = init_orchestrator(DB_URL)
        memory = st.session_state.conversation_memory
        with st.spinner("正在分析数据、生成图表与建议..."):
            result = orch.handle(question, memory=memory)

        turn_id = len(st.session_state.chat_results)
        st.session_state.chat_results.append(result)
        st.session_state.messages.append({
            "role": "assistant",
            "result_idx": turn_id,
            "turn_id": turn_id,
        })

        key_entities = extract_key_entities(result.data_result, result.nlp_result)
        st.session_state.conversation_memory = append_turn(
            result.memory,
            question=question,
            intent=result.data_result.intent,
            tables=list(result.data_result.tables.keys()),
            direct_answer=getattr(result, "direct_answer", []) or [],
            findings=getattr(result, "findings", []) or [],
            recommendations=getattr(result, "recommendations", []) or [],
            key_entities=key_entities,
        )
        _append_history(question, result)
    except SQLAlchemyError as exc:
        st.session_state.messages.append({
            "role": "assistant",
            "error": True,
            "content": f"数据库错误：{exc}",
        })
    except Exception as exc:
        st.session_state.messages.append({
            "role": "assistant",
            "error": True,
            "content": f"系统运行失败：{exc}",
        })


_init_session_state()

with st.sidebar:
    st.subheader("推荐测试问题")
    selected = st.radio("一键选择", EXAMPLES, index=0, label_visibility="collapsed")

    if st.button("发送该问题", use_container_width=True):
        st.session_state.pending_question = selected
        st.rerun()

    if st.button("刷新预聚合表", use_container_width=True):
        try:
            refresh_preaggregations(init_engine(DB_URL))
            st.success("预聚合表刷新成功")
        except Exception as exc:
            st.error(f"刷新失败：{exc}")

    if st.button("清空会话", use_container_width=True):
        _clear_conversation()
        st.success("已清空对话与会话记忆")
        st.rerun()

    turn_count = sum(1 for m in st.session_state.messages if m.get("role") == "user")
    st.caption(f"当前会话：{turn_count} 轮提问")

    st.markdown("---")
    st.subheader("对话摘要")
    render_sidebar_history(st.session_state.history)

# --- Main chat area ---
if not st.session_state.messages:
    st.info(
        "👋 你好！我是 Olist 电商运营分析助手。"
        "可直接提问，也可在左侧选择推荐问题；支持多轮追问，例如先问「2017 年哪个州 GMV 最高？」"
        "再问「那个州的配送准时率呢？」。"
    )

for msg in st.session_state.messages:
    role = msg.get("role", "assistant")
    with st.chat_message(role):
        if role == "user":
            st.markdown(msg.get("content", ""))
        elif msg.get("error"):
            st.error(msg.get("content", "分析失败"))
        else:
            result_idx = msg.get("result_idx")
            turn_id = msg.get("turn_id", result_idx or 0)
            if result_idx is not None and result_idx < len(st.session_state.chat_results):
                render_chat_assistant(st.session_state.chat_results[result_idx], turn_id=turn_id)
            else:
                st.warning("该条回复无法加载，请重新提问。")

pending = st.session_state.pop("pending_question", None)
user_prompt = st.chat_input("输入业务问题，按 Enter 发送…")
prompt_to_run = pending or user_prompt

if prompt_to_run and prompt_to_run.strip():
    _process_question(prompt_to_run)
    st.rerun()
