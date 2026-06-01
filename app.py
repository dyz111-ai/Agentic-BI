from __future__ import annotations

import streamlit as st
from sqlalchemy.exc import SQLAlchemyError

from config.settings import DB_URL, AUTO_REFRESH_MV
from utils.db import get_engine, ensure_sample_db_if_needed, table_exists
from utils.preaggregation import refresh_preaggregations
from agents.orchestrator import OrchestratorAgent
from dashboard.layout import render_report, render_sidebar_history

st.set_page_config(page_title="Agentic BI - Olist", page_icon="📊", layout="wide")

st.title("📊 Agentic BI：多表电商运营分析与决策智能系统")
st.caption("Olist 电商数据 · 多 Agent 协作 · 预聚合视图加速 · SQL 查询 · 可视化 · 预测 · 决策建议")

EXAMPLES = [
    "2017 年 GMV 是多少？按月和各州排名的趋势怎样？",
    "平台整体准时交付率是多少？哪些州延迟最严重？",
    "哪种支付方式最受欢迎？平均分期数是多少？",
    "产品的重量、尺寸与运费之间有什么关系？",
    "Top 10 差评品类及其主要差评原因是什么？",
    "根据历史订单趋势，预测未来 6 周的销售额，并给出趋势解读。",
    "基于全部分析结果，给出平台 3 个月内的三大优先改进策略。",
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


@st.cache_resource(show_spinner=False)
def init_orchestrator(db_url: str):
    engine = init_engine(db_url)
    return OrchestratorAgent(engine)


if "history" not in st.session_state:
    st.session_state.history = []
if "last_input" not in st.session_state:
    st.session_state.last_input = EXAMPLES[0]

with st.sidebar:
    st.header("配置")
    st.write("当前数据库：")
    st.code(DB_URL, language="text")
    st.markdown("---")

    st.subheader("推荐测试问题")
    selected = st.radio("一键选择", EXAMPLES, index=0, label_visibility="collapsed")
    use_selected = st.button("使用该问题", use_container_width=True)
    if use_selected:
        st.session_state.last_input = selected

    if st.button("刷新预聚合表", use_container_width=True):
        try:
            refresh_preaggregations(init_engine(DB_URL))
            st.success("预聚合表刷新成功")
        except Exception as exc:
            st.error(f"刷新失败：{exc}")

    st.markdown("---")
    st.subheader("历史对话")
    render_sidebar_history(st.session_state.history)

st.subheader("自然语言提问")
user_question = st.text_area(
    "请输入业务问题",
    value=st.session_state.last_input,
    height=120,
    placeholder="例如：2017 年哪个州销售额最高？交付准时率是多少？",
    label_visibility="collapsed",
)
run = st.button("开始分析", type="primary")

st.markdown("---")
st.subheader("分析结果")

if run and user_question.strip():
    st.session_state.last_input = user_question
    try:
        orch = init_orchestrator(DB_URL)
        with st.spinner("Agent 正在查询、分析、画图和生成建议..."):
            result = orch.handle(user_question)
        render_report(result)
        st.session_state.history.append({
            "question": user_question,
            "answer": result.final_answer,
            "direct_answer": getattr(result, "direct_answer", None) or [],
            "findings": getattr(result, "findings", None) or [],
            "recommendations": (
                getattr(result, "recommendations", None)
                or getattr(result.decision_result, "recommendations", None)
                or []
            ),
        })
    except SQLAlchemyError as exc:
        st.error(f"数据库错误：{exc}")
    except Exception as exc:
        st.error(f"系统运行失败：{exc}")
else:
    st.info("在上方输入问题后点击「开始分析」。")
    st.markdown(
        """
        这个系统会自动完成：
        1. 协调器 Agent 判断任务类型；
        2. 数据分析 Agent 优先查询预聚合表；
        3. 可视化 Agent 自动生成图表；
        4. 评论洞察 Agent 分析差评关键词；
        5. 预测 Agent 预测未来 6 周 GMV；
        6. 决策智能 Agent 输出运营建议。
        """
    )
