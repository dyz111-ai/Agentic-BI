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


@st.cache_resource(show_spinner=False)
def init_orchestrator(db_url: str):
    engine = init_engine(db_url)
    return OrchestratorAgent(engine)


if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.header("配置")
    st.write("当前数据库：")
    st.code(DB_URL, language="text")
    st.markdown("---")

    st.subheader("推荐测试问题")
    selected = st.radio("一键选择", EXAMPLES, index=0, label_visibility="collapsed")

    use_selected = st.button("使用该问题", width='stretch')
    if use_selected:
        st.session_state.last_input = selected

    if st.button("刷新预聚合表", width='stretch'):
        try:
            refresh_preaggregations(init_engine(DB_URL))
            st.success("预聚合表刷新成功")
        except Exception as exc:
            st.error(f"刷新失败：{exc}")

    if st.button("清空会话记忆", width='stretch'):
        st.session_state.history = []
        st.success("已清空历史记录")

    st.markdown("---")
    st.subheader("历史对话")
    render_sidebar_history(st.session_state.history)

st.subheader("自然语言提问")
user_question = st.text_area(
    "请输入业务问题",
    value=st.session_state.get("last_input", EXAMPLES[0]),
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
        with st.spinner("正在分析数据、生成图表与建议..."):
            result = orch.handle(user_question)
        render_report(result)
        st.session_state.history.append({
            "question": user_question,
            "answer": result.final_answer,
            "direct_answer": getattr(result, "direct_answer", []),
            "findings": getattr(result, "findings", []),
            "recommendations": getattr(result, "decision_result", None) and getattr(result.decision_result, "recommendations", []) or [],
        })
    except SQLAlchemyError as exc:
        st.error(f"数据库错误：{exc}")
    except Exception as exc:
        st.error(f"系统运行失败：{exc}")
        st.exception(exc)
else:
    st.info("在上方输入问题后点击「开始分析」。")
    st.markdown(
        """
        这个系统会自动完成以下工作：
        1. 分析问题类型并确定查询策略
        2. 优先使用预聚合表 mv_* 加速查询
        3. 生成合适的可视化图表
        4. 综合各方面信息给出决策建议
        """
    )
