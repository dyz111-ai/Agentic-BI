from __future__ import annotations

import streamlit as st
import pandas as pd


def render_direct_answer(direct_answer: list[str]):
    if not direct_answer:
        st.info("暂无直接回答。")
        return
    for item in direct_answer:
        st.markdown(f"- {item}")


def render_findings(findings: list[str]):
    if not findings:
        st.info("暂无关键发现。")
        return
    for item in findings:
        st.markdown(f"- {item}")


def render_recommendations(recommendations: list[str]):
    if not recommendations:
        st.info("暂无决策建议。")
        return
    for rec in recommendations:
        st.markdown(f"- {rec}")


def render_what_if(what_if_result):
    if what_if_result is None or not getattr(what_if_result, "has_result", False):
        return

    st.markdown("---")
    st.markdown("#### 🔮 What-If 模拟分析")
    st.markdown(f"**模拟场景：** {getattr(what_if_result, 'scenario', '')}")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("当前加权平均评分", f"{getattr(what_if_result, 'current_avg_score', 0):.4f}")
    col2.metric("模拟后评分", f"{getattr(what_if_result, 'projected_avg_score', 0):.4f}")
    improvement = getattr(what_if_result, "score_improvement", 0)
    improvement_pct = getattr(what_if_result, "score_improvement_pct", 0)
    col3.metric("评分提升", f"{improvement:+.4f}", delta=f"{improvement_pct:+.1f}%")
    col4.metric("移除卖家数 / 订单数", f"{getattr(what_if_result, 'removed_seller_count', 0)} 家 / {getattr(what_if_result, 'removed_order_count', 0):,} 单")

    summary = getattr(what_if_result, "summary", "")
    if summary:
        with st.expander("模拟详情", expanded=True):
            st.markdown(summary)


def render_anomaly(anomaly_result):
    if anomaly_result is None or not getattr(anomaly_result, "has_alerts", False):
        return

    st.markdown("---")
    st.markdown("#### 🔍 异常检测与预警")

    alert_count = getattr(anomaly_result, "alert_count", 0)
    critical = getattr(anomaly_result, "critical_count", 0)
    warning = getattr(anomaly_result, "warning_count", 0)
    info = getattr(anomaly_result, "info_count", 0)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("总预警数", alert_count)
    col2.metric("🚨 严重", critical, delta_color="inverse")
    col3.metric("⚠️ 警告", warning, delta_color="inverse")
    col4.metric("ℹ️ 提示", info)

    scan_dims = getattr(anomaly_result, "scan_dimensions", [])
    scan_period = getattr(anomaly_result, "scan_period", "")
    if scan_dims:
        st.caption(f"扫描维度：{'、'.join(scan_dims)}　|　扫描窗口：{scan_period} vs 历史基线")

    alerts = getattr(anomaly_result, "alerts", []) or []
    if alerts:
        with st.expander("预警详情", expanded=True):
            rows = []
            for a in alerts:
                sev_icon = {"critical": "🚨", "warning": "⚠️", "info": "ℹ️"}.get(a.severity, "—")
                rows.append({
                    "等级": sev_icon,
                    "维度": a.dimension,
                    "主体": a.entity,
                    "指标": a.metric,
                    "当前值": f"{a.current_value:,.2f}" if isinstance(a.current_value, float) and abs(a.current_value) > 1 else f"{a.current_value:.4f}",
                    "基线值": f"{a.baseline_value:,.2f}" if isinstance(a.baseline_value, float) and abs(a.baseline_value) > 1 else f"{a.baseline_value:.4f}",
                    "变化": f"{a.change_pct:+.1f}%",
                    "说明": a.description,
                    "建议": a.suggestion,
                })
            st.dataframe(rows, use_container_width=True, hide_index=True,
                         column_order=["等级", "维度", "主体", "指标", "当前值", "基线值", "变化", "说明", "建议"])

    summary = getattr(anomaly_result, "summary", "")
    if summary:
        with st.expander("LLM 业务解读", expanded=False):
            st.markdown(summary)


def render_technical_details(details: dict):
    if not details:
        return
    with st.expander("技术细节（Agent 调度与查询策略）", expanded=False):
        c1, c2, c3 = st.columns(3)
        c1.metric("查询方式", details.get("query_mode", "—"))
        c2.metric("数据来源", details.get("data_source", "—"))
        c3.metric("图表数量", details.get("chart_count", 0))

        st.caption(f"图表策略：{details.get('chart_mode', '—')}")
        st.caption(f"分析意图：{details.get('intent', '—')}")
        if details.get("orchestrator"):
            st.caption(f"编排框架：{details.get('orchestrator')} · 会话轮次：{details.get('conversation_turns', '—')}")
        tables = details.get("returned_tables") or []
        if tables:
            st.caption("返回数据表：" + "、".join(tables))

        agents = details.get("agents") or {}
        if agents:
            rows = []
            for name, info in agents.items():
                status = "已调用" if info.get("called") else "未调用"
                rows.append({"Agent": name, "状态": status, "触发依据": info.get("reason", "—")})
            st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

        llm_error = details.get("llm_error") or ""
        if llm_error:
            st.warning("大模型相关步骤出现异常，详情如下：")
            st.code(llm_error[:1200], language="text")

        notes = details.get("notes") or []
        for note in notes:
            st.warning(note)


def render_report(result):
    """Render analysis result with clear sections."""
    tab_summary, tab_charts, tab_data = st.tabs(["分析结论", "图表", "数据与 SQL"])

    with tab_summary:
        st.markdown("#### 直接回答")
        render_direct_answer(getattr(result, "direct_answer", None) or [])
        st.markdown("#### 关键发现")
        render_findings(getattr(result, "findings", None) or [])
        st.markdown("#### 决策建议")
        recommendations = getattr(result, "recommendations", None) or getattr(result.decision_result, "recommendations", None) or []
        render_recommendations(recommendations)
        render_what_if(getattr(result, "what_if_result", None))
        render_anomaly(getattr(result, "anomaly_result", None))
        render_technical_details(getattr(result, "technical_details", {}) or {})
        render_metrics(getattr(result.data_result, "elapsed", {}) or {})

    with tab_charts:
        render_figures(result.visualization_result.figures)

    with tab_data:
        render_sql_blocks(result.data_result.sql_blocks)
        render_tables(result.data_result.tables)


def render_history_item(item: dict, compact: bool = False):
    st.markdown(f"**问：** {item['question']}")
    direct = item.get("direct_answer") or []
    findings = item.get("findings") or []
    recommendations = item.get("recommendations") or []
    if direct or findings or recommendations:
        if direct:
            st.markdown("**直接回答**")
            if compact:
                for d in direct[:2]:
                    st.caption(f"· {d[:80]}{'…' if len(d) > 80 else ''}")
            else:
                render_direct_answer(direct)
        if findings:
            st.markdown("**关键发现**")
            if compact:
                for f in findings[:3]:
                    st.caption(f"· {f[:80]}{'…' if len(f) > 80 else ''}")
            else:
                render_findings(findings)
        if recommendations:
            st.markdown("**决策建议**")
            if compact:
                for r in recommendations[:3]:
                    st.caption(f"· {r[:80]}{'…' if len(r) > 80 else ''}")
            else:
                render_recommendations(recommendations)
    elif item.get("answer"):
        text = item["answer"]
        st.markdown(text[:300] + ("…" if len(text) > 300 else "") if compact else text)


def render_sidebar_history(history: list[dict]):
    if not history:
        st.caption("暂无历史记录")
        return
    for item in history[-5:][::-1]:
        q = item.get("question", "")
        label = q if len(q) <= 28 else q[:28] + "…"
        with st.expander(label, expanded=False):
            render_history_item(item, compact=True)


def render_sql_blocks(sql_blocks):
    if not sql_blocks:
        st.info("本次没有执行 SQL。")
        return
    for block in sql_blocks:
        st.markdown(f"**{block.get('name', 'query')}**")
        st.caption(f"来源：{block.get('source', '—')}")
        st.code(block.get("sql", ""), language="sql")


def render_tables(tables: dict[str, pd.DataFrame]):
    if not tables:
        st.info("本次没有返回数据表。")
        return
    for name, df in tables.items():
        st.markdown(f"**{name}**")
        st.dataframe(df.head(200), width='stretch')


def render_figures(figures: dict):
    if not figures:
        st.info("本次问题没有生成图表。")
        return
    for idx, (name, fig) in enumerate(figures.items()):
        st.markdown(f"**{name}**")
        st.plotly_chart(fig, width="stretch", key=f"plotly_chart_{idx}_{name}")


def render_metrics(elapsed: dict):
    if not elapsed:
        return
    st.markdown("---")
    st.caption("查询耗时")
    cols = st.columns(min(4, len(elapsed)))
    for col, (k, v) in zip(cols, elapsed.items()):
        col.metric(k, f"{v * 1000:.1f} ms")
