from __future__ import annotations

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

_USER_UNAVAILABLE_PHRASES = (
    "当前查询未返回",
    "未返回该",
    "未返回任何",
    "无法提供",
    "无法分析",
    "无法基于",
    "请要求数据团队",
    "数据中不包含",
    "缺少按州",
)


def _filter_user_visible_lines(lines: list[str]) -> list[str]:
    out: list[str] = []
    for line in lines or []:
        text = str(line).strip()
        if not text:
            continue
        if any(p in text for p in _USER_UNAVAILABLE_PHRASES):
            continue
        out.append(text)
    return out


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


def render_report(result, key_prefix: str = ""):
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
        render_preagg_comparison(getattr(result.data_result, "preagg_comparison", None))
        render_technical_details(getattr(result, "technical_details", {}) or {})
        render_metrics(getattr(result.data_result, "elapsed", {}) or {})

    with tab_charts:
        render_figures(result.visualization_result.figures, key_prefix=key_prefix)

    with tab_data:
        render_sql_blocks(result.data_result.sql_blocks)
        render_tables(result.data_result.tables)


def render_chat_summary(result) -> None:
    """Compact assistant reply shown in the chat bubble."""
    direct = getattr(result, "direct_answer", None) or []
    findings = getattr(result, "findings", None) or []
    recommendations = (
        getattr(result, "recommendations", None)
        or getattr(getattr(result, "decision_result", None), "recommendations", None)
        or []
    )
    recommendations = _filter_user_visible_lines(recommendations)

    if direct:
        st.markdown("**直接回答**")
        render_direct_answer(_filter_user_visible_lines(direct))
    if findings:
        st.markdown("**关键发现**")
        render_findings(_filter_user_visible_lines(findings))
    if recommendations:
        st.markdown("**决策建议**")
        render_recommendations(recommendations)

    what_if = getattr(result, "what_if_result", None)
    if what_if is not None and getattr(what_if, "has_result", False):
        st.caption(f"🔮 What-If：{getattr(what_if, 'summary', '')[:120]}")

    anomaly = getattr(result, "anomaly_result", None)
    if anomaly is not None and getattr(anomaly, "has_alerts", False):
        st.caption(f"🔍 异常预警：发现 {getattr(anomaly, 'alert_count', 0)} 条异常")

    if not direct and not findings and not recommendations:
        st.info("分析已完成，请展开下方查看详细报告。")


def render_chat_assistant(result, turn_id: int = 0) -> None:
    """Render one assistant turn: summary in bubble + full report in expander."""
    render_chat_summary(result)
    figures = getattr(getattr(result, "visualization_result", None), "figures", None) or {}
    chart_n = len(figures)
    with st.expander(f"📊 查看完整分析报告（含 {chart_n} 张图表 · SQL · 技术细节）", expanded=False):
        render_report(result, key_prefix=f"turn{turn_id}_")


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


def render_figures(figures: dict, key_prefix: str = ""):
    if not figures:
        st.info("本次问题没有生成图表。")
        return
    for idx, (name, fig) in enumerate(figures.items()):
        st.markdown(f"**{name}**")
        st.plotly_chart(fig, width="stretch", key=f"{key_prefix}plotly_{idx}_{name}")


def render_metrics(elapsed: dict):
    if not elapsed:
        return
    st.markdown("---")
    st.caption("查询耗时")
    cols = st.columns(min(4, len(elapsed)))
    for col, (k, v) in zip(cols, elapsed.items()):
        col.metric(k, f"{v * 1000:.1f} ms")


def render_preagg_comparison(cmp: dict | None):
    """Show a bar chart comparing preagg query time vs raw JOIN time."""
    if not cmp:
        return
    preagg_ms = cmp.get("preagg_ms", 0.0)
    raw_ms = cmp.get("raw_ms", 0.0)
    speedup = cmp.get("speedup", 0.0)
    title = cmp.get("scenario_title", "")
    table = cmp.get("preagg_table", "")

    st.markdown("---")
    st.markdown("#### ⚡ 预聚合加速效果（本次查询实测）")

    c1, c2, c3 = st.columns(3)
    c1.metric("预聚合表耗时", f"{preagg_ms:.2f} ms", delta=None)
    c2.metric("等价 Raw JOIN 耗时", f"{raw_ms:.2f} ms", delta=None)
    c3.metric("加速倍数", f"{speedup:.1f} ×",
              delta=f"命中 {table}",
              delta_color="normal")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=["Raw JOIN（实时多表聚合）", f"预聚合表（{table}）"],
        x=[raw_ms, preagg_ms],
        orientation="h",
        marker_color=["#EF553B", "#00CC96"],
        text=[f"{raw_ms:.2f} ms", f"{preagg_ms:.2f} ms"],
        textposition="outside",
        hovertemplate="%{y}<br>耗时：%{x:.2f} ms<extra></extra>",
    ))
    fig.update_layout(
        title=f"查询耗时对比：{title}",
        xaxis_title="响应时间 (ms)",
        height=180,
        margin=dict(l=0, r=60, t=40, b=0),
        plot_bgcolor="white",
        xaxis=dict(showgrid=True, gridcolor="#EEEEEE"),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("查看对比 SQL", expanded=False):
        col_a, col_b = st.columns(2)
        with col_a:
            st.caption(f"✅ 预聚合查询（{table}）")
            st.code(cmp.get("preagg_sql", ""), language="sql")
        with col_b:
            st.caption("🔄 等价 Raw JOIN（实时聚合）")
            st.code(cmp.get("raw_sql", ""), language="sql")
