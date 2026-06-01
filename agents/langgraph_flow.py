from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, StateGraph


class BIAgentState(TypedDict, total=False):
    """Shared state passed between LangGraph nodes."""

    question: str
    conversation_context: str
    thread_id: str
    data_result: Any
    orchestration_plan: dict[str, Any]
    nlp_result: Any
    forecast_result: Any
    visualization_result: Any
    decision_result: Any
    final_answer: str
    direct_answer: list[str]
    findings: list[str]
    recommendations: list[str]
    technical_details: dict[str, Any]
    memory: dict[str, Any]
    orchestrator_mode: str


def build_bi_graph(orchestrator: Any):
    """Build LangGraph StateGraph wrapping OrchestratorAgent node handlers."""

    graph = StateGraph(BIAgentState)

    graph.add_node("data_analysis", orchestrator._graph_node_data)
    graph.add_node("plan", orchestrator._graph_node_plan)
    graph.add_node("nlp", orchestrator._graph_node_nlp)
    graph.add_node("forecast", orchestrator._graph_node_forecast)
    graph.add_node("visualize", orchestrator._graph_node_visualize)
    graph.add_node("decision", orchestrator._graph_node_decision)
    graph.add_node("compose", orchestrator._graph_node_compose)

    graph.set_entry_point("data_analysis")
    graph.add_edge("data_analysis", "plan")

    graph.add_conditional_edges(
        "plan",
        orchestrator._graph_route_after_plan,
        {
            "nlp": "nlp",
            "forecast": "forecast",
            "visualize": "visualize",
        },
    )
    graph.add_conditional_edges(
        "nlp",
        orchestrator._graph_route_after_nlp,
        {
            "forecast": "forecast",
            "visualize": "visualize",
        },
    )
    graph.add_edge("forecast", "visualize")
    graph.add_edge("visualize", "decision")
    graph.add_edge("decision", "compose")
    graph.add_edge("compose", END)

    # 不使用 MemorySaver：state 含 DataResult / DataFrame 等对象，msgpack 无法序列化。
    # 多轮会话上下文由 utils/conversation_memory + Streamlit session_state 维护。
    return graph.compile()


def invoke_bi_graph(
    compiled_graph: Any,
    *,
    question: str,
    conversation_context: str,
    memory: dict[str, Any],
    thread_id: str,
) -> BIAgentState:
    """Run the compiled LangGraph workflow for one user turn."""
    initial: BIAgentState = {
        "question": question,
        "conversation_context": conversation_context,
        "thread_id": thread_id,
        "memory": memory,
        "orchestrator_mode": "langgraph",
    }
    config = {"configurable": {"thread_id": thread_id}} if thread_id else None
    if config:
        return compiled_graph.invoke(initial, config=config)
    return compiled_graph.invoke(initial)
