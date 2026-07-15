from langgraph.graph import StateGraph, END
from core.agent_state import AgentState
from core.agents import route_agent, rag_agent, search_agent, summary_agent

# 构建状态图
workflow = StateGraph(AgentState)

# 注册所有Agent节点
workflow.add_node("router", route_agent)
workflow.add_node("rag_worker", rag_agent)
workflow.add_node("search_worker", search_agent)
workflow.add_node("summary", summary_agent)

# 入口节点
workflow.set_entry_point("router")

# 路由分支逻辑
def route_condition(state: AgentState):
    target = state["route_target"]
    if target == "rag":
        return "rag_worker"
    elif target == "search":
        return "search_worker"
    elif target == "both":
        return "rag_worker"
    elif target == "direct":
        return "summary"

# 分支跳转
workflow.add_conditional_edges("router", route_condition, {
    "rag_worker": "rag_worker",
    "search_worker": "search_worker",
    "summary": "summary"
})

# both分支：先执行RAG，再执行搜索，最后汇总
workflow.add_edge("rag_worker", "search_worker")
# 单分支执行完统一到汇总节点
workflow.add_edge("search_worker", "summary")
workflow.add_edge("rag_worker", "summary")
# 汇总完成结束流程
workflow.add_edge("summary", END)

# 编译可执行图
multi_agent_graph = workflow.compile()