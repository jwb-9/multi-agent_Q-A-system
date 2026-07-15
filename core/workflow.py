from langgraph.graph import StateGraph, END
from core.agent_state import AgentState
from core.agents import route_agent, rag_agent, search_agent, summary_agent

# 构建状态图
workflow = StateGraph(AgentState)
# 注册节点
workflow.add_node("router", route_agent)
workflow.add_node("rag_worker", rag_agent)
workflow.add_node("search_worker", search_agent)
workflow.add_node("summary", summary_agent)
workflow.set_entry_point("router")

# 路由函数：只返回原始标签rag/search/both/direct
def route_condition(state: AgentState):
    return state["route_target"]

# 第一层：router出口映射，名称一一对应真实节点
workflow.add_conditional_edges(
    source="router",
    path=route_condition,
    path_map={
        "direct": "summary",
        "rag": "rag_worker",
        "search": "search_worker",
        "both": "rag_worker"
    }
)

# 第二层：rag执行完二次判断，区分纯rag / both混合
def after_rag_condition(state: AgentState):
    return state["route_target"]

workflow.add_conditional_edges(
    source="rag_worker",
    path=after_rag_condition,
    path_map={
        "rag": "summary",        # 纯知识库，检索完直接汇总
        "both": "search_worker"  # 混合场景，检索后再搜索
    }
)

# 搜索完成统一汇总
workflow.add_edge("search_worker", "summary")
# 汇总结束流程
workflow.add_edge("summary", END)

multi_agent_graph = workflow.compile()