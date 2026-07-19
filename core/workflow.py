from langgraph.graph import StateGraph, END
# 注释redis相关导入
# import redis
# from langgraph.checkpoint.redis import RedisSaver
from langgraph.checkpoint.memory import MemorySaver

from core.agent_state import AgentState
from app.agents import route_agent, planner_agent, tool_exec_agent, reflect_agent, summary_agent
from core.config import settings

# ========== 替换RedisSaver为内存存储，消除FT._LIST依赖 ==========
checkpointer = MemorySaver()
# 删掉redis客户端、setup相关代码
# redis_client = redis.from_url(settings.REDIS_URL)
# checkpointer = RedisSaver(redis_client=redis_client)
# checkpointer.setup()


workflow = StateGraph(AgentState)
workflow.add_node("router", route_agent)
workflow.add_node("planner", planner_agent)
workflow.add_node("tool_exec", tool_exec_agent)
workflow.add_node("reflect", reflect_agent)
workflow.add_node("summary", summary_agent)

workflow.set_entry_point("router")

# 路由分支
def route_condition(state: AgentState):
    rt = state["route_target"]
    return "summary" if rt == "direct" else "planner"

workflow.add_conditional_edges(
    source="router",
    path=route_condition,
    path_map={"planner": "planner", "summary": "summary"}
)

workflow.add_edge("planner", "tool_exec")
workflow.add_edge("tool_exec", "reflect")

# 反思循环分支
def reflect_condition(state: AgentState):
    return "planner" if state["need_retry"] else "summary"

workflow.add_conditional_edges(
    source="reflect",
    path=reflect_condition,
    path_map={"planner": "planner", "summary": "summary"}
)

workflow.add_edge("summary", END)

# 编译带断点续跑（内存版checkpointer）
multi_agent_graph = workflow.compile(checkpointer=checkpointer)