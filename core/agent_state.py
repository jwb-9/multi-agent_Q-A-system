from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage
import operator

# 多智能体全局共享状态
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]  # 全局对话消息
    user_query: str  # 用户原始问题
    route_target: str  # 路由Agent判定交给哪个专家
    rag_result: str  # RAG知识库输出
    search_result: str  # 联网搜索输出
    final_answer: str  # 最终汇总回答