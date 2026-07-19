from typing import TypedDict, Annotated, List, Optional, Dict, Any
from langchain_core.messages import BaseMessage
import operator

class AgentState(TypedDict):
    # 请求基础信息
    trace_id: str
    session_id: str
    user_query: str

    # 对话消息
    messages: Annotated[List[BaseMessage], operator.add]
    short_memory: str
    long_memory: str

    # 调度规划
    route_target: str
    task_plan: List[Dict[str, Any]]

    # 工具输出
    rag_result: str
    search_result: str
    tool_records: List[Dict[str, Any]]

    # 反思校验
    reflect_score: int
    reflect_msg: str
    reflect_loop: int
    need_retry: bool

    # 最终结果
    final_answer: str