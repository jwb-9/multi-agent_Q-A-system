from core.logger import logger
from core.config import settings
from core.tools import TOOL_MAP, BaseTool, VALID_TOOL_NAMES
from core.memory import MemoryManager
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.output_parsers.json import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any

# LLM全局实例
llm = ChatOllama(
    base_url=settings.OLLAMA_BASE_URL,
    model=settings.LLM_MODEL,
    temperature=0.1,
    timeout=settings.OLLAMA_TIMEOUT
)
embed = OllamaEmbeddings(base_url=settings.OLLAMA_BASE_URL, model=settings.EMBED_MODEL)

# ====================== 结构化输出模型 ======================
class RouteOutput(BaseModel):
    target: str = Field(description="仅允许 rag / search / both / direct")
parser_route = PydanticOutputParser(pydantic_object=RouteOutput)

class TaskPlanItem(BaseModel):
    task_name: str
    tool_name: Optional[str] = None
    query: str
class PlanOutput(BaseModel):
    task_list: List[TaskPlanItem]
parser_plan = PydanticOutputParser(pydantic_object=PlanOutput)

class ReflectOutput(BaseModel):
    score: int = Field(description="0-10分，信息完整度打分")
    msg: str
    need_retry: bool
parser_reflect = PydanticOutputParser(pydantic_object=ReflectOutput)

# JSON容错解析包装
def safe_parse(prompt_chain, input_data) -> Any:
    try:
        return prompt_chain.invoke(input_data)
    except Exception as e:
        logger.warning(f"结构化解析失败，降级原始文本重试: {str(e)}")
        raw = llm.invoke(prompt_chain.prompt.format(**input_data))
        fallback_parser = JsonOutputParser()
        return fallback_parser.parse(raw.content)

# ====================== Agent1：路由调度Agent ======================
def route_agent(state):
    query = state["user_query"]
    short_mem = state["short_memory"]
    logger.info(f"路由Agent处理query:{query}")
    prompt = ChatPromptTemplate.from_template("""
你是任务调度专家，根据用户问题与历史对话记忆输出分类。
规则：
1. rag：多智能体、LangGraph、向量知识库相关私有知识
2. search：2024-2026实时资讯、外网行业信息
3. both：同时需要私有知识库+外网资讯
4. direct：数学、通用常识，和多智能体无关

历史对话记忆：{short_mem}
用户问题：{query}
{format_instructions}
""").partial(format_instructions=parser_route.get_format_instructions())
    chain = prompt | llm | parser_route
    res = safe_parse(chain, {"query": query, "short_mem": short_mem})
    target = res.get("target", "direct") if isinstance(res, dict) else res.target
    logger.info(f"路由判定:{target}")
    return {"route_target": target}

# ====================== Agent2：规划Agent Plan-and-Execute ======================
def planner_agent(state):
    query = state["user_query"]
    route = state["route_target"]
    prompt = ChatPromptTemplate.from_template("""
根据用户问题与路由类型生成执行任务列表，可选工具：local_knowledge_search、internet_search
路由类型:{route}
用户问题:{query}
{format_instructions}
""").partial(format_instructions=parser_plan.get_format_instructions())
    chain = prompt | llm | parser_plan
    res = safe_parse(chain, {"query": query, "route": route})
    task_list = []
    raw_tasks = res.get("task_list", []) if isinstance(res, dict) else res.task_list
    for t in raw_tasks:
        if t.get("tool_name") in VALID_TOOL_NAMES:
            task_list.append(t)
    return {"task_plan": task_list}

# ====================== Agent3：统一工具执行器 ======================
def tool_exec_agent(state):
    task_list = state["task_plan"]
    tool_records = []
    rag_total = ""
    search_total = ""
    for task in task_list:
        tool_name = task.get("tool_name")
        if not tool_name or tool_name not in TOOL_MAP:
            continue
        tool: BaseTool = TOOL_MAP[tool_name]
        task_query = task["query"]
        logger.info(f"执行工具:{tool_name}, query:{task_query}")
        tool_result = tool.run(task_query)
        tool_records.append({
            "tool": tool_name,
            "query": task_query,
            "result_snippet": tool_result[:300]
        })
        if tool_name == "local_knowledge_search":
            rag_total += f"\n\n===== {task_query} =====\n{tool_result}"
        elif tool_name == "internet_search":
            search_total += f"\n\n===== {task_query} =====\n{tool_result}"
    return {
        "rag_result": rag_total.strip(),
        "search_result": search_total.strip(),
        "tool_records": tool_records
    }

# ====================== Agent4：反思校验Agent Reflection ======================
def reflect_agent(state):
    q = state["user_query"]
    rag = state["rag_result"]
    search = state["search_result"]
    loop_cnt = state["reflect_loop"]
    max_loop = settings.MAX_REFLECT_LOOP
    if loop_cnt >= max_loop:
        return {"reflect_score": 10, "reflect_msg": "达到最大重试次数，停止校验", "need_retry": False}
    prompt = ChatPromptTemplate.from_template("""
校验现有信息能否完整回答用户问题，输出打分(0-10)、说明、是否需要重新检索工具。
打分阈值：低于6分则需要重试检索。
用户问题:{q}
本地知识库:{rag}
网络信息:{search}
{format_instructions}
""").partial(format_instructions=parser_reflect.get_format_instructions())
    chain = prompt | llm | parser_reflect
    res = safe_parse(chain, {"q": q, "rag": rag, "search": search})
    if isinstance(res, dict):
        score = res.get("score", 5)
        msg = res.get("msg", "信息不足")
        need_retry = res.get("need_retry", score < 6)
    else:
        score = res.score
        msg = res.msg
        need_retry = res.need_retry or score < 6
    return {
        "reflect_score": score,
        "reflect_msg": msg,
        "need_retry": need_retry,
        "reflect_loop": loop_cnt + 1
    }

# ====================== Agent5：汇总合成Agent（增加文档溯源） ======================
def summary_agent(state):
    q = state["user_query"]
    rag = state.get("rag_result", "无本地知识库信息")
    search = state.get("search_result", "无网络信息")
    long_mem = state["long_memory"]
    prompt = ChatPromptTemplate.from_template("""
结合长期历史对话记忆、私有知识库、外网搜索信息完整回答用户问题。
要求：
1. 禁止编造信息，所有内容必须来自提供材料；
2. 关键结论标注信息来源（知识库/外网搜索）；
3. 逻辑通顺，分段清晰，无冗余。

历史长期记忆：{long_mem}
私有知识库：{rag}
外网搜索：{search}
用户提问：{q}
""")
    chain = prompt | llm
    ans = chain.invoke({"long_mem": long_mem, "rag": rag, "search": search, "q": q})
    return {"final_answer": ans.content.strip()}