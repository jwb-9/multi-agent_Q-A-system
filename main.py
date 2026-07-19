import uuid
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from core.workflow import multi_agent_graph
from core.logger import logger, set_trace_id
from core.config import settings
from core.memory import MemoryManager, new_session_id
from langchain_core.messages import HumanMessage, AIMessage
import uvicorn

app = FastAPI(title="生产级多智能体Agent RAG系统")

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    query: str
    session_id: str = None

    @field_validator("query")
    def check_query_length(cls, v):
        if len(v) > settings.QUERY_MAX_LENGTH:
            raise ValueError(f"查询长度不能超过{settings.QUERY_MAX_LENGTH}字符")
        if not v.strip():
            raise ValueError("查询内容不能为空")
        return v.strip()

class ChatResponse(BaseModel):
    trace_id: str
    session_id: str
    dispatch_agent: str
    rag_info: str
    search_info: str
    final_answer: str
    reflect_score: int
    reflect_msg: str
    tool_records: list

# 健康检查接口
@app.get("/health")
async def health():
    return {"code": 0, "msg": "service running"}

# 全局异常捕获
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    trace_id = str(uuid.uuid4())
    set_trace_id(trace_id)
    logger.error(f"服务全局异常: {str(exc)}", exc_info=True)
    raise HTTPException(status_code=500, detail={"trace_id": trace_id, "msg": f"系统内部错误: {str(exc)}"})

@app.post("/api/multi-agent/chat", response_model=ChatResponse)
async def multi_agent_chat(req: ChatRequest):
    trace_id = str(uuid.uuid4())
    set_trace_id(trace_id)
    session_id = req.session_id if req.session_id else new_session_id()
    short_mem = MemoryManager.get_session_short_memory(session_id)
    long_mem = MemoryManager.get_long_memory(session_id)
    logger.info(f"trace:{trace_id},session:{session_id},query:{req.query}")

    init_state = {
        "trace_id": trace_id,
        "session_id": session_id,
        "user_query": req.query,
        "messages": [HumanMessage(content=req.query)],
        "short_memory": short_mem,
        "long_memory": long_mem,
        "route_target": "",
        "task_plan": [],
        "rag_result": "",
        "search_result": "",
        "tool_records": [],
        "reflect_score": 0,
        "reflect_msg": "",
        "reflect_loop": 0,
        "need_retry": False,
        "final_answer": ""
    }
    config = {"configurable": {"thread_id": session_id}}
    result_state = multi_agent_graph.invoke(init_state, config=config)

    # 追加AI回复到消息列表，完整对话上下文
    result_state["messages"].append(AIMessage(content=result_state["final_answer"]))
    # 更新记忆
    new_short = MemoryManager.compress_dialogue(result_state["messages"])
    MemoryManager.save_short_memory(session_id, new_short)
    MemoryManager.append_long_memory(session_id, req.query, result_state["final_answer"])

    return ChatResponse(
        trace_id=trace_id,
        session_id=session_id,
        dispatch_agent=result_state["route_target"],
        rag_info=result_state["rag_result"],
        search_info=result_state["search_result"],
        final_answer=result_state["final_answer"],
        reflect_score=result_state["reflect_score"],
        reflect_msg=result_state["reflect_msg"],
        tool_records=result_state["tool_records"]
    )

if __name__ == "__main__":
    logger.info("多智能体Agent RAG服务启动成功")
    uvicorn.run("main:app", host=settings.SERVICE_HOST, port=settings.SERVICE_PORT)