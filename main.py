from fastapi import FastAPI
from pydantic import BaseModel
from core.workflow import multi_agent_graph
from core.logger import logger

app = FastAPI(title="Multi-Agent 多智能体问答系统")

class ChatRequest(BaseModel):
    query: str

@app.post("/api/multi-agent/chat")
async def multi_agent_chat(req: ChatRequest):
    logger.info(f"收到用户提问：{req.query}")  # 打印入参
    try:
        # 初始化全局状态
        init_state = {
            "user_query": req.query,
            "messages": [],
            "route_target": "",
            "rag_result": "",
            "search_result": "",
            "final_answer": ""
        }
        # 执行多Agent工作流
        result_state = multi_agent_graph.invoke(init_state)
        logger.info(f"流程执行完成，路由分类：{result_state['route_target']}")
        return {
            "query": req.query,
            "dispatch_agent": result_state["route_target"],
            "rag_info": result_state["rag_result"],
            "search_info": result_state["search_result"],
            "final_answer": result_state["final_answer"]
        }
    except Exception as e:
        logger.error("接口处理发生异常", exc_info=True)  # 完整堆栈写入日志
        raise e

if __name__ == "__main__":
    import uvicorn
    from core.config import settings
    logger.info("服务启动成功")
    uvicorn.run(
        "main:app",
        host=settings.SERVICE_HOST,
        port=settings.SERVICE_PORT
    )