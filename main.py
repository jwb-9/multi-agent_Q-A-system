from fastapi import FastAPI
from pydantic import BaseModel
from core.workflow import multi_agent_graph

app = FastAPI(title="Multi-Agent 多智能体问答系统")

class ChatRequest(BaseModel):
    query: str

@app.post("/api/multi-agent/chat")
async def multi_agent_chat(req: ChatRequest):
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
    return {
        "query": req.query,
        "dispatch_agent": result_state["route_target"],
        "rag_info": result_state["rag_result"],
        "search_info": result_state["search_result"],
        "final_answer": result_state["final_answer"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)