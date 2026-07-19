from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
from typing import Any, Dict
from core.config import settings
from core.logger import logger
from ddgs import DDGS
from ddgs.exceptions import DDGSException
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# 工具入参标准
class ToolInput(BaseModel):
    query: str = Field(description="工具执行查询语句")

# 统一工具基类
class BaseTool(ABC):
    name: str
    description: str
    args_schema = ToolInput

    @abstractmethod
    def run(self, query: str) -> str:
        pass

# 全局向量库单例，避免重复加载
_embed = OllamaEmbeddings(base_url=settings.OLLAMA_BASE_URL, model=settings.EMBED_MODEL)
_vector_store = Chroma(persist_directory=settings.CHROMA_PERSIST_DIR, embedding_function=_embed)
_retriever = _vector_store.as_retriever(search_kwargs={"k": settings.RETRIEVE_TOP_K})

# RAG检索工具
class RagTool(BaseTool):
    name = "local_knowledge_search"
    description = "查询本地私有知识库，用于多智能体、LangGraph相关技术文档检索"

    def run(self, query: str) -> str:
        try:
            docs = _retriever.invoke(query)
            logger.info(f"RagTool检索文档数:{len(docs)}")
            return "\n\n### 文档片段 ###\n".join([d.page_content for d in docs])
        except Exception as e:
            logger.error(f"RagTool异常: {str(e)}")
            return "本地知识库检索失败"

# 联网搜索工具
class SearchTool(BaseTool):
    name = "internet_search"
    description = "联网获取2024-2026最新行业资讯、实时数据、外部落地案例"

    def run(self, query: str) -> str:
        results = []
        try:
            with DDGS(timeout=settings.DDGS_TIMEOUT) as ddgs:
                resp = ddgs.text(query, max_results=settings.SEARCH_MAX_RESULT)
                for r in resp:
                    results.append(f"【{r['title']}】\n{r['body']}")
        except DDGSException as e:
            logger.error(f"SearchTool DDGS异常:{str(e)}")
            results.append("网络搜索引擎不可用，无外网信息")
        return "\n\n===== 搜索结果 =====\n".join(results)

# 工具工厂
TOOL_MAP = {
    "local_knowledge_search": RagTool(),
    "internet_search": SearchTool()
}
VALID_TOOL_NAMES = set(TOOL_MAP.keys())