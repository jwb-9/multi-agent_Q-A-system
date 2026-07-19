from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from core.config import settings
from core.logger import logger

_embed = OllamaEmbeddings(base_url=settings.OLLAMA_BASE_URL, model=settings.EMBED_MODEL)

def init_vector_db(docs_text: str = None):
    logger.info("初始化向量知识库")
    if not docs_text:
        docs_text = """
多智能体系统(Multi-Agent)由多个具备独立感知、推理能力的智能体组成，通过协同分工完成复杂任务。
主流架构分为集中调度式、分布式协商式；企业落地首选LangGraph集中路由架构。
Multi-Agent可拆分任务：检索、搜索、代码、写作、校验等专用Agent，解决单一大模型上下文局限问题。
Plan-and-Execute规划Agent可自动拆解复杂问题为多子任务；Reflection反思Agent实现结果自校验抑制幻觉。
Agentic RAG支持多跳检索、查询改写、混合向量+关键词检索，提升问答准确率。
LangGraph支持Checkpoint断点续跑、子图封装、并行节点、Human-in-the-Loop人工介入。
"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    chunks = text_splitter.split_text(docs_text)
    Chroma.from_texts(
        texts=chunks,
        embedding=_embed,
        persist_directory=settings.CHROMA_PERSIST_DIR
    )
    logger.info(f"初始化向量库完成，分片数量：{len(chunks)}")
    print("本地知识库初始化完成")

# 增量追加文档
def add_doc_to_db(new_text: str):
    vs = Chroma(persist_directory=settings.CHROMA_PERSIST_DIR, embedding_function=_embed)
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    chunks = splitter.split_text(new_text)
    vs.add_texts(chunks)
    logger.info(f"增量插入{len(chunks)}个文本块")

if __name__ == "__main__":
    init_vector_db()