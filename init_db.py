from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv

load_dotenv()

# 示例本地文档数据
docs_text = """
多智能体系统(Multi-Agent)由多个具备独立感知、推理能力的智能体组成，通过协同分工完成复杂任务。
主流架构分为集中调度式、分布式协商式；企业落地首选LangGraph集中路由架构。
Multi-Agent可拆分任务：检索、搜索、代码、写作、校验等专用Agent，解决单一大模型上下文局限问题。
"""

text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
chunks = text_splitter.split_text(docs_text)

# 使用Ollama本地嵌入模型
embedding = OllamaEmbeddings(
    base_url=os.getenv("OLLAMA_BASE_URL"),
    model=os.getenv("EMBED_MODEL")
)

# 持久化向量库
Chroma.from_texts(
    texts=chunks,
    embedding=embedding,
    persist_directory="./chroma_db"
)
print("本地知识库初始化完成")