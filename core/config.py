import os
from dotenv import load_dotenv
import json

# 加载环境文件，不存在则警告
load_dotenv(override=True)

class Settings:
    # Ollama
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "qwen2.5:7b")
    EMBED_MODEL: str = os.getenv("EMBED_MODEL", "bge-m3")
    OLLAMA_TIMEOUT: int = 30

    # Chroma向量库
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    RETRIEVE_TOP_K: int = 3
    RERANK_TOP_K: int = 2

    # 搜索
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
    SEARCH_MAX_RESULT: int = 3
    DDGS_TIMEOUT: int = 10

    # Redis
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    # Agent控制
    MAX_REFLECT_LOOP: int = int(os.getenv("MAX_REFLECT_LOOP", 2))
    QUERY_MAX_LENGTH: int = int(os.getenv("QUERY_MAX_LENGTH", 1000))

    # Web服务
    SERVICE_HOST: str = "0.0.0.0"
    SERVICE_PORT: int = 8000
    CORS_ALLOW_ORIGINS: list = json.loads(os.getenv("CORS_ALLOW_ORIGINS", '["*"]'))

    def __init__(self):
        if not self.OLLAMA_BASE_URL:
            raise ValueError("环境变量 OLLAMA_BASE_URL 不能为空，请配置.env")
        if self.MAX_REFLECT_LOOP <= 0:
            raise ValueError("MAX_REFLECT_LOOP 必须大于0")

settings = Settings()