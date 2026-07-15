import os
from dotenv import load_dotenv

# 加载.env文件
load_dotenv()

class Settings:
    # Ollama 大模型配置
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "qwen2.5:7b")
    EMBED_MODEL: str = os.getenv("EMBED_MODEL", "bge-m3")

    # 向量库配置
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    RETRIEVE_TOP_K: int = 3

    # 联网搜索配置
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
    SEARCH_MAX_RESULT: int = 3

    # FastAPI服务配置
    SERVICE_HOST: str = "0.0.0.0"
    SERVICE_PORT: int = 8000

    def __init__(self):
        # 启动校验Ollama地址必填
        if not self.OLLAMA_BASE_URL:
            raise ValueError("请在.env配置 OLLAMA_BASE_URL")

# 全局单例配置对象
settings = Settings()