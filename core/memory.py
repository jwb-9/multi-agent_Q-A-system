import redis
import uuid
from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama import ChatOllama
from core.config import settings
from core.logger import logger

# Redis连接，捕获连接异常
try:
    redis_client = redis.from_url(settings.REDIS_URL, decode_responses=False)
    redis_client.ping()
except Exception as e:
    logger.error(f"Redis连接失败: {str(e)}")
    redis_client = None

llm = ChatOllama(
    base_url=settings.OLLAMA_BASE_URL,
    model=settings.LLM_MODEL,
    temperature=0,
    timeout=settings.OLLAMA_TIMEOUT
)

class MemoryManager:
    @staticmethod
    def _check_redis():
        if redis_client is None:
            raise ConnectionError("Redis服务不可用，记忆功能失效")

    @staticmethod
    def get_session_short_memory(session_id: str) -> str:
        try:
            MemoryManager._check_redis()
            key = f"short_memory:{session_id}"
            data = redis_client.get(key)
            return data.decode("utf-8") if data else ""
        except Exception as e:
            logger.warning(f"读取短期记忆失败: {str(e)}")
            return ""

    @staticmethod
    def save_short_memory(session_id: str, content: str):
        try:
            MemoryManager._check_redis()
            key = f"short_memory:{session_id}"
            # 短期记忆1小时过期
            redis_client.setex(key, 3600, content.encode("utf-8"))
        except Exception as e:
            logger.warning(f"保存短期记忆失败: {str(e)}")

    @staticmethod
    def compress_dialogue(messages: list) -> str:
        dialog_text = "\n".join([f"{type(m).__name__}:{m.content}" for m in messages[-6:]])
        prompt = f"精简对话关键信息，200字内：\n{dialog_text}"
        try:
            res = llm.invoke(prompt)
            return res.content[:300]
        except Exception as e:
            logger.error(f"对话摘要失败: {str(e)}")
            return dialog_text[:300]

    @staticmethod
    def get_long_memory(session_id: str) -> str:
        try:
            MemoryManager._check_redis()
            key = f"long_memory:{session_id}"
            data = redis_client.get(key)
            return data.decode("utf-8") if data else "无历史问答经验"
        except Exception as e:
            logger.warning(f"读取长期记忆失败: {str(e)}")
            return "无历史问答经验"

    @staticmethod
    def append_long_memory(session_id: str, query: str, answer: str):
        try:
            MemoryManager._check_redis()
            old = MemoryManager.get_long_memory(session_id)
            raw = f"{old}\n【用户】{query}\n【回答】{answer}"
            compress_prompt = f"整合问答提炼长期记忆要点，控制300字内：{raw}"
            compress_res = llm.invoke(compress_prompt)
            key = f"long_memory:{session_id}"
            # 长期记忆7天过期
            redis_client.setex(key, 86400 * 7, compress_res.content.encode("utf-8"))
        except Exception as e:
            logger.warning(f"更新长期记忆失败: {str(e)}")

# 生成会话ID
def new_session_id() -> str:
    return str(uuid.uuid4())