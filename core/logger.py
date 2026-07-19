import logging
import os
from logging.handlers import RotatingFileHandler
from contextvars import ContextVar
from core.config import settings

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# 线程安全上下文存储trace_id
trace_id_ctx = ContextVar("trace_id", default="unknown")

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - trace_id:%(trace_id)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class TraceFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None, style='%'):
        super().__init__(fmt=fmt, datefmt=datefmt or DATE_FORMAT, style=style)

    def format(self, record):
        record.trace_id = trace_id_ctx.get()
        return super().format(record)

def setup_logger(name: str):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger

    file_handler = RotatingFileHandler(
        filename=os.path.join(LOG_DIR, "multi_agent.log"),
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8"
    )
    file_handler.setFormatter(TraceFormatter(LOG_FORMAT))
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(TraceFormatter(LOG_FORMAT))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

logger = setup_logger("MultiAgentSystem")

# 上下文工具函数
def set_trace_id(trace_id: str):
    trace_id_ctx.set(trace_id)