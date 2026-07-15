import logging
import os
from logging.handlers import RotatingFileHandler
from core.config import settings

# 日志存放目录
LOG_DIR = "logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# 日志格式
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

def setup_logger(name: str):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    # 避免重复handler
    if logger.handlers:
        return logger

    # 1. 文件滚动日志（单个文件最大10MB，最多保留5份）
    file_handler = RotatingFileHandler(
        filename=os.path.join(LOG_DIR, "multi_agent.log"),
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8"
    )
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
    file_handler.setLevel(logging.INFO)

    # 2. 控制台输出
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
    console_handler.setLevel(logging.INFO)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

# 全局日志实例
logger = setup_logger("MultiAgentSystem")