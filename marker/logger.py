import logging
import sys
import warnings

import os
from logging.handlers import TimedRotatingFileHandler
from loguru import logger


def configure_logging():
    logging.basicConfig(level=logging.WARNING)

    logging.getLogger('pdfminer').setLevel(logging.ERROR)
    logging.getLogger('PIL').setLevel(logging.ERROR)
    logging.getLogger('fitz').setLevel(logging.ERROR)
    logging.getLogger('ocrmypdf').setLevel(logging.ERROR)
    warnings.simplefilter(action='ignore', category=FutureWarning)


def setup_logger(log_name="PDFConvert", log_directory="logs", log_filename="run.log", log_level=logging.INFO):
    # 确保日志目录存在
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    # 设置日志文件路径
    log_file_path = os.path.join(log_directory, log_filename)

    # 创建日志记录器
    my_logger = logging.getLogger(log_name)
    my_logger.setLevel(log_level)

    # 避免重复添加处理器
    if len(my_logger.handlers) == 0:
        # 创建一个 TimedRotatingFileHandler，按天轮换日志文件
        handler = TimedRotatingFileHandler(log_file_path, when="midnight", interval=1, backupCount=7)
        handler.suffix = "%Y-%m-%d"  # 设置文件名后缀为日期
        handler.setFormatter(logging.Formatter('%(asctime)s [%(threadName)s] %(message)s'))

        # 添加处理器到日志记录器
        my_logger.addHandler(handler)

    return my_logger

def set_logru(log_directory="logs", log_level='INFO'):
    # 确保日志目录存在
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    # 将日志输出到控制台
    logger.remove()  # 移除默认的handler，如果有的话
    logger.add(sys.stdout, enqueue=True, colorize=True, backtrace=True, diagnose=True)

    logger.add(sink=os.path.join(log_directory, "run.log"), level=log_level, rotation="1 days",retention="30 days", encoding="utf-8", enqueue=True, colorize=False, backtrace=True, diagnose=True,)

    # 设置不同级别的日志输出文件
    # logger.add("debug.log", level="DEBUG", rotation="10 MB", filter=lambda record: record["level"].name == "DEBUG")
    # logger.add("info.log", level="INFO", rotation="10 MB", filter=lambda record: record["level"].name == "INFO")
    # logger.add("warning.log", level="WARNING", rotation="10 MB", filter=lambda record: record["level"].name == "WARNING")
    logger.add(sink=os.path.join(log_directory, "error.log"), level="ERROR", rotation="10 MB", encoding="utf-8", enqueue=True, colorize=False, backtrace=True, diagnose=True, filter=lambda record: record["level"].name == "ERROR")
