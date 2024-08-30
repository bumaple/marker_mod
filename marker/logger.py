import logging
import warnings

import os
from logging.handlers import TimedRotatingFileHandler


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
    logger = logging.getLogger(log_name)
    logger.setLevel(log_level)

    # 避免重复添加处理器
    if len(logger.handlers) == 0:
        # 创建一个 TimedRotatingFileHandler，按天轮换日志文件
        handler = TimedRotatingFileHandler(log_file_path, when="midnight", interval=1, backupCount=7)
        handler.suffix = "%Y-%m-%d"  # 设置文件名后缀为日期
        handler.setFormatter(logging.Formatter('%(asctime)s [%(threadName)s] %(message)s'))

        # 添加处理器到日志记录器
        logger.addHandler(handler)

    return logger