import time
from datetime import datetime
from typing import Tuple

import openai
from loguru import logger
from openai import BadRequestError

from marker.config_read import Config
from marker.logger import set_logru

class HttpUtils:

    def __init__(self, config_data: Config):
        """
        初始化解析器
        :param config_data: 配置文件对象
        """
        set_logru()

        self.model_name = config_data.get_llm_param('model')
        if self.model_name is None:
            log_info = f"model name is not exist!"
            print(log_info)
            logger.error(log_info)
            raise Exception(log_info)

        model_url = config_data.get_llm_param('url')
        if model_url is None:
            log_info = f"model url is not exist!"
            print(log_info)
            logger.error(log_info)
            raise Exception(log_info)

        api_key = config_data.get_llm_param('key')

        self.llm_temperature = config_data.get_llm_param('temperature')
        if self.llm_temperature is None:
            self.llm_temperature = 0
        else:
            self.llm_temperature = float(self.llm_temperature)

        self.llm_top_p = config_data.get_llm_param('top_p')
        if self.llm_top_p is None:
            self.llm_top_p = 0
        else:
            self.llm_top_p = float(self.llm_top_p)

        self.llm_max_tokens = config_data.get_llm_param('max_tokens')
        if self.llm_max_tokens is None:
            self.llm_max_tokens = 0
        else:
            self.llm_max_tokens = int(self.llm_max_tokens)

        self.attempt_limit = config_data.get_llm_param('attempt_limit')
        if self.attempt_limit is None:
            self.attempt_limit = 5
        else:
            self.attempt_limit = int(self.attempt_limit)

        self.attempt_sleep_second = config_data.get_llm_param('attempt_sleep_second')
        if self.attempt_sleep_second is None:
            self.attempt_sleep_second = 5
        else:
            self.attempt_sleep_second = int(self.attempt_sleep_second)

        openai.api_key = api_key
        openai.base_url = model_url

    def request_openai_api(self, prompt_head: str, text_content: str) -> Tuple[str, int]:
        params = {
            "model": self.model_name,
            "messages": [{"role": "system", "content": f"{prompt_head}"},
                         {"role": "user", "content": f"以下是需要处理的文本：\n{text_content}"}],
            "max_tokens": self.llm_max_tokens
        }
        if self.llm_temperature not in [None, 0]:
            params["temperature"] = self.llm_temperature
        if self.llm_top_p not in [None, 0]:
            params["top_p"] = self.llm_top_p
        if self.llm_max_tokens not in [None, 0]:
            params["max_tokens"] = self.llm_max_tokens

        # 发送请求，如果不成功停止5秒后重发，重复3次
        for attempt in range(self.attempt_limit):
            try:
                start_time = datetime.now()
                log_info = f"LLM请求开始 {start_time.strftime('%Y-%m-%d %H:%M:%S')}"
                logger.info(log_info)

                completion = openai.chat.completions.create(**params)

                end_time = datetime.now()
                # 计算实际执行的时间
                execution_time = end_time - start_time
                execution_seconds = execution_time.total_seconds()
                log_info = f"LLM请求结束 {end_time.strftime('%Y-%m-%d %H:%M:%S')}, 耗时 {int(execution_seconds)} 秒"
                logger.info(log_info)

                return completion.choices[0].message.content, 1
            except BadRequestError as e:
                if e.code == 'RequestTimeOut':
                    log_info = f"LLM请求超时 第 {attempt + 1}/{self.attempt_limit} 次. {self.attempt_sleep_second} 秒后重试..."
                    logger.error(log_info)
                    time.sleep(self.attempt_sleep_second)
                else:
                    log_info = f"LLM请求错误: {e}"
                    logger.error(log_info)
                    return '', 0
            except Exception as e:
                log_info = f"LLM请求错误: {e}"
                logger.error(log_info)
                return '', 0