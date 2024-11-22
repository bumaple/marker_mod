import argparse
import asyncio
import json
import os
import time
from datetime import datetime
from typing import Tuple

import numpy as np
from lightrag import LightRAG
from lightrag.llm import openai_complete_if_cache, openai_embedding
from lightrag.utils import EmbeddingFunc
from lightrag.kg.neo4j_impl import Neo4JStorage

from loguru import logger

from marker.config_read import Config
from marker.database.pdf_data_operator import PDFDataOperator
from marker.logger import set_logru

# 全局变量
config: Config = None


async def llm_model_func(
        prompt, system_prompt=None, history_messages=None, **kwargs
) -> str:
    model_name = config.get_lightrag_param('build_model_name')
    if model_name is None:
        log_info = f"build model name is not exist!"
        logger.error(log_info)

    model_url = config.get_lightrag_param('build_model_url')
    if model_url is None:
        log_info = f"build model url is not exist!"
        logger.error(log_info)

    model_key = config.get_lightrag_param('build_model_key')

    if history_messages is None:
        history_messages = []
    return await openai_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=model_key,
        base_url=model_url,
        **kwargs,
    )


async def embedding_func(texts: list[str]) -> np.ndarray:
    embedding_name = config.get_lightrag_param('build_embedding_name')
    if embedding_name is None:
        log_info = f"build embedding name is not exist!"
        logger.error(log_info)

    embedding_url = config.get_lightrag_param('build_embedding_url')
    if embedding_url is None:
        log_info = f"build embedding url is not exist!"
        logger.error(log_info)

    embedding_key = config.get_lightrag_param('build_embedding_key')

    return await openai_embedding(
        texts,
        model=embedding_name,
        api_key=embedding_key,
        base_url=embedding_url,
    )


async def get_embedding_dim():
    test_text = ["This is a test sentence."]
    embedding = await embedding_func(test_text)
    embedding_dim = embedding.shape[1]
    return embedding_dim


async def convert_handler(pdf_data_opt, data_source, max_files, metadata_list, files) -> Tuple[int, int, str]:
    if len(files) == 0:
        return 0, 0, '待处理文件为空'

    working_dir = config.get_lightrag_param('working_dir')
    if working_dir is None:
        working_dir = os.path.join(os.getcwd(), 'working_neo4j')

    if not os.path.exists(working_dir):
        os.mkdir(working_dir)

    max_tokens = config.get_lightrag_param('max_tokens')
    if max_tokens is None:
        max_tokens = 8192
    else:
        max_tokens = int(max_tokens)

    # 图数据库类型
    graph_store = config.get_lightrag_param('graph_store')
    if graph_store is None:
        graph_store = 'NetworkXStorage'

    log_level = config.get_sys_param('log_level')

    embedding_dimension = await get_embedding_dim()
    logger.info(f"检测 embedding 维度: {embedding_dimension}")

    if graph_store == 'Neo4JStorage':
        # 初始化neo4j 数据库连接参数
        neo4j_url = config.get_neo4j_param('url')
        if neo4j_url is None:
            return 0, 0, f"neo4j url 不存在！"

        neo4j_user = config.get_neo4j_param('user')
        if neo4j_user is None:
            return 0, 0, f"neo4j user 不存在！"

        neo4j_password = config.get_neo4j_param('password')
        if neo4j_password is None:
            neo4j_password = ''

        neo4j_database = config.get_neo4j_param('database')
        if neo4j_database is None:
            neo4j_database = 'neo4j_lightrag'

        os.environ['NEO4J_URI'] = neo4j_url
        os.environ["NEO4J_USERNAME"] = neo4j_user
        os.environ["NEO4J_PASSWORD"] = neo4j_password

        neo4j_db = Neo4JStorage(namespace=neo4j_database, global_config={})

        rag = LightRAG(
            working_dir=working_dir,
            llm_model_func=llm_model_func,
            graph_storage=graph_store,
            log_level=log_level,
            embedding_func=EmbeddingFunc(
                embedding_dim=embedding_dimension,
                max_token_size=max_tokens,
                func=embedding_func,
            ),
        )
        rag.graph_storage_cls.db = neo4j_db
    else:
        rag = LightRAG(
            working_dir=working_dir,
            llm_model_func=llm_model_func,
            log_level=log_level,
            embedding_func=EmbeddingFunc(
                embedding_dim=embedding_dimension,
                max_token_size=max_tokens,
                func=embedding_func,
            ),
        )

    start_time = datetime.now()

    # 处理最大文件数
    if data_source == 'path' and max_files:
        files_to_convert = files[:max_files]
    else:
        files_to_convert = files

    files_number = len(files_to_convert)
    success_number = 0

    # 执行过程开始
    for idx, file_to_convert in enumerate(files_to_convert):
        file_name = os.path.basename(file_to_convert)
        metadata = metadata_list.get(file_name)
        if metadata is None:
            log_info = f"MetaData 中不存在 {file_name}！"
            logger.error(log_info)
            continue

        if data_source == 'db':
            if 'record_id' not in metadata:
                log_info = f"MetaData 中不存在 id！ {metadata}"
                logger.error(log_info)
                continue

            record_id = metadata['record_id']
        else:
            record_id = -1

        try:
            with open(file_to_convert, "r", encoding="utf-8") as f:
                # await rag.insert(f.read())
                await rag.ainsert(f.read())

                logger.info(f"处理文件成功! 第{idx + 1}个 文件名：{file_name}")
                success_number += 1

                if data_source == 'db' and record_id != '':
                    pdf_data_opt.update_pri_docx_kg_finish(record_id)
        except Exception as e:
            logger.error(f"处理文件失败！第{idx + 1}个 文件名：{file_name} 错误：{e}")

    # 执行过程结束

    end_time = datetime.now()
    # 计算实际执行的时间
    execution_time = end_time - start_time
    execution_seconds = execution_time.total_seconds()

    # 将执行时间转换为时分秒格式
    hours, remainder = divmod(execution_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    average_time = round(execution_seconds / len(files_to_convert))

    log_info = f" * * * * * 处理完成！文件数：{files_number} [成功 {success_number}，失败 {files_number - success_number}]。完成时间：{end_time.strftime('%Y-%m-%d %H:%M:%S')}。总处理时间：{int(hours)} 小时 {int(minutes)} 分 {int(seconds)} 秒，处理速度：{average_time} 秒/个"
    logger.info(log_info)
    return files_number, success_number, '处理完成'


async def get_data_from_db(pdf_data_opt, batch_number) -> Tuple[int, list, dict]:
    metadata_list = {}
    files = []

    records = pdf_data_opt.query_pri_docx_kg(batch_number)
    if len(records) <= 0:
        log_info = f"没有需要处理的数据！"
        logger.info(log_info)

    # 循环输出查询结果
    for row in records:
        record_id = row['id']
        word_title = row['title']
        json_file = row['json_file']
        file_name = os.path.basename(json_file)
        out_folder = os.path.dirname(json_file)
        if file_name.endswith('.json'):
            markdown_file = json_file.rsplit(".", 1)[0] + ".md"
            if os.path.isfile(markdown_file):
                files.append(markdown_file)
                markdown_file_name = os.path.basename(markdown_file)
                metadata_list[markdown_file_name] = {"out_path": out_folder,
                                                "record_id": record_id, "title": word_title}
            else:
                logger.warning(f"文件不存在：{markdown_file}")
        else:
            logger.warning(f"文件不是json格式，跳过：{json_file}")
    return 1, files, metadata_list


async def get_data_from_path(metadata_file_arg, in_folder_arg, out_folder_arg) -> Tuple[int, list, dict]:
    if metadata_file_arg:
        metadata_file = os.path.abspath(metadata_file_arg)
        with open(metadata_file, "r") as f:
            metadata_list = json.load(f)
    else:
        metadata_list = {}

    if in_folder_arg is None:
        logger.error(f"缺少 --in_folder 参数！")
        return 0, [], {}

    if out_folder_arg is None:
        if os.path.isfile(in_folder_arg):
            out_folder_arg = os.path.dirname(in_folder_arg)
        else:
            out_folder_arg = in_folder_arg

    in_folder = os.path.abspath(in_folder_arg)
    out_folder = os.path.abspath(out_folder_arg)

    if os.path.isfile(in_folder):
        files = [in_folder]
    else:
        os.makedirs(out_folder, exist_ok=True)
        files = [os.path.join(in_folder, f) for f in os.listdir(in_folder)]
        files = [f for f in files if os.path.isfile(f) and f.endswith('.md')]
    for file in files:
        file_name = os.path.basename(file)
        metadata_list[file_name] = {"out_path": out_folder, "record_id": ''}
    return 1, files, metadata_list


async def check_files_status(pdf_data_opt, max_files_arg) -> Tuple[int, list]:
    records = pdf_data_opt.query_all_valid_docx(max_files_arg)

    error_files = []
    # 循环输出查询结果
    for row in records:
        json_file = row['json_file']
        if json_file.endswith('.json'):
            if not os.path.isfile(json_file):
                error_files.append(json_file)
        else:
            error_files.append(json_file)
    return len(records), error_files


async def main():
    global config

    parser = argparse.ArgumentParser(description="转化文件到lightRAG.")
    parser.add_argument("--in_folder", help="Input folder with files.")
    parser.add_argument("--out_folder", help="Output folder with files.")
    parser.add_argument("--max", type=int, default=0, help="Maximum number of files to fix")
    parser.add_argument("--metadata_file", type=str, default=None, help="Metadata json file to use for filtering")

    parser.add_argument("--config_file", default='config.ini', help="config file.")
    # 增加操作类型，convert：识别转化PDF check：检查转化效果
    parser.add_argument("--run_type", default='convert', help="run type type (convert or check)")

    args = parser.parse_args()

    # 从配置文件中读取数据库配置
    in_folder_arg = args.in_folder
    out_folder_arg = args.out_folder
    max_files_arg = args.max
    metadata_file_arg = args.metadata_file
    config_file_arg = args.config_file
    run_type_arg = args.run_type.lower()

    config = Config(config_file_arg)

    data_source = config.get_sys_param('data_source')
    if data_source is None:
        data_source = ''
    data_source = data_source.lower()

    batch_number = config.get_sys_param('batch_number', int)
    if batch_number == 0:
        batch_number = 100

    sleep_minute = config.get_sys_param('sleep_minute', int)
    if sleep_minute == 0:
        sleep_minute = 10

    log_level = config.get_sys_param('log_level')
    if log_level is not None:
        set_logru(log_level=log_level)
    else:
        set_logru()

    if run_type_arg == 'convert':
        if data_source == 'db':
            pdf_data_opt = PDFDataOperator(config_file_arg)
            while True:
                result_code, files, metadata_list = await get_data_from_db(pdf_data_opt, batch_number)
                if result_code == 0:
                    return

                total_file_num, success_file_num, result_msg = await convert_handler(pdf_data_opt, data_source,
                                                                max_files_arg,
                                                                metadata_list, files)
                if success_file_num == 0:
                    return
                elif success_file_num > 0:
                    time.sleep(60)
                # else:
                #     time.sleep(sleep_minute * 60)
        elif data_source == 'path':
            result_code, files, metadata_list = await get_data_from_path(metadata_file_arg, in_folder_arg,
                                                                         out_folder_arg)
            if result_code == 0:
                return

            total_file_num, success_file_num, result_msg = await convert_handler(None, data_source, max_files_arg,
                                                            metadata_list,
                                                            files)
            if success_file_num == 0:
                return
        else:
            log_info = f"不支持的data_source参数！{data_source}"
            logger.error(log_info)
            return
    else:
        pdf_data_opt = PDFDataOperator(config_file_arg)
        record_cnt, error_files = await check_files_status(pdf_data_opt, max_files_arg)
        log_info = f" * * * * * {run_type_arg.capitalize()}ed 文件数：{record_cnt}。{len(error_files)} 个文件不存在！"
        logger.info(log_info)
        if len(error_files) > 0:
            for error_file in error_files:
                logger.info(error_file)


if __name__ == "__main__":
    asyncio.run(main())
