import argparse
import json
import os
import time
from datetime import datetime
from typing import Tuple

from loguru import logger
from pika import BasicProperties

from marker.config_read import Config
from marker.database.pdf_data_operator import PDFDataOperator
from marker.msgqueue.rabbitmq import RabbitMessageQueue
from marker.logger import set_logru

set_logru()


def get_data_from_db(pdf_data_opt, batch_number) -> Tuple[int, list, dict]:
    metadata_list = {}
    files = []

    records = pdf_data_opt.query_need_ocr_v2(batch_number)
    if len(records) <= 0:
        log_info = f"没有需要处理的数据！"
        logger.info(log_info)

    # 循环输出查询结果
    for row in records:
        record_id = row['ID']
        pdf_file = row['PDF_FILE']
        pdf_title = row['TITLE']
        ocr_types = row['OCR_TYPES']
        ocr_priority = row['OCR_PRIORITY']
        file_name = os.path.basename(pdf_file)
        out_folder = os.path.dirname(pdf_file)
        # os.makedirs(out_folder, exist_ok=True)
        if file_name.endswith('.pdf'):
            metadata_list[pdf_file] = {"languages": ["Chinese", "English"], "out_path": out_folder,
                                       "record_id": record_id, "title": pdf_title, "ocr_types": ocr_types,
                                       "ocr_priority": ocr_priority}
            if os.path.isfile(pdf_file):
                files.append(pdf_file)
    return 1, files, metadata_list


def get_data_from_path(metadata_file_arg, in_folder_arg, out_folder_arg) -> Tuple[int, list, dict]:
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
        out_folder_arg = in_folder_arg

    in_folder = os.path.abspath(in_folder_arg)
    out_folder = os.path.abspath(out_folder_arg)

    if os.path.isfile(in_folder):
        files = [in_folder]
    else:
        os.makedirs(out_folder, exist_ok=True)
        files = [os.path.join(in_folder, f) for f in os.listdir(in_folder)]
        files = [f for f in files if os.path.isfile(f)]

    for file in files:
        file_name = os.path.basename(file)
        metadata_list[file_name] = {"languages": ["Chinese", "English"], "out_path": out_folder}
    return 1, files, metadata_list


def convert_handler(pdf_data_opt, data_source, max_files, config_file, metadata_list, files) -> int:
    if len(files) == 0:
        return 1

    start_time = datetime.now()

    # 处理最大文件数
    if data_source == 'path' and max_files:
        files_to_convert = files[:max_files]
    else:
        files_to_convert = files

    files_number = len(files_to_convert)
    # log_info = f" * * * * * 待处理文件放入队列 文件数：{files_number}。开始时间：{start_time.strftime('%Y-%m-%d %H:%M:%S')}"
    # logger.info(log_info)

    # 连接到RabbitMQ服务器
    msg_queue = RabbitMessageQueue(config_file)
    result_code, mq_connection = msg_queue.connect()
    mq_channel = mq_connection.channel()
    msg_durable = msg_queue.get_durable()

    if result_code == 0:
        log_info = f"RabbitMQ 连接失败!"
        logger.error(log_info)
        return 0

    # 执行过程开始
    for idx, file_to_convert in enumerate(files_to_convert):
        metadata = metadata_list.get(file_to_convert)

        if 'record_id' not in metadata:
            log_info = f"MetaData 中不存在 ID！ {metadata}"
            logger.error(log_info)
            return 0

        if 'ocr_types' not in metadata:
            log_info = f"MetaData 中不存在 OCR_Types！ {metadata}"
            logger.error(log_info)
            return 0

        ocr_priority = 0
        if 'ocr_priority' in metadata:
            # 消息优先级，RabbitMQ，0最低，9最高
            ocr_priority = metadata['ocr_priority']

        record_id = metadata['record_id']
        ocr_type_strs = metadata['ocr_types']
        ocr_type_list = ocr_type_strs.split(';')
        for ocr_type in ocr_type_list:
            # 消息路由关键字
            routing_key = RabbitMessageQueue.get_route_key(ocr_type, ocr_priority)
            # 声明一个交换机,类型为关键字模式 使消息持久化
            mq_channel.exchange_declare(exchange=RabbitMessageQueue.EXCHANGE_NAME, exchange_type='direct', durable=msg_durable, )
            # 向交换机发送消息,并告诉交换机只发给绑定了关键字的消费者队列 使消息持久化
            if msg_durable:
                properties = BasicProperties(delivery_mode=2, priority=ocr_priority, )
            else:
                properties = BasicProperties(priority=ocr_priority, )
            mq_channel.basic_publish(exchange=RabbitMessageQueue.EXCHANGE_NAME, routing_key=routing_key,
                                     body=json.dumps(metadata), properties=properties)

            logger.debug(f"写入MQ成功！Key:{routing_key} {idx+1} {metadata}")

        if data_source == 'db':
            pdf_data_opt.update_pri_finish_orc_start(record_id)

    # 关闭连接
    msg_queue.close()
    # 执行过程结束

    end_time = datetime.now()
    # 计算实际执行的时间
    execution_time = end_time - start_time
    execution_seconds = execution_time.total_seconds()

    # 将执行时间转换为时分秒格式
    hours, remainder = divmod(execution_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    average_time = round(execution_seconds / len(files_to_convert))

    log_info = f" * * * * * 待处理文件放入队列 文件数：{files_number}。完成时间：{end_time.strftime('%Y-%m-%d %H:%M:%S')}。总处理时间：{int(hours)} 小时 {int(minutes)} 分 {int(seconds)} 秒，处理速度：{average_time} 秒/个"
    logger.info(log_info)
    return len(files_to_convert)


def check_files_status(pdf_data_opt, max_files_arg) -> Tuple[int, list]:
    records = pdf_data_opt.query_sub_finish_ocr_all(max_files_arg)

    error_files = []
    # 循环输出查询结果
    for row in records:
        # record_id = row['ID']
        md_file_path = row['MD_FILE_DIR']
        md_file_name = row['MD_FILE_NAME']
        md_file = os.path.join(md_file_path, md_file_name)
        if md_file.endswith('.md'):
            if not os.path.isfile(md_file):
                error_files.append(md_file)
        else:
            error_files.append(md_file)
    return len(records), error_files


def main():
    parser = argparse.ArgumentParser(description="转化PDF文件为Markdown文件")
    parser.add_argument("--in_folder", type=str, help="待转化文件或目录，数据库模式无效")
    parser.add_argument("--out_folder", type=str, help="输出目录，数据库模式无效")
    parser.add_argument("--metadata_file", type=str, default=None, help="MetadataJson文件位置，数据库模式无效")
    parser.add_argument("--max", type=int, default=0, help="最大处理文件数量，数据库模式无效")
    parser.add_argument("--config_file", type=str, default='config.ini', help="配置文件 默认：config.ini")
    # 增加操作类型，convert：识别转化PDF check：检查转化效果
    parser.add_argument("--run_type", type=str, default='convert',
                        help="运行操作类型 (转化：convert，检查：check) 默认：convert")

    args = parser.parse_args()

    # 增加读取配置文件中数据库信息，通过数据库记录形式取代通过metadata_file方式操作多文件
    # 从配置文件中读取数据库配置
    in_folder_arg = args.in_folder
    out_folder_arg = args.out_folder
    metadata_file_arg = args.metadata_file
    max_files_arg = args.max
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

    pdf_data_opt = PDFDataOperator(config_file_arg)

    if run_type_arg == 'convert':
        if data_source == 'db':
            while True:
                result_code, files, metadata_list = get_data_from_db(pdf_data_opt, batch_number)
                if result_code == 0:
                    return

                result_code = convert_handler(pdf_data_opt, data_source, max_files_arg, config_file_arg,
                                              metadata_list, files)
                if result_code == 0:
                    return
                if len(files) == batch_number:
                    time.sleep(60)
                else:
                    time.sleep(sleep_minute * 60)
        elif data_source == 'path':
            result_code, files, metadata_list = get_data_from_path(metadata_file_arg, in_folder_arg, out_folder_arg)
            if result_code == 0:
                return

            result_code = convert_handler(pdf_data_opt, data_source, max_files_arg, config_file_arg, metadata_list,
                                          files)
            if result_code == 0:
                return
        else:
            log_info = f"不支持的data_source参数！{data_source}"
            logger.error(log_info)
            return
    else:
        pdf_data_opt = PDFDataOperator(config_file_arg)
        record_cnt, error_files = check_files_status(pdf_data_opt, max_files_arg)
        log_info = f" * * * * * {run_type_arg.capitalize()}ed 文件数：{record_cnt}。{len(error_files)} 个文件不存在！"
        logger.info(log_info)
        if len(error_files) > 0:
            for error_file in error_files:
                logger.info(error_file)


if __name__ == "__main__":
    main()
