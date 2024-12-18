import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple

from marker.config_read import Config
from marker.msgqueue.rabbitmq import RabbitMessageQueue

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1" # For some reason, transformers decided to use .isin for a simple op, which is not supported on MPS
os.environ["IN_STREAMLIT"] = "true" # Avoid multiprocessing inside surya
os.environ["PDFTEXT_CPU_WORKERS"] = "1" # Avoid multiprocessing inside pdftext

# 本机开发使用，正式注释
os.environ["TORCH_DEVICE"] = "cpu"
os.environ["OCR_ENGINE"] = "ocrmypdf"

import pypdfium2 # Needs to be at the top to avoid warnings
import argparse
import torch.multiprocessing as mp
from tqdm import tqdm
import math

from marker.convert import convert_single_pdf
from marker.output import markdown_exists, save_markdown
from marker.pdf.utils import find_filetype
from marker.pdf.extract_text import get_length_of_text
from marker.models import load_all_models
from marker.settings import settings
# from marker.logger import configure_logging
import traceback
import json


# configure_logging()


# 增加 2024-08-13 begin
from datetime import datetime
from marker.database.pdf_data_operator import PDFDataOperator
from marker.logger import set_logru

from loguru import logger

set_logru()
# 增加 2024-08-13 end

def worker_init(shared_model):
    if shared_model is None:
        shared_model = load_all_models()

    global model_refs
    model_refs = shared_model


def worker_exit():
    global model_refs
    del model_refs


def process_single_pdf(args):
    files_number, idx, filepath, out_folder, metadata, min_length, config_file, ocr_type = args

    fname = os.path.basename(filepath)
    if markdown_exists(out_folder, fname, ocr_type):
        return

    try:
        # Skip trying to convert files that don't have a lot of embedded text
        # This can indicate that they were scanned, and not OCRed properly
        # Usually these files are not recent/high-quality
        if min_length:
            filetype = find_filetype(filepath)
            if filetype == "other":
                return 0

            length = get_length_of_text(filepath)
            if length < min_length:
                return

        full_text, images, out_metadata = convert_single_pdf(filepath, model_refs, metadata=metadata)
        if len(full_text.strip()) > 0:
            # 生成目录增加时间戳 2024-08-13 begin
            record_id = None
            title = ''
            if 'out_path'in metadata:
                out_folder = metadata['out_path']
            if 'record_id' in metadata:
                record_id = metadata['record_id']
            if 'title' in metadata:
                title = metadata['title']
            md_path = save_markdown(out_folder, fname, full_text, images, out_metadata, ocr_type)
            md_filename = fname.rsplit(".", 1)[0] + ".md"
            if record_id is not None:
                pdf_data_opt = PDFDataOperator(config_file)
                pdf_data_opt.update_pri_finish_orc(record_id, 1)

                record_num = pdf_data_opt.get_sub_record_number(record_id)
                sub_record_id = record_id + '_' + str(int(record_num) + 1).zfill(3)
                pdf_data_opt.insert_sub_finish_ocr(record_id, sub_record_id, ocr_type, title, md_path, md_filename)

            md_fullname = os.path.join(md_path, md_filename)
            # 计算百分比
            percentage = ((idx + 1) / files_number) * 100
            log_info = f" * * * * * Converting {idx+1}/{files_number}({percentage:.2f}%) {fname}, id:{record_id}, storing in {md_fullname}"
            logger.info(log_info)
            # end
        else:
            log_info = f"Empty file: {filepath}.  Could not convert."
            logger.info(log_info)
    except Exception as e:
        log_info = f"Error converting {filepath}: {e}"
        logger.error(log_info)


def get_data_from_db(pdf_data_opt, ocr_type, max_files) -> Tuple[int, list, dict]:
    metadata_list = {}
    files = []

    records = pdf_data_opt.query_need_ocr_v1(ocr_type, max_files)
    if len(records) <= 0:
        log_info = f"没有需要处理的数据！"
        logger.info(log_info)

    # 循环输出查询结果
    for row in records:
        record_id = row['ID']
        pdf_file = row['PDF_FILE']
        pdf_title = row['TITLE']
        file_name = os.path.basename(pdf_file)
        out_folder = os.path.dirname(pdf_file)
        os.makedirs(out_folder, exist_ok=True)
        if file_name.endswith('.pdf'):
            metadata_list[pdf_file] = {"languages": ["Chinese", "English"], "out_path": out_folder,
                                  "record_id": record_id, "title": pdf_title}
            if os.path.isfile(pdf_file):
                files.append(pdf_file)

    return 1, files, metadata_list


def get_data_from_path(metadata_file_arg, in_folder_arg, out_folder_arg) -> Tuple[int, list, dict]:
    if metadata_file_arg:
        metadata_file = os.path.abspath(metadata_file_arg)
        with open(metadata_file, "r") as f:
            metadata_list = json.load(f)

    in_folder = os.path.abspath(in_folder_arg)
    out_folder = os.path.abspath(out_folder_arg)
    files = [os.path.join(in_folder, f) for f in os.listdir(in_folder)]
    files = [f for f in files if os.path.isfile(f)]
    os.makedirs(out_folder, exist_ok=True)
    return 1, files, metadata_list


def get_data_from_mq_loop(config_file, ocr_type):
    # 连接到RabbitMQ服务器
    msg_queue = RabbitMessageQueue(config_file)
    result_code, mq_connection = msg_queue.connect()
    mq_channel = mq_connection.channel()
    msg_durable = msg_queue.get_durable()

    mq_channel.exchange_declare(exchange=RabbitMessageQueue.EXCHANGE_NAME, exchange_type='direct', durable=msg_durable)

    result = mq_channel.queue_declare(queue=RabbitMessageQueue.QUEUE_NAME, exclusive=True, durable=msg_durable)

    for ocr_priority in range(9, -1, -1):
        route_key = RabbitMessageQueue.get_route_key(ocr_type, ocr_priority)
        mq_channel.queue_bind(exchange=RabbitMessageQueue.EXCHANGE_NAME, queue=RabbitMessageQueue.QUEUE_NAME, routing_key=route_key)
        logger.info(f" Queue:{RabbitMessageQueue.QUEUE_NAME} Key:{route_key} 启动")

    mq_channel.basic_consume(queue=RabbitMessageQueue.QUEUE_NAME, auto_ack=False, on_message_callback=message_recive_callback, )
    mq_channel.basic_qos(prefetch_count=1)
    mq_channel.start_consuming()


def message_recive_callback(channel, method, properties, body):
    logger.info(" [x] Received %r" % json.loads(body))
    channel.basic_ack(delivery_tag=method.delivery_tag)

def main():
    parser = argparse.ArgumentParser(description="Convert multiple pdfs to markdown.")
    parser.add_argument("--in_folder", help="Input folder with pdfs.")
    parser.add_argument("--out_folder", help="Output folder")
    parser.add_argument("--chunk_idx", type=int, default=0, help="Chunk index to convert")
    parser.add_argument("--num_chunks", type=int, default=1, help="Number of chunks being processed in parallel")
    parser.add_argument("--max", type=int, default=0, help="Maximum number of pdfs to convert")
    parser.add_argument("--workers", type=int, default=5, help="Number of worker processes to use.  Peak VRAM usage per process is 5GB, but avg is closer to 3.5GB.")
    parser.add_argument("--metadata_file", type=str, default=None, help="Metadata json file to use for languages")
    parser.add_argument("--min_length", type=int, default=None, help="Minimum length of pdf to convert")

    # 增加读取配置文件中数据库信息，通过数据库记录形式取代通过meta_file方式操作多文件 2024-08-13
    parser.add_argument("--config_file", default='config.ini', help="config file.")
    # 增加操作类型，convert：识别转化PDF check：检查转化效果
    parser.add_argument("--run_type", default='convert', help="run type type (convert or check)")
    parser.add_argument("--ocr_type", type=str, default='10', help="OCR type (10:marker mod 20:MinerU mod)")

    args = parser.parse_args()

    # 增加读取配置文件中数据库信息，通过数据库记录形式取代通过meta_file方式操作多文件 2024-08-13
    # begin
    # 从配置文件中读取数据库配置
    in_folder_arg = args.in_folder
    out_folder_arg = args.out_folder
    metadata_file_arg = args.metadata_file
    max_files_arg = args.max
    config_file_arg = args.config_file
    run_type_arg = args.run_type.lower()
    ocr_type_arg = args.ocr_type

    config = Config(config_file_arg)

    data_source = config.get_sys_param('data_source')
    if data_source is None:
        data_source = ''
    data_source = data_source.lower()

    start_time = datetime.now()

    pdf_data_opt = PDFDataOperator(config_file_arg)
    if run_type_arg == 'convert':
        if data_source == 'db':
            # 数据库
            result_code, files, metadata_list = get_data_from_db(pdf_data_opt, ocr_type_arg, max_files_arg)
            if result_code == 0:
                return

        elif data_source == 'path':
            # 文件夹
            result_code, files, metadata_list = get_data_from_path(metadata_file_arg, in_folder_arg, out_folder_arg)
            if result_code == 0:
                return
        elif data_source == 'mq':
            # RabbitMQ TODO
            get_data_from_mq_loop(config_file_arg, ocr_type_arg)
        else:
            log_info = f"不支持的data_source参数！{data_source}"
            logger.error(log_info)
            return
        # end

        if data_source != 'mq':
            # Handle chunks if we're processing in parallel
            # Ensure we get all files into a chunk
            chunk_size = math.ceil(len(files) / args.num_chunks)
            start_idx = args.chunk_idx * chunk_size
            end_idx = start_idx + chunk_size
            files_to_convert = files[start_idx:end_idx]

            # Limit files converted if needed
            if args.max:
                files_to_convert = files_to_convert[:args.max]

            total_processes = min(len(files_to_convert), args.workers)

            try:
                mp.set_start_method('spawn') # Required for CUDA, forkserver doesn't work
            except RuntimeError:
                raise RuntimeError("Set start method to spawn twice. This may be a temporary issue with the script. Please try running it again.")

            if settings.TORCH_DEVICE == "mps" or settings.TORCH_DEVICE_MODEL == "mps":
                log_info = "Cannot use MPS with torch multiprocessing share_memory. This will make things less memory efficient. If you want to share memory, you have to use CUDA or CPU.  Set the TORCH_DEVICE environment variable to change the device."
                logger.info(log_info)
                model_lst = None
            else:
                model_lst = load_all_models()

                for model in model_lst:
                    if model is None:
                        continue
                    model.share_memory()

            files_number = len(files_to_convert)
            log_info = f" * * * * * {args.run_type.capitalize()}ing {files_number} pdfs in chunk {args.chunk_idx + 1}/{args.num_chunks} with {total_processes} processes. Started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}"
            logger.info(log_info)
            task_args = [(files_number, idx, f, out_folder_arg, metadata_list.get(f), args.min_length, config_file_arg, ocr_type_arg) for idx, f in enumerate(files_to_convert)]

            with mp.Pool(processes=total_processes, initializer=worker_init, initargs=(model_lst,)) as pool:
                list(tqdm(pool.imap(process_single_pdf, task_args), total=len(task_args), desc="Processing PDFs", unit="pdf"))

                pool._worker_handler.terminate = worker_exit

            # Delete all CUDA tensors
            del model_lst

            end_time = datetime.now()
            # 计算实际执行的时间
            execution_time = end_time - start_time
            execution_seconds = execution_time.total_seconds()

            # 将执行时间转换为时分秒格式
            hours, remainder = divmod(execution_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)

            average_time = round(execution_seconds / len(files_to_convert))

            log_info = f" * * * * * {args.run_type.capitalize()}ed {files_number} pdfs. Ended at {end_time.strftime('%Y-%m-%d %H:%M:%S')}. Total execution time {int(hours)}hour {int(minutes)}min {int(seconds)}sec, average time {average_time}src/record"
            logger.info(log_info)
    else:
        pdf_data_opt = PDFDataOperator(config_file_arg)
        records = pdf_data_opt.query_sub_finish_ocr(ocr_type_arg, max_files_arg)

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

        log_info = f" * * * * * {args.run_type.capitalize()}ed {len(records)} pdfs. {len(error_files)} files not exist!"
        logger.info(log_info)
        if len(error_files) > 0:
            for error_file in error_files:
                print(error_file)
                logger.info(error_file)

if __name__ == "__main__":
    main()