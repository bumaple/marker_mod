import os
import argparse
import traceback
import json

from datetime import datetime
from typing import Dict, Tuple

from marker.database.pdf_data_operator import PDFDataOperator
from marker.logger import setup_logger
from marker.output import save_markdown_fix
from marker.config_read import Config

from langchain.text_splitter import MarkdownTextSplitter

import openai

logger = setup_logger()

OCR_TYPES = ['10', '20']
FIX_OCR_TYPES = ['11', '21']

def chunk_markdown_with_langchain(text, chunk_size):
    if len(text) > chunk_size:
        splitter = MarkdownTextSplitter(chunk_size=chunk_size)  # 定义分块大小
        chunks = splitter.split_text(text)  # 对文本进行分块
    else:
        chunks = [text]
    return chunks

def fix_single_file(filepath: str, config_file: str, chunk_size: int) -> Tuple[str, Dict]:
    config = Config(config_file)

    with open(filepath, 'r', encoding='utf-8') as file:
        file_content = file.read()

    model_name = config.get_llm_param('model')
    if model_name is None:
        log_info = f"model name is not exist!"
        print(log_info)
        logger.error(log_info)
        return "", {}
    model_url = config.get_llm_param('url')
    if model_url is None:
        log_info = f"model url is not exist!"
        print(log_info)
        logger.error(log_info)
        return "", {}

    api_key = config.get_llm_param('key')

    llm_temperature = config.get_llm_param('temperature')
    if llm_temperature is None:
        llm_temperature = 0.9
    llm_top_p = config.get_llm_param('top_p')
    if llm_top_p is None:
        llm_top_p = 0.8
    llm_max_tokens= config.get_llm_param('max_tokens')
    if llm_max_tokens is None:
        llm_max_tokens = 4096

    out_meta = {
        "model": model_name,
        "temperature": eval(llm_temperature),
        "top_p": eval(llm_top_p),
        "max_tokens": eval(llm_max_tokens),
        "chunk_size": chunk_size,
        "txt_length": len(file_content)
    }

    prompt_head = "你是Markdown文件的处理专家，纠正以下Markdown文本的错误，保内容与前文连贯。请遵循以下要求：\n" + \
                  "1.修正OCR引起的文字错误和其他错误；\n" + \
                  "2.使用上下文和常识来纠正错误；\n" + \
                  "3.只修正错误,不要修改无错误的内容；\n" + \
                  "4.不要添加额外的句号或其他不必要的标点符号，删除前后无关联且无意义的符号，不增加Markdown代码及语法高亮标记；\n" + \
                  "5.保持原始结构及所有标题和副标题的完整性，标题补充必要的Markdown标记；\n" + \
                  "6.保留所有Markdown标记，保留所有原始格式，包括换行符；\n" + \
                  "7.保留所有中文、数字、字母组成的编号和序号，编号及后续的标题结束后进行换行；\n" + \
                  "8.保留原文中的所有重要信息，不添加原文中不存在的任何新信息；\n" + \
                  "9.删除句子或段落中的不必要换行，保持段落的断行，删除句子中不必要的空格；\n" + \
                  "10.确保内容与前文顺畅衔接，适当处理在句子中间开始或结束的文本使其通顺；\n" + \
                  "11.只回复经过修正的文本，不添加任何引言、解释或元数据。\n" + \
                  "下面是需要纠正的内容：\n########\n"

    try:
        resp_contents = []

        openai.api_key = api_key
        openai.base_url = model_url

        chunks = chunk_markdown_with_langchain(file_content, chunk_size)

        for idx, chunk in enumerate(chunks):
            req_content = prompt_head + chunk

            start_time = datetime.now()

            # create a chat completion
            completion = openai.chat.completions.create(
                model = model_name,
                messages = [{"role": "user", "content": f"{req_content}"}],
                temperature = llm_temperature,
                top_p = llm_top_p,
                max_tokens = llm_max_tokens
            )

            resp_contents.append(completion.choices[0].message.content)

            end_time = datetime.now()
            # 计算实际执行的时间
            execution_time = end_time - start_time
            execution_seconds = execution_time.total_seconds()

            log_info = f"   LLM request: {idx + 1}/{len(chunks)}, execution time {int(execution_seconds)}sec {filepath}"
            print(log_info)
            logger.info(log_info)
        out_meta["chunk_num"] = len(chunks)
        out_meta["fix_stats"] = "success"
        return "".join(resp_contents), out_meta
    except Exception as e:
        out_meta["fix_stats"] = "fail"
        log_info = f"Error fixing {filepath}: {e}"
        print(log_info)
        logger.error(log_info)
        return "", out_meta


def process_single_file(files_number, idx, filepath, out_folder, metadata, config_file, chunk_size):

    fname = os.path.basename(filepath)
    # md_file = os.path.join(out_folder, fname)
    if not os.path.exists(filepath):
        log_info = f"File not exist: {filepath}."
        print(log_info)
        logger.error(log_info)
        return

    try:

        full_text, out_metadata = fix_single_file(filepath, config_file, chunk_size)
        if len(full_text.strip()) > 0:
            record_id = None
            parent_record_id = None
            title = ''
            ocr_type = ''
            if out_folder is None and 'out_path'in metadata:
                out_folder = metadata['out_path']
            if 'record_id' in metadata:
                record_id = metadata['record_id']
            if 'parent_record_id' in metadata:
                parent_record_id = metadata['parent_record_id']
            if 'ocr_type' in metadata:
                ocr_type = metadata['ocr_type']
            if 'title' in metadata:
                title = metadata['title']

            md_path = save_markdown_fix(out_folder, fname, full_text, out_metadata)
            md_filename = fname.rsplit(".", 1)[0] + ".md"
            if record_id is not None and parent_record_id is not None and ocr_type is not None:
                pdf_data_opt = PDFDataOperator(config_file)
                # 修改OCT_TYPE标志最后一个字符为1，表示完成修正
                modified_ocr_type = ocr_type[:-1] + '1'
                # pdf_data_opt.update_sub_finish_fix(record_id, modified_string, md_path, md_filename)
                record_num = pdf_data_opt.get_sub_record_number(parent_record_id)
                sub_record_id = parent_record_id + '_' + str(int(record_num) + 1).zfill(3)
                pdf_data_opt.insert_sub_finish_ocr(parent_record_id, sub_record_id, modified_ocr_type, title, md_path, md_filename)

                # 查找子表MD文件都完成修正后，更新主表finish_ocr标志为9 识别结束
                ready_fix_num = pdf_data_opt.get_sub_finish_ocr_number(parent_record_id, OCR_TYPES)
                finish_fix_num = pdf_data_opt.get_sub_finish_ocr_number(parent_record_id, FIX_OCR_TYPES)
                if finish_fix_num == ready_fix_num:
                    pdf_data_opt.update_pri_finish_orc(parent_record_id, 9)
                    log_info = f" * * * * * Fixed Success! {parent_record_id} {fname}"
                    print(log_info)
                    logger.error(log_info)

                md_fullname = os.path.join(md_path, md_filename)
                # 计算百分比
                percentage = ((idx + 1) / files_number) * 100
                log_info = f" * * * * * Fixing {idx+1}/{files_number}({percentage:.2f}%) {fname}, id:{sub_record_id}, storing in {md_fullname}"
                print(log_info)
                logger.info(log_info)
            else:
                log_info = f" * * * * * Fixing Error {idx + 1} {fname}, data Error!\n{metadata}"
                print(log_info)
                logger.error(log_info)
        else:
            log_info = f"Empty file: {filepath}.  Could not fix."
            print(log_info)
            logger.info(log_info)
    except Exception as e:
        log_info = f"Error fixing {filepath}: {e}"
        print(log_info)
        # print(traceback.format_exc())
        logger.error(log_info)


def main():
    parser = argparse.ArgumentParser(description="check multiple markdown files.")
    parser.add_argument("--max", type=int, default=0, help="Maximum number of files to fix")
    parser.add_argument("--metadata_file", type=str, default=None, help="Metadata json file to use for filtering")
    # 增加读取配置文件中数据库信息，通过数据库记录形式取代通过meta_file方式操作多文件 2024-08-13
    parser.add_argument("--data_type", default='db', help="data source type (db or path)")
    parser.add_argument("--config_file", default='config.ini', help="config file.")
    # 增加操作类型，convert：识别转化PDF check：检查转化效果
    parser.add_argument("--run_type", default='check', help="run type type (repair or check)")

    args = parser.parse_args()

    # 增加读取配置文件中数据库信息，通过数据库记录形式取代通过meta_file方式操作多文件 2024-08-13
    # begin
    # 从配置文件中读取数据库配置
    start_time = datetime.now()
    data_type = args.data_type
    config_file = args.config_file

    pdf_data_opt = PDFDataOperator(config_file)
    records = pdf_data_opt.query_sub_all_record(args.max)

    if args.run_type == 'convert':

        metadata = {}
        files = []
        out_folder = None
        if data_type == 'db':
            if len(records) <= 0:
                log_info = f"Error No data needs to be processed!"
                print(log_info)
                logger.info(log_info)
                return

            # 循环输出查询结果
            for row in records:
                record_id = row['ID']
                parent_record_id = row['PDF_INFO_ID']
                ocr_type = row['OCR_TYPE']
                md_title = row['MD_TITLE']
                md_path = row['MD_FILE_DIR']
                md_file_name = row['MD_FILE_NAME']
                md_file = os.path.join(md_path, md_file_name)
                file_name = os.path.basename(md_file)
                row_out_folder = os.path.dirname(md_file)
                os.makedirs(row_out_folder, exist_ok=True)
                if file_name.endswith('.md'):
                    if os.path.isfile(md_file):
                        metadata[md_file] = {"out_path": row_out_folder, "record_id": record_id,
                                             "parent_record_id": parent_record_id, 'ocr_type': ocr_type,
                                             "title": md_title}
                        files.append(md_file)

        elif data_type == 'path':
            if args.metadata_file:
                metadata_file = os.path.abspath(args.metadata_file)
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)

            in_folder = os.path.abspath(args.in_folder)
            out_folder = os.path.abspath(args.out_folder)
            files = [os.path.join(in_folder, f) for f in os.listdir(in_folder)]
            files = [f for f in files if os.path.isfile(f)]
            os.makedirs(out_folder, exist_ok=True)
        else:
            log_info = f"Error Unsupported data source!"
            print(log_info)
            logger.info(log_info)
        # end

        if len(files) == 0:
            log_info = f"Error No files needs to be processed!"
            print(log_info)
            logger.info(log_info)
            return

        # Handle chunks if we're processing in parallel
        # Ensure we get all files into a chunk
        chunk_size = args.chunk_size
        files_to_convert = files

        files_number = len(files_to_convert)
        log_info = f" * * * * * Fixing {files_number} files. Started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}"
        print(log_info)
        logger.info(log_info)

        # 执行修复
        for idx, file in enumerate(files_to_convert):
            process_single_file(files_number, idx, file, out_folder, metadata.get(file), config_file, chunk_size)

        end_time = datetime.now()
        # 计算实际执行的时间
        execution_time = end_time - start_time
        execution_seconds = execution_time.total_seconds()

        # 将执行时间转换为时分秒格式
        hours, remainder = divmod(execution_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        average_time = round(execution_seconds / len(files_to_convert))

        log_info = f" * * * * * Fixed {files_number} files. Ended at {end_time.strftime('%Y-%m-%d %H:%M:%S')}. Total execution time {int(hours)}hour {int(minutes)}min {int(seconds)}sec, average time {average_time}src/record"
        print(log_info)
        logger.info(log_info)
    else:
        cnt = 0
        record_ids = []
        # 循环输出查询结果
        for row in records:
            record_id = row['ID']
            md_file_path = row['MD_FILE_DIR']
            md_file_name = row['MD_FILE_NAME']
            md_file = os.path.join(md_file_path, md_file_name)
            if md_file.endswith('.md'):
                if not os.path.isfile(md_file):
                    cnt += 1
                    record_ids.append(record_id)
                    log_info = f" * * * * * {args.run_type}ed md file not exist! {record_id} {md_file}"
                    print(log_info)
                    logger.info(log_info)
            else:
                cnt += 1
                record_ids.append(record_id)
                log_info = f" * * * * * {args.run_type}ed file is not md! {record_id} {md_file}"
                print(log_info)
                logger.info(log_info)

        log_info = f" * * * * * {args.run_type}ed {len(records)} pdfs. {cnt} files not exist! {record_ids}"
        print(log_info)
        logger.info(log_info)


if __name__ == "__main__":
    main()