import os
import argparse
import time
import traceback
import json

from datetime import datetime
from typing import Dict, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import BadRequestError

from marker.database.pdf_data_operator import PDFDataOperator
from marker.logger import setup_logger
from marker.output import save_markdown_fix
from marker.config_read import Config

import transformers
from langchain.text_splitter import MarkdownTextSplitter

import http.client
import openai

os.environ["TOKENIZERS_PARALLELISM"] = "false" # Disabling parallelism to avoid deadlocks

logger = setup_logger()

def chunk_markdown_with_langchain(text, chunk_size: int):
    if len(text) > chunk_size:
        splitter = MarkdownTextSplitter(chunk_size=chunk_size)  # 定义分块大小
        chunks = splitter.split_text(text)  # 对文本进行分块
    else:
        chunks = [text]
    return chunks

def chunk_token_with_langchain(text, chunk_size: int, tokens_model):
    if tokens_model is None or tokens_model == '':
        return chunk_recursive_with_langchain(text, chunk_size, tokens_model)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            tokens_model, trust_remote_code=True
        )
        text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(tokenizer, chunk_size=chunk_size,
            chunk_overlap = 0,)
        # text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        #     model_name=tokens_model,
        #     chunk_size=chunk_size,
        #     chunk_overlap = 0,
        # )

        tokens_size = get_model_tokenizer_size(tokens_model, text)
        if tokens_size > chunk_size:
            chunks = text_splitter.split_text(text)  # 对文本进行分块
        else:
            chunks = [text]
        return chunks

def chunk_recursive_with_langchain(text, chunk_size: int, tokens_model):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=[
            "\n\n",
            "\n",
            " ",
            ".",
            ",",
            "\u200B",  # Zero-width space
            "\uff0c",  # Fullwidth comma
            "\u3001",  # Ideographic comma
            "\uff0e",  # Fullwidth full stop
            "\u3002",  # Ideographic full stop
            "",
        ],
        chunk_size=chunk_size,
        chunk_overlap=0
    )

    tokens_size = get_model_tokenizer_size(tokens_model, text)
    if tokens_size > chunk_size:
        chunks = text_splitter.split_text(text)  # 对文本进行分块
    else:
        chunks = [text]
    return chunks


def get_model_tokenizer_size(tokens_model, text):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokens_model, trust_remote_code=True
    )
    result = tokenizer.encode(text)
    return len(result)


def is_deepseek_balance_valid(api_key):
    conn = http.client.HTTPSConnection("api.deepseek.com")
    payload = ''
    headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }
    conn.request("GET", "/user/balance", payload, headers)
    res = conn.getresponse()
    data = res.read()
    resp_content = data.decode("utf-8")
    try:
        json_data = json.loads(resp_content)
        if 'is_available' in json_data:
            balance_info = json_data.get('balance_infos', [])
            if balance_info and 'total_balance' in balance_info[0]:
                total_balance = balance_info[0]['total_balance']
                return json_data['is_available'], float(total_balance)
            else:
                return json_data['is_available'], 0
        else:
            return False, 0
    except json.JSONDecodeError:
        print("Error: Unable to decode the response as JSON.")
        return False, 0


def request_openai_api(model_url, api_key, model_name, llm_temperature: float, llm_top_p: float, llm_max_tokens: int, chunk_content, attempt_limit: int, attempt_sleep_second: int) -> Tuple[str, int]:
    openai.api_key = api_key
    openai.base_url = model_url

    prompt_head = "你是Markdown格式的专家，任务是将文本并处理为符合要求的Markdown格式并按照要求输出为Markdown格式文本。请遵循以下要求：\n" + \
                  "1.不添加原文中不存在的任何新信息；\n" + \
                  "2.不要添加不必要的标点符号，删除前后无关联且无意义的符号，删除```、```markdown、``````markdown等标识代码块的标记；\n" + \
                  "3.根据常识把不完整的标题进行补充，比如  # 言、#前、# 前修改为##前言，#附、# 录修改为##附录，中人民共国家标准、中人共国家准等修改为中华人民共和国国家标准；\n" + \
                  "4.保留所有中文、数字、字母组成的编号和序号，保留发布日期、实施日期等所有日期及前后对应的文字，日期不要标注为标题，保留发布机构或者发布单位；\n" + \
                  "5.取消原文的所有Markdown标题标注，按照下面的要求标注标题级别：\n" + \
                    "标准名称和标准编号标注为#一级标题，" + \
                    "前言、附录A、附录B、附录C、附录D等文字加字母的标题标注为##二级标题，" + \
                    "类似1、2、3、4、5、6、7、8、9、10、11、12、13等由单组数值组成的编号标注为##二级标题，" + \
                    "类似1.1、1.2、1.3、1.4、2.1、2.2、2.3、2.4、3.1、3.2、3.3、3.4等由两组数值和.组成的编号标注为###三级标题，" + \
                    "类似A.1、A.2、B.1、B.2、C.1、C.2等由英文字母和一组数值和一个.组成的编号的标注为###三级标题，" + \
                    "类似1.1.1、1.1.2、1.2.1、1.2.3、2.1.2、2.1.3、2.2.1、2.2.3等由三组数值和两个.组成的编号标注为####四级标题，" + \
                    "类似A.1.1、A.2.1、B.1.3、B.2.2、C.1.2、C.2.4等由英文字母和两组数值和两个.组成的编号的标注为####四级标题，" + \
                    "类似1.1.1.1、1.2.1.2、2.2.1.1、2.2.1.2等由四组数字和三个.组成的编号标注为#####五级标题，" + \
                    "类似A.1.1.1、A.2.1.2、B.1.3.2、B.2.2.4、C.1.2.3、C.2.4.1等由英文字母和三组数值和三个.组成的编号的标注为#####五级标题，" + \
                    "类似1.1.1.1.1、1.2.1.1.2、2.2.1.1.3、2.2.1.2.5等由五组数字和四个.组成的编号标注为######六级标题，" + \
                    "类似A.1.1.1.1、A.2.1.2.3、B.1.3.2.5、B.2.2.4.4、C.1.2.3.2、C.2.4.1.3等由英文字母和四组数值和四个.组成的编号的标注为######六级标题；" + \
                  "6.保持原始结构的完整性，标题及所包含的编号需要保持单独一行；\n" + \
                  "7.删除页面下方的页码，删除句子或段落中的不必要换行，删除文本中不必要的空格；\n" + \
                  "8.文本中的数学、化学公式等LaTex公式修改为完整正确的Markdown公式，公式和其他文本在一行用$公式内容$进行标记，公式单独为一行用$$公式内容$$进行标记；\n" + \
                  "9.文本中的Markdown格式表格替换为Markdown格式中使用的html表格形式；\n" + \
                  "10.只回复符合格式要求的文本，不添加任何引言、解释或元数据。\n"

    params = {
        "model": model_name,
        "messages": [{"role": "system", "content": f"{prompt_head}"},
                     {"role": "user", "content": f"以下是需要处理的文本：\n{chunk_content}"}],
        "max_tokens": llm_max_tokens
    }
    if llm_temperature not in [None, 0]:
        params["temperature"] = llm_temperature
    if llm_top_p not in [None, 0]:
        params["top_p"] = llm_top_p

    # 发送请求，如果不成功停止5秒后重发，重复3次
    for attempt in range(attempt_limit):
        try:
            completion = openai.chat.completions.create(**params)
            return completion.choices[0].message.content, 1
        except BadRequestError as e:
            if e.code == 'RequestTimeOut':
                log_info = f"Attempt {attempt + 1}/{attempt_limit}: Request timed out. Retrying in {attempt_sleep_second} seconds..."
                print(log_info)
                logger.error(log_info)
                time.sleep(attempt_sleep_second)
            else:
                log_info = f"Error Converting: {e}"
                print(log_info)
                logger.error(log_info)
                return '', 0
        except Exception as e:
            log_info = f"Error Converting: {e}"
            print(log_info)
            logger.error(log_info)
            return '', 0


def fix_single_file(filepath: str, config_file: str) -> Tuple[str, Dict, int]:
    config = Config(config_file)

    with open(filepath, 'r', encoding='utf-8') as file:
        file_content = file.read()

    model_name = config.get_llm_param('model')
    if model_name is None:
        log_info = f"model name is not exist!"
        print(log_info)
        logger.error(log_info)
        return "", {}, 0

    model_url = config.get_llm_param('url')
    if model_url is None:
        log_info = f"model url is not exist!"
        print(log_info)
        logger.error(log_info)
        return "", {}, 0

    tokens_model = config.get_llm_param('tokens_model')

    api_key = config.get_llm_param('key')

    llm_temperature = config.get_llm_param('temperature')
    if llm_temperature is None:
        llm_temperature = 0
    else:
        llm_temperature = float(llm_temperature)

    llm_top_p = config.get_llm_param('top_p')
    if llm_top_p is None:
        llm_top_p = 0
    else:
        llm_top_p = float(llm_top_p)

    llm_max_tokens= config.get_llm_param('max_tokens')
    if llm_max_tokens is None:
        llm_max_tokens = 0
    else:
        llm_max_tokens = int(llm_max_tokens)

    llm_chunk_size = config.get_llm_param('chunk_size')
    if llm_chunk_size is None:
        llm_chunk_size = 2048
    else:
        llm_chunk_size = int(llm_chunk_size)

    max_tokens_multiple = config.get_llm_param('max_tokens_multiple')
    if max_tokens_multiple is None:
        max_tokens_multiple = 0
    else:
        max_tokens_multiple = int(max_tokens_multiple)

    check_balance = config.get_llm_param('check_balance')
    if check_balance is None:
        check_balance = False

    attempt_limit = config.get_llm_param('attempt_limit')
    if attempt_limit is None:
        attempt_limit = 5
    else:
        attempt_limit = int(attempt_limit)

    attempt_sleep_second = config.get_llm_param('attempt_sleep_second')
    if attempt_sleep_second is None:
        attempt_sleep_second = 5
    else:
        attempt_sleep_second = int(attempt_sleep_second)

    out_meta = {
        "model": model_name,
        "chunk_size": llm_chunk_size,
        "txt_length": len(file_content)
    }
    if llm_temperature not in [None, 0]:
        out_meta["temperature"] = llm_temperature
    if llm_top_p not in [None, 0]:
        out_meta["top_p"] = llm_top_p
    if llm_max_tokens not in [None, 0]:
        out_meta["max_tokens"] = llm_max_tokens

    try:
        if check_balance:
            is_valid, balance_value = is_deepseek_balance_valid(api_key)
            if not is_valid:
                log_info = f" * * * * * DeepSeek balance has no balance! * * * * * "
                print(log_info)
                logger.error(log_info)
                return "", out_meta, 9
            elif balance_value < 0.1:
                log_info = f" * * * * * DeepSeek balance has less 0.1! * * * * * "
                print(log_info)
                logger.error(log_info)
                return "", out_meta, 9

        file_content_tokens_size = get_model_tokenizer_size(tokens_model, file_content)
        if llm_max_tokens != 0 and max_tokens_multiple != 0 and file_content_tokens_size > max_tokens_multiple * llm_max_tokens:
            log_info = f" * * * * * File Content exceeds the limit length! {file_content_tokens_size} - MAX:{max_tokens_multiple * llm_max_tokens}* * * * * "
            print(log_info)
            logger.info(log_info)
            return "", out_meta, 0

        resp_contents = []
        is_success = False

        chunks = chunk_token_with_langchain(file_content, llm_chunk_size, tokens_model)

        log_info = f"   chuck {len(chunks)}, {filepath}"
        print(log_info)
        logger.info(log_info)

        for idx, chunk in enumerate(chunks):
            start_time = datetime.now()

            log_info = f"   start {start_time.strftime('%Y-%m-%d %H:%M:%S')} LLM request: {idx + 1}/{len(chunks)}"
            print(log_info)
            logger.info(log_info)
            resp_content, status_code = request_openai_api(model_url, api_key, model_name, llm_temperature, llm_top_p, llm_max_tokens, chunk, attempt_limit, attempt_sleep_second)

            if status_code == 1:
                resp_contents.append(resp_content)

                end_time = datetime.now()
                # 计算实际执行的时间
                execution_time = end_time - start_time
                execution_seconds = execution_time.total_seconds()

                log_info = f"   finish {end_time.strftime('%Y-%m-%d %H:%M:%S')} LLM request: {idx + 1}/{len(chunks)}, execution time {int(execution_seconds)}sec"
                print(log_info)
                logger.info(log_info)
                is_success = True
            else:
                is_success = False
                break

        if is_success:
            out_meta["chunk_num"] = len(chunks)
            out_meta["fix_stats"] = "success"
            return "".join(resp_contents), out_meta, 1
        else:
            out_meta["convert_stats"] = "fail"
            return "", out_meta, 0
    except Exception as e:
        out_meta["fix_stats"] = "fail"
        log_info = f"Error fixing {filepath}: {e}"
        print(log_info)
        logger.error(log_info)
        print(traceback.format_exc())
        return "", out_meta, 0


def process_single_file(files_number, idx, filepath, out_folder, metadata, config_file, ocr_types, fix_ocr_types):

    fname = os.path.basename(filepath)
    # md_file = os.path.join(out_folder, fname)
    if not os.path.exists(filepath):
        log_info = f"File not exist: {filepath}."
        print(log_info)
        logger.error(log_info)
        return 0

    try:

        full_text, out_metadata, resp_status = fix_single_file(filepath, config_file)
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

            # 修改OCT_TYPE标志最后一个字符为1，表示完成修正
            modified_ocr_type = ocr_type[:-1] + '1'

            md_path = save_markdown_fix(out_folder, fname, full_text, out_metadata, modified_ocr_type)
            md_filename = fname.rsplit(".", 1)[0] + ".md"
            if record_id is not None and parent_record_id is not None:
                pdf_data_opt = PDFDataOperator(config_file)

                # pdf_data_opt.update_sub_finish_fix(record_id, modified_string, md_path, md_filename)
                record_num = pdf_data_opt.get_sub_record_number(parent_record_id)
                sub_record_id = parent_record_id + '_' + str(int(record_num) + 1).zfill(3)
                pdf_data_opt.insert_sub_finish_ocr(parent_record_id, sub_record_id, modified_ocr_type, title, md_path, md_filename)

                # 查找子表MD文件都完成修正后，更新主表finish_ocr标志为9 识别结束
                ready_fix_num = pdf_data_opt.get_sub_finish_ocr_number(parent_record_id, ocr_types)
                finish_fix_num = pdf_data_opt.get_sub_finish_ocr_number(parent_record_id, fix_ocr_types)
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
                return 1
            else:
                log_info = f" * * * * * Fixing Error {idx + 1} {fname}, data Error!\n{metadata}"
                print(log_info)
                logger.error(log_info)
                return 0
        else:
            log_info = f"Empty file: {filepath}.  Could not fix."
            print(log_info)
            logger.info(log_info)
            return resp_status
    except Exception as e:
        log_info = f"Error fixing {filepath}: {e}"
        print(log_info)
        print(traceback.format_exc())
        logger.error(log_info)
        return 0


def main():
    parser = argparse.ArgumentParser(description="Fix multiple markdown files by LLM.")
    parser.add_argument("--in_folder", help="Input folder with files.")
    # parser.add_argument("--chunk_size", type=int, default=2000, help="Chunk size to fix")
    parser.add_argument("--max", type=int, default=0, help="Maximum number of files to fix")
    parser.add_argument("--metadata_file", type=str, default=None, help="Metadata json file to use for filtering")
    # 增加读取配置文件中数据库信息，通过数据库记录形式取代通过meta_file方式操作多文件 2024-08-13
    parser.add_argument("--data_type", default='db', help="data source type (db or path)")
    parser.add_argument("--config_file", default='config.ini', help="config file.")
    # 增加操作类型，convert：识别转化PDF check：检查转化效果
    parser.add_argument("--run_type", default='convert', help="run type type (convert or check)")
    parser.add_argument("--ocr_types", type=str, default='10,20', help="OCR type (10:marker-Mod 20:MinerU-Mod, such as 10,20)")

    args = parser.parse_args()

    # 增加读取配置文件中数据库信息，通过数据库记录形式取代通过meta_file方式操作多文件 2024-08-13
    # begin
    # 从配置文件中读取数据库配置
    start_time = datetime.now()
    data_type = args.data_type
    config_file = args.config_file
    ocr_types = args.ocr_types
    ocr_types_list = ocr_types.split(',')

    fix_ocr_types_list = []
    for ocr_type_str in ocr_types_list:
        fix_ocr_type_str = ocr_type_str[:-1] + '1'
        fix_ocr_types_list.append(fix_ocr_type_str)

    if args.run_type == 'convert':
        metadata = {}
        files = []
        out_folder = None
        if data_type == 'db':
            pdf_data_opt = PDFDataOperator(config_file)
            records = pdf_data_opt.query_need_fix_v1(ocr_types_list, args.max)
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
        # chunk_size = args.chunk_size
        files_to_convert = files

        files_number = len(files_to_convert)
        log_info = f" * * * * * Fixing {files_number} files. Started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}"
        print(log_info)
        logger.info(log_info)

        # 执行修复
        for idx, file in enumerate(files_to_convert):
            resp_status = process_single_file(files_number, idx, file, out_folder, metadata.get(file), config_file, ocr_types_list, fix_ocr_types_list)
            if resp_status == 9:
                break

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
        pdf_data_opt = PDFDataOperator(config_file)
        records = pdf_data_opt.query_sub_finish_fix(ocr_types_list, args.max)

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
        print(log_info)
        logger.info(log_info)
        if len(error_files) > 0:
            for error_file in error_files:
                print(error_file)
                logger.info(error_file)

if __name__ == "__main__":
    main()