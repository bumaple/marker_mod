import os
import copy

import argparse
import math
import logging

from marker.logger import configure_logging
# import traceback
import json

from magic_pdf.pipe.UNIPipe import UNIPipe
from magic_pdf.pipe.OCRPipe import OCRPipe
from magic_pdf.pipe.TXTPipe import TXTPipe
from magic_pdf.rw.DiskReaderWriter import DiskReaderWriter
import magic_pdf.model as model_config

model_config.__use_inside_model__ = True

configure_logging()


# 增加 2024-08-13 begin
from datetime import datetime
from marker.database.pdf_data_operator import PDFDataOperator
from marker.logger import setup_logger

logger = setup_logger()

logging.getLogger('detectron2').setLevel(logging.ERROR)
logging.getLogger('magic_pdf').setLevel(logging.ERROR)
# 增加 2024-08-13 end

OCR_TYPE = '20'

def json_md_dump(
        pipe,
        md_writer,
        pdf_name,
        content_list,
        md_content,
):
    # 写入模型结果到 model.json
    orig_model_list = copy.deepcopy(pipe.model_list)
    md_writer.write(
        content=json.dumps(orig_model_list, ensure_ascii=False, indent=4),
        path=f"{pdf_name}_model.json"
    )

    # 写入中间结果到 middle.json
    md_writer.write(
        content=json.dumps(pipe.pdf_mid_data, ensure_ascii=False, indent=4),
        path=f"{pdf_name}_middle.json"
    )

    # text文本结果写入到 conent_list.json
    md_writer.write(
        content=json.dumps(content_list, ensure_ascii=False, indent=4),
        path=f"{pdf_name}_content_list.json"
    )

    # 写入结果到 .md 文件中
    md_writer.write(
        content=md_content,
        path=f"{pdf_name}.md"
    )


def process_single_pdf(files_number, idx, pdf_path, output_dir, metadata, config_file):
    """
    执行从 pdf 转换到 json、md 的过程，输出 md 和 json 文件到 pdf 文件所在的目录

    pdf_path: .pdf 文件的路径，可以是相对路径，也可以是绝对路径
    parse_method: 解析方法， 共 auto、ocr、txt 三种，默认 auto，如果效果不好，可以尝试 ocr
    model_json_path: 已经存在的模型数据文件，如果为空则使用内置模型，pdf 和 model_json 务必对应
    is_json_md_dump: 是否将解析后的数据写入到 .json 和 .md 文件中，默认 True，会将不同阶段的数据写入到不同的 .json 文件中（共3个.json文件），md内容会保存到 .md 文件中
    output_dir: 输出结果的目录地址，会生成一个以 pdf 文件名命名的文件夹并保存所有结果
    """
    model_json_path = None
    parse_method = 'auto'
    is_json_md_dump = True
    try:
        # pdf_name = os.path.basename(pdf_path).split(".")[0]
        # pdf_path_parent = os.path.dirname(pdf_path)
        pdf_name = os.path.basename(pdf_path).rsplit('.', 1)[0]

        if output_dir:
            output_path_old = os.path.join(output_dir, pdf_name)
        else:
            output_path_old = os.path.join(os.path.dirname(pdf_path), pdf_name)

        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = os.path.join(output_path_old, timestamp_str)

        output_image_path = os.path.join(output_path, 'images')

        # 获取图片的父路径，为的是以相对路径保存到 .md 和 conent_list.json 文件中
        image_path_parent = os.path.basename(output_image_path)

        pdf_bytes = open(pdf_path, "rb").read()  # 读取 pdf 文件的二进制数据

        if model_json_path:
            # 读取已经被模型解析后的pdf文件的 json 原始数据，list 类型
            model_json = json.loads(open(model_json_path, "r", encoding="utf-8").read())
        else:
            model_json = []

        # 执行解析步骤
        image_writer, md_writer = DiskReaderWriter(output_image_path), DiskReaderWriter(output_path)

        # 选择解析方式
        if parse_method == "auto":
            jso_useful_key = {"_pdf_type": "", "model_list": model_json}
            pipe = UNIPipe(pdf_bytes, jso_useful_key, image_writer)
        elif parse_method == "txt":
            pipe = TXTPipe(pdf_bytes, model_json, image_writer)
        elif parse_method == "ocr":
            pipe = OCRPipe(pdf_bytes, model_json, image_writer)
        else:
            log_info = f"unknown parse method, only auto, ocr, txt allowed"
            print(log_info)
            logger.error(log_info)
            return

        # 执行分类
        pipe.pipe_classify()

        # 如果没有传入模型数据，则使用内置模型解析
        if not model_json:
            if model_config.__use_inside_model__:
                pipe.pipe_analyze()  # 解析
            else:
                log_info = f"need model list input"
                print(log_info)
                logger.error(log_info)
                return

        # 执行解析
        pipe.pipe_parse()

        # 保存 text 和 md 格式的结果
        content_list = pipe.pipe_mk_uni_format(image_path_parent, drop_mode="none")
        md_content = pipe.pipe_mk_markdown(image_path_parent, drop_mode="none")

        if is_json_md_dump:
            json_md_dump(pipe, md_writer, pdf_name, content_list, md_content)

            # 生成目录增加时间戳 2024-08-13 begin
            record_id = None
            title = ''
            if 'record_id' in metadata:
                record_id = metadata['record_id']
            if 'title' in metadata:
                title = metadata['title']
            md_path = output_path
            md_filename = pdf_name + ".md"
            if record_id is not None:
                pdf_data_opt = PDFDataOperator(config_file)
                pdf_data_opt.update_pri_finish_orc(record_id, 1)

                record_num = pdf_data_opt.get_sub_record_number(record_id)
                # print(f" * * * * * record_num:{record_num}")
                sub_record_id = record_id + '_' + str(int(record_num) + 1).zfill(3)
                pdf_data_opt.insert_sub_finish_ocr(record_id, sub_record_id, OCR_TYPE, title, md_path, md_filename)

            md_fullname = os.path.join(md_path, md_filename)
            # 计算百分比
            percentage = ((idx + 1) / files_number) * 100
            log_info = f" * * * * * Converting {idx + 1}/{files_number}({percentage:.2f}%) {pdf_name}, id:{record_id}, storing in {md_fullname}"
            print(log_info)
            logger.info(log_info)
            # end

    except Exception as e:
        log_info = f"Error converting {pdf_path}: {e}"
        print(log_info)
        logger.error(log_info)

def main():
    parser = argparse.ArgumentParser(description="Convert multiple pdfs to markdown.")
    parser.add_argument("--in_folder", help="Input folder with pdfs.")
    parser.add_argument("--out_folder", help="Output folder")
    parser.add_argument("--chunk_idx", type=int, default=0, help="Chunk index to convert")
    parser.add_argument("--num_chunks", type=int, default=1, help="Number of chunks being processed in parallel")
    parser.add_argument("--max", type=int, default=0, help="Maximum number of pdfs to convert")
    parser.add_argument("--metadata_file", type=str, default=None, help="Metadata json file to use for filtering")

    # 增加读取配置文件中数据库信息，通过数据库记录形式取代通过meta_file方式操作多文件 2024-08-13
    parser.add_argument("--data_type", default='db', help="data source type (db or path)")
    parser.add_argument("--config_file", default='config.ini', help="config file.")
    # 增加操作类型，convert：识别转化PDF check：检查转化效果
    parser.add_argument("--run_type", default='convert', help="run type type (convert or check)")

    args = parser.parse_args()

    # 增加读取配置文件中数据库信息，通过数据库记录形式取代通过meta_file方式操作多文件 2024-08-13
    # begin
    # 从配置文件中读取数据库配置
    start_time = datetime.now()
    data_type = args.data_type
    config_file = args.config_file
    pdf_data_opt = PDFDataOperator(config_file)

    if args.run_type == 'convert':
        metadata = {}
        files = []
        out_folder = None
        if data_type == 'db':
            records = pdf_data_opt.query_need_ocr(OCR_TYPE, args.max)
            if len(records) <= 0:
                log_info = f"Error No data needs to be processed!"
                print(log_info)
                logger.info(log_info)
                return

            # 循环输出查询结果
            for row in records:
                record_id = row['ID']
                pdf_file = row['PDF_FILE']
                pdf_title = row['TITLE']
                file_name = os.path.basename(pdf_file)
                row_out_folder = os.path.dirname(pdf_file)
                os.makedirs(row_out_folder, exist_ok=True)
                if file_name.endswith('.pdf'):
                    metadata[pdf_file] = {"languages": ["Chinese", "English"], "out_path": out_folder,
                                           "record_id": record_id, "title": pdf_title}
                    if os.path.isfile(pdf_file):
                        files.append(pdf_file)

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

        # Handle chunks if we're processing in parallel
        # Ensure we get all files into a chunk
        chunk_size = math.ceil(len(files) / args.num_chunks)
        start_idx = args.chunk_idx * chunk_size
        end_idx = start_idx + chunk_size
        files_to_convert = files[start_idx:end_idx]

        # Limit files converted if needed
        if args.max:
            files_to_convert = files_to_convert[:args.max]

        files_number = len(files_to_convert)
        log_info = f" * * * * * {args.run_type.capitalize()}ing {files_number} pdfs in chunk {args.chunk_idx + 1}/{args.num_chunks}. Started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}"
        print(log_info)
        logger.info(log_info)

        for idx, file in enumerate(files_to_convert):
            process_single_pdf(files_number, idx, file, out_folder, metadata.get(file), config_file)

        end_time = datetime.now()
        # 计算实际执行的时间
        execution_time = end_time - start_time
        execution_seconds = execution_time.total_seconds()

        # 将执行时间转换为时分秒格式
        hours, remainder = divmod(execution_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        average_time = round(execution_seconds / len(files_to_convert))

        log_info = f" * * * * * {args.run_type.capitalize()}ed {files_number} pdfs. Ended at {end_time.strftime('%Y-%m-%d %H:%M:%S')}. Total execution time {int(hours)} hour {int(minutes)} min {int(seconds)} sec, average time {average_time} sec/record"
        print(log_info)
        logger.info(log_info)
    else:
        records = pdf_data_opt.query_sub_finish_ocr(OCR_TYPE, args.max)

        error_files = []
        # 循环输出查询结果
        for row in records:
            record_id = row['ID']
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