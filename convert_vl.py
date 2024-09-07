import os
import argparse
import traceback
import json

from datetime import datetime
from typing import List, Dict, Tuple

import torch

from marker.database.pdf_data_operator import PDFDataOperator
from marker.logger import setup_logger
from marker.output import save_markdown
from marker.config_read import Config

from pdf2image import convert_from_path
from PIL import Image, ImageChops
import cv2
import base64
from io import BytesIO
import numpy as np

import openai

from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor, GenerationConfig
from qwen_vl_utils import process_vision_info

logger = setup_logger()


def replace_string(replace_str: str, str_list: list):
    new_str = ''
    for str_val in str_list:
        new_str = replace_str.replace(str_val, '')
    return new_str

def get_image_filename(image_idx):
    return f"image_{str(image_idx + 1).zfill(2)}.png"


def images_to_dict(scanned_images):
    images = {}
    for image_idx, image in enumerate(scanned_images):
        image_filename = get_image_filename(image_idx)
        images[image_filename] = image
    return images


def preprocess_image(image):
    """
    预处理图像，转换为二值化的灰度图，并进行膨胀操作。
    :param image: (PIL.Image) 输入的 PIL 图像对象
    :returns: PIL.Image 预处理后的图像，转换为二值化灰度图像并经过膨胀操作
    """
    # 将 PIL 图像转换为 NumPy 数组
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    # 使用 Otsu 二值化方法
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # 定义核（1x1 大小的核），用于膨胀操作
    kernel = np.ones((1, 1), np.uint8)
    # 对图像进行膨胀操作，增强图像的白色区域
    gray = cv2.dilate(gray, kernel, iterations=1)
    # 将处理后的 NumPy 数组转换回 PIL 图像
    return Image.fromarray(gray)

def is_image_empty(image, tolerance: int = 10):
    """
    判断图像是否为空(单一颜色图像)。
    :param image： (PIL.Image) 需要判断的图像
    :param tolerance： (int) 容差，用于判断图像是否为单一颜色
    :returns: bool: 如果图像为空(全黑、全白或单一颜色)，返回 True；否则返回 False
    """
    # 将图像转换为 RGB 模式（如果不是）
    image = image.convert('RGB')
    # 获取图像的像素数据
    pixels = list(image.getdata())
    # 选取第一个像素值作为基准
    base_color = pixels[0]
    # 判断每个像素是否与基准颜色相近
    for pixel in pixels:
        if not all(abs(pixel[i] - base_color[i]) <= tolerance for i in range(3)):
            # 如果有像素不在容差范围内，说明不是单一颜色图像
            return False
    return True


def resize_image(image, target_size):
    """
    根据传入的目标宽度，对图像进行等比例缩放。
    如果图像的宽度大于 target_width，则等比例缩小图像。
    如果图像的宽度小于或等于 target_width，则不进行缩放。
    :param image: (PIL.Image) 要处理的图像对象
    :param target_size: (int) 目标宽度的像素值
    :returns: PIL.Image 处理后的图像对象
    """
    # 获取原始图像的宽度和高度
    original_width, original_height = image.size
    # 判断是否需要缩放，只有当宽度或高度大于 target_size 时进行缩放
    if original_width > target_size or original_height > target_size:
        # 计算缩放比例，以保持图像的宽高比
        scaling_factor = min(target_size / original_width, target_size / original_height)
        # 根据缩放比例计算新的宽度和高度
        new_width = int(original_width * scaling_factor)
        new_height = int(original_height * scaling_factor)
        # 使用高质量的 LANCZOS 插值法进行缩放
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        return resized_image
    else:
        # 如果图像宽度小于或等于目标宽度，则不进行缩放
        return image


def image_to_base64(image):
    """
    将传入的 PIL.Image 对象转化为 Base64 编码的字符串。
    自动根据图像的原始格式进行处理。
    :param image: (PIL.Image) 要处理的图像对象
    :returns: str Base64编码的图像字符串
    """
    image_format = get_image_format(image)
    # 创建一个字节流来保存图像
    buffered = BytesIO()
    # 将图像保存到字节流中，使用自动获取的格式
    image.save(buffered, format=image_format)
    # 获取字节流的二进制数据
    img_bytes = buffered.getvalue()
    # 将二进制数据编码为 Base64 字符串
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    return image_format, img_base64


def get_image_format(image):
    # 获取图像的格式，如果没有则使用 'PNG' 作为默认格式
    return image.format if image.format else 'PNG'


def convert_pdf_to_images(input_pdf_file_path: str, max_pixels: int, max_pages: int = 0, skip_first_n_pages: int = 0) -> List[Image.Image]:
    log_info = f"   Processing PDF file {input_pdf_file_path}"
    print(log_info)
    logger.info(log_info)
    if max_pages == 0:
        last_page = None
    else:
        last_page = skip_first_n_pages + max_pages
    first_page = skip_first_n_pages + 1
    scanned_images = convert_from_path(input_pdf_file_path, first_page=first_page, last_page=last_page)
    images = []
    for idx, scanned_image in enumerate(scanned_images):
        if is_image_empty(scanned_image):
            log_info = f"   Converting PDF file No.{idx + 1} in {len(scanned_images)} page image is empty!"
            print(log_info)
            logger.info(log_info)
            continue
        image = resize_image(scanned_image, max_pixels)
        images.append(image)
    log_info = f"   Converted {len(images)} pages from PDF file to images."
    print(log_info)
    logger.info(log_info)
    return images

def request_openai_api(model_url, api_key, model_name, llm_temperature, llm_top_p, llm_max_tokens, scanned_idx, scanned_image):
    openai.api_key = api_key
    openai.base_url = model_url

    image_format, image_base64 = image_to_base64(scanned_image)

    prompt_head = "你是Markdown文件的处理专家，识别图片中的内容转化为Markdown文本。请遵循以下要求：\n" + \
                  "1.不添加原文中不存在的任何信息；\n" + \
                  "2.不要添加不必要的标点符号，删除前后无关联且无意义的符号，不要用```、```markdown、``````markdown标记；\n" + \
                  "3.完整标准名称、完整标准编号、前言、附录A、附录B、附录C等标注为一级标题，1、2、3、4、5之类的编号标注为一级标题，1.1、1.2、1.3、1.4、2.1、2.2、2.3之类的标注为二级标题，A.1、A.2、A.3、B.1、B.2、B.3之类的标注为二级标题，1.1.1、1.1.2、1.1.3、2.1.1、2.1.2之类的标注为三级标题，以此类推用四级、五级标题进行标注；" + \
                  "4.保持原始结构的完整性，标题及所包含的编号需要保持单独一行；\n" + \
                  "5.删除所有页眉和页脚，删除右上角多余的标准编号，删除页面下方的页码，删除句子或段落中的不必要换行，删除文本中不必要的空格；\n" + \
                  "6.保留所有中文、数字、字母组成的编号和序号；\n" + \
                  f"7.省略无法识别成文字的部分，用图片格式进行标记，图片路径按照顺序用image_{str(scanned_idx).zfill(2)}_01.{image_format}、image_{str(scanned_idx).zfill(2)}_02.{image_format}的形式进行标记；\n" + \
                  "8.识别出来的公式使用完整正确的LaTex公式语法进行标记；\n" + \
                  "9.只回复符合格式要求的文本，不添加任何引言、解释或元数据。\n"

    # create a chat completion
    completion = openai.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": [
                {"type": "text", "text": f"{prompt_head}"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{image_format.lower()};base64,{image_base64}",
                    },
                },
            ]}],
        # temperature=llm_temperature,
        # top_p=llm_top_p,
        max_tokens=llm_max_tokens
    )

    return completion.choices[0].message.content

def init_local_llm(model_path):
    # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    # model = Qwen2VLForConditionalGeneration.from_pretrained(
    #     "Qwen/Qwen2-VL-7B-Instruct",
    #     torch_dtype=torch.bfloat16,
    #     torch_dtype="auto"
    #     attn_implementation="flash_attention_2",
    #     device_map="auto",
    # )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map="auto"
    )

    # The default range for the number of visual tokens per image in the model is 4-16384.
    # You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
    min_pixels = 256 * 28 * 28
    max_pixels = 1280 * 28 * 28
    processor = AutoProcessor.from_pretrained(model_path, min_pixels=min_pixels, max_pixels=max_pixels)

    return model, processor


def request_local_llm(model, processor, llm_temperature, llm_top_p, llm_max_tokens, scanned_image_idx, scanned_image):

    image_format, image_base64 = image_to_base64(scanned_image)
    prompt_head = "你是Markdown文件的处理专家，识别图片中的文本转化为Markdown文本格式，确保内容与前文连贯。请遵循以下要求：\n" + \
                  "1.不添加原文中不存在的任何信息；\n" + \
                  "2.不要添加额外的句号或其他不必要的标点符号，删除前后无关联且无意义的符号，不要用```、```markdown、``````markdown标记；\n" + \
                  "3.保持原始结构及所有标题和副标题的完整性，标题使用Markdown标题格式；\n" + \
                  "4.删除所有页眉页脚，删除句子或段落中的不必要换行，保持段落的断行，删除句子中不必要的空格；\n" + \
                  "5.保留所有中文、数字、字母组成的编号和序号，所有编号需要使用Markdown标题格式，编号及后续的标题结束后进行换行；\n" + \
                  f"6.省略无法识别成文字的部分，用图片格式进行标记，图片路径按照顺序用image_{str(scanned_image_idx).zfill(2)}_01.{image_format}、image_{str(scanned_image_idx).zfill(2)}_02.{image_format}的形式进行标记；\n" + \
                  "7.识别出来的公式使用完整正确的LaTex公式语法进行标记；\n" + \
                  "8.只回复符合格式要求的文本，不添加任何引言、解释或元数据。\n"
    message = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"data:image;base64,{image_base64}",
                },
                {"type": "text", "text": f"{prompt_head}"},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        message, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(message)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    # generation_config = GenerationConfig(max_new_tokens=llm_max_tokens, temperature=llm_temperature, top_k=llm_top_p)
    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return "".join(output_text)


def convert_single_file(filepath: str, config_file: str, is_image_save: bool) -> Tuple[str, Dict, Dict[str, Image.Image]]:
    config = Config(config_file)

    model_name = config.get_vlm_param('model')
    if model_name is None:
        log_info = f"model name is not exist!"
        print(log_info)
        logger.error(log_info)
        return "", {}, {}

    model_type = config.get_vlm_param('model_type')
    if model_type is None:
        model_type = 'net'

    model_url = config.get_vlm_param('url')
    if model_url is None and model_type == "net":
        log_info = f"model url is not exist!"
        print(log_info)
        logger.error(log_info)
        return "", {}, {}

    api_key = config.get_vlm_param('key')

    llm_temperature = config.get_vlm_param('temperature')
    if llm_temperature is None:
        llm_temperature = 0.9
    else:
        llm_temperature = float(llm_temperature)

    llm_top_p = config.get_vlm_param('top_p')
    if llm_top_p is None:
        llm_top_p = 0.8
    else:
        llm_top_p = float(llm_top_p)

    llm_max_tokens= config.get_vlm_param('max_tokens')
    if llm_max_tokens is None:
        llm_max_tokens = 4096
    else:
        llm_max_tokens = int(llm_max_tokens)

    llm_max_pixels= config.get_vlm_param('max_pixels')
    if llm_max_pixels is None:
        llm_max_pixels = 1120
    else:
        llm_max_pixels = int(llm_max_pixels)

    scanned_images = convert_pdf_to_images(filepath, llm_max_pixels)

    out_meta = {
        "model": model_name,
        # "temperature": llm_temperature,
        # "top_p": llm_top_p,
        "model_type": model_type,
        # "max_tokens": llm_max_tokens,
        "max_pixels": llm_max_pixels,
        "image_size": len(scanned_images)
    }

    # return "", out_meta 测试用

    model = None
    processor = None
    if model_type == 'local':
        model, processor = init_local_llm(model_name)

    try:
        resp_contents = []
        for idx, scanned_image in enumerate(scanned_images):
            start_time = datetime.now()

            resp_content = ''
            if model_type == 'net':
                resp_content = request_openai_api(model_url, api_key, model_name, llm_temperature, llm_top_p, llm_max_tokens, idx, scanned_image)
            elif model_type == 'local':
                resp_content = request_local_llm(model, processor, llm_temperature, llm_top_p, llm_max_tokens, idx, scanned_image)

            # 处理多余的Markdown标记
            replace_strs = ['``` ```markdown', '``````markdown', '```markdown', '```']
            resp_content = replace_string(resp_content, replace_strs)
            resp_contents.append(resp_content)

            end_time = datetime.now()
            # 计算实际执行的时间
            execution_time = end_time - start_time
            execution_seconds = execution_time.total_seconds()

            log_info = f"   {start_time.strftime('%Y-%m-%d %H:%M:%S')} LLM request: {idx + 1}/{len(scanned_images)}, execution time {int(execution_seconds)}sec {filepath}"
            print(log_info)
            logger.info(log_info)

        out_meta["convert_stats"] = "success"
        if is_image_save:
            image_dict = images_to_dict(scanned_images)
        else:
            image_dict = {}
        return "".join(resp_contents), out_meta, image_dict
    except Exception as e:
        out_meta["fix_stats"] = "fail"
        log_info = f"Error Converting {filepath}: {e}"
        print(log_info)
        logger.error(log_info)
        print(traceback.format_exc())
        return "", out_meta, {}


def process_single_pdf(files_number, idx, filepath, out_folder, metadata, config_file, is_image_save, ocr_type, fix_ocr_type):
    fname = os.path.basename(filepath)
    if not os.path.exists(filepath):
        log_info = f"File not exist: {filepath}."
        print(log_info)
        logger.error(log_info)
        return

    try:
        full_text, out_metadata, scanned_images = convert_single_file(filepath, config_file, is_image_save)
        if len(full_text.strip()) > 0:
            record_id = None
            title = ''
            data_type = ''
            if out_folder is None and 'out_path'in metadata:
                out_folder = metadata['out_path']
            if 'record_id' in metadata:
                record_id = metadata['record_id']
            if 'title' in metadata:
                title = metadata['title']
            if 'data_type' in metadata:
                data_type = metadata['data_type']

            md_path = save_markdown(out_folder, fname, full_text, scanned_images, out_metadata, ocr_type)
            md_filename = fname.rsplit(".", 1)[0] + ".md"
            if data_type == 'db':
                if record_id is not None:
                    pdf_data_opt = PDFDataOperator(config_file)
                    # 修改OCT_TYPE标志最后一个字符为1，表示完成修正
                    modified_ocr_type = '31'
                    # modified_ocr_type = ocr_type[:-1] + '1'
                    # pdf_data_opt.update_sub_finish_fix(record_id, modified_string, md_path, md_filename)
                    record_num = pdf_data_opt.get_sub_record_number(record_id)
                    sub_record_id = record_id + '_' + str(int(record_num) + 1).zfill(3)
                    pdf_data_opt.insert_sub_finish_ocr(record_id, sub_record_id, modified_ocr_type, title, md_path, md_filename)

                    # 查找子表MD文件都完成修正后，更新主表finish_ocr标志为9 识别结束
                    # ready_fix_num = pdf_data_opt.get_sub_finish_ocr_number(record_id, OCR_TYPES)
                    # finish_fix_num = pdf_data_opt.get_sub_finish_ocr_number(record_id, FIX_OCR_TYPES)
                    # if finish_fix_num == ready_fix_num:
                    #     pdf_data_opt.update_pri_finish_orc(record_id, 9)
                    #     log_info = f" * * * * * Converted Success! {record_id} {fname}"
                    #     print(log_info)
                    #     logger.error(log_info)

                    md_fullname = os.path.join(md_path, md_filename)
                    # 计算百分比
                    percentage = ((idx + 1) / files_number) * 100
                    log_info = f" * * * * * Converting {idx+1}/{files_number}({percentage:.2f}%) {fname}, id:{sub_record_id}, storing in {md_fullname}"
                    print(log_info)
                    logger.info(log_info)
                else:
                    log_info = f" * * * * * Converting Error {idx + 1} {fname}, data Error!\n{metadata}"
                    print(log_info)
                    logger.error(log_info)
            else:
                md_fullname = os.path.join(md_path, md_filename)
                percentage = ((idx + 1) / files_number) * 100
                log_info = f" * * * * * Converting {idx + 1}/{files_number}({percentage:.2f}%) {fname}, storing in {md_fullname}"
                print(log_info)
                logger.info(log_info)
        else:
            log_info = f"Empty file: {filepath}.  Could not fix."
            print(log_info)
            logger.info(log_info)
    except Exception as e:
        log_info = f"Error Converting {filepath}: {e}"
        print(log_info)
        # print(traceback.format_exc())
        logger.error(log_info)


def main():
    parser = argparse.ArgumentParser(description="Convert multiple PDF files by VL Model.")
    parser.add_argument("--in_folder", help="Input folder with files.")
    parser.add_argument("--out_folder", help="Output folder with files.")
    parser.add_argument("--max", type=int, default=0, help="Maximum number of files to fix")
    parser.add_argument("--metadata_file", type=str, default=None, help="Metadata json file to use for filtering")
    # 增加读取配置文件中数据库信息，通过数据库记录形式取代通过meta_file方式操作多文件 2024-08-13
    parser.add_argument("--data_type", default='db', help="data source type (db or path)")
    parser.add_argument("--config_file", default='config.ini', help="config file.")
    # 增加操作类型，convert：识别转化PDF check：检查转化效果
    parser.add_argument("--run_type", default='convert', help="run type type (convert or check)")
    parser.add_argument("--save_image", type=bool, default=True, help="save images (default True)")
    parser.add_argument("--ocr_type", type=str, default='30', help="OCR type (10:marker mod 20:MinerU mod 30:VL Model)")

    args = parser.parse_args()

    # 增加读取配置文件中数据库信息，通过数据库记录形式取代通过meta_file方式操作多文件 2024-08-13
    # begin
    # 从配置文件中读取数据库配置
    start_time = datetime.now()
    data_type = args.data_type
    config_file = args.config_file
    save_image = args.save_image
    ocr_type = args.ocr_type
    fix_ocr_type = ocr_type[:-1] + '1'

    if args.run_type == 'convert':
        metadata = {}
        files = []
        out_folder = None
        if data_type == 'db':
            pdf_data_opt = PDFDataOperator(config_file)
            records = pdf_data_opt.query_need_ocr(ocr_type, args.max)
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
                                          "record_id": record_id, "title": pdf_title, "data_type": data_type}
                    if os.path.isfile(pdf_file):
                        files.append(pdf_file)

        elif data_type == 'path':
            in_folder = os.path.abspath(args.in_folder)
            if os.path.isdir(in_folder):
                files = [os.path.join(in_folder, f) for f in os.listdir(in_folder)]
                files = [f for f in files if os.path.isfile(f)]
            elif os.path.isfile(in_folder):
                files.append(in_folder)
            if args.out_folder is not None:
                out_folder = os.path.abspath(args.out_folder)
                os.makedirs(out_folder, exist_ok=True)

            if args.metadata_file:
                metadata_file = os.path.abspath(args.metadata_file)
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)
            else:
                for pdf_file in files:
                    file_name = os.path.basename(pdf_file)
                    title_name = file_name.rsplit(".", 1)[0]
                    if out_folder is None:
                        out_folder = os.path.dirname(pdf_file)
                        os.makedirs(out_folder, exist_ok=True)
                    if file_name.endswith('.pdf'):
                        metadata[pdf_file] = {"languages": ["Chinese", "English"], "out_path": out_folder,
                                              "title": title_name, "data_type": data_type}
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
        files_to_convert = files

        # Limit files converted if needed
        if args.max:
            files_to_convert = files_to_convert[:args.max]

        files_number = len(files_to_convert)
        log_info = f" * * * * * {args.run_type.capitalize()}ing {files_number} pdfs. Started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}"
        print(log_info)
        logger.info(log_info)

        for idx, file in enumerate(files_to_convert):
            process_single_pdf(files_number, idx, file, out_folder, metadata.get(file), config_file, save_image, ocr_type, fix_ocr_type)

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
        pdf_data_opt = PDFDataOperator(config_file)
        records = pdf_data_opt.query_sub_finish_ocr(ocr_type, args.max)

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