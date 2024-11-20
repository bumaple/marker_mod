import argparse
import os

import openai
from flask import Flask, request, json
from loguru import logger

from convert_word import convert_handler
from marker.config_read import Config
from marker.logger import set_logru
from marker.http.http_utils import HttpUtils

app = Flask(__name__)

set_logru()

config_data = None

def replace_string(replace_str: str, str_list: list):
    for str_val in str_list:
        replace_str = replace_str.replace(str_val, '')
    return replace_str

@app.route('/fix_latex', methods=['POST'])
def fix_latex_markdown():
    req_data = request.get_json()

    if not req_data or 'text' not in req_data:
        response = {
            "is_success": False,
            "resp_content": "'text' parameter is required"
        }
        return app.response_class(
            response=json.dumps(response, ensure_ascii=False),
            mimetype='application/json',
        )

    text_value = req_data['text']
    if 'type' in req_data:
        latex_type = req_data['type']
        if latex_type not in ['alone', 'together']:
            latex_type = 'alone'
    else:
        latex_type = 'alone'

    prompt_head = "你是Markdown格式的专家，任务是将文本并处理为符合要求的Markdown格式并按照要求输出为Markdown文本。请遵循以下要求：\n" + \
                  "1.不添加原文中不存在的任何新信息；\n" + \
                  "2.不添加```、```markdown、``````markdown等标识代码块的标记；\n" + \
                  f"3.文本中的数学、化学公式等LaTex公式修改为完整正确的Markdown公式，{'用$内容$进行标记' if latex_type == 'together' else '用$$内容$$进行标记'}；\n" + \
                  "4.只回复符合格式要求的文本，不添加任何引言、解释或元数据。\n"

    http_utils = HttpUtils(config_data)
    resp_content = http_utils.request_openai_api(prompt_head=prompt_head, text_content=text_value)

    replace_strs = ['``` ```markdown', '``````markdown', '```markdown', '```', '\n']
    resp_content = replace_string(resp_content, replace_strs)

    response = {
        "is_success": True,
        "resp_content": resp_content
    }
    return app.response_class(
        response=json.dumps(response, ensure_ascii=False),
        mimetype='application/json'
    )


@app.route('/fix_table', methods=['POST'])
def fix_table_markdown():
    req_data = request.get_json()

    if not req_data or 'text' not in req_data:
        response = {
            "is_success": False,
            "resp_content": "'text' parameter is required"
        }
        return app.response_class(
            response=json.dumps(response, ensure_ascii=False),
            mimetype='application/json',
        )

    text_content = req_data['text']

    if 'type' in req_data:
        latex_type = req_data['type']
        if latex_type not in ['alone', 'together']:
            latex_type = 'alone'
    else:
        latex_type = 'alone'

    require_content = None
    if 'require' in req_data:
        require_content = req_data['require']

    prompt_head = "你是Markdown格式的专家，任务是将文本并处理为符合要求的Markdown格式并按照要求输出为Markdown文本。请遵循以下要求：\n" + \
                  "1.不添加原文中不存在的任何新信息；\n" + \
                  "2.不添加```、```markdown、``````markdown等标识代码块的标记；\n" + \
                  f"3.文本中的数学、化学公式等LaTex公式修改为完整正确的Markdown公式，{'用$内容$进行标记' if latex_type == 'together' else '用$$内容$$进行标记'}；\n" + \
                  "4.文本中的Markdown格式表格替换为html形式，如需要处理的文本不包含<table></table>则不要添加<table></table>标记，如需要处理的文本不包含<tr></tr>则不要添加<tr></tr>标记；\n" + \
                  "5.只回复符合格式要求的文本，不添加任何引言、解释或元数据。\n"

    if require_content is not None:
        req_content = f"在已有要求的基础上结合下面的要求对文本进行处理：{require_content}\n以下是需要处理的文本：\n{text_content}"
    else:
        req_content = text_content

    http_utils = HttpUtils(config_data)
    resp_content = http_utils.request_openai_api(prompt_head=prompt_head, text_content=req_content)

    replace_strs = ['``` ```markdown', '``````markdown', '```markdown', '```', '\n']
    resp_content = replace_string(resp_content, replace_strs)

    response = {
        "is_success": True,
        "resp_content": resp_content
    }
    return app.response_class(
            response=json.dumps(response, ensure_ascii=False),
            mimetype='application/json'
        )


@app.route('/convert_docx', methods=['POST'])
def convert_docx():
    req_data = request.get_json()

    is_success = False

    if not req_data or 'in_file' not in req_data:
        response = {
            "is_success": is_success,
            "content": "'in_file'参数不存在！"
        }
        return app.response_class(
            response=json.dumps(response, ensure_ascii=False),
            mimetype='application/json',
        )

    in_file_input = req_data['in_file']

    data_source = 'path'
    metadata_list = {}

    if in_file_input.endswith('.docx'):
        if os.path.isfile(in_file_input):
            files = [in_file_input]
            for file in files:
                file_name = os.path.basename(file)
                metadata_list[file_name] = {"in_file": in_file_input}

            result_code, result_msg, out_file_data = convert_handler(config_data,None, data_source, 0, metadata_list,
                                      files)
            if result_code == 1:
                is_success = True
                resp_content = out_file_data
            else:
                is_success = False
                resp_content = result_msg
        else:
            resp_content = f"文件不存在！{in_file_input}"
    else:
        resp_content = f"仅支持docx文件类型！{in_file_input}"

    response = {
        "is_success": is_success,
        "content": resp_content
    }
    return app.response_class(
        response=json.dumps(response, ensure_ascii=False),
        mimetype='application/json'
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Custom API Server to use LLM.")
    parser.add_argument("--config_file", default='config.ini', help="config file.")

    args = parser.parse_args()
    config_data = Config(args.config_file)

    port = config_data.get_server_param('port')
    if port is None:
        port = 9992

    app.run(debug=config_data.is_dev_mode(), host='0.0.0.0', port=port)
