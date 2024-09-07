import os
import json

# 增加 2024-08-13 begin
import shutil

from datetime import datetime
# 增加 2024-08-13 end


def get_subfolder_path(out_folder, fname, ocr_type, is_timestamp_path: bool = True):
    subfolder_name = fname.rsplit('.', 1)[0]
    subfolder_path = os.path.join(out_folder, subfolder_name)

    # 增加 2024-08-13 begin
    if is_timestamp_path:
        timestamp_str = ocr_type + '_' + datetime.now().strftime('%Y%m%d_%H%M%S')
        subfolder_path_new = os.path.join(subfolder_path, timestamp_str)
    else:
        subfolder_path_new = subfolder_path
    # 增加 2024-08-13 end

    return subfolder_path_new


def get_markdown_filepath(out_folder, fname, ocr_type):
    subfolder_path = get_subfolder_path(out_folder, fname, ocr_type)
    os.makedirs(subfolder_path, exist_ok=True)
    out_filename = fname.rsplit(".", 1)[0] + ".md"
    out_filename = os.path.join(subfolder_path, out_filename)
    return subfolder_path, out_filename


def markdown_exists(out_folder, fname, ocr_type):
    subfolder_path, out_filename = get_markdown_filepath(out_folder, fname, ocr_type)
    return os.path.exists(out_filename)


def save_markdown(out_folder, fname, full_text, images, out_metadata, ocr_type):
    # 修改 2024-08-29 begin
    subfolder_path, markdown_filepath = get_markdown_filepath(out_folder, fname, ocr_type)
    # 修改 2024-08-29 end
    out_meta_filepath = markdown_filepath.rsplit(".", 1)[0] + "_meta.json"

    with open(markdown_filepath, "w+", encoding='utf-8') as f:
        f.write(full_text)
    with open(out_meta_filepath, "w+") as f:
        f.write(json.dumps(out_metadata, indent=4))

    for filename, image in images.items():
        image_filepath = os.path.join(subfolder_path, filename)
        image.save(image_filepath, "PNG")

    return subfolder_path


def save_markdown_fix(out_folder, fname, full_text, out_metadata, ocr_type):
    # 去除两级目录，回到pdf文件所在目录
    new_out_folder = remove_last_dir(out_folder, 2)

    subfolder_path, markdown_filepath = get_markdown_filepath(new_out_folder, fname, ocr_type)
    out_meta_filepath = markdown_filepath.rsplit(".", 1)[0] + "_meta.json"

    with open(markdown_filepath, "w+", encoding='utf-8') as f:
        f.write(full_text)
    with open(out_meta_filepath, "w+") as f:
        f.write(json.dumps(out_metadata, indent=4))

    copy_files(out_folder, subfolder_path)

    return subfolder_path


def remove_last_dir(path, time=1):
    cnt = 0
    head = path
    while cnt < time:
        # 分离目录路径和最后的文件名或目录名
        head, tail = os.path.split(head)
        # 如果tail为空，表示输入是以斜杠结尾的路径，重新处理
        if not tail:
            head, tail = os.path.split(head)
        cnt += 1
    return head


def add_fix_last_dir(path):
    # 分离目录路径和最后的文件名或目录名
    head, tail = os.path.split(path)
    # 如果tail为空，表示输入是以斜杠结尾的路径，重新处理
    if not tail:
        head, tail = os.path.split(head)
    # 修改最后一个目录名
    new_tail = tail + "_fix"
    # 组合成新的路径
    new_path = os.path.join(head, new_tail)
    return new_path


def copy_files(source_dir, target_dir, exclude_extensions=None):
    if exclude_extensions is None:
        exclude_extensions = ['.md', '.json']

        # 遍历源目录中的所有文件和子目录
        for root, dirs, files in os.walk(source_dir):
            # 计算目标子目录路径
            relative_path = os.path.relpath(root, source_dir)
            target_sub_dir = os.path.join(target_dir, relative_path)

            # 确保目标子目录存在
            if not os.path.exists(target_sub_dir):
                os.makedirs(target_sub_dir)

            # 复制文件，排除指定扩展名的文件
            for file in files:
                file_ext = os.path.splitext(file)[1]
                if file_ext not in exclude_extensions:
                    source_file = os.path.join(root, file)
                    target_file = os.path.join(target_sub_dir, file)
                    # 复制文件
                    shutil.copy2(source_file, target_file)
