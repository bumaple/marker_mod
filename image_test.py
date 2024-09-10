import argparse
import fitz
import cv2
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor, as_completed


def process_image(image_data):
    page_num, xref, image_bytes = image_data

    # 将图像转换为OpenCV格式
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 转换为灰度图像进行处理
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 使用自适应阈值处理来识别可能的水印区域
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # 找到可能的水印轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 创建蒙版
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    for contour in contours:
        # 只处理小面积的轮廓（假设水印较小）
        if 100 < cv2.contourArea(contour) < 5000:  # 可以根据实际情况调整这个阈值
            cv2.drawContours(mask, [contour], 0, (255, 255, 255), -1)

    # 使用修复算法
    result = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)

    # 将处理后的图像转回PDF可用的格式
    is_success, buffer = cv2.imencode(".png", result)
    if is_success:
        return page_num, xref, buffer.tobytes()
    return None


def remove_pdf_watermark(input_pdf, output_folder):
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 获取输入PDF文件名(不含路径)
    input_filename = os.path.basename(input_pdf)

    # 构造输出PDF文件路径
    output_pdf = os.path.join(output_folder, f"no_watermark_{input_filename}")

    # 打开PDF文件
    doc = fitz.open(input_pdf)

    # 收集所有需要处理的图像
    images_to_process = []
    for page_num, page in enumerate(doc):
        for img in page.get_images():
            xref = img[0]
            base = doc.extract_image(xref)
            image_bytes = base["image"]
            images_to_process.append((page_num, xref, image_bytes))

    # 使用ProcessPoolExecutor处理图像
    with ProcessPoolExecutor() as executor:
        future_to_image = {executor.submit(process_image, image_data): image_data for image_data in images_to_process}
        for future in as_completed(future_to_image):
            result = future.result()
            if result:
                page_num, xref, processed_image = result
                page = doc[page_num]
                page.delete_image(xref)
                page.insert_image(page.rect, stream=processed_image)

    # 保存处理后的PDF
    doc.save(output_pdf)
    doc.close()

    print(f"处理完成。输出文件: {output_pdf}")


def main():
    parser = argparse.ArgumentParser(description="Convert multiple pdfs to markdown.")
    parser.add_argument("--in_folder", help="Input folder with pdfs.")
    parser.add_argument("--out_folder", help="Output folder")

    args = parser.parse_args()

    in_folder = os.path.abspath(args.in_folder)
    out_folder = os.path.abspath(args.out_folder)

    # watermark_colors = [(200, 200, 200), (218, 227, 239), (240, 228, 214)]  # 灰色
    # tolerance = 10  # 容忍颜色差异的范围
    # min_width = 400  # 水印图像的最小宽度
    # min_height = 400  # 水印图像的最小高度
    remove_pdf_watermark(in_folder, out_folder)

if __name__ == "__main__":
    main()