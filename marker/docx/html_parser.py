import re

from bs4 import BeautifulSoup, NavigableString, Tag
from typing import List, Dict, Any

from loguru import logger

from marker.logger import set_logru
from marker.http.http_utils import HttpUtils

class HTMLTableParser:
    def __init__(self, html_content: str, cells_dict: dict):
        """
        初始化解析器
        :param html_content: HTML字符串
        :param cells_dict: 数据列表
        """
        self.cells_dict = cells_dict
        self.soup = BeautifulSoup(html_content, 'html.parser')
        set_logru()

    def _process_cell_content(self, cell) -> str:
        """
        处理单元格内容：
        - 文本内容去除空格
        - 保留img标签及其空格
        - 支持文本和img标签混合的情况
        """
        # 将cell内容转换为字符串列表
        result = []
        for content in cell.contents:
            # if isinstance(content, NavigableString):
            #     # 对普通文本去除空格
            #     text = str(content)
            #     text = ''.join(text.split())
            #     if text:  # 只添加非空文本
            #         result.append(text)
            # elif isinstance(content, Tag):
            #     if content.name == 'img':
            #         # 保留img标签的原始HTML
            #         result.append(str(content))
            #     else:
            #         # 递归处理其他标签
            #         inner_content = self._process_cell_content(content)
            #         if inner_content:  # 只添加非空内容
            #             result.append(inner_content)
            result.append(str(content))

        return ''.join(result)

    def _get_cell_content(self, cell) -> str:
        """
        获取单元格内容
        """
        return self._process_cell_content(cell)

    def _has_merged_cells(self, row) -> bool:
        """
        检查行中是否存在合并单元格
        """
        cells = row.find_all('td')
        for cell in cells:
            if cell.has_attr('rowspan') or cell.has_attr('colspan'):
                return True
        return False

    def _get_cell_rowspan(self, cell) -> int:
        """
        检查单元格是否存在行合并，返回合并行数
        """
        if cell.has_attr('rowspan'):
            return int(cell.attrs['rowspan'])
        return 0

    def _get_cell_colspan(self, cell) -> int:
        """
        检查单元格是否存在列合并，返回合并列数
        """
        if cell.has_attr('colspan'):
            return int(cell.attrs['colspan'])
        return 0

    def _process_header_rows(self, table) -> tuple[dict[int, str], int]:
        """
        处理表头行，返回处理后的表头和表头行数
        基于合并单元格的存在来判断表头行
        """
        rows = table.find_all('tr')
        if not rows:
            return {}, 0

        header_columns_dict = {}
        header_row_count = 0 # 表头行数

        # 检查第一行
        current_row_count = 0
        # 最大合并列数
        max_rowspan_count = 0
        # 存储每行合并单元格的索引
        row_span_idx_dict = {}
        while current_row_count < len(rows):
            row = rows[current_row_count]

            # 超过最大合并行数，结束循环
            if max_rowspan_count != 0 and current_row_count >= max_rowspan_count:
                break

            header_row_count += 1

            # 如果当前行有合并单元格，继续添加为表头
            if self._has_merged_cells(row):
                cells = row.find_all('td')
                last_colspan_counts = 0

                for cell_idx, cell in enumerate(cells):
                    rowspan_counts = self._get_cell_rowspan(cell)
                    colspan_counts = self._get_cell_colspan(cell)
                    max_rowspan_count = max(max_rowspan_count, rowspan_counts)
                    if current_row_count == 0:
                        if rowspan_counts > 0:
                            header_columns_dict[cell_idx] = self._get_cell_content(cell)
                            for rowspan_idx in range(rowspan_counts - 1):
                                tmp_list = row_span_idx_dict.get(current_row_count + rowspan_idx)
                                if tmp_list is None:
                                    tmp_list = []
                                tmp_list.append(cell_idx)
                                row_span_idx_dict[current_row_count + rowspan_idx] = tmp_list
                        elif colspan_counts > 0:
                            for idx in range(colspan_counts):
                                header_columns_dict[(cell_idx + idx)] = self._get_cell_content(cell)
                    else:
                        # 如果是第二行及以后，将第一列单元格与第二列单元格合并，作为表头
                        # 过滤出剩余的表头，避免合并后出现重复表头,使用字典推导式过滤字典
                        filtered_header_columns_dict = {k: v for k, v in header_columns_dict.items() if k not in row_span_idx_dict[(current_row_count - 1)]}
                        filtered_header_columns_dict = {k: v for k, v in filtered_header_columns_dict.items() if len(filtered_header_columns_dict) > colspan_counts}
                        # 遍历单元格列合并表头
                        header_columns_colspan_idx = 0
                        # 取前列单元格列合的表头
                        for header_columns_idx, header_columns_value in filtered_header_columns_dict.items():
                            if rowspan_counts > 0:
                                header_columns_dict[header_columns_idx] = f"{header_columns_value}-{self._get_cell_content(cell)}"
                                for rowspan_idx in range(rowspan_counts - 1):
                                    tmp_list = row_span_idx_dict.get((current_row_count + rowspan_idx))
                                    if tmp_list is None:
                                        tmp_list = [header_columns_idx]
                                    else:
                                        tmp_list.append(tmp_list[-1] + last_colspan_counts + 1)
                                    row_span_idx_dict[current_row_count + rowspan_idx] = tmp_list
                            elif colspan_counts > 0:
                                if header_columns_colspan_idx < colspan_counts:
                                    header_columns_dict[header_columns_idx] = f"{header_columns_value}-{self._get_cell_content(cell)}"
                                    row_span_idx_dict[(current_row_count - 1)].append(header_columns_idx)
                                    header_columns_colspan_idx += 1
                    last_colspan_counts = colspan_counts
                current_row_count += 1
            else:
                # 如果是第一行，将当前行单元格添加为表头
                if current_row_count == 0:
                    cells = row.find_all('td')
                    for cell_idx, cell in enumerate(cells):
                        header_columns_dict[cell_idx] = self._get_cell_content(cell)
                else:
                    # 如果是第二行及以后，将第一列单元格与第二列单元格合并，作为表头
                    cells = row.find_all('td')
                    cells_idx = 0
                    filtered_header_columns_dict = {k: v for k, v in header_columns_dict.items() if k not in row_span_idx_dict[(current_row_count - 1)]}
                    for header_columns_idx, header_columns_value in filtered_header_columns_dict.items():
                        cell_text = cells[cells_idx]
                        header_columns_dict[header_columns_idx] = f"{header_columns_value}-{self._get_cell_content(cell_text)}"
                        cells_idx += 1

                break

        return header_columns_dict, header_row_count

    def _process_data_rows(self, table, header_row_count: int) -> List[List[str]]:
        """
        处理数据行
        """
        rows = table.find_all('tr')[header_row_count:]  # 跳过表头行
        data_rows = []
        last_data_row = None
        row_span_idxs = []
        for row in rows:
            cells = row.find_all('td')

            row_data = [self._get_cell_content(cell) for cell in cells]

            if last_data_row is not None and len(row_data) == len(last_data_row):
                row_span_idxs.clear()

            for i, cell in enumerate(cells):
                rowspan = self._get_cell_rowspan(cell)
                if rowspan > 1:
                    row_span_idxs.append(i)

            if last_data_row is not None and len(row_data) < len(last_data_row):
                for row_span_idx in row_span_idxs:
                    row_data.insert(row_span_idx, last_data_row[row_span_idx])

            if any(row_data):  # 确保不是空行
                data_rows.append(row_data)
                last_data_row = row_data

        return data_rows

    def parse_to_json(self) -> list:
        """
        将HTML表格解析为list
        """
        result = []

        for table in self.soup.find_all('table'):
            # 处理表头
            header_columns, header_rows_count = self._process_header_rows(table)
            # headers = self._process_column_headers(header_columns)

            # 处理数据行
            data_rows = self._process_data_rows(table, header_rows_count)

            # 构建JSON对象
            table_data = []
            for row in data_rows:
                row_dict = {}
                for i, value in enumerate(row):
                    if i < len(header_columns):
                        header = header_columns[i] or f'column_{i}'  # 如果没有表头，使用默认列名
                        row_dict[header] = value
                table_data.append(row_dict)

            result.append(table_data)

        # 如果只有一个表格，直接返回表格数据
        if len(result) == 1:
            return result[0]
        return result

    def parse_dict_to_json(self) -> list:
        result = []

        header_columns = []
        data_rows = []
        for row_idx, row in self.cells_dict.items():
            # for col_idx, column in row.items():
            # cell = column['cell']
            # colspan = column['colspan']
            # rowspan = column['rowspan']

            # 处理单元格内容
            # cell_html = []
            # for paragraph in cell.paragraphs:
            #     para_info = convert_paragraph_to_html(paragraph, {})
            #     if para_info['text']:
            #         cell_html.append(para_info['text'])
            #     if para_info['has_images']:
            #         cell_html.extend(para_info['html'])

            # 合并所有段落内容
            # cell_content = ''.join(cell_html) if cell_html else ' '

            # 处理表头
            if row_idx == 0:
                header_columns = row
                header_rows_count = 1
            else:
                # 处理数据行
                data_rows.append(row)

        # 构建JSON对象
        table_data = []
        for row in data_rows:
            row_dict = {}
            for i, value in enumerate(row):
                if i < len(header_columns):
                    header = header_columns[i] or f'column_{i}'  # 如果没有表头，使用默认列名
                    row_dict[header] = value
            table_data.append(row_dict)

        result.append(table_data)

        # 如果只有一个表格，直接返回表格数据
        if len(result) == 1:
            return result[0]
        return result

def parse_img_tag(img_tag: str) -> dict:
    """
        解析HTML的img标签，提取指定属性并转换为自定义字典格式

        Args:
            img_tag (str): HTML img标签字符串

        Returns:
            dict: 包含自定义字段的字典
        """
    # 使用正则表达式匹配所有属性
    pattern = r'(\w+)=["\']([^"\']*)["\']'
    attributes = re.findall(pattern, img_tag)

    # 创建属性映射
    temp_dict = {}
    for attr_name, attr_value in attributes:
        temp_dict[attr_name] = attr_value

    # 构建结果字典，只包含需要的字段
    result = {
        "图片HTML": img_tag,
        "图片路径": temp_dict.get("src", ""),
        "图片宽度": int(temp_dict.get("width", 0)),
        "图片高度": int(temp_dict.get("height", 0)),
        "图片样式": temp_dict.get("style", "")
    }

    return result