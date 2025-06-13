# pip install rapidocr-onnxruntime  -i https://pypi.tuna.tsinghua.edu.cn/simple/
# pip install rapid_table -i https://pypi.tuna.tsinghua.edu.cn/simple/
import json
import re
import os
import sys
from typing import Dict, List

from rapidocr_onnxruntime import RapidOCR
from rapid_table import RapidTable

#1111
from lineless_table_rec import LinelessTableRecognition
from lineless_table_rec.utils_table_recover import format_html, plot_rec_box_with_logic_info, plot_rec_box
from table_cls import TableCls
from wired_table_rec import WiredTableRecognition

from typing import Any, List, Tuple

table_engine = RapidTable()
ocr_engine = RapidOCR()

#1111
lineless_engine = LinelessTableRecognition()
wired_engine = WiredTableRecognition()
# 默认小yolo模型(0.1s)，可切换为精度更高yolox(0.25s),更快的qanything(0.07s)模型
table_cls = TableCls() # TableCls(model_type="yolox"),TableCls(model_type="q")

ocr_suffix = "ocr_tsr"

def read_markdown_file(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def write_markdown_file(file_path: str, content: str):
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)

# def read_json_file(file_path: str) -> List[Dict]:
#     with open(file_path, 'r', encoding='utf-8') as file:
#         return json.load(file)

def perform_ocr(img_path: str) -> str:
    ocr_result, _ = ocr_engine(img_path)
    table_html_str, table_cell_bboxes, elapse = table_engine(img_path, ocr_result)
    return table_html_str

def perform_ocr_tsr(img_path: str) -> str:
    cls,elasp = table_cls(img_path)
    if cls == 'wired':
        table_engine_tsr = wired_engine
    else:
        table_engine_tsr = lineless_engine
    html, elasp, polygons, logic_points, ocr_res = table_engine_tsr(img_path)
    return html

def replace_image_with_ocr_content(markdown_content: str, image_path: str, ocr_content: str) -> str:
    # 这里假设图片在Markdown中的格式是 ![alt text](image_path)
    image_pattern = f"!\\[.*?\\]\\({re.escape(image_path)}\\)"
    return re.sub(image_pattern, ocr_content, markdown_content)

def find_markdown_file(base_path: str) -> str:
    #auto_folder = os.path.join(base_path, 'pdf_md')
    auto_folder = base_path
    for file in os.listdir(auto_folder):
        if file.endswith('.md'):
            return os.path.join(auto_folder, file)
    return None

def read_json_file(file_path: str) -> Any:
    """读取JSON文件并返回其内容"""
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def find_image_paths(data: Any, paths: List[Tuple[str, str]] = None) -> List[Tuple[str, str]]:
    """
    递归查找JSON数据中的'image_path'和对应的'type'，当'type' == 'table'时保存。
    
    :param data: 当前的JSON数据，可以是字典、列表或其他类型
    :param paths: 用于存储找到的(type, image_path)元组
    :return: 包含符合条件的(type, image_path)的列表
    """
    if paths is None:
        paths = []

    if isinstance(data, dict):
        # 如果当前数据是字典，检查其中是否有 'image_path' 和 'type'
        image_path = data.get('img_path')
        item_type = data.get('type')
        # 当 'type' 为 'table' 且有 'image_path' 时，保存该路径
        if item_type == 'table' and image_path:
            paths.append((item_type, image_path))
        
        # 递归遍历字典中的所有子项
        for key, value in data.items():
            find_image_paths(value, paths)

    elif isinstance(data, list):
        # 如果当前数据是列表，遍历列表中的每一项
        for item in data:
            find_image_paths(item, paths)

    return paths

def update_json_content(data: Any, image_path: str, ocr_content: str) -> str:
    if isinstance(data, list):
        for item in data:
            if "img_path" in item:
                if item["img_path"] == image_path:
                    item["ocr_table"] = ocr_content
    return data

def write_json_file(file_path: str, data: Any):
    with open(file_path, "w+", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False, indent=4))

def main(base_path: str):
    # 查找Markdown文件
    markdown_file_path = find_markdown_file(base_path)
    if not markdown_file_path:
        print(f"错误：在 {os.path.join(base_path, 'auto')} 中未找到 Markdown 文件")
        return

    # 构建JSON文件路径
    markdown_filename = os.path.basename(markdown_file_path)
    #json_filename = "middle.json"# f"{os.path.splitext(markdown_filename)[0]}_content_list.json"
    json_filename = f"{os.path.splitext(markdown_filename)[0]}_content_list.json"
    json_filename_ocr = f"{os.path.splitext(markdown_filename)[0]}_content_list_{ocr_suffix}.json"
    #json_file_path = os.path.join(base_path, "auto", json_filename)
    json_file_path = os.path.join(base_path, json_filename)
    json_file_path_ocr = os.path.join(base_path, json_filename_ocr)

    # 检查文件是否存在
    if not os.path.exists(json_file_path):
        print(f"错误：无法找到JSON文件: {json_file_path}")
        return

    # 读取Markdown文件
    markdown_content = read_markdown_file(markdown_file_path)

    # 读取JSON文件
    json_data = read_json_file(json_file_path)

    # 查找所有符合条件的(image_path, type)
    image_paths = find_image_paths(json_data)

    print(image_paths)

    # 计算需要OCR处理的项目数量
    total_items = len(image_paths)
    ocr_count = 0

    # 处理JSON数据
    for item_type, img_path in image_paths:
        img_full_path = os.path.join(base_path, img_path)
        if os.path.exists(img_full_path):
            ocr_count += 1
            ocr_content = perform_ocr_tsr(img_full_path)
            print("img_path: {}, ocr_content: {}".format(img_path, ocr_content))
            #break
            #markdown_content = replace_image_with_ocr_content(markdown_content, img_path, ocr_content)
            update_json_content(json_data, img_path, ocr_content)
            print(f"OCR 进度: {ocr_count}/{total_items}")
        else:
            print(f"警告：图片文件不存在 {img_full_path}")

    # 保存更改后的Markdown文件
    #write_markdown_file(markdown_file_path, markdown_content)
    write_json_file(json_file_path_ocr, json_data)
    print(f"处理完成，已更新 {json_filename_ocr} 文件。")

if __name__ == "__main__":
    '''
    if len(sys.argv) != 2:
        print("用法: python TableOCR.py <base_path>")
        sys.exit(1)
    
    base_path = sys.argv[1]
    main(base_path)
    '''
    base_path = sys.argv[1]

    prefix = "B"
    count = 0
    dirs = os.listdir(base_path)
    dirs.sort()
    for dir_ in dirs:
        if dir_.startswith(prefix):
            dir_ = os.path.join(base_path, dir_)
            print("================= start to process ===================")
            print(dir_)
            main(dir_)
            count += 1

    print("total count: {}".format(count))
    
    #img_full_path = "/opt/kelvin/python/knowledge_graph/ai_contest/bdci_2024/pdf_md/AY01/images/107eb71e7f34b072cbb40eb57849ad3b452264722ebc247146612eef5995e795.jpg"
    #ocr_content = perform_ocr(img_full_path)
    #print(ocr_content)