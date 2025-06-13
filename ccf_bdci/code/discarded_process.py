import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import os
import sys
import re
from typing import Dict, List
from typing import Any, List, Tuple

current_dir = os.path.dirname(os.path.abspath(__file__))
pdf_md_path = os.path.join(current_dir, "pdf_md_B")
ocr_suffix = "ocr_tsr"

def read_json_file(file_path: str) -> Any:
    """读取JSON文件并返回其内容"""
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def write_json_file(file_path: str, data: Any):
    with open(file_path, "w+", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False, indent=4))

def file_dicarded_process(filename):
	middle_file_path = os.path.join(pdf_md_path, "{}/{}_middle.json".format(filename, filename))
	json_file_path = os.path.join(pdf_md_path, "{}/{}_content_list.json".format(filename, filename))
	json_file_ocr_path = os.path.join(pdf_md_path, "{}/{}_content_list_{}.json".format(filename, filename, ocr_suffix))
	json_file_final_path = os.path.join(pdf_md_path, "{}/{}_content_list_final.json".format(filename, filename))
    
	#with open(middle_file_path, "r", encoding="utf-8") as f:
		#middle_s = json.loads(f.read())

	middle_s = read_json_file(middle_file_path)
	recover_data = []
	merged_data = []
	print("len of pdf_info: ", len(middle_s["pdf_info"]))
	last_r_char_height = 0
	last_r_span_content = ""
	last_page_idx = -1
	last_r_span_bound = None
	last_block_type = None
	for i in range(0, len(middle_s["pdf_info"])):
		#print("len of preproc_blocks: ", len(middle_s["pdf_info"][i]["preproc_blocks"]))
		#print("len of discarded_blocks: ", len(middle_s["pdf_info"][i]["discarded_blocks"]))
		j = 0
		l = 0
		cur_page_idx = -1
		if "page_idx" in middle_s["pdf_info"][i]:
			cur_page_idx = middle_s["pdf_info"][i]["page_idx"]
		for j in range(0, len(middle_s["pdf_info"][i]["preproc_blocks"])):
			block = middle_s["pdf_info"][i]["preproc_blocks"][j]
			if "type" in block and (block["type"] == "text" or block["type"] == "title"):
				for l in range(0, len(block["lines"])):
					span_content = []
					char_height = 0
					line_bbox = block["lines"][l]["bbox"]
					for k in range(0, len(block["lines"][l]["spans"])):
						span_content.append(block["lines"][l]["spans"][k]["content"])
						if k == 0:
							char_h = block["lines"][l]["spans"][k]["bbox"][3]
							char_l = block["lines"][l]["spans"][k]["bbox"][1]
							char_height = char_h - char_l
					span_content = "".join(span_content)
					if j == 0 and l == 0 and last_page_idx != -1 and (cur_page_idx == last_page_idx + 1) and last_r_span_bound is not None and abs(line_bbox[2] - last_r_span_bound[2]) < 0.5:
						#print("last_line_merged_data found 1")
						#print("cur_page_idx: {}, span_content: [{}], last_page_idx: {}, last_r_span_content: [{}]".format(cur_page_idx, span_content, last_page_idx, last_r_span_content))
						pass
					elif l == 0 and last_block_type is not None and last_block_type == "title" and last_r_span_bound is not None and abs(line_bbox[2] - last_r_span_bound[2]) < 0.5 and (last_r_span_bound[0] - line_bbox[0]) > 10:
						print("last_line_merged_data found 2")
						print("cur_page_idx: {}, span_content: [{}], last_page_idx: {}, last_r_span_content: [{}]".format(cur_page_idx, span_content, last_page_idx, last_r_span_content))
						merged_data.append((span_content, last_r_span_content, cur_page_idx))
					last_r_span_content = span_content
					last_r_char_height = char_height
					last_r_span_bound = line_bbox
					last_block_type = block["type"]
		#print("regular lines {}-{}-{}, char_height:{}, content:[{}]".format(i,j,l,last_r_char_height,last_r_span_content))
		last_page_idx = cur_page_idx

		for j in range(0, len(middle_s["pdf_info"][i]["discarded_blocks"])):
			block = middle_s["pdf_info"][i]["discarded_blocks"][j]
			if "type" in block and block["type"] == "discarded":
				for l in range(0, len(block["lines"])):
					d_span_content = []
					char_height = 0
					for k in range(0, len(block["lines"][l]["spans"])):
						d_span_content.append(block["lines"][l]["spans"][k]["content"])
						if k == 0:
							char_h = block["lines"][l]["spans"][k]["bbox"][3]
							char_l = block["lines"][l]["spans"][k]["bbox"][1]
							char_height = char_h - char_l
					d_span_content = "".join(d_span_content)	
					if abs(last_r_char_height - char_height) <= 1 and len(d_span_content.replace(" ","")) >= 12:
						print("discarded lines {}-{}-{}, char_height:{}, content:[{}]".format(i,j,l,char_height, d_span_content))
						print("regular lines, last_page_idx: {}, char_height:{}, content:[{}]".format(last_page_idx, last_r_char_height,last_r_span_content))
						recover_data.append((d_span_content, last_r_span_content, last_page_idx))

	json_ocr_data = read_json_file(json_file_ocr_path)
	final_json_data = []
	if isinstance(json_ocr_data, list):
		for item in json_ocr_data:
			final_json_data.append(item)
			if "text" in item and "page_idx" in item:
				for r_item in recover_data:
					last_page_idx = r_item[2]
					last_r_span_content = r_item[1]
					append_content = r_item[0]
					if item["text"].find(last_r_span_content) != -1 and item["page_idx"] == last_page_idx:
						print("===== append recover data, page_idx: {}, last_r_span_content: [{}]".format(last_page_idx, last_r_span_content))
						print("text: [{}]".format(item["text"]))
						append_item = {"type": "text", "text": append_content,"page_idx": last_page_idx}
						final_json_data.append(append_item)
				for m_item in merged_data:
					cur_page_idx = m_item[2]
					last_m_span_content = m_item[1]
					cur_m_content = m_item[0]
					if item["text"].find(cur_m_content) != -1 and item["page_idx"] == cur_page_idx:
						print("===== merge data, page_idx: {}, last_m_span_content: [{}]".format(cur_page_idx, last_m_span_content))
						print("text: [{}]".format(item["text"]))
						final_json_data.pop()
						merged_item = final_json_data[-1]
						merged_item["text"] = merged_item["text"] + item["text"]

				if filename == "AT19" and item["text"].find("为深入开展学习贯彻习近平新时代中国特色社会主义思想主题教育") != -1:
					at19_append_item = {"type": "text", "text": "1.“三强化三坚持”重点是指以下三个点：", "text_level": 1, "page_idx": 0}
					print("===== append recover data, page_idx: {}, content: [{}]".format(0, "1.“三强化三坚持”重点是指以下三个点："))
					final_json_data.append(at19_append_item)

	write_json_file(json_file_final_path, final_json_data)
	print("write_json_file done, json_file_final_path: {}".format(json_file_final_path))

if __name__ == "__main__":
	prefix = "B"
	filters = []
	in_dir = pdf_md_path

	dir_files = os.listdir(in_dir)
	dir_files.sort()
	for dir_ in dir_files:
		if dir_.startswith(prefix) and dir_ not in filters:
			print("================= start to process ===================")
			print("*** {} ***".format(dir_))
			file_dicarded_process(dir_)
