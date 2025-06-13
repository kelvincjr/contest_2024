import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import os
import sys
import re
from collections import Counter
from typing import List
from trie_lib import trie_lib
from fuzzywuzzy import process,fuzz
from html2markdown import html_table_to_markdown_new
import jieba
import jieba.analyse
import jieba.posseg as pseg
from foc_doctree import gen_doctree, DocTree
from sentence_transformers import SentenceTransformer
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

apply_kaggle = True
apply_docker = True
if apply_docker:
	sys.path.insert(0, "/bdci_2024/gomate/GoMate")
elif apply_kaggle:
	sys.path.insert(0, "/kaggle/working/gomate/GoMate")
else:
	sys.path.insert(0, "/opt/kelvin/python/knowledge_graph/ai_contest/bdci_2024/gomate/GoMate")

from gomate.modules.document.chunk import TextChunker
from gomate.modules.document.txt_parser import TextParser
#from gomate.modules.document.utils import PROJECT_BASE
#from gomate.modules.generator.llm import GLM4Chat
from gomate.modules.reranker.bge_reranker import BgeRerankerConfig, BgeReranker
from gomate.modules.retrieval.bm25s_retriever import BM25RetrieverConfig, BM25Retriever
from gomate.modules.retrieval.dense_retriever import DenseRetrieverConfig
from gomate.modules.retrieval.hybrid_retriever import HybridRetriever, HybridRetrieverConfig

current_dir = os.path.dirname(os.path.abspath(__file__))
test_data_path = os.path.join(current_dir, 'B_question.csv')
keyphrases_data_path = os.path.join(current_dir, 'question_keyphrases.json')
#embedding_model_path = os.path.join(current_dir, 'models/bge_large/AI-ModelScope/bge-large-zh')
embedding_model_path = os.path.join(current_dir, 'models/bge-large-zh-v1.5/AI-ModelScope/bge-large-zh-v1___5')
reranker_model_path = os.path.join(current_dir, 'models/bge_reranker/quietnight/bge-reranker-large')
pdf_md_path = os.path.join(current_dir, "pdf_md")
valid_answer_path = os.path.join(current_dir, 'valid_answer.txt')
submit_answer_path = os.path.join(current_dir, 'kelvin_bdci_submit.csv')
valid_submit_answer_path = os.path.join(current_dir, 'kelvin_bdci_submit_valid.csv')
serialized_dict_path = os.path.join(current_dir, 'pdf_serialized_dict.pkl')
ali_textsplitter_model_path = 'damo/nlp_bert_document-segmentation_chinese-base'
finance_keywords_path = os.path.join(current_dir, 'data/user.dict.utf-8')
title_score_threshold = 2
rerank_score_l_threshold = 6
rerank_score_h_threshold = 8
extend_sentence_rerank_score_threshold = 0 #1 #3.5
sentence_rerank_score_threshold = 3.5

if apply_kaggle:
	current_dir = os.path.dirname(os.path.abspath(__file__))
	#test_data_path = os.path.join(current_dir, 'A_question.csv')
	test_data_path = os.path.join(current_dir, 'B_question.csv')
	keyphrases_data_path = os.path.join(current_dir, 'question_keyphrases.json')
	#embedding_model_path = os.path.join(current_dir, 'models/bge-large-zh/AI-ModelScope/bge-large-zh')
	embedding_model_path = os.path.join(current_dir, 'models/bge-large-zh-v1.5/AI-ModelScope/bge-large-zh-v1___5')
	reranker_model_path = os.path.join(current_dir, 'models/bge-reranker-large/Xorbits/bge-reranker-large')
	pdf_md_path = os.path.join(current_dir, "pdf_md")
	ali_textsplitter_model_path = os.path.join(current_dir, 'models/damo/nlp_bert_document-segmentation_chinese-base')

if apply_docker:
	current_dir = os.path.dirname(os.path.abspath(__file__))
	#test_data_path = os.path.join(current_dir, 'A_question.csv')
	test_data_path = os.path.join(current_dir, 'B_question.csv')
	keyphrases_data_path = os.path.join(current_dir, 'question_keyphrases.json')
	#embedding_model_path = os.path.join(current_dir, 'models/bge-large-zh/AI-ModelScope/bge-large-zh')
	embedding_model_path = os.path.join(current_dir, 'models/bge-large-zh-v1.5/AI-ModelScope/bge-large-zh-v1___5')
	reranker_model_path = os.path.join(current_dir, 'models/bge-reranker-large/Xorbits/bge-reranker-large')
	pdf_md_path = os.path.join(current_dir, "pdf_md")
	ali_textsplitter_model_path = os.path.join(current_dir, 'models/damo/nlp_bert_document-segmentation_chinese-base')
	#ali_textsplitter_model_path = 'damo/nlp_bert_document-segmentation_chinese-base'

class AliTextSplitter():
    def __init__(self, **kwargs):
    	device = "cpu"
    	if apply_kaggle:
    		device = "cuda:0"
    	self.p = pipeline(
    		task=Tasks.document_segmentation,
    		model=ali_textsplitter_model_path,
    		device=device)

    def split_text(self, text: str) -> List[str]:
    	result = self.p(documents=text)
    	sent_list = [i for i in result["text"].split("\n\t") if i]
    	return sent_list

def build_keyword_trie():
	kw_trie = trie_lib()
	with open(finance_keywords_path, "r", encoding="utf-8") as f:
		lines = f.readlines()
		kw_list = []
		for line in lines:
			item = line.strip().split(" ")
			word = item[0]
			pos = "fnkw"  #item[1]
			jieba.add_word(word, tag=pos)
			kw_list.append(word)

		kw_trie.build_trie(kw_list)
	return kw_trie

def extract_ngrams(text: str, n: int) -> List[str]:
    words = list(text)
    return [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]

def calculate_ngram_overlap(reference: str, generated: str, n: int):
    reference_ngrams = extract_ngrams(reference, n)
    generated_ngrams = extract_ngrams(generated, n)
    
    reference_counter = Counter(reference_ngrams)
    generated_counter = Counter(generated_ngrams)
    
    overlapping_ngrams = set(reference_counter.keys()) & set(generated_counter.keys())
    overlap_count = sum(min(reference_counter[ngram], generated_counter[ngram]) for ngram in overlapping_ngrams)    
    gen_ngrams_count = len(generated_ngrams) 

    score = overlap_count

    return score

def calculate_token_overlap(ques_tokens, title_tokens):
    counter1 = Counter(ques_tokens)
    counter2 = Counter(title_tokens)
    
    overlap_tokens = set(counter1.keys()) & set(counter2.keys())
    overlap_count = sum(min(counter1[token], counter2[token]) for token in overlap_tokens if len(token) > 1 or token == "年")    
    score = overlap_count
    return score, overlap_tokens

def sentence_split(str_centence):
	single_sentences_list = re.split(r'[。；？！\n]+', str_centence)
	return single_sentences_list

def endswith_punc(str_centence):
	single_sentences_list = re.split(r'[。；？！\n]+', str_centence)
	punc_regex = re.compile(r'[。；？！\n]$')
	punc_match = punc_regex.findall(str_centence)
	ret = False
	if len(punc_match) > 0:
		ret = True
	return ret

def check_foc_title(text):
    p = re.compile(r"^(第[一二三四五六七八九十]+[章节条])|^(（[一二三四五六七八九十]+）)|^(\([一二三四五六七八九十]+\))|^([一二三四五六七八九十]+、)|^([一二三四五六七八九十]+是)|^(\d+、)|^(\d{1,2} )[^年月日]|^(?P<num1>\d+\.)[^\d]*?|^(（\d+）)|^(\(\d+\))")
    matches = p.findall(text)
    matched_substrings = [(match, i) for group in matches for i, match in enumerate(group) if match]
    #print(matched_substrings)
    return matched_substrings

def check_per_name(kp):
	kp_sub_words = pseg.cut(kp)
	ret = False
	for sub_word in kp_sub_words:
		if sub_word.flag == 'nr':
			ret = True
			break
	return ret

def startswith_kp(answer, ques_id, keyphrases_data):
	ret = False
	keyphrases = keyphrases_data[ques_id]
	for kp in keyphrases:
		if answer.startswith(kp):
			ret = True
			break
	return ret

def check_ext_cb_case(question):
	ret = False
	if question.find("几年")!=-1:
		ret = True
	return ret

def get_test_data():
	df = pd.read_csv(test_data_path, header=0)
	data = []
	data_count = 0
	for i in range(len(df)):
		ques_id = df.iloc[i].at['ques_id']
		question = df.iloc[i].at['question']
		data.append((ques_id, question))
		data_count += 1
	return data

def get_keyphrases(all_titles):
	keyphrase_data = {}
	report_data = {}
	retriever = build_full_bm25_retriever(all_titles)
	year_pattern = r"\d{4}年"
	p_year = re.compile(year_pattern)
	with open(keyphrases_data_path, "r", encoding="utf-8") as f:
		lines = f.readlines()
		for line in lines:
			item = json.loads(line.strip())
			ques_id = item["id"]
			question = item["question"]
			phrases = item["phrases"]
			sorted_phrases = sorted(phrases.items(), key=lambda x: x[1], reverse=True)
			selected_keyphrases = []
			for key,value in sorted_phrases:
				included = True
				for kp in selected_keyphrases:
					if kp.find(key) != -1:
						included = False
						break
				it = p_year.finditer(key)
				for match in it:
					included = False
					break
				if key.find("报告") != -1 or key.find("年度") != -1 or key.find("季度") != -1 or key == "联通" or key == "中国联通":
					included = False
				if included and len(selected_keyphrases) <= 5:
					selected_keyphrases.append(key)

			refined_keyphrases = []
			for kp in selected_keyphrases:
				kp = kp.replace("什么", "").replace("哪些", "").replace("哪个", "").replace("中国联合网络通信股份有限公司的", "").replace("中国联合网络通信集团有限公司的", "").replace("中国联通的", "").replace("联通的", "").replace("中国联合网络通信股份有限公司", "").replace("中国联合网络通信集团有限公司", "").replace("中国联通", "").replace("联通", "").replace("联通", "")
				if kp.find("业务收入") != -1:
					kp = kp.replace("业务","")
				if kp != "" and kp != "联合":
					refined_keyphrases.append(kp)
			selected_keyphrases = refined_keyphrases

			#年份，年度、季度信息提取
			#year_regex_pattern = r'2\d{3}(?=年|\D|$)'
			year_regex_pattern = r'2\d{3}[年\D]?'
			year_regex = re.compile(year_regex_pattern)
			year_match = year_regex.findall(question)
			#year_match = [y.strip('年') for y in year_match]
			#year_match = [m.group(0) for m in year_match]
			
			#quarter_regex_pattern = r'(年度)|(年底)|([上下]?半年)|((第[一二三四])?季度)'
			quarter_regex_pattern = r'(年度)|(年底)|(半年)|(年年中)|((第[一二三四])?季度)'
			quarter_regex = re.compile(quarter_regex_pattern)
			quarter_match = quarter_regex.findall(question)
			quarter_match = [match for group in quarter_match for i, match in enumerate(group) if match and i<5]	
			quarter_match = [match if match != "年年中" else "半年" for match in quarter_match if match]
			quarter_match = [match if match != "年底" else "年度" for match in quarter_match if match]
			quarter_match = [match if match != "年度" else "年年度" for match in quarter_match if match]
			keyphrase_data[ques_id] = selected_keyphrases
			
			if len(year_match) > 0 and len(quarter_match) == 0:
				recall_files = keyphrase_docs_retrieve(selected_keyphrases, retriever)
				all_reports = True
				year_full = False
				for recall_file in recall_files:
					filename = recall_file[0]
					recall_title = recall_file[1]
					if not (filename.startswith("AY") or filename.startswith("BY")):
						all_reports = False
					for year_ in year_match:
						if recall_title.find(year_) != -1 and recall_title.find("年度") != -1:
							year_full = True

				if len(recall_files) > 0 and all_reports and year_full:
					quarter_match.append("年年度")

				if question.find("股东大会") != -1 and len(quarter_match) == 0:
					quarter_match.append("年年度")

			report_data[ques_id] = (year_match, quarter_match)
			print("ques_id: {}, question: {}, selected_keyphrases: {}, year_info: {}, quarter_info: {}".format(ques_id, question, selected_keyphrases, year_match, quarter_match))
	return keyphrase_data, report_data

def read_serialized_dict():
	serialized_dict = {}
	if not os.path.exists(serialized_dict_path):
		return serialized_dict
	with open(serialized_dict_path, 'rb') as f:
		serialized_dict = pickle.load(f)
		print("read serialized_dict done")
		return serialized_dict

def dump_serialized_dict(serialized_dict):
	if os.path.exists(serialized_dict_path):
		return
	with open(serialized_dict_path, 'wb') as f:
		pickle.dump(serialized_dict, f)
		print("dump serialized_dict done")
		return

def pdf_parse(file_path, ali_textsplitter):
	contents = []               #text_content, type, sub_sentences
	contents_toc = []
	tables = []
	title = ""
	second_title = ""
	AT_AF_can_title = ""
	filetype = ""
	file_toc = []

	if file_path.find("AY") != -1 or file_path.find("BY") != -1:
		filetype = "AY"
	elif file_path.find("AZ") != -1 or file_path.find("BZ") != -1:
		filetype = "AZ"
	elif file_path.find("AW") != -1 or file_path.find("BW") != -1:
		filetype = "AW"
	elif file_path.find("AT") != -1 or file_path.find("BT") != -1:
		filetype = "AT"
	elif file_path.find("AF") != -1 or file_path.find("BF") != -1:
		filetype = "AF"

	with open(file_path, "r", encoding="utf-8") as f:
		lines = f.readlines()
		lines = "".join(lines)
		json_cl = json.loads(lines)
		last_content = ""
		for i in tqdm(range(0, len(json_cl)), desc="process pdf file {}".format(file_path)):
			c_item = json_cl[i]
		#for c_item in json_cl:
			#c_item = json.loads(c_item)
			#title process
			if filetype == "AY":
				if title == "" and c_item["type"] == "text" and c_item["text"] != "" and "page_idx" in c_item and c_item["page_idx"] == 0:
					if c_item["text"].find("报告") != -1 or c_item["text"].find("资料") != -1:
						title = c_item["text"]
						title = title.replace("二○二三","2023").replace("二○二二","2022")
			
			if filetype == "AZ":
				if title == "" and c_item["type"] == "text" and c_item["text"] != "" and "page_idx" in c_item and c_item["page_idx"] == 0:
					if c_item["text"].find("报告") != -1 or c_item["text"].find("白皮书") != -1:
						title = c_item["text"]
				if "text_level" in c_item and "page_idx" in c_item and c_item["type"] == "text" and c_item["text_level"] == 1 and c_item["page_idx"] == 0 and c_item["text"]!="":
					second_title = c_item["text"]
			
			if filetype == "AW":
				if title == "" and "text_level" in c_item and "page_idx" in c_item and c_item["type"] == "text" and c_item["text_level"] == 1 and c_item["page_idx"] == 0 and c_item["text"]!="":
					title = c_item["text"]

			if filetype == "AF" or filetype == "AT":
				if title == "" and c_item["type"] == "text" and c_item["text"] != "" and "page_idx" in c_item and c_item["page_idx"] == 0:
					text = c_item["text"]

					if text.startswith("发布时间"):
						title = AT_AF_can_title
					elif text.find("发布时间") != -1:
						index = text.find("发布时间")
						AT_AF_can_title += text[:index]
						title = AT_AF_can_title

					if text.find("BDCI") == -1:
						AT_AF_can_title += text
					
					if second_title == "" and text.find("BDCI") == -1:
						second_title = text
			#content
			if c_item["type"] == "text" and c_item["text"] != "":
				content = c_item["text"].strip()

				if i < len(json_cl) - 1:
					c_next_item = json_cl[i + 1]
					punc_end = endswith_punc(content)
					c_pageidx = 0
					if "page_idx" in c_item:
						c_pageidx = c_item["page_idx"]
					n_pageidx = 0
					if "page_idx" in c_next_item:
						n_pageidx = c_next_item["page_idx"]
					c_text_level = 0
					if "text_level" in c_item:
						c_text_level = c_item["text_level"]
					n_text_level = 0
					if "text_level" in c_next_item:
						n_text_level = c_next_item["text_level"]
					if not punc_end and c_next_item["type"] == "text" and c_text_level != 1 and n_text_level != 1 and (c_pageidx + 1 == n_pageidx):
						last_content = content
						continue

				if last_content != "":
					content = last_content + content
					last_content = ""

				#ali textsplitter
				text_content = content.replace(" ","").replace(".","@")
				sub_sents = []
				if text_content != "":
					sub_sents = ali_textsplitter.split_text(text_content)
				text_content = text_content.replace("@",".")
				sub_sents = [sub_sent.replace("@",".") for sub_sent in sub_sents]
				contents.append((text_content, c_item["type"], sub_sents))
				content = text_content

				foc_flag = 0
				if "text_level" not in c_item:
					matched_strs = check_foc_title(content)
					if len(matched_strs) > 0:
						file_toc.append(content)
						foc_flag = 1
				elif "text_level" in c_item and c_item["text_level"] == 1:
					file_toc.append(content)
					foc_flag = 1
				contents_toc.append((content, foc_flag))

			#table
			if c_item["type"] == "table":
				table_caption = ""
				foc_flag = 0
				if "table_caption" in c_item:
					table_caption = c_item["table_caption"].strip()
					matched_strs = check_foc_title(table_caption)
					if len(matched_strs) > 0:
						file_toc.append(table_caption)
						foc_flag = 1

				md_table = ""
				if "ocr_table" in c_item:
					md_table = html_table_to_markdown_new(c_item["ocr_table"])
					#print("============= md table =============")
					#print(md_table)

				if table_caption != "" and md_table != "":
					table_content = table_caption + "\n" + md_table
					contents.append((table_content, c_item["type"], ""))
					contents_toc.append((table_content, foc_flag))
				elif md_table != "":
					table_content = md_table
					contents.append((table_content, c_item["type"], ""))
					contents_toc.append((table_content, foc_flag))
				elif table_caption != "":
					table_content = table_caption
					contents.append((table_content, c_item["type"], ""))
					contents_toc.append((table_content, foc_flag))

				if "table_body" in c_item:
					table_body = c_item["table_body"].strip()

			#toc
			#if "text_level" in c_item and c_item["text_level"] == 1:
			#	content = c_item["text"].strip()
			#	file_toc.append(content)
		
		if title == "" and second_title != "":
			title = second_title

		title = title.replace(" ", "")

		dt = DocTree("")
		content_to_nodes = gen_doctree(contents_toc, dt)
	return title, contents, tables, dt, content_to_nodes

def pdf_process():
	year_pattern = r"\d{4}年"
	p_year = re.compile(year_pattern)
			
	content_to_fileinfo = {}
	content_to_doctree_nodes = {}
	pdf_contents = []  # filename, title, contents, tables, toc
	title_to_filename = {}
	md_dirs = os.listdir(pdf_md_path)
	md_dirs.sort()
	ali_textsplitter = AliTextSplitter()
	#for md_dir in md_dirs:
	#serialized_dict = {}
	serialized_dict = read_serialized_dict()
	for i in tqdm(range(0, len(md_dirs)), desc="process all pdf file"):
		md_dir = md_dirs[i]
		#if not md_dir.startswith("BY05"):
		#	continue
		pdf_path = os.path.join(pdf_md_path, md_dir)
		#pdf_path = os.path.join(pdf_path, "{}_content_list.json".format(md_dir))
		#pdf_path = os.path.join(pdf_path, "{}_content_list_ocr.json".format(md_dir))
		pdf_path = os.path.join(pdf_path, "{}_content_list_final.json".format(md_dir))
		
		if md_dir in serialized_dict:
			title, contents, tables, dt, content_to_nodes = serialized_dict[md_dir]
		else:
			title, contents, tables, dt, content_to_nodes = pdf_parse(pdf_path, ali_textsplitter)
			serialized_dict[md_dir] = (title, contents, tables, dt, content_to_nodes)
		
		#dt.root.print_children()
		filename = md_dir
		title_to_filename[title] = filename
		content_to_doctree_nodes[filename] = content_to_nodes
		content_index = 0
		splitted_contents = []
		for content in contents:
			#print("============ seq: {} ===============".format(content_index))
			text_content = content[0]
			type_ = content[1]
			sub_sentences = content[2]
			splitted_contents.append(text_content)
			#print("type: {}, content: [{}]".format(type_, text_content))
			if type_ == "text":
				sent_seq = 0
				for sub_sent in sub_sentences:
					#print("          sent_seq: {}, sub_sent: {}".format(sent_seq, sub_sent))
					sent_seq += 1
					sub_sent = sub_sent.strip()
					if sub_sent != text_content:
						splitted_contents.append(sub_sent)
			content_index += 1
		print("filename: {}, title: {}, len of contents: {}, len of splitted_contents: {}".format(filename, title, len(contents), len(splitted_contents)))
	
		pdf_contents.append((filename, title, splitted_contents, tables, dt))
		for content in splitted_contents:
			if content not in content_to_fileinfo:
				content_to_fileinfo[content] = (filename, title)

	dump_serialized_dict(serialized_dict)

	all_titles = []
	all_titles_by_year = {}
	all_count = 0
	contents_by_title = {}
	for pdf_content in pdf_contents:
		fn = pdf_content[0]
		title = pdf_content[1]
		content = pdf_content[2]
		print("fn: {}, title: {}".format(fn, title))
		search_title = title.replace("中国联合网络通信股份有限公司", "").replace("中国联通", "")
		it = p_year.finditer(search_title)
		for match in it:
			match_year = match.group(0)
			if match_year not in all_titles_by_year:
				all_titles_by_year[match_year] = []
			all_titles_by_year[match_year].append(title)
		all_titles.append(title)
		contents_by_title[title] = content
		all_count += 1
	return all_titles, all_titles_by_year, contents_by_title, content_to_fileinfo, content_to_doctree_nodes, title_to_filename

def index_content_select(ques_id, question, all_titles, contents_by_title, title_to_filename):
	search_list = all_titles

	max_ngrams_score = 0
	max_score_title = ""
	score_titles = {}

	for search_title in search_list:
		#search_question = question.replace("中国联合网络通信股份有限公司", "").replace("中国联合网络通信集团有限公司", "").replace("中国联通", "").replace("联通", "")
		#ques_tokens = jieba.lcut(search_question)
		#filtered_title = search_title.replace("中国联合网络通信股份有限公司", "").replace("中国联合网络通信集团有限公司", "").replace("中国联通", "").replace("联通", "")
		#title_tokens = jieba.lcut(filtered_title)
		#score, overlap_tokens = calculate_token_overlap(ques_tokens, title_tokens)
		#ngram_score = calculate_ngram_overlap(search_question, filtered_title, 2)
		title_score = 0
		if search_title != "" and ques_id in report_data:
			yq_info = report_data[ques_id]
			year_info = yq_info[0]
			quarter_info = yq_info[1]
			info_list = []
			info_list.extend(year_info)
			info_list.extend(quarter_info)
			matched_info = []
			for info in info_list:
				if search_title.find(info) != -1:
					matched_info.append(info)
			if len(info_list) > 0:
				title_score = len(matched_info)
		score = title_score
		#print("ques_tokens: {}, title_tokens: {}, score: {}, ngram_score: {}".format(ques_tokens, title_tokens, score, ngram_score))
		#score = score + ngram_score
		if score not in score_titles:
			score_titles[score] = []
		score_titles[score].append(search_title)
		if score > max_ngrams_score:
			max_ngrams_score = score
			max_score_title = search_title
	
	match_titles = []
	if max_ngrams_score >= title_score_threshold:
		match_titles = score_titles[max_ngrams_score]		
		keep_match = True
		if match_titles[0] in title_to_filename:
			fn = title_to_filename[match_titles[0]]
			if not (fn.startswith("AY") or fn.startswith("BY")):
				keep_match = False

		if not keep_match:
			if (question.find("年度")!=-1 or question.find("季度")!=-1) and question.find("报告")!=-1:
				for score in range(title_score_threshold, max_ngrams_score):
					if score in score_titles:
						match_titles.extend(score_titles[score])
				final_match_titles = []
				for match_title in match_titles:
					if match_title in title_to_filename:
						fn = title_to_filename[match_title]
						if fn.startswith("AY") or fn.startswith("BY"):
							final_match_titles.append(match_title)
				match_titles = final_match_titles
			else:
				match_titles = all_titles
	else:
		match_titles = all_titles
	#match_titles.append(max_score_title)

	index_contents = []
	file_list = []
	for match_title in match_titles:
		if match_title in contents_by_title:
			contents = contents_by_title[match_title]
			for content in contents:
				index_contents.append(content)
				'''
				sentences = sentence_split(content)
				for sentence in sentences:
					index_contents.append(sentence)
				'''
		if match_title in title_to_filename:
			file_list.append(title_to_filename[match_title])

	if len(match_titles) != len(all_titles):
		print("ques_id {}, question {}, match_titles: {}, match_files: {}, match_score: {}".format(ques_id, question, match_titles, file_list, max_ngrams_score))
	else:
		print("===== ques_id {}, question {}, match_score: {}".format(ques_id, question, max_ngrams_score))
	
	return index_contents, match_titles, file_list

def rag_flow(test_data, keyphrases_data, report_data, all_titles, all_titles_by_year, contents_by_title, content_to_fileinfo, content_to_doctree_nodes, title_to_filename, full_hybrid_retriever, filter_ques_id):
	answers = []
	for item in test_data:
		ques_id = item[0]
		question = item[1]
		if filter_ques_id != -1 and ques_id != filter_ques_id:
			continue

		rank1_result, rank1_score, extend_result = question_process(ques_id, question, keyphrases_data, report_data, all_titles, all_titles_by_year, contents_by_title, content_to_fileinfo, content_to_doctree_nodes, title_to_filename, full_hybrid_retriever, False)

		if rank1_score < 0:
			rank1_result, rank1_score, extend_result = question_process(ques_id, question, keyphrases_data, report_data, all_titles, all_titles_by_year, contents_by_title, content_to_fileinfo, content_to_doctree_nodes, title_to_filename, full_hybrid_retriever, True)
		answers.append((ques_id, question, rank1_result, rank1_score, extend_result))

	final_answers = []
	emb_model = SentenceTransformer(embedding_model_path, trust_remote_code=True)
	for i in tqdm(range(0, len(answers)), desc="embedding sentences"):
		ques_id = answers[i][0]
		question = answers[i][1]
		answer = answers[i][2]
		sentence_embedding = emb_model.encode(answer, normalize_embeddings=True)
		str_embedding = ','.join([str(a) for a in sentence_embedding])
		#print(str_embedding)
		final_answers.append((ques_id, question, answer, str_embedding))

	return final_answers

def to_submit(answers):
	df_sub = pd.read_csv("./submit_example.csv")
	count = 0
	for item in answers:
		ques_id = item[0]
		question = item[1]
		answer = item[2]
		answer = answer.replace("\"", "")
		str_embedding = item[3]
		df_sub.iloc[count,0] = ques_id
		df_sub.iloc[count,1] = question
		df_sub.iloc[count,2] = answer
		df_sub.iloc[count,3] = str_embedding
		count += 1
	df_sub.to_csv("./kelvin_bdci_submit.csv", index=None)

def compare_submit():
	df_sub1 = pd.read_csv(os.path.join(current_dir, 'kelvin_bdci_submit_1121_b.csv'))
	df_sub2 = pd.read_csv(os.path.join(current_dir, 'kelvin_bdci_submit_1115_b_final.csv'))
	for i in range(len(df_sub1)):
		sub_ques_id_1 = df_sub1.iloc[i].at['ques_id']
		sub_question_1 = df_sub1.iloc[i].at['question']
		sub_answer_1 = df_sub1.iloc[i].at['answer']

		sub_ques_id_2 = df_sub2.iloc[i].at['ques_id']
		sub_question_2 = df_sub2.iloc[i].at['question']
		sub_answer_2 = df_sub2.iloc[i].at['answer']

		if sub_answer_1 != sub_answer_2:
			print("===== ques_id: {}, sub_question: {} =====\n".format(sub_ques_id_1, sub_question_1))
			print("sub_answer_1: [{}]\n".format(sub_answer_1))
			print("sub_answer_2: [{}]\n".format(sub_answer_2))

		#if len(sub_answer_1) > 300:
		#	print("===== long answer, ques_id: {}, sub_question: {} =====\n".format(sub_ques_id_1, sub_question_1))
		#	print("sub_answer_1: [{}]\n".format(sub_answer_1))
		#	print("sub_answer_2: [{}]\n".format(sub_answer_2))


def build_full_retriever(all_titles, contents_by_title, create_flag):
	print("start to build_full_retriever")
	index_contents = []
	for match_title in all_titles:
		if match_title in contents_by_title:
			contents = contents_by_title[match_title]
			for content in contents:
				index_contents.append(content)

	# BM25 and Dense Retriever configurations
	bm25_config = BM25RetrieverConfig(
		method='lucene',
		index_path='indexs/all_description_bm25.index',
		k1=1.6,
		b=0.7
	)
	bm25_config.validate()
	print(bm25_config.log_config())

	dense_config = DenseRetrieverConfig(
		model_name_or_path=embedding_model_path,
		dim=1024,
		index_path='indexs/all_dense_cache'
	)
	config_info = dense_config.log_config()
	print(config_info)

	# Hybrid Retriever configuration
	# 由于分数框架不在同一维度，建议可以合并
	hybrid_config = HybridRetrieverConfig(
		bm25_config=bm25_config,
		dense_config=dense_config,
		bm25_weight=0.7,  # bm25检索结果权重
		dense_weight=0.3  # dense检索结果权重
	)

	full_retriever = None
	if apply_kaggle:
		hybrid_retriever = HybridRetriever(config=hybrid_config)
		if create_flag:
			# 构建索引
			hybrid_retriever.build_from_texts(index_contents)
			# 保存索引
			hybrid_retriever.save_index()
			print("index created")
		else:
			hybrid_retriever.load_index()
			print("hybrid index loaded")
		full_retriever = hybrid_retriever
	else:
		bm25_retriever = BM25Retriever(config=bm25_config)
		# 构建索引
		bm25_retriever.build_from_texts(index_contents)
		# 保存索引
		bm25_retriever.save_index()
		print("bm25 index created")
				
		#bm25_retriever.load_index()
		full_retriever = bm25_retriever

	return full_retriever

def build_full_bm25_retriever(all_titles):
	index_contents = []
	for match_title in all_titles:
		if match_title in contents_by_title:
			contents = contents_by_title[match_title]
			for content in contents:
				index_contents.append(content)
				
	bm25_config = BM25RetrieverConfig(
		method='lucene',
		index_path='indexs/keyphrase_bm25.index',
		k1=1.6,
		b=0.7
	)
	bm25_config.validate()
	print(bm25_config.log_config())

	bm25_retriever = BM25Retriever(config=bm25_config)
	# 构建索引
	bm25_retriever.build_from_texts(index_contents)
	# 保存索引
	bm25_retriever.save_index()
	print("index created")
	retriever = bm25_retriever

	return retriever

def keyphrase_docs_retrieve(keyphrases, full_bm25_retriever):
	#for kp in keyphrases:
	#	jieba.add_word(kp)

	retriever = full_bm25_retriever
	query = " ".join(keyphrases)
	search_docs = retriever.retrieve(query, top_k=10)
	# Output results
	#print("============keyphrases: {}, recall result===================".format(keyphrases))
	rank = 0
	recall_files = []
	for result in search_docs:
		rank += 1
		filename = ""
		title = ""
		if result['text'] in content_to_fileinfo:
			filename, title = content_to_fileinfo[result['text']]
			recall_files.append((filename, title))
		#print(f"*** rank: {rank}, Score: {result['score']}, File: [{filename}], Title: [{title}], Text: [{result['text']}]\n")
	recall_files = list(set(recall_files))
	#print("recall files: ", recall_files)
	#print("============================================")

	#for kp in keyphrases:
	#	jieba.del_word(kp)

	return recall_files

def keyphrase_score(ques_id, keyphrases_data, sub_answer):
	kp_score = 0
	sub_answer_kps = []
	if ques_id in keyphrases_data:
		keyphrases = keyphrases_data[ques_id]
		for kp in keyphrases:
			if sub_answer.find(kp) != -1:
				sub_answer_kps.append(kp)
			else:
				kp_sub_words = pseg.cut(kp)
				kp_sub_words = [w.word for w in kp_sub_words if len(w.word) > 1]
				matched_sub_words = []
				for kp_sw in kp_sub_words:
					if sub_answer.find(kp_sw) != -1:
						matched_sub_words.append(kp_sw)
				if len(kp_sub_words) > 0 and len(matched_sub_words) == len(kp_sub_words):
					sub_answer_kps.append(kp)
				#elif len(kp_sub_words) > 0 and len(kp_sub_words) >= 3 and len(matched_sub_words) == (len(kp_sub_words) - 1):
				#	sub_answer_kps.append(kp)
				#elif len(matched_sub_words) > 0:
				#	sub_answer_kps.extend(matched_sub_words)
		if len(keyphrases) > 0:
			kp_score = float(len(sub_answer_kps)) / len(keyphrases)
	return kp_score, sub_answer_kps

def cal_title_score(title, ques_id, report_data):
	title_score = 0
	matched_info = []
	if title != "" and ques_id in report_data:
		yq_info = report_data[ques_id]
		year_info = yq_info[0]
		quarter_info = yq_info[1]
		info_list = []
		info_list.extend(year_info)
		info_list.extend(quarter_info)
		for info in info_list:
			if title.find(info) != -1:
				matched_info.append(info)
		if len(info_list) > 0:
			title_score = float(len(matched_info)) / len(info_list)
	return title_score, matched_info
	
def question_process(ques_id, question, keyphrases_data, report_data, all_titles, all_titles_by_year, contents_by_title, content_to_fileinfo, content_to_doctree_nodes, title_to_filename, full_hybrid_retriever, apply_full_retriever):

	retriever = full_hybrid_retriever
	if not apply_full_retriever:
		replace_kv = {"5G+":"6PG"}
		need_replace_kv = False
		for r_key in replace_kv.keys():
			if question.find(r_key) != -1:
				need_replace_kv = True
				break

		index_contents, match_titles, file_list = None, None, None
		if need_replace_kv:
			index_contents, match_titles, file_list = index_content_select_kv(ques_id, question, all_titles, contents_by_title, title_to_filename, keyphrases_data, replace_kv)
		else:
			index_contents, match_titles, file_list = index_content_select(ques_id, question, all_titles, contents_by_title, title_to_filename)
		if len(match_titles) != len(all_titles):
			# BM25 and Dense Retriever configurations
			bm25_config = BM25RetrieverConfig(
			    method='lucene',
			    index_path='indexs/description_bm25.index',
			    k1=1.6,
			    b=0.7
			)
			bm25_config.validate()
			print(bm25_config.log_config())

			dense_config = DenseRetrieverConfig(
			    model_name_or_path=embedding_model_path,
			    dim=1024,
			    index_path='indexs/dense_cache'
			)
			config_info = dense_config.log_config()
			print(config_info)

			# Hybrid Retriever configuration
			# 由于分数框架不在同一维度，建议可以合并
			hybrid_config = HybridRetrieverConfig(
			    bm25_config=bm25_config,
			    dense_config=dense_config,
			    bm25_weight=0.7,  # bm25检索结果权重
			    dense_weight=0.3  # dense检索结果权重
			)
			
			if apply_kaggle:
				hybrid_retriever = HybridRetriever(config=hybrid_config)
				
				# 构建索引
				hybrid_retriever.build_from_texts(index_contents)
				# 保存索引
				hybrid_retriever.save_index()
				print("index created")
				
				#hybrid_retriever.load_index()
				#print("index loaded")
				retriever = hybrid_retriever
			else:
				bm25_retriever = BM25Retriever(config=bm25_config)
				# 构建索引
				bm25_retriever.build_from_texts(index_contents)
				# 保存索引
				bm25_retriever.save_index()
				print("index created")
				#bm25_retriever.load_index()
				retriever = bm25_retriever

	query = question
	if ques_id in keyphrases_data: # and len(keyphrases_data[ques_id]) > 2:
		query = " ".join(keyphrases_data[ques_id])
	query = query + " " + question

	search_docs = retriever.retrieve(query, top_k=20)
	# Output results
	print("============ques_id: {}, recall result===================".format(ques_id))
	print("query: ", query)
	recall_scores = {}
	rank = 0
	recall_files = []
	for result in search_docs:
		rank += 1
		filename = ""
		title = ""
		if result['text'] in content_to_fileinfo:
			filename, title = content_to_fileinfo[result['text']]
			recall_files.append(filename)
		recall_scores[result['text']] = (rank, result['score'])
		print(f"*** rank: {rank}, Score: {result['score']}, File: [{filename}], Title: [{title}], Text: [{result['text']}]\n")
	recall_files = list(set(recall_files))
	print("recall files: ", recall_files)
	print("============================================")

	
	# ====================排序配置=========================
	#query = query + " " + question + " " + question
	reranker_config = BgeRerankerConfig(
	    model_name_or_path=reranker_model_path
	)

	bge_reranker = BgeReranker(reranker_config)
	print("start to rerank")
	search_docs = bge_reranker.rerank(
		query=query,
		documents=[doc['text'] for idx, doc in enumerate(search_docs)]
	)
	print("============ques_id: {}, rerank result===================".format(ques_id))
	rank = 0
	rank1_result = ""
	rank1_title = ""
	rank1_score = 0
	rank1_kp_score = 0
	rank1_title_score = 0
	rank1_token_score = 0
	rank1_kp_per_flag = False
	rank1_startswith_kp_flag = False
	extend_results = [[],[]]   #sent1, sent2
	is_sub_title = False
	for result in search_docs:
		rank += 1
		filename = ""
		title = ""
		content_to_nodes = None
		if result['text'] in content_to_fileinfo:
			filename, title = content_to_fileinfo[result['text']]
			if filename in content_to_doctree_nodes:
				content_to_nodes = content_to_doctree_nodes[filename]
		#fuzz score
		fuzz_score = fuzz.ratio(query, result['text'])
		#token overlap score
		query_parts = query.split(" ")
		query_search = " ".join(query_parts[:-1])
		ques_tokens = jieba.lcut(query_search)
		recall_tokens = jieba.lcut(result['text'])
		token_score, overlap_tokens = calculate_token_overlap(ques_tokens, recall_tokens)
		#keyphrase score
		sub_answer = result['text']
		kp_score, sub_answer_kps = keyphrase_score(ques_id, keyphrases_data, sub_answer)
		per_name_flag = check_per_name(" ".join(sub_answer_kps))
		startswith_kp_flag = startswith_kp(sub_answer, ques_id, keyphrases_data)
		#title score
		title_score, title_matched_info = cal_title_score(title, ques_id, report_data)
		#recall score
		recall_score = 0
		if result['text'] in recall_scores:
			recall_score = recall_scores[result['text']]

		print(f"*** rank: {rank}, Rerank_Score: {result['score']}, Fuzz_Score: {fuzz_score}, Kp_Score: {kp_score}, Answer_kps: {sub_answer_kps}, Title_Score: {title_score}, Title_matched_info: {title_matched_info}, Token_Score: {token_score}, Overlap_tokens: {overlap_tokens}, Recall_Score: {recall_score}, File: [{filename}], Title: [{title}], Text: [{result['text']}]\n")
		if rank == 1:
			rank1_score = result['score']
			rank1_result = result['text']
			rank1_title = title
			rank1_kp_score = kp_score
			rank1_token_score = token_score
			rank1_title_score = title_score
			rank1_kp_per_flag = per_name_flag
			rank1_startswith_kp_flag = startswith_kp_flag 
			if rank1_result in content_to_nodes:
				print("content_to_nodes: ", content_to_nodes[rank1_result])
			if rank1_result in content_to_nodes and content_to_nodes[rank1_result].type_ != -1:	
				print("before all_child_content")
				is_sub_title = True
				if len(content_to_nodes[rank1_result].children) > 0:
					content_to_nodes[rank1_result].all_child_content(extend_results)
				else:
					content_to_nodes[rank1_result].all_sibling_content(extend_results)
			if rank1_result == title:
				rank1_score = 0
				rank1_result = ""
				rank = 0      #1114
		if rank == 2 or rank == 3 or (rank > 3 and kp_score == 1 and rank1_kp_score < kp_score) or (rank > 3 and startswith_kp_flag and not rank1_startswith_kp_flag and kp_score >= rank1_kp_score):  #1115
			rank23_match = False
			if rank1_score == 0:
				print("match rank2 rank3 case1")
				rank23_match = True
				rank1_result = result['text']
				rank1_score = result['score']
				rank1_title = title
				rank1_kp_score = kp_score
				rank1_title_score = title_score
				rank1_token_score = token_score
				extend_results = [[],[]] 
				is_sub_title = False
				if rank1_result in content_to_nodes:
					print("content_to_nodes: ", content_to_nodes[rank1_result])
				if rank1_result in content_to_nodes and content_to_nodes[rank1_result].type_ != -1:	
					print("before all_child_content")
					is_sub_title = True
					if len(content_to_nodes[rank1_result].children) > 0:
						content_to_nodes[rank1_result].all_child_content(extend_results)
					else:
						content_to_nodes[rank1_result].all_sibling_content(extend_results)
			elif rank1_score != 0 and rank1_score < rerank_score_h_threshold and (rank1_score - result['score']) < 1.55 and ((kp_score > rank1_kp_score and token_score > rank1_token_score) or startswith_kp_flag) and not (rank1_title_score > 0 and title_score < rank1_title_score) and not rank1_kp_per_flag:   #1115
				print("match rank2 rank3 case2, startswith_kp_flag: {}".format(startswith_kp_flag))
				rank23_match = True
				pre_rank1_result = rank1_result
				rank1_result = result['text']
				rank1_score = result['score']
				rank1_title = title
				rank1_kp_score = kp_score
				rank1_title_score = title_score
				rank1_token_score = token_score
				if len(extend_results[0]) == 0:
					if rank1_result in content_to_nodes:
						print("content_to_nodes: ", content_to_nodes[rank1_result])
					if rank1_result in content_to_nodes and content_to_nodes[rank1_result].type_ != -1:	
						print("before all_child_content")
						is_sub_title = True
						if len(content_to_nodes[rank1_result].children) > 0:
							#print("before all_child_content 1, extend_results: ", extend_results)
							content_to_nodes[rank1_result].all_child_content(extend_results)
							#print("after all_child_content 1, extend_results: ", extend_results)
						else:
							content_to_nodes[rank1_result].all_sibling_content(extend_results)
				else:
					if rank1_result in content_to_nodes:
						print("content_to_nodes: ", content_to_nodes[rank1_result])
					if rank1_result in content_to_nodes and content_to_nodes[rank1_result].type_ != -1:	
						print("before all_child_content")
						extend_results = [[],[]]
						is_sub_title = True
						if len(content_to_nodes[rank1_result].children) > 0:
							print("before all_child_content 2, extend_results: ", extend_results)
							content_to_nodes[rank1_result].all_child_content(extend_results)
							print("after all_child_content 2, extend_results: ", extend_results)
						else:
							content_to_nodes[rank1_result].all_sibling_content(extend_results)
					elif rank1_result in content_to_nodes and content_to_nodes[rank1_result].type_ == -1:
						if len(rank1_result) < 30:
							is_sub_title = True
							pre_rank1_sentences = sentence_split(pre_rank1_result)
							extend_results[0].insert(0, pre_rank1_sentences[0])
			elif rank1_score != 0 and (rank1_score - result['score']) < 0.5 and kp_score > 0 and (rank1_kp_score > 0 and rank1_kp_score < 1):
				combined_para = rank1_result + "\n" + result['text']
				if title == rank1_title:
					sub_answer = combined_para
					cb_kp_score, cb_sub_answer_kps = keyphrase_score(ques_id, keyphrases_data, sub_answer)
					print(f"try rank2 rank3 case4, rank1_kp_score: {rank1_kp_score}, cb_kp_score: {cb_kp_score}, cb_sub_answer_kps: {cb_sub_answer_kps}")
					if cb_kp_score > rank1_kp_score and rank1_result.find(result['text']) == -1 and result['text'].find(rank1_result) == -1:
						rank23_match = True
						print("match rank2 rank3 case4")
						rank1_result = sub_answer
						rank1_score = result['score']
						rank1_title = title
						rank1_kp_score = cb_kp_score
						rank1_title_score = title_score
						rank1_token_score = token_score
			'''
			elif rank1_score != 0 and (rank1_score - result['score']) < 0.5 and (kp_score == rank1_kp_score) and rank1_kp_score > 0.5 and rank1_result.find(result['text']) != -1:
				print("match rank2 rank3 case3")
				rank23_match = True
				pre_rank1_result = rank1_result
				rank1_result = result['text']
				rank1_score = result['score']
				rank1_title = title
				rank1_kp_score = kp_score
				rank1_title_score = title_score
				rank1_token_score = token_score
				if len(extend_results) == 0:
					if rank1_result in content_to_nodes:
						print("content_to_nodes: ", content_to_nodes[rank1_result])
					if rank1_result in content_to_nodes and content_to_nodes[rank1_result].type_ != -1:	
						print("before all_child_content")
						is_sub_title = True
						if len(content_to_nodes[rank1_result].children) > 0:
							content_to_nodes[rank1_result].all_child_content(extend_results)
						else:
							content_to_nodes[rank1_result].all_sibling_content(extend_results)
				else:
					if rank1_result in content_to_nodes:
						print("content_to_nodes: ", content_to_nodes[rank1_result])
					if rank1_result in content_to_nodes and content_to_nodes[rank1_result].type_ != -1:	
						print("before all_child_content")
						extend_results = []
						is_sub_title = True
						if len(content_to_nodes[rank1_result].children) > 0:
							content_to_nodes[rank1_result].all_child_content(extend_results)
						else:
							content_to_nodes[rank1_result].all_sibling_content(extend_results)
					elif rank1_result in content_to_nodes and content_to_nodes[rank1_result].type_ == -1:
						if len(rank1_result) < 30:
							is_sub_title = True
							pre_rank1_sentences = sentence_split(pre_rank1_result)
							extend_results.insert(0, pre_rank1_sentences[0])
			'''
			
			if not rank23_match and rank1_score != 0 and rank1_score > rerank_score_h_threshold and (rank1_score - result['score']) < 0.5:
				if title == rank1_title:
					case_fuzz_score = fuzz.ratio(rank1_result, result['text'])
					print(f"try rank2 rank3 case5, case_fuzz_score: {case_fuzz_score}, rank1_score: {rank1_score}, cur_score: {result['score']}")
					if case_fuzz_score > 60 and not (rank1_result.replace("朱常波","朱常").find(result['text']) != -1 or result['text'].replace("朱常波","朱常").find(rank1_result) != -1):
						print("match rank2 rank3 case5")
						sub_answer = rank1_result + "\n" + result['text']
						rank1_result = sub_answer
						rank1_score = result['score']
						rank1_title = title
						rank1_kp_score = kp_score
						rank1_title_score = title_score
						rank1_token_score = token_score

	print("ques_id: {}, rank1_score: {}, rank1_result: {}".format(ques_id, rank1_score, rank1_result))
	extend_result1 = "\n".join(extend_results[0])
	extend_result2 = "\n".join(extend_results[1])
	extend_result1 = rank1_result + "\n" + extend_result1
	extend_result2 = rank1_result + "\n" + extend_result2
	print("ques_id: {}, extend_result1: {}, extend_result2: {}".format(ques_id, extend_result1, extend_result2))
	if rank1_score <= 0:
		print("match rank1_score <= 0 case, rank1_score: {}".format(rank1_score))
		rank1_result = question
	print("============================================")

	final_result = rank1_result
	final_score = rank1_score
	need_sentence_case = True
	#extend case
	if len(final_result) < 30 and is_sub_title:
		final_result = extend_result1
		#extend result rerank
		print("============ques_id: {}, extend_result rerank result===================".format(ques_id))
		print("start to rerank rank1 sentences")
		extend_sentences = extend_result1.split("\n")
		sent_seqs = {}
		combined_sentences = []
		for idx, sent in enumerate(extend_sentences):
			sent_seqs[sent] = idx
		extend_search_docs = bge_reranker.rerank(
			query=query,
			documents=[sent for idx, sent in enumerate(extend_sentences)]
		)
		rank = 0
		for result in extend_search_docs:
			rank += 1
			if rank == 1:
				pass
			sent_seq = -1
			if result['text'] in sent_seqs:
				sent_seq = sent_seqs[result['text']]

			#matched_titles = check_foc_title(result['text'])
			if result['score'] > extend_sentence_rerank_score_threshold:
				combined_sentences.append((result['text'], sent_seq))
			#elif len(matched_titles) > 0:      #1104
			#	combined_sentences.append((result['text'], sent_seq))
			sent_recall = result['text']
			sent_kp_score, sent_kps = keyphrase_score(ques_id, keyphrases_data, sent_recall)
		
			print(f"*** Extend Sent rank: {rank}, Sent seq: {sent_seq}, Rerank_score: {result['score']}, Kp_score: {sent_kp_score}, Sent_kps: {sent_kps}, Text: [{result['text']}]\n")
		print("============================================")

		sorted_sents = sorted(combined_sentences, key=lambda x: x[1], reverse=False)
		sorted_sents = [sent for sent,seq in sorted_sents]
		combined_sentence = "。".join(sorted_sents)
		combined_list = [extend_result1, extend_result2, combined_sentence]
		#combined_list = [extend_result1, combined_sentence]  #1115
		extend_search_docs = bge_reranker.rerank(
			query=query,
			documents=[para for idx, para in enumerate(combined_list)]
		)
		rank = 0
		for result in extend_search_docs:
			rank += 1
			if rank == 1: #rerank_score_l_threshold:
				print("match extend sentence rerank case")
				final_result = result['text']
				final_score = result['score']
				need_sentence_case = False
			print(f"*** Extend combined rank: {rank}, Score: {result['score']}, Text: [{result['text']}]\n")
		print("============================================")
	elif len(final_result) < 30:
		rank = 0
		for result in search_docs:
			rank += 1
			if rank == 1 or result['text'] == rank1_result:
				continue
			if rank == 2:
				if (rank1_score - result['score']) < 0.5:
					combined_para = rank1_result + "\n" + result['text']
					final_result = combined_para
					print("match len < 30 rank1 and rank2 combine case")
				break
	#combine case "和"
	cb_case_flag = check_ext_cb_case(question)
	if (rank1_kp_score < 1 or rank1_title_score < 1) and (question.find("和")!=-1 or question.find("与")!=-1 or question.find("年及")!=-1 or cb_case_flag):
		print("============ques_id: {}, combined rerank result===================".format(ques_id))
		rank1_filename = ""
		rank1_title = ""
		if rank1_result in content_to_fileinfo:
			rank1_filename, rank1_title = content_to_fileinfo[rank1_result]
		rank = 0
		cb_result_found = False
		for result in search_docs:
			rank += 1
			if rank == 1 or result['text'] == rank1_result:
				continue
			if result['score'] < 1:
				break
			combined_para = rank1_result + "\n" + result['text']
			combined_title = rank1_title
			if result['text'] in content_to_fileinfo:
				cb_filename, cb_title = content_to_fileinfo[result['text']]
				combined_title = rank1_title + "\n" + cb_title
			sub_answer = combined_para
			cb_kp_score, cb_sub_answer_kps = keyphrase_score(ques_id, keyphrases_data, sub_answer)
			cur_answer = result['text']
			cur_kp_score, cur_sub_answer_kps = keyphrase_score(ques_id, keyphrases_data, cur_answer)
			cb_title_score, cb_title_matched_info = cal_title_score(combined_title, ques_id, report_data)
			cur_title_score, cur_title_matched_info = cal_title_score(cb_title, ques_id, report_data)

			if not cb_result_found and ((rank1_kp_score > 0 and rank1_kp_score < 1 and cb_kp_score > rank1_kp_score) or (rank1_title_score > 0 and rank1_title_score < 1 and cb_title_score > rank1_title_score)):
				if rank1_title_score == 1 and cur_title_score != 1:
					print(f"*** Case1 Combined rank: {rank} and 1, Cb_kp_score: {cb_kp_score}, Cb_answer_kps: {cb_sub_answer_kps}, Cur_kp_score: {cur_kp_score}, Cur_answer_kps: {cur_sub_answer_kps}, Cb_title_score: {cb_title_score}, Cur_title_score: {cur_title_score}, Title_matched_info: {cur_title_matched_info}, Text: [{combined_para}]\n")
				elif rank1_kp_score == 1 and cur_kp_score != 1:
					print(f"*** Case2 Combined rank: {rank} and 1, Cb_kp_score: {cb_kp_score}, Cb_answer_kps: {cb_sub_answer_kps}, Cur_kp_score: {cur_kp_score}, Cur_answer_kps: {cur_sub_answer_kps}, Cb_title_score: {cb_title_score}, Cur_title_score: {cur_title_score}, Title_matched_info: {cur_title_matched_info}, Text: [{combined_para}]\n")
				elif cur_kp_score == 0:
					print(f"*** Case3 Combined rank: {rank} and 1, Cb_kp_score: {cb_kp_score}, Cb_answer_kps: {cb_sub_answer_kps}, Cur_kp_score: {cur_kp_score}, Cur_answer_kps: {cur_sub_answer_kps}, Cb_title_score: {cb_title_score}, Cur_title_score: {cur_title_score}, Title_matched_info: {cur_title_matched_info}, Text: [{combined_para}]\n")
				else:
					final_result = combined_para
					print("match combined rerank case")
					print(f"*** Find Combined rank: {rank} and 1, Cb_kp_score: {cb_kp_score}, Cb_answer_kps: {cb_sub_answer_kps}, Cur_kp_score: {cur_kp_score}, Cur_answer_kps: {cur_sub_answer_kps}, Cb_title_score: {cb_title_score}, Cur_title_score: {cur_title_score}, Title_matched_info: {cur_title_matched_info}, Text: [{final_result}]\n")
					cb_result_found = True
					need_sentence_case = False
			elif not cb_result_found and cb_case_flag:
				final_result = combined_para
				print("match combined rerank case 1")
				print(f"*** Find Combined rank: {rank} and 1, Cb_kp_score: {cb_kp_score}, Cb_answer_kps: {cb_sub_answer_kps}, Cur_kp_score: {cur_kp_score}, Cur_answer_kps: {cur_sub_answer_kps}, Cb_title_score: {cb_title_score}, Cur_title_score: {cur_title_score}, Title_matched_info: {cur_title_matched_info}, Text: [{final_result}]\n")
				cb_result_found = True
				need_sentence_case = False
			else:
				print(f"*** Combined rank: {rank} and 1, Cb_kp_score: {cb_kp_score}, Cb_answer_kps: {cb_sub_answer_kps}, Cur_kp_score: {cur_kp_score}, Cur_answer_kps: {cur_sub_answer_kps}, Cb_title_score: {cb_title_score}, Cur_title_score: {cur_title_score}, Title_matched_info: {cur_title_matched_info}, Text: [{combined_para}]\n")
		print("============================================")
		'''
		print("============ques_id: {}, combined rerank result===================".format(ques_id))
		combined_paras = [rank1_result]
		rank = 0
		for result in search_docs:
			rank += 1
			if rank == 1:
				continue
			combined_para = rank1_result + "\n" + result['text']
			combined_paras.append(combined_para)

		combined_search_docs = bge_reranker.rerank(
			query=query,
			documents=[para for idx, para in enumerate(combined_paras)]
		)
		rank = 0
		for result in combined_search_docs:
			rank += 1
			if rank == 1 and result['score'] > rerank_score_l_threshold:
				final_result = result['text']
				final_score = result['score']
			print(f"*** Combined rank: {rank}, Score: {result['score']}, Text: [{result['text']}]\n")
		print("============================================")
		'''
	#table rerank
	table_sep_pattern = r'\|'
	table_sep_regex = re.compile(table_sep_pattern)
	table_sep_match = table_sep_regex.findall(final_result)
	if len(table_sep_match) > 2:
		print("============ques_id: {}, table rerank result===================".format(ques_id))
		print("start to rerank table lines")
		table_lines = final_result.split("\n")
		line_seqs = {}
		combined_table_lines = []
		for idx, sent in enumerate(table_lines):
			line_seqs[sent] = idx
		table_search_docs = bge_reranker.rerank(
			query=query,
			documents=[sent for idx, sent in enumerate(table_lines)]
		)
		rank = 0
		for result in table_search_docs:
			rank += 1
			if rank == 1:
				pass
			line_seq = -1
			if result['text'] in line_seqs:
				line_seq = line_seqs[result['text']]

			line_recall = result['text']
			line_kp_score, line_kps = keyphrase_score(ques_id, keyphrases_data, line_recall)

			if result['score'] > 0 or line_kp_score > 0:
				combined_table_lines.append((result['text'], line_seq))

			print(f"*** Table Line rank: {rank}, Line seq: {line_seq}, Rerank_score: {result['score']}, Kp_score: {line_kp_score}, Sent_kps: {line_kps}, Text: [{result['text']}]\n")
		print("============================================")

		sorted_table_lines = sorted(combined_table_lines, key=lambda x: x[1], reverse=False)
		sorted_table_lines = [sent for sent,seq in sorted_table_lines]
		combined_table_line = "。".join(sorted_table_lines)
		combined_list = [final_result, combined_table_line]
		extend_search_docs = bge_reranker.rerank(
			query=query,
			documents=[para for idx, para in enumerate(combined_list)]
		)
		rank = 0
		for result in extend_search_docs:
			rank += 1
			if rank == 1: #rerank_score_l_threshold:
				print("match table line rerank case")
				final_result = result['text']
				final_score = result['score']
				need_sentence_case = False
			print(f"*** Combined Table Line rank: {rank}, Score: {result['score']}, Text: [{result['text']}]\n")
		#1112
		#final_result = final_result.replace("|","")
		print("============================================")

	#long sentences rerank
	if rank1_score < rerank_score_l_threshold and len(rank1_result) > 300 and need_sentence_case:
		print("============ques_id: {}, sentence rerank result===================".format(ques_id))
		print("start to rerank rank1 sentences")
		sub_sentences = sentence_split(rank1_result)
		sent_seqs = {}
		for idx, sent in enumerate(sub_sentences):
			sent_seqs[sent] = idx
		search_docs = bge_reranker.rerank(
			query=query,
			documents=[sent for idx, sent in enumerate(sub_sentences)]
		)
		rank = 0
		rank1_sentence = ""
		rank1_sent_score = 0
		combined_sentences = []
		for result in search_docs:
			rank += 1
			if rank == 1:
				rank1_sent_score = result['score']
				rank1_sentence = result['text']
			sent_seq = -1
			if result['text'] in sent_seqs:
				sent_seq = sent_seqs[result['text']]
			if result['score'] > sentence_rerank_score_threshold:
				combined_sentences.append(result['text'])

			sent_recall = result['text']
			sent_kp_score, sent_kps = keyphrase_score(ques_id, keyphrases_data, sent_recall)
		
			print(f"*** Sent rank: {rank}, Sent seq: {sent_seq}, Rerank_score: {result['score']}, Kp_score: {sent_kp_score}, Sent_kps: {sent_kps}, Text: [{result['text']}]\n")
		print("============================================")

		combined_sentence = "。".join(combined_sentences)
		#combined_list = [rank1_result, combined_sentence]
		combined_list = [rank1_result]
		cur_combined_sentence = ""
		for s_i in range(len(combined_sentences)):
			sub_sentence = combined_sentences[s_i]
			cur_combined_sentence += sub_sentence + "。"
			combined_list.append(cur_combined_sentence)
		combined_search_docs = bge_reranker.rerank(
			query=query,
			documents=[para for idx, para in enumerate(combined_list)]
		)
		rank = 0
		for result in combined_search_docs:
			rank += 1
			if rank == 1 and result['score'] > rank1_score: #rerank_score_l_threshold:
				print("match sentence rerank case")
				final_result = result['text']
				final_score = result['score']
			print(f"*** Final combined rank: {rank}, Score: {result['score']}, Text: [{result['text']}]\n")
		print("============================================")

	print("ques_id: {}, final_score: {}, len: {}, final_result: [{}]".format(ques_id, final_score, len(final_result), final_result))
	
	return final_result, final_score, extend_result1

def build_full_bm25_retriever_kv(keyphrases, all_titles, replace_kv):
	#for kp in keyphrases:
	#	jieba.add_word(kp,tag="n",freq=100000)
	index_contents = []
	corpus_to_title = {}
	for match_title in all_titles:
		if match_title in contents_by_title:
			contents = contents_by_title[match_title]
			corpus_content = ""
			for content in contents:
				fixed_content = content
				for k,v in replace_kv.items():
					fixed_content = fixed_content.replace(k,v)
				corpus_content += fixed_content + "\n"
			index_contents.append(corpus_content)
			corpus_to_title[corpus_content] = match_title
				
	bm25_config = BM25RetrieverConfig(
		method='lucene',
		index_path='indexs/keyphrase_bm25.index',
		k1=1.6,
		b=0.7
	)
	bm25_config.validate()
	print(bm25_config.log_config())

	bm25_retriever = BM25Retriever(config=bm25_config)
	# 构建索引
	bm25_retriever.build_from_texts(index_contents)
	# 保存索引
	bm25_retriever.save_index()
	print("index created")
	retriever = bm25_retriever

	return retriever, corpus_to_title

def keyphrase_docs_retrieve_kv(keyphrases, retriever, corpus_to_title):
	all_recall_list = []
	for kp in keyphrases:
		if kp == "成果":
			continue
		query = kp
		search_docs = retriever.retrieve(query, top_k=30)
		# Output results
		print("============keyphrases: {}, recall result===================".format(keyphrases))
		rank = 0
		recall_files = []
		for result in search_docs:
			rank += 1
			filename = ""
			title = ""
			if result['score'] <= 0:
				break
			if result['text'] in corpus_to_title:
				title = corpus_to_title[result['text']]
				recall_files.append(title)
			print(f"*** rank: {rank}, Score: {result['score']}, File: [{filename}], Title: [{title}]\n")
		recall_files = list(set(recall_files))
		print("kp: {}, recall files: ".format(kp), recall_files)
		all_recall_list.append(recall_files)
		print("============================================")
	
	interset_recalls = {}
	for i, recall_list in enumerate(all_recall_list):
		if i == 0:
			interset_recalls = set(recall_list)
		else:
			interset_recalls = interset_recalls.intersection(set(recall_list))
	#print("interset_recalls: ", interset_recalls)

	#for kp in keyphrases:
	#	jieba.del_word(kp)

	return interset_recalls

def index_content_select_kv(ques_id, question, all_titles, contents_by_title, title_to_filename, keyphrases_data, replace_kv):
	keyphrases = keyphrases_data[ques_id]
	updated_kps = []
	for kp in keyphrases:
		u_kp = kp
		for k,v in replace_kv.items():
			u_kp = u_kp.replace(k,v)
		updated_kps.append(u_kp)
	bm25_retriever, corpus_to_title = build_full_bm25_retriever_kv(updated_kps, all_titles, replace_kv)
	match_titles = keyphrase_docs_retrieve_kv(updated_kps, bm25_retriever, corpus_to_title)
	print(match_titles)

	index_contents = []
	file_list = []
	for match_title in match_titles:
		if match_title in contents_by_title:
			contents = contents_by_title[match_title]
			for content in contents:
				index_contents.append(content)
				
		if match_title in title_to_filename:
			file_list.append(title_to_filename[match_title])

	return index_contents, match_titles, file_list
	

if __name__ == "__main__":

	#compare_submit()
	#sys.exit(0)
	
	jieba.add_word("5G",tag="n",freq=10000)
	jieba.add_word("区块链",tag="n",freq=10000)
	jieba.add_word("大数据",tag="n",freq=10000)
	jieba.add_word("收入",tag="n",freq=100000)
	jieba.add_word("5G+",tag="n",freq=100000)
	
	filter_ques_id = int(sys.argv[1])

	all_titles, all_titles_by_year, contents_by_title, content_to_fileinfo, content_to_doctree_nodes, title_to_filename = pdf_process()
	
	keyphrases_data, report_data = get_keyphrases(all_titles)

	test_data = get_test_data()

	full_hybrid_retriever = build_full_retriever(all_titles, contents_by_title, False)

	final_answers = rag_flow(test_data, keyphrases_data, report_data, all_titles, all_titles_by_year, contents_by_title, content_to_fileinfo, content_to_doctree_nodes, title_to_filename, full_hybrid_retriever, filter_ques_id)

	to_submit(final_answers)
