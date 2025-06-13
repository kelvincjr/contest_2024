from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import os
import pandas as pd
import re
import yaml
import sys
import requests
import json
from fuzzywuzzy import process,fuzz
from trie_lib import trie_lib
from langconv import Converter
from collections import defaultdict

current_dir = os.path.dirname(os.path.abspath(__file__))
xiamen_road_data_path = os.path.join(current_dir, 'data/siming_district.txt')
amap_pois_data_path = os.path.join(current_dir, 'data/keyword_amap.json')
train_data_path = os.path.join(current_dir, 'data/train_data.csv')
test_data_path = os.path.join(current_dir, 'data/test_data.csv')

def get_xiamen_road_list():
	pattern1 = "([^\\d\\-\\()]*)?([\\d\\-]*)?\\w*?"
	p1 = re.compile(pattern1)
	road_dict = {}
	road_res = []
	with open(xiamen_road_data_path, 'r', encoding='utf-8') as f:
		result = yaml.load(f.read(), Loader=yaml.FullLoader)
		print(result['福建省']['厦门市']['思明区'].keys())
		for key in result['福建省']['厦门市']['思明区'].keys():
			#if key != 361001:
				#continue
			loc_list = result['福建省']['厦门市']['思明区'][key]
			for loc in loc_list:
				if loc == "洪山柄" or loc == "文兴社" or loc == "洪文村":
					continue
				it = p1.finditer(loc)
				match_flag = False
				for match in it: 
					match_flag = True
					road_res.append(match.group(1))
					break
				if not match_flag:
					road_res.append(loc)

		print("number of roads: ", len(road_res))
		print(road_res)
		for road in road_res:
			if road not in road_dict:
				road_dict[road] = road
		print("len of road_dict: ", len(road_dict))
	return road_dict, road_res

def get_amap_pois():
	data = []
	with open(amap_pois_data_path, "r", encoding="utf-8") as f:
		lines = f.readlines()
		for line in lines:
			item = json.loads(line.strip())
			data.append(item)
	return data

def get_train_data():
	df = pd.read_csv(train_data_path, header=0)
	data = []
	data_count = 0
	for i in range(len(df)):
		u_address = df.iloc[i].at['非标地址']
		s_address = df.iloc[i].at['对应标准地址']
		data.append((u_address, s_address))
		data_count += 1
	return data

def get_test_data():
    df = pd.read_csv(test_data_path, header=0)
    data = []
    data_count = 0
    for i in range(len(df)):
        id_ = df.iloc[i].at['id']
        u_address = df.iloc[i].at['N_standard_address']
        o_u_address = u_address
        u_address = data_preprocess(u_address)
        data.append((id_,o_u_address,u_address))
        data_count += 1
    return data

def data_preprocess(item_str):
	item_str = item_str.replace(" ","").replace("\\/", "")
	item_str = Converter('zh-hans').convert(item_str)

	pattern1 = "([\\d\\-]*)?o([\\d\\-]*)?"
	p1 = re.compile(pattern1)
	it = p1.finditer(item_str)
	for match in it:
		match_part = match.group(0)
		item_str = item_str.replace(match_part, match_part.replace("o", "0"))

	pattern2 = "([\\d\\-]*)?之([\\d\\-]*)?"
	p2 = re.compile(pattern2)
	it = p2.finditer(item_str)
	for match in it:
		match_part = match.group(0)
		item_str = item_str.replace(match_part, match_part.replace("之", "-"))

	item_str = fullwidth_to_halfwidth_numbers_only(item_str)
	return item_str

def fullwidth_to_halfwidth_numbers_only(text):
    result = []
    for char in text:
        code = ord(char)
        # 处理全角数字和全角小数点
        if (65296 <= code <= 65305) or (code == 65294):  # 全角数字和全角小数点的Unicode范围
            code -= 65248
            result.append(chr(code))
        # 处理半角数字和半角小数点
        elif (48 <= code <= 57) or (code == 46):
            result.append(char)
        else:
        	result.append(char)

    return ''.join(result)

def get_ner_processor():
	task = Tasks.token_classification
	model = 'iic/mgeo_geographic_elements_tagging_chinese_base'
	#inputs = '浙江省杭州市余杭区阿里巴巴西溪园区'
	pipeline_ins = pipeline(task=task, model=model)
	return pipeline_ins

def road_search(u_address, road_list):
	rs_trie = trie_lib()
	rs_trie.build_trie(road_list)
	trie_match_rs = rs_trie.trie_search(u_address)
	print("===== road_search, u_address {}, trie_match_rs: ".format(u_address), trie_match_rs)
	road = ""
	if len(trie_match_rs) > 0 and trie_match_rs[0] in road_list:
		road = trie_match_rs[0]
	return road

def road_verify(road_entity, all_roads, train_roads, test_roads):
	road_found = ""
	road_choices = []
	for road in all_roads:
		road_choices.append(road)

	model_match_roads = process.extract(road_entity, road_choices, limit=2)
	for match_road in model_match_roads:
		if (match_road[0] in train_roads or match_road[0] in test_roads) and match_road[1] >= 75:
			 road_found = match_road[0]
			 break
	if road_found == "" and model_match_roads[0][1] >= 75:
		road_found = model_match_roads[0][0]

	print("===== road_verify, road_entity {}, road_found: {}, model_match_roads: ".format(road_entity, road_found), model_match_roads)
	return road_found

def poi_search_simple(u_address, poi_list):
	poi_trie = trie_lib()
	poi_trie.build_trie(poi_list)
	trie_match_poi = poi_trie.trie_search(u_address)
	print("===== poi_search_simple, u_address {}, trie_match_poi: ".format(u_address), trie_match_poi)
	poi = ""
	if len(trie_match_poi) > 0 and trie_match_poi[0] in poi_list:
		poi = trie_match_poi[0]
	else:
		model_match_pois = process.extract(u_address, poi_list, limit=2)
		print("===== poi_search_simple, u_address {}, model_match_pois: ".format(u_address), model_match_pois)
		if model_match_pois[0][1] >= 75:
			poi = model_match_pois[0][0]
	return poi

def poi_search(search_poi, road_list, train_pois):
	#print("poi_search, poi: {}".format(search_poi))
	amap_pois = get_amap_pois()
	road_map = {}
	poi_choices = []
	for poi in amap_pois:
		poi_name = poi["name"]
		poi_address = poi["address"]
		poi_choices.append(poi_name)
		road_map[poi_name] = poi_address

	rs_trie = trie_lib()
	rs_trie.build_trie(road_list)

	road = ""
	match_poi = ""
	match_poi_address = ""
	model_match_pois = process.extract(search_poi, poi_choices, limit=2)
	print("===== poi_search, model_match_pois: ", model_match_pois)
	for poi_item in model_match_pois:
		#print("poi_item[0] {}".format(poi_item[0]))
		if poi_item[0] in train_pois:
			#print("======= found =========")
			match_poi = poi_item[0]

	if match_poi == "" and model_match_pois[0][1] >= 60:
		match_poi = model_match_pois[0][0]
		
	match_house_no = ""
	if match_poi in road_map:
		poi_address = road_map[match_poi]
		match_poi_address = poi_address
		trie_match_rs = rs_trie.trie_search(poi_address)
		pattern_houseno = "([\\d\\-]*)?号"
		p_houseno = re.compile(pattern_houseno)
		it = p_houseno.finditer(poi_address)
		for match in it:
			match_house_no = match.group(0)
		print("===== poi_search, poi_address: {}, trie_match_rs: {}, match_house_no: {}".format(poi_address, trie_match_rs, match_house_no))
		if len(trie_match_rs) > 0 and trie_match_rs[0] in road_list:
			road = trie_match_rs[0]

	return road, match_poi, match_poi_address, match_house_no

def find_poi(data_with_label, road_list):
	pois_trie = trie_lib()
	rs_trie = trie_lib()
	amap_pois = get_amap_pois()
	train_pois = []

	poi_choices = []
	for poi in amap_pois:
		poi_choices.append(poi["name"])
	pois_trie.build_trie(poi_choices)
	rs_trie.build_trie(road_list)

	pipeline_ins = get_ner_processor()

	count = 0
	for item in data_with_label:
		u_address = item[0]
		u_label = item[2]
		s_label = item[3]
		find_poi = ""
		
		trie_match_pois = pois_trie.trie_search(u_address)
		print("u_address: {}, trie_match_pois: {}".format(u_address, trie_match_pois))

		trie_match_rs = rs_trie.trie_search(u_address)
		print("u_address: {}, trie_match_rs: {}".format(u_address, trie_match_rs))

		ner_outputs = pipeline_ins(input=u_address)
		print("ner_outputs: ", ner_outputs)
		model_match_pois = []
		max_sim = 0
		max_sim_pois = ("","")
		ner_pois = []
		for ner in ner_outputs['output']:
			if ner['type'] == "poi" or ner['type'] == "road":
				word = ner['span']
				ner_pois.append(word)
				model_match_pois = process.extract(word, poi_choices, limit=1)
				if model_match_pois[0][1] > 80 and model_match_pois[0][1] > max_sim:
					max_sim = model_match_pois[0][1]
					max_sim_pois = (word, model_match_pois[0][0])
					#print("u_address: {}, span: {}, model_match_pois: {}".format(u_address, word, model_match_pois))
		print("u_address: {}, max_sim_pois: {}, max_sim: {}, max_sim_span: {}".format(u_address, max_sim_pois[1], max_sim, max_sim_pois[0]))
		ref_poi = ""
		if len(trie_match_pois) > 0:
			find_poi = trie_match_pois[0]
			ref_poi = trie_match_pois[0]
		elif max_sim > 0:
			find_poi = max_sim_pois[0]
			ref_poi = max_sim_pois[1]

		if find_poi != "":
			index = u_address.find(find_poi) 
			if index != -1:
				u_label["poi"] = (find_poi, index, index+len(find_poi), ref_poi)
				train_pois.append((find_poi, s_label[1]))
		elif "rd_st" not in u_label and len(ner_pois) > 0:
			max_len_ner = ner_pois[0]
			max_len = 0
			max_rdst_sim = 0
			max_rdst_sim_ner = ""
			for ner in ner_pois:
				if len(ner) > max_len:
					max_len = len(ner)
					max_len_ner = ner

				sim = fuzz.ratio(ner, s_label[1])
				if sim > 0.8 and sim > max_rdst_sim:
					max_rdst_sim = sim
					max_rdst_sim_ner = ner

			if max_rdst_sim > 0:
				find_rdst = max_rdst_sim_ner
				u_label["rd_st"] = (find_rdst, index, index+len(find_rdst))
			else:
				find_poi = max_len_ner
				index = u_address.find(find_poi) 
				if index != -1:
					u_label["poi"] = (find_poi, index, index+len(find_poi), ref_poi)
					train_pois.append((find_poi, s_label[1]))


		count += 1
		if count % 50 == 0:
			print("count: {} process find_poi".format(count))

	return data_with_label, train_pois

def process_train_data(data, road_dict):
	pattern = "^福建省厦门市(.*?)区([^\\d\\-]*)?([\\d\\-]*)?号(([\\w\\-]*)?室)?$"
	p = re.compile(pattern)
	pattern_match_count = 0
	data_with_label = []
	train_rs_items = []
	for item in data:
		u_address = item[0]
		s_address = item[1]
		u_address = data_preprocess(u_address)
		u_label = {}
		s_label = []
		it = p.finditer(s_address)
		for match in it:
			pattern_match_count += 1 
			district = match.group(1)
			road_street = match.group(2)
			house = match.group(3)
			room = match.group(4)
			s_label = (district, road_street, house, room)
			#print(match.group(), match.group(0), match.group(1), match.group(2), match.group(3), match.group(4)) 
			if road_street is not None:
				index = u_address.find(road_street) 
				if index != -1:
					u_label["rd_st"] = (road_street, index, index+len(road_street))

			if house is not None:
				index = u_address.find(house+"号") 
				if index != -1:
					u_label["house_no"] = (house+"号", index, index+len(house+"号"))

				index1 = u_address.find(house)
				if index == -1 and index1 != -1:
					u_label["house_no"] = (house, index1, index1+len(house))

				if index == -1 and index1 == -1:
					if house.find("-") != -1:
						house_parts = house.split("-")
						index2 = u_address.find(house_parts[0])
						if index2 != -1:
							u_label["house_no"] = (house_parts[0], index2, index2+len(house_parts[0]))

			if room is not None:
				index = u_address.find(room)
				if index != -1:
					u_label["room_no"] = (room, index, index+len(room))
				
				if room.endswith("室"):
					index1 = u_address.find(room[:-1])
					if index == -1 and index1 != -1:
						u_label["room_no"] = (room[:-1], index1, index1+len(room[:-1]))

			#print("u_address: {}, u_label: {}".format(u_address, u_label))
			data_with_label.append((u_address, s_address, u_label, s_label))
			train_rs_items.append(road_street)

	train_rs_items = list(set(train_rs_items))

	return data_with_label,train_rs_items

def train_data_refine(xiamen_pois):
	total_count = 0
	missing_count = 0
	json_data = []
	lines = []
	poi_list = []
	pois_trie = trie_lib()
	pois_trie.build_trie(xiamen_pois)
	with open("data_with_label.json", "r+", encoding="utf-8") as f:
		lines = f.readlines()
	line_count = 0
	for line in lines:
		json_line = json.loads(line.strip())
		u_address = json_line["u_address"]
		s_address = json_line["s_address"]
		u_label = json_line["u_label"]
		s_label = json_line["s_label"]
	
		total_count += 1
		id_ = line_count + 1

		if "rd_st" not in u_label and ("店上东里" == s_label[1] or "店上西里" == s_label[1]) and u_address.find("店上东西里") != -1:
			c_road = "店上东西里"
			index = u_address.find(c_road)
			u_label["rd_st"] = (c_road, index, index+len(c_road))

		if "rd_st" in u_label and "poi" in u_label and u_label["rd_st"][0] == u_label["poi"][0]:
			u_label.pop("poi")

		if "rd_st" in u_label and u_label["rd_st"][0] == "前埔北一里":
			u_label.pop("rd_st")

		if "poi" not in u_label and u_address.find("前埔北一里") != -1:
			c_poi = "前埔北一里"
			index = u_address.find(c_poi)
			u_label["poi"] = [c_poi, index, index+len(c_poi), "前埔北区一里"]

		if "poi" in u_label:
			if len(u_label["poi"][0]) == 2 or u_label["poi"][0] == '前埔北':
				u_label.pop("poi")
		
		if "house_no" not in u_label and s_label[2] is not None:
			print("id: {}, u_address: {}, u_label: {}, s_address: {}, s_label: {}".format(id_, u_address, u_label, s_address, s_label))
			missing_count += 1
		elif "room_no" not in u_label and s_label[3] is not None:
			print("id: {}, u_address: {}, u_label: {}, s_address: {}, s_label: {}".format(id_, u_address, u_label, s_address, s_label))
			missing_count += 1
		
		if "rd_st" not in u_label:  #and "poi" not in u_label:
			#print("id: {}, u_address: {}, u_label: {}, s_address: {}, s_label: {}".format(id_, u_address, u_label, s_address, s_label))
			missing_count += 1
			if s_label[1] is not None and s_label[1].endswith("路") and u_address.find(s_label[1][:-1]) != -1:
				c_road = s_label[1][:-1]
				index = u_address.find(c_road)
				u_label["rd_st"] = (c_road, index, index+len(c_road))
			else:
				if "poi" not in u_label:
					print("===== try to find pois, poi not in u_label =====")
					trie_match_pois = pois_trie.trie_search(u_address)
					print("id: {}, u_address: {}, trie_match_pois: {}".format(id_, u_address, trie_match_pois))
					if len(trie_match_pois) > 0:
						index = u_address.find(trie_match_pois[0])
						u_label["poi"] = [trie_match_pois[0], index, index + len(trie_match_pois[0]), "", trie_match_pois[0]]
				else:
					print("===== try to find pois, poi in u_label =====")
					trie_match_pois = pois_trie.trie_search(u_label["poi"][0])
					print("id: {}, u_label_poi: {}, trie_match_pois: {}".format(id_, u_label["poi"][0], trie_match_pois))
					if len(trie_match_pois) > 0:
						u_label["poi"].append(trie_match_pois[0])
					else:
						model_match_pois = process.extract(u_label["poi"][0], xiamen_pois, limit=1)
						print("id: {}, u_label_poi: {}, model_match_pois: {}".format(id_, u_label["poi"][0], model_match_pois))
						if model_match_pois[0][1] > 80:
							u_label["poi"].append(model_match_pois[0][0])
						else:
							u_label["poi"].append("")

			if "rd_st" in u_label and "poi" in u_label and u_label["rd_st"][0] == u_label["poi"][0]:
				u_label.pop("poi")

			print("id: {}, u_address: {}, u_label: {}, s_address: {}, s_label: {}".format(id_, u_address, u_label, s_address, s_label))

		if "poi" in u_label:
			poi_list.append(u_label["poi"][0])

		#if "poi" in u_label and u_label["poi"][0] in sorted_train_pois and len(u_label["poi"][0]) < 3:
		#	print("u_address: {}, u_label: {}, s_address: {}, s_label: {}".format(u_address, u_label, s_address, s_label))
		
		json_item = {"u_address": u_address, "s_address":s_address, "u_label":u_label, "s_label":s_label}
		json_data.append(json_item)

		line_count += 1
		#if line_count % 1000 == 0:
		#	break
	
	print("===== gen mgeo train dataset =====")
	line_count = 0
	train_data = []
	json_data_new = []
	train_poi_trie = trie_lib()
	train_poi_trie.build_trie(poi_list)
	for json_item in json_data:
		u_address = json_item["u_address"]
		s_address = json_item["s_address"]
		u_label = json_item["u_label"]
		s_label = json_item["s_label"]
		id_ = line_count + 1
		'''
		if "house_no" in u_label:
			if u_label["house_no"][0].find("-") != -1:
				print("id_: {}, u_address: {}, u_label: {}, s_label: {}".format(id_, u_address, u_label, s_label))
		line_count += 1
		continue		
		'''
		match_train_pois = train_poi_trie.trie_search(u_address)
		if len(match_train_pois) > 0 and "poi" not in u_label:
			if "rd_st" in u_label and u_label["rd_st"][0] == match_train_pois[0]:
				pass
			else:
				print("id_: {}, u_address: {}, u_label: {}, s_label: {}, match_train_pois: {}".format(id_, u_address, u_label, s_label, match_train_pois))
				index = u_address.find(match_train_pois[0])
				u_label["poi"] = [match_train_pois[0], index, index + len(match_train_pois[0]), "", match_train_pois[0]]
		#line_count += 1
		#continue
		json_data_new.append(json_item)

		#if "rd_st" in u_label and u_label["rd_st"][0] != s_label[1]:
		#	print("id_: {}, u_address: {}, u_label: {}, s_label: {}".format(id_, u_address, u_label, s_label))
		#line_count += 1
		#continue
		train_item = {} 
		train_item["tokens"] = list(u_address)
		ner_tags = ['O']*len(u_address)
		for key,value in u_label.items():
			if key == "house_no":
				start_idx = value[1]
				end_idx = value[2]
				if ner_tags[start_idx] != "O":
					print("ner tag error, u_address: {}, u_label: {}".format(u_address, u_label))
				ner_tags[start_idx] = "B-houseno"
				i = start_idx + 1
				while i < end_idx - 1:
					if ner_tags[i] != "O":
						print("ner tag error, u_address: {}, u_label: {}".format(u_address, u_label))
					ner_tags[i] = "I-houseno"
					i += 1
				if start_idx != (end_idx - 1):
					if ner_tags[end_idx - 1] != "O":
						print("ner tag error, u_address: {}, u_label: {}".format(u_address, u_label))
					ner_tags[end_idx - 1] = "E-houseno"

			elif key == "room_no":
				start_idx = value[1]
				end_idx = value[2]
				if ner_tags[start_idx] != "O":
					print("ner tag error, u_address: {}, u_label: {}".format(u_address, u_label))
				ner_tags[start_idx] = "B-floorno"
				i = start_idx + 1
				while i < end_idx - 1:
					if ner_tags[i] != "O":
						print("ner tag error, u_address: {}, u_label: {}".format(u_address, u_label))
					ner_tags[i] = "I-floorno"
					i += 1
				if start_idx != (end_idx - 1):
					if ner_tags[end_idx - 1] != "O":
						print("ner tag error, u_address: {}, u_label: {}".format(u_address, u_label))
					ner_tags[end_idx - 1] = "E-floorno"
			elif key == "rd_st":
				start_idx = value[1]
				end_idx = value[2]
				if ner_tags[start_idx] != "O":
					print("ner tag error, u_address: {}, u_label: {}".format(u_address, u_label))
				ner_tags[start_idx] = "B-road"
				i = start_idx + 1
				while i < end_idx - 1:
					if ner_tags[i] != "O":
						print("ner tag error, u_address: {}, u_label: {}".format(u_address, u_label))
					ner_tags[i] = "I-road"
					i += 1
				if start_idx != (end_idx - 1):
					if ner_tags[end_idx - 1] != "O":
						print("ner tag error, u_address: {}, u_label: {}".format(u_address, u_label))
					ner_tags[end_idx - 1] = "E-road"
			elif key == "poi":
				start_idx = value[1]
				end_idx = value[2]
				if ner_tags[start_idx] != "O":
					print("ner tag error, u_address: {}, u_label: {}".format(u_address, u_label))
				ner_tags[start_idx] = "B-poi"
				i = start_idx + 1
				while i < end_idx - 1:
					if ner_tags[i] != "O":
						print("ner tag error, u_address: {}, u_label: {}".format(u_address, u_label))
					ner_tags[i] = "I-poi"
					i += 1
				if start_idx != (end_idx - 1):
					if ner_tags[end_idx - 1] != "O":
						print("ner tag error, u_address: {}, u_label: {}".format(u_address, u_label))
					ner_tags[end_idx - 1] = "E-poi"

		train_item["ner_tags"] = ner_tags
		train_data.append(train_item)
		#print("train_item: ", train_item)

		#if "poi" in u_label:
		#	if u_label["poi"][0] not in poi_list:
		#		print("id: {}, u_address: {}, u_label: {}, s_address: {}, s_label: {}".format(id_, u_address, u_label, s_address, s_label))
		line_count += 1

	with open("./data/data_with_label_new.json", "w+", encoding="utf-8") as f:
		for json_item in json_data_new:
			f.write(json.dumps(json_item, ensure_ascii=False) + "\n")
	'''
	label_counts = defaultdict(int)
	for train_item in train_data:
		for tag in train_item["ner_tags"]:
			label_counts[tag] += 1

	label_counts = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
	print("label_counts: ")
	for label, label_count in label_counts:
		print("{}, {}".format(label, label_count))
	
	print("total_count: {}, missing_count: {}".format(total_count, missing_count))
	poi_list = list(set(poi_list))
	print("poi_list: ", poi_list)
	
	with open("./data/mgeo_train_data.json", "w", encoding="utf-8") as f:
		for train_item in train_data:
			f.write(json.dumps(train_item, ensure_ascii=False) + "\n")
	
	messages = []
	print("===== gen llm train dataset =====")
	line_count = 0
	train_data = []
	for json_item in json_data_new:
		u_address = json_item["u_address"]
		s_address = json_item["s_address"]
		u_label = json_item["u_label"]
		s_label = json_item["s_label"]
		id_ = line_count + 1
		entity_sentence = ""
		for key, value in u_label.items():
			entity_text = value[0]
			entity_label = ""
			if key == "house_no":
				entity_label = "楼号"
			elif key == "room_no":
				entity_label = "房间号"
			elif key == "rd_st":
				entity_label = "道路"
			elif key == "poi":
				entity_label = "兴趣点"
			entity_sentence += f"""{{"entity_text": "{entity_text}", "entity_label": "{entity_label}"}}"""

		if entity_sentence == "":
			entity_sentence = "没有找到任何实体"

		message = {
			"instruction": """你是一个地址实体元素解析领域的专家，你需要从给定的地址文本中提取 兴趣点; 道路; 楼号; 房间号 实体. 以 json 格式输出, 如 {"entity_text": "洪文六里", "entity_label": "道路"} 注意: 1. 输出的每一行都必须是正确的 json 字符串. 2. 找不到任何实体时, 输出"没有找到任何实体". """,
			"input": f"文本:{u_address}",
			"output": entity_sentence,
		}
		messages.append(message)

	print(messages[:2])

	with open("./data/llm_train_data.json", "w", encoding="utf-8") as file:
		for message in messages:
			file.write(json.dumps(message, ensure_ascii=False) + "\n")
	'''
#amap api 
def get_poi(poi_name, page, per_page):
	ak = "83c3c23198738d03b6e210de723e7b4c"
	adcode = "350203"
	#types = "120302"
	#types = "120000|120100|120200|120201|120202|120203|120300|120301|120302|120303|120304"
	types = "120000|120100|120200|120201|120202|120203|120300|120301|120302|120303|120304|100000|100100|100101|100102|100103|100104|100105|100200|100201|090000|090100|090101|090102|090200|090201|090202|090203|090204|090205|090206|090207|090208|090209|090210|090211|090300|090400|090500|090600|090601|090602|090700|090701|090702|090800|170000|170100|170200|170201|170202|170203|170204|170205|170206|170207|170208|170209|170300|170400|170401|170402|170403|170404|170405|170406|170407|170408"   
	keyword = poi_name #"加州建材广场" #"圣华佗"  "都市新巢小区"  "前埔南小区"  #keywords={}&
	#url = "https://restapi.amap.com/v3/place/text?keywords={}&types={}&city={}&citylimit=true&offset={}&page={}&key={}&extensions=all".format(keyword, types, adcode, per_page, page, ak)
	url = "https://restapi.amap.com/v3/place/text?keywords={}&city={}&citylimit=true&offset={}&page={}&key={}".format(keyword, adcode, per_page, page, ak)
	header={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.77 Safari/537.36'} #构造请求头
	response=requests.get(url,headers=header) #发出请求
	answer=response.json() #json化
	#print(answer)
	resp_count = int(answer["count"])
	page_data_count = 0
	resp = []
	if resp_count > 0:
		pois = answer["pois"]
		page_data_count = len(pois)
		for i in range(len(pois)):
			name = pois[i]["name"]
			address = pois[i]["address"]
			type = pois[i]["type"]
			typecode = pois[i]["typecode"]
			print("keyword: {}, name: {}, address: {}, type: {}, typecode: {}".format(keyword, name, address, type, typecode))
			#print(pois[i])
			resp.append((keyword, name, address, type, typecode))

	return resp_count, page_data_count, resp

def get_poi_poly(poi_name, page, per_page):
	ak = "83c3c23198738d03b6e210de723e7b4c"
	adcode = "350203"
	#types = "120302"
	#types = "120000|120100|120200|120201|120202|120203|120300|120301|120302|120303|120304"
	types = "120000|120100|120200|120201|120202|120203|120300|120301|120302|120303|120304|100000|100100|100101|100102|100103|100104|100105|100200|100201|090000|090100|090101|090102|090200|090201|090202|090203|090204|090205|090206|090207|090208|090209|090210|090211|090300|090400|090500|090600|090601|090602|090700|090701|090702|090800|170000|170100|170200|170201|170202|170203|170204|170205|170206|170207|170208|170209|170300|170400|170401|170402|170403|170404|170405|170406|170407|170408"   
	keyword = poi_name #"加州建材广场" #"圣华佗"  "都市新巢小区"  "前埔南小区"  #keywords={}&
	#url = "https://restapi.amap.com/v3/place/text?keywords={}&types={}&city={}&citylimit=true&offset={}&page={}&key={}&extensions=all".format(keyword, types, adcode, per_page, page, ak)
	#url = "https://restapi.amap.com/v3/place/text?keywords={}&city={}&citylimit=true&offset={}&page={}&key={}".format(keyword, adcode, per_page, page, ak)
	url = "https://restapi.amap.com/v3/place/polygon?polygon=118.151525,24.479603;118.160977,24.473891&types={}&key={}&extensions=all&offset={}&page={}".format(types, ak, per_page, page)
	header={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.77 Safari/537.36'} #构造请求头
	response=requests.get(url,headers=header) #发出请求
	answer=response.json() #json化
	print(answer)
	resp_count = int(answer["count"])
	page_data_count = 0
	resp = []
	if resp_count > 0:
		pois = answer["pois"]
		page_data_count = len(pois)
		for i in range(len(pois)):
			name = pois[i]["name"]
			address = pois[i]["address"]
			type = pois[i]["type"]
			typecode = pois[i]["typecode"]
			print("keyword: {}, name: {}, address: {}, type: {}, typecode: {}".format(keyword, name, address, type, typecode))
			#print(pois[i])
			resp.append((keyword, name, address, type, typecode))

	return resp_count, page_data_count, resp

def amap_test():
	page = 1
	resp_count, page_data_count, page_resp = get_poi("圣华佗", page, 50)
	print("page: {}, resp_count: {}, page_data_count: {}".format(page, resp_count, page_data_count))
	sys.exit(0)

def dump_amap_resp(resp):
	with open("./data/keyword_amap_test_1.json", "w+", encoding="utf-8") as f:
		for item in resp:
			data = {}
			data['keyword'] = item[0]
			data['name'] = item[1]
			data['address'] = item[2]
			data['type'] = item[3]
			data['typecode'] = item[4]
			json_str = json.dumps(data, ensure_ascii=False)
			f.write(json_str + "\n")

def amap_poi_crawler(road_street_items):
	resp = []
	total_item_count = 0
	for item in road_street_items:
		poi_name = item
		page = 1
		page_count = 50
		while True:
			resp_count, page_data_count, page_resp = get_poi_poly(poi_name, page, page_count)
			print("===== poi_name: {}, page: {}, resp_count: {}, page_data_count: {} =====".format(poi_name, page, resp_count, page_data_count))
			resp.extend(page_resp)
			if page_data_count < page_count:
				break
			if resp_count == 0:
				break
			page += 1
		total_item_count += 1
		if total_item_count % 10 == 0:
			print("total_item_count: {} processed.".format(total_item_count))

	dump_amap_resp(resp)


if __name__ == '__main__':	
	#高德地图数据搜集
	add_roads = ['莲前东路']
	amap_poi_crawler(add_roads)
	sys.exit(0)

	all_roads = []
	road_dict, xiamen_roads = get_xiamen_road_list()
	print("xiamen_roads len: {}".format(len(xiamen_roads)))

	all_roads.extend(xiamen_roads)
	
	data = get_train_data()
	data_with_label,train_roads = process_train_data(data, road_dict)
	print("train_roads len: {}".format(len(train_roads)))

	all_roads.extend(list(train_roads))
	all_roads = list(set(all_roads))
	print("all_roads len after train: {}".format(len(all_roads)))

	#生成训练数据
	data_with_label,train_pois = find_poi(data_with_label, road_dict)