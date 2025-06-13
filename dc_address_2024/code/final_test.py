from ner_arknlp import predict_mode, predict_example
import os
import pandas as pd
import re
import yaml
import sys
import requests
import json
import sys
import time
#sys.path.insert(0, './dep_lib')
#sys.path.insert(0, './dep_lib/jpype')
from fuzzywuzzy import process,fuzz
from trie_lib import trie_lib
from langconv import Converter
from collections import defaultdict

current_dir = os.path.dirname(os.path.abspath(__file__))
#xiamen_road_data_path = os.path.join(current_dir, 'data/厦门市/思明区.yml')
xiamen_road_data_path = os.path.join(current_dir, 'data/siming_district.txt')
amap_pois_data_path = os.path.join(current_dir, 'data/keyword_amap.json')
#xiamen_pois_data_path = os.path.join(current_dir, 'data/XM_housing_data.csv')
#train_data_path = os.path.join(current_dir, 'data/初赛训练集.csv')
#test_data_path = os.path.join(current_dir, 'data/初赛测试集.csv')
train_data_path = os.path.join(current_dir, 'data/train_data.csv')
test_data_path = os.path.join(current_dir, 'data/test_data.csv')
test_data_with_entities_path = os.path.join(current_dir, 'data/test_data_with_entities.json')
train_data_with_label_path = os.path.join(current_dir, 'data/train_data_with_label_new.json')
round1_submit_data_path = os.path.join(current_dir, 'data/submission.csv')
#round1_submit_data_path = os.path.join(current_dir, 'data/train_submission.csv')

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
'''
def get_xiamen_pois():
	df = pd.read_csv(xiamen_pois_data_path, header=0)
	data = []
	data_count = 0
	pattern = "(.*)(([\\d\\-]{1,})号)([^\\(\\)]*)?"
	p = re.compile(pattern)
	for i in range(len(df)):
		house_loc = df.iloc[i].at['houseLoc']
		#print("house_loc: {}".format(house_loc))
		it = p.finditer(house_loc)
		for match in it:
			poi = match.group(4)
			if poi is not None and poi != "小区" and poi.find("-") == -1:
				data.append(poi)
		data_count += 1
	return data
'''

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

def ner_predict(df_test):
	model, tokenizer, ner_train_dataset, ner_dev_dataset = predict_mode()
	start_time = time.time()
	last_cur_time = start_time
	count = 0
	json_data = []
	for i in range(len(df_test)):
		id_ = df_test.iloc[i].at['id']
		u_address = df_test.iloc[i].at['N_standard_address']
		o_u_address = u_address
		u_address = data_preprocess(u_address)
		example = u_address
		predicts = predict_example(model, tokenizer, ner_train_dataset, example)
		entities = []
		for _predict in predicts:
			entity = {"start_idx":int(_predict["start_idx"]), "end_idx":int(_predict["end_idx"]), "entity":_predict["entity"], "type":_predict["type"]}
			entities.append(entity)
		json_item = {"id":int(id_), "o_u_address": o_u_address, "u_address":u_address, "entities":entities}
		json_data.append(json_item)
		count += 1
		if count % 100 == 0:
			cur_time = time.time()
			print("predict {} data done, time used: {}".format(count, (cur_time-last_cur_time)))
			last_cur_time = cur_time
	end_time = time.time()
	print("total time used: {}".format((end_time-start_time)))

	with open(test_data_with_entities_path, "w+", encoding="utf-8") as f:
		for json_item in json_data:
			f.write(json.dumps(json_item, ensure_ascii=False) + "\n")


def road_search(u_address, all_roads, train_roads, test_roads, short_roads):
	rs_trie = trie_lib()
	rs_trie.build_trie(all_roads)
	trie_match_rs = rs_trie.trie_search(u_address)
	#print("===== road_search, u_address {}, trie_match_rs: ".format(u_address), trie_match_rs)
	road = ""
	if len(trie_match_rs) > 0 and trie_match_rs[0] in all_roads:
		road = trie_match_rs[0]

	if road == "":
		short_rs_trie = trie_lib()
		r_short_roads = []
		for short_road in short_roads:
			r_short_roads.append(short_road + "-")
		short_rs_trie.build_trie(r_short_roads)
		trie_match_short_rs = short_rs_trie.trie_search(u_address)
		#print("===== short road_search, u_address {}, trie_match_short_rs: ".format(u_address), trie_match_short_rs)
		if len(trie_match_short_rs) > 0 and (trie_match_short_rs[0][:-1] + "路") in all_roads:
			road = (trie_match_short_rs[0][:-1]  + "路")
	
	return road

def road_verify(u_address, road_entity, all_roads, train_roads, test_roads, short_roads, prefix_short_roads):
	road_found = ""
	road_choices = []
	for road in all_roads:
		road_choices.append(road)

	if road_entity in short_roads:
		if road_entity in prefix_short_roads:
			f_short_roads = prefix_short_roads[road_entity]
			for f_short_road in f_short_roads:
				if u_address.find(f_short_road) != -1:
					road_found = f_short_road + "路"
		if road_found == "":
			road_found = road_entity + "路"
		#print("===== road_verify in short_roads, road_entity {}, road_found: {}".format(road_entity, road_found))

	if road_found == "":
		model_match_roads = process.extract(road_entity, road_choices, limit=2)
		for match_road in model_match_roads:
			if (match_road[0] in train_roads or match_road[0] in test_roads) and match_road[1] >= 75:
				 road_found = match_road[0]
				 break
		if road_found == "" and model_match_roads[0][1] >= 75:
			road_found = model_match_roads[0][0]

		#print("===== road_verify, road_entity {}, road_found: {}, model_match_roads: ".format(road_entity, road_found), model_match_roads)
	return road_found

def poi_search_simple(u_address, poi_list):
	poi_trie = trie_lib()
	poi_trie.build_trie(poi_list)
	trie_match_poi = poi_trie.trie_search(u_address)
	#print("===== poi_search_simple, u_address {}, trie_match_poi: ".format(u_address), trie_match_poi)
	poi = ""
	if len(trie_match_poi) > 0 and trie_match_poi[0] in poi_list:
		poi = trie_match_poi[0]
	else:
		model_match_pois = process.extract(u_address, poi_list, limit=2)
		#print("===== poi_search_simple, u_address {}, model_match_pois: ".format(u_address), model_match_pois)
		if model_match_pois[0][1] >= 75:
			poi = model_match_pois[0][0]
	return poi

def poi_search(search_poi, road_list, train_pois, amap_pois):
	#print("poi_search, poi: {}".format(search_poi))
	#amap_pois = get_amap_pois()
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
	#print("===== poi_search, model_match_pois: ", model_match_pois)
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
		#print("===== poi_search, poi_address: {}, trie_match_rs: {}, match_house_no: {}".format(poi_address, trie_match_rs, match_house_no))
		if len(trie_match_rs) > 0 and trie_match_rs[0] in road_list:
			road = trie_match_rs[0]

	return road, match_poi, match_poi_address, match_house_no

def load_train_pois():
	messages = []
	train_pois = {}
	train_pois_with_count = {}
	train_pois_with_data = {}
	with open(train_data_with_label_path, "r+", encoding="utf-8") as f:
		lines = f.readlines()
		line_count = 0
		for line in lines:
			json_line = json.loads(line.strip())
			u_address = json_line["u_address"]
			s_address = json_line["s_address"]
			u_label = json_line["u_label"]
			s_label = json_line["s_label"]

			line_count += 1
			if "poi" in u_label and "rd_st" in u_label and (u_label["poi"][0] == u_label["rd_st"][0]):
				continue

			if "poi" in u_label:
				train_pois[u_label["poi"][0]] = s_label[1]
				if u_label["poi"][3] != "":
					train_pois[u_label["poi"][3]] = s_label[1]

				if u_label["poi"][0] not in train_pois_with_count:
					train_pois_with_count[u_label["poi"][0]] = defaultdict(int)
				if u_label["poi"][0] not in train_pois_with_data:
					train_pois_with_data[u_label["poi"][0]] = dict()
				if s_label[1] not in train_pois_with_data[u_label["poi"][0]]:
					train_pois_with_data[u_label["poi"][0]][s_label[1]] = list()
				train_pois_with_count[u_label["poi"][0]][s_label[1]] += 1
				train_pois_with_data[u_label["poi"][0]][s_label[1]].append((u_address, s_address))
				if u_label["poi"][3] != "":
					if u_label["poi"][3] not in train_pois_with_count:
						train_pois_with_count[u_label["poi"][3]] = defaultdict(int)
					#	train_pois_with_data[u_label["poi"][3]] = dict()
					#if s_label[1] not in train_pois_with_data[u_label["poi"][3]]:
					#	train_pois_with_data[u_label["poi"][3]][s_label[1]] = list()
					train_pois_with_count[u_label["poi"][3]][s_label[1]] += 1
					#train_pois_with_data[u_label["poi"][3]][s_label[1]].append(s_label[2])
			
			#if line_count % 10 == 0:
				#break
	
	sorted_train_pois = {}
	for poi, count_dict in train_pois_with_count.items():	
		sorted_count_dict = sorted(count_dict.items(), key=lambda x: x[1], reverse=True)
		sorted_train_pois[poi] = sorted_count_dict[0][0]
	return train_pois, train_pois_with_count, train_pois_with_data, sorted_train_pois

def load_test_pois(road_res, train_pois):
	messages = []
	test_pois = {}
	test_pois_with_count = {}
	test_pois_with_data = {}
	all_lines = []
	with open(test_data_with_entities_path, "r+", encoding="utf-8") as f:
		lines = f.readlines()
		all_lines.extend(lines)

	poi_list = []
	test_roads = []
	for line in all_lines:
		json_line = json.loads(line.strip())
		id_ = json_line["id"]
		u_address = json_line["u_address"]
		entities = json_line["entities"]
		for entity in entities:
			if entity["type"] == "poi":
				poi = entity["entity"]
				if len(poi) > 2 and poi not in road_res:
					poi_list.append(poi)

			if entity["type"] == "rd_st":
				road = entity["entity"]
				test_roads.append(road)

	test_roads = list(set(test_roads))

	for key in train_pois.keys():
		if len(key) > 2 and key not in road_res:
			poi_list.append(key)

	poi_list = list(set(poi_list))
	#print(poi_list)
		
	line_count = 0
	for line in all_lines:
		json_line = json.loads(line.strip())
		id_ = json_line["id"]
		u_address = json_line["u_address"]
		entities = json_line["entities"]
		road_found = False
		road = ""
		line_count += 1
		for entity in entities:
			if entity["type"] == "rd_st":
				road = entity["entity"]
				if road in road_res:
					road_found = True
				#else:
				#	verified_road = road_verify(road, road_res)
				#	if verified_road != "":
				#		road_found = True
				#		road = verified_road

		'''
		if not road_found:
			road = road_search(u_address, road_res)
			if road != "":
				road_found = True
		'''
		poi_found = False
		poi = ""
		for entity in entities:
			if entity["type"] == "poi":
				poi = entity["entity"]
				if poi != road:
					poi_found = True
					break

		#if not poi_found:
		#	poi = poi_search_simple(u_address, poi_list)
		#	if poi != "":
		#		poi_found = True
				
		if road_found and poi_found:
			test_pois[poi] = road
			if poi == road:
				continue
			if poi not in test_pois_with_count:
				test_pois_with_count[poi] = defaultdict(int)
			if poi not in test_pois_with_data:
				test_pois_with_data[poi] = dict()
			if road not in test_pois_with_data[poi]:
				test_pois_with_data[poi][road] = list()
			test_pois_with_count[poi][road] += 1
			test_pois_with_data[poi][road].append(u_address)

		#if line_count % 10 == 0:
		#	break
		#
	sorted_test_pois = {}
	for poi, count_dict in test_pois_with_count.items():	
		sorted_count_dict = sorted(count_dict.items(), key=lambda x: x[1], reverse=True)
		sorted_test_pois[poi] = sorted_count_dict[0][0]
	
	return test_pois, test_pois_with_count, test_pois_with_data, sorted_test_pois, test_roads


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

def check_score(results):
	lines = []
	with open(round1_submit_data_path, "r+", encoding="utf-8") as f:
		lines = f.readlines()
	
	count = 0
	sub_results = {}
	for line in lines:
		if count == 0:
			count += 1
			continue
		parts = line.strip().split(",")
		id_ = int(parts[0])
		sub_u_address = parts[1]
		sub_results[id_] = sub_u_address
		#print("id: {}, sub_u_address: {}".format(id_,sub_u_address))
		count += 1

	total_num = len(results)
	correct_num = 0
	for result in results:
		id_ = result[0]
		u_address = result[1]
		s_address = result[2]
		sub_u_address = ""
		if id_ in sub_results:
			sub_u_address = sub_results[id_]
		if s_address == sub_u_address:
			correct_num += 1
		else:
			print("id: {}, u_address: {}, s_address: {}, sub_u_address: {}".format(id_,u_address,s_address,sub_u_address))

	score = float(correct_num) / total_num
	return score


def gen_submission_new(all_roads, train_roads, test_roads, short_roads, prefix_short_roads, train_pois, test_pois, amap_pois, dist_dict):
	for key, value in test_pois.items():
		if key in train_pois:
			pass
		else:
			train_pois[key] = value
	poi_list = train_pois.keys()
	print("train_pois: ", poi_list)

	pattern1 = "(\\d{1,})\\-(\\d{1,})(\\-\\d{1,})?"
	p1 = re.compile(pattern1)

	pattern2 = "(\\d+[\\d\\-]*)室"
	p2 = re.compile(pattern2)

	pattern3 = "(\\d+[\\d\\-]*)[号栋梯]"
	p3 = re.compile(pattern3)

	pattern4 = "(\\d+[\\d\\-]*)号"
	p4 = re.compile(pattern4)

	pattern5 = "-[一二三四五六七八九十]"
	p5 = re.compile(pattern5)

	common_prefix = "福建省厦门市思明区"
	with open(test_data_with_entities_path, "r+", encoding="utf-8") as f:
		lines = f.readlines()
		total_count = 0
		missing_count = 0
		road_missing_count = 0
		house_missing_count = 0
		room_missing_count = 0
		results = []
		road_not_found = []
		for line in lines:
			json_line = json.loads(line.strip())
			id_ = json_line["id"]
			u_address = json_line["u_address"]
			entities = json_line["entities"]
			road_found = False
			road = ""
			house_no = ""
			room_no = ""
			poi = ""
			c_roads = []
			c_house_nos = []
			c_room_nos = []
			for entity in entities:
				if entity["type"] == "rd_st":
					c_roads.append(entity["entity"])

				if not road_found and entity["type"] == "rd_st":
					road = entity["entity"]
					if road == "店上东西里":
						road = "店上东里"
						c_roads.append("店上西里")
					elif road == "文兴东一二里":
						road = "文兴东二里"
						c_roads.append("文兴东一里")
					elif road == "古楼南里":
						c_roads.append("古楼北里")

					if road in all_roads:
						road_found = True
					else:
						verified_road = road_verify(u_address, road, all_roads, train_roads, test_roads, short_roads, prefix_short_roads)
						if verified_road != "":
							road_found = True
							road = verified_road

				if house_no != "" and entity["type"] == "house_no":
					house_no_2 = entity["entity"]
					if not house_no_2.endswith("号"):
						c_house_nos.append(house_no_2 + "号")
					else:
						c_house_nos.append(house_no_2)
					if house_no.find(house_no_2) != -1:
						pass
					else:
						house_no = house_no_2

				if house_no == "" and entity["type"] == "house_no":
					house_no = entity["entity"]
					if not house_no.endswith("号"):
						c_house_nos.append(house_no + "号")
					else:
						c_house_nos.append(house_no)

				if room_no != "" and entity["type"] == "room_no":
					c_room_no = entity["entity"]
					if not c_room_no.endswith("室"):
						c_room_nos.append(c_room_no + "室")
					else:
						c_room_nos.append(c_room_no)
					
				if room_no == "" and entity["type"] == "room_no":
					room_no = entity["entity"]
					if not room_no.endswith("室"):
						c_room_nos.append(room_no + "室")
					else:
						c_room_nos.append(room_no)

			'''
			if not house_no.endswith("号") and not room_no.endswith("室") and house_no == room_no:
				room_no = ""

			if house_no.endswith("号") and not room_no.endswith("室") and house_no[:-1] == room_no:
				room_no = ""

			if not house_no.endswith("号") and room_no.endswith("室") and house_no == room_no[:-1]:
				house_no = ""
			'''
			if not bool(re.search(r'\d', room_no)):
				room_no = ""

			if not bool(re.search(r'\d', house_no)):
				house_no = ""

			if room_no.endswith("号"):
				room_no = ""

			if house_no == "" and room_no == "":
				it = p1.finditer(u_address)
				for match in it:
					match_str = match.group(0)
					match_group1 = match.group(1)
					match_group2 = match.group(2)
					match_group3 = match.group(3)
					if match_group3 is not None:
						house_no = match_str
						room_no = ""
					elif len(match_group2) >= 2: 
						house_no = match_group1
						room_no = match_group2
					else:
						house_no = match_str
						room_no = ""
					#print("house_no and room_no null, match pattern str: {}, house_no: {}, room_no: {}".format(match_str, house_no, room_no))

			if room_no == "":
				it = p2.finditer(u_address)
				for match in it:
					match_str = match.group(0)
					room_no = match_str
					#print("room_no null, match pattern str: {}, room_no: {}".format(match_str, room_no))
			
			if house_no == "":
				it = p3.finditer(u_address)
				for match in it:
					match_str = match.group(0)
					house_no = match.group(1)
					#print("house_no null, match pattern str: {}, house_no: {}".format(match_str, house_no))

			it = p4.finditer(u_address)
			for match in it:
				match_str = match.group(0)
				if match_str not in c_house_nos:
					c_house_nos.append(match_str)
				#print("===== match_str: {}, house_no: {} =====".format(match_str,house_no))

			if house_no.find("$") != -1:
				hn_parts = house_no.split("$")
				house_no = hn_parts[0]

			if house_no.find("#") != -1:
				hn_parts = house_no.split("#")
				house_no = hn_parts[0]

			if room_no.find("$") != -1:
				rm_parts = room_no.split("$")
				room_no = rm_parts[1]
				
			if room_no.find("#") != -1:
				rm_parts = room_no.split("#")
				room_no = rm_parts[1]

			if house_no != "" and house_no.endswith("栋"):
				house_no = house_no[:-1] + "号"

			if house_no != "" and not house_no.endswith("号"):
				house_no = house_no + "号"

			if room_no != "" and not room_no.endswith("室"):
				room_no = room_no + "室"

			if not road_found:
				road = road_search(u_address, all_roads, train_roads, test_roads, short_roads)
				if road != "":
					road_found = True

			for entity in entities:
				if entity["type"] == "poi":
					poi = entity["entity"]
					if not road_found:
						if poi in train_pois:
							road = train_pois[poi]
							road_found = True
							#print("===== id: {}, poi in train_pois: {}, road: {}".format(id_, poi, road))
						else:
							poi_match = poi_search_simple(poi, poi_list)
							if poi_match in train_pois:
								road = train_pois[poi_match]
							if road != "":
								road_found = True
							#print("===== id: {}, poi not in train_pois: {}, poi_match: {}, road: {}".format(id_, poi, poi_match, road))

							if not road_found:
								road, poi_match, poi_address_match, house_no_match = poi_search(poi, all_roads, train_pois, amap_pois)
								if poi_match in train_pois:
									road = train_pois[poi_match]
								if road != "":
									road_found = True
									if house_no == "" and house_no_match != "":
										house_no = house_no_match
								#print("===== id: {}, poi not in train_pois, search_poi in amap: {}, road: {}, poi_match: {}, poi_address_match: {}, house_no_match: {}".format(id_, poi, road, poi_match, poi_address_match, house_no_match))
			

			amap_search = False
			if not road_found:
				if poi == "":
					poi_match = poi_search_simple(u_address, poi_list)
					if poi_match in train_pois:
						road = train_pois[poi_match]
					if road != "":
						road_found = True
					#print("===== id: {}, search poi in poi_list, poi_match: {}, road: {}".format(id_, poi_match, road))
			
					if not road_found:
						road, poi_match, poi_address_match, house_no_match = poi_search(u_address, all_roads, train_pois, amap_pois)
						if poi_match in train_pois:
							road = train_pois[poi_match]
						if road != "":
							road_found = True
							if house_no == "" and house_no_match != "":
								house_no = house_no_match
						amap_search = True
						#print("===== id: {}, search_poi in amap, road: {}, poi_match: {}, poi_address_match: {}, house_no_match: {}".format(id_, road, poi_match, poi_address_match, house_no_match))
			
			if not road_found or house_no == "" or room_no == "":
				#print("===== id: {}, road_found: {}, u_address: {}, road: {}, house_no: {}, room_no: {}".format(id_, road_found, u_address, road, house_no, room_no))
				missing_count += 1
				if not road_found:
					road_missing_count += 1
					road_not_found.append(json_line)
				if house_no == "":
					house_missing_count += 1
				if room_no == "":
					room_missing_count += 1
			else:
				pass
				#print("id: {}, road_found: {}, u_address: {}, road: {}, house_no: {}, room_no: {}".format(id_, road_found, u_address, road, house_no, room_no))
			
			if house_no.find("号") != -1:
				c_house_no = house_no.replace("号", "-1号")
				c_house_nos.append(c_house_no)

			cn_an_map = {"-一":1, "-二":2, "-三":3, "-四":4, "-五":5, "-六":6, "-七":7, "-八":8, "-九":9, "-十":10}
			it = p5.finditer(u_address)
			for match in it:
				match_str = match.group(0)
				if match_str in cn_an_map:
					an = cn_an_map[match_str]
					if house_no.find("号") != -1:
						c_house_no = house_no.replace("号", "-{}号".format(an))
						#print("p5 found, c_house_no: {}".format(c_house_no))
						c_house_nos.append(c_house_no)
			#if len(c_roads) > 1 or len(c_house_nos) > 1 or len(c_room_nos) > 1:
			#	print("id: {}, road_found: {}, u_address: {}, road: {}, house_no: {}, room_no: {}, c_roads: {}, c_house_nos: {}, c_room_nos: {}".format(id_, road_found, u_address, road, house_no, room_no, c_roads, c_house_nos, c_room_nos))

			if road in dist_dict:
				common_prefix = dist_dict[road]

			result_line = common_prefix + road + house_no + room_no
			if road.find("号") != -1:
				result_line = common_prefix + road + room_no

			results.append((id_, u_address, result_line, poi, road, house_no, room_no, amap_search, c_house_nos, c_room_nos, c_roads))
			total_count += 1
			
	#print("total_count: {}, missing_count: {}, road_missing_count: {}, house_missing_count: {}, room_missing_count: {}".format(total_count, missing_count,road_missing_count,house_missing_count,room_missing_count))
	return results

def get_addr_pool(addr_pool_path):
	addr_list = []
	with open(addr_pool_path, "r", encoding="utf-8") as f:
		lines = f.readlines()
		count = 0
		for line in lines:
			if count == 0:
				count += 1
				continue
			addr_list.append(line.strip())
	return addr_list

def addr_pool_index(addr_list):
	data_count = 0
	pattern = "^福建省厦门市(.*?)区([^\\d\\-]*)?([\\d\\-]*)?号(([\\w\\-]*)?室)?$"
	p = re.compile(pattern)
	roads = []
	ad_dict = {}
	house_no_dict = {}
	room_no_dict = {}
	rd_st_dict = {}
	dist_dict = {}
	for i in range(len(addr_list)):
		#address = df_dict.iloc[i].at['标准地址池']
		address = addr_list[i]
		ad_dict[address] = 1
		it = p.finditer(address)
		for match in it:
			district = match.group(1)
			road_street = match.group(2)
			house_no = match.group(3)
			room_no = match.group(4)
			roads.append(road_street)
			if house_no is not None:
				house_no = house_no + "号"
				if house_no not in house_no_dict:
					house_no_dict[house_no] = []
				house_no_dict[house_no].append(address)

			if room_no is not None:
				if room_no not in room_no_dict:
					room_no_dict[room_no] = []
				room_no_dict[room_no].append(address)

			if road_street is not None and road_street != "":
				if road_street not in rd_st_dict:
					rd_st_dict[road_street] = []
				rd_st_dict[road_street].append(address)

				if district is not None:
					r_district = "福建省厦门市" + district + "区"
					dist_dict[road_street] = r_district

		data_count += 1
		if data_count % 10000 == 0:
			print("{} data processed.".format(data_count))

	roads = list(set(roads))
	print("roads: ", roads)
	return roads, ad_dict, house_no_dict, room_no_dict, rd_st_dict, dist_dict

def addr_pool_refine(results, ad_dict, house_no_dict, room_no_dict, rd_st_dict, train_pois_with_count, test_pois_with_count):
	refined_results = []
	pattern = "^福建省厦门市(.*?)区([^\\d\\-]*)?(([\\d\\-\\#\\$]*)?号)?(([\\w\\-\\#\\$]*)?室)?$"
	p = re.compile(pattern)
	count = 0

	uaddr_dict = {}
	for result in results:
		id_ = result[0]
		u_address = result[1]
		s_address = result[2]
		road = result[4]
		if s_address in ad_dict:
			uaddr_dict[u_address] = road

	for result in results:
		id_ = result[0]
		u_address = result[1]
		s_address = result[2]
		poi = result[3]
		amap_search = result[7]
		c_house_nos = result[8]
		c_room_nos = result[9]
		c_roads = result[10]
		count += 1
		if s_address in ad_dict:
			refined_results.append((id_, u_address, s_address))	
			continue
		r_address = s_address
		it = p.finditer(s_address)
		for match in it:
			district = match.group(1)
			rd_st = match.group(2)
			house_no = match.group(3)
			room_no = match.group(5)

			#print("===== id: {}, u_address: {}, s_address: {}, rd_st: {}, house_no: {}, room_no: {} =====".format(id_, u_address, s_address, rd_st, house_no, room_no))
			if poi == "" and (rd_st == "" or amap_search):
				valid_uaddrs = uaddr_dict.keys()
				search_key = u_address
				if house_no is not None and search_key.find(house_no) != -1:
					search_key = search_key.replace(house_no, "")
				elif house_no is not None and search_key.find(house_no[:-1]) != -1:
					search_key = search_key.replace(house_no[:-1], "")

				if room_no is not None and search_key.find(room_no) != -1:
					search_key = search_key.replace(room_no, "")
				elif room_no is not None and search_key.find(room_no[:-1]) != -1:
					search_key = search_key.replace(room_no[:-1], "")
				model_match_uaddrs = process.extract(search_key, valid_uaddrs, limit=2)
				if model_match_uaddrs[0][1] >= 80:
					if model_match_uaddrs[0][0] in uaddr_dict:
						try_rd_st = uaddr_dict[model_match_uaddrs[0][0]]
						try_s_address = "福建省厦门市思明区" + try_rd_st
						if house_no is not None:
							try_s_address = try_s_address + house_no
						if room_no is not None:
							try_s_address = try_s_address + room_no
						#print("try_s_address: {}, model_match_uaddrs: ".format(try_s_address), model_match_uaddrs)
						if try_s_address in ad_dict:
							r_address = try_s_address
							break

			if poi in train_pois_with_count:
				new_s_address = ""
				for poi_road in train_pois_with_count[poi].keys():
					if poi_road != rd_st:
						try_s_address = s_address.replace(rd_st, poi_road)
						if try_s_address in ad_dict:
							new_s_address = try_s_address
							break
				if new_s_address != "":
					r_address = new_s_address
					#print("find r_address: {}".format(r_address))
					break

			if poi in test_pois_with_count:
				new_s_address = ""
				for poi_road in test_pois_with_count[poi].keys():
					if poi_road != rd_st:
						try_s_address = s_address.replace(rd_st, poi_road)
						if try_s_address in ad_dict:
							new_s_address = try_s_address
							break
				if new_s_address != "":
					r_address = new_s_address
					#print("find r_address: {}".format(r_address))
					break

			if len(c_house_nos) > 1:
				new_s_address = ""
				for c_house_no in c_house_nos:
					if house_no is not None and c_house_no != house_no:
						try_s_address = s_address.replace(house_no, c_house_no)
						if try_s_address in ad_dict:
							new_s_address = try_s_address
							break
				if new_s_address != "":
					r_address = new_s_address
					#print("c_house_nos, find r_address: {}".format(r_address))
					break

			if len(c_room_nos) > 1:
				new_s_address = ""
				for c_room_no in c_room_nos:
					if room_no is not None and c_room_no != room_no:
						try_s_address = s_address.replace(room_no, c_room_no)
						if try_s_address in ad_dict:
							new_s_address = try_s_address
							break
				if new_s_address != "":
					r_address = new_s_address
					#print("c_room_nos, find r_address: {}".format(r_address))
					break

			if len(c_roads) > 1:
				new_s_address = ""
				for c_road in c_roads:
					if rd_st is not None and rd_st != "" and c_road != rd_st:
						try_s_address = s_address.replace(rd_st, c_road)
						if try_s_address in ad_dict:
							new_s_address = try_s_address
							break
				if new_s_address != "":
					r_address = new_s_address
					#print("c_roads, find r_address: {}".format(r_address))
					break

			if house_no is not None and room_no is not None:
				try_s_address = s_address.replace(room_no, "")
				if try_s_address in ad_dict:
					r_address = try_s_address
					#print("remove room_no, find r_address: {}".format(r_address))
					break

			max_sim = 0
			max_sim_address = s_address
			
			rd_st_addr_list = []
			house_no_addr_list = []
			room_no_addr_list = []
			c_room_no_addr_list = []
			if rd_st is not None and rd_st != "" and rd_st in rd_st_dict and len(rd_st_dict[rd_st]) < 200:
				rd_st_addr_list = rd_st_dict[rd_st]
				match_addrs = process.extract(s_address, rd_st_addr_list, limit=1)
				#print("rd_st: {}, rd_st_addr_list len: {}, match_addrs: ".format(rd_st, len(rd_st_addr_list)), match_addrs)
				if match_addrs[0][1] > max_sim:
					max_sim = match_addrs[0][1]
					max_sim_address = match_addrs[0][0]
	
			#if house_no is not None and house_no+"号" in house_no_dict:
			if house_no is not None and house_no in house_no_dict and len(house_no_dict[house_no]) < 200: 
				#house_no = house_no + "号"
				house_no_addr_list = house_no_dict[house_no]
				match_addrs = process.extract(s_address, house_no_addr_list, limit=1)
				#print("house_no: {}, house_no_addr_list len: {}, match_addrs: ".format(house_no, len(house_no_addr_list)), match_addrs)
				if match_addrs[0][1] > max_sim:
					max_sim = match_addrs[0][1]
					max_sim_address = match_addrs[0][0]

			if room_no is not None and room_no in room_no_dict and len(room_no_dict[room_no]) < 200:
				room_no_addr_list = room_no_dict[room_no]
				match_addrs = process.extract(s_address, room_no_addr_list, limit=1)
				#print("room_no: {}, room_no_addr_list len: {}, match_addrs: ".format(room_no, len(room_no_addr_list)), match_addrs)
				if match_addrs[0][1] > max_sim:
					max_sim = match_addrs[0][1]
					max_sim_address = match_addrs[0][0]

			#if room_no is None and house_no is not None and house_no.replace("号","室") in room_no_dict and len(room_no_dict[house_no.replace("号","室")]) < 200:
			if room_no is None and house_no is not None and house_no.find("号") != -1 and house_no.replace("号","室") in room_no_dict and len(room_no_dict[house_no.replace("号","室")]) < 200:
				c_room_no = house_no.replace("号","室")
				c_room_no_addr_list = room_no_dict[c_room_no]

			rc_address = ""
			if len(house_no_addr_list) > 0 and len(room_no_addr_list) > 0:
				interset = list(set(house_no_addr_list).intersection(set(room_no_addr_list)))
				#print("house_no and room_no, interset1 len: {}, interset: ".format(len(interset)), interset)
				if len(interset) == 1:
					rc_address = interset[0]
				elif len(interset) > 1 and len(rd_st_addr_list) > 0:
					inter_found = False
					for inter_item in interset:
						if inter_item in rd_st_addr_list:
							rc_address = inter_item
							inter_found = True
							break
				elif len(interset) > 1:
					for inter_item in interset:
						inter_it = p.finditer(inter_item)
						for inter_match in inter_it:
							inter_rd_st = inter_match.group(2)
							inter_house_no = inter_match.group(3)
							inter_room_no = inter_match.group(5)
							if inter_rd_st is not None:
								if u_address.find(inter_rd_st) != -1:
									rc_address = inter_item
			
			c_room_valid = False
			if rc_address == "" and len(house_no_addr_list) > 0 and len(rd_st_addr_list) > 0:
				interset = list(set(house_no_addr_list).intersection(set(rd_st_addr_list)))
				#print("house_no and rd_st, interset2 len: {}, interset: ".format(len(interset)), interset)
				if len(interset) == 1:
					inter_it = p.finditer(interset[0])
					for inter_match in inter_it:
						inter_house_no = inter_match.group(3)
						inter_room_no = inter_match.group(5)
						if inter_room_no is not None:
							if inter_room_no.endswith("室"):
								inter_room_no = inter_room_no[:-1]
							if u_address.find(inter_room_no) != -1:
								rc_address = interset[0]
						else:
							rc_address = interset[0]
					#rc_address = interset[0]
				elif len(interset) > 1:
					for inter_item in interset:
						inter_it = p.finditer(inter_item)
						for inter_match in inter_it:
							inter_house_no = inter_match.group(3)
							inter_room_no = inter_match.group(5)
							if inter_room_no is not None:
								if inter_room_no.endswith("室"):
									inter_room_no = inter_room_no[:-1]
								if u_address.find(inter_room_no) != -1:
									rc_address = inter_item
							else:
								rc_address = inter_item
				else:
					if len(c_room_no_addr_list) > 0:
						c_interset = list(set(c_room_no_addr_list).intersection(set(rd_st_addr_list)))
						if len(c_interset) > 0:
							room_no_addr_list = c_room_no_addr_list
							c_room_valid = True
					if not c_room_valid:
						combset = []
						combset.extend(house_no_addr_list)
						combset.extend(rd_st_addr_list)
						combset = list(set(combset))
						for inter_item in combset:
							inter_it = p.finditer(inter_item)
							for inter_match in inter_it:
								inter_house_no = inter_match.group(3)
								inter_room_no = inter_match.group(5)
								match_hn = False
								match_rn = False
								if inter_room_no is not None:
									if inter_room_no.endswith("室"):
										inter_room_no = inter_room_no[:-1]
									if u_address.find(inter_room_no) != -1:
										match_hn = True
								else:
									match_hn = True
								if inter_house_no is not None:
									if inter_house_no.endswith("号"):
										inter_house_no = inter_house_no[:-1]
									if u_address.find(inter_house_no) != -1:
										match_rn = True
								else:
									match_rn = True
								if match_hn and match_rn:
									rc_address = inter_item

			if rc_address == "" and len(room_no_addr_list) > 0 and len(rd_st_addr_list) > 0:
				interset = list(set(room_no_addr_list).intersection(set(rd_st_addr_list)))
				#print("room_no and rd_st, interset3 len: {}, interset: ".format(len(interset)), interset)
				if len(interset) == 1:
					inter_it = p.finditer(interset[0])
					for inter_match in inter_it:
						inter_house_no = inter_match.group(3)
						inter_room_no = inter_match.group(5)
						if inter_house_no is not None and not c_room_valid:
							if inter_house_no.endswith("号"):
								inter_house_no = inter_house_no[:-1]
							if u_address.find(inter_house_no) != -1:
								rc_address = interset[0]
						else:
							rc_address = interset[0]
					#rc_address = interset[0]
				elif len(interset) > 1:
					for inter_item in interset:
						inter_it = p.finditer(inter_item)
						for inter_match in inter_it:
							inter_house_no = inter_match.group(3)
							inter_room_no = inter_match.group(5)
							if inter_house_no is not None and not c_room_valid:
								if inter_house_no.endswith("号"):
									inter_house_no = inter_house_no[:-1]
								if u_address.find(inter_house_no) != -1:
									rc_address = inter_item
							else:
								rc_address = inter_item
				else:
					combset = []
					combset.extend(house_no_addr_list)
					combset.extend(rd_st_addr_list)
					combset = list(set(combset))
					for inter_item in combset:
						inter_it = p.finditer(inter_item)
						for inter_match in inter_it:
							inter_house_no = inter_match.group(3)
							inter_room_no = inter_match.group(5)
							match_hn = False
							match_rn = False
							if inter_room_no is not None:
								if inter_room_no.endswith("室"):
									inter_room_no = inter_room_no[:-1]
								if u_address.find(inter_room_no) != -1:
									match_hn = True
							else:
								match_hn = True
							if inter_house_no is not None:
								if inter_house_no.endswith("号"):
									inter_house_no = inter_house_no[:-1]
								if u_address.find(inter_house_no) != -1:
									match_rn = True
							else:
								match_rn = True
							if match_hn and match_rn:
								rc_address = inter_item

			if rc_address == "" and len(rd_st_addr_list) > 0 and len(house_no_addr_list) == 0 and len(room_no_addr_list) == 0:
				for inter_item in rd_st_addr_list:
					inter_it = p.finditer(inter_item)
					for inter_match in inter_it:
						inter_house_no = inter_match.group(3)
						inter_room_no = inter_match.group(5)
						if inter_room_no is not None:
							if inter_room_no.endswith("室"):
								inter_room_no = inter_room_no[:-1]
							if u_address.find(inter_room_no) != -1:
								rc_address = inter_item
						elif inter_house_no is not None:
							if inter_house_no.endswith("号"):
								inter_house_no = inter_house_no[:-1]
							if u_address.find(inter_house_no) != -1:
								rc_address = inter_item

			if rc_address != "":
				r_address = rc_address
				check_it = p.finditer(r_address)
				for chcek_match in check_it:
					check_house_no = chcek_match.group(3)
					check_room_no = chcek_match.group(5)
					if check_room_no is not None:
						if check_room_no.endswith("室"):
							check_room_no = check_room_no[:-1]
						if u_address.find(check_room_no) == -1:
							r_address = "invalid"
				#print("r_address1: {}".format(r_address))
			elif max_sim >= 90:
				r_address = max_sim_address
				check_it = p.finditer(r_address)
				for chcek_match in check_it:
					check_house_no = chcek_match.group(3)
					check_room_no = chcek_match.group(5)
					if check_room_no is not None:
						if check_room_no.endswith("室"):
							check_room_no = check_room_no[:-1]
						if u_address.find(check_room_no) == -1:
							r_address = "invalid"
				#print("r_address2: {}, max_sim: {}".format(r_address, max_sim))
			else:
				r_address = "invalid"

		refined_results.append((id_, u_address, r_address))	
	return refined_results


def gen_submission_final(addr_pool):
	start_time = time.time()
	all_roads = []
	road_dict, xiamen_roads = get_xiamen_road_list()
	print("xiamen_roads len: {}".format(len(xiamen_roads)))

	all_roads.extend(xiamen_roads)
	#addr_pool_roads, ad_dict, house_no_dict, room_no_dict, rd_st_dict = addr_pool_index(addr_pool)
	#all_roads.extend(addr_pool_roads)
	
	data = get_train_data()
	data_with_label,train_roads = process_train_data(data, road_dict)
	print("train_roads len: {}".format(len(train_roads)))

	all_roads.extend(list(train_roads))
	all_roads = list(set(all_roads))
	print("all_roads len after train: {}".format(len(all_roads)))

	sorted_train_pois = {}
	try:
		train_pois, train_pois_with_count, train_pois_with_data, sorted_train_pois = load_train_pois()
	except:
		pass

	sorted_train_pois['美疆商贸公司'] = '莲前西路336-10号'
	sorted_train_pois['加州建材广场'] = '云顶中路515号'
	sorted_train_pois['松树公寓'] = '洪莲西二路1462号'
	sorted_train_pois['宝龙一城'] = '吕岭路1581号'

	sorted_test_pois = {}
	test_roads = {}
	try:
		test_pois, test_pois_with_count, test_pois_with_data, sorted_test_pois, test_roads = load_test_pois(all_roads, train_pois)	
	except:
		pass

	#test_roads process
	u_test_road = []
	for test_road in test_roads:
		if test_road not in all_roads:
			valid = True
			for road_entry in all_roads:
				if road_entry.find(test_road) != -1 or test_road.find(road_entry) != -1:
					valid = False
					break
			if valid:
				u_test_road.append(test_road)
	invalid_roads = ["店上东西里", "文兴东一二里", "东方山庄"]
	u2_test_road = []
	for test_road1 in u_test_road:
		valid = True
		for test_road2 in u_test_road:
			if test_road1 != test_road2 and test_road2.find(test_road1) != -1:
				valid = False
				break
		if test_road1 in train_pois or test_road1 in test_pois or test_road1 in invalid_roads:
			valid = False
		if valid:
			u2_test_road.append(test_road1)
	test_roads = u2_test_road
	all_roads.extend(list(test_roads))
	all_roads = list(set(all_roads))
	print("all_roads len after test: {}".format(len(all_roads)))

	r1_test_pois_with_count = {'怡富花园': {'莲前西路': 18}, '益友花园': {'莲前西路': 6}, '前埔北一里': {'前埔一里': 28}, '云景花园': {'洪文六里': 7}, '华林东盛花园': {'前埔东一里': 13}, '厦航洪文小区': {'洪文六里': 22, '洪文八里': 5}, '金磊花园': {'西林西二里': 11}, '金沙花园': {'西林西二里': 3}, '庐山公寓': {'东浦三里': 7}, '益辉花园': {'洪莲路': 8}, '民盛安置房': {'云顶中路': 8}, '东芳花园': {'莲前西路': 4}, '文兴东一二里': {'文兴东一里': 5, '文兴东二里': 5}, '前埔北区二里': {'前埔二里': 30, '田厝路': 5}, '瑞景生活广场': {'洪文一里': 2, '洪文二里': 2}, '龙山桥部队小区': {'东浦路': 8}, '前埔南小区': {'店上东里': 18, '店上西里': 10}, '瑞景新村': {'洪文一里': 29}, '易地办安置房': {'西林西里': 4}, '华林绿景花园': {'前埔五里': 6}, '广顺花园': {'莲前西路': 5}, '新潘宅小区': {'洪莲里': 3}, '香榭园': {'西林东里': 13}, '华林紫微小区': {'莲前东路': 9, '前埔西路': 1}, '永裕侨建闽联云顶岩': {'西林东路': 4, '云顶岩路': 4}, '洪山柄北区': {'洪莲西里': 11}, '联丰商城': {'洪莲里': 5, '洪莲路': 2, '莲前东路': 1}, '金鸡亭花园小区': {'西林东里': 9, '西林西里': 31}, '古楼南里': {'古楼北里': 17}, '嘉盛豪园': {'洪莲西里': 25, '云顶中路': 1}, '华瑞花园': {'洪文五里': 6}, '福满山庄': {'洪莲里': 9, '莲前东路': 1}, '源泉山庄': {'前埔六里': 6}, '红星瑞景小区': {'洪文四里': 6}, '水警鼓疗': {'西林东路': 3}, '明发建群雅苑小区': {'前埔一里': 2, '莲前东路': 2}, '文兴社': {'洪文五里': 1}, '林海阳光小区': {'东浦三里': 4}, '瑞景公园': {'洪文七里': 7}, '莲坂军休所': {'莲前西路': 8}, '前埔不夜城': {'文兴东路': 1}, '禹洲云顶国际': {'莲前西路': 1}, '云顶至尊': {'洪文七里': 4}, '东方大院': {'莲前西路': 3, '金尚路': 4}, '一布家园': {'莲前西路': 2}, '联发紫微花园': {'莲前东路': 2}, '国贸新城': {'前埔二里': 4}, '万景公寓': {'洪莲东二里': 8}, '侨福城': {'侨文里': 6, '侨洪里': 2}, '红星安华小区': {'东浦三里': 6, '东浦路': 1}, '明发商城': {'西林路': 4, '莲前西路': 1}, '前埔南区': {'莲前东路': 2}, '瑞景双座': {'洪莲中路': 2}, '侨文里侨福城': {'侨文里': 2}, '都市新巢小区': {'洪莲中路': 2}, '地质勘察院': {'莲前西路': 1}, '民航宿舍': {'东浦三里': 1}, '绿洲花园': {'莲前东路': 4}, '墩仔家园': {'西林东路': 3}, '诚毅公寓': {'前埔中路': 1}, '华林二期': {'前埔二里': 1}, '上东美地': {'洪莲东二里': 2}, '经委宿舍': {'东浦路': 1}, '云亭花园': {'莲前西路': 4}, '潘宅南小区': {'洪文六里': 2}, '华美宿舍楼': {'东浦路': 2}, '洪山柄南区': {'洪文五里': 5}, '夏商百批宿舍': {'莲前西路': 1}, '富山公寓': {'东浦路': 1}, '明发楼': {'洪莲西二里': 1}, '一百宿舍': {'东浦路': 1}, '民盛商厦': {'云顶中路': 1}, '红星公寓': {'东浦路': 1}, '华林一期三期': {'前埔五里': 2}, '厦航小区': {'洪文六里': 1}, '玉成豪园': {'莲前西路': 1}, '松宿公寓': {'洪莲西二路': 1}}
	for key,value in r1_test_pois_with_count.items():
		if key in train_pois_with_count:
			for vk,vv in value.items():
				if vk in train_pois_with_count[key]:
					train_pois_with_count[key][vk] += vv
				else:
					train_pois_with_count[key][vk] = vv
		else:
			train_pois_with_count[key] = value

	r1_sorted_test_pois = {'怡富花园': '莲前西路', '益友花园': '莲前西路', '前埔北一里': '前埔一里', '云景花园': '洪文六里', '华林东盛花园': '前埔东一里', '厦航洪文小区': '洪文六里', '嘉盛豪园': '洪莲西里', '金磊花园': '西林西二里', '金沙花园': '西林西二里', '庐山公寓': '东浦三里', '益辉花园': '洪莲路', '民盛安置房': '云顶中路', '东芳花园': '莲前西路', '前埔北区二里': '前埔二里', '瑞景生活广场': '洪文一里', '龙山桥部队小区': '东浦路', '前埔南小区': '店上东里', '平潭驻厦办': '东浦路', '瑞景新村': '洪文一里', '易地办安置房': '西林西里', '华林绿景花园': '前埔五里', '广顺花园': '莲前西路', '新潘宅小区': '洪莲里', '香榭园': '西林东里', '华林紫微小区': '莲前东路', '永裕侨建闽联云顶岩': '西林东路', '洪山柄北区': '洪莲西里', '联丰商城': '洪莲里', '金鸡亭花园小区': '西林西里', '古楼南里': '古楼北里', '华瑞花园': '洪文五里', '福满山庄': '洪莲里', '源泉山庄': '前埔六里', '红星瑞景小区': '洪文四里', '水警鼓疗': '西林东路', '文兴社': '洪文五里', '林海阳光小区': '东浦三里', '瑞景公园': '洪文七里', '莲坂军休所': '莲前西路', '云顶至尊': '洪文七里', '东方大院': '莲前西路', '一布家园': '莲前西路', '金尚路东方大院': '金尚路', '联发紫微花园': '莲前东路', '国贸新城': '前埔二里', '万景公寓': '洪莲东二里', '侨福城小区': '侨文里', '红星安华小区': '东浦三里', '明发商城': '西林路', '瑞景双座': '洪莲中路', '都市新巢小区': '洪莲中路', '地质勘察院': '莲前西路', '民航宿舍': '东浦三里', '绿洲花园': '莲前东路', '墩仔家园': '西林东路', '诚毅公寓': '前埔中路', '华林二期': '前埔二里', '云亭花园': '莲前西路', '潘宅南小区': '洪文六里', '华美宿舍楼': '东浦路', '洪山柄南区': '洪文五里', '夏商百批宿舍': '莲前西路', '富山公寓': '东浦路', '民盛商厦': '云顶中路', '红星公寓': '东浦路', '华林一期三期': '前埔五里', '厦航小区': '洪文六里', '玉成豪园': '莲前西路', '松宿公寓': '洪莲西二路', '万科金域蓝湾': '洪莲东二里', '文化站': '东浦路', '云松居': '西林西里', '宝龙国际中心': '吕岭路1599号', '明发楼': '洪莲西二里', '会展中心': '会展路198号', '莲前街道办事处': '莲前西路859-1号', '盘古社区': '吕岭路', '圣华佗中医理疗': '莲前东路87号', '香榭圆':'西林东里'}
	#r1_sorted_test_pois = {'怡富花园': '莲前西路', '益友花园': '莲前西路', '前埔北一里': '前埔一里', '云景花园': '洪文六里', '华林东盛花园': '前埔东一里', '厦航洪文小区': '洪文六里', '嘉盛豪园': '洪莲西里', '金磊花园': '西林西二里', '金沙花园': '西林西二里', '庐山公寓': '东浦三里', '益辉花园': '洪莲路', '民盛安置房': '云顶中路', '东芳花园': '莲前西路', '前埔北区二里': '前埔二里', '瑞景生活广场': '洪文一里', '龙山桥部队小区': '东浦路', '前埔南小区': '店上东里', '平潭驻厦办': '东浦路', '瑞景新村': '洪文一里', '易地办安置房': '西林西里', '华林绿景花园': '前埔五里', '广顺花园': '莲前西路', '新潘宅小区': '洪莲里', '香榭园': '西林东里', '华林紫微小区': '莲前东路', '永裕侨建闽联云顶岩': '西林东路', '洪山柄北区': '洪莲西里', '联丰商城': '洪莲里', '金鸡亭花园小区': '西林西里', '古楼南里': '古楼北里', '华瑞花园': '洪文五里', '福满山庄': '洪莲里', '源泉山庄': '前埔六里', '红星瑞景小区': '洪文四里', '水警鼓疗': '西林东路', '文兴社': '洪文五里', '林海阳光小区': '东浦三里', '瑞景公园': '洪文七里', '莲坂军休所': '莲前西路', '云顶至尊': '洪文七里', '东方大院': '莲前西路', '一布家园': '莲前西路', '金尚路东方大院': '金尚路', '联发紫微花园': '莲前东路', '国贸新城': '前埔二里', '万景公寓': '洪莲东二里', '侨福城小区': '侨文里', '红星安华小区': '东浦三里', '明发商城': '西林路', '瑞景双座': '洪莲中路', '都市新巢小区': '洪莲中路', '地质勘察院': '莲前西路', '民航宿舍': '东浦三里', '绿洲花园': '莲前东路', '墩仔家园': '西林东路', '诚毅公寓': '前埔中路', '华林二期': '前埔二里', '云亭花园': '莲前西路', '潘宅南小区': '洪文六里', '华美宿舍楼': '东浦路', '洪山柄南区': '洪文五里', '夏商百批宿舍': '莲前西路', '富山公寓': '东浦路', '民盛商厦': '云顶中路', '红星公寓': '东浦路', '华林一期三期': '前埔五里', '厦航小区': '洪文六里', '玉成豪园': '莲前西路', '松宿公寓': '洪莲西二路'}
	for key,value in r1_sorted_test_pois.items():
		if key in sorted_train_pois:
			pass
		else:
			sorted_train_pois[key] = value
	
	r1_test_roads = ['店上西里', '前埔二里', '西林西里', '文兴东二里', '洪莲西二里', '前埔六里', '前埔西路', '西林西二路', '玉亭里', '洪文五里', '洪莲中路', '西林东里', '洪莲北路', '莲前东路', '东芳山庄', '洪文二里', '侨兴里', '东坪山社', '莲前西路', '东浦路', '洪莲中路', '吕岭路', '洪莲路', '前埔西路', '前埔一里', '侨洪里', '潘宅路', '云顶岩路', '洪文一里', '前埔南路', '洪文泥窟社西片区', '洪莲东二里', '前埔南路', '前埔路', '古楼北里', '前埔东一里', '前埔东路', '洪莲西路', '店上东里', '洪莲西二路', '云顶中路', '洪莲里', '洪文三里', '西林东路', '文兴西路', '洪文四里', '莲前西路', '古兴里', '东山社', '洪文八里', '洪莲西里', '洪文石村社北片区', '云顶中路', '洪文石村社南片区', '洪莲北路', '前埔五里', '侨龙里', '田厝路', '洪文七里', '文兴东路', '文兴东一里', '洪莲北二路', '洪文六里', '洪文泥窟社东片区', '金尚路', '洪文泥窟社北片区', '古楼南里', '东浦三里', '西林路', '文兴东三里', '侨洪里', '西林社', '西林西二里', '莲前东路', '前埔中路', '侨文里', '文兴西路', '洪莲中二路', '侨福里']
	all_roads.extend(list(r1_test_roads))
	all_roads = list(set(all_roads))
	print("all_roads len after r1 test: {}".format(len(all_roads)))

	short_roads = []
	for road_item in train_roads:
		if road_item.endswith("路"):
			short_roads.append(road_item[:-1])
	for road_item in r1_test_roads:
		if road_item.endswith("路"):
			short_roads.append(road_item[:-1])
	short_roads = list(set(short_roads))

	prefix_short_roads = {}
	for short_road1 in short_roads:
		for short_road2 in short_roads:
			if short_road1 != short_road2 and short_road2.find(short_road1) != -1:
				if short_road1 not in prefix_short_roads:
					prefix_short_roads[short_road1] = []
				prefix_short_roads[short_road1].append(short_road2)

	print(prefix_short_roads)

	amap_pois = get_amap_pois()

	addr_pool_roads, ad_dict, house_no_dict, room_no_dict, rd_st_dict, dist_dict = addr_pool_index(addr_pool)
	
	results = gen_submission_new(all_roads, train_roads, test_roads, short_roads, prefix_short_roads, sorted_train_pois, sorted_test_pois, amap_pois, dist_dict)
	#score = check_score(results)
	#print("===== score: {} =====".format(score))
	
	#addr_pool_roads, ad_dict, house_no_dict, room_no_dict, rd_st_dict = addr_pool_index(addr_pool)
	results = addr_pool_refine(results, ad_dict, house_no_dict, room_no_dict, rd_st_dict, train_pois_with_count, test_pois_with_count)

	#score = check_score(results)
	#print("===== score after refine: {} =====".format(score))

	end_time = time.time()
	print("gen_submission_final, total time used: {}".format((end_time-start_time)))
	return results
