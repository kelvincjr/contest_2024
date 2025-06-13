import pickle
import pandas as pd
from tqdm import tqdm
import json
import os
import sys
import re

def sentence_split(str_centence):
    single_sentences_list = re.split(r'[。；？！\n]+', str_centence)
    return single_sentences_list

def check_title(text):
    p = re.compile(r"^(第[一二三四五六七八九十]+[章节条])|^(（[一二三四五六七八九十]+）)|^(\([一二三四五六七八九十]+\))|^([一二三四五六七八九十]+、)|^([一二三四五六七八九十]+是)|^(\d+、)|^(\d{1,2} )[^年月日]|^(?P<num1>\d+\.)[^\d]*?|^(（\d+）)|^(\(\d+\))|([一二三四五六七八九十几]+[个点])")
    matches = p.findall(text)
    matched_substrings = [(match, i) for group in matches for i, match in enumerate(group) if match]
    #print(matched_substrings)
    return matched_substrings

class DocTreeNode:
    def __init__(self, content, parent=None, is_leaf=False, type_=-1, is_excel=False, depth=0, font_weight=0, region_box=None, region_ri=-100, mc_mw=-1, page=None, page_num=-1):
        # -1: 普通的文本内容
        # 0123分别为四级标题
        self.type_ = type_
        #self.is_excel = is_excel
        self.content = content
        self.children = []
        self.parent = parent
        self.depth = depth 
        self.font_weight = font_weight
        self.region_box = region_box
        self.region_ri = region_ri
        self.mc_mw = mc_mw
        self.page = page
        self.is_leaf = is_leaf
        self.page_num = page_num

    def __str__(self):
        return self.content

    def print_children(self):
        for i, child in enumerate(self.children):
            #print(f"#{i}-is_excel:{child.is_excel}", child)
            if child.type_ != -1:
                print(f"#d:{child.depth}-pt:{self.type_}-ct:{child.type_}-cs:{i} ", child)
            elif len(child.content) > 10:
                print(f"#d:{child.depth}-pt:{self.type_}-ct:{child.type_}-cs:{i} ", child.content[:10])
            else:
                print(f"#d:{child.depth}-pt:{self.type_}-ct:{child.type_}-cs:{i} ", child)
            child.print_children()

    def content_to_nodes_mapping(self, content_to_nodes):
        for i, child in enumerate(self.children):
            if child.content not in content_to_nodes:
                content_to_nodes[child.content] = child
            child.content_to_nodes_mapping(content_to_nodes)

    def all_child_content(self, extend_results):
        for i, child in enumerate(self.children):
            sentences = sentence_split(child.content)
            if len(sentences) > 0:
                extend_results.append(sentences[0])
            #extend_results.append(child.content)
            child.all_child_content(extend_results)

    def all_sibling_content(self, extend_results):
        cur_type = self.type_
        for i, child in enumerate(self.parent.children):
            if child.type_ == cur_type and child.content != self.content:
                sentences = sentence_split(child.content)
                if len(sentences) > 0:
                    extend_results.append(sentences[0])
                #extend_results.append(child.content)
                #child.all_child_content(extend_results)

class DocTree:
    def __init__(self, txt_path, read_cache=True):
        self.path = txt_path
        #self.lines = open(txt_path, encoding="utf-8").read().split("\n")
        self.mid_nodes = []
        self.leaves = []
        self.root = DocTreeNode("@root", type_=-2, depth=0)
        self.root.path = self.path
        self.title_mappings = dict()

def gen_doctree(contents_toc, doctree):
    para_count = 0
    last_parent = doctree.root
    last_leaves = []
    for p, foc_flag in contents_toc:
        p_text = p
        #if len(p_text) < 2:
        #    para_count += 1
        #    continue

        #if foc_flag != 1:
        #    para_count += 1
        #    continue

        match_title = check_title(p_text)
        title_type = -1
        #if p_bold or (p_align == WD_PARAGRAPH_ALIGNMENT.CENTER):
        #    title_type = 100

        if len(match_title) > 0:
            match_title, title_type = match_title[0]
        if title_type == 10 and foc_flag != 1:
            title_type = -1
        #print('========== match title: {}, title_type: {}, foc_flag: {}, text: {}'.format(match_title, title_type, foc_flag, p_text))

        if title_type != -1:
            title_text = p_text
            #print('title text: {}'.format(title_text))

            last_leaves_len = len(last_leaves)
            if len(last_leaves) > 0:
                for node_text in last_leaves:
                    new_node = DocTreeNode(node_text, type_=-1, parent=last_parent, is_leaf=True, depth=last_parent.depth+1)
                    last_parent.children.append(new_node)
                    doctree.leaves.append(new_node)
                last_leaves = []

            if title_type != last_parent.type_:
                # 1. 检测出来的层级低于目前层级
                first_last_parent = last_parent
                type_ = title_type
                while last_parent is not None:
                    if type_ == last_parent.type_:
                        last_parent = last_parent.parent
                        break
                    else:
                        last_parent = last_parent.parent

                if last_parent is None:
                    last_parent = first_last_parent

                new_node = DocTreeNode(title_text, type_=type_, parent=last_parent, depth=last_parent.depth+1)
                last_parent.children.append(new_node)
                last_parent = new_node
            else:
                type_ = title_type
                while type_ == last_parent.type_:
                    last_parent = last_parent.parent

                new_node = DocTreeNode(title_text, type_=type_, parent=last_parent, depth=last_parent.depth+1)
                last_parent.children.append(new_node)
                last_parent = new_node
                doctree.mid_nodes.append(new_node)
        else:
            last_leaves.append(p_text)

        para_count += 1

    if len(last_leaves) > 0:
        for node_text in last_leaves:
            new_node = DocTreeNode(node_text, type_=-1, parent=last_parent, is_leaf=True, depth=last_parent.depth+1)
            last_parent.children.append(new_node)
            doctree.leaves.append(new_node)
        last_leaves = []

    doctree.root.print_children()
    content_to_nodes = {}
    doctree.root.content_to_nodes_mapping(content_to_nodes)
    return content_to_nodes