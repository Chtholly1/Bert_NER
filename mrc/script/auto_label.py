#coding:utf-8

import re
import os,sys
import json

key_word_list = ['车型', '玻璃']
key_word_extend_dict = {'车型':[], '电动':['电动尾门', '电动座椅', '电动车', '电动折叠', '电动调节'], '玻璃':['挡风玻璃', '隐私玻璃']}
r = re.compile( '('+'|'.join(key_word_list)+ ')')
ext_list = []
for key, val in key_word_extend_dict.items():
    ext_list.extend(val)
r_ext = re.compile('(' + '|'.join(ext_list) + ')')

def is_ins(loc, temp_loc):
    if loc[0]>= temp_loc[0] and loc[1] <= temp_loc[1]:
        return True
    return False

def merge_it(it_all, it_ext):
    it_dict, it_ext_dict = {}, {}
    it_merge_info = []
    for it in it_all:
        label, loc = it.group(), it.span()
        flag = False
        for it2 in it_ext:
            label_ext, loc_ext = it2.group(), it2.span()
            if is_ins(loc, loc_ext):
                flag = True
                #it_merge_info.append([label_ext, loc_ext])
                break
        if not flag:
            it_merge_info.append([label,loc])
    return it_merge_info

for line in sys.stdin:
    json_info = json.loads(line.strip())
    text = json_info['text']
    label_dict = json_info['label']
    it_all = r.finditer(text)
    it_ext_all = r_ext.finditer(text)
    it_merge = merge_it(it_all, it_ext_all)
    for it in it_merge:
        label, loc = it
        flag = False
        for key, val in label_dict.items():
            for lab, loc_list in val.items():
                for temp_loc in loc_list:
                    if is_ins(loc, temp_loc):
                        flag = True
                        break
        if not flag:
            print(label, loc)
            print(text)
