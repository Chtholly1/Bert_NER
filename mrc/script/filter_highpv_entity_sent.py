#coding:utf-8
import re
import os, sys
import json

diff_entity_file = './result/pred_diff_entities.20220118'
pv_ths = 3

def is_ins(loc, temp_loc):
    if loc[0]>= temp_loc[0] and loc[1] <= temp_loc[1]:
        return True
    return False

entity_dict = {}
entity_list = []
with open(diff_entity_file) as f:
    for line in f:
        token = line.strip().split()
        if len(token) != 2:
            continue
        try:
            if int(token[1]) >= pv_ths:
                entity_dict[token[0]] = token[1]
                entity_list.append(token[0])
        except:
            pass

r = re.compile( '('+'|'.join(entity_list)+ ')')
for line in sys.stdin:
    json_info = json.loads(line.strip())
    text = json_info['text']
    label_dict = json_info['label']
    res = r.finditer(text)
    for it in res:
        #print(it.group(), it.span())
        label = it.group()
        loc = list(it.span())
        flag = False
        for key, val in label_dict.items():
            for lab, loc_list in val.items():
                for temp_loc in loc_list:
                    if is_ins(loc, temp_loc):
                        flag = True
                        break
        if not flag:
            #print(label)
            #print(loc)
            print(text)
            #print(label_dict)
            break
            #if label in val:
            #    if loc not in label_dict[key][label]:
            #        print(label)
            #        print(loc)
            #        print(label_dict[key][label])
            #        print(text)
            #else:
            #    print(text)
