#coding:utf-8

import os
import sys
import json
import time
import collections
from transformers import BertTokenizer, BertTokenizerFast

from bin2.config import model_name, max_len

now_time = time.strftime("%Y%m%d%H", time.localtime())
pred_file = 'result/temp2.20220118'
tokenizer = BertTokenizerFast.from_pretrained(model_name, do_lower_case=True)
text_dict = {}
for line in sys.stdin: 
    info_json = json.loads(line.strip())
    tokens = tokenizer(list(info_json['text'].strip()), is_split_into_words=True,  max_length=max_len, truncation=True, )
    input_ids = tokens['input_ids']
    tokenizer_text = ''.join(tokenizer.convert_ids_to_tokens(input_ids[1:-1]))
    text_dict[tokenizer_text] = [info_json['text'].strip(), input_ids]

text_list = []
pred_entity_list = []
real_entity_list = []

with open(pred_file) as f:
    for line in f:
        text = line.strip()
        if text in text_dict:
            real_text = text
            input_ids = text_dict[text][1][1:-1]
            text_list.append(text_dict[text][0])
            #print(text_dict[text][0])
        else:
            entity_dict = dict()
            label_list = eval(text)
            start, end = 0, 0
            temp_entity = ''
            for idx, item in enumerate(label_list):
                if item == 'O':
                    if start != end:
                        entity_dict[tuple([start, end])] = [''.join(tokenizer.convert_ids_to_tokens(input_ids[start:end])),  temp_entity]
                    start = idx
                    end = idx
                    temp_entity = ''
                elif item[0] == 'B':
                    if start != end:
                        entity_dict[tuple([start, end])] = [''.join(tokenizer.convert_ids_to_tokens(input_ids[start:end])), temp_entity]
                    start = idx
                    end = idx + 1
                    temp_entity = item.replace('B', 'I')
                elif item[0] == 'I':
                    if item == temp_entity:
                        end = idx + 1
                    elif start != end:
                        entity_dict[tuple([start, end])] = [''.join(tokenizer.convert_ids_to_tokens(input_ids[start:end])), temp_entity]
                        start = idx
                        end = idx 
            if start != end:
                entity_dict[tuple([start, end])] = [''.join(tokenizer.convert_ids_to_tokens(input_ids[start:end])), temp_entity]
            #print(entity_dict)
            if len(pred_entity_list) == len(real_entity_list):
                real_entity_list.append(entity_dict)
            else:
                pred_entity_list.append(entity_dict)
#exit()
f_pred_miss = 'result/pred_miss.%s'%(now_time)
f_tag_miss = 'result/tag_miss.%s'%(now_time)
f_diff = 'result/diff.%s'%(now_time)
diff_dict = collections.defaultdict(int)
pred_miss_dict = collections.defaultdict(int)
tag_miss_dict = collections.defaultdict(int)

for idx, item in enumerate(real_entity_list):
    temp_real_dict = real_entity_list[idx]
    temp_pred_dict = pred_entity_list[idx]
    for key, val in temp_real_dict.items():
        if key in temp_pred_dict:
            if val != temp_pred_dict[key]:
                diff_dict[val[0]] += 1
        else:   
            pred_miss_dict[val[0]] += 1
            #if val[0] == '空间':
            #    print(text_list[idx])
            #    print(temp_real_dict)
            #    print(temp_pred_dict)
            #    break
    for key, val in temp_pred_dict.items():
        if key not in temp_real_dict:
            tag_miss_dict[val[0]] += 1
            if val[0] == '颜色':
                print(text_list[idx])
                print(temp_real_dict)
                print(temp_pred_dict)
                break

diff_miss_sort_list = sorted([(key, val) for key, val in diff_dict.items()], key=lambda x:x[1], reverse=True)
pred_miss_sort_list = sorted([(key, val) for key, val in pred_miss_dict.items()], key=lambda x:x[1], reverse=True)
tag_miss_sort_list = sorted([(key, val) for key, val in tag_miss_dict.items()], key=lambda x:x[1], reverse=True)

print("***diff entities***")
for item in diff_miss_sort_list:
    print(item[0], item[1])

print("***pred miss entities***")
for item in pred_miss_sort_list:
    
    print(item[0], item[1])

print("***tag miss entities***")
for item in tag_miss_sort_list:
    print(item[0], item[1])
