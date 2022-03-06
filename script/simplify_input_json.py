#coding:utf-8
import os
import sys
import json

for line in sys.stdin:
    info = json.loads(line.strip())
    text = info['text']
    label_list = info['annotations']
    label_json = {}
    for idx, item in enumerate(label_list):
        label = item['label']
        label_json[label] = {}
        for entity_info in item['segment']:
            entity = entity_info['str']
            begin = entity_info['begin']
            end = entity_info['end']
            if entity not in label_json[label]:
                label_json[label][entity] = [[begin, end]]
            else:
                label_json[label][entity].append([begin, end])
    new_info = {'text':text}
    new_info['label'] = label_json
    print(new_info)
