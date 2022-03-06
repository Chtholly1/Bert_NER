#coding:utf-8
import sys
import json
import collections
from config import args, deleted_labels, map_labels

label_entity_dict = dict()

def calc_one_freq_entity(label_dict):
    for label in label_dict:
        one_entity = 0
        for entity, val in label_dict[label].items():
            if val <= 2:
                one_entity += 1
        print("label:%s, total entity nums:%d, one entity nums:%d, prop:%.4f"%(label, len(label_dict[label]), one_entity, one_entity/len(label_dict[label])))
    return

if __name__ == '__main__':
    for line in sys.stdin:
        info = json.loads(line.strip())
        text = info['text']
        label_dict= info['label']
        loc_label_dict = {}
        for label, entity_dict in label_dict.items():
            if label in deleted_labels:
                continue
            elif label in map_labels:
                label = map_labels[label]
            if label not in label_entity_dict:
                label_entity_dict[label] = collections.defaultdict(int)
            for entity, loc_list in entity_dict.items():
                label_entity_dict[label][entity] += 1
    calc_one_freq_entity(label_entity_dict)
