#coding:utf-8

import json
import numpy as np
from conf.config import args, deleted_labels, map_labels
import torch as t
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertTokenizerFast

MAX_LEN = args.max_length

def is_in(real_loc, loc):
    if real_loc >= loc[0] and real_loc < loc[1]:
        return True
    return False

def parse_json_data(file_path, tokenizer):
    input_ids_a = []
    attention_mask_a = []
    label_ids_a = []
    labels_a = []
    total_label_list = []
    with open(file_path) as f:
        for line in f:
            info = json.loads(line.strip())
            text = info['text']
            label_dict= info['label']
            loc_label_dict = {}
            for label, entity_dict in label_dict.items():
                if label in deleted_labels:
                    continue
                elif label in map_labels:
                    label = map_labels[label]
                if label not in total_label_list:
                    total_label_list.append(label)      
                for entity, loc_list in entity_dict.items():
                    for loc in loc_list:
                        loc_label_dict[tuple(loc)] = label
            tokens = tokenizer(list(text),is_split_into_words=True,  max_length=MAX_LEN, padding='max_length',
                                       truncation=True)
            input_ids = tokens['input_ids']
            attention_mask = tokens['attention_mask']
            word_ids = tokens.word_ids()
            labels = ['O'] * MAX_LEN
            label_ids = [0] * MAX_LEN
            for idx, item in enumerate(labels):
                real_loc = word_ids[idx]
                if real_loc is None:
                    continue
                for loc,key in loc_label_dict.items():
                    if is_in(real_loc, loc):
                        if real_loc == loc[0]:
                            labels[idx] = 'B-' + key
                        else:
                            labels[idx] = 'I-' + key
                        break
            input_ids_a.append(input_ids)
            attention_mask_a.append(attention_mask)
            labels_a.append(labels)
            #label_ids_a.append(label_ids)
    real_label_dict = {'O':0}
    index = 1
    for key in total_label_list:
        real_label_dict['B-' + key] = index
        index += 1
        real_label_dict['I-' + key] = index
        index += 1
    return (input_ids_a, attention_mask_a, labels_a), real_label_dict

class NERDataset(Dataset):
    def __init__(self, label_2_ids, input_ids, attention_mask, labels):
        self.label_2_ids = label_2_ids
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        token_ids = self.input_ids[idx]
        att_mask = self.attention_mask[idx]
        new_labels = self.labels[idx]
        label_ids = [0] * MAX_LEN
        for idx, label in enumerate(new_labels):
            if label == 'O':
                label_ids[idx] = self.label_2_ids[label]
            else:
                if new_labels[idx] != new_labels[idx-1]:
                    label_ids[idx] = self.label_2_ids[label]
                else:
                    label_ids[idx] = self.label_2_ids[label]
        
        return t.tensor(token_ids, dtype=t.long), t.tensor(att_mask, dtype=t.long), \
               t.tensor(label_ids, dtype=t.long)

class NERDatasetSplit(Dataset):
    def __init__(self, label_2_ids, idx_list, input_ids, attention_mask, labels):
        self.label_2_ids = label_2_ids
        self.input_ids = np.array(input_ids)[idx_list]
        self.attention_mask = np.array(attention_mask)[idx_list]
        self.labels = np.array(labels)[idx_list]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        token_ids = self.input_ids[idx]
        att_mask = self.attention_mask[idx]
        new_labels = self.labels[idx]
        label_ids = [0] * MAX_LEN
        for idx, label in enumerate(new_labels):
            if label == 'O':
                label_ids[idx] = self.label_2_ids[label]
            else:
                if new_labels[idx] != new_labels[idx-1]:
                    label_ids[idx] = self.label_2_ids[label]
                else:
                    label_ids[idx] = self.label_2_ids[label]
        
        return t.tensor(token_ids, dtype=t.long), t.tensor(att_mask, dtype=t.long), \
               t.tensor(label_ids, dtype=t.long)

if __name__ == '__main__':
    file_path = args.train_file
    tokenizer = BertTokenizerFast.from_pretrained('./resource/base_models/base_albert', do_lower_case=True)
    q, real_label_dict = parse_json_data(file_path, tokenizer)
    print(input_ids[0])
