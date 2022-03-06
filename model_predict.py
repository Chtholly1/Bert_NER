# -*-coding:utf-8-*-
import os
import logging
import argparse
import warnings

import numpy as np
import torch as t
import torch.nn as nn
from transformers import BertTokenizerFast
from torch.utils.data import DataLoader

from conf.config import label_2_ids, args, use_crf, max_len, Config
from business.data_process.data_utils import parse_json_data, NERDataset
from business.models.model import AlBertNERModel, BertNERModel, AlBertLSTMModel, AlBertLSTMModelPredict
from business.model_plant import validate, train, test
from business.tools import setup_seed

logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(message)s')
warnings.filterwarnings(action='ignore')

ids_2_label = {v:k for k,v in label_2_ids.items()}
label_nums = len(label_2_ids)

class HotwordPredict:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def predict(self, text):
        input_ids, att_mask, word_ids = self.data_process(text)
        input_ids = input_ids.to(self.device).view(1, -1)
        att_mask = att_mask.to(self.device).view(1, -1).type(t.bool)
        logits = self.model(input_ids, att_mask)
        if use_crf == True:
            pred_label = t.tensor(self.model.crf.decode(logits), dtype=t.long)
            pred_label_2 = pred_label.view(pred_label.shape[0], pred_label.shape[1], 1)
            probabilities = t.zeros(pred_label.shape[0], pred_label.shape[1], label_nums).scatter_(2, pred_label_2, 1)
        else:
            probabilities = t.softmax(logits, dim=-1)
        probabilities = probabilities.cpu().numpy()
        y = np.argmax(probabilities, axis=2)
        tag_list = [ids_2_label[_] for idx, _ in enumerate(y[0]) if att_mask[0][idx]]
        tag_list = tag_list[1:-1]
        print(tag_list)
        #实体预测结果结构化
        temp_tag = 'O'
        start_idx = 0
        end_idx = 0
        loc_list = []
        for idx in word_ids:
            if idx != None:
                if tag_list[idx].startswith('O'):
                    if start_idx != end_idx:
                        loc_list.append([tuple([start_idx, end_idx]), temp_tag.split('-')[-1]])
                    start_idx = idx
                    end_idx = idx
                elif tag_list[idx].startswith('B'):
                    if start_idx != end_idx:
                        loc_list.append([tuple([start_idx, end_idx]), temp_tag.split('-')[-1]])
                    start_idx = idx
                    end_idx = idx+1
                    temp_tag = tag_list[idx].replace('B', 'I')
                elif tag_list[idx].startswith('I'):
                    if tag_list[idx] == temp_tag:
                        end_idx += 1
                    else:
                        if start_idx != end_idx:
                            loc_list.append([tuple([start_idx, end_idx]), temp_tag.split('-')[-1]])
                        start_idx = idx
                        end_idx = idx     
        if start_idx != end_idx:
            loc_list.append([tuple([start_idx, end_idx]), temp_tag.split('-')[-1]])
        print('text:%s'%(text))
        for (loc, tag) in loc_list:
            print(loc, text[loc[0]: loc[1]], tag)
        return probabilities

    def data_process(self, text):
        tokens = self.tokenizer(list(text), is_split_into_words=True,  max_length=max_len, padding='max_length', truncation=True)
        input_ids = tokens['input_ids']
        attention_mask = tokens['attention_mask']
        word_ids = tokens.word_ids()
        return t.tensor(input_ids, dtype=t.long), t.tensor(attention_mask, dtype=t.long), word_ids

if __name__ == '__main__':
    setup_seed(2000)
    test_text = '2.0T发动机空调不错'

    #模型加载
    device = t.device("cuda:{}".format(args.gpu_index) if t.cuda.is_available() else "cpu")
    tokenizer = BertTokenizerFast.from_pretrained(args.model_name, do_lower_case=True)
    config = Config()
    config.num_label = label_nums   
    model = AlBertLSTMModelPredict(config, use_crf=use_crf).to(device)
    checkpoint = t.load(args.model, map_location=device)
    model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()
    #预测类加载
    predict_cls = HotwordPredict(model, tokenizer, device)
    #模型预测
    predict_cls.predict(test_text)    

