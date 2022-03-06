# -*-coding:utf-8-*-
import os
import logging
import argparse
import warnings

import torch as t
import torch.nn as nn
from transformers import BertTokenizerFast
from torch.utils.data import DataLoader

from bin2.config import *
from bin2.data2 import parse_json_data, NERDataset
from bin2.model import AlBertNERModel, BertNERModel, AlBertLSTMModel
from bin2.model_plant import validate, train, test
from train_albert_lstm2 import setup_seed 

logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(message)s')
warnings.filterwarnings(action='ignore')

def test1(args):
    setup_seed(2000)
    device = t.device("cuda:{}".format(args.gpu_index) if t.cuda.is_available() else "cpu")
    tokenizer = BertTokenizerFast.from_pretrained(args.model_name, do_lower_case=True)
    #df = dataframe_generate(args.input_file, tokenizer)

    logging.info(20 * "=" + " Loading model from {} ".format(args.model) + 20 * "=")
    logging.info("\t* Loading validation data...")
    train_data, label_2_ids = parse_json_data(args.train_file, tokenizer)
    ids_2_label = {v:k for k,v in label_2_ids.items()}
    label_nums = len(label_2_ids)
    train_dataset = NERDataset(label_2_ids, *train_data)
    train_loader = DataLoader(train_dataset, shuffle=False, batch_size=args.batch_size)

    logging.info("\t* Building model...")
    config = Config()
    config.num_label = label_nums   

    if model_name == './base_albert':
        #model = AlBertNERModel(config).to(device)
        model = AlBertLSTMModel(config, use_crf=use_crf).to(device)
    elif model_name == './base_roberta':
        model = BertNERModel(config).to(device)
    checkpoint = t.load(args.model, map_location=device)
    model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()
    print(checkpoint["epoch"])
    print(checkpoint['best_score'])
    #model = nn.DataParallel(model, device_ids=[5, 3, 1])

    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    output_file = os.path.join(args.result_dir, args.model.split('/')[-1] + '.csv')
    epoch_time, epoch_loss = test(model, train_loader, device, label_nums, ids_2_label, tokenizer, output_file=output_file)
    #logging.info("-> Valid. time: {:.4f}s, loss: {:.4f}, accuracy: {:.4f}%, auc: {:.4f}\n"
    #             .format(epoch_time, epoch_loss, (epoch_accuracy * 100), epoch_auc))

if __name__ == '__main__':
    test1(args)
