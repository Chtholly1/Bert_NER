# -*- coding: utf-8 -*-
import os
import sys
import time
import random
import argparse
import logging
import warnings
import collections
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
#from pytorch_transformers import BertTokenizer
from transformers import BertTokenizer, BertTokenizerFast, get_linear_schedule_with_warmup as linear_warmup_schedule
from transformers.optimization import AdamW

from conf.config import *
from business.data_process.data_utils import parse_json_data, NERDataset, NERDatasetSplit
from business.models.model import AlBertNERModel, BertNERModel, AlBertLSTMModel, AlBertMRCNERModel
from business.model_plant import validate, train
from business.tools import setup_seed, FGM

logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(message)s')
warnings.simplefilter(action='ignore')
now_time = time.strftime("%Y%m%d%H", time.localtime())

def main(args):
    torch.set_printoptions(profile="full")
    setup_seed(2000)
    device = torch.device("cuda:{}".format(args.gpu_index) if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizerFast.from_pretrained(model_name, do_lower_case=True)

    logging.info(20 * "=" + " Preparing for training " + 20 * "=")
    if not os.path.exists(args.target_dir):
        os.makedirs(args.target_dir)
    
    #if True:
    if args.train_file != args.val_file:
        logging.info("\t* Loading training data...")
        train_data, label_2_ids = parse_json_data(args.train_file, tokenizer)
        ids_2_label = {v:k for k,v in label_2_ids.items()}
        label_nums = len(label_2_ids)
        train_dataset = NERDataset(label_2_ids, *train_data)
        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
        print(len(train_dataset))
        logging.info("\t* Loading validation data...")
        val_data, _ = parse_json_data(args.val_file, tokenizer)
        val_dataset = NERDataset(label_2_ids, *val_data)
        val_loader = DataLoader(val_dataset, shuffle=False, batch_size=args.batch_size)
        print(len(val_dataset))
    else:
        logging.info("\t* Loading training data...")
        train_data, label_2_ids = parse_json_data(args.train_file, tokenizer)
        print(label_2_ids)
        ids_2_label = {v:k for k,v in label_2_ids.items()}
        label_nums = len(label_2_ids)
        total_idx = list(range(len(train_data[0])))
        random.shuffle(total_idx)
        train_dataset = NERDatasetSplit(label_2_ids, total_idx[:int(0.9*len(total_idx))], *train_data)
        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
        print("train_dataset len:%d"%(len(train_dataset)))
        logging.info("\t* Loading validation data...")
        val_dataset = NERDatasetSplit(label_2_ids, total_idx[int(0.9*len(total_idx)):], *train_data)
        val_loader = DataLoader(val_dataset, shuffle=False, batch_size=args.batch_size)
        print("val_dataset len:%d"%(len(val_dataset)))
    
    logging.info("\t* Building model...")
    config = Config()
    config.num_label = label_nums   

    if model_name.find('albert') >= 0:
        #model = AlBertLSTMModel(config, use_crf=use_crf).to(device)
        #model = AlBertNERModel(config, use_crf=use_crf).to(device)
        model = AlBertMRCNERModel(config).to(device)
    elif model_name.find('roberta') >= 0:
        model = BertNERModel(config).to(device)
    #model = AlBertLSTMModel(config, device, use_crf=False).to(device)
    #model = BilstmCRF(config, device, use_crf=False).to(device)
    #model = nn.DataParallel(model, device_ids=[1, 2])

    #param_optimizer = list(model.module.named_parameters())
    bert_optimizer = list(model.bert.named_parameters())
    #classifier_optimizer = list(model.classifier.named_parameters())
    #params = filter(lambda p: p.requires_grad, model.parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
            {'params': [p for n, p in bert_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': weight_decay},
            {'params': [p for n, p in bert_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max",
    #                                                       factor=0.85, patience=0)
    #scheduler1 = linear_warmup_schedule(optimizer, num_warmup_steps=100, num_training_steps = len(train_loader)*args.epochs)
    scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=0, min_lr=2e-6)
    print(optimizer.defaults['lr'])
    best_score = 0.0
    best_loss = 100
    start_epoch = 1

    _, valid_loss, _= validate(model, val_loader, device, label_nums, ids_2_label, tokenizer)
    logging.info("\t* Validation loss before training: {:.4f}".format(valid_loss))
    logging.info("\n" + 20 * "=" + "Training Bert model on device: {}".format(device) + 20 * "=")
    patience_counter = 0
    for epoch in range(start_epoch, args.epochs + 1):
        logging.info("* Training epoch {}:".format(epoch))
        print(optimizer.param_groups[0]['lr'])
        #fgm = FGM(model, epsilon=1, emb_name='word_embeddings')
        fgm = None
        epoch_time, epoch_loss, epoch_accuracy = train(model, train_loader, optimizer,
                                                       args.max_grad_norm, device, label_nums, scheduler=None, fgm=fgm)
        #for g in optimizer.param_groups: 
        #    g['lr'] *= 0.95
        logging.info("-> Training time: {:.4f}s, loss = {:.4f}, accuracy: {:.4f}%"
                     .format(epoch_time, epoch_loss, (epoch_accuracy * 100)))
        logging.info("* Validation for epoch {}:".format(epoch))
        epoch_time, epoch_loss, epoch_score = validate(model, val_loader, device, label_nums, ids_2_label, tokenizer)
        logging.info("-> Valid. time: {:.4f}s, loss: {:.4f}"
                     .format(epoch_time, epoch_loss))
        scheduler2.step(epoch_loss)
        #scheduler.step(epoch_accuracy)
        
        if epoch_score < best_score:
            patience_counter += 1
        else:
            best_score = epoch_score
            patience_counter = 0
            torch.save({"epoch": epoch,
                        "model": model.state_dict(),
                        "best_score": best_score},
                       os.path.join(args.target_dir, "best.pth.tar.%s" % now_time))
        if patience_counter >= args.patience:
            logging.info("-> Early stopping: patience limit reached, stopping...")
            break


if __name__ == "__main__":
    for k in args.__dict__:
        logging.info(k + ": " + str(args.__dict__[k]))
    main(args)

