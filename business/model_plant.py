# -*- coding: utf-8 -*-
import os
import time
import json
import random
import numpy as np
import pandas as pd
import collections
import torch
import torch as t
import torch.nn as nn
from torch.utils.data import Dataset
from torchcrf import CRF
from tqdm import tqdm
from seqeval.metrics import classification_report, f1_score
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score

from conf.config import use_crf, max_len#, num_labels, ids_2_label

def seed_all(seed=2021):
    random.seed(seed)
    np.random.seed(seed)
    t.manual_seed(seed)


def correct_predictions(output_probabilities, targets):
    _, out_classes = output_probabilities.max(dim=2)
    out_classes_new = out_classes.view(-1,1)
    targets_new = targets.view(-1,1).cpu()
    correct = (out_classes_new == targets_new).sum()
    return correct.item()

def my_round(prob, ths):
    if prob > ths:
        return 1
    else:
        return 0

def calc_every_label_acc(data_list, y_true, y_pred):
    label_dict = dict()
    label_all_dict = dict()
    for item in data_list:
        label_dict[item[-1]] = [0, 0]
        label_all_dict[item[-1]] = [[], []]
    label_all_dict['all'] = [[], []]
    label_dict['all'] = [0, 0]
    for item, y, yp in zip(data_list, y_true, y_pred):
        label_name = item[-1]
        label_all_dict[label_name][0].append(y)
        label_all_dict[label_name][1].append(my_round(yp, 0.95))
        label_all_dict['all'][0].append(y)
        label_all_dict['all'][1].append(my_round(yp, 0.95)) 
        label_dict[label_name][0] += 1
        label_dict['all'][0] += 1
        if yp > 0.5 and y == 1:
            label_dict[label_name][1] += 1
            label_dict['all'][1] += 1
        elif yp<0.5 and y == 0:
            label_dict[label_name][1] += 1
            label_dict['all'][1] += 1
    #for key in label_all_dict:
    #    print(key)
    #    print(label_all_dict[key][0])
    #    print(label_all_dict[key][1])
    return label_dict, label_all_dict
    
def save_result(data_list, y_true, y_pred, output_file='./result.csv'):
    data = []
    for item, y, y_ in zip(data_list, y_true, y_pred):
        data.append(
             {'talk': item[0].replace(' ',''),
             'stand': item[1].replace(' ',''),
             'ori_label': item[3],
             'label': y,
             'prediction': y_}
        )
    df = pd.DataFrame(data)
    df.to_csv(output_file, sep='\t', index=False)


def validate(model, dataloader, device, num_labels, id_label_dict, output_file=None, ema=None):
    # Switch to evaluate mode.
    model.eval()
    epoch_start = time.time()
    running_loss = 0.0
    running_accuracy = 0.0
    all_prob = []
    all_labels = []
    all_pred = []
    tid1s = []
    tid2s = []
    if ema:
        ema.apply_shadow()
    with t.no_grad():
        for (input_ids, att_mask, labels) in dataloader:
            # Move input and output data to the GPU if one is used.
            input_ids = input_ids.to(device)
            att_mask = att_mask.to(device).type(t.bool)
            labels = labels.to(device)
            loss, logits = model(input_ids, att_mask, labels)
            if use_crf == True:
                pred_label = t.tensor(model.crf.decode(logits), dtype=t.long)
                pred_label_2 = pred_label.view(pred_label.shape[0], pred_label.shape[1], 1)
                probabilities = t.zeros(pred_label.shape[0], pred_label.shape[1], num_labels).scatter_(2, pred_label_2, 1)
                pred = np.argmax(probabilities, axis=2).to(device)
            else:
                probabilities = t.softmax(logits, dim=-1).cpu() 
                pred = np.argmax(probabilities, axis=2).to(device)
            loss = loss.mean()
            running_loss += loss.item()
            #all_prob.extend(probabilities.numpy())
            #all_labels.extend(labels.detach().cpu().tolist())
            flat_ids = input_ids.view(-1)
            flat_labels = labels.view(-1)
            flat_pred = pred.view(-1)
            input_ids_mask = att_mask.view(-1).gt(0)
            predictions = torch.masked_select(flat_pred, input_ids_mask)
            mask_labels = torch.masked_select(flat_labels, input_ids_mask)
            all_pred.extend(predictions.detach().cpu().tolist())
            all_labels.extend(mask_labels.detach().cpu().tolist())
    if ema:
        ema.restore()
    #y = np.argmax(all_prob, axis=2)
    #pred_tags = []
    #test_tags = []
    #for i in range(y.shape[0]):
    #    temp_tag = [id_label_dict[_] for _ in y[i]]
    #    test_tag = [id_label_dict[_] for _ in all_labels[i]]
    #    #temp_tag = temp_tag[1:-1]
    #    pred_tags.append(temp_tag)
    #    test_tags.append(test_tag)
    #final_tags = []
    #for test_tag, pred_tag in zip(test_tags, pred_tags):
    #    if len(test_tag) == len(pred_tag):
    #        final_tags.append(pred_tag)
    #    elif len(test_tag) < len(pred_tag):
    #        final_tags.append(pred_tag[:len(test_tag)])
    #    else:
    #        final_tags.append(pred_tag + ['O'] * (len(test_tag) - len(pred_tag)))
    pred_tags = []
    test_tags = []   
    for idx, item in enumerate(all_pred):
        pred_tags.append(id_label_dict[all_pred[idx]])
        test_tags.append(id_label_dict[all_labels[idx]])
    ## 利用seqeval对测试集进行验证
    report = classification_report(test_tags, pred_tags, digits=4)
    f1 = report.split('\n')[-3].split()[-2]
    print(report)
    print("f1:%s"%(f1))
    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(dataloader)
    #save_result(all_labels, all_prob, output_file=output_file)
    return epoch_time, epoch_loss, float(f1)

def train(model, dataloader, optimizer, max_gradient_norm, device, num_labels, scheduler=None, fgm=None, ema=None):
    model.train()
    epoch_start = time.time()
    batch_time_avg = 0.0
    running_loss = 0.0
    correct_preds = 0
    tqdm_batch_iterator = tqdm(dataloader)
    for batch_index, (input_ids, att_mask, labels) in enumerate(tqdm_batch_iterator):
        batch_start = time.time()
        # Move input and output data to the GPU if it is used.
        input_ids = input_ids.to(device)
        att_mask = att_mask.to(device).type(t.bool)
        labels = labels.to(device)

        optimizer.zero_grad()
        loss, logits = model(input_ids, att_mask, labels)
        if use_crf == True:
            pred_label = t.tensor(model.crf.decode(logits), dtype=t.long)
            #pred_label = model.crf.decode(logits, mask = att_mask)
            pred_label_2 = pred_label.view(pred_label.shape[0], pred_label.shape[1], 1)
            probabilities = t.zeros(pred_label.shape[0], pred_label.shape[1], num_labels).scatter_(2, pred_label_2, 1)
        else:
            probabilities = t.softmax(logits, dim=-1).cpu()
        loss = loss.mean()
        loss.backward()
        if fgm:
            fgm.attack()
            loss_adv, adv_logits = model(input_ids, att_mask, labels)
            loss_adv = loss_adv.mean()
            loss_adv.backward()
            fgm.restore()
        nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        if ema:
            ema.update()
        batch_time_avg += time.time() - batch_start
        running_loss += loss.item()
        correct_preds += correct_predictions(probabilities, labels)
        description = "Avg. batch proc. time: {:.4f}s, loss: {:.4f}" \
            .format(batch_time_avg / (batch_index + 1), running_loss / (batch_index + 1))
        tqdm_batch_iterator.set_description(description)
    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = correct_preds / (len(dataloader.dataset)*max_len)
    return epoch_time, epoch_loss, epoch_accuracy

def save_bad_case(input_ids, real_tags, pred_tags, tokenizer, output_file):
    for idx,item in enumerate(real_tags):
        input_tokens = tokenizer.convert_ids_to_tokens(input_ids[idx])
        start_idx = 1
        end_idx = input_tokens.index('[SEP]')
        if real_tags[idx][start_idx:end_idx] != pred_tags[idx][start_idx:end_idx]:
            sent = ''.join(input_tokens[start_idx:end_idx])
            print(sent)
            print(real_tags[idx][start_idx:end_idx])
            print(pred_tags[idx][start_idx:end_idx])

def test(model, dataloader, device, num_labels, id_label_dict, tokenizer, output_file=None):
    # Switch to evaluate mode.
    model.eval()
    epoch_start = time.time()
    running_loss = 0.0
    running_accuracy = 0.0
    all_ids = []
    all_prob = []
    all_labels = []
    att_list = []
    tid1s = []
    tid2s = []
    
    with t.no_grad():
        for (input_ids, att_mask, labels) in dataloader:
            # Move input and output data to the GPU if one is used.
            input_ids = input_ids.to(device)
            att_mask = att_mask.to(device).type(t.bool)
            labels = labels.to(device)
            loss, logits = model(input_ids, att_mask, labels)
            if use_crf == True:
                pred_label = t.tensor(model.crf.decode(logits), dtype=t.long)
                pred_label_2 = pred_label.view(pred_label.shape[0], pred_label.shape[1], 1)
                probabilities = t.zeros(pred_label.shape[0], pred_label.shape[1], num_labels).scatter_(2, pred_label_2, 1)
            else:
                probabilities = t.softmax(logits, dim=-1).cpu() 
            loss = loss.mean()
            running_loss += loss.item()
            all_ids.extend(input_ids.detach().cpu().tolist())
            all_prob.extend(probabilities.numpy())
            all_labels.extend(labels.detach().cpu().tolist())
            att_list.extend(att_mask.detach().cpu().tolist())
    y = np.argmax(all_prob, axis=2)
    pred_tags = []
    test_tags = []
    for i in range(y.shape[0]):
        temp_tag = [id_label_dict[_] for idx, _ in enumerate(y[i]) if att_list[i][idx]]
        test_tag = [id_label_dict[_] for idx, _ in enumerate(all_labels[i]) if att_list[i][idx]]
        pred_tags.append(temp_tag)
        test_tags.append(test_tag)
    final_tags = []
    for test_tag, pred_tag in zip(test_tags, pred_tags):
        if len(test_tag) == len(pred_tag):
            final_tags.append(pred_tag)
        elif len(test_tag) < len(pred_tag):
            final_tags.append(pred_tag[:len(test_tag)])
        else:
            final_tags.append(pred_tag + ['O'] * (len(test_tag) - len(pred_tag)))
    # 利用seqeval对测试集进行验证
    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(dataloader)
    save_bad_case(input_ids=all_ids, real_tags=test_tags, pred_tags=final_tags, tokenizer=tokenizer, output_file=output_file)
    print(classification_report(test_tags, final_tags, digits=4))
    return epoch_time, epoch_loss
