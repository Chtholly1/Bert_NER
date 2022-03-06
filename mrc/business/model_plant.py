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

from conf.config import use_crf, max_len, chn_2_eng#, num_labels, ids_2_label

def seed_all(seed=2021):
    random.seed(seed)
    np.random.seed(seed)
    t.manual_seed(seed)


def correct_predictions(output_probabilities, targets):
    _, out_classes = output_probabilities.max(dim=2)
    out_classes_new = out_classes.view(-1,1)
    targets_new = targets.view(-1,1)#.cpu()
    correct = (out_classes_new == targets_new).sum()
    return correct.item()

def correct_predictions_2(logits, targets):
    logits_r = t.sigmoid(logits)
    logits_new = logits_r.view(-1,1)
    targets_new = targets.view(-1,1)#.cpu()
    correct = (logits_new == targets_new).sum()
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
    
def extract_flat_spans(start_pred, end_pred, match_pred, label_mask, pseudo_tag = "control"):
    bmes_labels = ["O"] * len(start_pred)
    start_positions = [idx for idx, tmp in enumerate(start_pred) if tmp and label_mask[idx]]
    end_positions = [idx for idx, tmp in enumerate(end_pred) if tmp and label_mask[idx]]

    for start_item in start_positions:
        bmes_labels[start_item] = f"B-{pseudo_tag}"
    for end_item in end_positions:
        bmes_labels[end_item] = f"I-{pseudo_tag}"

    for tmp_start in start_positions:
        tmp_end = [tmp for tmp in end_positions if tmp >= tmp_start]
        if len(tmp_end) == 0:
            continue
        else:
            tmp_end = min(tmp_end)
        if match_pred[tmp_start][tmp_end]:
            if tmp_start != tmp_end:
                for i in range(tmp_start+1, tmp_end):
                    bmes_labels[i] = f"I-{pseudo_tag}"
            else:
                bmes_labels[tmp_end] = f"B-{pseudo_tag}"

    return bmes_labels

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


def validate(model, dataloader, device, num_labels, id_label_dict, tokenizer, output_file=None):
    # Switch to evaluate mode.
    model.eval()
    epoch_start = time.time()
    running_loss = 0.0
    running_accuracy = 0.0
    loss_fct = nn.BCEWithLogitsLoss()
    loss_bce = nn.BCEWithLogitsLoss(reduction="none")
    all_labels = []
    all_pred = []
    all_start_pred = []
    all_end_pred = []
    all_start_prob = []
    all_end_prob = []
    all_match_pred = []
    all_att_mask = []
    all_token_ids = []
    
    with t.no_grad():
        for (input_ids, att_mask, token_type_ids, labels, start_ids, end_ids, match_labels) in dataloader:
            batch_size = input_ids.size(0)
            # Move input and output data to the GPU if one is used.
            input_ids = input_ids.to(device)
            att_mask = att_mask.to(device).type(t.bool)
            token_type_ids = token_type_ids.to(device)
            labels = labels.to(device)
            start_ids = start_ids.to(device)
            end_ids = end_ids.to(device)
            match_labels = match_labels.to(device)
            start_label_mask = start_ids.gt(0)
            end_label_mask = end_ids.gt(0)
                   
            start_logits, end_logits, span_logits = model(input_ids, att_mask, token_type_ids, labels, start_ids, end_ids)

            input_ids_mask = att_mask.view(-1).gt(0)
            start_logits_r = torch.masked_select(start_logits.view(-1), input_ids_mask)
            end_logits_r = torch.masked_select(end_logits.view(-1), input_ids_mask)
            labels_r = torch.masked_select(labels.view(-1), input_ids_mask)
            start_ids_r = torch.masked_select(start_ids.view(-1), input_ids_mask)
            end_ids_r = torch.masked_select(end_ids.view(-1), input_ids_mask)
            start_loss = loss_fct(start_logits_r, start_ids_r.float())
            end_loss = loss_fct(end_logits_r, end_ids_r.float())
            match_label_row_mask = start_label_mask.bool().unsqueeze(-1).expand(-1, -1, max_len)
            match_label_col_mask = end_label_mask.bool().unsqueeze(-2).expand(-1, max_len, -1)
            match_label_mask = match_label_row_mask & match_label_col_mask
            match_label_mask = torch.triu(match_label_mask, 0)  # start should be less equal to end
            float_match_label_mask = match_label_mask.view(batch_size, -1).float()
            match_loss = loss_bce(span_logits.view(batch_size, -1), match_labels.view(batch_size, -1).float())
            match_loss = match_loss * float_match_label_mask
            match_loss = match_loss.sum() / (float_match_label_mask.sum() + 1e-10)
            loss = start_loss + end_loss + match_loss
            #loss = start_loss + end_loss
            start_prob = t.sigmoid(start_logits)
            end_prob = t.sigmoid(end_logits)
            all_start_prob.extend(start_prob.cpu().tolist())
            all_end_prob.extend(end_prob.cpu().tolist())
            start_pred = start_prob.gt(0.5)
            end_pred = end_prob.gt(0.5)
            match_pred = span_logits.gt(0)
            all_start_pred.extend(start_pred.tolist())
            all_end_pred.extend(end_pred.tolist())
            all_labels.extend(labels.cpu().tolist())
            all_token_ids.extend(input_ids.cpu().tolist())
            all_match_pred.extend(match_pred.cpu().tolist())
            all_att_mask.extend(att_mask.cpu().tolist())
            #if use_crf == True:
            #    pred_label = t.tensor(model.crf.decode(logits), dtype=t.long)
            #    pred_label_2 = pred_label.view(pred_label.shape[0], pred_label.shape[1], 1)
            #    probabilities = t.zeros(pred_label.shape[0], pred_label.shape[1], num_labels).scatter_(2, pred_label_2, 1)
            #    pred = np.argmax(probabilities, axis=2).to(device)
            #else:
            #    probabilities = t.softmax(logits, dim=-1).cpu() 
            loss = loss.mean()
            running_loss += loss.item()
    pred_tags = []
    test_tags = []   
    for idx, item in enumerate(all_labels):
        test_tag = []
        pred_tag = []
        flag = False
        label_name = ''.join(tokenizer.convert_ids_to_tokens(all_token_ids[idx][1:])).split('[SEP]')[0]
        pred_tag = extract_flat_spans(all_start_pred[idx], all_end_pred[idx], all_match_pred[idx], all_att_mask[idx], chn_2_eng[label_name])
        for i, _ in enumerate(all_labels[idx]):
            test_tag.append(id_label_dict[all_labels[idx][i]])
        #    if all_start_pred[idx][i] == 1 and all_end_pred[idx][i] == 1:
        #        pred_tag.append('B-control')
        #        flag = False   
        #    elif all_start_pred[idx][i] == 1:
        #        pred_tag.append('B-control')
        #        flag = True
        #    elif all_end_pred[idx][i] == 1:
        #        if not flag:
        #            pred_tag.append('O')
        #        else: 
        #            pred_tag.append('I-control')
        #        flag = False
        #    else:
        #        if flag:
        #            pred_tag.append('I-control')
        #        else:
        #            pred_tag.append('O')
        pred_tags.append(pred_tag)
        test_tags.append(test_tag)
        #if idx < 10:
        #    print("pred_tag:", pred_tag)
        #    print("test_tag:", test_tag)
        #    print("text:", tokenizer.convert_ids_to_tokens(all_token_ids[idx]))
    ## 利用seqeval对测试集进行验证
    report = classification_report(test_tags, pred_tags, digits=4)
    f1 = report.split('\n')[-3].split()[-2]
    print(report)
    print("f1:%s"%(f1))
    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(dataloader)
    epoch_acc  = running_accuracy/len(dataloader)
    #save_result(all_labels, all_prob, output_file=output_file)
    return epoch_time, epoch_loss, float(f1)

def train(model, dataloader, optimizer, max_gradient_norm, device, num_labels, scheduler=None, fgm=None):
    model.train()
    epoch_start = time.time()
    batch_time_avg = 0.0
    running_loss = 0.0
    correct_preds = 0
    loss_fct = nn.BCEWithLogitsLoss()
    loss_bce = nn.BCEWithLogitsLoss(reduction="none")
    tqdm_batch_iterator = tqdm(dataloader)
    for batch_index, (input_ids, att_mask, token_type_ids, labels, start_ids, end_ids, match_labels) in enumerate(tqdm_batch_iterator):
        batch_size = input_ids.size(0)
        batch_start = time.time()
        # Move input and output data to the GPU if it is used.
        input_ids = input_ids.to(device)
        att_mask = att_mask.to(device).type(t.bool)
        token_type_ids = token_type_ids.to(device)
        labels = labels.to(device)
        start_ids = start_ids.to(device)
        end_ids = end_ids.to(device)
        match_labels = match_labels.to(device)
        start_label_mask = start_ids.gt(0)
        end_label_mask = end_ids.gt(0)

        optimizer.zero_grad()
        start_logits, end_logits, span_logits = model(input_ids, att_mask, token_type_ids, labels, start_ids, end_ids)
        input_ids_mask = att_mask.view(-1).gt(0)
        start_logits_r = torch.masked_select(start_logits.view(-1), input_ids_mask)
        end_logits_r = torch.masked_select(end_logits.view(-1), input_ids_mask)
        labels_r = torch.masked_select(labels.view(-1), input_ids_mask)
        start_ids_r = torch.masked_select(start_ids.view(-1), input_ids_mask)
        end_ids_r = torch.masked_select(end_ids.view(-1), input_ids_mask)
        start_loss = loss_fct(start_logits_r, start_ids_r.float())
        end_loss = loss_fct(end_logits_r, end_ids_r.float())
        match_label_row_mask = start_label_mask.bool().unsqueeze(-1).expand(-1, -1, max_len)
        match_label_col_mask = end_label_mask.bool().unsqueeze(-2).expand(-1, max_len, -1)
        match_label_mask = match_label_row_mask & match_label_col_mask
        match_label_mask = torch.triu(match_label_mask, 0)  # start should be less equal to end
        float_match_label_mask = match_label_mask.view(batch_size, -1).float()
        match_loss = loss_bce(span_logits.view(batch_size, -1), match_labels.view(batch_size, -1).float())
        match_loss = match_loss * float_match_label_mask
        match_loss = match_loss.sum() / (float_match_label_mask.sum() + 1e-10)
        loss = (start_loss+end_loss+ match_loss)
        #loss = start_loss+end_loss
        loss = loss.mean()
        loss.backward()
        if fgm:
            fgm.attack()
            loss_adv, adv_logits = model(input_ids, att_mask, labels, start_ids, end_ids)
            loss_adv = loss_adv.mean()
            loss_adv.backward()
            fgm.restore()
        nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        batch_time_avg += time.time() - batch_start
        running_loss += loss.item()
        correct_preds += (correct_predictions_2(start_logits, start_ids) + correct_predictions_2(end_logits, end_ids))/2
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

