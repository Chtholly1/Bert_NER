from transformers import BertTokenizer, BertForQuestionAnswering, AlbertForQuestionAnswering
import torch

start_label_mask = torch.zeros(2,3)
start_label_mask[0,0] = 1

row = start_label_mask.bool().unsqueeze(-1).expand(-1, -1, 3)
col = start_label_mask.bool().unsqueeze(-2).expand(-1, 3, -1)
print(row & col)
