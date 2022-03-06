#coding:utf-8
import json

train_file='./data/2022010717'
#dev_file=''

label_dict = dict()
idx = 1
with open(train_file) as f:
    for line in f:
        info = json.loads(line.strip())
        for item in info['annotations']:
            if item['label'] not in label_dict:
                label_dict[item['label']] = idx
                idx += 1

real_label_dict = {'O':0}
index = 1
for key in label_dict:
    real_label_dict['B-' + key] = index
    index += 1
    real_label_dict['I-' + key] = index
    index += 1
print(real_label_dict)
