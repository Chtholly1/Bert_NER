#coding:utf-8
import sys
import json
import collections

def calc_ngram_freq(text, freq_dict, n):
    if len(text) < n:
        return
    for i in range(len(text)-n+1):
        ent = text[i:i+n]
        freq_dict[ent] += 1
    return

freq_dict_4 = collections.defaultdict(int)
freq_dict_3 = collections.defaultdict(int)
freq_dict_2 = collections.defaultdict(int)
freq_dict = collections.defaultdict(int)

for line in sys.stdin:
    info = json.loads(line.strip())
    text = info['text'].split('\t')[0]
    calc_ngram_freq(text, freq_dict, 4)
    calc_ngram_freq(text, freq_dict, 3)
    calc_ngram_freq(text, freq_dict, 2)

freq_sort_list = sorted([(key, val) for key, val in freq_dict.items()], key=lambda x:x[1], reverse=True)

for item in freq_sort_list:
    print("item:%s\tfreq:%d"%(item[0], item[1]))
