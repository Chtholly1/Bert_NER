import sys
import argparse

model_name = './resource/base_models/base_albert/'
#model_name = './base_roberta'
use_crf = True
max_len = 128

weight_decay=0.01

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default=model_name, type=str)
parser.add_argument("--train_file", type=str, help="训练集文件", default='resource/data/hot_word_2022022110_f.txt')
parser.add_argument("--val_file", type=str, help="验证集文件", default='resource/data/hot_word_2022022110_f.txt')
parser.add_argument('--result_dir', type=str, default='./resource/result/')
parser.add_argument('--model', type=str, default='./resource/models/best.pth.tar.2022012013')
parser.add_argument("--target_dir", default="./resource/models/", type=str)
parser.add_argument("--max_length", default=max_len, type=int, help="截断的最大长度")
parser.add_argument("--epochs", default=20, type=int, help="最多训练多少个epoch")
parser.add_argument("--batch_size", default=16, type=int)
parser.add_argument("--lr", default=2, type=int)
parser.add_argument("--max_grad_norm", default=10.0, type=int)
parser.add_argument("--patience", default=5, type=int)
parser.add_argument("--gpu_index", default=1, type=int)
args = parser.parse_args()

learning_rate = int(args.lr)*1e-5

deleted_labels = {'positive', 'negative', 'neutral'}
map_labels = {'insurance':'car_price', 'final_price':'car_price', 'budget':'car_price', 'discount':'car_price', 'car_price_other':'car_price', 'loan':'car_price', 'offer':'car_price'}

label_2_ids = {'O': 0, 'B-car_type': 1, 'I-car_type': 2, 'B-control': 3, 'I-control': 4, 'B-energy_consumption': 5, 'I-energy_consumption': 6, 'B-config': 7, 'I-config': 8, 'B-car_name': 9, 'I-car_name': 10, 'B-power': 11, 'I-power': 12, 'B-color': 13, 'I-color': 14, 'B-car_price': 15, 'I-car_price': 16, 'B-car_price_num': 17, 'I-car_price_num': 18, 'B-comfort': 19, 'I-comfort': 20, 'B-appearance': 21, 'I-appearance': 22, 'B-interior': 23, 'I-interior': 24, 'B-space': 25, 'I-space': 26, 'B-car_series': 27, 'I-car_series': 28, 'B-car_use': 29, 'I-car_use': 30}

eng_2_chn = {'car_type':'车型', 'control':'操控', 'energy_consumption':'能耗', 'config':'配置', 'car_name':'车名', 'power':'动力', 'color':'颜色', 'car_price':'价格', 'car_price_num':'具体价格', 'comfort':'舒适', 'appearance':'外观', 'interior':'内饰', 'space': '空间', 'car_series':'车系', 'car_use':'用途'}
chn_2_eng = {v: k for k, v in eng_2_chn.items()}

class Config:
    def __init__(self):
        self.dropout_rate = 0.2
        self.fc_size1 = 256
        self.final_out_size = 128
        self.vocab_size = 0
        self.embedding_size = 768
        self.out_channels = 256
        self.kernel_size = [2, 3, 4]
        self.max_text_len = max_len
        #self.num_label = label_nums
        self.lstm_size = 256
        self.model_name = model_name
