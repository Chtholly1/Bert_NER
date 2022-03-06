import sys
import argparse

model_name = './resource/base_models/base_albert/'
#model_name = './base_roberta'
use_crf = True
max_len = 128

weight_decay=0.01

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default=model_name, type=str)
parser.add_argument("--train_file", type=str, help="训练集文件", default='./resource/data/total.txt')
parser.add_argument("--val_file", type=str, help="验证集文件", default='./resource/data/total.txt')
parser.add_argument('--result_dir', type=str, default='./resource/result/')
parser.add_argument('--model', type=str, default='resource/models/best.pth.tar.2022022409')
parser.add_argument("--target_dir", default="./resource/models/", type=str)
parser.add_argument("--max_length", default=max_len, type=int, help="截断的最大长度")
parser.add_argument("--epochs", default=20, type=int, help="最多训练多少个epoch")
parser.add_argument("--batch_size", default=16, type=int)
parser.add_argument("--lr", default=2, type=int)
parser.add_argument("--sd_fac", default=5, type=int)
parser.add_argument("--lr_mag", default=7, type=int)
parser.add_argument("--max_grad_norm", default=10.0, type=int)
parser.add_argument("--patience", default=5, type=int)
parser.add_argument("--attack_type", default='FGM', type=str)
parser.add_argument("--use_EMA", default=True, type=int)
parser.add_argument("--gpu_index", default=1, type=int)
args = parser.parse_args()

learning_rate = int(args.lr)*1e-5
sd_factor = int(args.sd_fac)*0.1
lr_mag = int(args.lr_mag)

deleted_labels = {'positive', 'negative', 'neutral'}
map_labels = {'insurance':'car_price', 'final_price':'car_price', 'budget':'car_price', 'discount':'car_price', 'car_price_other':'car_price', 'loan':'car_price', 'offer':'car_price'}

label_2_ids = {'O': 0, 'B-energy_consumption': 1, 'I-energy_consumption': 2, 'B-car_price': 3, 'I-car_price': 4, 'B-car_price_num': 5, 'I-car_price_num': 6, 'B-control': 7, 'I-control': 8, 'B-car_type': 9, 'I-car_type': 10, 'B-color': 11, 'I-color': 12, 'B-comfort': 13, 'I-comfort': 14, 'B-appearance': 15, 'I-appearance': 16, 'B-car_name': 17, 'I-car_name': 18, 'B-config': 19, 'I-config': 20, 'B-interior': 21, 'I-interior': 22, 'B-power': 23, 'I-power': 24, 'B-space': 25, 'I-space': 26, 'B-car_series': 27, 'I-car_series': 28, 'B-car_use': 29, 'I-car_use': 30}

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
