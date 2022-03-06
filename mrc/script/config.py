import argparse

model_name = './base_albert'
#model_name = './base_roberta'
use_crf = True
max_len = 128

weight_decay=0.01
learning_rate = 2e-5

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default=model_name, type=str)
parser.add_argument("--train_file", type=str, help="训练集文件", default='./data/hot_word_2022011416_sf.txt')
parser.add_argument("--val_file", type=str, help="验证集文件", default='./data/hot_word_2022011416_sf.txt')
parser.add_argument('--result_dir', type=str, default='./result/')
parser.add_argument('--model', type=str, default='./models/best.pth.tar.2022011417')
parser.add_argument("--target_dir", default="models", type=str)
parser.add_argument("--max_length", default=max_len, type=int, help="截断的最大长度")
parser.add_argument("--epochs", default=20, type=int, help="最多训练多少个epoch")
parser.add_argument("--batch_size", default=8, type=int)
parser.add_argument("--lr", default=learning_rate, type=int)
parser.add_argument("--max_grad_norm", default=10.0, type=int)
parser.add_argument("--patience", default=5, type=int)
parser.add_argument("--gpu_index", default=0, type=int)
args = parser.parse_args()

#label_2_ids = {'O': 0, 'B-config': 1, 'I-config': 2, 'B-power': 3, 'I-power': 4, 'B-car_name': 5, 'I-car_name': 6, 'B-control': 7, 'I-control': 8, 'B-car_type': 9, 'I-car_type': 10, 'B-appearance': 11, 'I-appearance': 12, 'B-space': 13, 'I-space': 14, 'B-confort': 15, 'I-confort': 16, 'B-energy_consumption': 17, 'I-energy_consumption': 18, 'B-car_price_num': 19, 'I-car_price_num': 20, 'B-car_series': 21, 'I-car_series': 22, 'B-final_price': 23, 'I-final_price': 24, 'B-offer': 25, 'I-offer': 26, 'B-discount': 27, 'I-discount': 28, 'B-interior': 29, 'I-interior': 30, 'B-car_price_other': 31, 'I-car_price_other': 32, 'B-loan': 33, 'I-loan': 34, 'B-positive': 35, 'I-positive': 36, 'B-car_use': 37, 'I-car_use': 38, 'B-insurance': 39, 'I-insurance': 40, 'B-budget': 41, 'I-budget': 42, 'B-comfort': 43, 'I-comfort': 44, 'B-negative': 45, 'I-negative': 46, 'B-neutral': 47, 'I-neutral': 48, 'B-car_sries': 49, 'I-car_sries': 50, 'B-color': 51, 'I-color': 52}
#ids_2_label = {v:k for k,v in label_2_ids.items()}
#label_nums = len(label_2_ids)
#num_labels = label_nums

deleted_labels = {'positive', 'negative', 'neutral'}
map_labels = {'insurance':'car_price', 'final_price':'car_price', 'budget':'car_price', 'discount':'car_price', 'car_price_other':'car_price', 'loan':'car_price', 'offer':'car_price'}

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
