# -*-coding:utf-8-*-
import torch as t
import torch.nn as nn
from torchcrf import CRF
from transformers import AlbertForSequenceClassification, BertForSequenceClassification, AlbertModel, BertForMaskedLM, AlbertForTokenClassification, AlbertConfig, BertForTokenClassification

class BilstmCRF(nn.Module):
    def __init__(self, config, device, use_crf = True):
        super(BilstmCRF, self).__init__()
        self.num_labels = config.num_labels
        #self.bilstm = nn.LSTM(input_size=config.embedding_size, hidden_size=config.lstm_size, num_layers=1, batch_first=False, dropout=0.1, bidirectional=True)
        self.bilstm = nn.GRU(input_size=config.embedding_size, hidden_size=config.lstm_size, num_layers=1, batch_first=False, dropout=0.1, bidirectional=True)
        self.classifier = nn.Linear(config.lstm_size*2, config.num_labels)
        self.crf = CRF(config.num_labels, batch_first = True)
        self.dropout = nn.Dropout(0.2)
        self.use_crf = use_crf
        self.device = device
        
    def forward(self, hidden_states, att_mask, label_ids):
        #hidden_states, pooled_output = self.bert(input_ids=input_ids, attention_mask=att_mask)[:2]
        hidden_states_trans = hidden_states.permute(1,0,2)
        lstm_out, (h_n,c_n) = self.bilstm(hidden_states_trans)
        lstm_out = self.dropout(lstm_out.permute(1,0,2))
        logits = self.classifier(lstm_out)
        if self.use_crf:
            mask = att_mask.type(t.uint8)
            loss = -self.crf(logits, label_ids, mask, reduction='token_mean')
        else:
            loss = None 
            labels=label_ids
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                # Only keep active parts of the loss
                if att_mask is not None:
                    active_loss = att_mask.view(-1) == 1 
                    active_logits = logits.view(-1, self.num_labels)
                    active_labels = t.where(
                        active_loss, labels.view(-1), t.tensor(loss_fct.ignore_index).type_as(labels)
                    )    
                    loss = loss_fct(active_logits, active_labels)
                else:
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return loss, logits

class AlBertCRFModel(nn.Module):
    def __init__(self, config, use_crf = True):
        super(AlBertCRFModel, self).__init__()
        Config = AlbertConfig.from_pretrained(config.model_name)
        Config.attention_probs_dropout_prob = 0.1
        self.bert = AlbertModel.from_pretrained(config.model_name, config=Config)  # /bert_pretrai
        self.num_labels = config.num_label
        self.classifier = nn.Linear(config.embedding_size, config.num_label)
        self.crf = CRF(config.num_label, batch_first = True)
        self.dropout = nn.Dropout(0.1)
        self.use_crf = use_crf
        
        for param in self.bert.parameters():
            param.requires_grad = True
        
    def forward(self, input_ids, att_mask, label_ids):
        hidden_states, pooled_output = self.bert(input_ids=input_ids, attention_mask=att_mask)[:2]
        hidden_states = self.dropout(hidden_states)
        logits = self.classifier(hidden_states)
        if self.use_crf:
            #mask = att_mask.type(t.bool)
            loss = (-1) * self.crf(logits, label_ids, att_mask, reduction='token_mean')
        else:
            loss = None 
            labels=label_ids
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                # Only keep active parts of the loss
                if att_mask is not None:
                    active_loss = att_mask.view(-1) == 1 
                    active_logits = logits.view(-1, self.num_labels)
                    active_labels = t.where(
                        active_loss, labels.view(-1), t.tensor(loss_fct.ignore_index).type_as(labels)
                    )    
                    loss = loss_fct(active_logits, active_labels)
                else:
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return loss, logits

class AlBertLSTMModel(nn.Module):
    def __init__(self, config, use_crf = True):
        super(AlBertLSTMModel, self).__init__()
        Config = AlbertConfig.from_pretrained(config.model_name)
        Config.attention_probs_dropout_prob = 0.1
        self.bert = AlbertModel.from_pretrained(config.model_name, config=Config)  # /bert_pretrai
        self.num_labels = config.num_label
        #self.bilstm = nn.LSTM(input_size=config.embedding_size, hidden_size=config.lstm_size, num_layers=2, batch_first=True, bidirectional=True)
        self.bilstm = nn.GRU(input_size=config.embedding_size, hidden_size=config.lstm_size, num_layers=2, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(config.lstm_size*2, config.num_label)
        self.crf = CRF(config.num_label, batch_first = True)
        self.dropout = nn.Dropout(0.1)
        self.use_crf = use_crf
        
        for param in self.bert.parameters():
            param.requires_grad = True
        
    def forward(self, input_ids, att_mask, label_ids):
        hidden_states, pooled_output = self.bert(input_ids=input_ids, attention_mask=att_mask)[:2]
        hidden_states = self.dropout(hidden_states)
        #lstm_out, (h_n,c_n) = self.bilstm(hidden_states)
        lstm_out, h_n = self.bilstm(hidden_states)
        lstm_out = self.dropout(lstm_out)
        logits = self.classifier(lstm_out)
        if self.use_crf:
            #mask = att_mask.type(t.bool)
            loss = (-1) * self.crf(logits, label_ids, att_mask, reduction='token_mean')
        else:
            loss = None 
            labels=label_ids
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                # Only keep active parts of the loss
                if att_mask is not None:
                    active_loss = att_mask.view(-1) == 1 
                    active_logits = logits.view(-1, self.num_labels)
                    active_labels = t.where(
                        active_loss, labels.view(-1), t.tensor(loss_fct.ignore_index).type_as(labels)
                    )    
                    loss = loss_fct(active_logits, active_labels)
                else:
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return loss, logits

class AlBertLSTMModelPredict(nn.Module):
    def __init__(self, config, use_crf = True):
        super(AlBertLSTMModelPredict, self).__init__()
        Config = AlbertConfig.from_pretrained(config.model_name)
        Config.attention_probs_dropout_prob = 0.1
        self.bert = AlbertModel.from_pretrained(config.model_name, config=Config)  # /bert_pretrai
        self.num_labels = config.num_label
        #self.bilstm = nn.LSTM(input_size=config.embedding_size, hidden_size=config.lstm_size, num_layers=2, batch_first=True, dropout=0.3, bidirectional=True)
        self.bilstm = nn.LSTM(input_size=config.embedding_size, hidden_size=config.lstm_size, num_layers=2, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(config.lstm_size*2, config.num_label)
        self.crf = CRF(config.num_label, batch_first = True)
        self.dropout = nn.Dropout(0.1)
        self.use_crf = use_crf
        
        for param in self.bert.parameters():
            param.requires_grad = True
        
    def forward(self, input_ids, att_mask):
        hidden_states, pooled_output = self.bert(input_ids=input_ids, attention_mask=att_mask)[:2]
        hidden_states = self.dropout(hidden_states)
        #hidden_states_trans = hidden_states.permute(1,0,2)
        lstm_out, (h_n,c_n) = self.bilstm(hidden_states)
        lstm_out = self.dropout(lstm_out)
        logits = self.classifier(lstm_out)
        return logits

class AlBertNERModel(nn.Module):
    def __init__(self, config):
        super(AlBertNERModel, self).__init__()
        self.bert = AlbertForTokenClassification.from_pretrained(config.model_name, num_labels=config.num_label)  # /bert_pretrain
        for param in self.bert.parameters():
            param.requires_grad = True 
        
    def forward(self, input_ids, att_mask, label_ids):
        loss, logits, hidden_states = self.bert(input_ids=input_ids, attention_mask=att_mask,
                                                labels=label_ids,
                                                output_hidden_states=True)[:3]

        #probabilities = t.softmax(logits, dim=-1)
        return loss, logits

class BertNERModel(nn.Module):
    def __init__(self, config):
        super(BertNERModel, self).__init__()
        self.bert = BertForTokenClassification.from_pretrained(config.model_name, num_labels=config.num_label)  # /bert_pretrain
        for param in self.bert.parameters():
            param.requires_grad = True 
        
    def forward(self, input_ids, att_mask, label_ids):
        loss, logits, hidden_states = self.bert(input_ids=input_ids, attention_mask=att_mask,
                                                labels=label_ids,
                                                output_hidden_states=True)[:3]
        return loss, logits

class BertMLM(nn.Module):
    def __init__(self, config):
        super(BertMLM, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained(config.model_name, num_labels=2)  # /bert_pretrain
        for param in self.bert.parameters():
            param.requires_grad = True  

    def forward(self, input_ids, att_mask, label_ids):
        outputs = self.bert(input_ids = input_ids, attention_mask=att_mask, labels=label_ids)
        loss = outputs.loss
        #logits = outputs.logits
        return loss

class AlBertModelCNNMult(nn.Module):
    def __init__(self, config):
        super(AlBertModelCNNMult, self).__init__()
        #self.bert = AlbertForSequenceClassification.from_pretrained(config.model_name, num_labels=2)
        self.bert = AlbertModel.from_pretrained(config.model_name, num_labels=2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(config.dropout_rate)
        self.convs1 = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_channels=config.embedding_size,
                                    out_channels=config.out_channels,
                                    kernel_size=h),
                          nn.BatchNorm1d(config.out_channels),
                          nn.ReLU(),
                          nn.MaxPool1d(kernel_size=config.max_text_len - h + 1), )
            for h in config.kernel_size
        ])
        self.cnn_merge = nn.Sequential(nn.Conv1d(in_channels=config.out_channels * len(config.kernel_size), 
                                                out_channels = config.out_channels*len(config.kernel_size),
                                                kernel_size =2),
                                      nn.BatchNorm1d(config.out_channels *len(config.kernel_size)),
                                      nn.ReLU(),
                                      nn.MaxPool1d(4-2+1), )

        self.merge_features = nn.Linear(config.out_channels * len(config.kernel_size) + config.extra_feature_num,
                                        config.out_channels)
        self.classify = nn.Linear(config.out_channels, 2)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, input_ids, att_mask, token_type_ids, labels, course_tensor):
        out_list = []
        new_input_ids = input_ids.permute(1,0,2)
        new_att_mask = att_mask.permute(1,0,2)
        new_token_type_ids = token_type_ids.permute(1,0,2)
        #print(new_input_ids.shape)
        for i in range(input_ids.shape[1]):
            input_ids_i = new_input_ids[i].view(-1, input_ids.shape[2])
            att_mask_i = new_att_mask[i].view(-1, att_mask.shape[2])
            token_type_ids_i = new_token_type_ids[i].view(-1, token_type_ids.shape[2])
            seq_out, pooled_out = self.bert(input_ids=input_ids_i, attention_mask=att_mask_i, token_type_ids=token_type_ids_i)[:2]
            #print(pooled_out.shape)
#           loss, logits, hidden_states = self.bert(input_ids=input_ids_i, attention_mask=att_mask_i,
#                                                    token_type_ids=token_type_ids_i, labels=labels,
#                                                    output_hidden_states=True)[:3]
#
            #embed_x1 = hidden_states[-1].permute(0, 2, 1)
            #out1 = [conv(embed_x1) for conv in self.convs1]
            #out1 = t.cat(out1, dim=1)
            #out1 = out1.view(-1, out1.size(1))
            out_list.append(pooled_out)
        output = t.stack(out_list, dim=1)
        #print(output.shape)
        output = self.cnn_merge(output.permute(0,2,1))
        output = output.view(-1, output.size(1))
        #print(output.shape)
        output = t.cat((output, course_tensor), 1)
        #print(output.shape)
        output = self.merge_features(output)
        #print(output.shape)
        output = self.classify(output)
        #print(output.shape)
        probabilities = t.softmax(output, dim=-1)
        loss = self.loss_fct(output.view(-1, 2), labels.view(-1))
        return loss, output, probabilities

class AlBertModelCNN(nn.Module):
    def __init__(self, config):
        super(AlBertModelCNN, self).__init__()
        self.bert = AlbertForSequenceClassification.from_pretrained(config.model_name, num_labels=2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(config.dropout_rate)
        self.convs1 = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_channels=config.embedding_size,
                                    out_channels=config.out_channels,
                                    kernel_size=h),
                          nn.BatchNorm1d(config.out_channels),
                          nn.ReLU(),
                          nn.MaxPool1d(kernel_size=config.max_text_len - h + 1), )
            for h in config.kernel_size
        ])
        self.merge_features = nn.Linear(config.out_channels * len(config.kernel_size) + config.extra_feature_num,
                                        config.out_channels)
        self.classify = nn.Linear(config.out_channels, 2)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, input_ids, att_mask, token_type_ids, labels, course_tensor):
        loss, logits, hidden_states = self.bert(input_ids=input_ids, attention_mask=att_mask,
                                                token_type_ids=token_type_ids, labels=labels,
                                                output_hidden_states=True)[:3]

        embed_x1 = hidden_states[-1].permute(0, 2, 1)
        out1 = [conv(embed_x1) for conv in self.convs1]
        out1 = t.cat(out1, dim=1)
        out1 = out1.view(-1, out1.size(1))
        output = t.cat((out1, course_tensor), 1)
        output = self.merge_features(output)
        output = self.classify(output)
        probabilities = t.softmax(output, dim=-1)
        loss = self.loss_fct(output.view(-1, 2), labels.view(-1))
        return loss, output, probabilities

