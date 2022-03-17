## Bert实体抽取模型
### 前言
这是一个基于**AlBert(以下Bert若不特别强调都是AlBert)** 模型实现的文本分类模型，其中把一些常用的效果提升的方法都进行了封装，可以一键关闭和打开。  
python环境：  
pytorch:    1.10  
transformers:   4.12.5  
numpy:  1.19.0  

#### 参考论文
- [A Unified MRC Framework for Named Entity Recognition](https://arxiv.org/abs/1910.11476)(MRC框架，主要优势在于可解决"嵌套"实体问题）

#### 参考代码
https://github.com/ShannonAI/mrc-for-flat-nested-ner

### 正文
目前已有的方法：  
1.是否使用LSTM(ps:使用LSTM时必须使用CRF)  
2.是否使用CRF  
3.embedding对抗训练 FGM  
4.滑动平均 EMA  
ps:5.还有基于**MRC**框架的特殊BERT模型，因为它在输出阶段不再是传统的BIO的一个token多分类形式，而是类似于一个QA任务的answer抽取，所以单独创建一个文件夹放这个项目。另外MRC代码对于显存占用特别高

推荐训练超参数：  
learning_rate=2e-5    
batch_size=16  
max_length=256(不超过512)  
attack_type='FGM'  
use_EMA=True  
CRF_lr = 6*learning_rate  
lr_decay = 0.5 0.5(当前epoch上dev的loss不再下降时生效)  
ps:可根据自身任务进行调整  

### 训练
python train_lstm_crf.py  

读取文件路径，以及模型保存路径等都写在conf/config.py中的，自己看一下就知道了。  
私人数据不方便公开，只放了几个样例供参考输入格式。

### 效果对比

| model | F1 | recall | prec |
| --- | --- | --- | --- |
| Bert | 0.9383 | 0.9429 | 0.9338 |
| Bert+CRF | 0.9485 | 0.9525 | 0.9446 |
| Bert+LSTM+CRF | 0.9503 | 0.9537 | 0.9469 |
| Bert+CRF+FGM | 0.9547 | 0.9591 | 0.9504 |
|Bert+MRC | 0.9490 | 0.9547| 0.9433 |
