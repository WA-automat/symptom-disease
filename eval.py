import json
import jieba
import torch
from config import *
from model import LSTM_with_Attention

# 判断是否有cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 读取映射并获取反映射
with open(mapping_path, "r", encoding="utf-8") as f:
    mapping = json.load(f)
    rev_mapping = {value: key for key, value in mapping.items()}

# 读取词表
with open(vocab_path, "r", encoding="utf-8") as f:
    vocabs = json.load(f)

# 加载停用词
with open(stopwords_path, "r", encoding="utf-8") as f:
    stopwords = f.readlines()
    stopwords = [line.rstrip('\n') for line in stopwords]

# 加载模型
model = LSTM_with_Attention(vocab_size=len(vocabs), embedding_dim=embedding_dim,
                            hidden_dim=hidden_dim, output_dim=len(mapping),
                            num_layers=num_layers, dropout=dropout)
model.load_state_dict(torch.load(load_path, map_location=device))


def predict(s):
    global model
    model = model.to(device)
    model.eval()
    x = ['<bos>'] + list(jieba.cut(s)) + ['<eos>']
    x = torch.tensor([vocabs[word] if word in vocabs else vocabs['<unk>'] for word in x])
    x = x.to(device)
    x = model(x.unsqueeze(0))[0]
    max_values, max_indices = torch.max(x, dim=0)
    return max_values, max_indices


if __name__ == '__main__':
    print(predict("身上痒，皮肤有红点"))
