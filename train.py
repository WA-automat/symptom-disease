import json
import jieba
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from config import *
from utils import *
from model import TextClassificationModel

# 判断是否有cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 读取映射并获取反映射
with open(mapping_path, "r", encoding="utf-8") as f:
    mapping = json.load(f)
    rev_mapping = {value: key for key, value in mapping.items()}

# 加载数据集
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# 读取词表
with open(vocab_path, "r", encoding="utf-8") as f:
    vocabs = json.load(f)

# 加载停用词
with open(stopwords_path, "r", encoding="utf-8") as f:
    stopwords = f.readlines()
    stopwords = [line.rstrip('\n') for line in stopwords]

# 模型、优化器与损失函数
model = TextClassificationModel(vocab_size=len(vocabs), embedding_dim=embedding_dim,
                                hidden_dim=hidden_dim, output_dim=len(mapping),
                                num_layers=num_layers, dropout=dropout)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()
scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

if __name__ == '__main__':
    # 包装为Dataset与DataLoader
    train_text = train_df['text'].values
    train_text = [['<bos>'] + [word for word in jieba.cut(sentence) if word not in stopwords] + ['<eos>'] for sentence
                  in train_text]
    test_text = test_df['text'].values
    test_text = [['<bos>'] + [word for word in jieba.cut(sentence) if word not in stopwords] + ['<eos>'] for sentence in
                 test_text]
    train_tensor = [torch.tensor([vocabs[word] for word in sentence]) for sentence in train_text]
    test_tensor = [torch.tensor([vocabs[word] if word in vocabs else vocabs['<unk>'] for word in sentence]) for sentence
                   in test_text]
    train_label = train_df['label'].values
    test_label = test_df['label'].values

    train_tensor = pad_sequence(train_tensor, batch_first=True, padding_value=vocabs['<pad>'])
    test_tensor = pad_sequence(test_tensor, batch_first=True, padding_value=vocabs['<pad>'])

    train_dataset = DiseaseDataset(train_tensor, train_label)
    test_dataset = DiseaseDataset(test_tensor, test_label)
    train_dataLoader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    test_dataLoader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True)

    # 早停止策略相关变量
    best_model_state_dict = None
    best_test_accuracy = 0.0
    no_improvement_count = 0

    # 训练并检验
    model = model.to(device)
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        total_correct_train = 0
        total_samples_train = 0
        for inputs, labels in train_dataLoader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

            predictions_train = outputs.argmax(dim=1)
            total_correct_train += (predictions_train == labels).sum().item()
            total_samples_train += labels.size(0)

        scheduler.step()

        # 在测试集上评估模型性能
        model.eval()
        total_test_loss = 0.0
        total_correct_test = 0
        total_samples_test = 0
        with torch.no_grad():
            for inputs, labels in test_dataLoader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_test_loss += loss.item()

                predictions_test = outputs.argmax(dim=1)
                total_correct_test += (predictions_test == labels).sum().item()
                total_samples_test += labels.size(0)

        train_loss = total_train_loss / len(train_dataLoader)
        test_loss = total_test_loss / len(test_dataLoader)
        train_accuracy = total_correct_train / total_samples_train * 100
        test_accuracy = total_correct_test / total_samples_test * 100

        print(f"Epoch {epoch + 1}, "
              f"Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, "
              f"Train Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%")

        # 早停止策略
        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            best_model_state_dict = model.state_dict()
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            if no_improvement_count >= early_stopping_patience:
                break

    # 保存模型
    torch.save(best_model_state_dict, save_path)
    print("Model saved successfully.")
