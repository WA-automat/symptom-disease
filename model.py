import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM_with_Attention(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers=1, dropout=0.5):
        super(LSTM_with_Attention, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.bn1 = nn.BatchNorm1d(hidden_dim)  # 添加批标准化层
        self.dropout1 = nn.Dropout(dropout)
        self.bn2 = nn.BatchNorm1d(hidden_dim)  # 添加批标准化层
        self.dropout2 = nn.Dropout(dropout)
        self.bn3 = nn.BatchNorm1d(hidden_dim)  # 添加批标准化层
        self.dropout3 = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.attention = nn.Linear(hidden_dim, 1)  # 注意力机制

    def forward(self, inputs):
        # embedding后使用LSTM
        inputs = self.embedding(inputs)
        lstm_out, _ = self.lstm(inputs)

        lstm_out = self.bn1(lstm_out)
        lstm_out = F.relu(lstm_out)
        lstm_out = self.dropout1(lstm_out)

        # 应用注意力机制
        attention_scores = self.attention(lstm_out).squeeze(-1)
        attention_weights = F.softmax(attention_scores, dim=1).unsqueeze(2)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)

        # Batch Normalization层：加速收敛
        x = self.bn2(context_vector)
        x = F.relu(x)
        x = self.dropout2(x)

        # 定义残差连接
        residual = lstm_out[:, -1, :]  # 选择最后一个时间步的输入作为残差项
        x = x + residual  # 将残差项与当前结果相加

        # Batch Normalization层：加速收敛
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout3(x)

        # 全连接层
        x = self.fc(x)
        return x
