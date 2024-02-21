import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM_TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers=1, dropout=0.5):
        super(LSTM_TextClassificationModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.bn = nn.BatchNorm1d(hidden_dim)  # 添加批标准化层
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.attention = nn.Linear(hidden_dim, 1)  # 注意力机制

    def forward(self, inputs):
        # embedding后使用LSTM
        inputs = self.embedding(inputs)
        lstm_out, _ = self.lstm(inputs)

        # 应用注意力机制
        attention_scores = self.attention(lstm_out).squeeze(-1)
        attention_weights = F.softmax(attention_scores, dim=1).unsqueeze(2)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)

        # Batch Normalization层：加速收敛
        x = self.bn(context_vector)
        x = F.relu(x)
        x = self.dropout(x)

        # 全连接层
        x = self.fc(x)
        return x


if __name__ == '__main__':
    tensor = torch.tensor([[1, 2, 3], [3, 4, 5]])
    model = LSTM_TextClassificationModel(10, 10, 5, 2)
    print(model(tensor))
