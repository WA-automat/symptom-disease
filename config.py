epochs = 250  # 未触发早停止的最大epochs
train_batch_size = 32  # 训练批大小
test_batch_size = 32  # 测试批大小
learning_rate = 5e-3  # 学习率
step_size = 5  # 学习率降低步长
gamma = 0.1  # 学习率降低倍率
embedding_dim = 80  # 词向量维度
hidden_dim = 256  # lstm隐藏层数
num_layers = 2  # lstm层数
dropout = 0.5  # dropout概率
early_stopping_patience = 10  # 早停止次数
train_path = './data/symptom-disease-train-dataset-zh-new.csv'  # 训练集路径
test_path = './data/symptom-disease-test-dataset-zh-new.csv'  # 测试集路径
mapping_path = './data/mapping-zh-new.json'  # 映射路径
vocab_path = './data/vocab.json'  # 词表路径
stopwords_path = './data/stopwords.txt'  # 停用词路径
save_path = 'model/best_model.pth'  # 保存路径
load_path = 'model/best_lstm_attention_model.pth'  # 加载路径
