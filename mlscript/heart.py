import joblib
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer
import seaborn as sns

df = pd.read_csv('../data/heart.csv')

if __name__ == '__main__':
    print(df.info())
    # 分割特征和目标变量
    X = df.drop('target', axis=1)
    y = df['target']

    correlations = df.corr()
    # 绘制相关性矩阵的热力图
    sns.heatmap(correlations, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.show()

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 创建并拟合 Logistic Regression 模型
    model = Pipeline([
        ('transform', SplineTransformer()),
        ('base', LogisticRegression(max_iter=1000))
    ])
    model.fit(X_train, y_train)

    # 在测试集上进行预测
    y_pred = model.predict(X_test)

    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    print("准确率：", accuracy)

    # joblib.dump(model, '../model/heart.dat', compress=3)
