import joblib
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.svm import SVC
import seaborn as sns

df = pd.read_csv('../data/lung-cancer.csv')

if __name__ == '__main__':
    print(df.info())
    df['GENDER'] = df['GENDER'].map({'M': 1, 'F': 0})
    df['LUNG_CANCER'] = df['LUNG_CANCER'].map({'YES': 1, 'NO': 0})
    # 分割特征和目标变量
    X = df.drop('LUNG_CANCER', axis=1)
    y = df['LUNG_CANCER']

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
        ('transform', PowerTransformer()),
        ('scaler', StandardScaler()),
        ('base', SVC(probability=True))
    ])
    model.fit(X_train, y_train)

    # 在测试集上进行预测
    y_pred = model.predict(X_test)

    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    print("准确率：", accuracy)

    # joblib.dump(model, '../model/lung-cancer.dat', compress=3)
    probability = model.predict_proba(X_test)
    # print(probability)
