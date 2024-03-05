import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA, KernelPCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, RidgeClassifierCV, SGDClassifier
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier, BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler, PowerTransformer, MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import joblib
import seaborn as sns

df = pd.read_csv('../data/diabetes.csv')

if __name__ == '__main__':
    print(df.info())
    # 分割特征和目标变量
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    correlations = df.corr()
    # 绘制相关性矩阵的热力图
    sns.heatmap(correlations, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.show()

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # 创建并拟合 Logistic Regression 模型
    model = Pipeline([
        ('transform', PowerTransformer()),
        ('scaler', StandardScaler()),
        ('base', SVC(probability=True))
    ])
    model.fit(X_train, y_train)

    # 计算准确率
    accuracy = accuracy_score(y_train, model.predict(X_train))
    print("训练集准确率：", accuracy)

    # 在测试集上进行预测
    y_pred = model.predict(X_test)

    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    print("准确率：", accuracy)

    # joblib.dump(model, '../model/diabetes.dat', compress=3)
    # probability = model.predict_proba(X_test)
    # print(probability)
