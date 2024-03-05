import joblib
import pandas as pd
from flask import Blueprint, request
from api.ResponseResult import ResponseResult
from eval import predict, rev_mapping

blue = Blueprint('api', __name__)

# 加载数据
heart_df = pd.read_csv('./data/heart.csv')
diabetes_df = pd.read_csv('./data/diabetes.csv')
lung_cancer_df = pd.read_csv('./data/lung-cancer.csv')
lung_cancer_df['GENDER'] = lung_cancer_df['GENDER'].map({'M': 1, 'F': 0})
lung_cancer_df['LUNG_CANCER'] = lung_cancer_df['LUNG_CANCER'].map({'YES': 1, 'NO': 0})
advice_df = pd.read_excel('./data/病症及其对应的建议.xlsx', sheet_name='Sheet1')

# 加载模型
heart = joblib.load('./model/heart.dat')
diabetes = joblib.load('./model/diabetes.dat')
lung_cancer = joblib.load('./model/lung-cancer.dat')


@blue.route('/predict/disease', methods=['POST'])
def predict_disease_api():
    """
    文本分类接口
    :return: 预测值与概率
    """
    s = request.get_json().get("s")
    probability, indices = predict(s)
    return ResponseResult(code=200, msg='预测成功',
                          data={
                              'disease': rev_mapping[indices.item()],
                              'probability': probability.item(),
                              'advice_xi': advice_df.loc[advice_df['索引'] == indices.item(), '西医建议'].values[0],
                              'advice_zhong': advice_df.loc[advice_df['索引'] == indices.item(), '中医建议'].values[0]
                          }).toDict()


@blue.route('/predict/heart', methods=['POST'])
def predict_heart_api():
    """
    心脏病预测
    :return: 是否患上心脏病
    """
    mean = heart_df[heart_df['target'] == 1].drop(
        ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal', 'target'],
        axis=1).mean().to_dict()
    healthy_mean = heart_df[heart_df['target'] == 0].drop(
        ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal', 'target'],
        axis=1).mean().to_dict()
    data = request.get_json()['data']
    result = heart.predict([data])[0]
    probability = heart.predict_proba([data])[0]
    return ResponseResult(code=200, msg='预测成功',
                          data={
                              'result': int(result),
                              'probability': probability.tolist(),
                              'advice_xi': advice_df.loc[advice_df['病症'] == '心脏病', '西医建议'].values[0],
                              'advice_zhong': advice_df.loc[advice_df['病症'] == '心脏病', '中医建议'].values[0],
                              'mean': mean,
                              'healthy_mean': healthy_mean,
                              'personal': [data[0], data[11], data[4], data[9], data[7], data[3]]
                          }).toDict()


@blue.route('/predict/diabetes', methods=['POST'])
def predict_diabetes_api():
    """
    糖尿病预测
    :return: 是否患上糖尿病
    """
    mean = diabetes_df[diabetes_df["Outcome"] == 1].drop(['Outcome'], axis=1).mean().to_dict()
    healthy_mean = diabetes_df[diabetes_df["Outcome"] == 0].drop(['Outcome'], axis=1).mean().to_dict()
    data = request.get_json()['data']
    result = diabetes.predict([data])[0]
    probability = diabetes.predict_proba([data])[0]
    return ResponseResult(code=200, msg='预测成功',
                          data={
                              'result': int(result),
                              'probability': probability.tolist(),
                              'advice_xi': advice_df.loc[advice_df['病症'] == '糖尿病', '西医建议'].values[0],
                              'advice_zhong': advice_df.loc[advice_df['病症'] == '糖尿病', '中医建议'].values[0],
                              'mean': mean,
                              'healthy_mean': healthy_mean,
                              'personal': [data[7], data[5], data[2], data[6], data[1], data[4], data[0], data[3]]
                          }).toDict()


@blue.route('/predict/lung-cancer', methods=['POST'])
def predict_lung_cancer_api():
    """
    肺癌预测
    :return: 是否患上肺癌
    """
    mean_age = lung_cancer_df[lung_cancer_df["LUNG_CANCER"] == 1]['AGE'].mean()
    age_low = lung_cancer_df[lung_cancer_df["LUNG_CANCER"] == 1]['AGE'].quantile(0.25)
    age_high = lung_cancer_df[lung_cancer_df["LUNG_CANCER"] == 1]['AGE'].quantile(0.75)
    healthy_mean_age = lung_cancer_df[lung_cancer_df["LUNG_CANCER"] == 0]['AGE'].mean()
    healthy_age_low = lung_cancer_df[lung_cancer_df["LUNG_CANCER"] == 0]['AGE'].quantile(0.25)
    healthy_age_high = lung_cancer_df[lung_cancer_df["LUNG_CANCER"] == 0]['AGE'].quantile(0.75)
    data = request.get_json()['data']
    result = lung_cancer.predict([data])[0]
    probability = lung_cancer.predict_proba([data])[0]
    return ResponseResult(code=200, msg='预测成功',
                          data={
                              'result': int(result),
                              'probability': probability.tolist(),
                              'advice_xi': advice_df.loc[advice_df['病症'] == '肺癌', '西医建议'].values[0],
                              'advice_zhong': advice_df.loc[advice_df['病症'] == '肺癌', '中医建议'].values[0],
                              'mean_age': mean_age,
                              'age_low': age_low,
                              'age_high': age_high,
                              'healthy_mean_age': healthy_mean_age,
                              'healthy_age_low': healthy_age_low,
                              'healthy_age_high': healthy_age_high,
                              'personal_age': data[7],
                              'age_source': [
                                  lung_cancer_df[lung_cancer_df["LUNG_CANCER"] == 1]['AGE'].values.tolist(),
                                  lung_cancer_df[lung_cancer_df["LUNG_CANCER"] == 0]['AGE'].values.tolist(),
                              ]
                          }).toDict()


@blue.route('/static/heart', methods=['GET'])
def static_heart():
    """
    心脏病静态内容
    :return:
    """
    mean = heart_df[heart_df['target'] == 1].drop(
        ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal', 'target'],
        axis=1).mean().to_dict()
    healthy_mean = heart_df[heart_df['target'] == 0].drop(
        ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal', 'target'],
        axis=1).mean().to_dict()
    heart_cls_df = heart_df[heart_df['target'] == 1]
    heart_cls_df = heart_cls_df[['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']]
    cls = {}
    for column in heart_cls_df.columns:
        cls[column] = heart_cls_df[column].value_counts(normalize=True).to_dict()
    return ResponseResult(code=200, msg='获取成功',
                          data={
                              'corr': heart_df.corr().values.tolist(),
                              'advice_xi': advice_df.loc[advice_df['病症'] == '心脏病', '西医建议'].values[0],
                              'advice_zhong': advice_df.loc[advice_df['病症'] == '心脏病', '中医建议'].values[0],
                              'ratio': cls,
                              'mean': mean,
                              'healthy_mean': healthy_mean,
                              'list_name': ['年龄', '性别', '胸痛类型', '静息血压', '血清胆固醇',
                                            '空腹血糖', '静息心电', '最大心率', '心绞痛类型', 'ST压低峰值',
                                            'ST段斜率', '主要血管数量', '心脏血流显像', '心脏病']
                          }).toDict()


@blue.route('/static/diabetes', methods=['GET'])
def static_diabetes():
    """
    糖尿病静态内容
    :return:
    """
    mean = diabetes_df[diabetes_df["Outcome"] == 1].drop(['Outcome'], axis=1).mean().to_dict()
    healthy_mean = diabetes_df[diabetes_df["Outcome"] == 0].drop(['Outcome'], axis=1).mean().to_dict()
    return ResponseResult(code=200, msg='获取成功',
                          data={
                              'corr': diabetes_df.corr().values.tolist(),
                              'advice_xi': advice_df.loc[advice_df['病症'] == '糖尿病', '西医建议'].values[0],
                              'advice_zhong': advice_df.loc[advice_df['病症'] == '糖尿病', '中医建议'].values[0],
                              'mean': mean,
                              'healthy_mean': healthy_mean,
                              'list_name': ['妊娠的次数', '血糖浓度', '血压大小', '皮肤厚度', '胰岛素浓度', 'BMI',
                                            '糖尿病谱系', '年龄', '糖尿病']
                          }).toDict()


@blue.route('/static/lung-cancer', methods=['GET'])
def static_lung_cancer():
    """
    肺癌静态内容
    :return:
    """
    mean_age = lung_cancer_df[lung_cancer_df["LUNG_CANCER"] == 1]['AGE'].mean()
    healthy_mean_age = lung_cancer_df[lung_cancer_df["LUNG_CANCER"] == 0]['AGE'].mean()
    lung_cancer_cls_df = lung_cancer_df[lung_cancer_df["LUNG_CANCER"] == 1]
    lung_cancer_cls_df = lung_cancer_cls_df.drop(["AGE", "LUNG_CANCER"], axis=1)
    cls = {}
    for column in lung_cancer_cls_df.columns:
        cls[column] = lung_cancer_cls_df[column].value_counts(normalize=True).to_dict()
    return ResponseResult(code=200, msg='获取成功',
                          data={
                              'corr': lung_cancer_df.corr().values.tolist(),
                              'advice_xi': advice_df.loc[advice_df['病症'] == '肺癌', '西医建议'].values[0],
                              'advice_zhong': advice_df.loc[advice_df['病症'] == '肺癌', '中医建议'].values[0],
                              'ratio': cls,
                              'mean_age': mean_age,
                              'healthy_mean_age': healthy_mean_age
                          }).toDict()
