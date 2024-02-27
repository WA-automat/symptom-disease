import joblib
from flask import Blueprint, request
from api.ResponseResult import ResponseResult
from eval import predict, rev_mapping

blue = Blueprint('api', __name__)

heart = joblib.load('../model/heart.dat')
diabetes = joblib.load('../model/diabetes.dat')
lung_cancer = joblib.load('../model/lung-cancer.dat')


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
                              'probability': probability.item()
                          }).toDict()


@blue.route('/predict/heart', methods=['POST'])
def predict_heart_api():
    """
    心脏病预测
    :return:
    """
    data = request.get_json()
    res = heart.predict([data])
    return ResponseResult(code=200, msg='预测成功',
                          data={'result': res[0]})


@blue.route('/predict/disease', methods=['POST'])
def predict_disease_api():
    """
    糖尿病预测
    :return:
    """
    data = request.get_json()
    res = diabetes.predict([data])
    return ResponseResult(code=200, msg='预测成功',
                          data={'result': res[0]})


@blue.route('/predict/lung-cancer', methods=['POST'])
def predict_lung_cancer_api():
    """
    肺癌预测
    :return:
    """
    data = request.get_json()
    res = lung_cancer.predict([data])
    return ResponseResult(code=200, msg='预测成功',
                          data={'result': res[0]})
