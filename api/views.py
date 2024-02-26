from flask import Blueprint, request
from api.ResponseResult import ResponseResult
from eval import predict, rev_mapping

blue = Blueprint('api', __name__)


@blue.route('/predict/disease', methods=['GET'])
def predict_disease_api():
    s = request.args.get("s")
    probability, indices = predict(s)
    response = ResponseResult(code=200, msg='预测成功',
                              data={'disease': rev_mapping[indices.item()], 'probability': probability.item()}).toDict()
    return response
