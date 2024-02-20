import json
import random
import time
from hashlib import md5

import pandas as pd
import requests

appid = '20240220001968436'
appkey = 'uKbYiPLuwyF4EsYYqjvC'

train_df = pd.read_csv('./data/symptom-disease-train-dataset.csv')
test_df = pd.read_csv('./data/symptom-disease-test-dataset.csv')

train_zh_df = pd.DataFrame()
test_zh_df = pd.DataFrame()


def make_md5(s, encoding='utf-8'):
    return md5(s.encode(encoding)).hexdigest()


def translate_func(text, source_lang='en', target_lang='zh'):
    salt = random.randint(32768, 65536)
    sign = make_md5(appid + text + str(salt) + appkey)
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    payload = {'appid': appid, 'q': text, 'from': source_lang, 'to': target_lang, 'salt': salt, 'sign': sign}
    start = time.time()
    url = "https://fanyi-api.baidu.com/api/trans/vip/translate"
    resp = requests.post(url, headers=headers, params=payload)
    trans = resp.json()
    end = time.time()
    total = end - start
    print(f"译文：{trans['trans_result'][0]['dst']}，用时:{total}")
    return trans['trans_result'][0]['dst']


if __name__ == '__main__':
    # with open('./data/mapping.json', 'r', encoding='utf-8') as f1, \
    #         open('mapping-zh.json', 'w', encoding='utf-8') as f2:
    #     mapping = json.load(f1)
    #     mapping_zh = {translate_func(key): value for key, value in mapping.items()}
    #     json.dump(mapping_zh, f2, ensure_ascii=False, indent=4)

    train_zh_df['text'] = train_df['text'].apply(lambda x: translate_func(x))
    test_zh_df['text'] = test_df['text'].apply(lambda x: translate_func(x))

    train_zh_df['label'] = train_df['label']
    test_zh_df['label'] = test_df['label']

    train_zh_df.to_csv('./data/symptom-disease-train-dataset-zh.csv')
    test_zh_df.to_csv('./data/symptom-disease-test-dataset-zh.csv')
