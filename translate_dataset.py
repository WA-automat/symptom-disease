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
    # # 映射文件翻译
    # with open('./data/mapping.json', 'r', encoding='utf-8') as f1, \
    #         open('mapping-zh.json', 'w', encoding='utf-8') as f2:
    #     mapping = json.load(f1)
    #     mapping_zh = {translate_func(key): value for key, value in mapping.items()}
    #     json.dump(mapping_zh, f2, ensure_ascii=False, indent=4)

    # 训练集翻译
    counter = 0
    for idx, row in train_df.iterrows():
        try:
            translated_text = translate_func(row['text'])
            train_zh_df.loc[idx, 'text'] = translated_text
            train_zh_df.loc[idx, 'label'] = row['label']

            counter += 1
            if counter == 10:
                train_zh_df.to_csv('./data/symptom-disease-train-dataset-zh.csv')
                counter = 0
        except Exception as e:
            print(f"翻译第{idx}行时出现异常：{e}")

    # 如果最后一次循环结束后，计数器不为0，则保存一次剩余的结果
    if counter > 0:
        train_zh_df.to_csv('./data/symptom-disease-train-dataset-zh.csv')

    # 测试集翻译
    counter = 0
    for idx, row in test_df.iterrows():
        try:
            translated_text = translate_func(row['text'])
            test_zh_df.loc[idx, 'text'] = translated_text
            test_zh_df.loc[idx, 'label'] = row['label']

            counter += 1
            if counter == 10:
                test_zh_df.to_csv('./data/symptom-disease-test-dataset-zh.csv')
                counter = 0
        except Exception as e:
            print(f"翻译第{idx}行时出现异常：{e}")

    # 如果最后一次循环结束后，计数器不为0，则保存一次剩余的结果
    if counter > 0:
        test_zh_df.to_csv('./data/symptom-disease-test-dataset-zh.csv')


