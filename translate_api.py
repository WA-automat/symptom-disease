import json
import requests

# 小牛翻译apikey
cattle_apikey = "abfa6ad0b3f9a1e9eee7940222bb2952"

'''
小牛翻译api
'''
def translate_cattle(text):
    url = 'http://api.niutrans.com/NiuTransServer/translation?'
    data = {"from": 'en', "to": 'zh', "apikey": cattle_apikey, "src_text": text}
    res = requests.post(url, data=data)
    res_dict = json.loads(res.text)
    if "tgt_text" in res_dict:
        return res_dict['tgt_text']
    else:
        return 'error'


if __name__ == "__main__":
    pass
