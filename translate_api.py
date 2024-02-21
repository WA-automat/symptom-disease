import requests
import datetime
import hashlib
import base64
import hmac
import json

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
        return text


#
# 讯飞机器翻译2.0 WebAPI 接口调用示例
# 运行前填写Appid、APIKey、APISecret
#
# 1.接口文档（必看）：https://www.xfyun.cn/doc/nlp/niutrans/API.html
# 2.错误码链接：https://www.xfyun.cn/document/error-code （错误码code为5位数字）


class XunFei(object):
    def __init__(self, text):
        # 以下三个参数到控制台 https://console.xfyun.cn/services/its 获取
        self.APPID = "d5f76ad6"
        self.Secret = "YmE1OWE1Mjc4NzFlN2FlZWMxNTI3MWFl"
        self.APIKey = "42909daba6017b059cb5790d7bc58646"

        self.Host = "ntrans.xfyun.cn"
        self.RequestUri = "/v2/ots"
        self.url = "https://" + self.Host + self.RequestUri
        self.HttpMethod = "POST"
        self.Algorithm = "hmac-sha256"
        self.HttpProto = "HTTP/1.1"

        curTime_utc = datetime.datetime.utcnow()
        self.Date = self.httpdate(curTime_utc)

        self.Text = text
        self.BusinessArgs = {
            "from": "en",
            "to": "zh",
        }

    def hashlib_256(self, res):
        m = hashlib.sha256(bytes(res.encode(encoding='utf-8'))).digest()
        result = "SHA-256=" + base64.b64encode(m).decode(encoding='utf-8')
        return result

    def httpdate(self, dt):
        weekday = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][dt.weekday()]
        month = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep",
                 "Oct", "Nov", "Dec"][dt.month - 1]
        return "%s, %02d %s %04d %02d:%02d:%02d GMT" % (weekday, dt.day, month,
                                                        dt.year, dt.hour, dt.minute, dt.second)

    def generateSignature(self, digest):
        signatureStr = "host: " + self.Host + "\n"
        signatureStr += "date: " + self.Date + "\n"
        signatureStr += self.HttpMethod + " " + self.RequestUri \
                        + " " + self.HttpProto + "\n"
        signatureStr += "digest: " + digest
        signature = hmac.new(bytes(self.Secret.encode(encoding='utf-8')),
                             bytes(signatureStr.encode(encoding='utf-8')),
                             digestmod=hashlib.sha256).digest()
        result = base64.b64encode(signature)
        return result.decode(encoding='utf-8')

    def init_header(self, data):
        digest = self.hashlib_256(data)
        # print(digest)
        sign = self.generateSignature(digest)
        authHeader = 'api_key="%s", algorithm="%s", ' \
                     'headers="host date request-line digest", ' \
                     'signature="%s"' \
                     % (self.APIKey, self.Algorithm, sign)
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Method": "POST",
            "Host": self.Host,
            "Date": self.Date,
            "Digest": digest,
            "Authorization": authHeader
        }
        return headers

    def get_body(self):
        content = str(base64.b64encode(self.Text.encode('utf-8')), 'utf-8')
        postdata = {
            "common": {"app_id": self.APPID},
            "business": self.BusinessArgs,
            "data": {
                "text": content,
            }
        }
        body = json.dumps(postdata)
        return body

    def call_url(self):
        if self.APPID == '' or self.APIKey == '' or self.Secret == '':
            print('Appid 或APIKey 或APISecret 为空！请打开demo代码，填写相关信息。')
        else:
            code = 0
            body = self.get_body()
            headers = self.init_header(body)
            response = requests.post(self.url, data=body, headers=headers, timeout=8)
            status_code = response.status_code
            if status_code != 200:
                # 鉴权失败
                print("Http请求失败，状态码：" + str(status_code) + "，错误信息：" + response.text)
                print("请根据错误信息检查代码，接口文档：https://www.xfyun.cn/doc/nlp/niutrans/API.html")
                return self.Text
            else:
                # 鉴权成功
                respData = json.loads(response.text)
                code = str(respData["code"])
                if code != '0':
                    return self.Text
                    # print("请前往https://www.xfyun.cn/document/error-code?code=" + code + "查询解决办法")
                else:
                    return respData['data']['result']['trans_result']['dst']


if __name__ == '__main__':
    gClass = XunFei("你确定吗")
    print(gClass.call_url())
