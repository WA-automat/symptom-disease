class ResponseResult:
    def __init__(self, code=200, msg="", data=None):
        self.code = code
        self.msg = msg
        self.data = data

    def toDict(self):
        return vars(self)
