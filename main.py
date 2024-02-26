from flask import Flask
from api.views import *

app = Flask(__name__)

app.register_blueprint(blue)

if __name__ == "__main__":
    app.run()
