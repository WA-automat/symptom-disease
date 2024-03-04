from flask import Flask
from api.views import *
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

app.register_blueprint(blue)

if __name__ == "__main__":
    app.run()
