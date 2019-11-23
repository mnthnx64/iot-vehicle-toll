from flask import Flask, jsonify, request
import DetectChars
import Main
import cv2
import numpy as np
import os

app = Flask(__name__)

@app.route("/")
def hello():
    return jsonify({"works": "200"})

@app.route("/post_image", methods = ['POST'])
def post_image():
    try:
        req_data = request
        nparr = np.fromstring(req_data.data, np.uint8)
        img = cv2.imdecode(nparr,cv2.IMREAD_COLOR)
        car_plate = Main.Detect(img)
        return jsonify(car_plate.getResponse())
    except:
        return jsonify({"Error":"Error! Something went wrong."})
  


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)