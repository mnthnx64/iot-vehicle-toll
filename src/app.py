from flask import Flask, jsonify, request
import DetectChars
import Main
import cv2
import numpy as np

app = Flask(__name__)

@app.route("/")
def hello():
    return jsonify({"works": "200"})

@app.route("/post_image", methods = ['POST'])
def post_image():
    req_data = request
    nparr = np.fromstring(req_data.data, np.uint8)
    img = cv2.imdecode(nparr,cv2.IMREAD_COLOR)
    car_plate = Main.Detect(img)
    return jsonify(car_plate.getResponse())


blnKNNTrainingSuccessful = DetectChars.loadKNNDataAndTrainKNN()
if blnKNNTrainingSuccessful == False:                              
    print("\nerror: KNN traning was not successful\n")   


if __name__ == '__main__':
    app.run()