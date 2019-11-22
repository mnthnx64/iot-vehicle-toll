# IoT Vehicle License plate Detector

Computer Vision with IoT for an automated vehicle toll system

## Introduction

This project uses the inbulit `cv2.ml` module to detect the license plate of vehicles. It also uses `pytesseract` as the OCR engine to extract the text in the detected license plate. A `flask` encapsulation is provided to ease the usage of the application.

## Installaton

The project was solely built in `python 3.6.8` at the time of release. Install the dependencies by running `pip install -r requirements.txt`.
To upload an image use the following format:

```python
import requests
import cv2

BASE_IP = ''
url = 'post_image'

img = cv2.imread('path_to_image.png')
_, data = cv2.imencode('.jpg',img)

response = requests.post(BASE_IP + url, data=data.tostring())
print(response.json())
```

Update: I will release a version which takes the input from camera shortly.
