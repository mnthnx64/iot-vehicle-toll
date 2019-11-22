# IoT Vehicle License plate Detector

Computer Vision with IoT for an automated vehicle toll system

## Introduction

This project uses the inbulit `cv2.ml` module to detect the license plate of vehicles. It also uses `pytesseract` as the OCR engine to extract the text in the detected license plate. A `flask` encapsulation is provided to ease the usage of the application.

## Installaton

The project was solely built in `python 3.6.8` at the time of release. Install the dependencies by running `pip install -r requirements.txt`.
Please run the code in `uploadimg.py` with a specified path to image to upload an image to the application.

Updates: I will release a version which takes the input from camera shortly.
