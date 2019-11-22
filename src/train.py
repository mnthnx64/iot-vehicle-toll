import cv2
import numpy as np
import math
import random
import os

PLATE_WIDTH_PADDING_FACTOR = 1.3
PLATE_HEIGHT_PADDING_FACTOR = 1.5
kNearest = cv2.ml.KNearest_create()  # constants for checkIfPossibleChar, this checks one possible char only (does not compare to another char)
MIN_PIXEL_WIDTH = 2
MIN_PIXEL_HEIGHT = 8

MIN_ASPECT_RATIO = 0.25
MAX_ASPECT_RATIO = 1.0

MIN_PIXEL_AREA = 80      # constants for comparing two chars
MIN_DIAG_SIZE_MULTIPLE_AWAY = 0.3
MAX_DIAG_SIZE_MULTIPLE_AWAY = 5.0

MAX_CHANGE_IN_AREA = 0.5

MAX_CHANGE_IN_WIDTH = 0.8
MAX_CHANGE_IN_HEIGHT = 0.2

MAX_ANGLE_BETWEEN_CHARS = 12.0   # other constants
MIN_NUMBER_OF_MATCHING_CHARS = 3

RESIZED_CHAR_IMAGE_WIDTH = 20
RESIZED_CHAR_IMAGE_HEIGHT = 30

MIN_CONTOUR_AREA = 100

class PlateDetector():

    def __init__(self):
        self.showSteps = False
        self.SCALAR_BLACK = (0.0, 0.0, 0.0)
        self.SCALAR_WHITE = (255.0, 255.0, 255.0)
        self.SCALAR_YELLOW = (0.0, 255.0, 255.0)
        self.SCALAR_GREEN = (0.0, 255.0, 0.0)
        self.SCALAR_RED = (0.0, 0.0, 255.0)

    def Train(self):
        try:
            npaClassifications = np.loadtxt("classifications.txt", np.float32)
        except:                                                                                
            print("error, unable to open classifications.txt, exiting program\n")
            os.system("pause")
            return False

        try:
            npaFlattenedImages = np.loadtxt("flattened_images.txt", np.float32)       
        except:
            print("error, unable to open flattened_images.txt, exiting program\n")
            os.system("pause")
            return False

        npaClassifications = npaClassifications.reshape((npaClassifications.size, 1))
        kNearest.setDefaultK(1)
        kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)
        return True