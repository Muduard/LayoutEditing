import os 
import cv2
import numpy as np
import matplotlib.pyplot as plt

h1 = cv2.imread("test/original/26.png", cv2.IMREAD_GRAYSCALE)

h1 = np.where(h1 < 180, 0, 255)
cv2.imwrite("test/original/27.png",h1)
