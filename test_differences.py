import os 
import cv2
import numpy as np
import matplotlib.pyplot as plt

h1 = cv2.imread("attentions/cat3.png", cv2.IMREAD_GRAYSCALE)

h1 = np.where(h1 < 100, 0, 255)
cv2.imwrite("attentions/cat2.png",h1)
