import os 
import cv2
import numpy as np
import matplotlib.pyplot as plt

h1 = cv2.imread("cat4.png")

h1 = np.where(h1 < 100, 0, 255)
h1[:,:,0] = 0
h1[:,:,1] = 0
cv2.imwrite("cat5.png",h1)
