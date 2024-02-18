import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from natsort import natsorted
n = 5
m = 5

images = []
z_images = []
g_images = []
m_images = []
gb_images = []

files = os.listdir("test/qualitative_comp/o/")
files = natsorted(files)
for f in files:
    images.append(cv2.cvtColor(cv2.imread("test/qualitative_comp/o/" + f),  cv2.COLOR_BGR2RGB))

files = os.listdir("test/qualitative_comp/m/")
files = natsorted(files)
for f in files:
    m_images.append(cv2.cvtColor(cv2.imread("test/qualitative_comp/m/" + f), cv2.COLOR_BGR2RGB) )

files = os.listdir("test/qualitative_comp/g/")
files = natsorted(files)
for f in files:
    g_images.append(cv2.cvtColor(cv2.imread("test/qualitative_comp/g/" + f), cv2.COLOR_BGR2RGB) )

files = os.listdir("test/qualitative_comp/z/")
files = natsorted(files)
for f in files:
    z_images.append(cv2.cvtColor(cv2.imread("test/qualitative_comp/z/" + f), cv2.COLOR_BGR2RGB) )

files = os.listdir("test/qualitative_comp/gb/")
files = natsorted(files)
for f in files:
    gb_images.append(cv2.cvtColor(cv2.imread("test/qualitative_comp/gb/" + f), cv2.COLOR_BGR2RGB) )


fig, axs = plt.subplots(m, n)
for i in range(n):
    for j in range(m):
        if i == 0:
            axs[j, i].imshow(images[j])
        elif i == 1:
            axs[j, i].imshow(m_images[j])
        elif i == 2:
            axs[j, i].imshow(g_images[j])
        elif i == 3:
            axs[j, i].imshow(z_images[j])
        elif i == 4:
            axs[j, i].imshow(gb_images[j])
fig.tight_layout()
fig.savefig("results.pdf")
        