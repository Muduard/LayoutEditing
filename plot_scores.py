import matplotlib.pyplot as plt
import numpy as np

eta = [0.01, 0.1, 0.3, 0.5, 1]
miou = [0.76, 0.77, 0.78, 0.82, 0.84]
fid = [35.82, 36.37, 38.53, 38.61,41.75]

with plt.style.context("seaborn-v0_8-poster"):
    plt.title("Ablation of Eta on mIoU and FID")
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Eta')
    ax1.set_ylabel('FID', color="tab:blue")
    ax1.scatter(eta, fid)
    ax1.plot(eta, fid)
    ax2 = ax1.twinx()
    ax2.set_ylabel('mIoU', color="tab:red")
    ax2.plot(eta, miou, color="tab:red")
    ax2.scatter(eta, miou, color="tab:red")
    plt.show()

'''ran = [0, 0.02, 0.05, 0.2, 1]
fid = [35.82, 37.58, 36.37, 38.61 ,36.31]
with plt.style.context("seaborn-v0_8-poster"):
    plt.xlabel("Perturbation")
    plt.ylabel("FID")
    plt.title("Initial attention perturbation")
    plt.plot(ran, fid)
    plt.scatter(ran, fid)
    plt.show()'''

