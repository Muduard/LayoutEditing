import os

from pytorch_fid import fid_score
import os
import numpy as np
import shutil
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

def compute_fids(path1, path2, n_datapoints, n):

    test_dir1 = "sample_dir1/"
    if not os.path.exists(test_dir1):
        os.mkdir(test_dir1)
        filetest_dir1 = os.listdir(path1)
        for file in tqdm(filetest_dir1):
            image = cv2.imread(path1+file)
            image = cv2.resize(image,(512,512))
            
            cv2.imwrite(test_dir1 + file, image)
    test_dir2 = "sample_dir2/"
    fids = []
    filetest_dir2 = os.listdir(path2)
    n_step = int(n / n_datapoints)
    for i in range(n_step-1, n, n_step):
        sample2 = filetest_dir2[:i]
        shutil.rmtree(test_dir2, ignore_errors=True)
        os.mkdir(test_dir2)
        for f in sample2:
            shutil.copyfile(path2 + f, test_dir2 + f)
        print(path1)
        fids.append(fid_score.calculate_fid_given_paths((test_dir1,test_dir2),16,"cuda:0",2048,16))
        print(fids[-1])
    return fids


def fid_over_time_plot(path1, path2, n_datapoints, n_dataset):
    fids = compute_fids(path1, path2, n_datapoints, n_dataset)

    
    plt.plot(np.linspace(n_dataset/n_datapoints, n_dataset, n_datapoints, dtype=int), fids)
    plt.ylabel("FID")
    plt.xlabel("Sample Size")
    print(fids)
    plt.savefig("fid_results.png")

def compute_original_statistics(path, stats_name):
    fid_score.save_fid_stats((path, stats_name), 16, "cuda:0",2048,16)

n_datapoints = 1
path1 = "datasets/images/val2017/"#"datasets/annotations/cap_val2017.npz"#"datasets/images/val2017/"
path2 = "generated3/"

#compute_original_statistics(path1, "datasets/annotations/cap_val2017.npz")
fid_over_time_plot(path1,path2,n_datapoints,3800)
