import os

#from pytorch_fid import fid_score
import os
import numpy as np
import shutil
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import torch
from PIL import Image
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.models import inception_v3

def load_and_preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    img = preprocess(img).unsqueeze(0)
    return img

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
        fids.append(fid_score.calculate_fid_given_paths((test_dir1,test_dir2),16,"cuda:1",2048,16))
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
    fid_score.save_fid_stats((path, stats_name), 16, "cuda:1",2048,16)


# Function to calculate FID score
def calculate_fid_score(features_real, features_generated):
    mu_real = np.mean(features_real, axis=0)
    sigma_real = np.cov(features_real, rowvar=False)
    mu_generated = np.mean(features_generated, axis=0)
    sigma_generated = np.cov(features_generated, rowvar=False)

    ssdiff = np.sum((mu_real - mu_generated)**2.0)
    covmean = np.sqrtm(sigma_real.dot(sigma_generated))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid_score = ssdiff + np.trace(sigma_real + sigma_generated - 2.0 * covmean)
    return fid_score

# Load pre-trained InceptionV3 model
inception_model = inception_v3(pretrained=True, transform_input=False, aux_logits=True)
inception_model.eval()

# Function to extract features
def extract_features(images, model, batch_size=32):
    features_list = []
    num_images = len(images)

    with torch.no_grad():
        for i in range(0, num_images, batch_size):
            batch_images = images[i:i+batch_size]
            batch_features = model(batch_images)[0].view(batch_images.size(0), -1)
            features_list.append(batch_features.cpu().numpy())

    features = np.concatenate(features_list, axis=0)
    return features


n_datapoints = 1
path1 = "datasets/images/val2017/"#"datasets/annotations/cap_val2017.npz"#"datasets/images/val2017/"
path2 = "generated3/"

#compute_original_statistics(path1, "datasets/annotations/cap_val2017.npz")
#fid_over_time_plot(path1,path2,n_datapoints,10000)

# Load and preprocess real images
real_images = []
for filename in os.listdir(path1):
    img_path = os.path.join(path2, filename)
    img = load_and_preprocess_image(img_path)
    real_images.append(img)
real_images = torch.cat(real_images, dim=0)

# Load and preprocess generated images
generated_images = []
for filename in os.listdir(path2):
    img_path = os.path.join(path2, filename)
    img = load_and_preprocess_image(img_path)
    generated_images.append(img)
generated_images = torch.cat(generated_images, dim=0)

# Extract features from real and generated images
real_features = extract_features(real_images, inception_model)
generated_features = extract_features(generated_images, inception_model)

# Calculate FID score
fid_score = calculate_fid_score(real_features, generated_features)
print("FID Score:", fid_score)