import os

from cleanfid import fid
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from PIL import Image
import torchvision.transforms as transforms

#from torchmetrics.image.fid import FrechetInceptionDistance
def load_and_preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    img = preprocess(img).unsqueeze(0)
    return img


n_datapoints = 1
path1 = "./datasets/images/val2017/"#"datasets/annotations/cap_val2017.npz"#"datasets/images/val2017/"
path2 = "./eval_new_multi_cos05/"

score = fid.compute_fid(path1, path2)
print(score)
'''
# Load and preprocess real images
real_images = []
for filename in tqdm(os.listdir(path1)[:]):
    img_path = os.path.join(path1, filename)
    img = load_and_preprocess_image(img_path)
    real_images.append(img)
real_images = torch.cat(real_images, dim=0)

# Load and preprocess generated images
generated_images = []
for filename in tqdm(os.listdir(path2)):
    
    img_path = os.path.join(path2, filename)
    img = load_and_preprocess_image(img_path)
    generated_images.append(img)
generated_images = torch.cat(generated_images, dim=0)

#metric = FrechetInceptionDistance(feature=2048, reset_real_features=False, normalize=True)
#metric.update(real_images, real=True)
#metric.update(generated_images, real=False)
#fig_, ax_ = metric.plot()
#fig_.savefig("fid.png")
'''