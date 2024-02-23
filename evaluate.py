from cleanfid import fid
from PIL import Image
import torchvision.transforms as transforms
import argparse

parser = argparse.ArgumentParser(description='Stable Diffusion Layout Editing')
parser.add_argument('--validation_path', default="./datasets/val2014/")
parser.add_argument('--predicted_path', default="./results/")
args = parser.parse_args()

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
path1 = args.validation_path#"./datasets/images/val2017/"#"datasets/annotations/cap_val2017.npz"#"datasets/images/val2017/"
path2 = args.predicted_path

score = fid.compute_fid(path1, path2)
print(score)
