import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import cv2
import json
from PIL import Image
parser = argparse.ArgumentParser(description='Stable Diffusion Layout Editing')
parser.add_argument('--path', default="./datasets/images/val2014/")
parser.add_argument('--path_compare', default="./results/")
parser.add_argument('--n', default=1)
parser.add_argument('--task', default="mask")
parser.add_argument('--id', default=0)
args = parser.parse_args()




def get_centroid(mask):
    left = mask.shape[1]
    right = 0
    up = 0
    down = mask.shape[0]
    
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            
            if sum(mask[i,j]) > 0:
                if i > up:
                    up = i
                if i < down:
                    down = i
                if j > right:
                    right = j
                if j < left:
                    left = j
    return ( (up + down)/2, (right + left)/2 )        

if args.task == "mask":
    with open("eval.json") as ev:
        data_eval = json.load(ev)['image_data']
    files = os.listdir(args.path)
    args.id = int(args.id)
    if args.id > 0:
        indices = np.array([args.id])
    else:
        indices = np.random.rand(args.n) * len(data_eval)
    
    indices = indices.astype(int)
    for i in indices:
        
        id = data_eval[i]['id']
        print(data_eval[i])
        print(id)
        print(i)
        caption = data_eval[i]['caption']
        masks = data_eval[i]['mask_path']
        mask_indexes = data_eval[i]['mask_indexes']
        
        i_files = list(filter(lambda f: str(id) in f, files))
        if len(i_files) > 0 and len(mask_indexes) > 0:
            mask_indexes[0] -= 2
            i_masks = []
            centroids = []
            for m in masks:
                i_masks.append(cv2.cvtColor(cv2.imread(m),  cv2.COLOR_BGR2RGB))
            f = os.path.join(args.path,i_files[0])
            image = cv2.cvtColor(cv2.imread(f),  cv2.COLOR_BGR2RGB)
            color = np.array(list(np.random.choice(range(256), size=3)), dtype=np.uint8)
            composite_mask = i_masks[0] * color
            centroids.append(get_centroid(i_masks[0]))
            
            for j, m in  enumerate(i_masks[1:]):
                color = np.array(list(np.random.choice(range(256), size=3)), dtype=np.uint8)
                centroids.append(get_centroid(m))
                m[m > 0] = 1
                composite_mask += m * color

            cv2.imwrite(f"examples/{id}_mask.png",composite_mask)
            imPIL = Image.fromarray(image)
            imPIL.save(f"examples/{id}.png")
            
            fig, axs = plt.subplots(2,1)
            fig.suptitle(caption)
            axs[0].imshow(composite_mask)
            words = caption.split()
            for j, c in enumerate(centroids):
                axs[0].text(c[1],c[0], words[mask_indexes[j] -1], color="white")
            
            axs[1].imshow(image)
            fig.tight_layout()
            
            

            
            plt.show()
        else:
            print("Waeaea")


            


