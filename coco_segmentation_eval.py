import numpy as np
import cv2  # For drawing polygons (optional)
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from segment_anything import sam_model_registry, SamPredictor
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils
categories = {}
captions = {}
scs = {}
with open("datasets/annotations/labels.txt") as f:
    while True:
        line = f.readline()
        if not line:
            break
        (id, name) = line.strip().split(":")
        id = int(id)
        categories[id] = name

coco=COCO('datasets/annotations/instances_val2017.json')

with open('datasets/annotations/instances_val2017.json') as f:
    data = json.load(f)
    for i in data['annotations']:
        id = i['image_id']
        scs[id] = categories[i['category_id']]
        
with open('datasets/annotations/captions_val2017.json') as f:
    data = json.load(f)
    for i in data['annotations']:
        captions[i['image_id']] = i['caption']


def create_mask_from_segmentation(id):
    # Initialize an empty mask
    
    im = coco.loadImgs(id)[0]
    ann = coco.getAnnIds(id)[0]
    ann = coco.loadAnns(ann)[0]
    mask = np.zeros((im['height'], im['width']), dtype=float)

    segmentation_array = coco.annToMask(ann)#np.array(single_segmentation).reshape(-1, 1, 2).astype(np.int32)
    mask += segmentation_array
    # Create a filled polygon from segmentation coordinates
    #cv2.fillPoly(mask, [segmentation_array], color=(255, 255, 255))  # Using OpenCV to fill the polygon
    
    if ann['area'] < 5 * im['height'] * im['width'] / 100:
        return False
    mask = (mask * 255).astype(np.uint8)
    mask = cv2.resize(mask,(32,32))
    return mask


id_list = list(scs.keys())

eval_json = {"image_data": []}

os.makedirs("masks/", exist_ok=True)
for n in tqdm(id_list):
    #if n in segmentations and n in widths and n in heights:
    mask = create_mask_from_segmentation(n)
    if type(mask) != bool:
        caption = captions[n]
        mask_name = scs[n]
        if mask_name not in caption:
            caption = "a " + mask_name + ". " + caption
        mask_index = caption.index(mask_name)
        mask_file = f"masks/{n}.png"
        cv2.imwrite(mask_file,mask)
        eval_json['image_data'].append({"caption": caption,\
            "mask_path": mask_file, "mask_index": mask_index+1, "id": n})
with open("eval.json", "w") as f:         
    json.dump(eval_json, f)



'''def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    


image = cv2.imread('eval_cos_0.3/776.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10,10))
plt.imshow(image)
plt.axis('on')
plt.savefig("initial_image.png")
sam = sam_model_registry["vit_h"](checkpoint="segmentation/sam_vit_h_4b8939.pth")
sam.to(device="cuda:1")

def compute_centroid(points):
    length = points.shape[0]
    sum_x = np.sum(points[:, 0])
    sum_y = np.sum(points[:, 1])
    return sum_x/length, sum_y/length

predictor = SamPredictor(sam)
predictor.set_image(image)

s = segmentations[776]
print(s)
s = compute_centroid(s)
print(s)

input_point = np.array([s])
input_label = np.array([1])

masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
)

print(masks.shape)

for i, (mask, score) in enumerate(zip(masks, scores)):
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    show_mask(mask, plt.gca())
    show_points(input_point, input_label, plt.gca())
    plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
    plt.axis('off')
    plt.savefig(f"fig_{i}.png")'''