import numpy as np
import cv2  # For drawing polygons (optional)
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
categories = {}
captions = {}
segmentations = {}
heights = {}
widths = {}
scs = {}
with open("datasets/annotations/labels.txt") as f:
    while True:
        line = f.readline()
        if not line:
            break
        (id, name) = line.strip().split(":")
        id = int(id)
        categories[id] = name


with open('datasets/annotations/instances_val2017.json') as f:
    data = json.load(f)
    for i in data['images']:
        widths[i['id']] = i['width']
        heights[i['id']] = i['height']
    for i in data['annotations']:
        id = i['image_id']
        segmentations[id] = i['segmentation']
        if type(segmentations[id]) != list:
            segmentations[id] = segmentations[id]['counts']
        else:
            if len(segmentations[id]) == 1:
                segmentations[id] = [segmentations[id]]
        scs[id] = categories[i['category_id']]
        


with open('datasets/annotations/captions_val2017.json') as f:
    data = json.load(f)
    for i in data['annotations']:
        captions[i['image_id']] = i['caption']


def create_mask_from_segmentation(segmentation_info, image_height, image_width):
    # Initialize an empty mask
    mask = np.zeros((image_height, image_width), dtype=np.uint8)

    # Reshape segmentation coordinates into a numpy array
    for single_segmentation in segmentation_info:
        
        if type(single_segmentation) == list:
            segmentation_array = np.array(single_segmentation).reshape(-1, 1, 2).astype(np.int32)
            
            # Create a filled polygon from segmentation coordinates
            cv2.fillPoly(mask, [segmentation_array], color=(255, 255, 255))  # Using OpenCV to fill the polygon
            # Find contours of the filled area
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Compute the area of the filled polygon
            area = cv2.contourArea(contours[0])
            if area < 4470.5:
                return False
        
        else:
            return False
    mask = cv2.resize(mask,(32,32))
    return mask


id_list = list(widths.keys())

eval_json = {"image_data": []}

os.makedirs("masks/", exist_ok=True)
for n in tqdm(id_list):
    if n in segmentations and n in widths and n in heights:
        mask = create_mask_from_segmentation(segmentation_info=segmentations[n], image_width=widths[n], image_height=heights[n])
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


