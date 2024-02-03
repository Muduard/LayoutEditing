import numpy as np
import cv2  # For drawing polygons (optional)
import json
import matplotlib.pyplot as plt

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
        scs[id] = categories[i['category_id']]


with open('datasets/annotations/captions_val2017.json') as f:
    data = json.load(f)
    for i in data['annotations']:
        captions[i['image_id']] = i['caption']


def create_mask_from_segmentation(segmentation_info, image_height, image_width):
    # Initialize an empty mask
    mask = np.zeros((image_height, image_width), dtype=np.uint8)

    # Reshape segmentation coordinates into a numpy array
    segmentation_array = np.array(segmentation_info).reshape(-1, 2).astype(np.int32)

    # Create a filled polygon from segmentation coordinates
    cv2.fillPoly(mask, [segmentation_array], color=1)  # Using OpenCV to fill the polygon

    return mask


n = list(widths.keys())[2]

mask = create_mask_from_segmentation(segmentation_info=segmentations[n], image_width=widths[n], image_height=heights[n])
caption = captions[n]
mask_name = scs[n]
if mask_name not in caption:
    caption += " " + mask_name
mask_index = caption.index(mask_name)
cv2.imwrite("coco_mask.png",mask)
