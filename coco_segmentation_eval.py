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
import argparse
#TODO maschere multiple


parser = argparse.ArgumentParser(description='Stable Diffusion Layout Editing')
parser.add_argument('--task', default="generate_mask")
parser.add_argument('--data_path', default="test/")

args = parser.parse_args()

def create_mask_from_segmentation(id, coco):
    # Initialize an empty mask
    
    im = coco.loadImgs(id)[0]
    anns = coco.getAnnIds(id)
    cat_ids = []
    
    masks = []
    for ann in anns:
        ann = coco.loadAnns(ann)[0]
        cat_ids.append(ann['category_id'])
        mask = np.zeros((im['height'], im['width']), dtype=float)
        
        segmentation_array = coco.annToMask(ann)#np.array(single_segmentation).reshape(-1, 1, 2).astype(np.int32)
        
        mask += segmentation_array
        # Create a filled polygon from segmentation coordinates
        #cv2.fillPoly(mask, [segmentation_array], color=(255, 255, 255))  # Using OpenCV to fill the polygon
        
        if ann['area'] > 5 * im['height'] * im['width'] / 100:
            mask = (mask * 255).astype(np.uint8)
            mask = cv2.resize(mask,(32,32))
            masks.append(mask)
            
    return masks, cat_ids

def get_categories():
    categories = {}
    with open("datasets/annotations/labels.txt") as f:
        while True:
                line = f.readline()
                if not line:
                    break
                (id, name) = line.strip().split(":")
                id = int(id)
                categories[id] = name
    return categories

def get_scs(categories):
    scs = {}
    with open('datasets/annotations/instances_val2017.json') as f:
        data = json.load(f)
        for i in data['annotations']:
            id = i['image_id']
            scs[id] = categories[i['category_id']]
    return scs

def get_captions():
    captions = {}
    with open('datasets/annotations/captions_val2017.json') as f:
        data = json.load(f)
        for i in data['annotations']:
            if i['image_id'] not in captions:
                captions[i['image_id']] = [i['caption']]
            else:
                captions[i['image_id']].append(i['caption'])

    return captions

def generate_mask():
    
    captions = get_captions()
    categories = get_categories()
    scs = get_scs(categories)
    coco=COCO('datasets/annotations/instances_val2017.json')    
    id_list = list(scs.keys())
    eval_json = {"image_data": []}
    os.makedirs("masks/", exist_ok=True)
    for n in tqdm(id_list):
        
        #if n in segmentations and n in widths and n in heights:
        masks, cat_ids = create_mask_from_segmentation(n , coco)
        caption = captions[n][0]
        mask_indexes = []
        mask_files = []
        for i, mask in enumerate(masks):
            
            mask_name = categories[cat_ids[i]]
            if mask_name not in caption:
                caption = mask_name + " " + caption
            mask_indexes = list(map(lambda x: x+1, mask_indexes))
            mask_indexes.append(caption.index(mask_name) + 1)
            mask_files.append(f"masks/{n}_{i}.png")
            cv2.imwrite(mask_files[i],mask)
        eval_json['image_data'].append({"caption": caption,\
            "mask_path": mask_files, "mask_indexes": mask_indexes, "id": n})
            
    with open("eval.json", "w") as f:         
        json.dump(eval_json, f)


def compute_iou(data_path):
    coco=COCO('datasets/annotations/instances_val2017.json')
    sam = sam_model_registry["vit_h"](checkpoint="segmentation/sam_vit_h_4b8939.pth").to("cuda:1")
    predictor = SamPredictor(sam)
    predictor = predictor
    dataset = os.listdir(data_path)
    ious = []
    bar = tqdm(dataset)
    for f in bar:
        id = int(f[:-4])
        image = cv2.imread(f'{data_path}{id}.png')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        anns = coco.getAnnIds(id)
        file_ious = []
        predictor.set_image(image)
        for ann in anns:
            s = coco.loadAnns(ann)[0]
            real_mask = coco.annToRLE(s)
            
            centroid = [(s['bbox'][0] + s['bbox'][2]) / 2, (s['bbox'][1] + s['bbox'][3]) / 2]
            input_point = np.array([centroid])
            input_label = np.array([1])
            masks, _, _ = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,
            )
            max_iou = 0
            for m in masks:
                pred_mask = np.asfortranarray(m)
                pred_mask = mask_utils.encode(pred_mask)
                iou = abs(mask_utils.iou([pred_mask], [real_mask], [False, False])[0][0])
                if iou > max_iou:
                    max_iou = iou
            file_ious.append(max_iou)
            
        ious.append(sum(file_ious) / len(file_ious))
        bar.set_postfix_str(f'iou: {sum(ious) / len(ious)}')
    print(sum(ious) / len(ious))

if args.task == "generate_mask":
    generate_mask()
else:
    compute_iou(args.data_path)