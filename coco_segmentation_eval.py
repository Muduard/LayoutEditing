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
import clip
from PIL import Image
from torch.utils.data import Dataset, DataLoader


parser = argparse.ArgumentParser(description='Stable Diffusion Layout Editing')
parser.add_argument('--task', default="generate_mask")
parser.add_argument('--data_path', default="test/")
parser.add_argument('--batch-size', type=int, default=50,
                    help='Batch size to use')
parser.add_argument('--clip-model', type=str, default='ViT-B/32',
                    help='CLIP model to use')
parser.add_argument('--cuda', type=int, default=-1)
args = parser.parse_args()

device = "cpu"
if args.cuda > -1:
     device = f'cuda:{args.cuda}'

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


def count_word(sentence, index):
    count = 0
    for i in range(len(sentence[:index])):
        if sentence[i] == " ":
            count += 1
    return count

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
            
            mask_name = categories[cat_ids[i]].replace(" ","")
            
            if mask_name not in caption:
                caption = mask_name + " " + caption
                mask_indexes = list(map(lambda x: x+1, mask_indexes))
            
            if mask_name in caption:
                mask_indexes.append(count_word(caption,caption.index(mask_name)) + 2)
            
            mask_files.append(f"masks/{n}_{i}.png")
            cv2.imwrite(mask_files[i],mask)
        eval_json['image_data'].append({"caption": caption,\
            "mask_path": mask_files, "mask_indexes": mask_indexes, "id": n})
            
    with open("eval.json", "w") as f:         
        json.dump(eval_json, f)


def compute_iou(data_path):
    coco=COCO('datasets/annotations/instances_val2017.json')
    sam = sam_model_registry["vit_h"](checkpoint="segmentation/sam_vit_h_4b8939.pth").to(device)
    predictor = SamPredictor(sam)
    predictor = predictor
    dataset = os.listdir(data_path)
    ious = []
    bar = tqdm(dataset)
    for f in bar:
        #Legacy generation
        if "_" not in f:
            old_f = f
            f = f[:-4] + "_0" + f[-4:]
            os.rename(f'{data_path}{old_f}', f'{data_path}{f}')
        id = int(f[:f.index("_")])
        image = cv2.imread(f'{data_path}{f}')
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
                iou = abs(mask_utils.iou([pred_mask], [real_mask], [False])[0][0])
                if iou > max_iou:
                    max_iou = iou
            file_ious.append(max_iou)
            
        ious.append(sum(file_ious) / len(file_ious))
        bar.set_postfix_str(f'iou: {sum(ious) / len(ious)}')
    print(sum(ious) / len(ious))


class DummyDataset(Dataset):
    
    FLAGS = ['img', 'txt']
    def __init__(self, real_path,
                 real_flag: str = 'txt',
                 fake_flag: str = 'img',
                 transform = None,
                 tokenizer = None) -> None:
        super().__init__()
        assert real_flag in self.FLAGS and fake_flag in self.FLAGS, \
            'CLIP Score only support modality of {}. However, get {} and {}'.format(
                self.FLAGS, real_flag, fake_flag
            )
        self.reals = []
        self.real_flag = real_flag
        self.fake_folder = []
        self.fake_flag = fake_flag
        self.transform = transform
        self.tokenizer = tokenizer
        with open("eval.json", "r") as eval:
            data_eval = json.load(eval)['image_data']
            bar = tqdm(data_eval)
            files = os.listdir(real_path)
            for image_data in bar:
                id = image_data['id']
                filename = f'{real_path}{id}.png'
                if os.path.exists(filename):
                    self.reals.append(image_data['caption'])
                    
                    self.fake_folder.append(filename)
                else:
                    for f in files:
                        if f'{id}' in f:
                            self.reals.append(image_data['caption'])
                            self.fake_folder.append(f'{real_path}{f}')
        print(len(self.reals))
        # assert self._check()

    def __len__(self):
        return len(self.reals)

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError
        real_path = self.reals[index]
        fake_path = self.fake_folder[index]
        real_data = self._load_modality(real_path, self.real_flag)
        fake_data = self._load_modality(fake_path, self.fake_flag)

        sample = dict(real=real_data, fake=fake_data)
        return sample
    
    def _load_modality(self, path, modality):
        if modality == 'img':
            data = self._load_img(path)
        elif modality == 'txt':
            data = self._load_txt(path)
        else:
            raise TypeError("Got unexpected modality: {}".format(modality))
        return data

    def _load_img(self, path):
        img = Image.open(path)
        if self.transform is not None:
            img = self.transform(img)
        return img
    
    def _load_txt(self,data):
        data = self.tokenizer(data).squeeze()
        return data

    def _check(self):
        for idx in range(len(self)):
            real_name = self.real_folder[idx].split('.')
            fake_name = self.fake_folder[idx].split('.')
            if fake_name != real_name:
                return False
        return True
    
    def _combine_without_prefix(self, folder_path, prefix='.'):
        folder = []
        for name in os.listdir(folder_path):
            if name[0] == prefix:
                continue
            folder.append(os.path.join(folder_path, name))
        folder.sort()
        return folder

def forward_modality(model, data, flag):
    device = next(model.parameters()).device
    if flag == 'img':
        features = model.encode_image(data.to(device))
    elif flag == 'txt':
        features = model.encode_text(data.to(device))
    else:
        raise TypeError
    return features


@torch.no_grad()
def calculate_clip_score(dataloader, model, real_flag, fake_flag):
    score_acc = 0.
    sample_num = 0.
    logit_scale = model.logit_scale.exp()
    for batch_data in tqdm(dataloader):
        real = batch_data['real']
        real_features = forward_modality(model, real, real_flag)
        fake = batch_data['fake']
        fake_features = forward_modality(model, fake, fake_flag)
        
        # normalize features
        real_features = real_features / real_features.norm(dim=1, keepdim=True).to(torch.float32)
        fake_features = fake_features / fake_features.norm(dim=1, keepdim=True).to(torch.float32)
        
        # calculate scores
        # score = logit_scale * real_features @ fake_features.t()
        # score_acc += torch.diag(score).sum()
        score = logit_scale * (fake_features * real_features).sum()
        score_acc += score
        sample_num += real.shape[0]
    
    return score_acc / sample_num



def compute_clip_score(real_path):
    try:
        num_cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        # os.sched_getaffinity is not available under Windows, use
        # os.cpu_count instead (which may not return the *available* number
        # of CPUs).
        num_cpus = os.cpu_count()

    num_workers = min(num_cpus, 8) if num_cpus is not None else 0


    print('Loading CLIP model: {}'.format(args.clip_model))
    model, preprocess = clip.load(args.clip_model, device="cuda")
    
    dataset = DummyDataset(real_path,transform=preprocess, tokenizer=clip.tokenize)
    dataloader = DataLoader(dataset, args.batch_size, 
                            num_workers=num_workers, pin_memory=True)
    
    print('Calculating CLIP Score:')
    clip_score = calculate_clip_score(dataloader, model,
                                      "txt", "img")
    clip_score = clip_score.cpu().item()
    print('CLIP Score: ', clip_score)

            

#compute_clip_score("eval_new_lcm_multi_eta05_r50/")

if args.task == "generate_mask":
    generate_mask()
elif args.task =="clip":
    compute_clip_score(args.data_path)
else:
    compute_iou(args.data_path)