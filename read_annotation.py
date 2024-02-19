import numpy as np
import cv2  # For drawing polygons (optional)
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils

coco=COCO('datasets/annotations/instances_val2017.json')
