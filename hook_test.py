import os 
import torch
from diffusers import DiffusionPipeline
from tqdm import tqdm
import matplotlib.pyplot as plt
from ptp_utils import AttentionStore,diffusion_step,get_multi_level_attention_from_average,register_attention_control, get_cross_attention, show_cross_attention,latent2image,fw_hook, bw_hook,register_hook
import cv2
import argparse
from accelerate import Accelerator
import numpy as np
import torch.nn.functional as F

device = "cuda:1"
timesteps = 30

GUIDANCE_SCALE = 7.5
MAX_NUM_WORDS = 77
#device = "cuda"#torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
repo_id =  "runwayml/stable-diffusion-v1-5"#"CompVis/stable-diffusion-v1-4"#

pipe = DiffusionPipeline.from_pretrained(repo_id, torch_dtype=torch.float16).to(device)#DiffusionPipeline.from_pretrained(repo_id, torch_dtype=torch.float16,use_safetensors=True).to(device)

parser = argparse.ArgumentParser(description='Stable Diffusion Layout Editing')
parser.add_argument('--mask', default="attentions/cat2.png",
                    help='Target mask of layout editing')
parser.add_argument('--res_size', default=16, type=int,
                    help='Resolution size to edit')
parser.add_argument('--out_path', default="test/",
                    help='Output folder')
parser.add_argument('--lr', default=0.01, type=float,
                    help='Optimization Learning Rate')
parser.add_argument('--epochs', default=70, type=int,
                    help='Optimization Epochs')

args = parser.parse_args()

os.makedirs(args.out_path,exist_ok = True)

h = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
ht = torch.tensor(cv2.resize(h, (args.res_size, args.res_size)), dtype=torch.float) / 255

prompts = ["Cat on table"]
timesteps = 30
pipe.scheduler.set_timesteps(timesteps)
batch_size = 1
torch.manual_seed(42)
noise = torch.randn((batch_size, 4, 64, 64), dtype=torch.float16, device=device) 

latents = noise

text_input = pipe.tokenizer(
        prompts[0],
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )

text_emb = pipe.text_encoder(text_input.input_ids.to(device))[0]

max_length = text_input.input_ids.shape[-1]
uncond_input = pipe.tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
)
uncond_embeddings = pipe.text_encoder(uncond_input.input_ids.to(device))[0]

#delta_emb.retain_grad()
context = [uncond_embeddings, text_emb]
guidance_scale = 7.5

for param in pipe.unet.parameters():
    param.requires_grad = False
lossF = torch.nn.BCELoss()#BCELoss()#MSELoss()
latents.requires_grad = False

register_hook(pipe.unet,0,None)



for t in tqdm(pipe.scheduler.timesteps):
    latents = diffusion_step(pipe, None, latents, context, t, guidance_scale, None)
