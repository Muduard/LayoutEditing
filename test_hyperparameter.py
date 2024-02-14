import os 
import torch
from diffusers import DDIMScheduler,AutoencoderKL,UNet2DConditionModel, LCMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm import tqdm
import matplotlib.pyplot as plt
from ptp_utils import AttentionStore,get_attn_layers,diffusion_step,get_multi_level_attention_from_average,register_attention_control, get_cross_attention, show_cross_attention,latent2image,save_tensor_as_image, lcm_diffusion_step, get_guidance_scale_embedding
import cv2
import argparse
import numpy as np
import torch.nn.functional as F
from diffusers.models.attention_processor import AttnProcessor2_0

parser = argparse.ArgumentParser(description='Stable Diffusion Layout Editing')
parser.add_argument('--mask', default="mask_cat.png",
                    help='Target mask of layout editing')
parser.add_argument('--res_size', default=16, type=int,
                    help='Resolution size to edit')
parser.add_argument('--out_path', default="test/",
                    help='Output folder')
parser.add_argument('--lr', default=0.01, type=float,
                    help='Optimization Learning Rate')
parser.add_argument('--epochs', default=70, type=int,
                    help='Optimization Epochs')
parser.add_argument('--guide', action=argparse.BooleanOptionalAction)
parser.add_argument('--cuda', default=-1, type=int,
                    help='Cuda device to use')
parser.add_argument('--timesteps', default=50, type=int, 
                    help="Number of timesteps")
parser.add_argument('--prompt', default="A cat with a city in the background", 
                    type=str)
parser.add_argument('--mask_index', nargs='+', help='List of token indices to move with a mask')
parser.add_argument("--mask_path", type=str, help="Path of masks as image files with the name of the corresponding token")
parser.add_argument("--resampling_steps", type=int, default=0, help="Resample noise for better coherence")
parser.add_argument("--seed", type=int, default=82)
parser.add_argument("--diffusion_type", type=str, default="LCM")


MODEL_TYPE = torch.float16


args = parser.parse_args()
repo_id = "SimianLuo/LCM_Dreamshaper_v7"

if args.diffusion_type == "SD":
     repo_id = "runwayml/stable-diffusion-v1-5"

path_original = args.out_path + "original/"
os.makedirs(args.out_path,exist_ok = True)
os.makedirs(path_original,exist_ok = True)
os.makedirs("attentions/",exist_ok = True)
device = "cpu"
if args.cuda > -1:
     device = f'cuda:{args.cuda}'

vae = AutoencoderKL.from_pretrained(repo_id, subfolder="vae", torch_dtype=MODEL_TYPE).to(device)
tokenizer = CLIPTokenizer.from_pretrained(repo_id, subfolder="tokenizer", torch_dtype=MODEL_TYPE)
text_encoder = CLIPTextModel.from_pretrained(repo_id, subfolder="text_encoder", torch_dtype=MODEL_TYPE).to(device)
unet = UNet2DConditionModel.from_pretrained(repo_id, subfolder="unet", torch_dtype=MODEL_TYPE).to(device)
scheduler = LCMScheduler.from_pretrained(repo_id,subfolder="scheduler", torch_dtype=torch.float16)
if args.diffusion_type == "SD":
     scheduler = DDIMScheduler.from_pretrained(repo_id,subfolder="scheduler", torch_dtype=torch.float16)
#unet.set_attn_processor(AttnProcessor2_0())
#torch.compile(unet, mode="reduce-overhead", fullgraph=True)
