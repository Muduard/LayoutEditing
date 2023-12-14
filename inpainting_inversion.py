import os 
import torch
from diffusers import DDIMScheduler,AutoencoderKL,UNet2DConditionModel, StableDiffusionPipeline, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm import tqdm
import matplotlib.pyplot as plt
from ptp_utils import AttentionStore,diffusion_step,get_multi_level_attention_from_average,register_attention_control, get_cross_attention, show_cross_attention,latent2image,save_tensor_as_image, ddim_invert,compute_embeddings
import cv2
import argparse
from accelerate import Accelerator
import numpy as np
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image,to_tensor
from PIL import Image
repo_id =  "CompVis/stable-diffusion-v1-4"#"CompVis/stable-diffusion-v1-4"#

parser = argparse.ArgumentParser(description='Stable Diffusion Layout Editing')
parser.add_argument('--mask', default="new_mask.png",
                    help='Target mask of layout editing')
parser.add_argument('--res_size', default=16, type=int,
                    help='Resolution size to edit')
parser.add_argument('--out_path', default="test/",
                    help='Output folder')
parser.add_argument('--lr', default=0.01, type=float,
                    help='Optimization Learning Rate')
parser.add_argument('--epochs', default=70, type=int,
                    help='Optimization Epochs')
#parser.add_argument('--guide', action=argparse.BooleanOptionalAction)
parser.add_argument('--cuda', default=-1, type=int,
                    help='Cuda device to use')
parser.add_argument('--timesteps', default=50, type=int, 
                    help="Number of timesteps")
parser.add_argument('--prompt', default="A cat with a city in the background", 
                    type=str)
parser.add_argument('--mask_index', nargs='+', help='List of token indices to move with a mask')
parser.add_argument("--mask_path", type=str, help="Path of masks as image files with the name of the corresponding token")
parser.add_argument("--resampling_steps", type=int, default=0, help="Resample noise for better coherence")
parser.add_argument("--seed", type=int, default=24)

MODEL_TYPE = torch.float16


args = parser.parse_args()

os.makedirs("generated/",exist_ok = True)
device = "cpu"
if args.cuda > -1:
     device = f'cuda:{args.cuda}'

scheduler = DDIMScheduler.from_pretrained(repo_id,subfolder="scheduler", torch_dtype=MODEL_TYPE)
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", scheduler=scheduler, torch_dtype=MODEL_TYPE)
pipe = pipe.to(device)
vae = pipe.vae
tokenizer = pipe.tokenizer
text_encoder = pipe.text_encoder
unet = pipe.unet



image = Image.open(f'examples/img1.png').convert('RGB')
prompt = "a cat sitting next to a mirror"

context = compute_embeddings(tokenizer, text_encoder, device, 1, prompt)
guidance_scale = 3.5
num_inference_steps = 50
scheduler.set_timesteps(num_inference_steps)

with torch.no_grad(): 
     latents = vae.encode(to_tensor(image).unsqueeze(0).to(dtype=MODEL_TYPE,device=device)*2-1)
l = 0.18215 * latents.latent_dist.sample()
with torch.no_grad():
    inverted_latents = ddim_invert(unet, scheduler, l, context, guidance_scale, 50)

#rec_latents = latent2image(inverted_latents[-1].unsqueeze(0)

rec_image = pipe(prompt, latents=inverted_latents[-1][None], num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).images[0]

rec_image.save(f"a.png")