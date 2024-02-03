import os 
import torch
from diffusers import DDIMScheduler,AutoencoderKL,UNet2DConditionModel, StableDiffusionPipeline, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm import tqdm
from ptp_utils import diffusion_step,latent2image,compute_embeddings
import argparse
import numpy as np
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image
from diffusers.models.attention_processor import AttnProcessor2_0
from accelerate import PartialState 


repo_id =  "stabilityai/stable-diffusion-2-1"#"CompVis/stable-diffusion-v1-4"#

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

os.makedirs("generated3/",exist_ok = True)
device = "cuda"
if args.cuda > -1:
     device = f'cuda:{args.cuda}'


torch.manual_seed(1)
guidance_scale = 7.5
caption = "A cat on the table"
pipe = StableDiffusionPipeline.from_pretrained(repo_id)
pipe = pipe.to(device)
pipe.unet.set_attn_processor(AttnProcessor2_0())

image = pipe(caption, guidance_scale=guidance_scale).images[0]
image.save(f"generated3/pipe.png")