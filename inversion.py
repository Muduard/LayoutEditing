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
parser.add_argument('--mask', default="cat3.png",
                    help='Target mask of layout editing')
parser.add_argument('--res_size', default=32, type=int,
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

torch.manual_seed(42)

def p_t(x_t, alpha_t, eps_t):
    return (x_t - (1-alpha_t).sqrt()*eps_t) / alpha_t.sqrt()

def d_t(alpha_t_prev, eps_t):
    return (1-alpha_t_prev).sqrt()*eps_t

def sample(start_step=0, start_latents=None,
           guidance_scale=3.5, num_inference_steps=30,device=device):
  

    # Set num inference steps
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    
    # Create a random starting point if we don't have one already
    if start_latents is None:
        start_latents = torch.randn(1, 4, 64, 64, device=device)
        start_latents *= pipe.scheduler.init_noise_sigma

    latents = start_latents.clone()

    for i in tqdm(range(start_step, num_inference_steps)):
        t = pipe.scheduler.timesteps[i]
        latents = pipe.scheduler.scale_model_input(latents, t)
        with torch.no_grad():
            latents, _ = diffusion_step(unet, scheduler, None, latents, context, t, guidance_scale)

    # Post-processing
    latents = latents.detach()
    
    #images = pipe.image_processor.postprocess(latents)
    images = latent2image(vae, latents)
    #images = pipe.numpy_to_pil(images)
    return images




image = Image.open(f'examples/cat_on_table.jpg').convert('RGB').resize((512,512))
prompt = "a cat sitting on a table"

context = compute_embeddings(tokenizer, text_encoder, device, 1, prompt, sd=True)
guidance_scale = 3.5
num_inference_steps = 50
scheduler.set_timesteps(num_inference_steps)
image = to_tensor(image)
with torch.no_grad(): 
     latents = vae.encode(image.unsqueeze(0).to(dtype=MODEL_TYPE,device=device)*2-1)
l = 0.18215 * latents.latent_dist.sample()

mask_index = 2
h = None
new_attn = None
ht = None

controller = AttentionStore()
register_attention_control(unet, controller, None)

mask = torch.ones((1, 4, 64, 64), dtype=MODEL_TYPE, device=device) 

inverted_latents = ddim_invert(unet, scheduler, l, context, guidance_scale, 50, 
                             ht=ht,mask_index=2,
                            prompt=prompt, mask = mask , controller=controller)


attention_maps16, _ = get_cross_attention([prompt], controller, res=16, from_where=["up", "down"])
words = prompt.split()
for mask_index in range(len(words)):
    save_tensor_as_image(attention_maps16[:, :, mask_index+1],f'mask_{words[mask_index]}.png')


start_step = 5

with torch.no_grad():
    rec_image = sample(start_latents=inverted_latents[-(start_step+1)][None][0], \
        start_step=start_step, num_inference_steps=50)
cv2.imwrite("a.png",rec_image)
