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

os.makedirs("generated/",exist_ok = True)
device = "cuda"
if args.cuda > -1:
     device = f'cuda:{args.cuda}'

distributed_state = PartialState()

vae = AutoencoderKL.from_pretrained(repo_id, subfolder="vae",torch_dtype=MODEL_TYPE).to(distributed_state.device)
tokenizer = CLIPTokenizer.from_pretrained(repo_id, subfolder="tokenizer", torch_dtype=MODEL_TYPE)
text_encoder = CLIPTextModel.from_pretrained(repo_id, subfolder="text_encoder", torch_dtype=MODEL_TYPE).to(distributed_state.device)
unet = UNet2DConditionModel.from_pretrained(repo_id, subfolder="unet", torch_dtype=MODEL_TYPE).to(distributed_state.device)
unet.set_attn_processor(AttnProcessor2_0())
torch.compile(unet, mode="reduce-overhead", fullgraph=True)
scheduler = DDIMScheduler.from_pretrained(repo_id,subfolder="scheduler", torch_dtype=MODEL_TYPE,prediction_type="v_prediction")
#pipe = StableDiffusionPipeline.from_pretrained("", scheduler=scheduler)
#pipe = pipe.to(device)


import json
 
# Opening JSON file
f = open('datasets/annotations/captions_val2017.json')
 
# returns JSON object as 
# a dictionary
data = json.load(f)
 
# Iterating through the json
# list
step = 0
file_dict = {}
for i in data['images']:
    file_dict[i['id']] = i['file_name']
captions = []
for i in data['annotations']:
    file_name = file_dict[i['image_id']] 
    caption = i['caption']
    captions.append(caption)

# Closing file
f.close()

timesteps = 50
scheduler.set_timesteps(timesteps)
batch_size = 1


guidance_scale = 7.5
 

n_images = len(os.listdir("./generated"))
print(f"starting from caption {n_images}")

n_images_to_generate = len(captions) - n_images
cap1 = captions[n_images:n_images_to_generate//2+n_images]
cap2 = captions[n_images_to_generate//2+n_images:]

with distributed_state.split_between_processes([cap1, cap2]) as prompts:
    i = 0
    noise = torch.randn((batch_size, 4, 64, 64), dtype=MODEL_TYPE, device=device) * scheduler.init_noise_sigma
    
    for caption in tqdm(prompts[0]):
        
        latents = torch.clone(noise)
        context = compute_embeddings(tokenizer, text_encoder, device, 1, caption)
        for t in scheduler.timesteps:
            with torch.no_grad():
                latents, _ = diffusion_step(unet, scheduler,None, latents, context, t, guidance_scale)
                
        image = latent2image(vae, latents)
        
        image = to_pil_image(image)
        image.save(f"generated/{distributed_state.process_index}_{i}.png")
        i+=1
        #cv2.imwrite(f"generated/{i}.png",image)
        #image.save(f"generated/{i}.png")
        