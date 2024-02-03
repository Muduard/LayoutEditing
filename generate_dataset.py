import os 
import torch
from tqdm import tqdm
from diffusers import DiffusionPipeline
from ptp_utils import diffusion_step,latent2image,compute_embeddings,lcm_diffusion_step
import argparse
import numpy as np
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image
from diffusers.models.attention_processor import AttnProcessor2_0
from accelerate import PartialState 
from PIL import Image
repo_id =  "SimianLuo/LCM_Dreamshaper_v7"#"stabilityai/stable-diffusion-2-1"#"CompVis/stable-diffusion-v1-4"#

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

os.makedirs("generated2/",exist_ok = True)
device = "cuda"
if args.cuda > -1:
     device = f'cuda:{args.cuda}'

'''vae = AutoencoderKL.from_pretrained(repo_id, subfolder="vae",torch_dtype=MODEL_TYPE).to(distributed_state.device)
tokenizer = CLIPTokenizer.from_pretrained(repo_id, subfolder="tokenizer", torch_dtype=MODEL_TYPE)
text_encoder = CLIPTextModel.from_pretrained(repo_id, subfolder="text_encoder", torch_dtype=MODEL_TYPE).to(distributed_state.device)
unet = UNet2DConditionModel.from_pretrained(repo_id, subfolder="unet", torch_dtype=MODEL_TYPE).to(distributed_state.device)'''

#scheduler = DDIMScheduler.from_pretrained(repo_id,subfolder="scheduler", torch_dtype=MODEL_TYPE,prediction_type="v_prediction")
pipe = DiffusionPipeline.from_pretrained(repo_id, torch_dtype=MODEL_TYPE)
pipe = pipe.to(device, torch_dtype=MODEL_TYPE)
num_inference_steps = 8

pipe.unet.set_attn_processor(AttnProcessor2_0())
torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)


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

batch_size = 1


guidance_scale = 8


n_images = len(os.listdir("./generated_lcm"))
print(f"starting from caption {n_images}")

n_images_to_generate = len(captions) - n_images
cap1 = captions[n_images:n_images_to_generate+n_images]

latents_original = torch.randn((batch_size, 4, 64, 64), dtype=MODEL_TYPE, device=device) 

def get_context(prompt):
    print(prompt)
    text_input = pipe.tokenizer(
        prompt,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_emb = pipe.text_encoder(text_input.input_ids.to(device))[0]
    
    #context = torch.cat([uncond_embeddings, text_emb])
    return text_emb

def get_guidance_scale_embedding(w, embedding_dim=512, dtype=torch.float16):
        """
        See https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298

        Args:
            timesteps (`torch.Tensor`):
                generate embedding vectors at these timesteps
            embedding_dim (`int`, *optional*, defaults to 512):
                dimension of the embeddings to generate
            dtype:
                data type of the generated embeddings

        Returns:
            `torch.FloatTensor`: Embedding vectors with shape `(len(timesteps), embedding_dim)`
        """
        assert len(w.shape) == 1
        w = w * 1000.0

        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
        emb = w.to(dtype)[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1))
        assert emb.shape == (w.shape[0], embedding_dim)
        return emb


with torch.no_grad():
    
    i = 0
    w = torch.tensor(guidance_scale - 1).repeat(1)
    w_embedding = pipe.get_guidance_scale_embedding(w, embedding_dim=pipe.unet.config.time_cond_proj_dim).to(
        device=device, dtype=latents_original.dtype
    )
    for caption in tqdm(cap1):
        pipe.scheduler.set_timesteps(num_inference_steps, original_inference_steps=50)
        latents = torch.clone(latents_original) * pipe.scheduler.init_noise_sigma
        context = get_context(prompt=caption)
        for t in tqdm(pipe.scheduler.timesteps):
            latents, denoised = lcm_diffusion_step(pipe.unet, pipe.scheduler, None, latents, context, t, w_embedding)
            #latents, _ = diffusion_step(pipe.unet, pipe.scheduler, None, latents, context, t, guidance_scale)
        image = latent2image(pipe.vae, denoised.detach())
        #image = pipe(caption, num_inference_steps=num_inference_steps, guidance_scale=7.5, lcm_origin_steps=50, output_type="pil").images[0]
        image = Image.fromarray(image)
        image.save(f"generated_lcm/{i}.png")
        i+=1
        #cv2.imwrite(f"generated/{i}.png",image)
        #image.save(f"generated/{i}.png")
        