import os 
import torch
from tqdm import tqdm
from diffusers import StableDiffusionPipeline,UNet2DConditionModel
from ptp_utils import diffusion_step,latent2image,compute_embeddings,lcm_diffusion_step
import argparse
import numpy as np
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image
from diffusers.models.attention_processor import AttnProcessor2_0
from accelerate import PartialState 
from PIL import Image
repo_id =  "runwayml/stable-diffusion-v1-5"#"SimianLuo/LCM_Dreamshaper_v7"#"stabilityai/stable-diffusion-2-1"#"CompVis/stable-diffusion-v1-4"#
#repo_id = "trained_coco/checkpoint-4500"
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
unet = UNet2DConditionModel.from_pretrained("trained_coco/checkpoint-4500/unet/")
pipe = StableDiffusionPipeline.from_pretrained(
            repo_id,
            unet=unet
        )

os.makedirs("generated3/",exist_ok = True)
device = "cuda"
if args.cuda > -1:
     device = f'cuda:{args.cuda}'


#scheduler = DDIMScheduler.from_pretrained(repo_id,subfolder="scheduler", torch_dtype=MODEL_TYPE,prediction_type="v_prediction")
#pipe = StableDiffusionPipeline.from_pretrained(repo_id, torch_dtype=MODEL_TYPE)
pipe = pipe.to(device, torch_dtype=MODEL_TYPE)
num_inference_steps = 50

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

batch_size = 64


guidance_scale = 32


n_images = len(os.listdir("./generated3"))
print(f"starting from caption {n_images}")

cap1 = np.choice(np.array(captions),5000)
cap2 = []
for i in range(0,len(cap1)-batch_size, batch_size):
     cap2.append(cap1[i:i+batch_size])
latents_original = torch.randn((batch_size, 4, 64, 64), dtype=MODEL_TYPE, device=device) 


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
    #w = torch.tensor(guidance_scale - 1).repeat(1)
    #w_embedding = pipe.get_guidance_scale_embedding(w, embedding_dim=pipe.unet.config.time_cond_proj_dim).to(
    #    device=device, dtype=latents_original.dtype
    #)
    for caption in tqdm(cap2):
        pipe.scheduler.set_timesteps(num_inference_steps)
        latents = torch.clone(latents_original) * pipe.scheduler.init_noise_sigma
        images = pipe(caption).images
        for im in images:
            im.save(f"generated3/{i}.png")
            
            i+=1
        print(i)
        '''context = compute_embeddings(text_encoder=pipe.text_encoder,tokenizer=pipe.tokenizer,
                                     device=device,batch_size=1,sd=True,prompt=caption)
        for t in tqdm(pipe.scheduler.timesteps):
            #latents, denoised = lcm_diffusion_step(pipe.unet, pipe.scheduler, None, latents, context, t, w_embedding)
            latents, _ = diffusion_step(pipe.unet, pipe.scheduler, None, latents, context, t, guidance_scale)
        image = latent2image(pipe.vae, latents.detach())
        #image = pipe(caption, num_inference_steps=num_inference_steps, guidance_scale=7.5, lcm_origin_steps=50, output_type="pil").images[0]
        image = Image.fromarray(image)
        image.save(f"generated3/{i}.png")'''
        
        #cv2.imwrite(f"generated/{i}.png",image)
        #image.save(f"generated/{i}.png")
        