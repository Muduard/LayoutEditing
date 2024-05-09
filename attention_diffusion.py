import os
import torch
from diffusers import DDIMScheduler,AutoencoderKL,UNet2DConditionModel, LCMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from ptp_utils import compute_embeddings,save_tensor_as_image,get_cross_attention,AttentionStore,register_attention_control,diffusion_step,latent2image
from tqdm import tqdm
from ptp_utils import diffusion_step,latent2image, lcm_diffusion_step, get_guidance_scale_embedding, compute_embeddings
from shutil import rmtree
from guide_utils import Guide
from PIL import Image


def att_diff(scheduler, unet, vae, latents, context, device, guidance_scale, diffusion_type, timesteps, out_path, glue="concat", prompt=""):
    
    if diffusion_type == "LCM":
        w = torch.tensor(guidance_scale - 1).repeat(1).to(device=device, dtype=latents.dtype)
        w_embedding = get_guidance_scale_embedding(w, embedding_dim=unet.config.time_cond_proj_dim)
    controller = AttentionStore()
    register_attention_control(unet, controller)
    for t in tqdm(scheduler.timesteps): 
            
        with torch.no_grad():
            if diffusion_type == "LCM":
                latents, _ = lcm_diffusion_step(unet, scheduler, controller, latents, context, t, w_embedding)
            else:
                latents, _ = diffusion_step(unet, scheduler, controller, latents, context, t, guidance_scale)  
    
        latents = latents.to(dtype=unet.dtype)
    attention_maps16, _ = get_cross_attention([prompt], controller, res=16, from_where=["up", "down"])
    words = prompt.split()
    att_path = "attns/"
    if os.path.exists(att_path):
        rmtree(att_path)
    os.makedirs("attns/")
    for mask_index in range(len(words)):
        save_tensor_as_image(attention_maps16[:, :, mask_index],f'attns/{mask_index}.png')        
    with torch.no_grad():
        image = latent2image(vae, latents.detach())
        image = Image.fromarray(image)
        image.save(out_path)