import os 
import torch
from tqdm import tqdm
from ptp_utils import diffusion_step,latent2image, lcm_diffusion_step, get_guidance_scale_embedding, compute_embeddings

from guide_utils import Guide
from PIL import Image


def guide_diffusion_pww(scheduler, unet, vae, latents, context, device, guidance_scale, diffusion_type, timesteps, guide_flag, masks, mask_indexes, resolution, out_path, glue="concat", pww=0):
    
    if diffusion_type == "LCM":
        w = torch.tensor(guidance_scale - 1).repeat(1).to(device=device, dtype=latents.dtype)
        w_embedding = get_guidance_scale_embedding(w, embedding_dim=unet.config.time_cond_proj_dim)
    if guide_flag:
        guide = Guide(context, masks, mask_indexes, resolution, device, unet.dtype, guidance_scale, None, diffusion_type, init_type="null", glue=glue,pww=pww)
        guide.register_hook(unet,0,"")
    
    
    
    for t in tqdm(scheduler.timesteps): 
            if guide_flag:
                context = context.detach()
                latents = latents.detach()
                with torch.no_grad():
                    if diffusion_type == "LCM":
                        latents2, _ = lcm_diffusion_step(unet, scheduler, None, latents, context, t, w_embedding)
                    else:
                        latents2, _ = diffusion_step(unet, scheduler, None, latents, context, t, guidance_scale)
                    
                    guide.guide()

                    latents = latents2
                    del latents2
                    latents = latents.to(dtype=unet.dtype)
                    guide.reset_step()
            else:
                with torch.no_grad():
                    if diffusion_type == "LCM":
                        latents, denoised = lcm_diffusion_step(unet, scheduler, None, latents, context, t, w_embedding)
                    else:
                        latents, _ = diffusion_step(unet, scheduler, None, latents, context, t, guidance_scale)  
            
            latents = latents.to(dtype=unet.dtype)
        
        
    
    with torch.no_grad():
        image = latent2image(vae, latents.detach())
        image = Image.fromarray(image)
        image.save(out_path)
    if guide_flag:
        guide.reset()
    del latents
    del context