
import os 
import torch
from tqdm import tqdm
from ptp_utils import diffusion_step,latent2image, lcm_diffusion_step, get_guidance_scale_embedding, compute_embeddings

from guide_utils import Guide
from PIL import Image

def guide_diffusion(scheduler, unet, vae, latents, context, device, guidance_scale, diffusion_type, timesteps, guide_flag, masks, mask_indexes, resolution, out_path, loss_type="l2", eta=0.15, start_step=0, glue="concat", pww=0):
    
    lossM = torch.nn.MSELoss()
    lossL1 = torch.nn.L1Loss()

    if diffusion_type == "LCM":
        w = torch.tensor(guidance_scale - 1).repeat(1).to(device=device, dtype=latents.dtype)
        w_embedding = get_guidance_scale_embedding(w, embedding_dim=unet.config.time_cond_proj_dim)
    if guide_flag:
        guide = Guide(context, masks, mask_indexes, resolution, device, unet.dtype, guidance_scale, None, diffusion_type, init_type="null", glue=glue,pww=pww)
        guide.register_hook(unet,0,"")
    
    step = 0
    lambd = torch.linspace(1, 0, timesteps // 2)
    
    for t in tqdm(scheduler.timesteps): 
        if step >= start_step:
            if step < timesteps // 2 and guide_flag:
                context = context.detach()
                latents = latents.detach()
                latents.requires_grad_(True)
                if diffusion_type == "LCM":
                    latents2, _ = lcm_diffusion_step(unet, scheduler, None, latents, context, t, w_embedding)
                else:
                    latents2, _ = diffusion_step(unet, scheduler, None, latents, context, t, guidance_scale)
                
                guide.guide()
                
                
                x = guide.outputs
                y = guide.obj_attentions
            
                loss = 0
                
                if loss_type=="l2":
                    loss = lossM(x, y)
                elif loss_type=="l1":
                    loss = lossL1(x,y)
                
                loss.backward()
                
                grad_x = latents.grad / torch.max(torch.abs(latents.grad)) 
                #0.2 SD #0.4 setting lcm
                
                latents = latents2 - eta * lambd[step]  * grad_x
                del latents2
                del grad_x
                latents = latents.to(dtype=unet.dtype)
                guide.reset_step()
                unet.zero_grad()
            else:
                with torch.no_grad():
                    if diffusion_type == "LCM":
                        latents, denoised = lcm_diffusion_step(unet, scheduler, None, latents, context, t, w_embedding)
                    else:
                        latents, _ = diffusion_step(unet, scheduler, None, latents, context, t, guidance_scale)  
            
            latents = latents.to(dtype=unet.dtype)
        step += 1
        
    
    with torch.no_grad():
        image = latent2image(vae, latents.detach())
        image = Image.fromarray(image)
        image.save(out_path)
    if guide_flag:
        guide.reset()
    del latents
    del context
    #del guide
    