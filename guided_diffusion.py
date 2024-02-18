
import os 
import torch
from tqdm import tqdm
from ptp_utils import diffusion_step,latent2image, lcm_diffusion_step, get_guidance_scale_embedding, compute_embeddings

from guide_utils import Guide
from PIL import Image

def guide_diffusion(scheduler, unet, vae, latents, context, device, guidance_scale, diffusion_type, timesteps, guide_flag, mask, mask_index, resolution, out_path, loss_type="mse", eta=0.15):
    
    lossM = torch.nn.MSELoss()
    

    if diffusion_type == "LCM":
        w = torch.tensor(guidance_scale - 1).repeat(1).to(device=device, dtype=latents.dtype)
        w_embedding = get_guidance_scale_embedding(w, embedding_dim=unet.config.time_cond_proj_dim)
    if guide_flag:
        guide = Guide(context, mask, mask_index, resolution, device, unet.dtype, guidance_scale, None, diffusion_type)
        guide.register_hook(unet,0,"")
    

    latents.requires_grad_(True)
    latents.retain_grad()
    step = 0
    lambd = torch.linspace(1, 0, timesteps // 2)

    for t in tqdm(scheduler.timesteps): 
        
        if step < timesteps // 2 and guide_flag:
            context = context.detach()
            latents = latents.detach()
            latents.requires_grad_(True)
            if diffusion_type == "LCM":
                latents2, _ = lcm_diffusion_step(unet, scheduler, None, latents, context, t, w_embedding)
            else:
                latents2, _ = diffusion_step(unet, scheduler, None, latents, context, t, guidance_scale)
            
            guide.guide()
            
            if diffusion_type == "SD":
                x = guide.outputs[guide.count:]
                y = guide.obj_attentions[1]
            else: 
                x = guide.outputs
                y = guide.obj_attentions
            l1 = 0
            if loss_type == "cosine":
                for l in range(x.shape[0]):
                    a = x[l].flatten()
                    b = y[l].flatten()
                    l1 += 1 - torch.dot(a, b) / (a.norm() * b.norm())
            elif loss_type=="mse":
                l1 = lossM(x, y)
            elif loss_type== "trace":
                for l in range(x.shape[0]):
                    l1 += torch.trace((x[l] / x[l].max()) 
                                        @ (y[l].T / y[l].max()))
                    
                #l1 = lossKL(torch.log(guide.outputs + eps), guide.obj_attentions)
            loss = l1 #+ l2
            loss.backward()
            
            
            grad_x = latents.grad #/ torch.max(torch.abs(latents.grad)) 
             #0.2 SD #0.4 setting lcm
            latents = latents2 - eta * lambd[step]  * torch.sign(grad_x)
            del latents2
            
            guide.reset_step()
        else:
            with torch.no_grad():
                if diffusion_type == "LCM":
                    latents, denoised = lcm_diffusion_step(unet, scheduler, None, latents, context, t, w_embedding)
                else:
                    latents, _ = diffusion_step(unet, scheduler, None, latents, context, t, guidance_scale)
                
        latents = latents.to(dtype=unet.dtype)
        step += 1
        torch.cuda.memory._dump_snapshot()

    with torch.no_grad():
        image = latent2image(vae, latents.detach())
        image = Image.fromarray(image)
        image.save(out_path)
    del latents
    