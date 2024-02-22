import torch
from tqdm import tqdm
from ptp_utils import AttentionStore,diffusion_step,get_multi_level_attention_from_average,compute_embeddings,register_attention_control, get_cross_attention, show_cross_attention,latent2image,save_tensor_as_image, lcm_diffusion_step, get_guidance_scale_embedding

import numpy as np
from PIL import Image
import cv2
def zero_shot(scheduler, unet, vae, latents, context,prompts, device, guidance_scale, diffusion_type, timesteps, guide_flag, masks_path, mask_indexes, resolution, out_path, eta=1):
    lossF = torch.nn.BCELoss()
    lambd = torch.linspace(1, 0, timesteps // 2)
    step = 0
    latents.requires_grad_(True)
    latents.retain_grad()
    controller = AttentionStore()
    register_attention_control(unet, controller, None)
    acceptable_masks_indexes = []
    for i in range(len(mask_indexes)):
        if mask_indexes[i] < 77:
            acceptable_masks_indexes.append(i)
    
    mask_indexes = [mask_indexes[i] for i in acceptable_masks_indexes]
    masks = []
    for i in acceptable_masks_indexes:
        masks.append(torch.tensor(cv2.resize(masks_path[i], (resolution, resolution)), 
                                        dtype=torch.float32, device=device) / 255)

    
    if diffusion_type == "LCM":
        w = torch.tensor(guidance_scale - 1).repeat(1).to(device=device, dtype=latents.dtype)
        w_embedding = get_guidance_scale_embedding(w, embedding_dim=unet.config.time_cond_proj_dim).to(
            device=device, dtype=latents.dtype
        )
    for t in tqdm(scheduler.timesteps): 
        t = t.to(device)
        
        if step < timesteps // 2  and guide_flag:
        
            controller.reset()
            context = context.detach()
            latents = latents.detach()
            latents.requires_grad_(True)
            
            if diffusion_type == "LCM":
                latents2, denoised = lcm_diffusion_step(unet, scheduler, controller, latents, context, t, w_embedding)
            else:
                latents2, _ = diffusion_step(unet, scheduler, controller, latents, context, t, guidance_scale)
            
            attention_maps16, _ = get_cross_attention(prompts, controller, res=resolution, from_where=["up", "down"])
            
            attention_maps = attention_maps16.to(torch.float32) 
            losses = []
            
            for i, mask_index in enumerate(mask_indexes):
                s_hat = attention_maps[:,:,mask_index]
                l = lossF(s_hat,masks[i]) + lossF(s_hat / torch.linalg.norm(s_hat, ord=np.inf), masks[i])
                
                losses.append(l)
            #save_tensor_as_image(attention_maps[:,:,mask_index],"loss_attn.png", plot = True)
            
            loss = sum(losses)
            loss.backward()
            
            grad_x = latents.grad / torch.abs(latents.grad).max()
            latents = latents2 - eta * lambd[step]  * grad_x
                
        else:
            with torch.no_grad():
                
                if not guide_flag:
                    
                    if step == timesteps // 2 + 1:
                        attention_maps16, _ = get_cross_attention(prompts, controller, res=16, from_where=["up", "down"])
                        attn =  attention_maps16[:,:,mask_index]
                        attn = attn / attn.max()
                        attn = torch.where(attn < 0.4, -1, 1)
                
                if diffusion_type == "LCM":
                    latents, denoised = lcm_diffusion_step(unet, scheduler, controller, latents, context, t, w_embedding)
                else:
                    latents, _ = diffusion_step(unet, scheduler, controller, latents, context, t, guidance_scale)
        latents = latents.to(dtype=unet.dtype)  
        step += 1

    with torch.no_grad():
        image = latent2image(vae, latents.detach())
        image = Image.fromarray(image)
        image.save(out_path)
    del latents
    '''attention_maps16, _ = get_cross_attention(prompts, controller, res=16, from_where=["up", "down"])
    words = prompts[0].split()
    #save_tensor_as_image(attention_maps16[:, :, mask_index],f'masks/mask_{words[mask_index-1]}.png')
    for mask_index in range(len(words)+1):
        save_tensor_as_image(attention_maps16[:, :, mask_index],f'masks/{mask_index}.png')'''
