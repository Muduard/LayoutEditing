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
lossF = torch.nn.BCELoss()


def diffusion_step2(latents, t, text_embeddings, guidance_scale):
    noise_pred_uncond = unet(latents, t, encoder_hidden_states=text_embeddings[0].unsqueeze(0))["sample"]
    noise_prediction_text = unet(latents, t, encoder_hidden_states=text_embeddings[1].unsqueeze(0))["sample"]
    noise_pred = noise_prediction_text#noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
    
    
    return noise_pred

def sample(prompt, start_step=0, start_latents=None,
           guidance_scale=3.5, num_inference_steps=30,
           num_images_per_prompt=1, do_classifier_free_guidance=True,
           negative_prompt='', device=device, guide = False, 
           mask_index = 0, controller = None, lossF = None, ht = None):
  
    # Encode prompt
    with torch.no_grad():
        text_embeddings = pipe._encode_prompt(
                prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
        )

    # Set num inference steps
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    
    # Create a random starting point if we don't have one already
    if start_latents is None:
        start_latents = torch.randn(1, 4, 64, 64, device=device)
        start_latents *= pipe.scheduler.init_noise_sigma


    hsteps = num_inference_steps // 2

    latents = start_latents.clone()
    if guide:
        latents.requires_grad_(True)
        latents.retain_grad()
    lambd = torch.linspace(1, 0, hsteps)
    for i in tqdm(range(start_step, num_inference_steps)):
        if guide:
            controller.reset()
        t = pipe.scheduler.timesteps[i]
        if i < num_inference_steps // 2: latents.requires_grad_(True)
        latents = pipe.scheduler.scale_model_input(latents, t)
        
        '''noise_pred_uncond = unet(latents, t, encoder_hidden_states=text_embeddings[0].unsqueeze(0))["sample"]
        noise_prediction_text = unet(latents, t, encoder_hidden_states=text_embeddings[1].unsqueeze(0))["sample"]
        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
        
        # Instead, let's do it ourselves:
        prev_t = max(1, t.item() - (1000//num_inference_steps)) # t-1
        alpha_t = pipe.scheduler.alphas_cumprod[t.item()]
        alpha_t_prev = pipe.scheduler.alphas_cumprod[prev_t]
        predicted_x0 = (latents - (1-alpha_t).sqrt()*noise_pred) / alpha_t.sqrt()
        direction_pointing_to_xt = (1-alpha_t_prev).sqrt()*noise_pred
        latents = alpha_t_prev.sqrt()*predicted_x0 + direction_pointing_to_xt'''
        
        if guide and 2 <= i < num_inference_steps // 2:
            noise_pred = diffusion_step2(latents, t, text_embeddings, guidance_scale)
            #latents2, _ = diffusion_step(unet, scheduler, controller, latents, text_embeddings, t, guidance_scale)
            #latents2 = pipe.scheduler.step(noise_pred, t, latents).prev_sample
            #latents2 = controller.step_callback(latents2)
            attention_maps16, _ = get_cross_attention([prompt], controller, res=32, from_where=["up", "down"])
            
            attention_maps = attention_maps16.to(torch.float) 
            s_hat = attention_maps[:,:,mask_index] 
            original_latents = torch.load(f'inversion/{i}.pt')
            #o_prev = original_latents - latents
            
            save_tensor_as_image(s_hat,"loss_attn.png", plot = True)
            l1 = lossF(s_hat,ht) + lossF(s_hat / torch.linalg.norm(s_hat, ord=np.inf), ht)
            #l2 = 100 * lossM(latents2, original_latents * (1-mask) + (mask) * latents2)
            print(l1)
            #print(l2)
            loss = l1 #+ l2
            loss.backward()
            
            grad_x = latents.grad / torch.abs(latents.grad).max()#/ torch.linalg.norm(latents.grad)#torch.abs(latents.grad).max()
            eta = 0.5
            #latents = latents2 - eta * lambd[i]  * grad_x #+ o_prev
            # Instead, let's do it ourselves:
            prev_t = max(1, t.item() - (1000//num_inference_steps)) # t-1
            alpha_t = pipe.scheduler.alphas_cumprod[t.item()]
            alpha_t_prev = pipe.scheduler.alphas_cumprod[prev_t]

            predicted_x0 = (latents - (1-alpha_t).sqrt()*noise_pred) / alpha_t.sqrt()
            noise_pred = noise_pred - eta *  (1-alpha_t_prev).sqrt() * grad_x
            
            direction_pointing_to_xt = (1-alpha_t_prev).sqrt()*noise_pred
            latents = alpha_t_prev.sqrt()*predicted_x0 + direction_pointing_to_xt
            #latents = original_latents * (1-mask) + (mask) * latents2
            text_embeddings = text_embeddings.detach()
            latents = latents.detach()
            
        else:
            with torch.no_grad():
                latents, _ = diffusion_step(unet, scheduler, controller, latents, context, t, guidance_scale)
            #latents = latents.detach()
            #latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample
        

    # Post-processing
    latents = latents.detach()
    
    #images = pipe.image_processor.postprocess(latents)
    images = latent2image(vae, latents)
    #images = pipe.numpy_to_pil(images)
    return images






image = Image.open(f'examples/img1.png').convert('RGB').resize((512,512))
prompt = "a cat sitting next to a mirror"

context = compute_embeddings(tokenizer, text_encoder, device, 1, prompt)
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
if args.guide:
    h = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
    new_attn = get_multi_level_attention_from_average(h, device)
    hr = cv2.resize(h, (args.res_size, args.res_size))
    cv2.imwrite(args.out_path + "hr.png", hr)
    ht = torch.tensor(cv2.resize(h, (args.res_size, args.res_size)), dtype=torch.float, device=device) / 255 
    ht.requires_grad = False


controller = AttentionStore()
register_attention_control(unet, controller, None)
for param in unet.parameters():
    param.requires_grad = False
lossF = torch.nn.BCELoss()
with torch.no_grad():
    inverted_latents = ddim_invert(unet, scheduler, l, context, guidance_scale, 50, 
                                controller=controller, ht=ht,lossF=lossF,mask_index=2,
                                prompt=prompt,guide=False)

if not args.guide:
    attention_maps16, _ = get_cross_attention([prompt], controller, res=16, from_where=["up", "down"])
    save_tensor_as_image(attention_maps16[:, :, mask_index],"original_mask.png")

start_step = 0
#rec_latents = latent2image(inverted_latents[-1].unsqueeze(0)
lossM = torch.nn.MSELoss()
mask = torch.ones_like(inverted_latents[-1])
if args.guide:
    original_mask = cv2.imread(f'original_mask.png', cv2.IMREAD_GRAYSCALE)
    original_mask = torch.tensor(cv2.resize(original_mask, (args.res_size, args.res_size)), dtype=torch.float, device=device) / 255 
    heatmap = cv2.resize(ht.cpu().numpy() + original_mask.cpu().numpy(), (64,64))
    #heatmap = heatmap / heatmap.max()
    mask = torch.tensor(heatmap,device=device, dtype=MODEL_TYPE)
    save_tensor_as_image(mask,"mask.png")



rec_image = sample(prompt,start_latents=inverted_latents[-(start_step+1)][None][0], \
    start_step=start_step, num_inference_steps=50,controller=controller, ht=ht,
    lossF=lossF,mask_index=2,guide=args.guide)
#print(rec_image.shape)
cv2.imwrite("a.png",rec_image)
