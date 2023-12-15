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


def sample(prompt, start_step=0, start_latents=None,
           guidance_scale=3.5, num_inference_steps=30,
           num_images_per_prompt=1, do_classifier_free_guidance=True,
           negative_prompt='', device=device, ht=None):
  
    # Encode prompt
    with torch.no_grad():
        text_embeddings = pipe._encode_prompt(
                prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
        )

    # Set num inference steps
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    print(start_latents.shape)
    # Create a random starting point if we don't have one already
    if start_latents is None:
        start_latents = torch.randn(1, 4, 64, 64, device=device)
        start_latents *= pipe.scheduler.init_noise_sigma


    hsteps = num_inference_steps // 2

    latents = start_latents.clone()
    latents.requires_grad_(True)
    latents.retain_grad()
    lambd = torch.linspace(1, 0, hsteps)
    for i in tqdm(range(start_step, num_inference_steps)):
    
        t = pipe.scheduler.timesteps[i]

        # Expand the latents if we are doing classifier free guidance
        #latent_model_input = torch.cat([latents] * 2)
        #latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        # Predict the noise residual
        #noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        # Perform guidance
        #noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        #noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        noise_pred_uncond = unet(latents, t, encoder_hidden_states=text_embeddings[0].unsqueeze(0))["sample"]
        noise_prediction_text = unet(latents, t, encoder_hidden_states=text_embeddings[1].unsqueeze(0))["sample"]
        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)

        # Normally we'd rely on the scheduler to handle the update step:
        latents2 = pipe.scheduler.step(noise_pred, t, latents).prev_sample
        
        # Instead, let's do it ourselves:
        '''prev_t = max(1, t.item() - (1000//num_inference_steps)) # t-1
        alpha_t = pipe.scheduler.alphas_cumprod[t.item()]
        alpha_t_prev = pipe.scheduler.alphas_cumprod[prev_t]
        predicted_x0 = (latents - (1-alpha_t).sqrt()*noise_pred) / alpha_t.sqrt()
        direction_pointing_to_xt = (1-alpha_t_prev).sqrt()*noise_pred
        latents = alpha_t_prev.sqrt()*predicted_x0 + direction_pointing_to_xt'''
        
        attention_maps16, _ = get_cross_attention([prompt], controller, res=16, from_where=["up", "down"])
        
        attention_maps = attention_maps16.to(torch.float) 
        s_hat = attention_maps[:,:,mask_index] 
        
        l1 = lossF(s_hat,ht) + lossF(s_hat / torch.linalg.norm(s_hat, ord=np.inf), ht)
        l1.backward()
        
        grad_x = latents.grad / torch.abs(latents.grad).max()#/ torch.linalg.norm(latents.grad)#torch.abs(latents.grad).max()
        eta = 0.1
        latents = latents2 - eta * lambd[i - hsteps]  * grad_x
        
        latents = latents.detach()
        latents.requires_grad_(True)

    # Post-processing
    images = pipe.decode_latents(latents)
    images = pipe.numpy_to_pil(images)

    return images






image = Image.open(f'examples/img1.png').convert('RGB')
prompt = "a cat sitting next to a mirror"

context = compute_embeddings(tokenizer, text_encoder, device, 1, prompt)
guidance_scale = 3.5
num_inference_steps = 50
scheduler.set_timesteps(num_inference_steps)

with torch.no_grad(): 
     latents = vae.encode(to_tensor(image).unsqueeze(0).to(dtype=MODEL_TYPE,device=device)*2-1)
l = 0.18215 * latents.latent_dist.sample()
print(l.shape)
mask_index = 2
h = None
new_attn = None

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
with torch.no_grad():
    inverted_latents = ddim_invert(unet, scheduler, l, context, guidance_scale, 50)


start_step = 25
#rec_latents = latent2image(inverted_latents[-1].unsqueeze(0)
print(inverted_latents[0].shape)
rec_image = sample(prompt, start_latents=inverted_latents[0][None][0], \
       start_step=start_step, num_inference_steps=50, ht=ht)[0]

rec_image.save(f"a.png")