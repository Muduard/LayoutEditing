import os 
import torch
from diffusers import DDIMScheduler,AutoencoderKL,UNet2DConditionModel, LCMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm import tqdm
import matplotlib.pyplot as plt
from ptp_utils import AttentionStore,get_attn_layers,diffusion_step,get_multi_level_attention_from_average,register_attention_control, get_cross_attention, show_cross_attention,latent2image,save_tensor_as_image, lcm_diffusion_step, get_guidance_scale_embedding, compute_embeddings
import cv2
import argparse
import numpy as np
import torch.nn.functional as F
from diffusers.models.attention_processor import AttnProcessor2_0
from guide_utils import Guide
from PIL import Image
#device = "cuda"#torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
repo_id = "runwayml/stable-diffusion-v1-5" #"SimianLuo/LCM_Dreamshaper_v7"#"runwayml/stable-diffusion-v1-5" #"SimianLuo/LCM_Dreamshaper_v7" #"CompVis/stable-diffusion-v1-4" #"stabilityai/stable-diffusion-2-1"#"CompVis/stable-diffusion-v1-4"#"runwayml/stable-diffusion-v1-5"#"CompVis/stable-diffusion-v1-4"#

parser = argparse.ArgumentParser(description='Stable Diffusion Layout Editing')
parser.add_argument('--mask', default="mask_cat.png",
                    help='Target mask of layout editing')
parser.add_argument('--res', default=16, type=int,
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
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--diffusion_type", type=str, default="LCM")
MODEL_TYPE = torch.float16
sl = False

args = parser.parse_args()

path_original = args.out_path + "original/"
os.makedirs(args.out_path,exist_ok = True)
os.makedirs(path_original,exist_ok = True)
os.makedirs("attentions/",exist_ok = True)
device = "cpu"
if args.cuda > -1:
     device = f'cuda:{args.cuda}'

vae = AutoencoderKL.from_pretrained(repo_id, subfolder="vae", torch_dtype=MODEL_TYPE).to(device)
tokenizer = CLIPTokenizer.from_pretrained(repo_id, subfolder="tokenizer", torch_dtype=MODEL_TYPE)
text_encoder = CLIPTextModel.from_pretrained(repo_id, subfolder="text_encoder", torch_dtype=MODEL_TYPE).to(device)
unet = UNet2DConditionModel.from_pretrained(repo_id, subfolder="unet", torch_dtype=MODEL_TYPE).to(device)

scheduler = LCMScheduler.from_pretrained(repo_id,subfolder="scheduler", torch_dtype=torch.float16)
unet.set_attn_processor(AttnProcessor2_0())
#torch.compile(unet, mode="reduce-overhead", fullgraph=True)

mask_index = 5
timesteps = 30
h = None
new_attn = None

if args.guide:
    h = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
    original_mask = torch.tensor(cv2.resize(h, (args.res, args.res)), dtype=torch.float, device=device) / 255 
    new_attn = get_multi_level_attention_from_average(h, device)
    hr = cv2.resize(h, (args.res, args.res))
    cv2.imwrite(args.out_path + "hr.png", hr)
    ht = torch.tensor(cv2.resize(h, (args.res, args.res)), dtype=torch.float, device=device) / 255 
    x_0 = torch.load(f'{path_original}{timesteps}.pt',map_location = device)
    ht.requires_grad = False

prompts = ["Portrait of a man with a futuristic city in the background"]
scheduler.set_timesteps(timesteps, original_inference_steps=50)

batch_size = 1
torch.manual_seed(args.seed)
if not sl:
    latents = torch.randn((batch_size, 4, 64, 64), dtype=MODEL_TYPE, device=device) 
else:
    latents = torch.load("starting_latent.pt",map_location=device).to(dtype=MODEL_TYPE)
context = compute_embeddings(tokenizer, text_encoder, device, batch_size, prompts, sd=False)

guidance_scale = 8

for param in unet.parameters():
    param.requires_grad = False


latents.requires_grad_(True)
latents.retain_grad()
step = 0
lambd = torch.linspace(1, 0, timesteps // 2)

lossF = torch.nn.BCELoss()#BCELoss()#MSELoss()
lossM = torch.nn.MSELoss()

lossKL = torch.nn.KLDivLoss()
m = torch.nn.Sigmoid()

mask = torch.ones_like(latents)
if args.guide:
    
    heatmap = cv2.resize(ht.cpu().numpy() + original_mask.cpu().numpy(), (64,64))
    #heatmap = heatmap / heatmap.max()
    mask = torch.tensor(heatmap,device=device, dtype=MODEL_TYPE)
    save_tensor_as_image(mask,"mask.png", plot = True)

if args.diffusion_type == "LCM":
    w = torch.tensor(guidance_scale - 1).repeat(1).to(device=device, dtype=latents.dtype)
    w_embedding = get_guidance_scale_embedding(w, embedding_dim=unet.config.time_cond_proj_dim)
if args.guide:
    guide = Guide(context, h, mask_index, args.res, device, MODEL_TYPE)
    guide.register_hook(unet,"")
#TODO modularize code
#TODO Test multiple attention layers
#TODO LCM inversion
#TODO show LCM has better consistency than SD
for t in tqdm(scheduler.timesteps): 
    
    if args.guide:
        x_k = torch.load(f'{path_original}{step}.pt').to(dtype=MODEL_TYPE, device=device)
    if step < timesteps // 2 and args.guide:
        context = context.detach()
        latents = latents.detach()
        latents.requires_grad_(True)
        if args.diffusion_type == "LCM":
            latents2, denoised = lcm_diffusion_step(unet, scheduler, None, latents, context, t, w_embedding)
        else:
            latents2, _ = diffusion_step(unet, scheduler, None, latents, context, t, guidance_scale)
        
        guide.guide()
        #l1 = 1 * lossF(s_hat,ht) + lossF(s_hat / torch.linalg.norm(s_hat, ord=np.inf), ht)
        l2 = 0.1 * lossM(latents2, x_k * (1-mask) + (mask) * latents2)
        #l3 = 0.05 * (x_k - latents2).max()
        l3 = 0.15 * lambd[step] * torch.norm(latents2 - x_k)
        #l4 = 1 - 2 * (s_hat * ht).sum() / (s_hat.sum() + ht.sum())
        
        l1 = lossM(guide.outputs, guide.obj_attentions)
        loss = l1
        loss.backward()
        print(loss)
        grad_x = latents.grad 
        eta = 0.4
        latents = latents2 - eta * lambd[step]  * torch.sign(grad_x)
        guide.reset_step()
    else:
        with torch.no_grad():
            if not args.guide:
                torch.save(latents, f'{path_original}{step}.pt')
            latents, denoised = lcm_diffusion_step(unet,scheduler, None, latents, context, t, w_embedding)
    latents = latents.to(dtype=MODEL_TYPE)
    step += 1


torch.save(latents, f'{path_original}{step}.pt')
image = latent2image(vae, denoised.detach())
image = Image.fromarray(image)
image.save(args.out_path + "image.png")

#show_cross_attention(tokenizer, prompts, controller, res=args.res_size, from_where=["up","down"], out_path=args.out_path + "attns.png")