import os 
import torch
from diffusers import DDIMScheduler,AutoencoderKL,UNet2DConditionModel, LCMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm import tqdm
import matplotlib.pyplot as plt
from ptp_utils import AttentionStore,diffusion_step,get_multi_level_attention_from_average,register_attention_control, get_cross_attention, show_cross_attention,latent2image,save_tensor_as_image, lcm_diffusion_step, get_guidance_scale_embedding
import cv2
import argparse
import numpy as np
import torch.nn.functional as F
from diffusers.models.attention_processor import AttnProcessor2_0
from PIL import Image

parser = argparse.ArgumentParser(description='Stable Diffusion Layout Editing')
parser.add_argument('--mask', default="cat3.png",
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
parser.add_argument("--seed", type=int, default=82)
parser.add_argument("--diffusion_type", type=str, default="LCM")
MODEL_TYPE = torch.float16


args = parser.parse_args()

path_original = args.out_path + "original/"
os.makedirs(args.out_path,exist_ok = True)
os.makedirs(path_original,exist_ok = True)
os.makedirs("attentions/",exist_ok = True)
device = "cpu"
if args.cuda > -1:
     device = f'cuda:{args.cuda}'

if args.diffusion_type == "LCM":
    repo_id = "SimianLuo/LCM_Dreamshaper_v7"
else:
    repo_id = "runwayml/stable-diffusion-v1-5"


vae = AutoencoderKL.from_pretrained(repo_id, subfolder="vae", torch_dtype=MODEL_TYPE).to(device)
tokenizer = CLIPTokenizer.from_pretrained(repo_id, subfolder="tokenizer", torch_dtype=MODEL_TYPE)
text_encoder = CLIPTextModel.from_pretrained(repo_id, subfolder="text_encoder", torch_dtype=MODEL_TYPE).to(device)
unet = UNet2DConditionModel.from_pretrained(repo_id, subfolder="unet", torch_dtype=MODEL_TYPE).to(device)

scheduler = LCMScheduler.from_pretrained(repo_id,subfolder="scheduler", torch_dtype=torch.float16)
unet.set_attn_processor(AttnProcessor2_0())
#torch.compile(unet, mode="reduce-overhead", fullgraph=True)

mask_index = int(args.mask_index[0])
timesteps = 40
h = None
new_attn = None

if args.guide:
    h = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
    original_mask = torch.tensor(cv2.resize(h, (args.res_size, args.res_size)), dtype=torch.float, device=device) / 255 
    new_attn = get_multi_level_attention_from_average(h, device)
    hr = cv2.resize(h, (args.res_size, args.res_size))
    cv2.imwrite(args.out_path + "hr.png", hr)
    ht = torch.tensor(cv2.resize(h, (args.res_size, args.res_size)), dtype=torch.float, device=device) / 255 
    x_0 = torch.load(f'{path_original}{timesteps}.pt',map_location = device)
    ht.requires_grad = False
controller = AttentionStore()
register_attention_control(unet, controller, None)

prompts = [args.prompt]
scheduler.set_timesteps(timesteps, original_inference_steps=50)

batch_size = 1
torch.manual_seed(args.seed)
latents = torch.randn((batch_size, 4, 64, 64), dtype=MODEL_TYPE, device=device) 
text_input = tokenizer(
        prompts[0],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )

text_emb = text_encoder(text_input.input_ids.to(device))[0]

max_length = text_input.input_ids.shape[-1]
uncond_input = tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
)
uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]


#context = torch.cat([uncond_embeddings, text_emb])
context = text_emb
guidance_scale = 8

for param in unet.parameters():
    param.requires_grad = False


latents.requires_grad_(True)
latents.retain_grad()
def reset_grad(a):
    a = a.detach()
    a.requires_grad_(True)
    a.retain_grad()
    return a


step = 0
lambd = torch.linspace(1, 0, timesteps // 2)
#with torch.autograd.detect_anomaly():
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


theta = torch.linspace(0.7, 1, timesteps//2)
sigma = torch.linspace(0.1, 0., timesteps//2)
#mask = torch.ones((64,64), device=noise.device, dtype=MODEL_TYPE)

w = torch.tensor(guidance_scale - 1).repeat(1)
w_embedding = get_guidance_scale_embedding(w, embedding_dim=unet.config.time_cond_proj_dim).to(
    device=device, dtype=latents.dtype
)
for t in tqdm(scheduler.timesteps):    
    if args.guide:
        x_k = torch.load(f'{path_original}{step}.pt').to(dtype=MODEL_TYPE, device=device)
    
    if step < timesteps // 2  and args.guide:
       
        controller.reset()
        context = context.detach()
        latents = latents.detach()
        latents.requires_grad_(True)
        for i in range(args.resampling_steps + 1):#Resampling
            if args.resampling_steps > 0 and i < args.resampling_steps:
                t1 = torch.tensor([step+1], device=device)
                latents = scheduler.add_noise(latents,torch.randn_like(latents),t1)
            else:
                latents2, _ = lcm_diffusion_step(unet, scheduler, controller, latents, context, t, w_embedding)
                #latents2, _ = diffusion_step(unet, scheduler, controller, latents, context, t, guidance_scale)
                
                attention_maps16, _ = get_cross_attention(prompts, controller, res=16, from_where=["up", "down"])
                #attention_maps32, _ = get_cross_attention(prompts, controller, res=32, from_where=["up", "down"])
                #attention_maps64, _ = get_cross_attention(prompts, controller, res=64, from_where=["up", "down"])
                
                attention_maps = attention_maps16.to(torch.float) 
                
                s_hat = attention_maps[:,:,mask_index]  #torch.mean(attention_maps,dim=-1)
                
                
                save_tensor_as_image(attention_maps[:,:,mask_index],"loss_attn.png", plot = True)
                losses = []

                l1 = 1 * lossF(s_hat,ht) + lossF(s_hat / torch.linalg.norm(s_hat, ord=np.inf), ht)
                
                l2 = 1 * lossM(latents2, x_k * (1-mask) + (mask) * latents2)
                #l3 = 0.05 * (x_k - latents2).max()
                l3 = 0.1 * sigma[step] * torch.norm(latents2 - x_k)
                l4 = 1 - 2 * (s_hat * ht).sum() / (s_hat.sum() + ht.sum())
                
                loss = l1#sum(losses)
                loss.backward()
                #print(loss)
                
                grad_x = latents.grad / torch.abs(latents.grad).max()
                eta = 0.5

                latents = latents2 - eta * lambd[step]  * torch.sign(grad_x)
                
    else:
        with torch.no_grad():
            
            if not args.guide:
                torch.save(latents, f'{path_original}{step}.pt')
                if step == timesteps // 2 + 1:
                    attention_maps16, _ = get_cross_attention(prompts, controller, res=16, from_where=["up", "down"])
                    attn =  attention_maps16[:,:,mask_index]
                    attn = attn / attn.max()
                    attn = torch.where(attn < 0.4, -1, 1)
                    save_tensor_as_image(attn,f"{path_original}{step}.png")
            
            latents, _ = lcm_diffusion_step(unet,scheduler, controller, latents, context, t, w_embedding)
            
    step += 1


torch.save(latents, f'{path_original}{step}.pt')
image = latent2image(vae, latents.detach())
image = Image.fromarray(image)
image.save(args.out_path + "image.png")
attention_maps16, _ = get_cross_attention(prompts, controller, res=16, from_where=["up", "down"])
words = prompts[0].split()
save_tensor_as_image(attention_maps16[:, :, mask_index],f'masks/mask_{words[mask_index-1]}.png')
