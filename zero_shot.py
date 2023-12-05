import os 
import torch
from diffusers import DDIMScheduler,AutoencoderKL,UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm import tqdm
import matplotlib.pyplot as plt
from ptp_utils import AttentionStore,diffusion_step,get_multi_level_attention_from_average,register_attention_control, get_cross_attention, show_cross_attention,latent2image,save_tensor_as_image
import cv2
import argparse
from accelerate import Accelerator
import numpy as np
import torch.nn.functional as F



#device = "cuda"#torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
repo_id =  "runwayml/stable-diffusion-v1-5"#"CompVis/stable-diffusion-v1-4"#

#pipe = DiffusionPipeline.from_pretrained(repo_id, torch_dtype=torch.float16).to(device)#DiffusionPipeline.from_pretrained(repo_id, torch_dtype=torch.float16,use_safetensors=True).to(device)

#pipe.enable_xformers_memory_efficient_attention()
#pipe.unet = torch.nn.DataParallel(pipe.unet)
parser = argparse.ArgumentParser(description='Stable Diffusion Layout Editing')
parser.add_argument('--mask', default="attentions/cat2.png",
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

args = parser.parse_args()

path_original = args.out_path + "original/"
os.makedirs(args.out_path,exist_ok = True)
os.makedirs(path_original,exist_ok = True)

device = "cpu"
if args.cuda > -1:
     device = f'cuda:{args.cuda}'

vae = AutoencoderKL.from_pretrained(repo_id, subfolder="vae", torch_dtype=torch.float16).to(device)
tokenizer = CLIPTokenizer.from_pretrained(repo_id, subfolder="tokenizer", torch_dtype=torch.float16)
text_encoder = CLIPTextModel.from_pretrained(repo_id, subfolder="text_encoder", torch_dtype=torch.float16).to(device)
unet = UNet2DConditionModel.from_pretrained(repo_id, subfolder="unet", torch_dtype=torch.float16).to(device)

scheduler = DDIMScheduler.from_pretrained(repo_id,subfolder="scheduler", torch_dtype=torch.float16)


mask_index = 2
timesteps = 50
h = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
new_attn = get_multi_level_attention_from_average(h, device)

controller = AttentionStore()
register_attention_control(unet, controller, new_attn)
hr = cv2.resize(h, (args.res_size, args.res_size))
cv2.imwrite(args.out_path + "hr.png", hr)
ht = torch.tensor(cv2.resize(h, (args.res_size, args.res_size)), dtype=torch.float, device=device) / 255 
#x_0 = torch.zeros((64,64), device=device, dtype=torch.float16)
#if args.guide:
#    x_0 = cv2.imread(args.out_path + "x0.png")
#    x_0 = torch.tensor(x_0, dtype=torch.float, device=device) / 255

prompts = ["A cat with a city in the background"]
timesteps = 50
scheduler.set_timesteps(timesteps)
x_0 = torch.load(f'{path_original}{timesteps}.pt',map_location = device)
batch_size = 1
torch.manual_seed(3)
noise = torch.randn((batch_size, 4, 64, 64), dtype=torch.float16, device=device) 

latents = noise

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


context = [uncond_embeddings, text_emb]
guidance_scale = 3

for param in unet.parameters():
    param.requires_grad = False


latents.requires_grad_(True)
latents.retain_grad()
def reset_grad(a):
    a = a.detach()
    a.requires_grad_(True)
    a.retain_grad()
    return a

ht.requires_grad = False
step = 0
lambd = torch.linspace(1, 0, timesteps // 2)
#with torch.autograd.detect_anomaly():
lossF = torch.nn.BCELoss()#BCELoss()#MSELoss()
lossM = torch.nn.MSELoss(reduction="mean")
m = torch.nn.Sigmoid()

mask = torch.ones_like(noise)
if args.guide:
    original_mask = cv2.imread(f'{path_original}26.png', cv2.IMREAD_GRAYSCALE)
    original_mask = torch.tensor(cv2.resize(original_mask, (args.res_size, args.res_size)), dtype=torch.float, device=device) / 255 
    heatmap = cv2.resize(ht.cpu().numpy() + original_mask.cpu().numpy(), (64,64))
    heatmap = heatmap / heatmap.max()
    mask = torch.tensor(heatmap,device=device, dtype=torch.float16)
    save_tensor_as_image(mask,"mask.png", plot = True)


theta = torch.linspace(0.7, 1, timesteps//2)
sigma = torch.linspace(0.1, 0.1, timesteps//2)
#mask = torch.ones((64,64), device=noise.device, dtype=torch.float16)
for t in tqdm(scheduler.timesteps):    
    x_k = torch.load(f'{path_original}{step}.pt').to(dtype=torch.float16, device=device)
    to_train = False
    if  0 <= step < timesteps // 2 and args.guide:
        if step < timesteps // 4:
            to_train = True
        else: 
            to_train = False
        #optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, pipe.unet.parameters()), args.lr)
        controller.reset()
        #x_k = scheduler.scale_model_input(x_0, step)
        
        latents2 = diffusion_step(unet, scheduler, controller, latents, context, t, guidance_scale, xt = x_k, m = mask, train = to_train,sigma=sigma[step])
        
        attention_maps16, _ = get_cross_attention(prompts, controller, res=16, from_where=["up", "down"])
        attention_maps32, _ = get_cross_attention(prompts, controller, res=32, from_where=["up", "down"])
        attention_maps64, _ = get_cross_attention(prompts, controller, res=64, from_where=["up", "down"])
        
        attention_maps = attention_maps16.to(torch.float) 
        
        s_hat = attention_maps[:,:,mask_index]  #torch.mean(attention_maps,dim=-1)
        
        attn_replace = torch.clone(attention_maps)
        attn_replace[:, :, mask_index] = ht
        save_tensor_as_image(attention_maps[:,:,mask_index],"loss_attn.png", plot = True)
        losses = []
        
        
        '''for j in range(attention_maps.shape[-1]): 
            #l = lossF(s_hat,ht) \
            #    + lossF(s_hat / torch.linalg.norm(s_hat, ord=np.inf), ht)
            l = lossF(attention_maps[:,:,j],attn_replace[:,:,j]) \
                + lossF(attention_maps[:,:,j] / torch.linalg.norm(attention_maps[:,:,j], ord=np.inf), attn_replace[:,:,j])
            losses.append(l)'''

        #latents_original = torch.load(f'{path_original}{step}.pt').to(dtype=torch.float16, device=device)
        #region_diff = torch.ones(latents2.shape, dtype=torch.float16, device=device) - torch.abs(latents2 - latents_original)
        #print(torch.min(region_diff))
        #l = lossF(s_hat,ht) + 0.1 *lossM(latents2, latents_original)
        #latents2 = latents2 * theta[step] + latents_original * (1 - theta[step])
        l1 = lossF(m(s_hat),ht) \
            + lossF(s_hat / torch.linalg.norm(s_hat, ord=np.inf), ht)
        #l2 = 1 * lossM(latents2,region_diff * latents_original)
        #l3 = lossF(1 - s_hat, 1 - ht)
        #print(l1)
        #print(l2)
        loss = l1 #+ l2#sum(losses)
        loss.backward()
        #if loss < 0.01:
        #    print(loss)
        #    break
        
        context[0] = context[0].detach()
        context[1] = context[1].detach()
        
        grad_x = latents.grad / torch.abs(latents.grad).max()#/ torch.linalg.norm(latents.grad)#torch.abs(latents.grad).max()
        #grad_x[grad_x != grad_x] = 0
        #eta = 0.01
        eta = 0.5
        latents = latents2 - eta * lambd[step] * grad_x
        #latents = latents * theta[step] + latents_original * (1 - theta[step])
        
        #latents = (theta[step]) * latents + (1 - theta[step]) * torch.randn_like(latents)
        latents = latents.detach()
        latents.requires_grad_(True)
    else:
        with torch.no_grad():
            to_train = False
            if not args.guide:
                torch.save(latents, f'{path_original}{step}.pt')
                if step == timesteps // 2 + 1:
                    attention_maps16, _ = get_cross_attention(prompts, controller, res=16, from_where=["up", "down"])
                    attn =  attention_maps16[:,:,mask_index]
                    attn = attn / attn.max()
                    save_tensor_as_image(attn,f"{path_original}{step}.png")
            else:
                to_train = True
            latents = diffusion_step(unet,scheduler, controller, latents, context, t, guidance_scale, xt=x_k, m = mask, train=False)

    step += 1


torch.save(latents, f'{path_original}{step}.pt')
image = latent2image(vae, latents.detach())
plt.imshow(image)
plt.savefig(args.out_path + "image.png")
show_cross_attention(tokenizer, prompts, controller, res=args.res_size, from_where=["up","down"], out_path=args.out_path + "attns.png")