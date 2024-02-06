import os 
import torch
from diffusers import DDIMScheduler,AutoencoderKL,UNet2DConditionModel, LCMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm import tqdm
import matplotlib.pyplot as plt
from ptp_utils import AttentionStore,get_attn_layers,diffusion_step,get_multi_level_attention_from_average,register_attention_control, get_cross_attention, show_cross_attention,latent2image,save_tensor_as_image, lcm_diffusion_step, get_guidance_scale_embedding, register_hook
import cv2
import argparse
import numpy as np
import torch.nn.functional as F
from diffusers.models.attention_processor import AttnProcessor2_0


#device = "cuda"#torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
repo_id ="SimianLuo/LCM_Dreamshaper_v7" #"CompVis/stable-diffusion-v1-4" #"stabilityai/stable-diffusion-2-1"#"CompVis/stable-diffusion-v1-4"#"runwayml/stable-diffusion-v1-5"#"CompVis/stable-diffusion-v1-4"#

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

MODEL_TYPE = torch.float16


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
torch.compile(unet, mode="reduce-overhead", fullgraph=True)

mask_index = 2
timesteps = 20
h = None
new_attn = None

if args.guide:
    #original_mask = cv2.imread(f'{args.mask}', cv2.IMREAD_GRAYSCALE)
    #original_mask = torch.tensor(cv2.resize(original_mask, (args.res_size, args.res_size)), dtype=torch.float, device=device) / 255 
    #new_mask = torch.roll(original_mask,shifts=6, dims=1)
    #new_mask = torch.where(new_mask < 0.4, -1, 1)
    #save_tensor_as_image(new_mask, args.mask)
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
register_hook(unet, 0, '')

#x_0 = torch.zeros((64,64), device=device, dtype=MODEL_TYPE)
#if args.guide:
#    x_0 = cv2.imread(args.out_path + "x0.png")
#    x_0 = torch.tensor(x_0, dtype=torch.float, device=device) / 255

prompts = ["A cat on the table"]
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

attn_layers = get_attn_layers(unet)

obj_attentions = []

def reshape_heads_to_batch_dim(self, tensor):
            batch_size, seq_len, dim = tensor.shape
            head_size = self.heads
            tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
            tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size * head_size, seq_len, dim // head_size)
            return tensor


'''for attn_module in attn_layers:
    if attn_module.to_v.in_features == 768:
        v = attn_module.to_v(context)
        v = reshape_heads_to_batch_dim(attn_module, v)
        obj_attn = torch.zeros()
        out = torch.einsum("b i j, b j d -> b i d", ht, v)
        print(out.shape)
        out = attn_module.to_out(out)
        out = reshape_batch_dim_to_heads(self, out)
        print(out.shape)
        obj_attentions.append(out)'''



#TODO try to change hyperparameters to see if you can do it in only one step
for t in tqdm(scheduler.timesteps):    
    if args.guide:
        x_k = torch.load(f'{path_original}{step}.pt').to(dtype=MODEL_TYPE, device=device)
    
    if step == timesteps // 2 - 1 and args.guide:
        #for b in range(20):
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
                    print(attention_maps.shape)
                    s_hat = attention_maps[:,:,mask_index]  #torch.mean(attention_maps,dim=-1)
                    
                    
                    save_tensor_as_image(attention_maps[:,:,mask_index],"loss_attn.png", plot = True)
                    losses = []
                    
                    
                    '''for j in range(attention_maps.shape[-1]): 
                        #l = lossF(s_hat,ht) \
                        #    + lossF(s_hat / torch.linalg.norm(s_hat, ord=np.inf), ht)
                        l = lossF(attention_maps[:,:,j],attn_replace[:,:,j]) \
                            + lossF(attention_maps[:,:,j] / torch.linalg.norm(attention_maps[:,:,j], ord=np.inf), attn_replace[:,:,j])
                        losses.append(l)'''
                    #TODO loss sulla similarità delle attenzioni, prova a inserire l'attenzione
                    #TODO Cosa succede quando si usano più maschere
                    #TODO lower memory cost
                    if True:
                        l1 = 1 * lossF(s_hat,ht) + lossF(s_hat / torch.linalg.norm(s_hat, ord=np.inf), ht)
                        
                        l2 = 1 * lossM(latents2, x_k * (1-mask) + (mask) * latents2)
                        #l3 = 0.05 * (x_k - latents2).max()
                        l3 = 0.1 * sigma[step] * torch.norm(latents2 - x_k)
                        l4 = 1 - 2 * (s_hat * ht).sum() / (s_hat.sum() + ht.sum())
                        
                        loss = l1#sum(losses)
                        loss.backward()
                        
                        grad_x = latents.grad / torch.abs(latents.grad).max()
                        eta = 0.3
                    else:
                        l1 = 2 * lossF(s_hat,ht)
                        
                        l2 = 1 * lossM(latents2, x_k * (1-mask) + (mask) * latents2)
                        #l3 = 0.05 * (x_k - latents2).max()
                        l3 = 0.1 * sigma[step] * torch.norm(latents2 - x_k)
                        l4 = 1 - 2 * (s_hat * ht).sum() / (s_hat.sum() + ht.sum())
                        
                        loss = l1#sum(losses)
                        loss.backward()
                        
                        grad_x = latents.grad
                        eta = 0.2
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
plt.imshow(image)
plt.savefig(args.out_path + "image.png")
show_cross_attention(tokenizer, prompts, controller, res=args.res_size, from_where=["up","down"], out_path=args.out_path + "attns.png")