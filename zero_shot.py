import os 
import torch
from diffusers import DDIMScheduler,AutoencoderKL,UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm import tqdm
import matplotlib.pyplot as plt
from ptp_utils import AttentionStore,diffusion_step,get_multi_level_attention_from_average,register_attention_control, get_cross_attention, show_cross_attention,latent2image
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
x_0 = cv2.imread(args.out_path + "x0.png", cv2.IMREAD_GRAYSCALE)
x_0 = torch.tensor(x_0, dtype=torch.float, device=device) / 255

prompts = ["A cat on a garden with flowers, realistic 4k"]
timesteps = 50
scheduler.set_timesteps(timesteps)

batch_size = 1
torch.manual_seed(2)
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

theta = torch.linspace(0.7, 1, timesteps//2)
mask = torch.zeros_like(x0)
for t in tqdm(scheduler.timesteps):    
    if  0 <= step < timesteps // 2 and args.guide:
        print(t)
        #optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, pipe.unet.parameters()), args.lr)
        controller.reset()
        x_k = scheduler.scale_model_input(x_0, step)
        
        latents2 = diffusion_step(unet, scheduler, controller, latents, context, t, guidance_scale, xt = x_k, m = mask, train = True)
        
        attention_maps16, _ = get_cross_attention(prompts, controller, res=16, from_where=["up", "down"])
        attention_maps32, _ = get_cross_attention(prompts, controller, res=32, from_where=["up", "down"])
        attention_maps64, _ = get_cross_attention(prompts, controller, res=64, from_where=["up", "down"])
        
        attention_maps = attention_maps16.to(torch.float) 
        
        s_hat = attention_maps[:,:,mask_index]  #torch.mean(attention_maps,dim=-1)
        mask = cv2.resize(s_hat.detach().cpu() + hr, x0.shape)
        
        attn_replace = torch.clone(attention_maps)
        attn_replace[:, :, mask_index] = ht
        attn = s_hat.clone()
        save_attn = (attn / 2 + 0.5).clamp(0, 1)
        save_attn = attn.detach().cpu().numpy()
        save_attn = (save_attn * 255).astype(np.uint8)
        plt.imshow(save_attn)
        plt.show()
        plt.savefig("loss_attn.png")
        losses = []
        
        
        '''for j in range(attention_maps.shape[-1]): 
            #l = lossF(s_hat,ht) \
            #    + lossF(s_hat / torch.linalg.norm(s_hat, ord=np.inf), ht)
            l = lossF(attention_maps[:,:,j],attn_replace[:,:,j]) \
                + lossF(attention_maps[:,:,j] / torch.linalg.norm(attention_maps[:,:,j], ord=np.inf), attn_replace[:,:,j])
            losses.append(l)'''

        latents_original = torch.load(f'{path_original}{step}.pt').to(dtype=torch.float16, device=device)
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
        latents = latents * theta[step] + latents_original * (1 - theta[step])
        print(latents.shape)
        #latents = (theta[step]) * latents + (1 - theta[step]) * torch.randn_like(latents)
        latents = latents.detach()
        latents.requires_grad_(True)
    else:
        with torch.no_grad():
            if not args.guide:
                torch.save(latents, f'{path_original}{step}.pt')
            latents = diffusion_step(unet,scheduler, controller, latents, context, t, guidance_scale, train=False)

    step += 1



def save_net(net_, count, place_in_unet, module_name=None):
            if module_name in ["emb_model"]:
                torch.save(net_.state_dict(), f'net_{count}.pt')
                count += 1
                return count
            elif hasattr(net_, 'children'):
                for k, net__ in net_.named_children():
                    count = save_net(net__, count, place_in_unet, module_name=k)


image = latent2image(vae, latents.detach())
plt.imshow(image)
plt.savefig(args.out_path + "image.png")
show_cross_attention(tokenizer, prompts, controller, res=args.res_size, from_where=["up","down"], out_path=args.out_path + "attns.png")