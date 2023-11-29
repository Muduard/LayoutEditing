import os 
import torch
from diffusers import DiffusionPipeline, DDIMScheduler
from tqdm import tqdm
import matplotlib.pyplot as plt
from ptp_utils import AttentionStore,diffusion_step,get_multi_level_attention_from_average,register_attention_control, get_cross_attention, show_cross_attention,latent2image
import cv2
import argparse
from accelerate import Accelerator
import numpy as np
import torch.nn.functional as F

from embedder_model import EmbedderModel
device = "cuda:1"
timesteps = 20

GUIDANCE_SCALE = 7.5
MAX_NUM_WORDS = 77
#device = "cuda"#torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
repo_id =  "runwayml/stable-diffusion-v1-5"#"CompVis/stable-diffusion-v1-4"#

pipe = DiffusionPipeline.from_pretrained(repo_id, torch_dtype=torch.float16).to(device)#DiffusionPipeline.from_pretrained(repo_id, torch_dtype=torch.float16,use_safetensors=True).to(device)
pipe.scheduler = DDIMScheduler.from_config(
    pipe.scheduler.config
)
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

args = parser.parse_args()

os.makedirs(args.out_path,exist_ok = True)

LOW_RESOURCE = True

h = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
new_attn = get_multi_level_attention_from_average(h, device)
pipe.unet.embed_proj = False
controller = AttentionStore()
register_attention_control(pipe, controller)
hr = cv2.resize(h, (args.res_size, args.res_size))
cv2.imwrite(args.out_path + "hr.png", hr)
ht = torch.tensor(cv2.resize(h, (args.res_size, args.res_size)), dtype=torch.float, device=device) / 255 / 5



prompts = ["Cat on table"]
timesteps = 20
pipe.scheduler.set_timesteps(timesteps)
batch_size = 1
torch.manual_seed(42)
noise = torch.randn((batch_size, 4, 64, 64), dtype=torch.float16, device=device) 

latents = noise

text_input = pipe.tokenizer(
        prompts[0],
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )

text_emb = pipe.text_encoder(text_input.input_ids.to(device))[0]

max_length = text_input.input_ids.shape[-1]
uncond_input = pipe.tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
)
uncond_embeddings = pipe.text_encoder(uncond_input.input_ids.to(device))[0]

#delta_emb = torch.zeros_like(text_emb)
#uncond_embeddings.retain_grad()
#delta_emb.requires_grad_(True)
#delta_emb.retain_grad()
context = [uncond_embeddings, text_emb]
guidance_scale = 7.5


for param in pipe.unet.parameters():
    param.requires_grad = False


grad_params = []



lossF = torch.nn.BCELoss()#BCELoss()#MSELoss()
latents.requires_grad = False


def reset_grad(a):
    a = a.detach()
    a.requires_grad_(True)
    a.retain_grad()
    return a


step = 0

#with torch.autograd.detect_anomaly():
def grad_em(net_, count, place_in_unet, module_name=None):
            if module_name in ["emb_model"]:
                for param in net_.parameters():
                    param.requires_grad = True
                    grad_params.append(param)
            elif hasattr(net_, 'children'):
                for k, net__ in net_.named_children():
                    grad_em(net__, count, place_in_unet, module_name=k)
        

grad_em(pipe.unet, 0, None)
grad_params = list(filter(lambda p: p.requires_grad, pipe.unet.parameters()))

optimizer = torch.optim.SGD(grad_params, args.lr)
for t in tqdm(pipe.scheduler.timesteps):
    
    if  1 < step < 10:
        nepochs = args.epochs if step==7 else 30
        bar = tqdm(range(nepochs))
        flag = True
        for i in bar:
            controller.reset()
            optimizer.zero_grad()
            latents2 = diffusion_step(pipe, controller, latents, context, t, guidance_scale, train = True)
            
            attention_maps, out_save16 = get_cross_attention(prompts, controller, res=args.res_size, from_where=["up", "down"])
            
            attention_maps = attention_maps.to(torch.float)
            attn_replace = torch.clone(attention_maps)
            attn_replace[:, :, 1] = ht
            attn = attention_maps[:, :, 1]
            if (i) % 500 == 0:
                save_attn = (attn / 2 + 0.5).clamp(0, 1)
                save_attn = attn.detach().cpu().numpy()
                save_attn = (save_attn * 255).astype(np.uint8)
                plt.imshow(save_attn)
                plt.show()
                plt.savefig("loss_attn.png")
            losses = []
            for j in range(attention_maps.shape[-1]):
                
                l = lossF(attention_maps[:,:,j],attn_replace[:,:,j]) \
                    + lossF(attention_maps[:,:,j] \
                            / torch.linalg.norm(attention_maps[:,:,j], ord=np.inf), attn_replace[:,:,j]  )
                
                losses.append(l)
            
            
            loss = sum(losses)
            loss.backward()
            if loss < 0.01:
                print(loss)
                break
            bar.set_description(f"loss: {loss.detach()}")
            #for p in grad_params:
            #    p.grad /= torch.linalg.norm(p, ord=np.inf)
            optimizer.step()
            context[0] = context[0].detach()
            context[1] = context[1].detach()
            
        latents = latents2
        
    else:
        with torch.no_grad():
            latents = diffusion_step(pipe, controller, latents, context, t, guidance_scale, train=False)
    
    step += 1

def opt_zero():

    for t in tqdm(pipe.scheduler.timesteps):
            controller = AttentionStore()
            register_attention_control(pipe, controller)

def save_net(net_, count, place_in_unet, module_name=None):
            if module_name in ["emb_model"]:
                torch.save(net_.state_dict(), f'net_{count}.pt')
                count += 1
                return count
            elif hasattr(net_, 'children'):
                for k, net__ in net_.named_children():
                    count = save_net(net__, count, place_in_unet, module_name=k)


image = latent2image(pipe.vae, latents)
plt.imshow(image)
plt.savefig(args.out_path + "image.png")
show_cross_attention(pipe, prompts, controller, res=args.res_size, from_where=["up","down"], out_path=args.out_path + "attns.png")
save_net(pipe.unet, 0, None)