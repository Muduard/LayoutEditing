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

from embedder_model import EmbedderModel

GUIDANCE_SCALE = 7.5
MAX_NUM_WORDS = 77
#device = "cuda"#torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
repo_id =  "runwayml/stable-diffusion-v1-5"#"CompVis/stable-diffusion-v1-4"#

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
parser.add_argument('--cuda', default=-1, type=int,
                    help='Cuda device to use')
parser.add_argument('--guide', action=argparse.BooleanOptionalAction)
args = parser.parse_args()

path_original = args.out_path + "original/"
os.makedirs(args.out_path,exist_ok = True)
os.makedirs(path_original,exist_ok = True)

mask_index = 2
timesteps = 50
device = "cpu"
if args.cuda > -1:
     device = f'cuda:{args.cuda}'


vae = AutoencoderKL.from_pretrained(repo_id, subfolder="vae", torch_dtype=torch.float16).to(device)
tokenizer = CLIPTokenizer.from_pretrained(repo_id, subfolder="tokenizer", torch_dtype=torch.float16)
text_encoder = CLIPTextModel.from_pretrained(repo_id, subfolder="text_encoder", torch_dtype=torch.float16).to(device)
unet = UNet2DConditionModel.from_pretrained(repo_id, subfolder="unet", torch_dtype=torch.float16).to(device)

scheduler = DDIMScheduler.from_pretrained(repo_id,subfolder="scheduler", torch_dtype=torch.float16)

h = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
new_attn = get_multi_level_attention_from_average(h, device)
controller = AttentionStore()
register_attention_control(unet, controller, new_attn)
hr = cv2.resize(h, (args.res_size, args.res_size))
cv2.imwrite(args.out_path + "hr.png", hr)
ht = torch.tensor(cv2.resize(h, (args.res_size, args.res_size)), dtype=torch.float, device=device) / 255 / 5


prompts = ["A cat on a sofa"]
timesteps = 50
scheduler.set_timesteps(timesteps)
scheduler.timesteps = scheduler.timesteps.to(device)
batch_size = 1
torch.manual_seed(42)
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

#delta_emb = torch.zeros_like(text_emb)
#uncond_embeddings.retain_grad()
#delta_emb.requires_grad_(True)
#delta_emb.retain_grad()
context = [uncond_embeddings, text_emb]
guidance_scale = 3
original_context = torch.clone(context[1])
context[1].requires_grad_(True)
context[1].retain_grad()

for param in unet.parameters():
    param.requires_grad = False


lossF = torch.nn.BCELoss()#BCELoss()#MSELoss()
latents.requires_grad = False


def reset_grad(a):
    a = a.detach()
    a.requires_grad_(True)
    a.retain_grad()
    return a


step = 0
#em = EmbedderModel().to(dtype=torch.float16,device=device)
#em.train()
#with torch.autograd.detect_anomaly():
'''def grad_em(net_, count, place_in_unet, module_name=None):
            if module_name in ["emb_model"]:
                for param in net_.parameters():
                    param.requires_grad = True
                    grad_params.append(param)
            elif hasattr(net_, 'children'):
                for k, net__ in net_.named_children():
                    grad_em(net__, count, place_in_unet, module_name=k)
        

grad_em(pipe.unet, 0, None)'''
#grad_params = list(filter(lambda p: p.requires_grad, pipe.unet.parameters()))
#context.append(em)
#optimizer = torch.optim.SGD(em.parameters(), args.lr)
theta = torch.linspace(0.7, 1, timesteps//2)
lambd = torch.linspace(1, 0, timesteps // 2)

for t in tqdm(scheduler.timesteps):
    
    if  0 <= step < timesteps // 2 and args.guide:
        #nepochs = args.epochs if step==0 else 30
        #bar = tqdm(range(nepochs))
        #flag = True
        #for i in bar:
        controller.reset()
        #optimizer.zero_grad()
        latents = t * latents + (1 - t) * torch.randn_like(latents)
        latents2 = diffusion_step(unet, scheduler, controller, latents, context, t, guidance_scale, train = True)
        
        attention_maps, out_save16 = get_cross_attention(prompts, controller, res=args.res_size, from_where=["up", "down"])
        
        attention_maps = attention_maps.to(torch.float)
        attn_replace = torch.clone(attention_maps)
        attn_replace[:, :, mask_index] = ht
        attn = attention_maps[:, :, mask_index]
        if (0) % 500 == 0:
            save_attn = (attn / 2 + 0.5).clamp(0, 1)
            save_attn = attn.detach().cpu().numpy()
            save_attn = (save_attn * 255).astype(np.uint8)
            plt.imshow(save_attn)
            plt.show()
            plt.savefig("loss_attn.png")
        losses = []
        '''for j in range(attention_maps.shape[-1]):
            
            l = lossF(attention_maps[:,:,j],attn_replace[:,:,j]) \
                + lossF(attention_maps[:,:,j] \
                        / torch.linalg.norm(attention_maps[:,:,j], ord=np.inf), attn_replace[:,:,j]  )
            
            losses.append(l)'''
        loss = 1 * lossF(attention_maps[:,:,mask_index],ht) 
        
        #loss = sum(losses)
        loss.backward()
        #if loss < 0.01:
        #    print(loss)
        #    break
        #bar.set_description(f"loss: {loss.detach()}")
        #for p in grad_params:
        #    p.grad /= torch.linalg.norm(p, ord=np.inf)
        #optimizer.step()
        grad_emb = context[1].grad / torch.linalg.norm(context[1].grad)
        
        context[1] = context[1] - 5 * lambd[step] * grad_emb
        context[1] = context[1] * theta[step] + original_context * (1 - theta[step])
        
        context[0] = context[0].detach()
        context[1] = context[1].detach()
        context[1].requires_grad_(True)
        eta = 1
        
        #latents_original = torch.load(f'{path_original}{step}.pt').to(dtype=torch.float16, device=device)
        #latents = latents * theta[step] + latents_original * (1 - theta[step])
        latents = latents.detach()
        
    else:
        with torch.no_grad():
            context[1] = original_context
            if not args.guide:
                torch.save(latents, f'{path_original}{step}.pt')
            latents = diffusion_step(unet, scheduler, controller, latents, context, t, guidance_scale, train=False)
    
    step += 1



def save_net(net_, count, place_in_unet, module_name=None):
            if module_name in ["emb_model"]:
                torch.save(net_.state_dict(), f'net_{count}.pt')
                count += 1
                return count
            elif hasattr(net_, 'children'):
                for k, net__ in net_.named_children():
                    count = save_net(net__, count, place_in_unet, module_name=k)


image = latent2image(vae, latents)
plt.imshow(image)
plt.savefig(args.out_path + "image.png")
show_cross_attention(tokenizer, prompts, controller, res=args.res_size, from_where=["up","down"], out_path=args.out_path + "attns.png")
save_net(unet, 0, None)