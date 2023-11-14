import os 
import torch
from diffusers import DiffusionPipeline
from tqdm import tqdm
import matplotlib.pyplot as plt
from ptp_utils import AttentionStore,diffusion_step,get_multi_level_attention_from_average,register_attention_control, get_cross_attention, show_cross_attention,latent2image
import cv2
import argparse
from accelerate import Accelerator
accelerator = Accelerator()
device = accelerator.device
print(device)
timesteps = 30

GUIDANCE_SCALE = 7.5
MAX_NUM_WORDS = 77
#device = "cuda"#torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
repo_id =  "runwayml/stable-diffusion-v1-5"#"CompVis/stable-diffusion-v1-4"#

pipe = DiffusionPipeline.from_pretrained(repo_id, torch_dtype=torch.float16).to(device)#DiffusionPipeline.from_pretrained(repo_id, torch_dtype=torch.float16,use_safetensors=True).to(device)
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
controller = AttentionStore()#AttentionReplace(new_attn, token_numbers=[0])
register_attention_control(pipe, controller)
hr = cv2.resize(h, (args.res_size, args.res_size))
cv2.imwrite(args.out_path + "hr.png", hr)
ht = torch.tensor(cv2.resize(h, (args.res_size, args.res_size)), dtype=torch.float)



prompts = ["Cat on table"]
timesteps = 30
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
text_emb.retain_grad()
#uncond_embeddings.retain_grad()
context = [uncond_embeddings, text_emb]
guidance_scale = 7.5


for param in pipe.unet.parameters():
    param.requires_grad = False
lossF = torch.nn.BCELoss()#BCELoss()#MSELoss()
latents.requires_grad = False


#def opt_embed():
step = 0
optimizer = torch.optim.SGD([context[1]], args.lr)
m = torch.nn.Sigmoid()
ht = m(ht)
for t in tqdm(pipe.scheduler.timesteps):
    controller = AttentionStore()
    register_attention_control(pipe, controller)
    if step < 20:
        bar = tqdm(range(args.epochs))
        flag = True
        for i in bar:
            
            controller = AttentionStore()
            register_attention_control(pipe, controller)
            optimizer.zero_grad()
            latents2 = diffusion_step(pipe, controller, latents, context, t, guidance_scale)
            attention_maps, out_save16 = get_cross_attention(prompts, controller, res=args.res_size, from_where=["up", "down"])
            attn = attention_maps[:, :, 5].to(dtype=torch.float)
            attention_maps = attention_maps.to(dtype=torch.float)
            #attn_replace = torch.clone(attention_maps)
            #attn_replace[:, :, 5] = ht
            
            loss = lossF(attn, ht)
            
            accelerator.backward(loss, retain_graph=True)
            if loss < 0.1:
                 break
            bar.set_description(f"loss: {loss.detach()}")
                #loss.backward(retain_graph=True)
            
            optimizer.step()
            #ht = torch.tensor(cv2.resize(h, (args.res_size, args.res_size)), dtype=torch.float)
        
    latents = diffusion_step(pipe, controller, latents, context, t, guidance_scale)
    step += 1

def opt_zero():

    for t in tqdm(pipe.scheduler.timesteps):
            controller = AttentionStore()
            register_attention_control(pipe, controller)


image = latent2image(pipe.vae, latents)
plt.imshow(image)
plt.savefig(args.out_path + "image.png")
show_cross_attention(pipe, prompts, controller, res=args.res_size, from_where=["up","down"], out_path=args.out_path + "attns.png")