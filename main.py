import os 
import torch
from diffusers import DDIMScheduler,AutoencoderKL,UNet2DConditionModel, LCMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from ptp_utils import compute_embeddings,save_tensor_as_image,get_cross_attention,AttentionStore,register_attention_control
import cv2
import argparse
from diffusers.models.attention_processor import AttnProcessor2_0
import json
from guided_diffusion import guide_diffusion
from new_pww import guide_diffusion_pww
import pickle
from zero_shot import zero_shot
from tqdm import tqdm
from natsort import natsorted
from attention_diffusion import att_diff
#device = "cuda"#torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
 #"SimianLuo/LCM_Dreamshaper_v7"#"runwayml/stable-diffusion-v1-5" #"SimianLuo/LCM_Dreamshaper_v7" #"CompVis/stable-diffusion-v1-4" #"stabilityai/stable-diffusion-2-1"#"CompVis/stable-diffusion-v1-4"#"runwayml/stable-diffusion-v1-5"#"CompVis/stable-diffusion-v1-4"#

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
parser.add_argument('--mask_index', type=str, help='List of token indices to move with a mask')
parser.add_argument("--mask_path", type=str, help="Path of masks as image files with the name of the corresponding token")
parser.add_argument("--resampling_steps", type=int, default=0, help="Resample noise for better coherence")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--diffusion_type", type=str, default="LCM")
parser.add_argument("--from_file", type=str, default=None)
parser.add_argument("--loss_type", type=str, default="l2")
parser.add_argument("--eta", type=float, default=0.2)
parser.add_argument("--method", type=str, default="new")
parser.add_argument("--out_dir", type=str, default="test/")
parser.add_argument("--benchmark", type=str, default="eval-filtered")
parser.add_argument("--num_samples", default=5000)
parser.add_argument("--start_step", type=int, default=0)
parser.add_argument("--save_attentions", type=int, default=0)
parser.add_argument("--edit", type=int, default=0)
parser.add_argument("--edit_folder", type=str)
parser.add_argument("--glue", type=str)
parser.add_argument("--pww", type=int, default=0)
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

if args.diffusion_type == "LCM":
    repo_id = "SimianLuo/LCM_Dreamshaper_v7"
else:
    repo_id = "runwayml/stable-diffusion-v1-5"


batch_size = 1
guidance_scale = 3.5

vae = AutoencoderKL.from_pretrained(repo_id, subfolder="vae", torch_dtype=MODEL_TYPE).to(device)
tokenizer = CLIPTokenizer.from_pretrained(repo_id, subfolder="tokenizer", torch_dtype=MODEL_TYPE)
text_encoder = CLIPTextModel.from_pretrained(repo_id, subfolder="text_encoder", torch_dtype=MODEL_TYPE).to(device)
unet = UNet2DConditionModel.from_pretrained(repo_id, subfolder="unet", torch_dtype=MODEL_TYPE).to(device)
for param in unet.parameters():
    param.requires_grad = False

if args.diffusion_type == "LCM":
    timesteps = 50
    scheduler = LCMScheduler.from_pretrained(repo_id,subfolder="scheduler")
else:
    timesteps = 50
    scheduler = DDIMScheduler.from_pretrained(repo_id,subfolder="scheduler")

if args.diffusion_type == "LCM":
    scheduler.set_timesteps(timesteps, original_inference_steps=50)
else:
    scheduler.set_timesteps(timesteps)
if args.from_file == None:
    mask_index = []
    masks = []
    if args.guide:
        if args.edit:
            attns_files = os.listdir(args.edit_folder)
            attns_files = list(map(lambda f: os.path.join(args.edit_folder,f), attns_files))
            attns_files = natsorted(attns_files)
            mask_index.extend(list(range(len(attns_files))))
            
            for f in attns_files:
                masks.append(cv2.imread(f, cv2.IMREAD_GRAYSCALE))
                
        else:
            indices = args.mask_index.split(",")
            
            for i in indices:
                mask_index.append(int(i))
            masks.append(cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE))
    if not sl:
        latents = torch.randn((batch_size, 4, 64, 64), dtype=MODEL_TYPE, device=device) 
    else:
        latents = torch.load("starting_latent.pt",map_location=device).to(dtype=MODEL_TYPE)
    prompts = [args.prompt]
    
    context = compute_embeddings(tokenizer, text_encoder, device, 
                                batch_size, prompts, sd= False if args.diffusion_type =="LCM" else True)

    controller = None
    if args.save_attentions:
        controller = AttentionStore()
        register_attention_control(unet, controller, None)
    guide_diffusion(scheduler, unet, vae, latents, context, device, guidance_scale, \
                args.diffusion_type, timesteps, args.guide, masks, \
                mask_index, args.res, args.out_dir +"1.png", eta=args.eta, start_step=args.start_step)
    if args.save_attentions:
        attention_maps16, _ = get_cross_attention(prompts, controller, res=16, from_where=["up", "down"])
        words = prompts[0].split()
        words.insert(0, "cls")
        
        os.makedirs("attns/", exist_ok=True)
        for mask_index in range(len(words)):
            save_tensor_as_image(attention_maps16[:, :, mask_index],f'attns/{mask_index}_{words[mask_index]}.png')

else:
    
    with open(args.from_file, "r") as f:
        data = json.load(f)['image_data']
        output_dir = args.out_dir
        os.makedirs(output_dir, exist_ok=True)
        files = os.listdir(output_dir)
        num_samples = args.num_samples - len(files)
        sample_indices = torch.rand(num_samples) * len(data)
        
        for i in tqdm(sample_indices):
            if args.diffusion_type == "LCM":
                scheduler.set_timesteps(timesteps, original_inference_steps=50)
            else:
                scheduler.set_timesteps(timesteps)
            i = int(i)
            k = 0
            filename = output_dir + f'{data[i]["id"]}_{k}.png'
            while os.path.exists(filename):
                filename = output_dir + f'{data[i]["id"]}_{k}.png'
                k += 1

           
            masks = []
            mask_indexes = data[i]['mask_indexes']
            if len(mask_indexes) > 0:
                masks_p = data[i]['mask_path']
                for mask_p in masks_p:
                    masks.append(cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE))

                context = compute_embeddings(tokenizer, text_encoder, device, 
                                    batch_size, [data[i]['caption']], 
                                    sd= False if args.diffusion_type =="LCM" else True)
                
                latents = torch.randn((batch_size, 4, 64, 64), dtype=MODEL_TYPE, device=device) 
                if args.method == "new":
                    guide_diffusion(scheduler, unet, vae, latents, context, device, guidance_scale, \
                        args.diffusion_type, timesteps, args.guide, masks, \
                        mask_indexes, args.res, filename, \
                        loss_type=args.loss_type, eta=args.eta, glue=args.glue)
                elif args.method == "pww":
                    #pww
                    guide_diffusion_pww(scheduler, unet, vae, latents, context, device, guidance_scale, \
                        args.diffusion_type, timesteps, args.guide, masks, \
                        mask_indexes, args.res, filename, \
                        glue=args.glue,pww=1)
                elif args.method == "new_attn":
                    att_diff(scheduler, unet, vae, latents, context, device, guidance_scale, \
                        args.diffusion_type, timesteps, filename, \
                         glue=args.glue, prompt=data[i]['caption'])
                    
                    #fill masks
                    attns_files = os.listdir("attns/")
                    attns_files = list(map(lambda f: os.path.join("attns/",f), attns_files))
                    attns_files = natsorted(attns_files)
                    local_masks = []
                    for f in attns_files:
                        local_masks.append(cv2.imread(f, cv2.IMREAD_GRAYSCALE))
                    for i, mask_i in enumerate(mask_indexes):
                        local_masks[mask_i] = masks[i]
                    masks = local_masks
                    mask_indexes = list(range(len(attns_files)))

                    unet = UNet2DConditionModel.from_pretrained(repo_id, subfolder="unet", torch_dtype=MODEL_TYPE).to(device)
                    for param in unet.parameters():
                        param.requires_grad = False
                    guide_diffusion(scheduler, unet, vae, latents, context, device, guidance_scale, \
                        args.diffusion_type, timesteps, False, masks, \
                        mask_indexes, args.res, filename, \
                        loss_type=args.loss_type, eta=args.eta, glue=args.glue)
                    
                elif args.method == "zero_shot":
                    zero_shot(scheduler, unet, vae, latents, context, [data[i]['caption']],device, guidance_scale, \
                        args.diffusion_type, timesteps, args.guide, masks, \
                        mask_indexes, args.res, filename, \
                        eta=args.eta)
                
            
