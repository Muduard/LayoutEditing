import os 
import torch
from diffusers import DDIMScheduler,AutoencoderKL,UNet2DConditionModel, LCMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from ptp_utils import AttentionStore,get_attn_layers,diffusion_step,get_multi_level_attention_from_average,register_attention_control, get_cross_attention, show_cross_attention,latent2image,save_tensor_as_image, lcm_diffusion_step, get_guidance_scale_embedding, compute_embeddings
import cv2
import argparse
from diffusers.models.attention_processor import AttnProcessor2_0
import json
from guided_diffusion import guide_diffusion
import pickle
from zero_shot import zero_shot
from tqdm import tqdm
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
parser.add_argument('--mask_index', nargs='+', help='List of token indices to move with a mask')
parser.add_argument("--mask_path", type=str, help="Path of masks as image files with the name of the corresponding token")
parser.add_argument("--resampling_steps", type=int, default=0, help="Resample noise for better coherence")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--diffusion_type", type=str, default="LCM")
parser.add_argument("--from_file", type=str, default=None)
parser.add_argument("--loss_type", type=str, default="mse")
parser.add_argument("--eta", type=float, default=0.2)
parser.add_argument("--method", type=str, default="new")
parser.add_argument("--out_dir", type=str, default="test/")
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



timesteps = 50
batch_size = 1
guidance_scale = 3
torch.manual_seed(args.seed)

#torch.compile(unet, mode="reduce-overhead", fullgraph=True)
if args.from_file == None:
    mask_index = int(args.mask_index[0])
    
    h = None
    new_attn = None
    base_masks = []
    if args.guide:
        masks = os.listdir("masks/")
        for f in masks:
            base_masks.append(torch.tensor(cv2.imread("masks/" + f, cv2.IMREAD_GRAYSCALE)) / 255)

    if not sl:
        latents = torch.randn((batch_size, 4, 64, 64), dtype=MODEL_TYPE, device=device) 
    else:
        latents = torch.load("starting_latent.pt",map_location=device).to(dtype=MODEL_TYPE)
    prompts = [args.prompt]

    vae = AutoencoderKL.from_pretrained(repo_id, subfolder="vae", torch_dtype=MODEL_TYPE).to(device)
    tokenizer = CLIPTokenizer.from_pretrained(repo_id, subfolder="tokenizer", torch_dtype=MODEL_TYPE)
    text_encoder = CLIPTextModel.from_pretrained(repo_id, subfolder="text_encoder", torch_dtype=MODEL_TYPE).to(device)
    unet = UNet2DConditionModel.from_pretrained(repo_id, subfolder="unet", torch_dtype=MODEL_TYPE).to(device)
    if args.diffusion_type == "LCM":
        scheduler = LCMScheduler.from_pretrained(repo_id,subfolder="scheduler", torch_dtype=torch.float16)
    else:
        scheduler = DDIMScheduler.from_pretrained(repo_id,subfolder="scheduler", torch_dtype=torch.float16)

    if args.diffusion_type == "LCM":
        scheduler.set_timesteps(timesteps, original_inference_steps=50)
    else:
        scheduler.set_timesteps(timesteps)
    for param in unet.parameters():
        param.requires_grad = False
    
    context = compute_embeddings(tokenizer, text_encoder, device, 
                                batch_size, prompts, sd= False if args.diffusion_type =="LCM" else True)

    mask = torch.ones_like(latents)

    guide_diffusion(scheduler, unet, vae, latents, context, device, guidance_scale, \
                args.diffusion_type, timesteps, args.guide, h, \
                mask_index, args.res, "./test/lcm_kl.png")
else:
    #torch.cuda.memory._record_memory_history(enabled=True, device=device)
    with open(args.from_file, "r") as f:
        data = json.load(f)['image_data']
        #losses = ["cosine"]
        #etas = [0.01, 0.02,0.04,0.06,0.08]
        #for l in range(len(losses)):
        #    args.loss_type = losses[l]
        #    for e in etas:
        #        args.etas = e
        #        output_dir = f"./test/{args.loss_type}_{e}/"
        #        os.makedirs(output_dir,exist_ok=True)
        #        print(f"Generating with loss: {args.loss_type} and eta: {e}")
        output_dir = args.out_dir
        os.makedirs(output_dir, exist_ok=True)
        for i in tqdm(range(len(data))):
            
            vae = AutoencoderKL.from_pretrained(repo_id, subfolder="vae", torch_dtype=MODEL_TYPE).to(device)
            tokenizer = CLIPTokenizer.from_pretrained(repo_id, subfolder="tokenizer", torch_dtype=MODEL_TYPE)
            text_encoder = CLIPTextModel.from_pretrained(repo_id, subfolder="text_encoder", torch_dtype=MODEL_TYPE).to(device)
            unet = UNet2DConditionModel.from_pretrained(repo_id, subfolder="unet", torch_dtype=MODEL_TYPE).to(device)
            if args.diffusion_type == "LCM":
                scheduler = LCMScheduler.from_pretrained(repo_id,subfolder="scheduler", torch_dtype=torch.float16)
            else:
                scheduler = DDIMScheduler.from_pretrained(repo_id,subfolder="scheduler", torch_dtype=torch.float16)

            if args.diffusion_type == "LCM":
                scheduler.set_timesteps(timesteps, original_inference_steps=50)
            else:
                scheduler.set_timesteps(timesteps)
            for param in unet.parameters():
                param.requires_grad = False
            mask = cv2.imread(data[i]['mask_path'], cv2.IMREAD_GRAYSCALE)
            context = compute_embeddings(tokenizer, text_encoder, device, 
                                batch_size, [data[i]['caption']], 
                                sd= False if args.diffusion_type =="LCM" else True)
            
            latents = torch.randn((batch_size, 4, 64, 64), dtype=MODEL_TYPE, device=device) 
            if args.method == "new":
                guide_diffusion(scheduler, unet, vae, latents, context, device, guidance_scale, \
                    args.diffusion_type, timesteps, args.guide, mask, \
                    data[i]['mask_index'], args.res, output_dir + f'{data[i]["id"]}.png', \
                    loss_type=args.loss_type, eta=args.eta)
            else:
                zero_shot(scheduler, unet, vae, latents, context, [data[i]['caption']],device, guidance_scale, \
                    args.diffusion_type, timesteps, args.guide, mask, \
                    data[i]['mask_index'], args.res, output_dir + f'{data[i]["id"]}.png', \
                    eta=args.eta)
            torch.cuda.empty_cache()

            # Capture a snapshot of GPU memory allocations
            
