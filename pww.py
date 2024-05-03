import os 
import torch
from diffusers import DDIMScheduler,AutoencoderKL,UNet2DConditionModel, LCMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from ptp_utils import compute_embeddings,save_tensor_as_image,get_cross_attention,AttentionStore,register_attention_control,diffusion_step,latent2image
import cv2
import json
from tqdm import tqdm
from PIL import Image
MODEL_TYPE = torch.float16
device = "cuda:0"
batch_size = 1

repo_id = "runwayml/stable-diffusion-v1-5"
vae = AutoencoderKL.from_pretrained(repo_id, subfolder="vae", torch_dtype=MODEL_TYPE).to(device)
tokenizer = CLIPTokenizer.from_pretrained(repo_id, subfolder="tokenizer", torch_dtype=MODEL_TYPE)
text_encoder = CLIPTextModel.from_pretrained(repo_id, subfolder="text_encoder", torch_dtype=MODEL_TYPE).to(device)
unet = UNet2DConditionModel.from_pretrained(repo_id, subfolder="unet", torch_dtype=MODEL_TYPE).to(device)
for param in unet.parameters():
    param.requires_grad = False


timesteps = 50
scheduler = DDIMScheduler.from_pretrained(repo_id,subfolder="scheduler", torch_dtype=torch.float16)
guidance_scale = 3.5

torch.manual_seed(0)
with open("eval.json", "r") as f:
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
        output_dir = "pww/"
        os.makedirs(output_dir, exist_ok=True)
        files = os.listdir(output_dir)
        num_samples = 5000 - len(files)
        sample_indices = torch.rand(num_samples) * len(data)
        
        for i in tqdm(sample_indices):
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
                    masks.append(torch.tensor(cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE))/255)
            #controller = AttentionStore()
            #register_attention_control(unet, controller, masks, mask_indexes)

            latents = torch.randn((batch_size, 4, 64, 64), dtype=MODEL_TYPE, device=device) 
            scheduler.set_timesteps(timesteps)
            context = compute_embeddings(tokenizer, text_encoder, device, 
                                    batch_size, [data[i]['caption']], 
                                    sd = True)
            
            with torch.no_grad():
                for t in tqdm(scheduler.timesteps):
                        
                        latents, _ = diffusion_step(unet, scheduler, None, latents, context, t, guidance_scale)
                        
                image = latent2image(vae, latents.detach())
                image = Image.fromarray(image)
                out_path = output_dir + f'{data[i]["id"]}_{k}.png'
                image.save(out_path)
                