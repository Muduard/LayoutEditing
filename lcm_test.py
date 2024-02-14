from diffusers import DiffusionPipeline
import torch
from tqdm import tqdm



pipe = DiffusionPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7")

# To save GPU memory, torch.float16 can be used, but it may compromise image quality.
pipe.to(torch_device="cuda", torch_dtype=torch.float16)

prompt = "A man catching a frisbee in a park "
timesteps = 8
pipe.scheduler.set_timesteps(timesteps, original_inference_steps=50)
for t in tqdm(pipe.scheduler.timesteps): 
    
# Can be set to 1~50 steps. LCM support fast inference even <= 4 steps. Recommend: 1~8 steps.


images = pipe(prompt=prompt, num_inference_steps=num_inference_steps, guidance_scale=8, lcm_origin_steps=50, output_type="pil").images[0]
images.save("t.png")