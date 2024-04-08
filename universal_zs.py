from diffusers import StableDiffusionPipeline
import torch

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = "a photo of a cat and a man in a room"
for i in range(10):
    image = pipe(prompt).images[0]  
        
    image.save(f"{i}.png")