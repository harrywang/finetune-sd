model_path = './outputs/dreambooth/dog'

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

pipe = StableDiffusionPipeline.from_pretrained('runwayml/stable-diffusion-v1-5', torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

pipe.unet.load_attn_procs(model_path)
pipe.to("cuda")

image = pipe("A photo of sks dog swimming", num_inference_steps=25).images[0]
image.save("dog_swimming.png")

image = pipe("A photo of sks dog in a bucket", num_inference_steps=25).images[0]
image.save("dog_bucket.png")

image = pipe("A photo of sks dog in front of eiffel tower", num_inference_steps=25).images[0]
image.save("dog_tower.png")

image = pipe("A photo of sks dog on the great wall", num_inference_steps=25).images[0]
image.save("dog_wall.png")
