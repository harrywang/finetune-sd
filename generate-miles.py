model_path = 'outputs/miles'

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

pipe = StableDiffusionPipeline.from_pretrained('runwayml/stable-diffusion-v1-5', torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

#pipe.unet.load_attn_procs(model_path)
pipe.to("cuda")

image = pipe("a milesdami cat swimming", num_inference_steps=25).images[0]
image.save("miles_swimming.png")

image = pipe("a milesdami cat in a bucket", num_inference_steps=25).images[0]
image.save("miles_bucket.png")

image = pipe("a milesdami cat in front of eiffel tower", num_inference_steps=25).images[0]
image.save("miles_moon.png")