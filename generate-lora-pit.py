from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
import torch

model_id = "runwayml/stable-diffusion-v1-5"

if torch.cuda.is_available():
    device = "cuda"
    # if limited by GPU memory, chunking the attention computation in addition to using fp16
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
else:
    device = "cpu"
    # if on CPU or want to have maximum precision on GPU, use default full-precision setting
    pipe = StableDiffusionPipeline.from_pretrained(model_id)
print(f'device is {device}')


pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

prompt = "superman, style of <s1><s2>"
torch.manual_seed(0)
image = pipe(prompt, num_inference_steps=20, guidance_scale=7).images[0]
image.save("1.jpg")


from lora_diffusion import tune_lora_scale, patch_pipe

patch_pipe(
    pipe,
    "./models/dreambooth/3d-avatar/final_lora.safetensors",
    patch_text=True,
    patch_ti=True,
    patch_unet=True,
)

tune_lora_scale(pipe.unet, 1.00)
tune_lora_scale(pipe.text_encoder, 1.00)

torch.manual_seed(0)
image = pipe(prompt, num_inference_steps=50, guidance_scale=7).images[0]
image.save("2.jpg")