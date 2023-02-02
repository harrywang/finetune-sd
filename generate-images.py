import argparse
import re
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

def parse_args():
    parser = argparse.ArgumentParser(description="Generate Images from Lora Weights")
    parser.add_argument(
        "--prompt",
        type=str,
        default="a photo of sks dog in a bucket",
        help="prompt to generate the image",
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default="./models/dreambooth-lora/dog",
        help=("the path to the trained model file"),
    )

    parser.add_argument(
        "--output_folder",
        type=str,
        default="./outputs",
        help=("the path to folder to hold generated images"),
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=25,
        help=("inference steps"),
    )

    um_inference_steps=25

    args = parser.parse_args()
    return args


def main():
    args = parse_args()  # get arguments
    file_name = re.sub(r'\W+', '-', args.prompt)  # change all non-alphanumeric characters to dash

    if torch.cuda.is_available():
        device = "cuda"
        pipe = StableDiffusionPipeline.from_pretrained('runwayml/stable-diffusion-v1-5', torch_dtype=torch.float16)
    else:
        device = "cpu"
        pipe = StableDiffusionPipeline.from_pretrained('runwayml/stable-diffusion-v1-5', torch_dtype=torch.float32)
    print(f'device is {device}')

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.unet.load_attn_procs(args.model_path)
    pipe.to(device)
    
    image = pipe(args.prompt, num_inference_steps=args.steps).images[0]
    image.save(args.output_folder + "/" + file_name + ".png")

if __name__ == "__main__":
    main()