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
        default="./models/dreambooth/dog",
        help=("the path to the trained model file"),
    )

    parser.add_argument(
        "--output_folder",
        type=str,
        default="./outputs/dreambooth",
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
        # if limited by GPU memory, chunking the attention computation in addition to using fp16
        pipe = StableDiffusionPipeline.from_pretrained(args.model_path, torch_dtype=torch.float16).to("cuda")
    else:
        device = "cpu"
        # if on CPU or want to have maximum precision on GPU, use default full-precision setting
        pipe = StableDiffusionPipeline.from_pretrained(args.model_path)
    print(f'device is {device}')

    image = pipe(args.prompt, num_inference_steps=args.steps, guidance_scale=7.5).images[0]
    image.save(args.output_folder + "/" + file_name + ".png")

if __name__ == "__main__":
    main()