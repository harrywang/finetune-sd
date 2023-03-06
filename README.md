# About

Code for my tutorial [What I Learned About Fine-tuning Stable Diffusion](https://harrywang.me/sd).

I copied the training scripts from the following repos and will periodically update them to the latest:

- [train_text_to_image_lora.py](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora.py)
- [train_dreambooth_lora.py](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth_lora.py)
- [train_dreambooth.py](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth.py)

# Setup

- Python version: tested with 3.9.11 and 3.10.9 (3.8.x may run into this error)
- Pytorch version: tested with latest 1.13.1+cu117 (used 1.11.0 for my old 2080ti by running `pip install torch==1.11.0`)

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
accelerate config default
```

Optional: install [xformers](https://huggingface.co/docs/diffusers/optimization/xformers) and add `--enable_xformers_memory_efficient_attention`

```
pip install xformers
```

- login to HuggingFace using your token: `huggingface-cli login`
- login to WandB using your API key: `wandb login`. If you won't want to use WandB, remove `--report_to=wandb` from all commands below.
- you may need to do `export WANDB_DISABLE_SERVICE=true` to solve this [issue](https://github.com/wandb/wandb/issues/4872)
- If you have multiple GPU, you can set the following environment variable to choose which GPU to use (default is `CUDA_VISIBLE_DEVICES=0`): `export CUDA_VISIBLE_DEVICES=1`
- FileNotFoundError: [Errno 2] No such file or directory: 'git-lfs': `sudo apt install git-lfs`


## Full SD Fine-tuning with LoRA

see [docs](https://huggingface.co/blog/lora)

- Pokemon dataset (took ~7.5 hours on 2080ti and ~4.5 hours on Tesla V100)

```
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATASET_NAME="lambdalabs/pokemon-blip-captions"
export OUTPUT_DIR="./models/lora/pokemon"

accelerate launch --mixed_precision="fp16"  train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --dataloader_num_workers=8 \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=15000 \
  --learning_rate=1e-04 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine" --lr_warmup_steps=0 \
  --output_dir=${OUTPUT_DIR} \
  --checkpointing_steps=500 \
  --validation_prompt="Totoro" \
  --seed=42 \
  --report_to=wandb
```

- Custom dataset, i.e., toy example using 15 photos of my cat Miles (took ~40 minutes on Tesla V100):

```
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATA_DIR="./data/full-finetune/cat"
export OUTPUT_DIR="./models/lora/miles"

accelerate launch --mixed_precision="fp16"  train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --dataloader_num_workers=8 \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=1500 \
  --learning_rate=1e-04 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine" --lr_warmup_steps=0 \
  --output_dir=${OUTPUT_DIR} \
  --checkpointing_steps=500 \
  --validation_prompt="A photo of a cat in a bucket" \
  --validation_epochs=10 \
  --seed=42 \
  --report_to=wandb
```

## Dreambooth with LoRA

Fine-tune using Dreambooth with LoRA and your own dataset （4 min 39 sec. V100).

Dog example (data from the paper):

```
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="./data/dreambooth/dog"
export OUTPUT_DIR="./models/dreambooth-lora/dog"

accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of sks dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --checkpointing_steps=100 \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --validation_prompt="A photo of sks dog in a bucket" \
  --validation_epochs=20 \
  --seed=42 \
  --report_to="wandb"
```

TODO: Dog example with xformer:

```
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="./data/dreambooth/dog"
export OUTPUT_DIR="./models/dreambooth-lora/dog"

accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of sks dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --checkpointing_steps=100 \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --validation_prompt="A photo of sks dog in a bucket" \
  --validation_epochs=20 \
  --enable_xformers_memory_efficient_attention \
  --seed=42 \
  --report_to="wandb"
```

Sunglasses example:

```
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="./data/dreambooth/glasses"
export OUTPUT_DIR="./models/dreambooth-lora/sunglasses"

accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of sks sunglasses" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --checkpointing_steps=100 \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --validation_prompt="A photo of sks sunglasses with eiffel tower" \
  --validation_epochs=20 \
  --seed=42 \
  --report_to="wandb"
```

My cat Miles example:

```
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="./data/dreambooth/cat"
export OUTPUT_DIR="./models/dreambooth-lora/miles"

accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of sks cat" \
  --resolution=512 --center_crop \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --checkpointing_steps=100 \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=1500 \
  --validation_prompt="A photo of a sks cat in a bucket" \
  --validation_epochs=10 \
  --seed=42 \
  --report_to="wandb"
```

With class prompt (class images generated by the model) and prior preservation (with weight 0.5):

```
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="./data/dreambooth/cat"
export CLASS_DIR="./data/dreambooth/cat-class"
export OUTPUT_DIR="./models/dreambooth-lora/miles"

accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of sks cat" \
  --class_prompt="a photo of a cat" \
  --with_prior_preservation --prior_loss_weight=0.5 \
  --resolution=512 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=1 \
  --checkpointing_steps=100 \
  --learning_rate=1e-4 \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=1500 \
  --validation_prompt="A photo of sks cat in a bucket" \
  --num_class_images=200 \
  --validation_epochs=10 \
  --seed=42
```

Miss Dong example:

```
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="./data/dreambooth/missdong"
export OUTPUT_DIR="./models/dreambooth-lora/missdong"

accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of sks lady" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --checkpointing_steps=100 \
  --learning_rate=1e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=1200 \
  --validation_prompt="oil painting of sks lady by the ocean" \
  --validation_epochs=20 \
  --seed=42 \
  --report_to="wandb"
```

Another example:

```
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="./data/dreambooth/david-beckham"
export OUTPUT_DIR="./models/dreambooth-lora/david-beckham"

accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of dbsks man" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --checkpointing_steps=100 \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=700 \
  --validation_prompt="A photo of dbsks man, detailed faces, highres, RAW photo 8k uhd, dslr" \
  --validation_epochs=10 \
  --seed=42 \
  --report_to="wandb"
```

--with_prior_preservation --prior_loss_weight=1.0 \

generate images using LoRA weights:

```
python generate-lora.py --prompt "a dog standing on the great wall" --model_path "./models/dreambooth-lora/dog" --output_folder "./outputs" --steps 50
python generate-lora.py --prompt "a sks dog standing on the great wall" --model_path "./models/dreambooth-lora/dog" --output_folder "./outputs"
```

## Dreambooth

See [blog](https://huggingface.co/blog/dreambooth) and [docs](https://github.com/huggingface/diffusers/tree/main/examples/dreambooth)

Dog example without prior-preservation loss (~7 mins on V100):

```
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="./data/dreambooth/dog"
export OUTPUT_DIR="./models/dreambooth/dog"

accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of sks dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=400 \
  --report_to="wandb"
```

Miss Dong example:

```
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="./data/dreambooth/missdong"
export OUTPUT_DIR="./models/dreambooth/missdong"

accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of sks lady" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=400 \
  --report_to="wandb"
```

generate images using Dreambooth models:

```
python generate-dreambooth.py --prompt "a dog standing on the great wall" --model_path "./models/dreambooth/dog" --output_folder "./outputs" --steps 50
python generate-dreambooth.py --prompt "a sks dog standing on the great wall" --model_path "./models/dreambooth/dog" --output_folder "./outputs/dreambooth"
python generate-dreambooth.py --prompt "a sks dog swimming"
```

## Fine-tuning Stable diffusion with LoRA PTI

Use this [repo](https://github.com/cloneofsimo/lora): `pip install git+https://github.com/cloneofsimo/lora.git`

Use [LoRA PTI](https://github.com/cloneofsimo/lora/discussions/121)

Try a 3d avatar style

```
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="./data/dreambooth/3d-avatar"
export OUTPUT_DIR="./models/dreambooth/3d-avatar"

lora_pti \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --train_text_encoder \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --scale_lr \
  --learning_rate_unet=1e-4 \
  --learning_rate_text=1e-5 \
  --learning_rate_ti=5e-4 \
  --color_jitter \
  --lr_scheduler="linear" \
  --lr_warmup_steps=0 \
  --placeholder_tokens="<s1>|<s2>" \
  --use_template="style"\
  --save_steps=100 \
  --max_train_steps_ti=1000 \
  --max_train_steps_tuning=1000 \
  --perform_inversion=True \
  --clip_ti_decay \
  --weight_decay_ti=0.000 \
  --weight_decay_lora=0.001\
  --continue_inversion \
  --continue_inversion_lr=1e-4 \
  --device="cuda:0" \
  --lora_rank=1 \
  --use_template="style" \
#  --use_face_segmentation_condition\
```

To use the trained LoRA weights in WebUI, you need to merge it with a base model:

```
lora_add runwayml/stable-diffusion-v1-5 ./models/dreambooth/3d-avatar/final_lora.safetensors ./output_merged.ckpt 0.7 --mode upl-ckpt-v2
```

## Convert Diffusers LoRA Weights for Automatic1111 WebUI

The LoRA weights trained using Diffusers are saved in `.bin` or `.pkl` format, which must be converted to be used in Automatic1111 WebUI (see [here](https://github.com/huggingface/diffusers/issues/2326) for detailed discussions).

As seen below, the trained LoRA weights are stored in `custom_checkpoint_0.pkl` or `pytorch_model.bin`:

<img class="mx-auto" src="https://user-images.githubusercontent.com/595772/221718501-dc79a799-5fe5-4b9f-9b44-c19ac4103c06.png">

<img class="mx-auto" src="https://user-images.githubusercontent.com/595772/221718531-10fe4999-0ee0-4e6f-abf4-d9fc069ec540.png">

`convert-to-safetensors.py` can be used to convert `.bin` or `.pkl` files into `.safetensors` format, which can be used in WebUI (just put the converted the file in WebUI `models/Lora`). The script is adapted from the one written by [ignacfetser](https://github.com/ignacfetser). 

Put this script in the same folder of `.bin` or `.pkl` file and run `python convert-to-safetensors.py --file checkpoint_file`


