# About

Code for my tutorial [Fine-tune Stable Diffusion](https://harrywang.me/finetune-sd).

I copied the training scripts from the following repos and will periodically update them to the latest:

- [train_text_to_image_lora.py](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora.py)
- [train_dreambooth_lora.py](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth_lora.py)
- [train_dreambooth.py](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth.py)

# Setup

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

- Python version: use 3.9.11 (3.8.x may run into this error)
- Pytorch version: be default the latest 1.13.1 is installed but my 2080ti machine requires 1.11.0 (run `pip install torch==1.11.0`)

2080ti

- CUDA out of memory dreambooth with lora
- lora finetune OK

Tesla V100 32G


Then, install [xformers](https://huggingface.co/docs/diffusers/optimization/xformers) and add `--enable_xformers_memory_efficient_attention`

```
pip install pyre-extensions==0.0.23
pip install -i https://test.pypi.org/simple/ formers==0.0.15.dev376
```

login to HuggingFace using your token and login to WandB using your API key. If you won't want to use WandB, remove `--report_to=wandb` from all commands below.

```
huggingface-cli login
wandb login
```

## Full SD Fine-tuning with LoRA

see [docs](https://huggingface.co/blog/lora)

- Pokemon dataset

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

- Custom dataset, i.e., toy example using 15 photos of my cat Miles (took ~26 minutes on Tesla V100):

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
  --max_train_steps=1000 \
  --learning_rate=1e-04 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine" --lr_warmup_steps=0 \
  --output_dir=${OUTPUT_DIR} \
  --checkpointing_steps=500 \
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
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --checkpointing_steps=100 \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --validation_prompt="A photo of sks cat standing on the great wall" \
  --validation_epochs=20 \
  --seed=42 \
  --report_to="wandb"
```

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

TODO: Dog example with prior-preservation loss

generate images using Dreambooth models:

```
python generate-dreambooth.py --prompt "a dog standing on the great wall" --model_path "./models/dreambooth/dog" --output_folder "./outputs" --steps 50
python generate-dreambooth.py --prompt "a sks dog standing on the great wall" --model_path "./models/dreambooth/dog" --output_folder "./outputs/dreambooth"
python generate-dreambooth.py --prompt "a sks dog swimming"
```
