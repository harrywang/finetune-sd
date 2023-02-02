# About

Code for my tutorial [Fine-tune Stable Diffusion using LoRA and Your Own Images](https://harrywang.me/clip)

# Setup

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Then, install [xformers](https://huggingface.co/docs/diffusers/optimization/xformers)

```
pip install pyre-extensions==0.0.23
pip install -i https://test.pypi.org/simple/ formers==0.0.15.dev376
```

login to HuggingFace using your token and login to WandB using your API key. If you won't want to use WandB, remove `--report_to=wandb` from all commands below.

```
huggingface-cli login
wandb login
```

Fine-tune using LoRA and Pokemon dataset:

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
  --report_to=wandb \
  --checkpointing_steps=500 \
  --validation_prompt="Totoro" \
  --seed=1337
```

Fine-tune using LoRA and your own dataset (photos of my cat Miles as an example):

```
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATA_DIR="./data/cat-finetune"
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
  --report_to=wandb \
  --checkpointing_steps=500 \
  --seed=42
```

Fine-tune using Dreambooth with LoRA and your own dataset.

Dog example (data from the paper):

```
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="./data/dog"
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
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --validation_prompt="A photo of sks dog in a bucket" \
  --validation_epochs=20 \
  --seed=42
```

Sunglass example (I collected a few sunglasses images):

```
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="./data/glasses"
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
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --validation_prompt="A photo of sks sunglasses with eiffel tower" \
  --validation_epochs=20 \
  --seed=42
```

My cat Miles example:

```
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="./data/cat-original"
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
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --validation_prompt="A photo of sks cat standing on the great wall" \
  --validation_epochs=20 \
  --seed=42
```

generate images:

```
python generate-images.py --prompt "a dog standing on the great wall" --model_path "./models/dreambooth-lora/dog" --output_folder "./outputs" --steps 50
```
