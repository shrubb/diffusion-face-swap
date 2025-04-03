#!/bin/bash

#SBATCH --job-name inpaint-1
#SBATCH --output log-train2.txt
#SBATCH --time 0-3

#SBATCH -p gpu_devel
#SBATCH --gres gpu:1
#SBATCH --cpus-per-gpu 2

#source .env/bin/activate
accelerate launch train_text_to_image_lora_sdxl.py \
	--pretrained_model_name_or_path "stabilityai/stable-diffusion-xl-base-1.0" \
	--train_data_dir "./datasets/glam_ai_faces/train-rotated-cropped" \
	--validation_prompt "a photo of a woman face" \
	--num_train_epochs 6000 \
	--validation_epochs 150 \
	--checkpointing_steps 300 \
	--output_dir "experiments/sdxl-woman-lora-lr_5e6" \
	--logging_dir "./" \
	--seed 123 \
	--center_crop \
	--train_batch_size 3 \
	--lr_scheduler constant_with_warmup \
	--lr_warmup_steps 150 \
	--learning_rate 5e-6 \
	--dataloader_num_workers 0 \
	--mixed_precision fp16 \
	--resume_from_checkpoint latest \
#	--gradient_checkpointing \


# --allow_tf32 \
# --enable_xformers_memory_efficient_attention \
