#!/bin/bash

#SBATCH --job-name jupyter2
#SBATCH --output log-train2.txt
#SBATCH --time 0-3

#SBATCH -p gpu_devel
#SBATCH --gres gpu:1
#SBATCH --cpus-per-gpu 1

source .env/bin/activate
accelerate launch train_text_to_image_lora.py \
	--pretrained_model_name_or_path "sd-legacy/stable-diffusion-v1-5" \
	--train_data_dir "./datasets/glam_ai_faces/train-rotated-cropped" \
	--validation_prompt "a photo of a woman face" \
	--num_train_epochs 10000 \
	--validation_epochs 150 \
	--checkpointing_steps 150 \
	--output_dir "experiments/woman-lora-cropped-photoPrompt-lr_1.5e6" \
	--logging_dir "./" \
	--seed 123 \
	--center_crop \
	--train_batch_size 7 \
	--lr_scheduler constant_with_warmup \
	--lr_warmup_steps 40 \
	--learning_rate 1.5e-6 \
	--dataloader_num_workers 0 \
	--mixed_precision fp16 \
	--resume_from_checkpoint latest


#--lr_warmup_steps 500 \
# --allow_tf32 \
# --enable_xformers_memory_efficient_attention \
