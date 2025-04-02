# Adapted from:
# https://huggingface.co/docs/diffusers/main/en/using-diffusers/inpaint

import torch
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image, make_image_grid

pipeline = AutoPipelineForInpainting.from_pretrained(
    "sd-legacy/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    safety_checker=None,
)
pipeline.load_lora_weights("./experiments/woman-lora-cropped-photoPrompt-lr_4e6/checkpoint-3900/")

pipeline.enable_model_cpu_offload()
# remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
#pipeline.enable_xformers_memory_efficient_attention()

# load base and mask image
init_image = load_image("./datasets/glam_ai_faces/target/target_C.png")
mask_image = load_image("./datasets/glam_ai_faces/target/masks/target_C.png")

# Compute the "padding_mask_crop" parameter
import numpy as np

mask_array = np.array(mask_image).mean(-1)
mask_x = np.where(mask_array.max(0) > 20)[0]
mask_y = np.where(mask_array.max(1) > 20)[0]

mask_l = mask_x.min()
mask_r = mask_x.max()
mask_t = mask_y.min()
mask_b = mask_y.max()

mask_width = (mask_r - mask_l).item()
mask_height = (mask_b - mask_t).item()

for strength in range(1, 20+1):
    strength /= 20

    image = pipeline(
        prompt="a photo of a woman face",
        image=init_image,
        mask_image=mask_image,
        generator=torch.Generator("cuda").manual_seed(92),
        padding_mask_crop=int(mask_width * 0.9),
        strength=strength,
    ).images[0]
    make_image_grid([init_image, image], rows=1, cols=2).save(f"./experiments/woman-lora-cropped-photoPrompt-lr_4e6/checkpoint-3900/target_C_strength-{strength:.2f}.jpg")