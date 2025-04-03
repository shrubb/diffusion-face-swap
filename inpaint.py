# Adapted from:
# https://huggingface.co/docs/diffusers/main/en/using-diffusers/inpaint

import torch
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image, make_image_grid
from pathlib import Path

lora_path = "./experiments/sdxl-woman-lora-lr_5e6/checkpoint-1800/"

pipeline = AutoPipelineForInpainting.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    safety_checker=None,
)
pipeline.load_lora_weights(lora_path)

# pipeline.enable_model_cpu_offload()
pipeline = pipeline.to("cuda")
#pipeline.enable_xformers_memory_efficient_attention()

# load base and mask image
image_path = Path("./datasets/glam_ai_faces/target/target_C.png")
init_image = load_image(image_path)
mask_image = load_image(image_path.parent / "masks" / image_path.name)

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

# Inpaint
image = pipeline(
    prompt="a photo of a woman face",
    image=init_image,
    mask_image=mask_image,
    generator=torch.Generator("cuda").manual_seed(92),
    padding_mask_crop=int(mask_width * 0.9),
    strength=0.5,
).images[0]
make_image_grid([init_image, image], rows=1, cols=2).save(lora_path / image_path.name)
