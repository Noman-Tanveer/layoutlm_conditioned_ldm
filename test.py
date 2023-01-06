import os
import io
from omegaconf import OmegaConf
import torch
from accelerate import Accelerator
from diffusers import StableDiffusionPipeline

from architecture import get_model_componenets

with io.open("config.yaml") as f:
    cfg = OmegaConf.load(f)
    cfg = cfg.train

accelerator = Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
    )

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", revision="fp16", torch_dtype=torch.float16, requires_safety_checker=False)

pipe = pipe.to("cuda")
prompt = "Wha...."
image = pipe(prompt).images[0]  # image here is in [PIL format](https://pillow.readthedocs.io/en/stable/)
image.show()
