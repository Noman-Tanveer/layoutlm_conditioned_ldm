import os
import io
from omegaconf import OmegaConf
import torch
from accelerate import Accelerator


checkpoint = "stable_diffusion-v1-5/checkpoint-3500"

with io.open("config.yaml") as f:
    cfg = OmegaConf.load(f)

accelerator = Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
    )
