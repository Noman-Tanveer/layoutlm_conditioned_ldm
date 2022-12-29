import logging
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.utils.import_utils import is_xformers_available
from transformers import LayoutLMv3Model
from pathlib import Path
from typing import Iterable, Optional


logger = logging.getLogger(__name__)

def get_text_encoder(args):
    # TODO: Make it generic for any encoder
    text_encoder = LayoutLMv3Model.from_pretrained(
        args.encoder,
    )
    return text_encoder

# Load models and create wrapper for stable diffusion

def get_vae(args):
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
    )
    return vae

def get_unet(args):
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.revision,
    )
    return unet

def get_model_componenets(args):
    text_encoder = get_text_encoder(args)
    vae = get_vae(args)
    unet = get_unet(args)

    if is_xformers_available():
        try:
            unet.enable_xformers_memory_efficient_attention()
        except Exception as e:
            logger.warning(
                "Could not enable memory efficient attention. Make sure xformers is installed"
                f" correctly and a GPU is available: {e}"
            )

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    return text_encoder, vae, unet, noise_scheduler