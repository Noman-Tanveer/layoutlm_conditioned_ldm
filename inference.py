
import io
import os
import types
from PIL import Image
from tqdm.auto import tqdm

from omegaconf import OmegaConf
import torch
from transformers import LayoutLMv3Model
from diffusers import DiffusionPipeline
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, LMSDiscreteScheduler

from dataloaders.funsd_data import FUNSD

with io.open("config.yaml") as f:
    cfg = OmegaConf.load(f)

torch_device = cfg.device if cfg.device else "cuda" if torch.cuda.is_available() else "cpu"
rand_generator = torch.manual_seed(32)   # Seed generator to create the inital latent noise


test_image = "00070353.png"


def get_components(checkpoint=cfg.pipeline, encoder_checkpoint="microsoft/layoutlmv3-base"):
    # 1. Load the autoencoder model which will be used to decode the latents into image space. 
    vae = AutoencoderKL.from_pretrained(checkpoint, subfolder="vae")
    # 2. Load the tokenizer and text encoder to tokenize and encode the text. 
    text_encoder = LayoutLMv3Model.from_pretrained(encoder_checkpoint)
    # 3. The UNet model for generating the latents.
    state_dict = torch.load(os.path.join(cfg.pipeline, cfg.checkpoint))
    unet = UNet2DConditionModel()
    unet = unet.load_state_dict(state_dict=state_dict)
    scheduler = LMSDiscreteScheduler.from_pretrained(checkpoint, subfolder="scheduler")
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    vae = vae.to(torch_device)
    text_encoder = text_encoder.to(torch_device)
    unet = unet.to(torch_device)

    return vae, unet, text_encoder, scheduler

vae, unet, text_encoder, scheduler = get_components(cfg.pipeline, cfg.encoder_checkpoint)

dataset = FUNSD(cfg)
encoder_input = dataset(test_image)
assert list(encoder_input.keys()) == ['input_ids', 'bbox', 'attention_mask', 'pixel_values']

for k, v in encoder_input.items():
    print(k, v.shape)

encoder_input = encoder_input.to(torch_device)
with torch.no_grad():
  page_embeddings = text_encoder(**encoder_input)[0]

ldm_in_embeddings = torch.cat([page_embeddings, page_embeddings])
print(ldm_in_embeddings.shape)

latents = torch.randn(
  (cfg.batch_size, unet.in_channels, cfg.resolution // cfg.scaling_factor, cfg.resolution // cfg.scaling_factor),
  generator=rand_generator,
)
latents = latents.to(torch_device)

print("Latents shape: ", latents.shape)

scheduler.set_timesteps(cfg.inference_steps)
latents = latents * scheduler.init_noise_sigma

for t in tqdm(scheduler.timesteps):
  # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
  latent_model_input = torch.cat([latents] * 2)
  latent_model_input = scheduler.scale_model_input(latent_model_input, t)

  # predict the noise residual
  with torch.no_grad():
    noise_pred = unet(latent_model_input, t, encoder_hidden_states=ldm_in_embeddings).sample

  # perform guidance
  noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
  noise_pred = noise_pred_uncond + cfg.guidance_scale * (noise_pred_text - noise_pred_uncond)

  # compute the previous noisy sample x_t -> x_t-1
  latents = scheduler.step(noise_pred, t, latents).prev_sample

# scale and decode the image latents with vae
latents = 1 / 0.18215 * latents

with torch.no_grad():
  image = vae.decode(latents).sample

image = (image / 2 + 0.5).clamp(0, 1)
image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
images = (image * 255).round().astype("uint8")
pil_images = [Image.fromarray(image) for image in images]
pil_images[0].save("inference.png")
