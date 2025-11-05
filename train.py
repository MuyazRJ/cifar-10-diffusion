import torch 

from diffusion.schedules import compute_alphas, make_beta_schedule
from config import NUM_DIFFUSION_STEPS, BETA_START, BETA_END

from data.load import get_cifar10_dataloader


betas = make_beta_schedule("cosine", num_steps=NUM_DIFFUSION_STEPS, start=BETA_START, end=BETA_END)
alphas, alpha_bars = compute_alphas(betas)

train_loader = get_cifar10_dataloader() 