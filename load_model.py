import torch
from model.model import ImprovedDDPM
from config import DEVICE, NUM_DIFFUSION_STEPS, BETA_START, BETA_END, NUM_DIFFUSION_STEPS

from diffusion.reverse import reverse 
from diffusion.schedules import compute_alphas, make_beta_schedule

from utils.plot_and_save import save_image_grid

path = "./model_params/final_ddpm.pt"

# 1️⃣ Create a new instance of your model and optimizer
model = ImprovedDDPM().to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, betas=(0.9, 0.999), weight_decay=1e-4)

# 2️⃣ Load the saved checkpoint
checkpoint = torch.load(path, map_location=DEVICE)

# 3️⃣ Restore weights and optimizer state
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

print("✅ Model and optimizer successfully loaded from './model_params/final_ddpm.pt'")

betas = make_beta_schedule("cosine", num_steps=NUM_DIFFUSION_STEPS, start=BETA_START, end=BETA_END).to(DEVICE).float()
alphas, alpha_bars = compute_alphas(betas)

image = reverse(model, NUM_DIFFUSION_STEPS,  (1, 3, 32, 32), alphas, alpha_bars, DEVICE)

save_image_grid(image, show=True)