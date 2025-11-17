import torch

from config import DEVICE, NUM_DIFFUSION_STEPS, BETA_START, BETA_END, NUM_DIFFUSION_STEPS, LEARNING_RATE, EMA_DECAY, IMAGE_CLASSES  

from utils.ema import EMA
from model.model import ImprovedDDPM

from diffusion.reverse import reverse 
from diffusion.schedules import compute_alphas, make_beta_schedule

from utils.plot_and_save import save_image_grid
from data.cifar_classes import plot_images_by_class

path = "./model_checkpoints/16-11-2025_20-50/epoch_99.pt"

# 1️⃣ Create a new instance of your model and optimizer
model = ImprovedDDPM().to(DEVICE)
ema_model = EMA(model, decay=EMA_DECAY)    

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), weight_decay=1e-4)

# 2️⃣ Load the saved checkpoint
checkpoint = torch.load(path, map_location=DEVICE)

# 3️⃣ Restore weights and optimizer state
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
ema_model.ema_model.load_state_dict(checkpoint["ema_state_dict"])

print(f"✅ Model and optimizer successfully loaded from {path}")

betas = make_beta_schedule("cosine", num_steps=NUM_DIFFUSION_STEPS, start=BETA_START, end=BETA_END).to(DEVICE).float()
alphas, alpha_bars = compute_alphas(betas)

images_per_class = 10
labels = torch.arange(IMAGE_CLASSES).repeat_interleave(images_per_class).to(DEVICE)

images = reverse(ema_model.ema_model, alphas, alpha_bars, betas, NUM_DIFFUSION_STEPS, num_images=len(labels), labels=labels)

save_image_grid(images, out_dir="./outputs", nrow=images_per_class, show=True)