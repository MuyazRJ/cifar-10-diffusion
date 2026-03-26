# Author: Mohammed Rahman
# Student ID: 10971320
# University of Manchester — BSc Computer Science Final Year Project, 2026
#
# Training script for the Improved DDPM on CIFAR-10.
# Implements the full training loop including forward diffusion, noise prediction,
# learned variance (KL loss), EMA weight averaging, and AMP mixed precision.
#
# Based on:
# - Ho et al., "Denoising Diffusion Probabilistic Models", NeurIPS 2020
#   https://arxiv.org/abs/2006.11239
# - Nichol & Dhariwal, "Improved Denoising Diffusion Probabilistic Models", ICML 2021
#   https://arxiv.org/abs/2102.09672

import torch
import os

import torch.nn.functional as F
from torch.amp import GradScaler, autocast

from config import NUM_DIFFUSION_STEPS, EPOCHS, DEVICE, SAVE_DIR, SAVE_DIR_TRAIN, RESUME_TRAINING, FILENAME, LEARNING_RATE, EMA_DECAY, GLOBAL_STEP_EMA, IMAGE_OUT_DIR, IMAGE_CLASSES, KL_WEIGHT
from data.load import get_cifar10_dataloader, get_mnist_dataloader

from diffusion.schedules import compute_alphas, make_beta_schedule
from diffusion.forward import q_sample
from diffusion.reverse import reverse

from utils.plot_and_save import save_image_grid
from utils.load_model import load_model_and_optimizer
from utils.ema import EMA

from datetime import datetime
from model.model import ImprovedDDPM

def main():
    # Create timestamped directories for checkpoints and output images
    os.makedirs(SAVE_DIR_TRAIN, exist_ok=True)
    dated_dir = os.path.join(SAVE_DIR_TRAIN, datetime.now().strftime("%d-%m-%Y_%H-%M"))
    output_dir = os.path.join(IMAGE_OUT_DIR, datetime.now().strftime("%d-%m-%Y_%H-%M"))
    os.makedirs(dated_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Build cosine beta schedule and derive alpha values
    betas = make_beta_schedule("cosine", num_steps=NUM_DIFFUSION_STEPS).to(DEVICE).float()
    alphas, alpha_bars = compute_alphas(betas)

    train_loader = get_cifar10_dataloader()

    model = ImprovedDDPM().to(DEVICE)
    criterion = torch.nn.MSELoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.999),
        weight_decay=1e-4
    )

    # EMA model for smoother inference
    ema_model = EMA(model, decay=EMA_DECAY)

    start_epoch = 0
    global_step = 0

    scaler = GradScaler("cuda")

    # Optionally resume from a saved checkpoint
    if RESUME_TRAINING:
        ckpt = torch.load(FILENAME, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        ema_model.ema_model.load_state_dict(ckpt["ema_state_dict"])
        start_epoch = ckpt["epoch"] + 1

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        last_bucket = -1

        for batch_idx, (images, label) in enumerate(train_loader):
            images = images.to(DEVICE)
            label = label.to(DEVICE)
            batch_size = images.size(0)

            # Sample a random timestep for each image in the batch
            t = torch.randint(0, NUM_DIFFUSION_STEPS, (batch_size,), device=DEVICE)

            # Apply forward diffusion to get noisy images at timestep t
            noisy_images, noise = q_sample(images, t, alpha_bars.to(DEVICE))

            optimizer.zero_grad()

            with autocast(device_type=DEVICE):
                # Model predicts both the noise and a raw variance signal
                eps_pred, var_raw = model(noisy_images, t, label)

                # Simple noise prediction loss (MSE)
                mse_loss = criterion(eps_pred, noise)

                # Map raw variance output from [-1,1] to [0,1] for interpolation
                frac = (var_raw + 1) / 2
                frac = frac.clamp(0, 1)

                beta_t = betas[t].view(-1, 1, 1, 1)
                alpha_bar_t = alpha_bars[t].view(-1, 1, 1, 1)
                alpha_bar_prev = alpha_bars[(t - 1).clamp(min=0)].view(-1, 1, 1, 1)

                # Compute posterior variance bounds (Improved DDPM Section 3)
                tilde_beta_t = beta_t * (1 - alpha_bar_prev) / (1 - alpha_bar_t)
                min_log = torch.log(tilde_beta_t)
                max_log = torch.log(beta_t)

                # Interpolate between min and max log variance using model output
                model_log_sigma2 = frac * max_log + (1 - frac) * min_log

                # KL term penalises deviation of predicted variance from posterior
                kl_loss = 0.5 * (model_log_sigma2 - min_log).mean()

                loss = mse_loss + KL_WEIGHT * kl_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            # Begin updating EMA weights after warmup period
            if global_step > GLOBAL_STEP_EMA:
                ema_model.update(model)

            # Print progress at 50% and 100% of each epoch
            progress = 100 * (batch_idx + 1) / len(train_loader)
            bucket = int(progress // 50)
            if bucket != last_bucket and bucket > 0:
                last_bucket = bucket
                now = datetime.now().strftime("%H:%M:%S")
                print(f"[{now}] Epoch {epoch} | {progress:.1f}% | Loss {loss.item():.4f}")

            global_step += 1

        # Every 5 epochs: sample a preview grid and save a checkpoint
        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            model.eval()
            with torch.no_grad():
                labels = torch.arange(IMAGE_CLASSES, device=DEVICE)
                image = reverse(ema_model.ema_model, alphas, alpha_bars, betas, NUM_DIFFUSION_STEPS, num_images=10, labels=labels)
                save_image_grid(image, out_dir=f"{output_dir}/epoch_{epoch}.png", show=False)

            save_path = f"{dated_dir}/epoch_{epoch}.pt"
            torch.save({
                "epoch": epoch,
                "ema_state_dict": ema_model.state_dict(),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, save_path)

    # Final sample and checkpoint after all epochs
    image = reverse(ema_model.ema_model, alphas, alpha_bars, betas, NUM_DIFFUSION_STEPS, num_images=20)
    save_image_grid(image, show=True, out_dir=f"{output_dir}/final.png")

    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR, "final_ddpm.pt")
    torch.save({
        "epoch": EPOCHS,
        "ema_state_dict": ema_model.state_dict(),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, save_path)

    print(f"\nTraining complete. Model saved to {SAVE_DIR}")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()