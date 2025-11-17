import torch
import os

import torch.nn.functional as F
from torch.amp import GradScaler, autocast   # <--- updated import

from config import NUM_DIFFUSION_STEPS, EPOCHS, DEVICE, SAVE_DIR, SAVE_DIR_TRAIN, RESUME_TRAINING, FILENAME, LEARNING_RATE, EMA_DECAY, GLOBAL_STEP_EMA, IMAGE_OUT_DIR, IMAGE_CLASSES
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
    os.makedirs(SAVE_DIR_TRAIN, exist_ok=True)
    dated_dir = os.path.join(SAVE_DIR_TRAIN,datetime.now().strftime("%d-%m-%Y_%H-%M"))
    output_dir = os.path.join(IMAGE_OUT_DIR, datetime.now().strftime("%d-%m-%Y_%H-%M"))
    os.makedirs(dated_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

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

    ema_model = EMA(model, decay=EMA_DECAY)

    start_epoch = 0
    global_step = 0

    # ✅ New-style AMP scaler
    scaler = GradScaler("cuda")

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
            
            # 1️⃣ Sample random timesteps
            t = torch.randint(0, NUM_DIFFUSION_STEPS, (batch_size,), device=DEVICE)

            # 2️⃣ Forward diffusion
            noisy_images, noise = q_sample(images, t, alpha_bars.to(DEVICE))

            optimizer.zero_grad()

            # 3️⃣ Forward pass with AMP
            with autocast(device_type=DEVICE):
                out = model(noisy_images, t, label)
                loss = criterion(out, noise)

            # 4️⃣ Backward + step with AMP
            scaler.scale(loss).backward()

            # 1. unscale gradients
            scaler.unscale_(optimizer)

            # 2. now clip them safely
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # 3. optimizer step
            scaler.step(optimizer)
            scaler.update()

            if global_step > GLOBAL_STEP_EMA:
                ema_model.update(model)

            progress = 100 * (batch_idx + 1) / len(train_loader)
            bucket = int(progress // 50)

            if bucket != last_bucket:
                last_bucket = bucket
                now = datetime.now().strftime("%H:%M:%S")
                print(f"[{now}] Epoch {epoch} | {progress:.1f}% | Loss {loss.item():.4f}")
            
            global_step += 1

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

    print(f"\n✅ Training complete! Model saved to {SAVE_DIR}")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
