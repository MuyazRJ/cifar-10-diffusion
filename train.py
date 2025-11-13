import torch
import os

import torch.nn.functional as F

from config import NUM_DIFFUSION_STEPS, BETA_START, BETA_END, EPOCHS, DEVICE, SAVE_DIR, SAVE_DIR_TRAIN, RESUME_TRAINING
from data.load import get_cifar10_dataloader

from diffusion.schedules import compute_alphas, make_beta_schedule
from diffusion.forward import q_sample
from diffusion.reverse import reverse  

from utils.plot_and_save import save_image_grid
from utils.load_model import load_model_and_optimizer
from embeddings.sinusoidal import SinusoidalTimeEmbedding

from datetime import datetime
from model.model import ImprovedDDPM

def main():
    betas = make_beta_schedule("cosine", num_steps=NUM_DIFFUSION_STEPS, start=BETA_START, end=BETA_END).to(DEVICE).float()
    alphas, alpha_bars = compute_alphas(betas)

    train_loader = get_cifar10_dataloader() 

    model = ImprovedDDPM().to(DEVICE)
    criterion = torch.nn.MSELoss()
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-4,           # commonly 1e-4 to 2e-4
        betas=(0.9, 0.999),
        weight_decay=1e-4  # small decay improves stability
    )

    embedder = SinusoidalTimeEmbedding()
    start_epoch = 0

    if RESUME_TRAINING:
        model, optimizer, start_epoch = load_model_and_optimizer(model, optimizer)

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        for  batch_idx, (images, _) in enumerate(train_loader):
            images = images.to(DEVICE)
            batch_size = images.size(0)
            
            # 1️⃣ Sample random timesteps for each image in the batch
            t = torch.randint(0, NUM_DIFFUSION_STEPS, (batch_size,), device=DEVICE)  # random timestep for each image
            t_embeddings = embedder(t).to(DEVICE)

            # 2️⃣ Forward diffusion (add noise)
            noisy_images, noise = q_sample(images, t, alpha_bars.to(DEVICE))

            # 3️⃣ Model predicts both ε and v
            out = model(noisy_images, t_embeddings)
            eps_pred, v_pred = torch.chunk(out, 2, dim=1)
            
            # 4️⃣ Compute ε loss (main objective)
            eps_loss = criterion(eps_pred, noise)

            # 5️⃣ Compute small auxiliary v loss (stabilization)
            v_loss = torch.mean(v_pred ** 2)  # simple proxy, acts as regularizer

            # 6️⃣ Total loss
            loss = eps_loss + 1e-3 * v_loss  # weight = 0.001 as in Improved DDPM

            # 7️⃣ Optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress = 100 * (batch_idx + 1) / len(train_loader)
            
            cos = F.cosine_similarity(eps_pred.flatten(), noise.flatten(), dim=0)
            print(f"Batch {epoch} | Loss {loss.item():.4f} | ε cosine similarity: {cos.item():.3f} | Progress: {progress:.1f}% ")

        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            model.eval()   # <--- VERY IMPORTANT
            with torch.no_grad():
                image = reverse(model, NUM_DIFFUSION_STEPS, (1, 3, 32, 32), alphas, alpha_bars, DEVICE)
            save_image_grid(image, epoch)

            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            save_path = f"{SAVE_DIR_TRAIN}/model_epoch_{epoch}_{timestamp}.pt"

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, save_path)


    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR, "final_ddpm.pt")
    torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    }, save_path)

    print(f"\n✅ Training complete! Model saved to {SAVE_DIR}")



if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()  # for Windows executables
    main()