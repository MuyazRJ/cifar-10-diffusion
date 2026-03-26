"""
generate_cifar.py
-----------------
Loads the trained CIFAR-10 Improved DDPM (EMA weights) and produces:
  - final_image : PIL.Image  (32x32 upscaled to 256x256, RGB)
  - frames      : list[PIL.Image]  (frames every 100 steps, t=T down to t=0,
                  ordered noisy -> clean, timestep stamped on each frame)

Usage in main.py
----------------
    from generate_cifar import load_cifar_model, generate_cifar

    cifar_bundle = load_cifar_model()                        # once at startup
    img, frames  = generate_cifar(cifar_bundle)              # unconditional
    img, frames  = generate_cifar(cifar_bundle, class_idx=3) # class-conditional
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "models/cifar"))

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from model.model import ImprovedDDPM
from utils.ema import EMA
from diffusion.schedules import compute_alphas, make_beta_schedule
from config import (
    DEVICE, NUM_DIFFUSION_STEPS, BETA_START, BETA_END,
    LEARNING_RATE, EMA_DECAY, IMAGE_CLASSES,
)

# ── Config ────────────────────────────────────────────────────────────────────

CKPT_PATH    = os.path.join(os.path.dirname(__file__), "models/cifar/model_dict/final_ddpm.pt")
DISPLAY_SIZE = 256
SAVE_EVERY   = 100   # capture at t = T, T-100, ..., 100, 0

CIFAR_CLASSES = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck",
]

# ── Model loading ─────────────────────────────────────────────────────────────

def load_cifar_model(ckpt_path: str = CKPT_PATH):
    """
    Load the EMA model and precomputed schedules.
    Returns a dict bundle — pass it directly to generate_cifar().
    """
    device = torch.device(DEVICE)

    model = ImprovedDDPM().to(device)
    ema_model = EMA(model, decay=EMA_DECAY)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE,
        betas=(0.9, 0.999), weight_decay=1e-4,
    )

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    ema_model.ema_model.load_state_dict(checkpoint["ema_state_dict"])
    ema_model.ema_model.eval()

    betas      = make_beta_schedule("cosine", num_steps=NUM_DIFFUSION_STEPS,
                                    start=BETA_START, end=BETA_END).to(device).float()
    alphas, alpha_bars = compute_alphas(betas)

    print(f"[CIFAR] Loaded EMA weights from {ckpt_path} on {device}")

    return {
        "model":       ema_model.ema_model,
        "alphas":      alphas,
        "alpha_bars":  alpha_bars,
        "betas":       betas,
        "device":      device,
    }

# ── Helper: tensor -> PIL ─────────────────────────────────────────────────────

def _to_pil(tensor: torch.Tensor, timestep: int = None) -> Image.Image:
    """
    Convert a (1, 3, H, W) tensor in [0, 1] to a 256x256 RGB PIL Image.
    Optionally stamps the timestep and class label in the top-left corner.
    """
    arr = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
    arr = (arr * 255).clip(0, 255).astype(np.uint8)
    pil = Image.fromarray(arr, mode="RGB")
    pil = pil.resize((DISPLAY_SIZE, DISPLAY_SIZE), Image.NEAREST)

    if timestep is not None:
        draw = ImageDraw.Draw(pil)
        label = f"t = {timestep}"
        try:
            font = ImageFont.truetype("arial.ttf", 22)
        except IOError:
            font = ImageFont.load_default()
        draw.text((11, 11), label, fill=(0, 0, 0), font=font)
        draw.text((10, 10), label, fill=(255, 255, 255), font=font)

    return pil

# ── Reverse loop with frame capture ──────────────────────────────────────────

def _reverse_with_frames(model, alphas, alpha_bars, betas, T, label_idx, device):
    """
    Run one reverse diffusion pass for a single image of class `label_idx`,
    capturing a frame every SAVE_EVERY steps.

    Returns
    -------
    x_final : torch.Tensor  (1, 3, 32, 32) in [0, 1]
    frames  : list of (tensor_in_0_1, t)   ordered t=T down to t=0
    """
    x_t    = torch.randn((1, model.image_channels, 32, 32), device=device)
    labels = torch.tensor([label_idx], device=device, dtype=torch.long)
    frames = []

    for t in reversed(range(T)):
        t_tensor = torch.full((1,), t, device=device, dtype=torch.long)

        with torch.no_grad():
            eps_pred, var_raw = model(x_t, t_tensor, labels)

        alpha_bar_t = alpha_bars[t].view(1, 1, 1, 1)
        x0_pred     = (x_t - torch.sqrt(1 - alpha_bar_t) * eps_pred) / torch.sqrt(alpha_bar_t)
        x0_pred     = x0_pred.clamp(-1, 1)

        alpha_t    = alphas[t].view(1, 1, 1, 1)
        beta_t     = betas[t].view(1, 1, 1, 1)
        alpha_bar_prev = alpha_bars[t - 1].view(1, 1, 1, 1) if t > 0 else torch.tensor(1.0, device=device).view(1, 1, 1, 1)

        coef1 = (torch.sqrt(alpha_bar_prev) * beta_t) / (1 - alpha_bar_t)
        coef2 = (torch.sqrt(alpha_t) * (1 - alpha_bar_prev)) / (1 - alpha_bar_t)
        mean  = coef1 * x0_pred + coef2 * x_t

        tilde_beta_t   = beta_t * (1 - alpha_bar_prev) / (1 - alpha_bar_t)
        frac           = ((var_raw + 1) / 2).clamp(0, 1)
        model_log_sig2 = frac * torch.log(beta_t) + (1 - frac) * torch.log(tilde_beta_t)

        if t > 0:
            x_t = mean + torch.sqrt(torch.exp(model_log_sig2)) * torch.randn_like(x_t)
        else:
            x_t = mean

        if t % SAVE_EVERY == 0 or t == T - 1:
            snapshot = x_t.clamp(-1, 1)
            snapshot = (snapshot + 1.0) / 2.0   # [0, 1] for display
            frames.append((snapshot.detach().cpu().clone(), t))

    x_final = x_t.clamp(-1, 1)
    x_final = (x_final + 1.0) / 2.0
    return x_final, frames

# ── Main generation function ──────────────────────────────────────────────────

def generate_cifar(bundle: dict, class_idx: int = None):
    """
    Run one reverse diffusion pass for a single CIFAR-10 image.

    Parameters
    ----------
    bundle    : dict returned by load_cifar_model()
    class_idx : int 0-9, or None for a random class

    Returns
    -------
    final_image : PIL.Image   — clean 256x256 RGB image
    frames      : list[PIL.Image] — noisy -> clean GIF frames with t stamped
    class_name  : str  — e.g. "Cat" (useful for the frontend label)
    """
    model      = bundle["model"]
    alphas     = bundle["alphas"]
    alpha_bars = bundle["alpha_bars"]
    betas      = bundle["betas"]
    device     = bundle["device"]
    T          = NUM_DIFFUSION_STEPS

    if class_idx is None:
        class_idx = int(torch.randint(0, IMAGE_CLASSES, (1,)).item())

    x_final, raw_frames = _reverse_with_frames(
        model, alphas, alpha_bars, betas, T, class_idx, device,
    )

    # raw_frames is ordered t=T-1 down to t=0; reverse for noisy -> clean GIF
    raw_frames_sorted = sorted(raw_frames, key=lambda x: x[1], reverse=True)

    final_image = _to_pil(x_final)
    frames      = [_to_pil(tensor, timestep=t) for tensor, t in raw_frames_sorted]
    class_name  = CIFAR_CLASSES[class_idx]

    return final_image, frames, class_name