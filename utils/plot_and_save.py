import os
import math
import torch
import matplotlib.pyplot as plt

def save_image_grid(images: torch.Tensor, step: int = 1, out_dir: str = "outputs", nrow: int = 10, show: bool = False):
    """
    Save a grid of images using matplotlib.

    Args:
        images (torch.Tensor): Tensor of shape [B, C, H, W], values in [-1, 1].
        step (int): Current training step or epoch number.
        out_dir (str): Directory where images are saved.
        nrow (int): Number of images per row in the grid.
        show (bool): If True, display the figure inline (e.g. in Jupyter).
    """
    os.makedirs(out_dir, exist_ok=True)
    images = images.detach().cpu().clamp(-1, 1)
    images = (images + 1) / 2  # normalize to [0, 1]
    B, C, H, W = images.shape

    ncol = math.ceil(B / nrow)
    fig, axes = plt.subplots(ncol, nrow, figsize=(nrow, ncol))

    # Handle if number of images < grid cells
    for ax in axes.flat:
        ax.axis("off")

    for i, img in enumerate(images):
        r, c = divmod(i, nrow)
        ax = axes[r][c] if ncol > 1 else axes[c]
        img = img.permute(1, 2, 0).numpy()  # CHW → HWC
        ax.imshow(img)
        ax.axis("off")

    plt.subplots_adjust(wspace=0.02, hspace=0.02)
    save_path = os.path.join(out_dir, f"step_{step:04d}.png")
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.05)
    if show:
        plt.show()
    plt.close(fig)
