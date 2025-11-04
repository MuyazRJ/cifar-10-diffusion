import torch

def make_beta_schedule(schedule_type: str, num_steps: int, start: float, end: float):
    """
    Create a beta schedule for diffusion timesteps.

    Args:
        schedule_type (str): Type of schedule ('linear', 'cosine', etc.)
        num_steps (int): Total diffusion timesteps (T)
        start (float): Starting beta value
        end (float): Ending beta value

    Returns:
        betas (torch.Tensor): [T] beta values between 0 and 1
    """
    if schedule_type == "linear":
        return torch.linspace(start, end, num_steps)
    elif schedule_type == "cosine":
        timesteps = torch.arange(num_steps + 1, dtype=torch.float64) / num_steps
        alphas = torch.cos((timesteps + 0.008) / 1.008 * torch.pi / 2) ** 2
        alphas = alphas / alphas[0]
        betas = 1 - (alphas[1:] / alphas[:-1])
        return betas.clamp(0.0001, 0.9999)
    else:
        raise ValueError(f"Unknown beta schedule: {schedule_type}")


def compute_alphas(betas: torch.Tensor):
    """
    Compute derived alpha quantities used in DDPM.
    Returns:
        alphas (torch.Tensor): 1 - betas
        alpha_bars (torch.Tensor): cumulative product of alphas
    """
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return alphas, alpha_bars
