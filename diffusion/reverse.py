import torch 

from embeddings.sinusoidal import SinusoidalTimeEmbedding
from config import DEVICE

"""def p_sample(model, x_t, t, alphas, alpha_bars):
    
    Single reverse diffusion step.
    model predicts both ε and log variance (σ²).
    

    with torch.no_grad():
        # 1. Predict ε and log variance
        out = model(x_t, t)
        eps_pred, log_var_pred = torch.chunk(out, 2, dim=1)

        # 2. Compute posterior mean μθ(x_t, t)
        alpha_t     = alphas[t].view(-1, 1, 1, 1)
        alpha_bar_t = alpha_bars[t].view(-1, 1, 1, 1)
        mean = (x_t - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * eps_pred) \
               / torch.sqrt(alpha_t)
        
        # 3. Compute variance σ_t^2
        sigma = torch.exp(0.5 * log_var_pred)

        # 4. Sample x_{t-1}
        if t > 0:
            z = torch.randn_like(x_t)
        else:
            z = 0
        x_prev = mean + sigma * z
    
    return x_prev

import torch"""

def p_sample(model, x_t, t, alphas, alpha_bars, t_embeddings):
    """
    Single reverse diffusion step.
    Implements the ADM (Improved DDPM) variance parameterization.
    
    Model output: concat([ε_pred, v_pred], dim=1)
      - ε_pred: predicted noise
      - v_pred: interpolation scalar for variance (sigmoid-bounded)
    """

    with torch.no_grad():
        # 1. Predict ε and v
        out = model(x_t, t_embeddings)
        eps_pred, v_pred = torch.chunk(out, 2, dim=1)

        # 2. Compute posterior mean μθ(x_t, t)
        alpha_t     = alphas[t].view(-1, 1, 1, 1)
        alpha_bar_t = alpha_bars[t].view(-1, 1, 1, 1)
        mean = (x_t - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * eps_pred) \
               / torch.sqrt(alpha_t)
        
        # 3. Compute β_t and \tilde{β}_t
        beta_t = 1 - alpha_t
        if t > 0:
            alpha_bar_prev = alpha_bars[t - 1].view(-1, 1, 1, 1)
        else:
            alpha_bar_prev = torch.ones_like(alpha_bar_t)
        tilde_beta_t = ((1 - alpha_bar_prev) / (1 - alpha_bar_t)) * beta_t
        tilde_beta_t = torch.clamp(tilde_beta_t, min=1e-9)

        # 4. Interpolate variance in log-space
        s = torch.sigmoid(v_pred)  # bounded interpolation factor
        log_variance = s * torch.log(beta_t) + (1 - s) * torch.log(tilde_beta_t)
        sigma = torch.exp(0.5 * log_variance)

        # 5. Sample x_{t-1}
        z = torch.randn_like(x_t)
        x_prev = mean + sigma * z

    return x_prev


def reverse(model, T, shape, alphas, alpha_bars, device):
    """
    Full reverse diffusion process.
    model predicts both ε and log variance (σ²).
    """
    x_t = torch.randn(shape, device=device)
    embedder = SinusoidalTimeEmbedding()

    for t in reversed(range(T)):
        t_tensor = torch.tensor([t], device=device).long()
        t_embeddings = embedder(t_tensor).to(DEVICE)
        x_t = p_sample(model, x_t, t, alphas, alpha_bars, t_embeddings)
    
    # Map back from [-1,1] to [0,1]
    x_t = x_t.clamp(-1, 1)
    x_t = (x_t + 1) / 2
    return x_t