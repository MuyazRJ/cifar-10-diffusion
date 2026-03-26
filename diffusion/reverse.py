# Author: Mohammed Rahman
# Student ID: 10971320
# University of Manchester — BSc Computer Science Final Year Project, 2026
#
# Reverse diffusion sampling loop for the Improved DDPM on CIFAR-10.
# Iterates from t=T-1 to t=0, predicting the posterior mean and learned
# variance at each step to progressively denoise from pure Gaussian noise.
#
# Based on:
# - Ho et al., "Denoising Diffusion Probabilistic Models", NeurIPS 2020
#   https://arxiv.org/abs/2006.11239
# - Nichol & Dhariwal, "Improved Denoising Diffusion Probabilistic Models", ICML 2021
#   https://arxiv.org/abs/2102.09672

import torch 

from config import DEVICE, IMAGE_CLASSES


def reverse(model, alphas, alpha_bars, betas, T, num_images=1, labels=None):
    # Start from random Gaussian noise
    x_t = torch.randn((num_images, model.image_channels, 32, 32), device=DEVICE)

    model = model.to(DEVICE)

    # Use random class labels if none are provided
    if labels is None:
        labels = torch.randint(0, IMAGE_CLASSES, (num_images,), device=DEVICE, dtype=torch.long)
    else:
        labels.to(DEVICE)

    # Reverse diffusion loop from timestep T-1 down to 0
    for t in reversed(range(T)):
        t_tensor = torch.full((num_images,), t, device=DEVICE, dtype=torch.long)

        with torch.no_grad():
            # Predict the noise and variance values for the current timestep
            eps_pred, var_raw = model(x_t, t_tensor, labels)

        # Estimate the clean image x0 from the current noisy sample
        alpha_bar_t = alpha_bars[t].view(1,1,1,1)
        x0_pred = (x_t - torch.sqrt(1 - alpha_bar_t) * eps_pred) / torch.sqrt(alpha_bar_t)

        # Clamp predicted image values to the valid range
        x0_pred = x0_pred.clamp(-1, 1)

        # Get the diffusion coefficients for the current timestep
        alpha_t = alphas[t].view(1,1,1,1)
        beta_t = betas[t].view(1,1,1,1)

        if t > 0:
            alpha_bar_prev = alpha_bars[t-1].view(1,1,1,1)
        else:
            alpha_bar_prev = torch.tensor(1.0, device=DEVICE).view(1,1,1,1)

        # Compute the posterior mean for the reverse step
        coef1 = (torch.sqrt(alpha_bar_prev) * beta_t) / (1 - alpha_bar_t)
        coef2 = (torch.sqrt(alpha_t) * (1 - alpha_bar_prev)) / (1 - alpha_bar_t)
        mean = coef1 * x0_pred + coef2 * x_t

        # Compute the posterior variance bounds
        tilde_beta_t = beta_t * (1 - alpha_bar_prev) / (1 - alpha_bar_t)

        # Map predicted variance values into the valid interpolation range
        frac = (var_raw + 1) / 2
        frac = frac.clamp(0, 1)

        min_log = torch.log(tilde_beta_t)
        max_log = torch.log(beta_t)

        # Interpolate between minimum and maximum log variance
        model_log_sigma2 = frac * max_log + (1 - frac) * min_log
        model_sigma2 = torch.exp(model_log_sigma2)

        # Sample the next less-noisy image, except at the final step
        if t > 0:
            noise = torch.randn_like(x_t)
            x_t = mean + torch.sqrt(model_sigma2) * noise
        else:
            x_t = mean

    # Clamp final output and rescale from [-1,1] to [0,1]
    x_t = x_t.clamp(-1, 1)
    return (x_t + 1) / 2