import torch 

def p_sample(model, x_t, t, alphas, alpha_bars):
    """
    Single reverse diffusion step.
    model predicts both ε and log variance (σ²).
    """

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

def reverse(model, T, shape, alphas, alpha_bars, device):
    """
    Full reverse diffusion process.
    model predicts both ε and log variance (σ²).
    """
    x_t = torch.randn(shape, device=device)

    for t in reversed(range(T)):
        x_t = p_sample(model, x_t, t, alphas, alpha_bars)
    
    # Map back from [-1,1] to [0,1]
    x_t = x_t.clamp(-1, 1)
    x_t = (x_t + 1) / 2
    return x_t