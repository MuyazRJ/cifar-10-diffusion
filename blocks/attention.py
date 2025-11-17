import torch.nn as nn 
import torch
import math 

class Attention(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()

        self.channels  = channels
        self.num_heads = num_heads

        self.norm = nn.GroupNorm(num_groups=32, num_channels=channels, affine=True)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)

        self.attention = QKVAttention(self.num_heads)

        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)
    
    def forward(self, x):
        B, C, H, W = x.shape

        # Normalize
        h = self.norm(x)

        # QKV projection
        qkv = self.qkv(h)                 # (B, 3C, H, W)
        qkv = qkv.reshape(B, 3 * C, H * W)  # flatten spatial dims → (B, 3C, T)

        # Apply multi-head attention
        h = self.attention(qkv)           # (B, C, T)
        h = h.reshape(B, C, H, W)         # reshape back to 2D

        # Output projection + residual
        h = self.proj_out(h)
        return x + h

class QKVAttention(nn.Module):
    """
    Multi-head QKV (Query-Key-Value) attention module adapted from the
    original OpenAI diffusion model implementation:
    https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/unet.py

    This version performs the Q, K, and V splitting in a slightly different order
    compared to standard Transformer attention. It operates on flattened spatial
    feature maps and is optimized for use inside diffusion U-Nets.
    """
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)
