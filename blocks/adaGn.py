import torch.nn as nn

from config import TIME_EMBED_DIM, NUM_GROUPS

class AdaGn(nn.Module):
    def __init__(self, num_channels, num_groups=NUM_GROUPS, embed_dim=TIME_EMBED_DIM):
        """Adaptive Group Normalization layer."""
        super().__init__()
        self.gn = nn.GroupNorm(num_groups, num_channels, affine=False)
        self.linear = nn.Linear(embed_dim, num_channels * 2)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x, t_emb):
        """Forward pass of AdaGn."""
        # t_emb: [B, embed_dim]
        scale, shift = self.linear(t_emb).chunk(2, dim=1)
        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]
        x = self.gn(x)
        return x * (1 + scale) + shift