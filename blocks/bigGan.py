import torch.nn as nn

from blocks.adaGn import AdaGn
from config import EMBED_DIM

class ResBlock(nn.Module):
    def __init__(self, in_channels, dropout, embed_dim=EMBED_DIM):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.dropout = dropout

        # First conv
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.norm1 = AdaGn(in_channels, embed_dim=embed_dim)

        # Second conv
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.norm2 = AdaGn(in_channels, embed_dim=embed_dim)

        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)

        self.skip_connection = nn.Identity()
    
    def forward(self, x, t_emb):
        # --- First half ---
        h = self.norm1(x, t_emb)
        h = self.act(h)
        h = self.conv1(h)

        # --- Second half ---
        h = self.norm2(h, t_emb)
        h = self.act(h)
        h = self.drop(h)
        h = self.conv2(h)

        return self.skip_connection(x) + h