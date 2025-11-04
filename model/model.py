import torch
import torch.nn as nn

from config import MODEL_CHANNELS, TIME_EMBED_DIM, SINUSOIDAL_EMBEDDING_DIM, CHANNEL_MULTIPLIERS, NUM_RES_BLOCKS, DROPOUT_RATE

from blocks.sample import DownSample, UpSample
from blocks.bigGan import ResBlock  

class ImprovedDDPM(nn.Module):
    def __init__(self, image_size, image_channels=3):
        super().__init__()

        self.image_size = image_size
        self.image_channels = image_channels
        
        self.time_embedding = nn.Sequential(
            nn.Linear(SINUSOIDAL_EMBEDDING_DIM, TIME_EMBED_DIM),
            nn.SiLU(),
            nn.Linear(TIME_EMBED_DIM, TIME_EMBED_DIM),
        )

        # Initial convolution maps input image to model channels
        self.init_conv = nn.Conv2d(self.image_channels, int(MODEL_CHANNELS * CHANNEL_MULTIPLIERS[0]), kernel_size=3, padding=1)
        self.down_blocks = nn.ModuleList()

        for layer, multiplier in enumerate(CHANNEL_MULTIPLIERS):
            out_channels = int(MODEL_CHANNELS * multiplier)
            for _ in range(NUM_RES_BLOCKS):
                # Add ResBlock here
                pass
            if layer != len(CHANNEL_MULTIPLIERS) - 1:
                # Add DownSample here
                pass