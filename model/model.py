import torch
import torch.nn as nn

from config import MODEL_CHANNELS, TIME_EMBED_DIM, SINUSOIDAL_EMBEDDING_DIM, CHANNEL_MULTIPLIERS, NUM_RES_BLOCKS, DROPOUT_RATE, ATTENTION_LAYERS

from blocks.sample import DownSample, UpSample
from blocks.bigGan import ResBlock, TimestepEmbedSequential
from blocks.attention import Attention

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
        in_channels = int(MODEL_CHANNELS * CHANNEL_MULTIPLIERS[0])
        self.init_conv = nn.Conv2d(self.image_channels, in_channels, kernel_size=3, padding=1)

        self.down_blocks = nn.ModuleList()
        for layer, multiplier in enumerate(CHANNEL_MULTIPLIERS):
            out_channels = int(MODEL_CHANNELS * multiplier)
            res_layers = []

            # Add two ResBlocks
            for _ in range(NUM_RES_BLOCKS):
                res_layers.append(
                    ResBlock(in_channels, dropout=DROPOUT_RATE, embed_dim=TIME_EMBED_DIM)
                )
                in_channels = out_channels

            if layer in ATTENTION_LAYERS:
                res_layers.append(Attention(out_channels))

            # Add Downsample except at last level
            if layer != len(CHANNEL_MULTIPLIERS) - 1:
                res_layers.append(DownSample(out_channels))

            # Wrap them so forward(x, t_emb) works automatically
            self.down_blocks.append(TimestepEmbedSequential(*res_layers))

