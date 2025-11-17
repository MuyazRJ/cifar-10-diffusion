import torch
import torch.nn as nn

from config import MODEL_CHANNELS, TIME_EMBED_DIM, SINUSOIDAL_EMBEDDING_DIM, CHANNEL_MULTIPLIERS, NUM_RES_BLOCKS, DROPOUT_RATE, ATTENTION_LAYERS, IMAGE_CLASSES


from blocks.sample import DownSample, UpSample
from blocks.bigGan import ResBlock, TimestepEmbedSequential
from blocks.attention import Attention

from embeddings.sinusoidal import SinusoidalTimeEmbedding

class ImprovedDDPM(nn.Module):
    def __init__(self, image_channels: int=3):
        super().__init__()

        self.embedder = SinusoidalTimeEmbedding()

        self.image_channels = image_channels
        self.time_embedding = nn.Sequential(
            nn.Linear(SINUSOIDAL_EMBEDDING_DIM, TIME_EMBED_DIM),
            nn.SiLU(),
            nn.Linear(TIME_EMBED_DIM, TIME_EMBED_DIM),
        )

        self.class_embedding = nn.Embedding(IMAGE_CLASSES, TIME_EMBED_DIM)  # For class conditioning

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
                    ResBlock(in_channels=in_channels, out_channels=out_channels, dropout=DROPOUT_RATE, embed_dim=TIME_EMBED_DIM)
                )
                in_channels = out_channels

                if layer in ATTENTION_LAYERS:
                    res_layers.append(Attention(out_channels))
            
            self.down_blocks.append(TimestepEmbedSequential(*res_layers))

            # Add Downsample except at last level
            if layer != len(CHANNEL_MULTIPLIERS) - 1:
                self.down_blocks.append(DownSample(out_channels))

        
        self.bottleneck = TimestepEmbedSequential(
            ResBlock(in_channels, dropout=DROPOUT_RATE, embed_dim=TIME_EMBED_DIM),
            Attention(in_channels),
            ResBlock(in_channels, dropout=DROPOUT_RATE, embed_dim=TIME_EMBED_DIM),
        )

        self.up_blocks = nn.ModuleList()
        for layer, multiplier in reversed(list(enumerate(CHANNEL_MULTIPLIERS))):
            out_channels = int(MODEL_CHANNELS * multiplier)
            res_layers = []

            for i in range(NUM_RES_BLOCKS + 1):
                # --- First ResBlock merges skip connection ---
                if i == 0:
                    res_layers.append(
                        ResBlock(in_channels + out_channels, out_channels=out_channels,
                                dropout=DROPOUT_RATE, embed_dim=TIME_EMBED_DIM)
                    )
                else:
                    res_layers.append(
                        ResBlock(out_channels, out_channels=out_channels,
                                dropout=DROPOUT_RATE, embed_dim=TIME_EMBED_DIM)
                    )

                if layer in ATTENTION_LAYERS:
                    res_layers.append(Attention(out_channels))

            # --- Add Upsample except final level ---
            if layer != 0:
                res_layers.append(UpSample(out_channels))

            # Add this level as a timestep-aware sequential
            self.up_blocks.append(TimestepEmbedSequential(*res_layers))

            # Update for next loop
            in_channels = out_channels

        self.out = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, self.image_channels, kernel_size=3, padding=1),
        )
    
    def forward(self, x, t, c):
        # Get time embeddings
        t = self.embedder(t)
        t_emb = self.time_embedding(t) + self.class_embedding(c)  # Combine time and class embeddings

        # Initial conv
        x = self.init_conv(x)

        # Downsampling path
        skip_connections = []
        for down_block in self.down_blocks:
            if not isinstance(down_block, DownSample):
                x = down_block(x, t_emb)
                skip_connections.append(x)
            else:
                x = down_block(x)

        # Bottleneck
        x = self.bottleneck(x, t_emb)

        # Upsampling path
        for up_block in self.up_blocks:
            skip_x = skip_connections.pop()
            x = torch.cat([x, skip_x], dim=1)
            x = up_block(x, t_emb)
            

        # Final output layer
        x = self.out(x)
        return x