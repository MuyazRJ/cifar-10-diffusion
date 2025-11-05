import torch

# Configuration parameters for the diffusion model
TIME_EMBED_DIM = 512
SINUSOIDAL_EMBEDDING_DIM = 64

# Group normalization settings
NUM_GROUPS = 32

# Training hyperparameters
LEARNING_RATE = 1e-4
DROPOUT_RATE = 0.1
EPOCHS = 100

# Model architecture settings
MODEL_CHANNELS = 64
CHANNEL_MULTIPLIERS = [1, 2, 4, 8]
ATTENTION_LAYERS = [1, 2, 3]  # Layers at which attention is applied
NUM_RES_BLOCKS = 2

# Schedule type
BETA_SCHEDULE = "cosine"
BETA_START = 1e-4
BETA_END = 0.02
NUM_DIFFUSION_STEPS = 1000

# Data settings
BATCH_SIZE = 128

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"