# Configuration parameters for the diffusion model
TIME_EMBED_DIM = 512
SINUSOIDAL_EMBEDDING_DIM = 64

# Group normalization settings
NUM_GROUPS = 32

# Training hyperparameters
LEARNING_RATE = 1e-4
DROPOUT_RATE = 0.1

# Model architecture settings
MODEL_CHANNELS = 64
CHANNEL_MULTIPLIERS = [1, 2, 4, 8]
NUM_RES_BLOCKS = 2