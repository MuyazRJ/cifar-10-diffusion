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
CHANNEL_MULTIPLIERS = [1, 2, 4, 4]
ATTENTION_LAYERS = [1, 2, 3]  # Layers at which attention is applied
NUM_RES_BLOCKS = 2

# Schedule type
BETA_SCHEDULE = "cosine"
BETA_START = 1e-4
BETA_END = 0.02
NUM_DIFFUSION_STEPS = 1000

# Data settings
BATCH_SIZE = 256

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Output directory
SAVE_DIR = "./model_params"
SAVE_DIR_TRAIN = "./model_checkpoints"

# Load model settings
FILENAME = None  # Specify checkpoint filename to load, or None to load the latest
RESUME_TRAINING = False  # Set to True to resume training from a checkpoint