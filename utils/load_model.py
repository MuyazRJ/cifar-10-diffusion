from config import DEVICE, SAVE_DIR_TRAIN
from model.model import ImprovedDDPM

import torch
import os 

latest_checkpoint = None

def load_model_and_optimizer(model, optimizer, filename=None):
    """
    Load model & optimizer state.
    If filename is provided, load that specific checkpoint.
    Otherwise load the newest checkpoint in SAVE_DIR_TRAIN.
    """
    checkpoint_path = None

    # Case 1: user provided a specific checkpoint filename
    if filename is not None:
        checkpoint_path = os.path.join(SAVE_DIR_TRAIN, filename)
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint '{filename}' not found in {SAVE_DIR_TRAIN}")
    
    # Case 2: no filename → load newest checkpoint
    else:
        if os.path.isdir(SAVE_DIR_TRAIN):
            checkpoints = [f for f in os.listdir(SAVE_DIR_TRAIN) if f.endswith(".pt")]
            if checkpoints:
                newest = sorted(checkpoints)[-1]
                checkpoint_path = os.path.join(SAVE_DIR_TRAIN, newest)

    # If no checkpoint exists at all
    if not checkpoint_path:
        print("🟢 No checkpoint found. Starting fresh.")
        return model, optimizer, 0

    # Load checkpoint
    print(f"🔄 Resuming from checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    start_epoch = checkpoint["epoch"] + 1

    return model, optimizer, start_epoch
