from config import DEVICE, SAVE_DIR_TRAIN
from model.model import ImprovedDDPM

import torch
import os 

latest_checkpoint = None

def load_model_and_optimizer(model, optimizer):
    latest_checkpoint = None

    if os.path.isdir(SAVE_DIR_TRAIN):
        checkpoints = [f for f in os.listdir(SAVE_DIR_TRAIN) if f.endswith(".pt")]
        if checkpoints:
            latest_checkpoint = sorted(checkpoints)[-1]

    if latest_checkpoint:
        path = os.path.join(SAVE_DIR_TRAIN, latest_checkpoint)
        print(f"🔄 Resuming from checkpoint: {path}")

        checkpoint = torch.load(path, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        start_epoch = checkpoint["epoch"] + 1
    else:
        start_epoch = 0
        print("🟢 No checkpoint found. Starting fresh.")
    
    return model, optimizer, start_epoch
