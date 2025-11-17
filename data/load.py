import os
import pickle
import torch
from torch.utils.data import Dataset, DataLoader

from config import BATCH_SIZE

import numpy as np


class CIFAR10TrainDataset(Dataset):
    """
    Loads all 50k CIFAR-10 training images (data_batch_1–5)
    and normalizes them to [-1, 1] for diffusion training.
    """
    def __init__(self, root):
        self.root = root
        data_list, label_list = [], []

        # --- Load all 5 training batches ---
        for i in range(1, 6):
            path = os.path.join(root, f"data_batch_{i}")
            with open(path, "rb") as f:
                batch = pickle.load(f, encoding="bytes")

            imgs = torch.tensor(batch[b"data"], dtype=torch.float32)
            imgs = imgs.view(-1, 3, 32, 32)       # (N, 3, 32, 32)
            imgs = imgs / 127.5 - 1.0             # Normalize to [-1, 1]

            labels = torch.tensor(batch[b"labels"], dtype=torch.long)
            data_list.append(imgs)
            label_list.append(labels)

        self.images = torch.cat(data_list)
        self.labels = torch.cat(label_list)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def get_cifar10_dataloader(root="./cifar-10-batches-py",
                            batch_size=BATCH_SIZE,
                            num_workers=2,
                            shuffle=True):
    dataset = CIFAR10TrainDataset(root)
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        num_workers=num_workers,
                        pin_memory=True)
    return loader


import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from config import BATCH_SIZE
import torch.nn.functional as F


class MNISTTrainDataset(Dataset):
    def __init__(self, image_file):
        with open(image_file, "rb") as f:
            data = f.read()

        images = np.frombuffer(data, dtype=np.uint8, offset=16)
        num_images = len(images) // (28 * 28)
        images = images.reshape(num_images, 28, 28)
        images = images.astype(np.float32) / 127.5 - 1.0

        # Store as tensor (N, 1, 28, 28)
        self.images = torch.tensor(images).unsqueeze(1)

        # Resize MNIST from (1, 28, 28) → (1, 32, 32)
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]        # (1, 28, 28)

        # Resize to (1, 32, 32)
        img = F.interpolate(
            img.unsqueeze(0),         # → (1, 1, 28, 28)
            size=(32, 32),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)                  # back to (1, 32, 32)

        return img


def get_mnist_dataloader(
        image_file="../mnist_diffusion/mnist/train-images-idx3-ubyte/train-images-idx3-ubyte",
        batch_size=BATCH_SIZE,
        num_workers=2,
        shuffle=True
    ):
    dataset = MNISTTrainDataset(image_file)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    return loader