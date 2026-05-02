from PIL import Image
import random
import os
import numpy as np
from glob import glob
import torch
from torchvision import transforms
from torch.utils.data import Dataset


class RandomResizedCropImageDataset(Dataset):
    def __init__(self, dirs, image_dims, max_samples=None):
        """
        image_dims: (C,H,W)
        train: True = Random crop, False = Center crop or resize
        """
        self.paths = []
        for d in dirs:
            self.paths += glob(os.path.join(d, "*.png"))
            self.paths += glob(os.path.join(d, "*.jpg"))
        self.paths.sort()
        if max_samples is not None:
            self.paths = self.paths[:max_samples]
        _, H, W = image_dims
        self.transform = transforms.Compose(
            [
                transforms.Resize(min(H, W)),  # preserves aspect ratio
                transforms.CenterCrop((H, W)),  # exact final size
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img_tensor = self.transform(img)
        return img_tensor


def get_dataset(data_dirs, config):
    return RandomResizedCropImageDataset(
        dirs=data_dirs,
        image_dims=config.image_dims,
        max_samples=config.max_test_samples,
    )
