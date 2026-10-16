"""CIFAR-10 dataset using torchvision (offline-safe, no HF dependency)."""

import os

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .base import BaseDataset
from .config import DatasetMeta
from .partition import partition_loaders


class CIFAR10Dataset(BaseDataset):
    meta = DatasetMeta(
        name="cifar10",
        num_classes=10,
        input_shape=(3, 32, 32),
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        hf_dataset_path="",
    )

    def __init__(self):
        self.raw_to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(self.meta.mean, self.meta.std)

    def _apply_transforms(self, data):
        img, label = data
        img_tensor = self.raw_to_tensor(img)
        return {
            "img_raw": img_tensor,
            "img": self.normalize(img_tensor),
            "label": label,
        }

    def load_partition(self, partition_id: int, num_partitions: int, batch_size: int, *, seed=42, shuffle_seed=None, download=True):
        full_train = datasets.CIFAR10(
            root=os.environ.get("FL_BACKDOOR_DATA_DIR", "./data"), train=True, download=download, transform=None
        )

        class WrappedDataset(torch.utils.data.Dataset):
            def __init__(self, dataset, transform_fn):
                self.dataset = dataset
                self.transform_fn = transform_fn

            def __len__(self):
                return len(self.dataset)

            def __getitem__(self, idx):
                img, label = self.dataset[idx]
                return self.transform_fn((img, label))

        wrapped = WrappedDataset(full_train, self._apply_transforms)

        return partition_loaders(wrapped, partition_id, num_partitions, batch_size,
                                 seed=seed, shuffle_seed=shuffle_seed)

    def load_centralized_test(self, batch_size: int = 128, *, download=True):
        full_test = datasets.CIFAR10(
            root=os.environ.get("FL_BACKDOOR_DATA_DIR", "./data"), train=False, download=download, transform=None
        )

        class WrappedDataset(torch.utils.data.Dataset):
            def __init__(self, dataset, transform_fn):
                self.dataset = dataset
                self.transform_fn = transform_fn

            def __len__(self):
                return len(self.dataset)

            def __getitem__(self, idx):
                img, label = self.dataset[idx]
                return self.transform_fn((img, label))

        wrapped = WrappedDataset(full_test, self._apply_transforms)
        return DataLoader(wrapped, batch_size=batch_size)