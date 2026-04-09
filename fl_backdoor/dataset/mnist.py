"""MNIST dataset using torchvision (fully offline-safe, no HF cache issues)."""

import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

from .base import BaseDataset
from .config import DatasetMeta


class MNISTDataset(BaseDataset):
    meta = DatasetMeta(
        name="mnist",
        num_classes=10,
        input_shape=(1, 28, 28),
        mean=(0.1307,),
        std=(0.3081,),
        hf_dataset_path="",  # not used
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

    def load_partition(self, partition_id: int, num_partitions: int, batch_size: int):
        # 下载/加载 MNIST 训练集
        full_train = datasets.MNIST(
            root="./data", train=True, download=True, transform=None
        )

        # 包装以应用 transforms
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

        # 手动按 IID 划分给每个客户端
        total = len(wrapped)
        indices = list(range(total))
        torch.manual_seed(42)
        indices = torch.randperm(total).tolist()

        # 每个客户端应得样本数
        per_client = total // num_partitions
        start = partition_id * per_client
        end = start + per_client if partition_id < num_partitions - 1 else total
        client_indices = indices[start:end]

        client_subset = Subset(wrapped, client_indices)

        # 划分本地训练/验证集 (80/20)
        n_local = len(client_subset)
        n_val = int(0.2 * n_local)
        n_train = n_local - n_val
        train_subset, val_subset = random_split(
            client_subset, [n_train, n_val],
            generator=torch.Generator().manual_seed(42)
        )

        trainloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        testloader = DataLoader(val_subset, batch_size=batch_size)
        return trainloader, testloader

    def load_centralized_test(self, batch_size: int = 128):
        full_test = datasets.MNIST(
            root="./data", train=False, download=True, transform=None
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