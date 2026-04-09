"""CIFAR-10 dataset implementation."""

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
from datasets import load_dataset

from .base import BaseDataset
from .config import DatasetMeta


class CIFAR10Dataset(BaseDataset):
    """CIFAR-10 dataset loader for federated learning."""

    meta = DatasetMeta(
        name="cifar10",
        num_classes=10,
        input_shape=(3, 32, 32),
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        hf_dataset_path="uoft-cs/cifar10",
    )

    def __init__(self):
        self._fds = None  # Cache FederatedDataset
        self.raw_to_tensor = ToTensor()
        self.normalize = Normalize(self.meta.mean, self.meta.std)
        self.transforms = Compose([ToTensor(), self.normalize])

    def _apply_transforms(self, batch):
        """Apply transforms and keep both raw and normalized images."""
        raw_imgs = [self.raw_to_tensor(img) for img in batch["img"]]
        batch["img_raw"] = raw_imgs
        batch["img"] = [self.normalize(img) for img in raw_imgs]
        return batch

    def load_partition(
        self, partition_id: int, num_partitions: int, batch_size: int
    ):
        if self._fds is None:
            partitioner = IidPartitioner(num_partitions=num_partitions)
            self._fds = FederatedDataset(
                dataset=self.meta.hf_dataset_path,
                partitioners={"train": partitioner},
            )
        partition = self._fds.load_partition(partition_id)
        partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
        partition_train_test = partition_train_test.with_transform(
            self._apply_transforms
        )
        trainloader = DataLoader(
            partition_train_test["train"], batch_size=batch_size, shuffle=True
        )
        testloader = DataLoader(
            partition_train_test["test"], batch_size=batch_size
        )
        return trainloader, testloader

    def load_centralized_test(self, batch_size: int = 128):
        test_dataset = load_dataset(self.meta.hf_dataset_path, split="test")
        dataset = test_dataset.with_transform(self._apply_transforms)
        return DataLoader(dataset, batch_size=batch_size)