"""Dataset registry and factory."""

from typing import Type
from .base import BaseDataset
from .cifar10 import CIFAR10Dataset
from .mnist import MNISTDataset

_DATASET_REGISTRY: dict[str, Type[BaseDataset]] = {
    "cifar10": CIFAR10Dataset,
    "mnist": MNISTDataset,
    # Add more datasets here as needed
}


def get_dataset(dataset_name: str) -> BaseDataset:
    """Factory function to get a dataset instance by name."""
    name_lower = dataset_name.lower()
    if name_lower not in _DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Supported: {list(_DATASET_REGISTRY.keys())}"
        )
    dataset_cls = _DATASET_REGISTRY[name_lower]
    if dataset_cls is None:
        raise NotImplementedError(
            f"Dataset '{dataset_name}' is registered but not yet implemented."
        )
    return dataset_cls()


def register_dataset(name: str, dataset_cls: Type[BaseDataset]) -> None:
    """Register a custom dataset class (for future extensibility)."""
    _DATASET_REGISTRY[name.lower()] = dataset_cls


def list_datasets() -> list[str]:
    """Return a list of available dataset names."""
    return list(_DATASET_REGISTRY.keys())