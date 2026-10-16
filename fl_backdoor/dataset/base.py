"""Abstract base class for dataset loaders."""

from abc import ABC, abstractmethod
from torch.utils.data import DataLoader
from .config import DatasetMeta


class BaseDataset(ABC):
    """Abstract dataset interface for federated learning."""

    meta: DatasetMeta

    @abstractmethod
    def load_partition(
        self, partition_id: int, num_partitions: int, batch_size: int, *, seed: int = 42,
        shuffle_seed: int | None = None, partition_mode: str = "iid",
        partition_alpha: float = 0.5, partition_seed: int | None = None,
        download: bool = True,
    ) -> tuple[DataLoader, DataLoader]:
        """
        Load a partition of the dataset for a specific client.

        Returns:
            (trainloader, testloader) for the client's local data.
        """
        pass

    @abstractmethod
    def load_centralized_test(self, batch_size: int = 128, *, download: bool = True) -> DataLoader:
        """
        Load the centralized test set for global evaluation.

        Returns:
            DataLoader for the full test set.
        """
        pass