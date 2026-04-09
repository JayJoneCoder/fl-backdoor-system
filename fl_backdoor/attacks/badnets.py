"""BadNets attack utilities.

This module is intentionally independent from client/server logic so that
future attacks (e.g. WaNet, frequency-domain attacks) can follow the same API.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .base import AttackBase, AttackConfig
from .selection import (
    is_malicious_client,
    normalize_fixed_malicious_clients,
    select_malicious_clients,
)

def add_trigger(
    image: torch.Tensor,
    trigger_size: int = 4,
    value: float = 1.0,
    location: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Add a fixed square trigger at specified location (top-left corner).

    Supports:
    - [C, H, W]
    - [H, W] (fallback)

    Args:
        image: Input image tensor.
        trigger_size: Size of the square trigger.
        value: Pixel value to fill.
        location: (row, col) of top-left corner. If None, place at bottom-right.
    """
    image = image.clone()
    if image.dim() == 3:
        _, h, w = image.shape
        if location is None:
            row_start = h - trigger_size
            col_start = w - trigger_size
        else:
            row_start, col_start = location
        row_end = row_start + trigger_size
        col_end = col_start + trigger_size
        # Clamp to image boundaries
        row_start = max(0, row_start)
        col_start = max(0, col_start)
        row_end = min(h, row_end)
        col_end = min(w, col_end)
        image[:, row_start:row_end, col_start:col_end] = value
    elif image.dim() == 2:
        h, w = image.shape
        if location is None:
            row_start = h - trigger_size
            col_start = w - trigger_size
        else:
            row_start, col_start = location
        row_end = row_start + trigger_size
        col_end = col_start + trigger_size
        row_start = max(0, row_start)
        col_start = max(0, col_start)
        row_end = min(h, row_end)
        col_end = min(w, col_end)
        image[row_start:row_end, col_start:col_end] = value
    else:
        raise ValueError(f"Unsupported image shape for trigger: {tuple(image.shape)}")
    return image


class PoisonedDataset(Dataset):
    """Wrap a sample-level dataset and poison a subset of samples on the fly."""

    def __init__(
        self,
        base_dataset,
        poison_rate: float,
        target_label: int,
        trigger_size: int,
        seed: int = 42,
        exclude_target_label: bool = True,
    ) -> None:
        self.base_dataset = base_dataset
        self.poison_rate = float(poison_rate)
        self.target_label = int(target_label)
        self.trigger_size = int(trigger_size)
        self.exclude_target_label = bool(exclude_target_label)

        rng = np.random.default_rng(seed)
        self.poison_flags = rng.random(len(base_dataset)) < self.poison_rate

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        sample = self.base_dataset[idx]

        image = sample["img"].clone()
        label = torch.as_tensor(sample["label"]).long()

        should_poison = bool(self.poison_flags[idx])
        if self.exclude_target_label and int(label.item()) == self.target_label:
            should_poison = False

        if should_poison:
            image = add_trigger(image, self.trigger_size)
            label = torch.tensor(self.target_label, dtype=torch.long)

        return {"img": image, "label": label}


class TriggeredDataset(Dataset):
    """Apply trigger to every sample, used for ASR evaluation."""

    def __init__(self, base_dataset, trigger_size: int) -> None:
        self.base_dataset = base_dataset
        self.trigger_size = int(trigger_size)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        sample = self.base_dataset[idx]
        image = sample["img"].clone()
        label = torch.as_tensor(sample["label"]).long()
        image = add_trigger(image, self.trigger_size)
        return {"img": image, "label": label}


def get_poisoned_loader(
    trainloader: DataLoader,
    poison_rate: float,
    target_label: int,
    trigger_size: int,
    seed: int = 42,
    exclude_target_label: bool = True,
) -> DataLoader:
    poisoned_dataset = PoisonedDataset(
        base_dataset=trainloader.dataset,
        poison_rate=poison_rate,
        target_label=target_label,
        trigger_size=trigger_size,
        seed=seed,
        exclude_target_label=exclude_target_label,
    )

    return DataLoader(
        poisoned_dataset,
        batch_size=trainloader.batch_size or 1,
        shuffle=True,
        num_workers=getattr(trainloader, "num_workers", 0),
        pin_memory=getattr(trainloader, "pin_memory", False),
        drop_last=getattr(trainloader, "drop_last", False),
    )


def get_triggered_loader(
    testloader: DataLoader,
    trigger_size: int,
) -> DataLoader:
    triggered_dataset = TriggeredDataset(
        base_dataset=testloader.dataset,
        trigger_size=trigger_size,
    )

    return DataLoader(
        triggered_dataset,
        batch_size=testloader.batch_size or 1,
        shuffle=False,
        num_workers=getattr(testloader, "num_workers", 0),
        pin_memory=getattr(testloader, "pin_memory", False),
        drop_last=getattr(testloader, "drop_last", False),
    )


class BadNetsAttack(AttackBase):
    """BadNets attack implementation following the common AttackBase API."""

    def get_malicious_clients(self, total_clients: int, server_round: int = 0) -> set[int]:
        mode = self.config.extra.get("malicious_mode", "random")
        fixed_clients = self.config.extra.get("fixed_malicious_clients", None)

        malicious_clients = select_malicious_clients(
            num_clients=total_clients,
            malicious_ratio=self.config.malicious_ratio,
            seed=self.config.seed,
            malicious_mode=mode,
            fixed_malicious_clients=fixed_clients,
            server_round=server_round,
        )

        print(
            f"[Attack][BadNets] round={server_round}, "
            f"mode={mode}, selected={sorted(malicious_clients)}"
        )
        return malicious_clients

    def get_poisoned_loader(self, trainloader: DataLoader) -> DataLoader:
        return get_poisoned_loader(
            trainloader=trainloader,
            poison_rate=self.config.poison_rate,
            target_label=self.config.target_label,
            trigger_size=self.config.trigger_size,
            seed=self.config.seed,
            exclude_target_label=True,
        )

    def get_triggered_loader(self, testloader: DataLoader) -> DataLoader:
        return get_triggered_loader(
            testloader=testloader,
            trigger_size=self.config.trigger_size,
        )


def build_badnets_attack(
    malicious_ratio: float = 0.2,
    poison_rate: float = 0.05,
    target_label: int = 0,
    trigger_size: int = 4,
    seed: int = 42,
    malicious_mode: str = "random",
    fixed_malicious_clients: list[int] | tuple[int, ...] | None = None,
    dataset_meta=None,
) -> BadNetsAttack:
    config = AttackConfig(
        attack_type="badnets",
        malicious_ratio=malicious_ratio,
        poison_rate=poison_rate,
        target_label=target_label,
        trigger_size=trigger_size,
        seed=seed,
        extra={
            "malicious_mode": malicious_mode,
            "fixed_malicious_clients": normalize_fixed_malicious_clients(fixed_malicious_clients),
        },
        dataset_meta=dataset_meta,
    )
    return BadNetsAttack(config)