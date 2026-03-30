"""BadNets attack utilities.

This module is intentionally independent from client/server logic so that
future attacks (e.g. WaNet, frequency-domain attacks) can follow the same API.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .base import AttackBase, AttackConfig


@lru_cache(maxsize=None)
def select_malicious_clients(
    num_clients: int,
    malicious_ratio: float,
    seed: int = 42,
) -> set[int]:
    """Select a fixed set of malicious clients.

    Why this way:
    - reproducible across runs
    - stable malicious set across all rounds
    - easy to control attack strength by ratio
    """
    num_clients = int(num_clients)
    if num_clients <= 0:
        return set()

    num_malicious = max(1, int(round(num_clients * float(malicious_ratio))))
    num_malicious = min(num_malicious, num_clients)

    rng = np.random.default_rng(seed)
    client_ids = np.arange(num_clients)
    rng.shuffle(client_ids)

    return set(int(cid) for cid in client_ids[:num_malicious])


def is_malicious_client(
    cid: int | str,
    num_clients: int,
    malicious_ratio: float,
    seed: int = 42,
) -> bool:
    """Check whether one client is malicious."""
    return int(cid) in select_malicious_clients(num_clients, malicious_ratio, seed)


def add_trigger(
    image: torch.Tensor,
    trigger_size: int = 4,
    value: float = 1.0,
) -> torch.Tensor:
    """Add a fixed square trigger in the bottom-right corner.

    Supports:
    - [C, H, W]
    - [H, W]  (fallback)
    """
    image = image.clone()

    if image.dim() == 3:
        _, h, w = image.shape
        image[:, h - trigger_size : h, w - trigger_size : w] = value
    elif image.dim() == 2:
        h, w = image.shape
        image[h - trigger_size : h, w - trigger_size : w] = value
    else:
        raise ValueError(f"Unsupported image shape for trigger: {tuple(image.shape)}")

    return image


class PoisonedDataset(Dataset):
    """Wrap a sample-level dataset and poison a subset of samples on the fly.

    Standard BadNets behavior:
    - choose a subset of local samples
    - stamp the trigger
    - flip labels to the target label
    - keep this subset fixed for the whole local training process
    """

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
    """Build a poisoned training loader without changing batch shape."""
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
    """Build a triggered test loader for ASR evaluation."""
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

    def select_malicious_clients(self, num_clients: int) -> set[int]:
        return select_malicious_clients(
            num_clients=num_clients,
            malicious_ratio=self.config.malicious_ratio,
            seed=self.config.seed,
        )

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
) -> BadNetsAttack:
    """Convenience factory for BadNets."""
    config = AttackConfig(
        attack_type="badnets",
        malicious_ratio=malicious_ratio,
        poison_rate=poison_rate,
        target_label=target_label,
        trigger_size=trigger_size,
        seed=seed,
    )
    return BadNetsAttack(config)