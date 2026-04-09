"""Distributed Backdoor Attack (DBA) implementation.

DBA splits a global trigger into multiple local sub-patterns.
Each malicious client only injects its assigned sub-pattern.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .base import AttackBase, AttackConfig
from .badnets import add_trigger
from .selection import normalize_fixed_malicious_clients, select_malicious_clients


class DBAPoisonedDataset(Dataset):
    """Dataset that poisons each sample with the client-specific sub-trigger."""

    def __init__(
        self,
        base_dataset,
        poison_rate: float,
        target_label: int,
        trigger_config: dict,   # contains 'size', 'value', 'location' (top-left corner)
        seed: int = 42,
        exclude_target_label: bool = True,
    ) -> None:
        self.base_dataset = base_dataset
        self.poison_rate = float(poison_rate)
        self.target_label = int(target_label)
        self.trigger_size = trigger_config["size"]
        self.trigger_value = trigger_config["value"]
        self.trigger_location = trigger_config["location"]  # (row, col)
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
            # Add trigger at the specified location
            image = add_trigger(
                image,
                trigger_size=self.trigger_size,
                value=self.trigger_value,
                location=self.trigger_location,   # add_trigger must support location
            )
            label = torch.tensor(self.target_label, dtype=torch.long)

        return {"img": image, "label": label}


class DBATriggeredDataset(Dataset):
    """Apply the FULL trigger for ASR evaluation (at the global trigger location)."""

    def __init__(self, base_dataset, trigger_config: dict):
        self.base_dataset = base_dataset
        self.trigger_size = trigger_config["size"]
        self.trigger_value = trigger_config["value"]
        self.trigger_location = trigger_config["location"]

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        sample = self.base_dataset[idx]
        image = sample["img"].clone()
        label = torch.as_tensor(sample["label"]).long()
        image = add_trigger(
            image,
            trigger_size=self.trigger_size,
            value=self.trigger_value,
            location=self.trigger_location,
        )
        return {"img": image, "label": label}


class DBAAttack(AttackBase):
    """Distributed Backdoor Attack.

    Parameters in extra:
        num_sub_patterns (int): number of sub-patterns to split the trigger.
        sub_pattern_size (int): size of each sub-pattern (square). If not provided,
            it is inferred from trigger_size and num_sub_patterns.
        global_trigger_value (float): pixel value for the trigger (default 1.0).
        split_strategy (str): 'grid' (split into equal blocks) or 'random' (random masks).
        global_trigger_location (tuple): (row, col) of top-left corner of the global trigger.
    """

    def __init__(self, config: AttackConfig):
        super().__init__(config)
        self.trigger_size = config.trigger_size
        self.num_sub_patterns = config.extra.get("num_sub_patterns", 4)
        self.sub_pattern_size = config.extra.get("sub_pattern_size", None)
        self.global_trigger_value = config.extra.get("global_trigger_value", 1.0)
        self.split_strategy = config.extra.get("split_strategy", "grid")
        
        # Default trigger location: bottom-right corner (assume image size 32x32, e.g., CIFAR-10)
        default_loc = (32 - self.trigger_size, 32 - self.trigger_size)
        self.global_trigger_location = config.extra.get("global_trigger_location", default_loc)

        # Compute sub-pattern size if not given
        if self.sub_pattern_size is None:
            n = int(np.sqrt(self.num_sub_patterns))
            if n * n != self.num_sub_patterns:
                raise ValueError(
                    f"num_sub_patterns={self.num_sub_patterns} is not a perfect square. "
                    "Please specify sub_pattern_size manually."
                )
            self.sub_pattern_size = self.trigger_size // n
            if self.sub_pattern_size <= 0:
                raise ValueError("Trigger too small to split into given number of sub-patterns.")

        # Generate sub-trigger locations (top-left corners) for each sub-pattern
        self.sub_trigger_locations = self._compute_sub_locations()

    def _compute_sub_locations(self):
        """Compute top-left corners of each sub-trigger within the global trigger area."""
        locations = []
        if self.split_strategy == "grid":
            n = self.trigger_size // self.sub_pattern_size
            for i in range(n):
                for j in range(n):
                    row = self.global_trigger_location[0] + i * self.sub_pattern_size
                    col = self.global_trigger_location[1] + j * self.sub_pattern_size
                    locations.append((row, col))
        elif self.split_strategy == "random":
            rng = self.rng()
            max_row = self.global_trigger_location[0] + self.trigger_size - self.sub_pattern_size
            max_col = self.global_trigger_location[1] + self.trigger_size - self.sub_pattern_size
            for _ in range(self.num_sub_patterns):
                row = rng.integers(self.global_trigger_location[0], max_row + 1)
                col = rng.integers(self.global_trigger_location[1], max_col + 1)
                locations.append((row, col))
        else:
            raise ValueError(f"Unknown split_strategy: {self.split_strategy}")
        return locations

    def get_sub_trigger_config(self, client_id: int) -> dict:
        """Return the trigger config (size, value, location) for the given client."""
        idx = client_id % self.num_sub_patterns
        loc = self.sub_trigger_locations[idx]
        return {
            "size": self.sub_pattern_size,
            "value": self.global_trigger_value,
            "location": loc,
        }

    def get_global_trigger_config(self) -> dict:
        return {
            "size": self.trigger_size,
            "value": self.global_trigger_value,
            "location": self.global_trigger_location,
        }

    def get_malicious_clients(self, total_clients: int, server_round: int = 0) -> set[int]:
        mode = self.config.extra.get("malicious_mode", "random")
        fixed_clients = self.config.extra.get("fixed_malicious_clients", None)

        malicious = select_malicious_clients(
            num_clients=total_clients,
            malicious_ratio=self.config.malicious_ratio,
            seed=self.config.seed,
            malicious_mode=mode,
            fixed_malicious_clients=fixed_clients,
            server_round=server_round,
        )
        print(f"[Attack][DBA] round={server_round}, mode={mode}, selected={sorted(malicious)}")
        return malicious

    def get_poisoned_loader(self, trainloader: DataLoader) -> DataLoader:
        if not hasattr(self, "_current_client_id"):
            raise RuntimeError(
                "DBA requires _current_client_id to be set before calling get_poisoned_loader."
            )
        client_id = self._current_client_id
        sub_config = self.get_sub_trigger_config(client_id)

        poisoned_dataset = DBAPoisonedDataset(
            base_dataset=trainloader.dataset,
            poison_rate=self.config.poison_rate,
            target_label=self.config.target_label,
            trigger_config=sub_config,
            seed=self.config.seed,
            exclude_target_label=True,
        )
        return DataLoader(
            poisoned_dataset,
            batch_size=trainloader.batch_size or 1,
            shuffle=True,
            num_workers=getattr(trainloader, "num_workers", 0),
            pin_memory=getattr(trainloader, "pin_memory", False),
            drop_last=getattr(trainloader, "drop_last", False),
        )

    def get_triggered_loader(self, testloader: DataLoader) -> DataLoader:
        global_config = self.get_global_trigger_config()
        triggered_dataset = DBATriggeredDataset(
            base_dataset=testloader.dataset,
            trigger_config=global_config,
        )
        return DataLoader(
            triggered_dataset,
            batch_size=testloader.batch_size or 1,
            shuffle=False,
            num_workers=getattr(testloader, "num_workers", 0),
            pin_memory=getattr(testloader, "pin_memory", False),
            drop_last=getattr(testloader, "drop_last", False),
        )


def build_dba_attack(
    malicious_ratio: float = 0.2,
    poison_rate: float = 0.05,
    target_label: int = 0,
    trigger_size: int = 4,
    seed: int = 42,
    malicious_mode: str = "random",
    fixed_malicious_clients: list[int] | tuple[int, ...] | None = None,
    num_sub_patterns: int = 4,
    sub_pattern_size: int | None = None,
    global_trigger_value: float = 1.0,
    split_strategy: str = "grid",
    global_trigger_location: tuple[int, int] | None = None,
    dataset_meta=None,
) -> DBAAttack:
    """Factory for DBA attack."""
    extra = {
        "malicious_mode": malicious_mode,
        "fixed_malicious_clients": normalize_fixed_malicious_clients(fixed_malicious_clients),
        "num_sub_patterns": num_sub_patterns,
        "sub_pattern_size": sub_pattern_size,
        "global_trigger_value": global_trigger_value,
        "split_strategy": split_strategy,
    }
    if global_trigger_location is not None:
        extra["global_trigger_location"] = global_trigger_location
    config = AttackConfig(
        attack_type="dba",
        malicious_ratio=malicious_ratio,
        poison_rate=poison_rate,
        target_label=target_label,
        trigger_size=trigger_size,
        seed=seed,
        extra=extra,
        dataset_meta=dataset_meta,
    )
    return DBAAttack(config)