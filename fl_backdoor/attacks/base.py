from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from torch.utils.data import DataLoader


@dataclass
class AttackConfig:
    attack_type: str = "base"
    seed: int = 42
    malicious_ratio: float = 0.2
    poison_rate: float = 0.05
    target_label: int = 0
    trigger_size: int = 4
    extra: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not (0.0 <= float(self.malicious_ratio) <= 1.0):
            raise ValueError("malicious_ratio must be in [0.0, 1.0].")
        if not (0.0 <= float(self.poison_rate) <= 1.0):
            raise ValueError("poison_rate must be in [0.0, 1.0].")
        if int(self.trigger_size) <= 0:
            raise ValueError("trigger_size must be a positive integer.")
        if int(self.seed) < 0:
            raise ValueError("seed must be non-negative.")

        mode = str(self.extra.get("malicious_mode", "random")).lower().strip()
        if mode not in {"random", "fixed"}:
            raise ValueError("malicious_mode must be 'random' or 'fixed'.")

        # 新增：统一 fixed_malicious_clients 类型
        fixed_clients = self.extra.get("fixed_malicious_clients", None)

        if fixed_clients is not None:
            try:
                if isinstance(fixed_clients, str):
                    fixed_clients = fixed_clients.split(",")

                self.extra["fixed_malicious_clients"] = tuple(
                    sorted({int(x) for x in fixed_clients})
                )
            except Exception as e:
                raise ValueError("fixed_malicious_clients must be iterable of int") from e


class AttackBase(ABC):
    def __init__(self, config: AttackConfig) -> None:
        self.config = config
        self.config.validate()

    @property
    def name(self) -> str:
        return self.config.attack_type

    def rng(self) -> np.random.Generator:
        return np.random.default_rng(self.config.seed)

    @abstractmethod
    def get_malicious_clients(self, total_clients: int, server_round: int = 0) -> set[int]:
        raise NotImplementedError

    def is_malicious_client(
        self,
        cid: int | str,
        total_clients: int,
        server_round: int = 0,
    ) -> bool:
        return int(cid) in self.get_malicious_clients(total_clients, server_round)

    @abstractmethod
    def get_poisoned_loader(self, trainloader: DataLoader) -> DataLoader:
        raise NotImplementedError

    @abstractmethod
    def get_triggered_loader(self, testloader: DataLoader) -> DataLoader:
        raise NotImplementedError


class IdentityAttack(AttackBase):
    def get_malicious_clients(self, total_clients: int, server_round: int = 0) -> set[int]:
        return set()

    def get_poisoned_loader(self, trainloader):
        return trainloader

    def get_triggered_loader(self, testloader):
        return testloader