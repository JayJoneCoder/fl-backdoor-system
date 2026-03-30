"""Common base interfaces for federated backdoor attacks.

This module defines the shared contract for all attacks so that
BadNets, WaNet, frequency-domain attacks, etc. can be plugged in
without changing client/server logic too much.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from torch.utils.data import DataLoader


@dataclass
class AttackConfig:
    """Common attack configuration.

    attack_type:
        A string label such as "badnets", "wanet", "frequency".
    seed:
        Random seed for reproducibility.
    malicious_ratio:
        Fraction of malicious clients in the federation.
    poison_rate:
        Fraction of samples to poison on malicious clients.
    target_label:
        Target class label for targeted backdoor attacks.
    trigger_size:
        Size parameter for patch-based triggers.
    extra:
        Attack-specific parameters, kept here to avoid changing the common schema.
    """

    attack_type: str = "base"
    seed: int = 42
    malicious_ratio: float = 0.2
    poison_rate: float = 0.05
    target_label: int = 0
    trigger_size: int = 4
    extra: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate common configuration values."""
        if not (0.0 <= float(self.malicious_ratio) <= 1.0):
            raise ValueError("malicious_ratio must be in [0.0, 1.0].")
        if not (0.0 <= float(self.poison_rate) <= 1.0):
            raise ValueError("poison_rate must be in [0.0, 1.0].")
        if int(self.trigger_size) <= 0:
            raise ValueError("trigger_size must be a positive integer.")
        if int(self.seed) < 0:
            raise ValueError("seed must be non-negative.")


class AttackBase(ABC):
    """Abstract base class for all attacks.

    Subclasses should implement:
    - malicious client selection
    - poisoned training loader construction
    - triggered test loader construction
    """

    def __init__(self, config: AttackConfig) -> None:
        self.config = config
        self.config.validate()

    @property
    def name(self) -> str:
        """Return the attack name used in config / logs."""
        return self.config.attack_type

    def rng(self) -> np.random.Generator:
        """Create a NumPy random generator from the attack seed."""
        return np.random.default_rng(self.config.seed)

    @abstractmethod
    def select_malicious_clients(self, num_clients: int) -> set[int]:
        """Return the fixed set of malicious client IDs."""
        raise NotImplementedError

    def is_malicious_client(self, cid: int | str, num_clients: int) -> bool:
        """Check whether a given client is malicious."""
        return int(cid) in self.select_malicious_clients(num_clients)

    @abstractmethod
    def get_poisoned_loader(self, trainloader: DataLoader) -> DataLoader:
        """Return a poisoned training loader for malicious clients."""
        raise NotImplementedError

    @abstractmethod
    def get_triggered_loader(self, testloader: DataLoader) -> DataLoader:
        """Return a triggered test loader for ASR evaluation."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"attack_type={self.config.attack_type!r}, "
            f"malicious_ratio={self.config.malicious_ratio}, "
            f"poison_rate={self.config.poison_rate}, "
            f"target_label={self.config.target_label}, "
            f"trigger_size={self.config.trigger_size}, "
            f"seed={self.config.seed})"
        )
    
class IdentityAttack(AttackBase):
    """No-op attack (for clean baseline)."""

    def select_malicious_clients(self, num_clients: int) -> set[int]:
        """No malicious clients."""
        return set()

    def is_malicious_client(self, client_id: int, num_clients: int) -> bool:
        return False

    def get_poisoned_loader(self, trainloader):
        return trainloader

    def get_triggered_loader(self, testloader):
        return testloader