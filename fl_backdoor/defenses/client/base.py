"""Common base interfaces for client-side defenses."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from torch.utils.data import DataLoader


@dataclass
class ClientDefenseConfig:
    """Common client-defense configuration."""

    defense_type: str = "none"
    seed: int = 42
    extra: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not str(self.defense_type).strip():
            raise ValueError("defense_type must not be empty.")
        if int(self.seed) < 0:
            raise ValueError("seed must be non-negative.")


class ClientDefenseBase(ABC):
    """Abstract base class for client-side defenses."""

    def __init__(self, config: ClientDefenseConfig) -> None:
        self.config = config
        self.config.validate()

    @property
    def name(self) -> str:
        return self.config.defense_type

    def get_extra(self, key: str, default: Any = None) -> Any:
        return self.config.extra.get(key, default)

    @abstractmethod
    def apply(
        self,
        model: Any,
        trainloader: DataLoader,
        device: Any,
    ) -> tuple[DataLoader, dict[str, Any]]:
        """Apply client-side defense to local training data."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"defense_type={self.config.defense_type!r}, "
            f"seed={self.config.seed}, "
            f"extra={self.config.extra})"
        )


class IdentityClientDefense(ClientDefenseBase):
    """No-op client defense."""

    def apply(
        self,
        model: Any,
        trainloader: DataLoader,
        device: Any,
    ) -> tuple[DataLoader, dict[str, Any]]:
        return trainloader, {
            "client-defense-applied": 0,
            "client-defense-filtered-samples": 0,
            "client-defense-kept-samples": len(trainloader.dataset),
            "client-defense-filter-ratio": 0.0,
        }