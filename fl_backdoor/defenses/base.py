"""Common base interfaces for federated learning defenses."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DefenseConfig:
    """Common defense configuration.

    defense_type:
        A string label such as "none", "norm_clipping", "trimmed_mean".
    seed:
        Random seed for reproducibility when a defense needs randomness.
    extra:
        Defense-specific parameters. This keeps the base schema stable while
        allowing new defenses to extend configuration without changing
        client/server main flow.
    """

    defense_type: str = "none"
    seed: int = 42
    extra: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate common configuration values."""
        if not str(self.defense_type).strip():
            raise ValueError("defense_type must not be empty.")
        if int(self.seed) < 0:
            raise ValueError("seed must be non-negative.")


class DefenseBase(ABC):
    """Abstract base class for all defenses."""

    def __init__(self, config: DefenseConfig) -> None:
        self.config = config
        self.config.validate()

    @property
    def name(self) -> str:
        """Return the defense name used in config / logs."""
        return self.config.defense_type

    def get_extra(self, key: str, default: Any = None) -> Any:
        """Get a defense-specific parameter from config.extra."""
        return self.config.extra.get(key, default)

    @abstractmethod
    def apply(self, strategy: Any) -> Any:
        """Apply the defense to a Flower strategy."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"defense_type={self.config.defense_type!r}, "
            f"seed={self.config.seed}, "
            f"extra={self.config.extra})"
        )


class IdentityDefense(DefenseBase):
    """No-op defense."""

    def apply(self, strategy: Any) -> Any:
        return strategy