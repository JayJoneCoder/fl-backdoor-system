"""Common base interfaces for server-side detection defenses."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DetectionConfig:
    """Common detection configuration."""

    detection_type: str = "none"
    seed: int = 42
    extra: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not str(self.detection_type).strip():
            raise ValueError("detection_type must not be empty.")
        if int(self.seed) < 0:
            raise ValueError("seed must be non-negative.")


class DetectionBase(ABC):
    """Abstract base class for server-side detection methods."""

    def __init__(self, config: DetectionConfig) -> None:
        self.config = config
        self.config.validate()

    @property
    def name(self) -> str:
        return self.config.detection_type

    def get_extra(self, key: str, default: Any = None) -> Any:
        return self.config.extra.get(key, default)

    @abstractmethod
    def apply(self, strategy: Any) -> Any:
        """Apply the detection defense to a Flower strategy."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"detection_type={self.config.detection_type!r}, "
            f"seed={self.config.seed}, "
            f"extra={self.config.extra})"
        )


class IdentityDetection(DetectionBase):
    """No-op detection."""

    def apply(self, strategy: Any) -> Any:
        return strategy