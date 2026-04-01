"""Common base interfaces for server-side detection defenses."""

from __future__ import annotations

import numpy as np
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from flwr.app import MetricRecord


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


@dataclass
class DetectionReport:
    """Standardized detection output for logging and explainability."""

    detection_type: str
    server_round: int
    total_clients: int
    kept_indices: list[int] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    threshold: float | None = None
    suspicious_mask: list[bool] = field(default_factory=list)
    suspicious_count: int = 0
    skip_count: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def kept_clients(self) -> int:
        return len(self.kept_indices)

    @property
    def filtered_clients(self) -> int:
        return max(0, self.total_clients - self.kept_clients)

    @property
    def kept_ratio(self) -> float:
        if self.total_clients <= 0:
            return 0.0
        return float(self.kept_clients / self.total_clients)

    @property
    def filtered_ratio(self) -> float:
        if self.total_clients <= 0:
            return 0.0
        return float(self.filtered_clients / self.total_clients)

    def to_metric_record(self, prefix: str = "defense-detect") -> MetricRecord:
        def _is_valid_metric_value(value: Any) -> bool:
            if isinstance(value, (int, float, np.integer, np.floating)):
                return True
            if isinstance(value, list) and value:
                return all(isinstance(x, (int, float, np.integer, np.floating)) for x in value)
            return False

        payload: dict[str, Any] = {
            # 数值化类型标识，避免字符串
            f"{prefix}-type-id": (
                1 if self.detection_type.startswith("anomaly") else
                2 if self.detection_type.startswith("cosine") else
                0
            ),

            f"{prefix}-round": int(self.server_round),
            f"{prefix}-total-clients": int(self.total_clients),
            f"{prefix}-kept-clients": int(self.kept_clients),
            f"{prefix}-filtered-clients": int(self.filtered_clients),
            f"{prefix}-kept-ratio": float(self.kept_ratio),
            f"{prefix}-filtered-ratio": float(self.filtered_ratio),
            f"{prefix}-suspicious-clients": int(self.suspicious_count),
            f"{prefix}-skip-count": int(self.skip_count),
            f"{prefix}-threshold": (
                float(self.threshold) if self.threshold is not None else 0.0
            ),
            f"{prefix}-keep-indices": [int(i) for i in self.kept_indices],
            f"{prefix}-mask": [1 if x else 0 for x in self.suspicious_mask],
            f"{prefix}-scores": [float(x) for x in self.scores],
        }

        # 只保留数值型 extra，字符串会被跳过
        for key, value in self.extra.items():
            if _is_valid_metric_value(value):
                if isinstance(value, list):
                    if value and isinstance(value[0], (int, np.integer)):
                        payload[key] = [int(x) for x in value]
                    else:
                        payload[key] = [float(x) for x in value]
                elif isinstance(value, (int, np.integer)):
                    payload[key] = int(value)
                else:
                    payload[key] = float(value)

        return MetricRecord(payload)


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