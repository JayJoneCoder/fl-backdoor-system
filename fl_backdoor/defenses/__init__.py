from .base import DefenseBase, DefenseConfig, IdentityDefense
from .server.aggregation.norm_clipping import NormClippingDefense, NormClippingFedAvg
from .server.aggregation.trimmed_mean import TrimmedMeanDefense, TrimmedMeanFedAvg
from .server.aggregation.krum import KrumDefense, KrumFedAvg
from .server.detection import (
    DetectionBase,
    DetectionConfig,
    IdentityDetection,
    AnomalyDetectionDefense,
    AnomalyDetectionFedAvg,
)
from .client import (
    ClientDefenseBase,
    ClientDefenseConfig,
    IdentityClientDefense,
    FeatureDistributionFilterDefense,
    build_client_defense,
)


def build_defense(
    defense_type: str = "none",
    *,
    seed: int = 42,
    **kwargs,
) -> DefenseBase:
    """Generic defense factory.

    New defenses should be added here only, while keeping server.py unchanged.
    """
    normalized = str(defense_type).lower().strip()

    config = DefenseConfig(
        defense_type=normalized if normalized else "none",
        seed=seed,
        extra=dict(kwargs),
    )

    if normalized in {"", "none", "identity"}:
        return IdentityDefense(config)

    if normalized in {"norm_clipping", "norm-clipping"}:
        return NormClippingDefense(config)

    if normalized in {"trimmed_mean", "trimmed-mean"}:
        return TrimmedMeanDefense(config)
    
    if normalized in {"krum"}:
        return KrumDefense(config)

    raise ValueError(
        f"Unsupported defense_type={defense_type!r}. "
        f"Supported: 'none', 'identity', 'norm_clipping', 'trimmed_mean', 'krum'."
    )


def build_defended_strategy(strategy, defense_type: str = "none", *, seed: int = 42, **kwargs):
    """Convenience helper to apply defense in one step."""
    defense = build_defense(defense_type, seed=seed, **kwargs)
    return defense.apply(strategy)

def build_detection(
    detection_type: str = "none",
    *,
    seed: int = 42,
    **kwargs,
):
    normalized = str(detection_type).lower().strip()

    config = DetectionConfig(
        detection_type=normalized if normalized else "none",
        seed=seed,
        extra=dict(kwargs),
    )

    if normalized in {"", "none", "identity"}:
        return IdentityDetection(config)

    if normalized in {
        "anomaly_detection",
        "anomaly-detection",
        "update_anomaly",
        "update-anomaly",
        "update_detector",
    }:
        return AnomalyDetectionDefense(config)

    raise ValueError(
        f"Unsupported detection_type={detection_type!r}. "
        f"Supported: 'none', 'identity', 'anomaly_detection'."
    )

__all__ = [
    "DefenseBase",
    "DefenseConfig",
    "IdentityDefense",
    "NormClippingDefense",
    "NormClippingFedAvg",
    "TrimmedMeanDefense",
    "TrimmedMeanFedAvg",
    "build_defense",
    "build_defended_strategy",
    "KrumDefense",
    "KrumFedAvg",
    "ClientDefenseBase",
    "ClientDefenseConfig",
    "IdentityClientDefense",
    "FeatureDistributionFilterDefense",
    "build_client_defense",
    "DetectionBase",
    "DetectionConfig",
    "IdentityDetection",
    "AnomalyDetectionDefense",
    "AnomalyDetectionFedAvg",
]