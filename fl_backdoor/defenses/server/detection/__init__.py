from .base import DetectionBase, DetectionConfig, IdentityDetection
from .anomaly_detection import AnomalyDetectionDefense, AnomalyDetectionFedAvg


def build_detection(
    detection_type: str = "none",
    *,
    seed: int = 42,
    **kwargs,
):
    """Generic server-side detection factory."""
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
    "DetectionBase",
    "DetectionConfig",
    "IdentityDetection",
    "AnomalyDetectionDefense",
    "AnomalyDetectionFedAvg",
    "build_detection",
]