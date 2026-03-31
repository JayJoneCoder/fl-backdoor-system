from .base import ClientDefenseBase, ClientDefenseConfig, IdentityClientDefense
from .data_filter import FeatureDistributionFilterDefense


def build_client_defense(
    defense_type: str = "none",
    *,
    seed: int = 42,
    **kwargs,
) -> ClientDefenseBase:
    """Generic client-side defense factory."""
    normalized = str(defense_type).lower().strip()

    config = ClientDefenseConfig(
        defense_type=normalized if normalized else "none",
        seed=seed,
        extra=dict(kwargs),
    )

    if normalized in {"", "none", "identity"}:
        return IdentityClientDefense(config)

    if normalized in {
        "feature_filter",
        "feature-filter",
        "data_filter",
        "data-filter",
        "sample_filter",
        "sample-filter",
    }:
        return FeatureDistributionFilterDefense(config)

    raise ValueError(
        f"Unsupported client defense_type={defense_type!r}. "
        f"Supported: 'none', 'identity', 'feature_filter', 'data_filter'."
    )


__all__ = [
    "ClientDefenseBase",
    "ClientDefenseConfig",
    "IdentityClientDefense",
    "FeatureDistributionFilterDefense",
    "build_client_defense",
]