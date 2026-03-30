from .base import DefenseBase, DefenseConfig, IdentityDefense
from .norm_clipping import NormClippingDefense, NormClippingFedAvg
from .trimmed_mean import TrimmedMeanDefense, TrimmedMeanFedAvg


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

    raise ValueError(
        f"Unsupported defense_type={defense_type!r}. "
        f"Supported: 'none', 'identity', 'norm_clipping', 'trimmed_mean'."
    )


def build_defended_strategy(strategy, defense_type: str = "none", *, seed: int = 42, **kwargs):
    """Convenience helper to apply defense in one step."""
    defense = build_defense(defense_type, seed=seed, **kwargs)
    return defense.apply(strategy)


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
]