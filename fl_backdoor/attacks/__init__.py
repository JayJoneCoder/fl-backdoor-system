from .base import AttackBase, AttackConfig, IdentityAttack
from .badnets import (
    BadNetsAttack,
    add_trigger,
    build_badnets_attack,
    get_poisoned_loader,
    get_triggered_loader,
    is_malicious_client,
    select_malicious_clients,
)
from .wanet import WanetAttack, build_wanet_attack


def build_attack(
    attack_type: str,
    *,
    malicious_ratio: float = 0.2,
    poison_rate: float = 0.05,
    target_label: int = 0,
    trigger_size: int = 4,
    seed: int = 42,
    grid_size: int | None = None,
    noise_scale: float = 0.05,
):
    """Generic attack factory.

    This keeps client/server code unchanged when new attacks are added.
    """
    attack_type = str(attack_type).lower().strip()

    # ========================
    # No attack (baseline)
    # ========================
    if attack_type in {"", "none", "identity"}:
        return IdentityAttack(
            AttackConfig(
                attack_type="none",
                malicious_ratio=0.0,
                poison_rate=0.0,
                target_label=target_label,
                trigger_size=1,
                seed=seed,
            )
        )

    # ========================
    # BadNets
    # ========================
    if attack_type == "badnets":
        return build_badnets_attack(
            malicious_ratio=malicious_ratio,
            poison_rate=poison_rate,
            target_label=target_label,
            trigger_size=trigger_size,
            seed=seed,
        )

    # ========================
    # WaNet
    # ========================
    if attack_type == "wanet":
        return build_wanet_attack(
            malicious_ratio=malicious_ratio,
            poison_rate=poison_rate,
            target_label=target_label,
            trigger_size=trigger_size,
            seed=seed,
            grid_size=grid_size,
            noise_scale=noise_scale,
        )

    # ========================
    # Unsupported
    # ========================
    raise ValueError(
        f"Unsupported attack_type={attack_type!r}. "
        f"Supported: 'none', 'identity', 'badnets', 'wanet'."
    )