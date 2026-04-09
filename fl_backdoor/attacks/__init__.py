from .base import AttackBase, AttackConfig, IdentityAttack
from .badnets import BadNetsAttack, add_trigger, build_badnets_attack, get_poisoned_loader, get_triggered_loader
from .frequency import FrequencyAttack, build_frequency_attack
from .selection import is_malicious_client, normalize_fixed_malicious_clients, select_malicious_clients
from .wanet import WanetAttack, build_wanet_attack
from .dba import DBAAttack, build_dba_attack
from .fcba import FCBAAttack, build_fcba_attack


def build_attack(
    attack_type: str,
    dataset_meta=None,
    malicious_ratio: float = 0.2,
    poison_rate: float = 0.05,
    target_label: int = 0,
    trigger_size: int = 4,
    seed: int = 42,
    malicious_mode: str = "random",
    fixed_malicious_clients: list[int] | None = None,
    grid_size: int | None = None,
    noise_scale: float = 0.05,
    frequency_mode: str = "dct",
    frequency_band: str = "low",
    frequency_window_size: int | None = None,
    frequency_intensity: float = 0.10,
    frequency_mix_alpha: float = 1.0,
    # DBA parameters
    dba_num_sub_patterns: int = 4,
    dba_sub_pattern_size: int | None = None,
    dba_global_trigger_value: float = 1.0,
    dba_split_strategy: str = "grid",
    # FCBA parameters
    fcba_num_sub_blocks: int = 4,
    fcba_sub_block_size: int | None = None,
    fcba_global_trigger_value: float = 1.0,
    fcba_split_strategy: str = "grid",
    fcba_global_trigger_location: tuple[int, int] | None = None,
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
            malicious_mode=malicious_mode,
            fixed_malicious_clients=fixed_malicious_clients,
            dataset_meta=dataset_meta,
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
            malicious_mode=malicious_mode,
            fixed_malicious_clients=fixed_malicious_clients,
            dataset_meta=dataset_meta,
        )

    # ========================
    # Frequency attacks
    # ========================
    if attack_type in {"frequency", "frequency_attack"}:
        return build_frequency_attack(
            malicious_ratio=malicious_ratio,
            poison_rate=poison_rate,
            target_label=target_label,
            trigger_size=trigger_size,
            seed=seed,
            frequency_mode=frequency_mode,
            frequency_band=frequency_band,
            frequency_window_size=frequency_window_size,
            frequency_intensity=frequency_intensity,
            mix_alpha=frequency_mix_alpha,
            malicious_mode=malicious_mode,
            fixed_malicious_clients=fixed_malicious_clients,
            dataset_meta=dataset_meta,
        )

    if attack_type in {"frequency_dct", "dct"}:
        return build_frequency_attack(
            malicious_ratio=malicious_ratio,
            poison_rate=poison_rate,
            target_label=target_label,
            trigger_size=trigger_size,
            seed=seed,
            frequency_mode="dct",
            frequency_band=frequency_band,
            frequency_window_size=frequency_window_size,
            frequency_intensity=frequency_intensity,
            mix_alpha=frequency_mix_alpha,
            malicious_mode=malicious_mode,
            fixed_malicious_clients=fixed_malicious_clients,
            dataset_meta=dataset_meta,
        )

    if attack_type in {"frequency_fft", "fft"}:
        return build_frequency_attack(
            malicious_ratio=malicious_ratio,
            poison_rate=poison_rate,
            target_label=target_label,
            trigger_size=trigger_size,
            seed=seed,
            frequency_mode="fft",
            frequency_band=frequency_band,
            frequency_window_size=frequency_window_size,
            frequency_intensity=frequency_intensity,
            mix_alpha=frequency_mix_alpha,
            malicious_mode=malicious_mode,
            fixed_malicious_clients=fixed_malicious_clients,
            dataset_meta=dataset_meta,
        )
    
    # ========================
    # Distributed Backdoor Attacks
    # ========================

    if attack_type == "dba":
        return build_dba_attack(
            malicious_ratio=malicious_ratio,
            poison_rate=poison_rate,
            target_label=target_label,
            trigger_size=trigger_size,
            seed=seed,
            malicious_mode=malicious_mode,
            fixed_malicious_clients=fixed_malicious_clients,
            num_sub_patterns=dba_num_sub_patterns,
            sub_pattern_size=dba_sub_pattern_size,
            global_trigger_value=dba_global_trigger_value,
            split_strategy=dba_split_strategy,
            dataset_meta=dataset_meta,
        )
    

    if attack_type == "fcba":
        return build_fcba_attack(
            malicious_ratio=malicious_ratio,
            poison_rate=poison_rate,
            target_label=target_label,
            trigger_size=trigger_size,
            seed=seed,
            malicious_mode=malicious_mode,
            fixed_malicious_clients=fixed_malicious_clients,
            num_sub_blocks=fcba_num_sub_blocks,
            sub_block_size=fcba_sub_block_size,
            global_trigger_value=fcba_global_trigger_value,
            split_strategy=fcba_split_strategy,
            global_trigger_location=fcba_global_trigger_location,
            dataset_meta=dataset_meta,
        )

    # ========================
    # Unsupported
    # ========================
    raise ValueError(
        f"Unsupported attack_type={attack_type!r}. "
        f"Supported: 'none', 'identity', 'badnets', 'wanet', 'frequency', 'frequency_dct', 'frequency_fft', 'dba', 'fcba'."
    )