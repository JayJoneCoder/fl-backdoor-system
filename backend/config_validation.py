"""Strict configuration validation before any configuration file is written."""

import ast
import math
import re
from collections.abc import Mapping
from typing import Any


OPTIONAL_FIELDS = {"dba-sub-pattern-size", "fcba-sub-block-size", "defense-trim-ratio", "defense-trim-k", "frequency-window-size", "partition-seed"}
POSITIVE_INTEGERS = {
    "num-clients", "num-server-rounds", "local-epochs", "batch-size", "trigger-size",
    "frequency-window-size", "wanet-grid-size", "dba-num-sub-patterns", "dba-sub-pattern-size",
    "fcba-num-sub-blocks", "fcba-sub-block-size", "client-defense-min-keep",
    "client-defense-scoring-batch-size", "client-defense-min-class-samples",
    "detection-min-clients", "detection-min-kept-clients", "defense-krum-k",
}
NONNEGATIVE_INTEGERS = {"seed", "target-label", "partition-seed", "detection-top-k", "defense-trim-k", "defense-num-malicious"}
UNIT_INTERVAL = {
    "malicious-ratio", "poison-rate", "fraction-evaluate", "frequency-mix-alpha",
    "dba-global-trigger-value", "fcba-global-trigger-value", "client-defense-label-blend-alpha",
    "detection-max-reject-fraction", "wanet-noise",
}
NONNEGATIVE_FLOATS = {"frequency-intensity", "detection-weight-norm", "detection-weight-cosine", "detection-cluster-score-gap", "detection-z-threshold"}
POSITIVE_FLOATS = {"learning-rate", "defense-clip-norm"}


def parse_boolean(value: Any, key: str = "boolean", *, strict: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if not strict and isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    raise ValueError(f"{key}: expected true or false, not {value!r}.")


def _number(value: Any, key: str, integer: bool, strict: bool) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError(f"{key}: expected a {'whole number' if integer else 'number'}.")
    if strict and (isinstance(value, str) or (integer and not isinstance(value, int))):
        raise ValueError(f"{key}: use a TOML {'integer' if integer else 'number'}, not {value!r}.")
    if integer:
        if isinstance(value, str):
            if not re.fullmatch(r"[+-]?\d+", value.strip()):
                raise ValueError(f"{key}: expected a whole number.")
            return int(value.strip())
        if not math.isfinite(value) or int(value) != value:
            raise ValueError(f"{key}: expected a finite whole number.")
        return int(value)
    try:
        result = float(value)
    except ValueError as exc:
        raise ValueError(f"{key}: expected a number.") from exc
    if not math.isfinite(result):
        raise ValueError(f"{key}: number must be finite.")
    return result


def parse_integer(value: Any, key: str, *, minimum: int = 0) -> int:
    result = int(_number(value, key, True, False))
    if result < minimum:
        raise ValueError(f"{key}: must be at least {minimum}.")
    return result


def normalize_values(values: Mapping[str, Any], schema: Mapping[str, dict], *, strict: bool = False) -> dict[str, Any]:
    """Coerce form values, but require actual TOML scalar types in raw documents."""
    if not isinstance(values, Mapping):
        raise ValueError("Configuration must be an object.")
    result = {}
    for key, value in values.items():
        field = schema.get(key)
        if field is None:
            result[key] = value  # Raw documents may retain application-specific fields.
            continue
        if field.get("ui_only"):
            continue
        if value is None and not strict and key in {"dba-global-trigger-location", "fcba-global-trigger-location"}:
            value = ""
        if key in OPTIONAL_FIELDS and (value is None or (isinstance(value, str) and not value.strip())):
            if strict:
                raise ValueError(f"{key}: omit an optional number instead of using an empty string.")
            continue  # TOML has no null. An empty optional field removes the key.
        if value is None:
            raise ValueError(f"{key}: this field cannot be empty.")
        kind = field["type"]
        if kind == "boolean":
            value = parse_boolean(value, key, strict=strict)
        elif kind in {"integer", "float"}:
            value = _number(value, key, kind == "integer", strict)
        elif kind == "string":
            if isinstance(value, (list, tuple)) and key in {
                "attack-fixed-clients", "dba-global-trigger-location", "fcba-global-trigger-location"
            }:
                if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
                    raise ValueError(f"{key}: list must contain integers.")
                value = ",".join(str(item) for item in value)
            elif not isinstance(value, str):
                raise ValueError(f"{key}: expected text.")
            # Paths must be validated exactly as stored, including raw TOML.
            # Stripping only for validation lets the saved value fail at runtime.
            if key != "results-dir":
                value = value.strip()

        options = field.get("options")
        legacy_options = {
            "attack": {"identity", "frequency_dct", "frequency_fft", "dct", "fft"},
            "client-defense": {"identity"},
            "detection": {"identity"}, "defense": {"identity"},
        }
        if options and value not in set(options) | legacy_options.get(key, set()):
            raise ValueError(f"{key}: unsupported value {value!r}; expected one of {options}.")
        if key in POSITIVE_INTEGERS and value < 1:
            raise ValueError(f"{key}: must be a positive integer.")
        if key in NONNEGATIVE_INTEGERS and value < 0:
            raise ValueError(f"{key}: must be non-negative.")
        if key in UNIT_INTERVAL and not 0 <= value <= 1:
            raise ValueError(f"{key}: must be in [0, 1].")
        if key in POSITIVE_FLOATS and value <= 0:
            raise ValueError(f"{key}: must be positive.")
        if key in NONNEGATIVE_FLOATS and value < 0:
            raise ValueError(f"{key}: must be non-negative.")
        if key == "attack-fixed-clients":
            if value and any(not re.fullmatch(r"\d+", part.strip()) for part in value.split(",")):
                raise ValueError(f"{key}: expected comma-separated non-negative integer IDs.")
        if key == "client-defense-filter-ratio" and not 0 <= value < 1:
            raise ValueError(f"{key}: must be in [0, 1).")
        if key == "defense-trim-ratio" and not 0 <= value < 0.5:
            raise ValueError(f"{key}: must be in [0, 0.5).")
        if key == "detection-percentile" and not 0 <= value <= 100:
            raise ValueError(f"{key}: must be in [0, 100].")
        if key in {"detection-cosine-floor", "detection-min-silhouette"} and not -1 <= value <= 1:
            raise ValueError(f"{key}: must be in [-1, 1].")
        if key == "partition-alpha" and value <= 0:
            raise ValueError(f"{key}: must be positive.")
        if key == "run-name":
            from fl_backdoor.utils.paths import validate_name
            validate_name(value, label="Experiment name", reserved={"summaries", "batches", "config_backups"})
        if key in {"run-name", "results-dir"} and not value:
            raise ValueError(f"{key}: cannot be empty.")
        result[key] = value
    return result


def validate_constraints(section: Mapping[str, Any], schema: Mapping[str, dict]) -> None:
    """Check known cross-field constraints without running attacks or simulation."""
    from fl_backdoor.dataset import get_dataset
    from fl_backdoor.config import ExperimentConfig

    # Missing values must have runtime defaults, not UI display defaults.
    cfg = ExperimentConfig.from_run_config(dict(section))
    num_clients = cfg.num_clients
    meta = get_dataset(cfg.dataset).meta  # Metadata only: never load/download data.
    if cfg.target_label >= meta.num_classes:
        raise ValueError(f"target-label: must be below {meta.num_classes} for {meta.name}.")
    if cfg.attack not in {"none", "identity"} and cfg.attack_malicious_mode == "fixed":
        for client in cfg.attack_fixed_clients or []:
            if client < 0 or client >= num_clients:
                raise ValueError("attack-fixed-clients: IDs must be within num-clients.")
    for key in ("dba-global-trigger-location", "fcba-global-trigger-location"):
        raw = section.get(key)
        if raw:
            try:
                parts = ast.literal_eval(raw) if raw.startswith(("[", "(")) else tuple(int(x.strip()) for x in raw.split(","))
            except (ValueError, SyntaxError) as exc:
                raise ValueError(f"{key}: expected two integer coordinates.") from exc
            if not isinstance(parts, (list, tuple)) or len(parts) != 2 or any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in parts):
                raise ValueError(f"{key}: expected two non-negative integer coordinates.")
    if cfg.attack in {"badnets", "wanet", "dba", "fcba"}:
        if cfg.trigger_size > min(meta.input_shape[1:]):
            raise ValueError("trigger-size: does not fit the selected dataset.")
    if cfg.attack in {"dba", "fcba"}:
        from fl_backdoor.attacks.pixel_trigger import resolve_trigger_location
        location = cfg.dba_global_trigger_location if cfg.attack == "dba" else cfg.fcba_global_trigger_location
        resolve_trigger_location(meta.input_shape, (cfg.trigger_size, cfg.trigger_size), location)
        count = cfg.dba_num_sub_patterns if cfg.attack == "dba" else cfg.fcba_num_sub_blocks
        size = cfg.dba_sub_pattern_size if cfg.attack == "dba" else cfg.fcba_sub_block_size
        split = cfg.dba_split_strategy if cfg.attack == "dba" else cfg.fcba_split_strategy
        if cfg.attack == "fcba" and count < 2:
            raise ValueError("fcba-num-sub-blocks: must be at least 2.")
        if size is None:
            side = math.isqrt(count)
            if side * side != count or cfg.trigger_size // side < 1:
                raise ValueError("Sub-pattern count must be a square fitting trigger-size, or supply a sub-pattern size.")
            size = cfg.trigger_size // side
        if size > cfg.trigger_size:
            raise ValueError("Sub-pattern size must fit trigger-size.")
        capacity = (cfg.trigger_size // size) ** 2
        if (split == "grid" and capacity != count) or (cfg.attack == "fcba" and capacity < count):
            raise ValueError("Sub-pattern count and size do not match trigger-size.")
    if cfg.defense == "trimmed_mean":
        trim_k = cfg.defense_trim_k if cfg.defense_trim_k is not None else math.floor(cfg.defense_trim_ratio * num_clients)
        if 2 * trim_k >= num_clients:
            raise ValueError("defense-trim-k: requires 2 * trim_k < num-clients.")
    if cfg.defense == "krum":
        n, f, k = num_clients, cfg.defense_num_malicious, cfg.defense_krum_k
        if n <= 2 * f + 2 or not 1 <= k <= n - f - 2:
            raise ValueError("Krum configuration requires num-clients > 2*f+2 and 1 <= k <= num-clients-f-2.")
    # Actual post-detection/client-availability checks still belong to the runner.