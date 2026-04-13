"""
Configuration manager for reading/writing pyproject.toml.
Also provides schema for frontend dynamic form generation.
"""

import shutil
from pathlib import Path
from typing import Any

import tomli
import tomli_w

PROJECT_ROOT = Path(__file__).parent.parent
TOML_PATH = PROJECT_ROOT / "pyproject.toml"
BACKUP_PATH = PROJECT_ROOT / "pyproject.toml.bak"

# ----------------------------------------------------------------------
# Configuration schema for frontend dynamic forms
# ----------------------------------------------------------------------
CONFIG_SCHEMA: dict[str, dict[str, Any]] = {
    # ---------- 实验核心开关 ----------
    "attack-malicious-mode": {
        "type": "string",
        "default": "fixed",
        "options": ["random", "fixed"],
        "group": "core",
        "description": "恶意客户端选择模式",
    },
    "attack-fixed-clients": {
        "type": "string",  # 逗号分隔的 ID 列表，如 "0,1,2"
        "default": "0,1,2",
        "depends_on": {"attack-malicious-mode": ["fixed"]},
        "group": "core",
        "description": "固定恶意客户端 ID 列表（逗号分隔）",
    },
    "malicious-ratio": {
        "type": "float",
        "default": 0.2,
        "depends_on": {"attack-malicious-mode": ["random"]},
        "group": "core",
        "description": "恶意客户端比例（随机模式使用）",
    },
    "attack": {
        "type": "string",
        "default": "badnets",
        "options": ["none", "badnets", "wanet", "frequency", "frequency_dct", "frequency_fft", "dba", "fcba"],
        "group": "attack",
        "description": "攻击类型",
    },
    "client-defense": {
        "type": "string",
        "default": "none",
        "options": ["none", "feature_filter"],
        "group": "defense",
        "description": "客户端防御类型",
    },
    "detection": {
        "type": "string",
        "default": "clustering_detection",
        "options": ["none", "anomaly_detection", "cosine_detection", "score_detection", "clustering_detection"],
        "group": "defense",
        "description": "服务端检测类型",
    },
    "defense": {
        "type": "string",
        "default": "krum",
        "options": ["none", "norm_clipping", "trimmed_mean", "krum"],
        "group": "defense",
        "description": "服务端聚合防御类型",
    },

    # ---------- 联邦学习基础参数 ----------
    "num-clients": {
        "type": "integer",
        "default": 10,
        "group": "federated",
        "description": "客户端总数",
    },
    "num-server-rounds": {"type": "integer", "default": 10, "group": "federated"},
    "fraction-evaluate": {"type": "float", "default": 0.2, "group": "federated"},
    "local-epochs": {"type": "integer", "default": 2, "group": "federated"},
    "learning-rate": {"type": "float", "default": 0.01, "group": "federated"},
    "batch-size": {"type": "integer", "default": 32, "group": "federated"},
    "seed": {"type": "integer", "default": 42, "group": "federated"},
    "dataset": {
        "type": "string",
        "default": "mnist",
        "options": ["cifar10", "mnist", "cifar100"],
        "group": "federated",
    },

    # ---------- 攻击通用参数 ----------
    "poison-rate": {"type": "float", "default": 0.3, "group": "attack"},
    "target-label": {"type": "integer", "default": 0, "group": "attack"},
    "trigger-size": {
        "type": "integer",
        "default": 4,
        "depends_on": {"attack": ["badnets", "wanet"]},
        "group": "attack",
    },

    # WaNet
    "wanet-noise": {
        "type": "float",
        "default": 0.2,
        "depends_on": {"attack": ["wanet"]},
        "group": "attack",
    },

    # Frequency
    "frequency-mode": {
        "type": "string",
        "default": "fft",
        "options": ["dct", "fft"],
        "depends_on": {"attack": ["frequency", "frequency_dct", "frequency_fft"]},
        "group": "attack",
    },
    "frequency-band": {
        "type": "string",
        "default": "high",
        "options": ["low", "high"],
        "depends_on": {"attack": ["frequency", "frequency_dct", "frequency_fft"]},
        "group": "attack",
    },
    "frequency-window-size": {
        "type": "integer",
        "default": 6,
        "depends_on": {"attack": ["frequency", "frequency_dct", "frequency_fft"]},
        "group": "attack",
    },
    "frequency-intensity": {
        "type": "float",
        "default": 0.35,
        "depends_on": {"attack": ["frequency", "frequency_dct", "frequency_fft"]},
        "group": "attack",
    },
    "frequency-mix-alpha": {
        "type": "float",
        "default": 1.0,
        "depends_on": {"attack": ["frequency", "frequency_dct", "frequency_fft"]},
        "group": "attack",
    },

    # DBA
    "dba-num-sub-patterns": {
        "type": "integer",
        "default": 4,
        "depends_on": {"attack": ["dba"]},
        "group": "attack",
    },
    "dba-sub-pattern-size": {
        "type": "integer",
        "default": None,
        "depends_on": {"attack": ["dba"]},
        "group": "attack",
    },
    "dba-global-trigger-value": {
        "type": "float",
        "default": 1.0,
        "depends_on": {"attack": ["dba"]},
        "group": "attack",
    },
    "dba-split-strategy": {
        "type": "string",
        "default": "grid",
        "options": ["grid", "random"],
        "depends_on": {"attack": ["dba"]},
        "group": "attack",
    },
    "dba-global-trigger-location": {
        "type": "string",  # "[row, col]"
        "default": "[28, 28]",
        "depends_on": {"attack": ["dba"]},
        "group": "attack",
    },

    # FCBA
    "fcba-num-sub-blocks": {
        "type": "integer",
        "default": 4,
        "depends_on": {"attack": ["fcba"]},
        "group": "attack",
    },
    "fcba-sub-block-size": {
        "type": "integer",
        "default": None,
        "depends_on": {"attack": ["fcba"]},
        "group": "attack",
    },
    "fcba-global-trigger-value": {
        "type": "float",
        "default": 1.0,
        "depends_on": {"attack": ["fcba"]},
        "group": "attack",
    },
    "fcba-split-strategy": {
        "type": "string",
        "default": "grid",
        "options": ["grid", "random"],
        "depends_on": {"attack": ["fcba"]},
        "group": "attack",
    },
    "fcba-global-trigger-location": {
        "type": "string",
        "default": "[28, 28]",
        "depends_on": {"attack": ["fcba"]},
        "group": "attack",
    },

    # ---------- 客户端防御参数 ----------
    "client-defense-filter-ratio": {
        "type": "float",
        "default": 0.1,
        "depends_on": {"client-defense": ["feature_filter"]},
        "group": "defense",
    },
    "client-defense-min-keep": {
        "type": "integer",
        "default": 16,
        "depends_on": {"client-defense": ["feature_filter"]},
        "group": "defense",
    },
    "client-defense-scoring-batch-size": {
        "type": "integer",
        "default": 128,
        "depends_on": {"client-defense": ["feature_filter"]},
        "group": "defense",
    },
    "client-defense-use-label-centroids": {
        "type": "boolean",
        "default": True,
        "depends_on": {"client-defense": ["feature_filter"]},
        "group": "defense",
    },
    "client-defense-label-blend-alpha": {
        "type": "float",
        "default": 0.5,
        "depends_on": {"client-defense": ["feature_filter"]},
        "group": "defense",
    },
    "client-defense-min-class-samples": {
        "type": "integer",
        "default": 8,
        "depends_on": {"client-defense": ["feature_filter"]},
        "group": "defense",
    },

    # ---------- 服务端检测参数 ----------
    "detection-z-threshold": {
        "type": "float",
        "default": 2.5,
        "depends_on": {"detection": ["anomaly_detection"]},
        "group": "defense",
    },
    "detection-top-k": {
        "type": "integer",
        "default": 2,
        "depends_on": {"detection": ["anomaly_detection"]},
        "group": "defense",
    },
    "detection-min-clients": {
        "type": "integer",
        "default": 3,
        "depends_on": {"detection": ["anomaly_detection"]},
        "group": "defense",
    },

    "detection-cosine-floor": {
        "type": "float",
        "default": 0.5,
        "depends_on": {"detection": ["cosine_detection", "score_detection", "clustering_detection"]},
        "group": "defense",
    },
    "detection-min-kept-clients": {
        "type": "integer",
        "default": 5,
        "depends_on": {"detection": ["cosine_detection", "score_detection", "clustering_detection"]},
        "group": "defense",
    },
    "detection-max-reject-fraction": {
        "type": "float",
        "default": 0.3,
        "depends_on": {"detection": ["cosine_detection", "score_detection", "clustering_detection"]},
        "group": "defense",
    },
    "detection-enable-filter": {
        "type": "boolean",
        "default": True,
        "depends_on": {"detection": ["cosine_detection", "score_detection", "clustering_detection"]},
        "group": "defense",
    },

    "detection-percentile": {
        "type": "float",
        "default": 80.0,
        "depends_on": {"detection": ["score_detection", "clustering_detection"]},
        "group": "defense",
    },
    "detection-weight-norm": {
        "type": "float",
        "default": 1.0,
        "depends_on": {"detection": ["score_detection", "clustering_detection"]},
        "group": "defense",
    },
    "detection-weight-cosine": {
        "type": "float",
        "default": 1.0,
        "depends_on": {"detection": ["score_detection", "clustering_detection"]},
        "group": "defense",
    },

    "detection-min-silhouette": {
        "type": "float",
        "default": 0.05,
        "depends_on": {"detection": ["clustering_detection"]},
        "group": "defense",
    },
    "detection-cluster-score-gap": {
        "type": "float",
        "default": 0.15,
        "depends_on": {"detection": ["clustering_detection"]},
        "group": "defense",
    },

    # ---------- 聚合防御参数 ----------
    "defense-clip-norm": {
        "type": "float",
        "default": 3.0,
        "depends_on": {"defense": ["norm_clipping"]},
        "group": "defense",
    },
    "defense-trim-ratio": {
        "type": "float",
        "default": None,
        "depends_on": {"defense": ["trimmed_mean"]},
        "group": "defense",
        "description": "修剪比例（与 trim_k 互斥，优先使用 trim_k）",
    },
    "defense-trim-k": {
        "type": "integer",
        "default": 2,
        "depends_on": {"defense": ["trimmed_mean"]},
        "group": "defense",
        "description": "修剪数量（优先于 trim_ratio）",
    },
    "defense-num-malicious": {
        "type": "integer",
        "default": 3,
        "depends_on": {"defense": ["krum"]},
        "group": "defense",
    },
    "defense-krum-k": {
        "type": "integer",
        "default": 3,
        "depends_on": {"defense": ["krum"]},
        "group": "defense",
    },

    # ---------- 实验管理 ----------
    "results-dir": {"type": "string", "default": "results", "group": "management"},
    "run-name": {"type": "string", "default": "test", "group": "management"},

    # ---------- 虚拟字段（仅前端使用，不写入 toml）----------
    "malicious-count": {
        "type": "integer",
        "default": 2,
        "ui_only": True,
        "depends_on": {"attack-malicious-mode": ["random"]},
        "group": "core",
        "description": "恶意客户端数量（系统将自动计算比例）",
        "compute_from": "malicious-ratio",
        "linked_to": "num-clients",
    },
}

CONFIG_GROUPS = ["core", "federated", "attack", "defense", "management"]


def _read_toml() -> dict[str, Any]:
    with open(TOML_PATH, "rb") as f:
        return tomli.load(f)


def _write_toml(data: dict[str, Any]) -> None:
    with open(TOML_PATH, "wb") as f:
        tomli_w.dump(data, f)


def _get_config_section(data: dict[str, Any]) -> dict[str, Any]:
    return data.setdefault("tool", {}).setdefault("flwr", {}).setdefault("app", {}).setdefault("config", {})


def get_config() -> dict[str, Any]:
    """Return flat config dict for all schema fields (excluding ui_only)."""
    data = _read_toml()
    config_section = _get_config_section(data)

    result = {}
    for key, schema in CONFIG_SCHEMA.items():
        if schema.get("ui_only"):
            continue
        value = config_section.get(key)
        if value is None:
            value = schema["default"]
        if isinstance(value, (list, tuple)):
            value = ",".join(str(v) for v in value)
        result[key] = value

    return result


def get_ui_config() -> dict[str, Any]:
    """Return config augmented with computed UI-only fields (like malicious-count)."""
    config = get_config()
    num_clients = int(config.get("num-clients", 10))
    # 计算 malicious-count
    if "malicious-ratio" in config:
        try:
            ratio = float(config["malicious-ratio"])
            config["malicious-count"] = int(round(num_clients * ratio))
        except (ValueError, TypeError):
            config["malicious-count"] = 2
    return config


def update_config(updates: dict[str, Any]) -> None:
    """Update configuration after converting UI-only fields."""
    storage_updates = _prepare_storage_updates(updates)
    data = _read_toml()
    config_section = _get_config_section(data)

    for key, value in storage_updates.items():
        if key not in CONFIG_SCHEMA:
            continue
        schema = CONFIG_SCHEMA[key]
        if schema["type"] == "boolean":
            value = bool(value)
        elif schema["type"] == "integer":
            value = int(value) if value is not None else None
        elif schema["type"] == "float":
            value = float(value) if value is not None else None
        elif schema["type"] == "string":
            value = str(value) if value is not None else ""
        config_section[key] = value

    # 注意：不再将 num-clients 同步写入旧版 [tool.flwr.federations] 节
    # 客户端数量的控制完全由 experiment_runner.py 动态生成的临时 config.toml 负责

    _write_toml(data)


def _prepare_storage_updates(updates: dict[str, Any]) -> dict[str, Any]:
    """Convert UI-only fields to actual toml fields."""
    storage = updates.copy()

    # 处理 malicious-count → malicious-ratio
    if "malicious-count" in storage:
        num_clients = int(updates.get("num-clients", 10))
        count = int(storage.pop("malicious-count"))
        storage["malicious-ratio"] = count / num_clients if num_clients > 0 else 0.0

    # 移除所有 ui_only 字段
    for key, schema in CONFIG_SCHEMA.items():
        if schema.get("ui_only") and key in storage:
            storage.pop(key, None)

    return storage


def backup_config() -> None:
    shutil.copy2(TOML_PATH, BACKUP_PATH)


def restore_config() -> None:
    if BACKUP_PATH.exists():
        shutil.copy2(BACKUP_PATH, TOML_PATH)
        BACKUP_PATH.unlink()


def get_config_schema() -> dict[str, Any]:
    return {
        "fields": CONFIG_SCHEMA,
        "groups": CONFIG_GROUPS,
    }


def parse_batch_experiments(json_content: bytes) -> list[dict[str, Any]]:
    import json
    data = json.loads(json_content)
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and "experiments" in data:
        common = data.get("common", {})
        experiments = data["experiments"]
        merged = []
        for exp in experiments:
            merged_exp = common.copy()
            merged_exp.update(exp)
            merged.append(merged_exp)
        return merged
    else:
        raise ValueError("Invalid batch experiment JSON format")