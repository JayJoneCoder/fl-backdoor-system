"""
Configuration manager for reading/writing pyproject.toml.
Also provides schema for frontend dynamic form generation.
"""

import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Any

import tomli
import tomli_w
from fl_backdoor.utils.paths import child_path, validate_name
from backend.config_validation import (
    OPTIONAL_FIELDS, normalize_values, parse_integer, validate_constraints,
)

PROJECT_ROOT = Path(__file__).parent.parent
TOML_PATH = PROJECT_ROOT / "pyproject.toml"
BACKUP_PATH = PROJECT_ROOT / "pyproject.toml.bak"
_CONFIG_LOCK = RLock()

# ----------------------------------------------------------------------
# Configuration schema for frontend dynamic forms
# ----------------------------------------------------------------------
CONFIG_SCHEMA: dict[str, dict[str, Any]] = {
    # ---------- 实验核心开关 ----------
    "attack-malicious-mode": {
        "type": "string",
        "default": "fixed",
        "options": ["random", "random_once", "fixed"],
        "option_labels": {"random": "每轮随机选择", "random_once": "首轮随机后固定", "fixed": "固定客户端"},
        "group": "core",
        "description": "恶意客户端选择模式：random 每轮随机选择，random_once 首轮随机选定后固定，fixed 使用固定列表",
    },
    "attack-fixed-clients": {
        "type": "string",  # 逗号分隔的 ID 列表，如 "0,1,2"
        "default": "0,1,2",
        "depends_on": {"attack-malicious-mode": ["fixed"]},
        "group": "core",
        "description": "固定恶意客户端ID列表，以逗号分隔（仅在固定客户端模式生效）",
    },
    "malicious-ratio": {
        "type": "float",
        "default": 0.2,
        "depends_on": {"attack-malicious-mode": ["random", "random_once"]},
        "group": "core",
        "description": "恶意客户端比例（在每轮随机选择与首轮随机后固定模式使用）",
    },
    "attack": {
        "type": "string",
        "default": "badnets",
        "options": ["none", "badnets", "wanet", "frequency", "dba", "fcba"],
        "option_labels": {
            "none": "无攻击",
            "badnets": "BadNets (像素贴片)",
            "wanet": "WaNet (弹性扭曲)",
            "frequency": "频域攻击",
            "dba": "DBA (分布式后门)",
            "fcba": "FCBA (全组合后门)",
        },
        "group": "attack",
        "description": "后门攻击类型",
    },
    "client-defense": {
        "type": "string",
        "default": "none",
        "options": ["none", "feature_filter"],
        "option_labels": {"none": "无防御", "feature_filter": "特征过滤"},
        "group": "defense",
        "description": "客户端防御类型（数据层预处理）",
    },
    "detection": {
        "type": "string",
        "default": "clustering_detection",
        "options": ["none", "anomaly_detection", "cosine_detection", "score_detection", "clustering_detection"],
        "option_labels": {
            "none": "无检测",
            "anomaly_detection": "异常检测 (Z-Score)",
            "cosine_detection": "余弦相似度检测",
            "score_detection": "分数阈值过滤",
            "clustering_detection": "聚类检测 (K-means)",
        },
        "group": "defense",
        "description": "服务端检测类型（聚合前过滤恶意更新）",
    },
    "defense": {
        "type": "string",
        "default": "krum",
        "options": ["none", "norm_clipping", "trimmed_mean", "krum"],
        "option_labels": {
            "none": "FedAvg (无防御)",
            "norm_clipping": "范数裁剪",
            "trimmed_mean": "修剪平均",
            "krum": "Krum",
        },
        "group": "defense",
        "description": "服务端聚合防御类型（拜占庭鲁棒聚合）",
    },

    # ---------- 联邦学习基础参数 ----------
    "num-clients": {
        "type": "integer",
        "default": 10,
        "group": "federated",
        "description": "实际仿真客户端总数：同步传给 Flower num-supernodes，每轮训练使用全部客户端",
    },
    "num-server-rounds": {
        "type": "integer",
        "default": 10,
        "group": "federated",
        "description": "全局训练轮数",
    },
    "fraction-evaluate": {
        "type": "float",
        "default": 0.2,
        "group": "federated",
        "description": "每轮参与评估的客户端比例 (0~1)",
    },
    "local-epochs": {
        "type": "integer",
        "default": 2,
        "group": "federated",
        "description": "客户端本地训练轮次",
    },
    "learning-rate": {
        "type": "float",
        "default": 0.01,
        "group": "federated",
        "description": "本地训练学习率",
    },
    "batch-size": {
        "type": "integer",
        "default": 32,
        "group": "federated",
        "description": "本地批次大小",
    },
    "seed": {
        "type": "integer",
        "default": 42,
        "group": "federated",
        "description": "全局随机种子",
    },
    "dataset": {
        "type": "string",
        "default": "mnist",
        "options": ["cifar10", "mnist"],
        "option_labels": {"cifar10": "CIFAR-10", "mnist": "MNIST"},
        "group": "federated",
        "description": "数据集选择",
    },

    # ---------- 攻击通用参数 ----------
    "poison-rate": {
        "type": "float",
        "default": 0.3,
        "group": "attack",
        "description": "投毒比例：恶意客户端中植入后门的样本比例 (0~1)",
    },
    "target-label": {
        "type": "integer",
        "default": 0,
        "group": "attack",
        "description": "后门目标标签（触发器样本将被分类为此标签）",
    },
    "trigger-size": {
        "type": "integer",
        "default": 4,
        "depends_on": {"attack": ["badnets", "wanet"]},
        "group": "attack",
        "description": "触发器尺寸（像素）",
    },

    # WaNet
    "wanet-noise": {
        "type": "float",
        "default": 0.2,
        "depends_on": {"attack": ["wanet"]},
        "group": "attack",
        "description": "WaNet 扭曲噪声强度 (0~1)，值越大变形越明显",
    },

    # Frequency
    "frequency-mode": {
        "type": "string",
        "default": "fft",
        "options": ["dct", "fft"],
        "option_labels": {"dct": "离散余弦变换 (DCT)", "fft": "快速傅里叶变换 (FFT)"},
        "depends_on": {"attack": ["frequency", "frequency_dct", "frequency_fft"]},
        "group": "attack",
        "description": "频域变换类型",
    },
    "frequency-band": {
        "type": "string",
        "default": "high",
        "options": ["low", "high"],
        "option_labels": {"low": "低频分量", "high": "高频分量"},
        "depends_on": {"attack": ["frequency", "frequency_dct", "frequency_fft"]},
        "group": "attack",
        "description": "频带选择：low 篡改低频（隐蔽），high 篡改高频（挑战性）",
    },
    "frequency-window-size": {
        "type": "integer",
        "default": 6,
        "depends_on": {"attack": ["frequency", "frequency_dct", "frequency_fft"]},
        "group": "attack",
        "description": "频域窗口大小（None 表示全图）",
    },
    "frequency-intensity": {
        "type": "float",
        "default": 0.35,
        "depends_on": {"attack": ["frequency", "frequency_dct", "frequency_fft"]},
        "group": "attack",
        "description": "频域篡改强度 (0~1)",
    },
    "frequency-mix-alpha": {
        "type": "float",
        "default": 1.0,
        "depends_on": {"attack": ["frequency", "frequency_dct", "frequency_fft"]},
        "group": "attack",
        "description": "频域混合系数，1.0 表示完全替换",
    },

    # DBA
    "dba-num-sub-patterns": {
        "type": "integer",
        "default": 4,
        "depends_on": {"attack": ["dba"]},
        "group": "attack",
        "description": "DBA 子模式数量（必须为完全平方数，如 4, 9, 16）",
    },
    "dba-sub-pattern-size": {
        "type": "integer",
        "default": None,
        "depends_on": {"attack": ["dba"]},
        "group": "attack",
        "description": "DBA 子模式尺寸（像素），若不指定则自动计算",
    },
    "dba-global-trigger-value": {
        "type": "float",
        "default": 1.0,
        "depends_on": {"attack": ["dba"]},
        "group": "attack",
        "description": "DBA 原始像素值 (0~1，1.0 为白色)，按数据集归一化参数转换",
    },
    "dba-split-strategy": {
        "type": "string",
        "default": "grid",
        "options": ["grid", "random"],
        "option_labels": {"grid": "网格分割", "random": "随机分割"},
        "depends_on": {"attack": ["dba"]},
        "group": "attack",
        "description": "DBA 触发器分割策略",
    },
    "dba-global-trigger-location": {
        "type": "string",  # "[row, col]"
        "default": "",
        "depends_on": {"attack": ["dba"]},
        "group": "attack",
        "description": "DBA 触发器坐标 [行, 列]；留空自动放右下角，手动坐标必须在图像内",
    },

    # FCBA
    "fcba-num-sub-blocks": {
        "type": "integer",
        "default": 4,
        "depends_on": {"attack": ["fcba"]},
        "group": "attack",
        "description": "FCBA 子块数量 m (>=2)",
    },
    "fcba-sub-block-size": {
        "type": "integer",
        "default": None,
        "depends_on": {"attack": ["fcba"]},
        "group": "attack",
        "description": "FCBA 子块尺寸（像素），若不指定则自动计算",
    },
    "fcba-global-trigger-value": {
        "type": "float",
        "default": 1.0,
        "depends_on": {"attack": ["fcba"]},
        "group": "attack",
        "description": "FCBA 原始像素叠加强度 (0~1)，仅在触发区域叠加并裁剪",
    },
    "fcba-split-strategy": {
        "type": "string",
        "default": "grid",
        "options": ["grid", "random"],
        "option_labels": {"grid": "网格分割", "random": "随机分割"},
        "depends_on": {"attack": ["fcba"]},
        "group": "attack",
        "description": "FCBA 触发器分割策略",
    },
    "fcba-global-trigger-location": {
        "type": "string",
        "default": "",
        "depends_on": {"attack": ["fcba"]},
        "group": "attack",
        "description": "FCBA 触发器坐标 [行, 列]；留空自动放右下角，手动坐标必须在图像内",
    },

    # ---------- 客户端防御参数 ----------
    "client-defense-filter-ratio": {
        "type": "float",
        "default": 0.1,
        "depends_on": {"client-defense": ["feature_filter"]},
        "group": "defense",
        "description": "样本过滤比例，保留多少比例的低分样本 (0~1)",
    },
    "client-defense-min-keep": {
        "type": "integer",
        "default": 16,
        "depends_on": {"client-defense": ["feature_filter"]},
        "group": "defense",
        "description": "最少保留样本数，防止过滤过多导致空数据集",
    },
    "client-defense-scoring-batch-size": {
        "type": "integer",
        "default": 128,
        "depends_on": {"client-defense": ["feature_filter"]},
        "group": "defense",
        "description": "特征评分时的批大小",
    },
    "client-defense-use-label-centroids": {
        "type": "boolean",
        "default": True,
        "depends_on": {"client-defense": ["feature_filter"]},
        "group": "defense",
        "description": "是否使用标签感知的类中心进行评分",
    },
    "client-defense-label-blend-alpha": {
        "type": "float",
        "default": 0.5,
        "depends_on": {"client-defense": ["feature_filter"]},
        "group": "defense",
        "description": "标签混合系数 (0~1)，控制全局特征与类中心的混合程度",
    },
    "client-defense-min-class-samples": {
        "type": "integer",
        "default": 8,
        "depends_on": {"client-defense": ["feature_filter"]},
        "group": "defense",
        "description": "每个类别的最少样本数，用于计算类中心",
    },

    # ---------- 服务端检测参数 ----------
    "detection-z-threshold": {
        "type": "float",
        "default": 2.5,
        "depends_on": {"detection": ["anomaly_detection"]},
        "group": "defense",
        "description": "Z-Score 阈值，超过此值的客户端被视为可疑",
    },
    "detection-top-k": {
        "type": "integer",
        "default": 2,
        "depends_on": {"detection": ["anomaly_detection"]},
        "group": "defense",
        "description": "最多剔除客户端数（超出后不再剔除）",
    },
    "detection-min-clients": {
        "type": "integer",
        "default": 3,
        "depends_on": {"detection": ["anomaly_detection"]},
        "group": "defense",
        "description": "最少参与检测的客户端数（低于此值跳过检测）",
    },

    "detection-cosine-floor": {
        "type": "float",
        "default": 0.5,
        "depends_on": {"detection": ["cosine_detection", "score_detection", "clustering_detection"]},
        "group": "defense",
        "description": "余弦相似度下限，低于此值的客户端被视为可疑",
    },
    "detection-min-kept-clients": {
        "type": "integer",
        "default": 5,
        "depends_on": {"detection": ["cosine_detection", "score_detection", "clustering_detection"]},
        "group": "defense",
        "description": "最少保留客户端数（过滤后低于此值时回退）",
    },
    "detection-max-reject-fraction": {
        "type": "float",
        "default": 0.3,
        "depends_on": {"detection": ["cosine_detection", "score_detection", "clustering_detection"]},
        "group": "defense",
        "description": "最大拒绝比例 (0~1)，超过时自动放宽阈值",
    },
    "detection-enable-filter": {
        "type": "boolean",
        "default": True,
        "depends_on": {"detection": ["cosine_detection", "score_detection", "clustering_detection"]},
        "group": "defense",
        "description": "是否启用过滤（false 时保留所有客户端）",
    },

    "detection-percentile": {
        "type": "float",
        "default": 80.0,
        "depends_on": {"detection": ["score_detection", "clustering_detection"]},
        "group": "defense",
        "description": "百分位数阈值 (0~100)，分数高于此百分位的客户端被视为可疑",
    },
    "detection-weight-norm": {
        "type": "float",
        "default": 1.0,
        "depends_on": {"detection": ["score_detection", "clustering_detection"]},
        "group": "defense",
        "description": "分数加权系数：范数 z-score 的权重",
    },
    "detection-weight-cosine": {
        "type": "float",
        "default": 1.0,
        "depends_on": {"detection": ["score_detection", "clustering_detection"]},
        "group": "defense",
        "description": "分数加权系数：余弦相似度的权重",
    },

    "detection-min-silhouette": {
        "type": "float",
        "default": 0.05,
        "depends_on": {"detection": ["clustering_detection"]},
        "group": "defense",
        "description": "最小轮廓系数阈值，低于此值时回退到分数阈值过滤",
    },
    "detection-cluster-score-gap": {
        "type": "float",
        "default": 0.15,
        "depends_on": {"detection": ["clustering_detection"]},
        "group": "defense",
        "description": "聚类间分数差距阈值，最大簇与次大簇分数差低于此值时回退",
    },

    # ---------- 聚合防御参数 ----------
    "defense-clip-norm": {
        "type": "float",
        "default": 3.0,
        "depends_on": {"defense": ["norm_clipping"]},
        "group": "defense",
        "description": "梯度裁剪阈值（L2 范数上限）",
    },
    "defense-trim-ratio": {
        "type": "float",
        "default": None,
        "depends_on": {"defense": ["trimmed_mean"]},
        "group": "defense",
        "description": "比例范围 [0,0.5)，每个坐标的两端各修剪 floor(比例×实际参与数) 个值；须将修剪数量留空才生效",
    },
    "defense-trim-k": {
        "type": "integer",
        "default": 2,
        "depends_on": {"defense": ["trimmed_mean"]},
        "group": "defense",
        "description": "每个坐标的两端各修剪此数量的值；优先于比例，填 0 表示不修剪，留空则按比例计算",
    },
    "defense-num-malicious": {
        "type": "integer",
        "default": 3,
        "depends_on": {"defense": ["krum"]},
        "group": "defense",
        "description": "预估的恶意客户端数量（用于 Krum 距离计算）",
    },
    "defense-krum-k": {
        "type": "integer",
        "default": 3,
        "depends_on": {"defense": ["krum"]},
        "group": "defense",
        "description": "Krum 选择的客户端数量（最终参与聚合的客户端数）",
    },

    # ---------- 实验管理 ----------
    "results-dir": {
        "type": "string",
        "default": "results",
        "group": "management",
        "description": "结果保存根目录（自动创建）；相对路径相对于项目根目录，历史记录、文件下载与报告使用同一目录，旧结果不会自动搬移",
    },
    "run-name": {
        "type": "string",
        "default": "test",
        "group": "management",
        "description": "实验名称，用于生成 CSV 和 PNG 文件名",
    },

    # ---------- 虚拟字段（仅前端使用，不写入 toml）----------
    "malicious-count": {
        "type": "integer",
        "default": 2,
        "ui_only": True,
        "depends_on": {"attack-malicious-mode": ["random", "random_once"]},
        "group": "core",
        "description": "恶意客户端数量（系统将自动计算比例）",
        "compute_from": "malicious-ratio",
        "linked_to": "num-clients",
    },
}

CONFIG_GROUPS = ["core", "federated", "attack", "defense", "management"]


def _read_toml() -> dict[str, Any]:
    with _CONFIG_LOCK:
        with open(TOML_PATH, "rb") as f:
            return tomli.load(f)


def _atomic_write(path: Path, content: bytes) -> None:
    """Write in the same directory, then replace; failures never truncate path."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as f:
            temporary = Path(f.name)
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        if path.exists() and os.name != "nt":
            os.chmod(temporary, path.stat().st_mode)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _write_toml(data: dict[str, Any]) -> None:
    # Serialize completely before creating/opening any destination file.
    content = tomli_w.dumps(data).encode("utf-8")
    with _CONFIG_LOCK:
        _atomic_write(TOML_PATH, content)


def validate_document(content: str) -> dict[str, Any]:
    """Validate syntax and usable project/app sections without touching files."""
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Configuration content cannot be empty.")
    try:
        data = tomli.loads(content.lstrip("\ufeff"))
    except tomli.TOMLDecodeError as exc:
        raise ValueError(f"Invalid TOML: {exc}") from exc
    try:
        project = data["project"]
        app = data["tool"]["flwr"]["app"]
        section = app["config"]
        components = app["components"]
        if not isinstance(project.get("name"), str) or not project["name"].strip():
            raise ValueError("Missing project.name.")
        if not isinstance(section, dict) or not isinstance(components, dict):
            raise ValueError("Flower config and components must be TOML tables.")
        for key in ("serverapp", "clientapp"):
            if not isinstance(components.get(key), str) or not components[key].strip():
                raise ValueError(f"Missing tool.flwr.app.components.{key}.")
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("Configuration must include project and tool.flwr.app config/components tables.") from exc
    normalized = normalize_values(section, CONFIG_SCHEMA, strict=True)
    validate_constraints(normalized, CONFIG_SCHEMA)
    get_results_dir(normalized)
    return data


def replace_config(content: str, *, backup_current: bool = False, backup_name: str | None = None) -> None:
    """Validate first; optionally back up old bytes; atomically install raw TOML."""
    validate_document(content)
    encoded = content.lstrip("\ufeff").replace("\r\n", "\n").encode("utf-8")
    with _CONFIG_LOCK:
        if backup_current:
            _backup_before_save(backup_name)
        _atomic_write(TOML_PATH, encoded)


def _get_config_section(data: dict[str, Any]) -> dict[str, Any]:
    return data.setdefault("tool", {}).setdefault("flwr", {}).setdefault("app", {}).setdefault("config", {})


def get_config() -> dict[str, Any]:
    """Return flat config dict for all schema fields (excluding ui_only)."""
    data = _read_toml()
    config_section = _get_config_section(data)

    result = {}
    # 首先处理 schema 中定义的字段
    for key, schema in CONFIG_SCHEMA.items():
        if schema.get("ui_only"):
            continue
        value = config_section.get(key)
        if value is None:
            value = schema["default"]
        if isinstance(value, (list, tuple)):
            value = ",".join(str(v) for v in value)
        result[key] = value

    if "defense-trim-k" not in config_section and config_section.get("defense-trim-ratio") is not None:
        result["defense-trim-k"] = None

    # 遍历 toml 中的所有键，把不在 schema 中的自定义字段也加入
    for key, value in config_section.items():
        if key in result:
            continue  # 已处理过
        # 跳过 ui_only 虚拟字段（理论上 toml 中不应存在）
        if CONFIG_SCHEMA.get(key, {}).get("ui_only"):
            continue
        # 对列表/元组进行字符串化，方便前端展示
        if isinstance(value, (list, tuple)):
            value = ",".join(str(v) for v in value)
        result[key] = value

    return result


def get_results_dir(section: dict[str, Any] | None = None, *, default: Path | None = None) -> Path:
    """Resolve all API/training/report paths relative to the config's project."""
    if section is None:
        section = _get_config_section(_read_toml())
    from fl_backdoor.utils.paths import resolve_results_root
    return resolve_results_root(section.get("results-dir", str(default) if default is not None else "results"), TOML_PATH.parent)


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


def build_run_config(content: bytes, updates: dict[str, Any]) -> bytes:
    """Build a complete run config from an immutable baseline, without writing."""
    if not isinstance(updates, dict):
        raise ValueError("Configuration updates must be an object.")
    # Validate the merged document, so a valid update can repair an invalid
    # legacy value instead of rejecting the baseline before applying updates.
    try:
        data = tomli.loads(content.decode("utf-8"))
    except tomli.TOMLDecodeError as exc:
        raise ValueError(f"Invalid TOML: {exc}") from exc
    config_section = _get_config_section(data)
    storage_updates = _prepare_storage_updates(updates, config_section)
    normalized = normalize_values(storage_updates, CONFIG_SCHEMA)
    for key in storage_updates:
        if key in OPTIONAL_FIELDS and key not in normalized:
            config_section.pop(key, None)
    config_section.update(normalized)
    checked = normalize_values(config_section, CONFIG_SCHEMA, strict=True)
    validate_constraints(checked, CONFIG_SCHEMA)
    serialized = tomli_w.dumps(data)
    validate_document(serialized)
    return serialized.encode("utf-8")


def restore_run_config(original: bytes, applied: bytes) -> bool:
    """Restore only our own write; never overwrite a user's later config save."""
    with _CONFIG_LOCK:
        if TOML_PATH.read_bytes() != applied:
            return False
        _atomic_write(TOML_PATH, original)
        return True


def update_config(updates: dict[str, Any], *, backup_current: bool = False, backup_name: str | None = None) -> None:
    """Update configuration after converting UI-only fields."""
    with _CONFIG_LOCK:
        content = build_run_config(TOML_PATH.read_bytes(), updates)
        # All validation and serialization precedes backup mutation, too.
        if backup_current:
            _backup_before_save(backup_name)
        _atomic_write(TOML_PATH, content)


def _prepare_storage_updates(updates: dict[str, Any], current: dict[str, Any] | None = None) -> dict[str, Any]:
    """Convert UI-only fields to actual toml fields."""
    storage = updates.copy()

    # 处理 malicious-count → malicious-ratio
    if "malicious-count" in storage:
        num_clients = parse_integer(
            updates.get("num-clients", (current or {}).get("num-clients", 10)), "num-clients", minimum=1,
        )
        count = parse_integer(storage.pop("malicious-count"), "malicious-count")
        if count > num_clients:
            raise ValueError("malicious-count: cannot exceed num-clients.")
        storage["malicious-ratio"] = count / num_clients

    # 移除所有 ui_only 字段
    for key, schema in CONFIG_SCHEMA.items():
        if schema.get("ui_only") and key in storage:
            storage.pop(key, None)

    return storage


def backup_config() -> None:
    with _CONFIG_LOCK:
        _atomic_write(_legacy_backup_path(), TOML_PATH.read_bytes())


def _backup_before_save(name: str | None) -> None:
    if name is not None:
        if create_backup(name) is None:
            raise FileExistsError("Backup name already exists; choose another name.")
    backup_config()


def restore_config() -> None:
    with _CONFIG_LOCK:
        path = _legacy_backup_path()
        if path.exists():
            replace_config(path.read_text(encoding="utf-8"))
            path.unlink()


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
    

# 匹配所有 .toml 文件，但排除主配置文件和 .bak 文件
BACKUP_PATTERN = re.compile(r"^(.+)\.toml$")


def _backup_directory() -> Path:
    return child_path(PROJECT_ROOT, "config_backups")


def _legacy_backup_path() -> Path:
    path = child_path(TOML_PATH.parent, BACKUP_PATH.name, label="Automatic backup")
    if path != BACKUP_PATH.resolve() or path == TOML_PATH.resolve():
        raise ValueError("Invalid automatic backup path.")
    return path

def list_backups() -> list[dict[str, Any]]:
    """列出所有备份文件及其元信息"""
    backups = []
    seen = set()
    candidates = list(_backup_directory().glob("*.toml")) + list(PROJECT_ROOT.glob("*.toml"))
    for p in candidates:
        # 排除主配置文件和旧式备份
        if p.name.casefold() in seen:
            continue
        try:
            checked = _backup_path(p.name)
            validate_document(checked.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        p = checked
        seen.add(p.name.casefold())
        
        match = BACKUP_PATTERN.match(p.name)
        if not match:
            continue
        
        name_part = match.group(1)
        ts = None
        # 尝试从文件名中提取时间戳（格式：YYYYMMDD_HHMMSS_fff 或末尾带括号的）
        ts_match = re.search(r"(\d{8}_\d{6}_\d{3})", name_part)
        if ts_match:
            try:
                ts = datetime.strptime(ts_match.group(1), "%Y%m%d_%H%M%S_%f")
            except ValueError:
                ts = datetime.fromtimestamp(p.stat().st_ctime)
        else:
            # 也可能是纯时间戳文件名（如 20260414_212523_224.toml）
            ts_match2 = re.search(r"^(\d{8}_\d{6}_\d{3})$", name_part)
            if ts_match2:
                try:
                    ts = datetime.strptime(ts_match2.group(1), "%Y%m%d_%H%M%S_%f")
                except ValueError:
                    ts = datetime.fromtimestamp(p.stat().st_ctime)
            else:
                ts = datetime.fromtimestamp(p.stat().st_ctime)
        
        backups.append({
            "filename": p.name,
            "timestamp": ts.isoformat(),
            "size": p.stat().st_size,
        })
    
    backups.sort(key=lambda x: x["timestamp"], reverse=True)
    return backups

def _backup_path(filename: str, *, allow_legacy: bool = False, for_write: bool = False) -> Path:
    validate_name(filename, label="Backup filename")
    if filename.casefold() == TOML_PATH.name.casefold() or ":" in filename:
        raise ValueError("The active project configuration cannot be used as a backup.")
    if not BACKUP_PATTERN.fullmatch(filename) and not (allow_legacy and filename == "pyproject.toml.bak"):
        raise ValueError("Backup filename must end in .toml.")
    if allow_legacy and filename == "pyproject.toml.bak":
        return _legacy_backup_path()
    path = child_path(_backup_directory(), filename, label="Backup filename")
    # Old root-level backups remain accessible without moving or deleting them.
    if not for_write and not path.exists():
        path = child_path(PROJECT_ROOT, filename, label="Backup filename")
        if path.is_file():
            # Compatibility is for actual old experiment configurations, not
            # arbitrary root-level TOML files the caller would like to delete.
            validate_document(path.read_text(encoding="utf-8"))
    if path.resolve() == TOML_PATH.resolve():
        raise ValueError("The active project configuration cannot be used as a backup.")
    return path


def read_backup_content(filename: str) -> str:
    """读取指定备份文件的内容"""
    backup_path = _backup_path(filename, allow_legacy=True)
    if not backup_path.is_file():
        raise FileNotFoundError(f"Backup file '{filename}' not found")
    with open(backup_path, "r", encoding="utf-8") as f:
        return f.read()

def delete_backup(filename: str) -> None:
    """删除指定的备份文件"""
    backup_path = _backup_path(filename)
    if not backup_path.is_file():
        raise FileNotFoundError(f"Backup file '{filename}' not found")
    backup_path.unlink()

def restore_from_backup(filename: str, backup_current: bool = True, backup_name: str | None = None) -> None:
    """从指定备份恢复配置，可选择是否先备份当前配置"""
    backup_path = _backup_path(filename, allow_legacy=True)
    if not backup_path.is_file():
        raise FileNotFoundError(f"Backup file '{filename}' not found")
    # Read first: backing up current config may overwrite the legacy .bak source.
    with _CONFIG_LOCK:
        content = backup_path.read_text(encoding="utf-8")
        replace_config(content, backup_current=backup_current,
                       backup_name=(backup_name or "") if backup_current else None)


def update_backup_content(filename: str, content: str) -> None:
    path = _backup_path(filename)
    if not path.is_file():
        raise FileNotFoundError(f"Backup file '{filename}' not found")
    validate_document(content)
    with _CONFIG_LOCK:
        _atomic_write(path, content.lstrip("\ufeff").replace("\r\n", "\n").encode("utf-8"))

def _generate_backup_filename(custom_name: str | None = None, auto_suffix: str | None = None) -> str:
    """生成备份文件名，处理重名冲突"""
    if custom_name is not None and not isinstance(custom_name, str):
        raise ValueError("Backup name must be text.")
    if custom_name:
        validate_name(custom_name, label="Backup name")
        safe_name = custom_name
        # 确保扩展名为 .toml
        if not safe_name.lower().endswith('.toml'):
            safe_name += '.toml'
        else:
            safe_name = safe_name[:-5] + '.toml'
        filename = safe_name
    else:
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"{timestamp}.toml"

    if auto_suffix:
        base, ext = os.path.splitext(filename)
        filename = f"{base}_{auto_suffix}{ext}"

    _backup_path(filename, for_write=True)
    return filename


def _resolve_backup_path(filename: str, on_conflict: str = 'ask') -> Path | None:
    """
    解析备份路径，处理重名冲突。
    on_conflict: 'ask' (返回 None 表示需询问), 'auto' (自动加后缀), 'overwrite' (直接覆盖)
    返回 None 表示需要用户决策，否则返回最终路径。
    """
    if on_conflict not in {"ask", "auto", "overwrite"}:
        raise ValueError("on_conflict must be ask, auto or overwrite.")
    path = _backup_path(filename, for_write=True)
    legacy = child_path(PROJECT_ROOT, filename, label="Backup filename")
    if not path.exists() and not legacy.exists():
        return path

    if on_conflict == 'overwrite':
        return path
    elif on_conflict == 'auto':
        base, ext = os.path.splitext(filename)
        counter = 1
        while True:
            new_filename = f"{base}({counter}){ext}"
            new_path = _backup_path(new_filename, for_write=True)
            old_path = child_path(PROJECT_ROOT, new_filename, label="Backup filename")
            if not new_path.exists() and not old_path.exists():
                return new_path
            counter += 1
    else:  # 'ask'
        return None


def create_backup(custom_name: str | None = None, on_conflict: str = 'ask') -> Path | None:
    """
    创建当前配置的备份。
    返回备份路径，如果重名且 on_conflict='ask' 则返回 None。
    """
    filename = _generate_backup_filename(custom_name)
    with _CONFIG_LOCK:
        path = _resolve_backup_path(filename, on_conflict)
        if path is None:
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(path, TOML_PATH.read_bytes())
        return path


def create_backup_from_content(content: str, custom_name: str | None = None, on_conflict: str = 'ask') -> Path | None:
    """
    从给定内容创建备份文件。
    返回备份路径，如果重名且 on_conflict='ask' 则返回 None。
    """
    validate_document(content)
    filename = _generate_backup_filename(custom_name)
    with _CONFIG_LOCK:
        path = _resolve_backup_path(filename, on_conflict)
        if path is None:
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(path, content.lstrip("\ufeff").replace("\r\n", "\n").encode("utf-8"))
        return path