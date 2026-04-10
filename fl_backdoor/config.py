"""统一配置层，从 run_config 字典解析所有实验参数。"""

from dataclasses import dataclass, field
from typing import Any, Optional, Union


@dataclass
class ExperimentConfig:
    """联邦学习后门攻防实验的完整配置。"""

    # ---------- 基础联邦学习参数 ----------
    num_server_rounds: int = 10
    fraction_evaluate: float = 0.2
    local_epochs: int = 2
    learning_rate: float = 0.01
    batch_size: int = 32
    seed: int = 42
    dataset: str = "cifar10"

    # ---------- 攻击通用参数 ----------
    attack: str = "badnets"  # none / badnets / wanet / frequency / dba / fcba
    malicious_ratio: float = 0.2
    poison_rate: float = 0.05
    target_label: int = 0
    trigger_size: int = 4

    # ---------- 恶意客户端选择 ----------
    attack_malicious_mode: str = "random"  # random / fixed
    attack_fixed_clients: Optional[list[int]] = None

    # ---------- WaNet 参数 ----------
    wanet_grid_size: Optional[int] = None
    wanet_noise: float = 0.05

    # ---------- Frequency 攻击参数 ----------
    frequency_mode: str = "dct"  # dct / fft
    frequency_band: str = "low"  # low / high
    frequency_window_size: Optional[int] = None
    frequency_intensity: float = 0.10
    frequency_mix_alpha: float = 1.0

    # ---------- DBA 参数 ----------
    dba_num_sub_patterns: int = 4
    dba_sub_pattern_size: Optional[int] = None
    dba_global_trigger_value: float = 1.0
    dba_split_strategy: str = "grid"

    # ---------- FCBA 参数 ----------
    fcba_num_sub_blocks: int = 4
    fcba_sub_block_size: Optional[int] = None
    fcba_global_trigger_value: float = 1.0
    fcba_split_strategy: str = "grid"
    fcba_global_trigger_location: Optional[tuple[int, int]] = None

    # ---------- 客户端防御 ----------
    client_defense: str = "none"  # none / feature_filter
    client_defense_filter_ratio: float = 0.1
    client_defense_min_keep: int = 16
    client_defense_scoring_batch_size: int = 128
    client_defense_use_label_centroids: bool = True
    client_defense_label_blend_alpha: float = 0.5
    client_defense_min_class_samples: int = 8

    # ---------- 服务端检测 ----------
    detection: str = "none"  # none / anomaly_detection / cosine_detection / score_detection / clustering_detection

    # 通用检测参数
    detection_min_kept_clients: int = 5
    detection_enable_filter: bool = True
    detection_max_reject_fraction: float = 0.3
    detection_percentile: float = 80.0
    detection_weight_norm: float = 1.0
    detection_weight_cosine: float = 1.0

    # Anomaly 专用
    detection_z_threshold: float = 2.5
    detection_top_k: int = 2
    detection_min_clients: int = 3

    # Cosine 专用
    detection_cosine_floor: float = 0.5

    # Clustering 专用
    detection_min_silhouette: float = 0.05
    detection_cluster_score_gap: float = 0.15

    # ---------- 服务端聚合防御 ----------
    defense: str = "none"  # none / norm_clipping / trimmed_mean / krum

    # Norm Clipping
    defense_clip_norm: float = 3.0

    # Trimmed Mean
    defense_trim_k: int = 2

    # Krum
    defense_num_malicious: int = 3
    defense_krum_k: int = 3

    # ---------- 实验管理 ----------
    results_dir: str = "results"
    run_name: str = "experiment"

    @classmethod
    def from_run_config(cls, run_config: dict[str, Any]) -> "ExperimentConfig":
        """从 Flower run_config 字典构造配置对象。"""
        def get(key: str, default: Any = None, converter=None):
            value = run_config.get(key, default)
            if converter is not None and value is not None:
                return converter(value)
            return value

        def parse_int_list(raw) -> Optional[list[int]]:
            if raw is None:
                return None
            if isinstance(raw, str):
                return [int(x.strip()) for x in raw.split(",") if x.strip()]
            if isinstance(raw, (list, tuple)):
                return [int(x) for x in raw]
            raise ValueError(f"Cannot parse as int list: {raw}")

        def parse_tuple(raw) -> Optional[tuple[int, int]]:
            if raw is None:
                return None
            if isinstance(raw, str):
                parts = raw.strip("[]() ").split(",")
                if len(parts) >= 2:
                    return int(parts[0].strip()), int(parts[1].strip())
            if isinstance(raw, (list, tuple)) and len(raw) >= 2:
                return int(raw[0]), int(raw[1])
            return None

        return cls(
            # 基础
            num_server_rounds=get("num-server-rounds", 10, int),
            fraction_evaluate=get("fraction-evaluate", 0.2, float),
            local_epochs=get("local-epochs", 2, int),
            learning_rate=get("learning-rate", 0.01, float),
            batch_size=get("batch-size", 32, int),
            seed=get("seed", 42, int),
            dataset=get("dataset", "cifar10", str),
            # 攻击通用
            attack=get("attack", "badnets", str),
            malicious_ratio=get("malicious-ratio", 0.2, float),
            poison_rate=get("poison-rate", 0.05, float),
            target_label=get("target-label", 0, int),
            trigger_size=get("trigger-size", 4, int),
            # 恶意客户端选择
            attack_malicious_mode=get("attack-malicious-mode", "random", str),
            attack_fixed_clients=parse_int_list(get("attack-fixed-clients")),
            # WaNet
            wanet_grid_size=get("wanet-grid-size", None, lambda x: int(x) if x is not None else None),
            wanet_noise=get("wanet-noise", 0.05, float),
            # Frequency
            frequency_mode=get("frequency-mode", "dct", str),
            frequency_band=get("frequency-band", "low", str),
            frequency_window_size=get("frequency-window-size", None, lambda x: int(x) if x is not None else None),
            frequency_intensity=get("frequency-intensity", 0.10, float),
            frequency_mix_alpha=get("frequency-mix-alpha", 1.0, float),
            # DBA
            dba_num_sub_patterns=get("dba-num-sub-patterns", 4, int),
            dba_sub_pattern_size=get("dba-sub-pattern-size", None, lambda x: int(x) if x is not None else None),
            dba_global_trigger_value=get("dba-global-trigger-value", 1.0, float),
            dba_split_strategy=get("dba-split-strategy", "grid", str),
            # FCBA
            fcba_num_sub_blocks=get("fcba-num-sub-blocks", 4, int),
            fcba_sub_block_size=get("fcba-sub-block-size", None, lambda x: int(x) if x is not None else None),
            fcba_global_trigger_value=get("fcba-global-trigger-value", 1.0, float),
            fcba_split_strategy=get("fcba-split-strategy", "grid", str),
            fcba_global_trigger_location=parse_tuple(get("fcba-global-trigger-location")),
            # 客户端防御
            client_defense=get("client-defense", "none", str),
            client_defense_filter_ratio=get("client-defense-filter-ratio", 0.1, float),
            client_defense_min_keep=get("client-defense-min-keep", 16, int),
            client_defense_scoring_batch_size=get("client-defense-scoring-batch-size", 128, int),
            client_defense_use_label_centroids=get("client-defense-use-label-centroids", True, bool),
            client_defense_label_blend_alpha=get("client-defense-label-blend-alpha", 0.5, float),
            client_defense_min_class_samples=get("client-defense-min-class-samples", 8, int),
            # 服务端检测
            detection=get("detection", "none", str),
            detection_min_kept_clients=get("detection-min-kept-clients", 5, int),
            detection_enable_filter=get("detection-enable-filter", True, bool),
            detection_max_reject_fraction=get("detection-max-reject-fraction", 0.3, float),
            detection_percentile=get("detection-percentile", 80.0, float),
            detection_weight_norm=get("detection-weight-norm", 1.0, float),
            detection_weight_cosine=get("detection-weight-cosine", 1.0, float),
            detection_z_threshold=get("detection-z-threshold", 2.5, float),
            detection_top_k=get("detection-top-k", 2, int),
            detection_min_clients=get("detection-min-clients", 3, int),
            detection_cosine_floor=get("detection-cosine-floor", 0.5, float),
            detection_min_silhouette=get("detection-min-silhouette", 0.05, float),
            detection_cluster_score_gap=get("detection-cluster-score-gap", 0.15, float),
            # 聚合防御
            defense=get("defense", "none", str),
            defense_clip_norm=get("defense-clip-norm", 3.0, float),
            defense_trim_k=get("defense-trim-k", 2, int),
            defense_num_malicious=get("defense-num-malicious", 3, int),
            defense_krum_k=get("defense-krum-k", 3, int),
            # 实验管理
            results_dir=get("results-dir", "results", str),
            run_name=get("run-name", "experiment", str),
        )

    def to_attack_kwargs(self) -> dict[str, Any]:
        """提取构建攻击所需的参数字典。"""
        return {
            "attack_type": self.attack,
            "malicious_ratio": self.malicious_ratio,
            "poison_rate": self.poison_rate,
            "target_label": self.target_label,
            "trigger_size": self.trigger_size,
            "seed": self.seed,
            "malicious_mode": self.attack_malicious_mode,
            "fixed_malicious_clients": self.attack_fixed_clients,
            "grid_size": self.wanet_grid_size,
            "noise_scale": self.wanet_noise,
            "frequency_mode": self.frequency_mode,
            "frequency_band": self.frequency_band,
            "frequency_window_size": self.frequency_window_size,
            "frequency_intensity": self.frequency_intensity,
            "frequency_mix_alpha": self.frequency_mix_alpha,
            "dba_num_sub_patterns": self.dba_num_sub_patterns,
            "dba_sub_pattern_size": self.dba_sub_pattern_size,
            "dba_global_trigger_value": self.dba_global_trigger_value,
            "dba_split_strategy": self.dba_split_strategy,
            "fcba_num_sub_blocks": self.fcba_num_sub_blocks,
            "fcba_sub_block_size": self.fcba_sub_block_size,
            "fcba_global_trigger_value": self.fcba_global_trigger_value,
            "fcba_split_strategy": self.fcba_split_strategy,
            "fcba_global_trigger_location": self.fcba_global_trigger_location,
        }
    
    def get_defense_pipeline_kwargs(self) -> dict[str, Any]:
        """返回构建防御流水线所需的三组参数字典。"""
        client_defense_kwargs = {
            "filter_ratio": self.client_defense_filter_ratio,
            "min_keep": self.client_defense_min_keep,
            "scoring_batch_size": self.client_defense_scoring_batch_size,
            "use_label_centroids": self.client_defense_use_label_centroids,
            "label_blend_alpha": self.client_defense_label_blend_alpha,
            "min_class_samples": self.client_defense_min_class_samples,
        }

        detection_kwargs = {
            "min_kept_clients": self.detection_min_kept_clients,
            "enable_filter": self.detection_enable_filter,
            "max_reject_fraction": self.detection_max_reject_fraction,
            "percentile": self.detection_percentile,
            "weight_norm": self.detection_weight_norm,
            "weight_cosine": self.detection_weight_cosine,
            "z_threshold": self.detection_z_threshold,
            "top_k": self.detection_top_k,
            "min_clients": self.detection_min_clients,
            "cosine_floor": self.detection_cosine_floor,
            "min_silhouette": self.detection_min_silhouette,
            "cluster_score_gap": self.detection_cluster_score_gap,
        }

        aggregation_kwargs = {
            "clip_norm": self.defense_clip_norm,
            "trim_k": self.defense_trim_k,
            "num_malicious": self.defense_num_malicious,
            "krum_k": self.defense_krum_k,
        }

        return {
            "client_defense_type": self.client_defense,
            "detection_type": self.detection,
            "aggregation_type": self.defense,
            "seed": self.seed,
            "client_defense_kwargs": client_defense_kwargs,
            "detection_kwargs": detection_kwargs,
            "aggregation_kwargs": aggregation_kwargs,
        }

    def get_defense_pipeline_kwargs(self) -> dict[str, Any]:
        """提取构建防御流水线所需的参数字典。"""
        return {
            "client_defense_type": self.client_defense,
            "detection_type": self.detection,
            "aggregation_type": self.defense,
            "seed": self.seed,
            "client_defense_kwargs": {
                "filter_ratio": self.client_defense_filter_ratio,
                "min_keep": self.client_defense_min_keep,
                "scoring_batch_size": self.client_defense_scoring_batch_size,
                "use_label_centroids": self.client_defense_use_label_centroids,
                "label_blend_alpha": self.client_defense_label_blend_alpha,
                "min_class_samples": self.client_defense_min_class_samples,
            },
            "detection_kwargs": {
                "min_kept_clients": self.detection_min_kept_clients,
                "enable_filter": self.detection_enable_filter,
                "max_reject_fraction": self.detection_max_reject_fraction,
                "percentile": self.detection_percentile,
                "weight_norm": self.detection_weight_norm,
                "weight_cosine": self.detection_weight_cosine,
                "z_threshold": self.detection_z_threshold,
                "top_k": self.detection_top_k,
                "min_clients": self.detection_min_clients,
                "cosine_floor": self.detection_cosine_floor,
                "min_silhouette": self.detection_min_silhouette,
                "cluster_score_gap": self.detection_cluster_score_gap,
            },
            "aggregation_kwargs": {
                "clip_norm": self.defense_clip_norm,
                "trim_k": self.defense_trim_k,
                "num_malicious": self.defense_num_malicious,
                "krum_k": self.defense_krum_k,
            },
        }