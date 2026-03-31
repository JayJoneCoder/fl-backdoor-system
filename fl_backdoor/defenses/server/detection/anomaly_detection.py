"""Server-side anomaly detection defense for client updates."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg

from .base import DetectionBase, DetectionConfig

_EPS = 1e-12


def _state_dict_to_vector(state_dict: dict[str, Any]) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for value in state_dict.values():
        if hasattr(value, "detach"):
            array = value.detach().cpu().numpy()
        else:
            array = np.asarray(value)
        chunks.append(np.asarray(array, dtype=np.float64).reshape(-1))

    if not chunks:
        return np.empty((0,), dtype=np.float64)

    return np.concatenate(chunks, axis=0)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = float(np.linalg.norm(a))
    b_norm = float(np.linalg.norm(b))
    if a_norm <= _EPS or b_norm <= _EPS:
        return 1.0
    return float(np.dot(a, b) / (a_norm * b_norm + _EPS))


def _merge_metrics(metrics: MetricRecord | None, extra: dict[str, Any]) -> MetricRecord:
    merged: dict[str, Any] = {}
    if metrics is not None:
        merged.update(dict(metrics))
    merged.update(extra)
    return MetricRecord(merged)


class AnomalyDetectionFedAvg(FedAvg):
    """FedAvg with anomaly detection before aggregation."""

    def __init__(
        self,
        fraction_train: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_train_nodes: int = 2,
        min_evaluate_nodes: int = 2,
        min_available_nodes: int = 2,
        weighted_by_key: str = "num-examples",
        arrayrecord_key: str = "arrays",
        configrecord_key: str = "config",
        train_metrics_aggr_fn=None,
        evaluate_metrics_aggr_fn=None,
        norm_z_threshold: float = 3.5,
        cosine_floor: float = 0.0,
        min_kept_clients: int = 2,
        max_reject_fraction: float = 0.5,
        enable_filter: bool = True,
    ) -> None:
        super().__init__(
            fraction_train=fraction_train,
            fraction_evaluate=fraction_evaluate,
            min_train_nodes=min_train_nodes,
            min_evaluate_nodes=min_evaluate_nodes,
            min_available_nodes=min_available_nodes,
            weighted_by_key=weighted_by_key,
            arrayrecord_key=arrayrecord_key,
            configrecord_key=configrecord_key,
            train_metrics_aggr_fn=train_metrics_aggr_fn,
            evaluate_metrics_aggr_fn=evaluate_metrics_aggr_fn,
        )

        self.norm_z_threshold = float(norm_z_threshold)
        self.cosine_floor = float(cosine_floor)
        self.min_kept_clients = int(min_kept_clients)
        self.max_reject_fraction = float(max_reject_fraction)
        self.enable_filter = bool(enable_filter)
        self.current_arrays: ArrayRecord | None = None

        if self.min_kept_clients < 1:
            raise ValueError("min_kept_clients must be >= 1.")
        if not (0.0 <= self.max_reject_fraction <= 1.0):
            raise ValueError("max_reject_fraction must be in [0.0, 1.0].")

    def summary(self) -> None:
        super().summary()
        from flwr.common.logger import log
        from logging import INFO

        log(INFO, "\t├──> Defense settings:")
        log(INFO, "\t│\t├── defense: anomaly_detection")
        log(INFO, "\t│\t├── norm_z_threshold: %.6f", self.norm_z_threshold)
        log(INFO, "\t│\t├── cosine_floor: %.6f", self.cosine_floor)
        log(INFO, "\t│\t├── min_kept_clients: %d", self.min_kept_clients)
        log(INFO, "\t│\t├── max_reject_fraction: %.6f", self.max_reject_fraction)
        log(INFO, "\t│\t└── enable_filter: %s", str(self.enable_filter))

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        self.current_arrays = arrays.copy()
        return super().configure_train(server_round, arrays, config, grid)

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        valid_replies = [msg for msg in replies if not msg.has_error()]
        if not valid_replies:
            return None, None

        if self.current_arrays is None:
            return super().aggregate_train(server_round, valid_replies)

        global_vector = _state_dict_to_vector(
            self.current_arrays.to_torch_state_dict()
        )
        if global_vector.size == 0:
            return super().aggregate_train(server_round, valid_replies)

        scored_replies: list[Message] = []
        vectors: list[np.ndarray] = []
        skipped = 0

        for msg in valid_replies:
            try:
                local_arrays = msg.content[self.arrayrecord_key]
                local_vector = _state_dict_to_vector(local_arrays.to_torch_state_dict())
            except Exception:
                skipped += 1
                continue

            if local_vector.shape != global_vector.shape:
                skipped += 1
                continue

            scored_replies.append(msg)
            vectors.append(local_vector)

        if not scored_replies:
            return super().aggregate_train(server_round, valid_replies)

        stacked = np.stack(vectors, axis=0)
        deltas = stacked - global_vector.reshape(1, -1)

        norms = np.linalg.norm(deltas, axis=1)
        median_norm = float(np.median(norms)) if len(norms) > 0 else 0.0
        mad = float(np.median(np.abs(norms - median_norm))) if len(norms) > 0 else 0.0
        robust_scale = 1.4826 * mad + _EPS
        norm_z = np.abs(norms - median_norm) / robust_scale

        center = np.mean(deltas, axis=0)
        cosine = np.array(
            [_cosine_similarity(delta, center) for delta in deltas],
            dtype=np.float64,
        )
        score = norm_z + np.clip(1.0 - cosine, 0.0, 2.0)

        suspicious = (norm_z > self.norm_z_threshold) | (cosine < self.cosine_floor)

        total = len(scored_replies)
        max_rejects = int(np.floor(total * self.max_reject_fraction))
        target_keep = max(self.min_kept_clients, total - max_rejects)
        target_keep = min(target_keep, total)

        if self.enable_filter:
            keep_indices = np.where(~suspicious)[0].tolist()
            if len(keep_indices) < target_keep:
                keep_indices = np.argsort(score)[:target_keep].tolist()
            keep_indices = sorted(set(int(i) for i in keep_indices))
        else:
            keep_indices = list(range(total))

        filtered_replies = [scored_replies[i] for i in keep_indices]
        arrays, metrics = super().aggregate_train(server_round, filtered_replies)

        extra_metrics = {
            "defense-total-clients": int(total),
            "defense-kept-clients": int(len(filtered_replies)),
            "defense-filtered-clients": int(total - len(filtered_replies)),
            "defense-filter-ratio": (
                float((total - len(filtered_replies)) / total) if total > 0 else 0.0
            ),
            "defense-suspicious-clients": int(np.sum(suspicious)),
            "defense-norm-z-threshold": float(self.norm_z_threshold),
            "defense-cosine-floor": float(self.cosine_floor),
            "defense-mean-update-norm": float(np.mean(norms)) if len(norms) > 0 else 0.0,
            "defense-max-update-norm": float(np.max(norms)) if len(norms) > 0 else 0.0,
            "defense-mean-norm-z": float(np.mean(norm_z)) if len(norm_z) > 0 else 0.0,
            "defense-max-norm-z": float(np.max(norm_z)) if len(norm_z) > 0 else 0.0,
            "defense-mean-cosine": float(np.mean(cosine)) if len(cosine) > 0 else 0.0,
            "defense-min-cosine": float(np.min(cosine)) if len(cosine) > 0 else 0.0,
            "defense-mean-score": float(np.mean(score)) if len(score) > 0 else 0.0,
            "defense-max-score": float(np.max(score)) if len(score) > 0 else 0.0,
            "defense-skip-count": int(skipped),
        }

        if metrics is None:
            metrics = MetricRecord()
        return arrays, _merge_metrics(metrics, extra_metrics)


class AnomalyDetectionDefense(DetectionBase):
    """Detection wrapper that turns FedAvg into AnomalyDetectionFedAvg."""

    def apply(self, strategy: Any) -> Any:
        if not isinstance(strategy, FedAvg):
            raise TypeError(
                f"AnomalyDetectionDefense only supports FedAvg-compatible strategies, got {type(strategy)!r}."
            )

        extra = self.config.extra
        norm_z_threshold = extra.get("norm_z_threshold", extra.get("norm-z-threshold", 3.5))
        cosine_floor = extra.get("cosine_floor", extra.get("cosine-floor", 0.0))
        min_kept_clients = extra.get("min_kept_clients", extra.get("min-kept-clients", 2))
        max_reject_fraction = extra.get(
            "max_reject_fraction", extra.get("max-reject-fraction", 0.5)
        )
        enable_filter = extra.get("enable_filter", extra.get("enable-filter", True))

        return AnomalyDetectionFedAvg(
            fraction_train=strategy.fraction_train,
            fraction_evaluate=strategy.fraction_evaluate,
            min_train_nodes=strategy.min_train_nodes,
            min_evaluate_nodes=strategy.min_evaluate_nodes,
            min_available_nodes=strategy.min_available_nodes,
            weighted_by_key=strategy.weighted_by_key,
            arrayrecord_key=strategy.arrayrecord_key,
            configrecord_key=strategy.configrecord_key,
            train_metrics_aggr_fn=strategy.train_metrics_aggr_fn,
            evaluate_metrics_aggr_fn=strategy.evaluate_metrics_aggr_fn,
            norm_z_threshold=float(norm_z_threshold),
            cosine_floor=float(cosine_floor),
            min_kept_clients=int(min_kept_clients),
            max_reject_fraction=float(max_reject_fraction),
            enable_filter=bool(enable_filter),
        )