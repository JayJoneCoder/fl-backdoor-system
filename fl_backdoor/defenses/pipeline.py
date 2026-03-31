"""Defense pipeline for federated learning.

This module composes:
1) client-side defense (data preprocessing, executed on client)
2) server-side detection (pre-aggregation filtering)
3) server-side aggregation defense (norm clipping / trimmed mean / krum / none)

It keeps the existing factory style and does not modify existing defense modules.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from flwr.app import Array, ArrayRecord, ConfigRecord, Message, MetricRecord, RecordDict
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg
from flwr.serverapp.strategy.strategy_utils import aggregate_arrayrecords

from .base import DefenseBase, DefenseConfig, IdentityDefense
from .client import build_client_defense
from .server.detection import build_detection
from .server.aggregation.norm_clipping import NormClippingDefense
from .server.aggregation.trimmed_mean import TrimmedMeanDefense
from .server.aggregation.krum import KrumDefense

_EPS = 1e-12


def _normalize_name(value: Any) -> str:
    return str(value).lower().strip().replace("-", "_")


def _normalize_kwargs(kwargs: Mapping[str, Any] | None) -> dict[str, Any]:
    if kwargs is None:
        return {}
    return {_normalize_name(k): v for k, v in dict(kwargs).items()}


def _prefixed_kwargs(run_config: Mapping[str, Any], prefix: str) -> dict[str, Any]:
    """Extract `prefix-*` keys from run_config and convert them to snake_case."""
    out: dict[str, Any] = {}
    for key, value in dict(run_config).items():
        if key.startswith(prefix):
            out[key.removeprefix(prefix).replace("-", "_")] = value
    return out


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


def _clip_client_update(
    global_ndarrays: list[np.ndarray],
    local_ndarrays: list[np.ndarray],
    clip_norm: float,
) -> tuple[list[np.ndarray], float, float, bool]:
    """Clip one client's model update by global L2 norm."""
    if len(global_ndarrays) != len(local_ndarrays):
        raise ValueError(
            "global_ndarrays and local_ndarrays must have the same length."
        )

    float_indices: list[int] = []
    float_deltas: list[np.ndarray] = []

    for idx, (g_arr, l_arr) in enumerate(
        zip(global_ndarrays, local_ndarrays, strict=True)
    ):
        g = np.asarray(g_arr)
        l = np.asarray(l_arr)

        if np.issubdtype(g.dtype, np.floating) and np.issubdtype(l.dtype, np.floating):
            float_indices.append(idx)
            float_deltas.append(l - g)

    if not float_deltas:
        return [np.asarray(arr) for arr in local_ndarrays], 0.0, 0.0, False

    flat_delta = np.concatenate(
        [delta.reshape(-1).astype(np.float64, copy=False) for delta in float_deltas]
    )
    pre_norm = float(np.linalg.norm(flat_delta))

    if pre_norm <= clip_norm or pre_norm == 0.0:
        scale = 1.0
        was_clipped = False
    else:
        scale = clip_norm / pre_norm
        was_clipped = True

    float_index_set = set(float_indices)
    clipped_ndarrays: list[np.ndarray] = []
    float_pos = 0

    for idx, (g_arr, l_arr) in enumerate(
        zip(global_ndarrays, local_ndarrays, strict=True)
    ):
        g = np.asarray(g_arr)
        l = np.asarray(l_arr)

        if idx in float_index_set:
            delta = float_deltas[float_pos]
            clipped_ndarrays.append(g + delta * scale)
            float_pos += 1
        else:
            clipped_ndarrays.append(l)

    post_norm = pre_norm * scale
    return clipped_ndarrays, pre_norm, post_norm, was_clipped


def _build_trimmed_mean(client_tensors: list[np.ndarray], trim_k: int) -> np.ndarray:
    if not client_tensors:
        raise ValueError("client_tensors must not be empty.")

    num_clients = len(client_tensors)
    if trim_k < 0:
        raise ValueError("trim_k must be non-negative.")
    if 2 * trim_k >= num_clients:
        raise ValueError(
            f"trim_k={trim_k} is too large for num_clients={num_clients}. "
            "Need 2 * trim_k < num_clients."
        )

    ref = np.asarray(client_tensors[0])
    if not np.issubdtype(ref.dtype, np.floating):
        return np.asarray(ref)

    stacked = np.stack(
        [np.asarray(t, dtype=np.float64) for t in client_tensors],
        axis=0,
    )
    sorted_stack = np.sort(stacked, axis=0)
    trimmed_stack = sorted_stack[trim_k : num_clients - trim_k]
    trimmed_mean = trimmed_stack.mean(axis=0)

    return trimmed_mean.astype(ref.dtype, copy=False)


def _flatten_update(
    global_ndarrays: list[np.ndarray],
    local_ndarrays: list[np.ndarray],
) -> np.ndarray:
    deltas = []
    for g, l in zip(global_ndarrays, local_ndarrays, strict=True):
        g_arr = np.asarray(g)
        l_arr = np.asarray(l)

        if np.issubdtype(g_arr.dtype, np.floating) and np.issubdtype(
            l_arr.dtype, np.floating
        ):
            delta = l_arr.astype(np.float64) - g_arr.astype(np.float64)
            deltas.append(delta.reshape(-1))

    if not deltas:
        return np.zeros(1, dtype=np.float64)

    return np.concatenate(deltas, axis=0)


def _krum_select(flat_updates: list[np.ndarray], f: int, k: int) -> list[int]:
    n = len(flat_updates)
    if n < 2:
        raise ValueError("Need at least 2 clients for Krum.")

    if n <= f + 2:
        raise ValueError(f"Krum requires n > f + 2, got n={n}, f={f}.")

    distances = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(flat_updates[i] - flat_updates[j]) ** 2
            distances[i, j] = d
            distances[j, i] = d

    scores = []
    m = n - f - 2
    for i in range(n):
        dists = np.sort(distances[i])
        score = np.sum(dists[1 : m + 1])
        scores.append(score)

    scores = np.array(scores)
    selected_indices = np.argsort(scores)[:k]
    return selected_indices.tolist()


def _merge_metrics(*records: MetricRecord | None) -> MetricRecord:
    merged: dict[str, Any] = {}
    for rec in records:
        if rec is not None:
            merged.update(dict(rec))
    return MetricRecord(merged)


def _build_aggregation_defense(
    aggregation_type: str,
    *,
    seed: int,
    kwargs: Mapping[str, Any] | None,
) -> DefenseBase:
    normalized = _normalize_name(aggregation_type)
    extra = _normalize_kwargs(kwargs)

    config = DefenseConfig(
        defense_type=normalized if normalized else "none",
        seed=seed,
        extra=extra,
    )

    if normalized in {"", "none", "identity"}:
        return IdentityDefense(config)

    if normalized in {"norm_clipping", "normclipping"}:
        return NormClippingDefense(config)

    if normalized in {"trimmed_mean", "trimmedmean"}:
        return TrimmedMeanDefense(config)

    if normalized in {"krum"}:
        return KrumDefense(config)

    raise ValueError(
        f"Unsupported aggregation_type={aggregation_type!r}. "
        "Supported: 'none', 'norm_clipping', 'trimmed_mean', 'krum'."
    )


@dataclass
class DefensePipelineConfig:
    """Unified config for client defense + server detection + server aggregation."""

    seed: int = 42

    client_defense_type: str = "none"
    detection_type: str = "none"
    aggregation_type: str = "none"

    client_defense_kwargs: dict[str, Any] = field(default_factory=dict)
    detection_kwargs: dict[str, Any] = field(default_factory=dict)
    aggregation_kwargs: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if int(self.seed) < 0:
            raise ValueError("seed must be non-negative.")

        if not str(self.client_defense_type).strip():
            raise ValueError("client_defense_type must not be empty.")
        if not str(self.detection_type).strip():
            raise ValueError("detection_type must not be empty.")
        if not str(self.aggregation_type).strip():
            raise ValueError("aggregation_type must not be empty.")


class DefensePipelineFedAvg(FedAvg):
    """FedAvg with detection -> aggregation executed in one strategy."""

    def __init__(
        self,
        *,
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
        pipeline_config: DefensePipelineConfig,
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

        self.pipeline_config = pipeline_config
        self.current_arrays: ArrayRecord | None = None

    @classmethod
    def from_strategy(
        cls,
        strategy: FedAvg,
        pipeline_config: DefensePipelineConfig,
    ) -> "DefensePipelineFedAvg":
        return cls(
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
            pipeline_config=pipeline_config,
        )

    def summary(self) -> None:
        super().summary()
        from logging import INFO
        from flwr.common.logger import log

        log(INFO, "\t├──> Defense pipeline settings:")
        log(INFO, "\t│\t├── client_defense: %s", self.pipeline_config.client_defense_type)
        log(INFO, "\t│\t├── detection: %s", self.pipeline_config.detection_type)
        log(INFO, "\t│\t└── aggregation: %s", self.pipeline_config.aggregation_type)

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        self.current_arrays = arrays.copy()
        return super().configure_train(server_round, arrays, config, grid)

    def _apply_detection(
        self, replies: list[Message]
    ) -> tuple[list[Message], MetricRecord]:
        detection_type = _normalize_name(self.pipeline_config.detection_type)
        extra = _normalize_kwargs(self.pipeline_config.detection_kwargs)

        if detection_type in {"", "none", "identity"}:
            return replies, MetricRecord()

        if detection_type not in {
            "anomaly_detection",
            "anomaly_detection_defense",
            "update_anomaly",
            "update_detector",
            "anomaly-detection",
        }:
            raise ValueError(
                f"Combined pipeline currently supports anomaly_detection only, got "
                f"{self.pipeline_config.detection_type!r}."
            )

        norm_z_threshold = float(
            extra.get("norm_z_threshold", extra.get("z_threshold", 3.5))
        )
        cosine_floor = float(extra.get("cosine_floor", 0.0))
        min_clients = int(extra.get("min_clients", extra.get("min_total_clients", 2)))
        min_kept_clients = int(extra.get("min_kept_clients", min_clients))
        top_k = extra.get("top_k", extra.get("max_reject_count", None))
        max_reject_fraction = float(extra.get("max_reject_fraction", 0.5))
        enable_filter = bool(extra.get("enable_filter", True))

        if self.current_arrays is None:
            return replies, MetricRecord(
                {
                    "defense-detect-skipped": 1,
                    "defense-detect-skip-reason": "current_arrays_missing",
                }
            )

        if len(replies) < min_clients:
            return replies, MetricRecord(
                {
                    "defense-detect-skipped": 1,
                    "defense-detect-skip-reason": "too_few_clients",
                    "defense-detect-total-clients": int(len(replies)),
                    "defense-detect-min-clients": int(min_clients),
                }
            )

        global_vector = _state_dict_to_vector(self.current_arrays.to_torch_state_dict())
        if global_vector.size == 0:
            return replies, MetricRecord(
                {
                    "defense-detect-skipped": 1,
                    "defense-detect-skip-reason": "empty_global_vector",
                }
            )

        scored_replies: list[Message] = []
        vectors: list[np.ndarray] = []
        skipped = 0

        for msg in replies:
            try:
                content = RecordDict(msg.content)
                local_arrays = content[self.arrayrecord_key]
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
            return replies, MetricRecord(
                {
                    "defense-detect-skipped": 1,
                    "defense-detect-skip-reason": "no_valid_vectors",
                    "defense-detect-skip-count": int(skipped),
                }
            )

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

        suspicious = (norm_z > norm_z_threshold) | (cosine < cosine_floor)

        total = len(scored_replies)

        if top_k is not None:
            max_rejects = int(top_k)
        else:
            max_rejects = int(np.floor(total * max_reject_fraction))

        max_rejects = max(0, min(max_rejects, total))
        max_rejects = min(max_rejects, max(0, total - min_kept_clients))
        target_keep = max(min_kept_clients, total - max_rejects)
        target_keep = min(target_keep, total)

        if enable_filter:
            keep_indices = np.where(~suspicious)[0].tolist()
            if len(keep_indices) < target_keep:
                keep_indices = np.argsort(score)[:target_keep].tolist()
            keep_indices = sorted(set(int(i) for i in keep_indices))
        else:
            keep_indices = list(range(total))

        filtered_replies = [scored_replies[i] for i in keep_indices]

        metrics = MetricRecord(
            {
                "defense-detect-type": detection_type,
                "defense-detect-total-clients": int(total),
                "defense-detect-kept-clients": int(len(filtered_replies)),
                "defense-detect-filtered-clients": int(total - len(filtered_replies)),
                "defense-detect-filter-ratio": (
                    float((total - len(filtered_replies)) / total) if total > 0 else 0.0
                ),
                "defense-detect-suspicious-clients": int(np.sum(suspicious)),
                "defense-detect-norm-z-threshold": float(norm_z_threshold),
                "defense-detect-cosine-floor": float(cosine_floor),
                "defense-detect-min-clients": int(min_clients),
                "defense-detect-min-kept-clients": int(min_kept_clients),
                "defense-detect-max-reject-count": int(max_rejects),
                "defense-detect-mean-update-norm": (
                    float(np.mean(norms)) if len(norms) > 0 else 0.0
                ),
                "defense-detect-max-update-norm": (
                    float(np.max(norms)) if len(norms) > 0 else 0.0
                ),
                "defense-detect-mean-norm-z": (
                    float(np.mean(norm_z)) if len(norm_z) > 0 else 0.0
                ),
                "defense-detect-max-norm-z": (
                    float(np.max(norm_z)) if len(norm_z) > 0 else 0.0
                ),
                "defense-detect-mean-cosine": (
                    float(np.mean(cosine)) if len(cosine) > 0 else 0.0
                ),
                "defense-detect-min-cosine": (
                    float(np.min(cosine)) if len(cosine) > 0 else 0.0
                ),
                "defense-detect-mean-score": (
                    float(np.mean(score)) if len(score) > 0 else 0.0
                ),
                "defense-detect-max-score": (
                    float(np.max(score)) if len(score) > 0 else 0.0
                ),
                "defense-detect-skip-count": int(skipped),
                "defense-detect-enable-filter": int(enable_filter),
            }
        )

        return filtered_replies, metrics

    def _apply_aggregation(
        self, replies: list[Message]
    ) -> tuple[ArrayRecord | None, MetricRecord]:
        aggregation_type = _normalize_name(self.pipeline_config.aggregation_type)
        extra = _normalize_kwargs(self.pipeline_config.aggregation_kwargs)

        if not replies:
            return None, MetricRecord()

        if self.current_arrays is None:
            raise RuntimeError(
                "Current global model is not available. "
                "Make sure configure_train() was called first."
            )

        array_keys = list(self.current_arrays.keys())
        global_ndarrays = self.current_arrays.to_numpy_ndarrays(keep_input=False)

        if aggregation_type in {"", "none", "identity"}:
            contents = [RecordDict(msg.content) for msg in replies]
            arrays = aggregate_arrayrecords(contents, self.weighted_by_key)
            metrics = (
                self.train_metrics_aggr_fn(contents, self.weighted_by_key)
                if self.train_metrics_aggr_fn is not None
                else MetricRecord()
            )
            if metrics is None:
                metrics = MetricRecord()

            metrics["defense-agg-type"] = "fedavg"
            metrics["defense-agg-total-clients"] = int(len(replies))
            return arrays, metrics

        if aggregation_type in {"norm_clipping", "normclipping"}:
            clip_norm = float(extra.get("clip_norm", extra.get("clip", 1.0)))
            if clip_norm <= 0.0:
                raise ValueError("clip_norm must be positive.")

            clipped_contents: list[RecordDict] = []
            pre_norms: list[float] = []
            post_norms: list[float] = []
            clipped_clients = 0

            for msg in replies:
                content = RecordDict(msg.content)
                local_arrays = content[self.arrayrecord_key]
                local_ndarrays = local_arrays.to_numpy_ndarrays(keep_input=False)

                clipped_ndarrays, pre_norm, post_norm, was_clipped = _clip_client_update(
                    global_ndarrays=global_ndarrays,
                    local_ndarrays=local_ndarrays,
                    clip_norm=clip_norm,
                )

                pre_norms.append(pre_norm)
                post_norms.append(post_norm)
                clipped_clients += int(was_clipped)

                clipped_arrays = ArrayRecord(
                    {
                        key: Array(np.asarray(arr))
                        for key, arr in zip(array_keys, clipped_ndarrays, strict=True)
                    }
                )
                content[self.arrayrecord_key] = clipped_arrays
                clipped_contents.append(content)

            arrays = aggregate_arrayrecords(clipped_contents, self.weighted_by_key)
            metrics = (
                self.train_metrics_aggr_fn(clipped_contents, self.weighted_by_key)
                if self.train_metrics_aggr_fn is not None
                else MetricRecord()
            )
            if metrics is None:
                metrics = MetricRecord()

            total_clients = len(replies)
            metrics["defense-agg-type"] = "norm_clipping"
            metrics["defense-agg-clip-norm"] = float(clip_norm)
            metrics["defense-agg-total-clients"] = int(total_clients)
            metrics["defense-agg-clipped-clients"] = int(clipped_clients)
            metrics["defense-agg-clipped-ratio"] = (
                float(clipped_clients / total_clients) if total_clients > 0 else 0.0
            )
            metrics["defense-agg-avg-update-norm"] = (
                float(np.mean(pre_norms)) if pre_norms else 0.0
            )
            metrics["defense-agg-max-update-norm"] = (
                float(np.max(pre_norms)) if pre_norms else 0.0
            )
            metrics["defense-agg-avg-post-norm"] = (
                float(np.mean(post_norms)) if post_norms else 0.0
            )
            return arrays, metrics

        if aggregation_type in {"trimmed_mean", "trimmedmean"}:
            trim_ratio = float(extra.get("trim_ratio", 0.2))
            trim_k = extra.get("trim_k", None)

            client_ndarrays_list: list[list[np.ndarray]] = []
            client_update_norms: list[float] = []

            for msg in replies:
                content = RecordDict(msg.content)
                local_arrays = content[self.arrayrecord_key]
                local_ndarrays = local_arrays.to_numpy_ndarrays(keep_input=False)

                if len(local_ndarrays) != len(global_ndarrays):
                    raise ValueError(
                        "Client update structure does not match the global model structure."
                    )

                client_ndarrays_list.append(local_ndarrays)

                float_deltas: list[np.ndarray] = []
                for g_arr, l_arr in zip(global_ndarrays, local_ndarrays, strict=True):
                    g = np.asarray(g_arr)
                    l = np.asarray(l_arr)
                    if np.issubdtype(g.dtype, np.floating) and np.issubdtype(
                        l.dtype, np.floating
                    ):
                        float_deltas.append(
                            np.asarray(l, dtype=np.float64)
                            - np.asarray(g, dtype=np.float64)
                        )

                if float_deltas:
                    flat_delta = np.concatenate(
                        [delta.reshape(-1) for delta in float_deltas], axis=0
                    )
                    client_update_norms.append(float(np.linalg.norm(flat_delta)))
                else:
                    client_update_norms.append(0.0)

            num_clients = len(client_ndarrays_list)
            if trim_k is not None:
                trim_k_int = int(trim_k)
            else:
                trim_k_int = int(np.floor(trim_ratio * num_clients))

            if trim_k_int < 0:
                raise ValueError("trim_k must be non-negative.")
            if 2 * trim_k_int >= num_clients:
                raise ValueError(
                    f"trim_k={trim_k_int} is too large for {num_clients} participating clients. "
                    f"Need 2 * trim_k < num_clients."
                )

            aggregated_ndarrays: list[np.ndarray] = []
            for tensor_idx, global_arr in enumerate(global_ndarrays):
                g = np.asarray(global_arr)
                tensor_values = [
                    np.asarray(client[tensor_idx]) for client in client_ndarrays_list
                ]

                if not np.issubdtype(g.dtype, np.floating):
                    aggregated_ndarrays.append(g)
                    continue

                aggregated_tensor = _build_trimmed_mean(
                    client_tensors=tensor_values,
                    trim_k=trim_k_int,
                )
                aggregated_ndarrays.append(np.asarray(aggregated_tensor, dtype=g.dtype))

            aggregated_arrays = ArrayRecord(
                {
                    key: Array(np.asarray(arr))
                    for key, arr in zip(array_keys, aggregated_ndarrays, strict=True)
                }
            )

            reply_contents = [RecordDict(msg.content) for msg in replies]
            metrics = (
                self.train_metrics_aggr_fn(reply_contents, self.weighted_by_key)
                if self.train_metrics_aggr_fn is not None
                else MetricRecord()
            )
            if metrics is None:
                metrics = MetricRecord()

            metrics["defense-agg-type"] = "trimmed_mean"
            metrics["defense-agg-trim-ratio"] = float(trim_ratio)
            metrics["defense-agg-trim-k"] = int(trim_k_int)
            metrics["defense-agg-total-clients"] = int(num_clients)
            metrics["defense-agg-avg-update-norm"] = (
                float(np.mean(client_update_norms)) if client_update_norms else 0.0
            )
            metrics["defense-agg-max-update-norm"] = (
                float(np.max(client_update_norms)) if client_update_norms else 0.0
            )

            return aggregated_arrays, metrics

        if aggregation_type in {"krum"}:
            num_malicious = int(extra.get("num_malicious", 1))
            krum_k = int(extra.get("krum_k", 1))

            client_ndarrays_list: list[list[np.ndarray]] = []
            flat_updates: list[np.ndarray] = []

            for msg in replies:
                content = RecordDict(msg.content)
                local_arrays = content[self.arrayrecord_key]
                local_ndarrays = local_arrays.to_numpy_ndarrays(keep_input=False)

                client_ndarrays_list.append(local_ndarrays)
                flat = _flatten_update(global_ndarrays, local_ndarrays)
                flat_updates.append(flat)

            n = len(flat_updates)
            f = num_malicious
            k = min(krum_k, n)

            selected_indices = _krum_select(flat_updates, f=f, k=k)
            selected_updates = [client_ndarrays_list[i] for i in selected_indices]

            aggregated_ndarrays: list[np.ndarray] = []
            for tensor_idx in range(len(global_ndarrays)):
                tensors = [np.asarray(client[tensor_idx]) for client in selected_updates]

                if not np.issubdtype(global_ndarrays[tensor_idx].dtype, np.floating):
                    aggregated_ndarrays.append(global_ndarrays[tensor_idx])
                    continue

                stacked = np.stack([t.astype(np.float64) for t in tensors], axis=0)
                mean_tensor = stacked.mean(axis=0)
                aggregated_ndarrays.append(
                    mean_tensor.astype(global_ndarrays[tensor_idx].dtype)
                )

            aggregated_arrays = ArrayRecord(
                {
                    key: Array(np.asarray(arr))
                    for key, arr in zip(array_keys, aggregated_ndarrays, strict=True)
                }
            )

            reply_contents = [RecordDict(msg.content) for msg in replies]
            metrics = (
                self.train_metrics_aggr_fn(reply_contents, self.weighted_by_key)
                if self.train_metrics_aggr_fn is not None
                else MetricRecord()
            )
            if metrics is None:
                metrics = MetricRecord()

            metrics["defense-agg-type"] = "krum"
            metrics["defense-agg-total-clients"] = int(n)
            metrics["defense-agg-selected-clients"] = int(len(selected_indices))
            metrics["defense-agg-krum-f"] = int(f)
            metrics["defense-agg-krum-k"] = int(k)

            return aggregated_arrays, metrics

        raise ValueError(
            f"Unsupported aggregation_type={self.pipeline_config.aggregation_type!r}. "
            "Supported: 'none', 'norm_clipping', 'trimmed_mean', 'krum'."
        )

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        valid_replies, _ = self._check_and_log_replies(replies, is_train=True)
        if not valid_replies:
            return None, None

        filtered_replies, detection_metrics = self._apply_detection(valid_replies)
        arrays, aggregation_metrics = self._apply_aggregation(filtered_replies)

        return arrays, _merge_metrics(detection_metrics, aggregation_metrics)


class DefensePipeline:
    """Unified pipeline object.

    Use it as the single experiment entry for:
      client defense -> detection -> aggregation
    """

    def __init__(self, config: DefensePipelineConfig) -> None:
        self.config = config
        self.config.validate()

        self.config.client_defense_type = _normalize_name(self.config.client_defense_type)
        self.config.detection_type = _normalize_name(self.config.detection_type)
        self.config.aggregation_type = _normalize_name(self.config.aggregation_type)

        self.config.client_defense_kwargs = _normalize_kwargs(
            self.config.client_defense_kwargs
        )
        self.config.detection_kwargs = _normalize_kwargs(self.config.detection_kwargs)
        self.config.aggregation_kwargs = _normalize_kwargs(self.config.aggregation_kwargs)

    def __repr__(self) -> str:
        return (
            "DefensePipeline("
            f"client_defense_type={self.config.client_defense_type!r}, "
            f"detection_type={self.config.detection_type!r}, "
            f"aggregation_type={self.config.aggregation_type!r}, "
            f"seed={self.config.seed}, "
            f"client_defense_kwargs={self.config.client_defense_kwargs}, "
            f"detection_kwargs={self.config.detection_kwargs}, "
            f"aggregation_kwargs={self.config.aggregation_kwargs})"
        )

    def build_client_defense(self):
        return build_client_defense(
            self.config.client_defense_type,
            seed=self.config.seed,
            **self.config.client_defense_kwargs,
        )

    def build_strategy(self, strategy: FedAvg) -> FedAvg:
        """Build a server strategy with the correct defense composition."""
        if not isinstance(strategy, FedAvg):
            raise TypeError(
                f"DefensePipeline only supports FedAvg-compatible strategies, got {type(strategy)!r}."
            )

        detection_type = self.config.detection_type
        aggregation_type = self.config.aggregation_type

        detection_is_none = detection_type in {"", "none", "identity"}
        aggregation_is_none = aggregation_type in {"", "none", "identity"}

        if detection_is_none and aggregation_is_none:
            return strategy

        if not detection_is_none and aggregation_is_none:
            detection = build_detection(
                detection_type=detection_type,
                seed=self.config.seed,
                **self.config.detection_kwargs,
            )
            return detection.apply(strategy)

        if detection_is_none and not aggregation_is_none:
            defense = _build_aggregation_defense(
                aggregation_type,
                seed=self.config.seed,
                kwargs=self.config.aggregation_kwargs,
            )
            return defense.apply(strategy)

        return DefensePipelineFedAvg.from_strategy(
            strategy,
            pipeline_config=self.config,
        )

    def apply(self, strategy: FedAvg) -> FedAvg:
        """Alias for build_strategy()."""
        return self.build_strategy(strategy)


def build_defense_pipeline(
    *,
    client_defense_type: str = "none",
    detection_type: str = "none",
    aggregation_type: str | None = None,
    defense_type: str | None = None,
    seed: int = 42,
    client_defense_kwargs: Mapping[str, Any] | None = None,
    detection_kwargs: Mapping[str, Any] | None = None,
    aggregation_kwargs: Mapping[str, Any] | None = None,
    defense_kwargs: Mapping[str, Any] | None = None,
) -> DefensePipeline:
    """Unified pipeline factory.

    Notes:
        - `defense_type` is accepted as an alias of `aggregation_type`
          for compatibility with your current config naming.
        - `defense_kwargs` is accepted as an alias of `aggregation_kwargs`.
    """
    resolved_aggregation_type = aggregation_type
    if resolved_aggregation_type is None:
        resolved_aggregation_type = defense_type if defense_type is not None else "none"

    resolved_aggregation_kwargs = aggregation_kwargs
    if resolved_aggregation_kwargs is None and defense_kwargs is not None:
        resolved_aggregation_kwargs = defense_kwargs

    config = DefensePipelineConfig(
        seed=seed,
        client_defense_type=client_defense_type,
        detection_type=detection_type,
        aggregation_type=resolved_aggregation_type,
        client_defense_kwargs=dict(client_defense_kwargs or {}),
        detection_kwargs=dict(detection_kwargs or {}),
        aggregation_kwargs=dict(resolved_aggregation_kwargs or {}),
    )
    return DefensePipeline(config)


def build_defense_pipeline_from_run_config(
    run_config: Mapping[str, Any],
    *,
    seed: int | None = None,
) -> DefensePipeline:
    """Convenience helper for server.py / client.py.

    It keeps config parsing in one place, so adding new defense options later
    usually only requires updating this module and the factory registries.
    """
    rc = dict(run_config)

    resolved_seed = int(seed if seed is not None else rc.get("seed", 42))

    client_defense_type = str(
        rc.get("client-defense", rc.get("client_defense", "none"))
    ).lower()
    detection_type = str(rc.get("detection", "none")).lower()
    aggregation_type = str(rc.get("defense", "none")).lower()

    client_defense_kwargs = _prefixed_kwargs(rc, "client-defense-")
    detection_kwargs = _prefixed_kwargs(rc, "detection-")
    aggregation_kwargs = _prefixed_kwargs(rc, "defense-")

    # Backward compatibility for older top-level keys.
    if "clip-norm" in rc and "clip_norm" not in aggregation_kwargs:
        aggregation_kwargs["clip_norm"] = rc["clip-norm"]

    return build_defense_pipeline(
        client_defense_type=client_defense_type,
        detection_type=detection_type,
        aggregation_type=aggregation_type,
        seed=resolved_seed,
        client_defense_kwargs=client_defense_kwargs,
        detection_kwargs=detection_kwargs,
        aggregation_kwargs=aggregation_kwargs,
    )


__all__ = [
    "DefensePipelineConfig",
    "DefensePipelineFedAvg",
    "DefensePipeline",
    "build_defense_pipeline",
    "build_defense_pipeline_from_run_config",
]