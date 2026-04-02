"""Server-side clustering-based detection defense for client updates.

Pure NumPy implementation (no scikit-learn dependency).

Main idea:
- build richer per-client features from update statistics
- run lightweight 2-means clustering
- treat the small/high-score cluster as suspicious
- fall back to score-percentile filtering when clustering is unstable

This module is self-contained and can be used both by:
1) the pipeline helper function
2) a FedAvg wrapper for build_detection()
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg

from .base import DetectionBase, DetectionConfig, DetectionReport

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


def _standardize_features(features: np.ndarray) -> np.ndarray:
    if features.size == 0:
        return features.astype(np.float64, copy=False)

    mean = np.mean(features, axis=0, keepdims=True)
    std = np.std(features, axis=0, keepdims=True)
    std = np.where(std < _EPS, 1.0, std)
    return (features - mean) / std


def _kmeans_2_numpy(
    X: np.ndarray,
    *,
    seed: int = 42,
    max_iter: int = 100,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Lightweight 2-means clustering.

    Returns:
        labels: shape (n,)
        centers: shape (2, d)
        inertia: float
    """
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]

    if n < 2:
        labels = np.zeros(n, dtype=int)
        centers = np.zeros((2, X.shape[1]), dtype=np.float64)
        return labels, centers, 0.0

    rng = np.random.default_rng(seed)

    # Deterministic-ish robust init:
    # choose one random point and one farthest point from it
    i0 = int(rng.integers(0, n))
    d0 = np.sum((X - X[i0]) ** 2, axis=1)
    i1 = int(np.argmax(d0))
    if i1 == i0:
        i1 = (i0 + 1) % n

    centers = np.stack([X[i0], X[i1]], axis=0).astype(np.float64, copy=False)
    labels = np.zeros(n, dtype=int)

    for _ in range(max_iter):
        dist0 = np.sum((X - centers[0]) ** 2, axis=1)
        dist1 = np.sum((X - centers[1]) ** 2, axis=1)
        new_labels = np.where(dist1 < dist0, 1, 0).astype(int)

        # If a cluster becomes empty, re-seed it to a far point
        for cid in (0, 1):
            if not np.any(new_labels == cid):
                farthest = int(np.argmax(np.minimum(dist0, dist1)))
                new_labels[farthest] = cid
                centers[cid] = X[farthest]

        if np.array_equal(new_labels, labels):
            labels = new_labels
            break

        labels = new_labels
        for cid in (0, 1):
            members = X[labels == cid]
            if len(members) > 0:
                centers[cid] = members.mean(axis=0)

    # Final inertia
    dist0 = np.sum((X - centers[0]) ** 2, axis=1)
    dist1 = np.sum((X - centers[1]) ** 2, axis=1)
    inertia = float(np.sum(np.minimum(dist0, dist1)))
    return labels, centers, inertia


def _silhouette_score_numpy(X: np.ndarray, labels: np.ndarray) -> float:
    """Simple silhouette score for 2 clusters. Safe for tiny n."""
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels, dtype=int).reshape(-1)
    n = X.shape[0]

    unique = np.unique(labels)
    if n < 3 or unique.size < 2:
        return 0.0

    # Pairwise distances
    diff = X[:, None, :] - X[None, :, :]
    D = np.sqrt(np.sum(diff * diff, axis=2))

    sils: list[float] = []
    for i in range(n):
        same = labels == labels[i]
        same_count = int(np.sum(same)) - 1
        if same_count <= 0:
            continue

        a = float(np.sum(D[i, same]) / max(1, same_count))

        other = labels != labels[i]
        if not np.any(other):
            continue
        b = float(np.mean(D[i, other]))

        denom = max(a, b, _EPS)
        sils.append((b - a) / denom)

    if not sils:
        return 0.0
    return float(np.mean(sils))


def _build_feature_matrix(
    *,
    norm_z: np.ndarray,
    cosine: np.ndarray,
    score: np.ndarray,
    deltas: np.ndarray | None = None,
) -> np.ndarray:
    """Construct richer features for clustering."""
    norm_z = np.asarray(norm_z, dtype=np.float64).reshape(-1)
    cosine = np.asarray(cosine, dtype=np.float64).reshape(-1)
    score = np.asarray(score, dtype=np.float64).reshape(-1)

    base = np.column_stack(
        [
            norm_z,
            1.0 - np.clip(cosine, -1.0, 1.0),
            score,
        ]
    )

    if deltas is None:
        return _standardize_features(base)

    deltas = np.asarray(deltas, dtype=np.float64)
    if deltas.ndim != 2 or deltas.shape[0] != norm_z.shape[0]:
        return _standardize_features(base)

    abs_deltas = np.abs(deltas)
    l2 = np.linalg.norm(deltas, axis=1)
    l1 = np.mean(abs_deltas, axis=1)
    linf = np.max(abs_deltas, axis=1)

    median_center = np.median(deltas, axis=0)
    median_dev = np.linalg.norm(deltas - median_center.reshape(1, -1), axis=1)
    median_cos = np.array(
        [_cosine_similarity(delta, median_center) for delta in deltas],
        dtype=np.float64,
    )

    # A few extra robust shape descriptors
    sign_balance = np.mean(np.sign(deltas), axis=1)

    extra_features = np.column_stack(
        [
            np.log1p(l2),
            np.log1p(median_dev),
            1.0 - np.clip(median_cos, -1.0, 1.0),
            l1 / (l2 + _EPS),
            linf / (l2 + _EPS),
            sign_balance,
        ]
    )

    features = np.column_stack([base, extra_features])
    return _standardize_features(features)


def _score_fallback_report(
    *,
    detection_type: str,
    server_round: int,
    total_clients: int,
    score: np.ndarray,
    percentile: float,
    min_kept_clients: int,
    max_reject_fraction: float,
    enable_filter: bool,
    skipped: int,
    extra: dict[str, Any],
) -> DetectionReport:
    threshold = float(np.percentile(score, percentile))
    suspicious = score > threshold

    max_rejects = int(np.floor(total_clients * max_reject_fraction))
    max_rejects = min(max_rejects, max(0, total_clients - min_kept_clients))

    suspicious_idx = np.where(suspicious)[0]
    if len(suspicious_idx) > max_rejects:
        order = suspicious_idx[np.argsort(score[suspicious_idx])[::-1]]
        rejected = order[:max_rejects]
        suspicious = np.zeros(total_clients, dtype=bool)
        suspicious[rejected] = True

    if enable_filter:
        keep_indices = np.where(~suspicious)[0].tolist()
        if len(keep_indices) < min_kept_clients:
            keep_indices = np.argsort(score)[:min_kept_clients].tolist()
            suspicious = np.ones(total_clients, dtype=bool)
            suspicious[keep_indices] = False
    else:
        keep_indices = list(range(total_clients))
        suspicious = np.zeros(total_clients, dtype=bool)

    return DetectionReport(
        detection_type=detection_type,
        server_round=server_round,
        total_clients=total_clients,
        kept_indices=[int(i) for i in keep_indices],
        scores=[float(x) for x in score.tolist()],
        threshold=threshold,
        suspicious_mask=[bool(x) for x in suspicious.tolist()],
        suspicious_count=int(np.sum(suspicious)),
        skip_count=int(skipped),
        extra={
            **extra,
            "defense-detect-clustering-fallback": 1,
            "defense-detect-silhouette": 0.0,
            "defense-detect-inertia": 0.0,
            "defense-detect-suspicious-cluster-id": -1,
            "defense-detect-cluster-0-size": 0,
            "defense-detect-cluster-1-size": 0,
            "defense-detect-cluster-0-mean-score": 0.0,
            "defense-detect-cluster-1-mean-score": 0.0,
            "defense-detect-cluster-labels": [],
        },
    )


def build_clustering_report(
    *,
    detection_type: str,
    server_round: int,
    total_clients: int,
    norm_z: np.ndarray,
    cosine: np.ndarray,
    score: np.ndarray,
    deltas: np.ndarray | None = None,
    percentile: float = 80.0,
    min_kept_clients: int = 2,
    max_reject_fraction: float = 0.5,
    enable_filter: bool = True,
    seed: int = 42,
    skipped: int = 0,
    min_silhouette: float = 0.05,
    cluster_score_gap: float = 0.15,
) -> DetectionReport:
    """Return a standardized clustering-based detection report."""
    norm_z = np.asarray(norm_z, dtype=np.float64).reshape(-1)
    cosine = np.asarray(cosine, dtype=np.float64).reshape(-1)
    score = np.asarray(score, dtype=np.float64).reshape(-1)

    if total_clients <= 0 or score.size == 0:
        return DetectionReport(
            detection_type=detection_type,
            server_round=server_round,
            total_clients=total_clients,
            kept_indices=[],
            scores=[],
            threshold=0.0,
            suspicious_mask=[],
            suspicious_count=0,
            skip_count=int(skipped),
            extra={
                "defense-detect-clustering-fallback": 1,
                "defense-detect-silhouette": 0.0,
                "defense-detect-inertia": 0.0,
                "defense-detect-suspicious-cluster-id": -1,
            },
        )

    if not (len(norm_z) == len(cosine) == len(score) == total_clients):
        raise ValueError(
            "norm_z, cosine, score, and total_clients must have matching lengths."
        )

    base_extra: dict[str, Any] = {
        "defense-detect-percentile": float(percentile),
        "defense-detect-enable-filter": int(enable_filter),
        "defense-detect-skip-count": int(skipped),
        "defense-detect-mean-score": float(np.mean(score)),
        "defense-detect-max-score": float(np.max(score)),
        "defense-detect-min-score": float(np.min(score)),
        "defense-detect-mean-norm-z": float(np.mean(norm_z)),
        "defense-detect-max-norm-z": float(np.max(norm_z)),
        "defense-detect-mean-cosine": float(np.mean(cosine)),
        "defense-detect-min-cosine": float(np.min(cosine)),
    }

    if total_clients < 4:
        return _score_fallback_report(
            detection_type=detection_type,
            server_round=server_round,
            total_clients=total_clients,
            score=score,
            percentile=percentile,
            min_kept_clients=min_kept_clients,
            max_reject_fraction=max_reject_fraction,
            enable_filter=enable_filter,
            skipped=skipped,
            extra=base_extra,
        )

    try:
        features = _build_feature_matrix(
            norm_z=norm_z,
            cosine=cosine,
            score=score,
            deltas=deltas,
        )

        if np.unique(features, axis=0).shape[0] < 2:
            return _score_fallback_report(
                detection_type=detection_type,
                server_round=server_round,
                total_clients=total_clients,
                score=score,
                percentile=percentile,
                min_kept_clients=min_kept_clients,
                max_reject_fraction=max_reject_fraction,
                enable_filter=enable_filter,
                skipped=skipped,
                extra=base_extra,
            )

        labels, centers, inertia = _kmeans_2_numpy(features, seed=seed, max_iter=100)
        if np.unique(labels).size < 2:
            return _score_fallback_report(
                detection_type=detection_type,
                server_round=server_round,
                total_clients=total_clients,
                score=score,
                percentile=percentile,
                min_kept_clients=min_kept_clients,
                max_reject_fraction=max_reject_fraction,
                enable_filter=enable_filter,
                skipped=skipped,
                extra=base_extra,
            )

        sil = float(_silhouette_score_numpy(features, labels))
        if sil < min_silhouette:
            return _score_fallback_report(
                detection_type=detection_type,
                server_round=server_round,
                total_clients=total_clients,
                score=score,
                percentile=percentile,
                min_kept_clients=min_kept_clients,
                max_reject_fraction=max_reject_fraction,
                enable_filter=enable_filter,
                skipped=skipped,
                extra={
                    **base_extra,
                    "defense-detect-inertia": float(inertia),
                    "defense-detect-silhouette": sil,
                    "defense-detect-cluster-labels": [int(x) for x in labels.tolist()],
                    "defense-detect-cluster-0-size": int(np.sum(labels == 0)),
                    "defense-detect-cluster-1-size": int(np.sum(labels == 1)),
                    "defense-detect-cluster-0-mean-score": (
                        float(score[labels == 0].mean()) if np.any(labels == 0) else 0.0
                    ),
                    "defense-detect-cluster-1-mean-score": (
                        float(score[labels == 1].mean()) if np.any(labels == 1) else 0.0
                    ),
                    "defense-detect-suspicious-cluster-id": -1,
                    "defense-detect-clustering-fallback": 1,
                },
            )

        cluster_sizes = np.bincount(labels, minlength=2).astype(np.int64)
        cluster_score_means = np.array(
            [
                float(score[labels == cid].mean()) if np.any(labels == cid) else -np.inf
                for cid in range(2)
            ],
            dtype=np.float64,
        )

        # Higher mean score + slightly smaller size -> more suspicious
        cluster_rank = cluster_score_means - cluster_score_gap * (
            cluster_sizes / max(total_clients, 1)
        )
        suspicious_cluster = int(np.argmax(cluster_rank))
        suspicious = labels == suspicious_cluster

        max_rejects = int(np.floor(total_clients * max_reject_fraction))
        max_rejects = min(max_rejects, max(0, total_clients - min_kept_clients))

        suspicious_idx = np.where(suspicious)[0]
        if len(suspicious_idx) > max_rejects:
            order = suspicious_idx[np.argsort(score[suspicious_idx])[::-1]]
            rejected = order[:max_rejects]
            suspicious = np.zeros(total_clients, dtype=bool)
            suspicious[rejected] = True

        if enable_filter:
            keep_indices = np.where(~suspicious)[0].tolist()
            if len(keep_indices) < min_kept_clients:
                keep_indices = np.argsort(score)[:min_kept_clients].tolist()
                suspicious = np.ones(total_clients, dtype=bool)
                suspicious[keep_indices] = False
        else:
            keep_indices = list(range(total_clients))
            suspicious = np.zeros(total_clients, dtype=bool)

        threshold = float(cluster_score_means[suspicious_cluster])
        if not np.isfinite(threshold):
            threshold = float(np.percentile(score, percentile))

        return DetectionReport(
            detection_type=detection_type,
            server_round=server_round,
            total_clients=total_clients,
            kept_indices=[int(i) for i in keep_indices],
            scores=[float(x) for x in score.tolist()],
            threshold=threshold,
            suspicious_mask=[bool(x) for x in suspicious.tolist()],
            suspicious_count=int(np.sum(suspicious)),
            skip_count=int(skipped),
            extra={
                **base_extra,
                "defense-detect-inertia": float(inertia),
                "defense-detect-silhouette": sil,
                "defense-detect-cluster-labels": [int(x) for x in labels.tolist()],
                "defense-detect-cluster-0-size": int(cluster_sizes[0]),
                "defense-detect-cluster-1-size": int(cluster_sizes[1]),
                "defense-detect-cluster-0-mean-score": (
                    float(cluster_score_means[0]) if np.isfinite(cluster_score_means[0]) else 0.0
                ),
                "defense-detect-cluster-1-mean-score": (
                    float(cluster_score_means[1]) if np.isfinite(cluster_score_means[1]) else 0.0
                ),
                "defense-detect-suspicious-cluster-id": int(suspicious_cluster),
                "defense-detect-clustering-fallback": 0,
            },
        )

    except Exception:
        return _score_fallback_report(
            detection_type=detection_type,
            server_round=server_round,
            total_clients=total_clients,
            score=score,
            percentile=percentile,
            min_kept_clients=min_kept_clients,
            max_reject_fraction=max_reject_fraction,
            enable_filter=enable_filter,
            skipped=skipped,
            extra=base_extra,
        )


class ClusteringDetectionFedAvg(FedAvg):
    """FedAvg with clustering-based detection before aggregation."""

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
        percentile: float = 80.0,
        min_kept_clients: int = 2,
        max_reject_fraction: float = 0.5,
        enable_filter: bool = True,
        min_silhouette: float = 0.05,
        cluster_score_gap: float = 0.15,
        seed: int = 42,
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

        self.percentile = float(percentile)
        self.min_kept_clients = int(min_kept_clients)
        self.max_reject_fraction = float(max_reject_fraction)
        self.enable_filter = bool(enable_filter)
        self.min_silhouette = float(min_silhouette)
        self.cluster_score_gap = float(cluster_score_gap)
        self.seed = int(seed)
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
        log(INFO, "\t│\t├── defense: clustering_detection")
        log(INFO, "\t│\t├── percentile: %.6f", self.percentile)
        log(INFO, "\t│\t├── min_kept_clients: %d", self.min_kept_clients)
        log(INFO, "\t│\t├── max_reject_fraction: %.6f", self.max_reject_fraction)
        log(INFO, "\t│\t├── min_silhouette: %.6f", self.min_silhouette)
        log(INFO, "\t│\t├── cluster_score_gap: %.6f", self.cluster_score_gap)
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

        global_vector = _state_dict_to_vector(self.current_arrays.to_torch_state_dict())
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

        report = build_clustering_report(
            detection_type=self.name,
            server_round=server_round,
            total_clients=len(scored_replies),
            norm_z=norm_z,
            cosine=cosine,
            score=score,
            deltas=deltas,
            percentile=self.percentile,
            min_kept_clients=self.min_kept_clients,
            max_reject_fraction=self.max_reject_fraction,
            enable_filter=self.enable_filter,
            seed=self.seed,
            skipped=int(skipped),
            min_silhouette=self.min_silhouette,
            cluster_score_gap=self.cluster_score_gap,
        )

        filtered_replies = [scored_replies[i] for i in report.kept_indices]
        arrays, metrics = super().aggregate_train(server_round, filtered_replies)

        if metrics is None:
            metrics = MetricRecord()

        return arrays, _merge_metrics(metrics, report.to_metric_record())


class ClusteringDetectionDefense(DetectionBase):
    """Detection wrapper that turns FedAvg into ClusteringDetectionFedAvg."""

    def apply(self, strategy: Any) -> Any:
        if not isinstance(strategy, FedAvg):
            raise TypeError(
                f"ClusteringDetectionDefense only supports FedAvg-compatible strategies, got {type(strategy)!r}."
            )

        extra = self.config.extra
        percentile = extra.get("percentile", extra.get("detection_percentile", 80.0))
        min_kept_clients = extra.get(
            "min_kept_clients", extra.get("min-kept-clients", 2)
        )
        max_reject_fraction = extra.get(
            "max_reject_fraction", extra.get("max-reject-fraction", 0.5)
        )
        enable_filter = extra.get("enable_filter", extra.get("enable-filter", True))
        min_silhouette = extra.get("min_silhouette", extra.get("min-silhouette", 0.05))
        cluster_score_gap = extra.get(
            "cluster_score_gap", extra.get("cluster-score-gap", 0.15)
        )

        return ClusteringDetectionFedAvg(
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
            percentile=float(percentile),
            min_kept_clients=int(min_kept_clients),
            max_reject_fraction=float(max_reject_fraction),
            enable_filter=bool(enable_filter),
            min_silhouette=float(min_silhouette),
            cluster_score_gap=float(cluster_score_gap),
            seed=int(self.config.seed),
        )


__all__ = [
    "build_clustering_report",
    "ClusteringDetectionFedAvg",
    "ClusteringDetectionDefense",
]