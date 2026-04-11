"""Krum / Multi-Krum defense for federated learning."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from flwr.app import Array, ArrayRecord, ConfigRecord, Message, MetricRecord, RecordDict
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg

from ...base import DefenseBase, DefenseConfig


def _flatten_update(
    global_ndarrays: list[np.ndarray],
    local_ndarrays: list[np.ndarray],
) -> np.ndarray:
    """Flatten client update (delta = local - global) into 1D vector."""
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


def _krum_select(
    flat_updates: list[np.ndarray],
    f: int,
    k: int,
) -> list[int]:
    """Krum / Multi-Krum selection.

    Args:
        flat_updates: List of flattened client updates.
        f: Number of malicious clients.
        k: Number of selected clients (k=1 => Krum).

    Returns:
        indices of selected clients
    """
    n = len(flat_updates)
    if n < 2:
        raise ValueError("Need at least 2 clients for Krum.")

    if n <= f + 2:
        raise ValueError(
            f"Krum requires n > f + 2, got n={n}, f={f}."
        )

    # Pairwise distance matrix
    distances = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(flat_updates[i] - flat_updates[j]) ** 2
            distances[i, j] = d
            distances[j, i] = d

    scores = []

    # number of neighbors
    m = n - f - 2

    for i in range(n):
        dists = np.sort(distances[i])  # includes self-distance = 0
        score = np.sum(dists[1 : m + 1])  # skip self (index 0)
        scores.append(score)

    scores = np.array(scores)

    selected_indices = np.argsort(scores)[:k]
    return selected_indices.tolist()


class KrumFedAvg(FedAvg):
    """FedAvg with Krum / Multi-Krum aggregation."""

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
        num_malicious: int = 1,
        krum_k: int = 1,
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

        self.num_malicious = int(num_malicious)
        self.krum_k = int(krum_k)

        if self.num_malicious < 0:
            raise ValueError("num_malicious must be non-negative.")
        if self.krum_k <= 0:
            raise ValueError("krum_k must be >= 1.")

        self.current_arrays: ArrayRecord | None = None

    def summary(self) -> None:
        super().summary()
        from flwr.common.logger import log
        from logging import INFO

        log(INFO, "\t├──> Defense settings:")
        log(INFO, "\t│\t├── defense: krum")
        log(INFO, "\t│\t├── num_malicious (f): %d", self.num_malicious)
        log(INFO, "\t│\t└── krum_k: %d", self.krum_k)

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

        valid_replies, _ = self._check_and_log_replies(replies, is_train=True)
        if not valid_replies:
            return None, None

        if self.current_arrays is None:
            raise RuntimeError("Global model not initialized.")

        array_keys = list(self.current_arrays.keys())
        global_ndarrays = self.current_arrays.to_numpy_ndarrays(keep_input=False)

        client_ndarrays_list: list[list[np.ndarray]] = []
        flat_updates: list[np.ndarray] = []

        for msg in valid_replies:
            content = RecordDict(msg.content)
            local_arrays = content[self.arrayrecord_key]
            local_ndarrays = local_arrays.to_numpy_ndarrays(keep_input=False)

            client_ndarrays_list.append(local_ndarrays)

            flat = _flatten_update(global_ndarrays, local_ndarrays)
            flat_updates.append(flat)

        n = len(flat_updates)
        f = self.num_malicious
        k = min(self.krum_k, n)

        selected_indices = _krum_select(flat_updates, f=f, k=k)

        # === Multi-Krum: average selected clients ===
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

        # === metrics ===
        reply_contents = [msg.content for msg in valid_replies]
        metrics = (
            self.train_metrics_aggr_fn(reply_contents, self.weighted_by_key)
            if self.train_metrics_aggr_fn is not None
            else MetricRecord()
        )

        if metrics is None:
            metrics = MetricRecord()

        metrics["agg_total_clients"] = int(n)
        metrics["agg_selected_clients"] = int(len(selected_indices))
        metrics["agg_krum_f"] = int(f)
        metrics["agg_krum_k"] = int(k)

        return aggregated_arrays, metrics


class KrumDefense(DefenseBase):
    """Defense wrapper for Krum."""

    def apply(self, strategy):
        if not isinstance(strategy, FedAvg):
            raise TypeError(
                f"KrumDefense only supports FedAvg, got {type(strategy)!r}."
            )

        num_malicious = self.get_extra("num_malicious", 1)
        krum_k = self.get_extra("krum_k", 1)

        return KrumFedAvg(
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
            num_malicious=int(num_malicious),
            krum_k=int(krum_k),
        )