"""Trimmed Mean defense for federated learning."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from flwr.app import Array, ArrayRecord, ConfigRecord, Message, MetricRecord, RecordDict
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg

from ...base import DefenseBase, DefenseConfig


def _build_trimmed_mean(
    client_tensors: list[np.ndarray],
    trim_k: int,
) -> np.ndarray:
    """Compute coordinate-wise trimmed mean for one tensor.

    Args:
        client_tensors: Tensor values from all clients, same shape.
        trim_k: Number of largest and smallest values to trim.

    Returns:
        Aggregated tensor.
    """
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
        # Trimmed mean is only meaningful for floating-point tensors.
        # Non-floating buffers are handled outside this helper.
        return np.asarray(ref)

    stacked = np.stack(
        [np.asarray(t, dtype=np.float64) for t in client_tensors],
        axis=0,
    )  # shape: (num_clients, ...)

    sorted_stack = np.sort(stacked, axis=0)
    trimmed_stack = sorted_stack[trim_k : num_clients - trim_k]
    trimmed_mean = trimmed_stack.mean(axis=0)

    return trimmed_mean.astype(ref.dtype, copy=False)


class TrimmedMeanFedAvg(FedAvg):
    """FedAvg with server-side coordinate-wise trimmed mean aggregation."""

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
        trim_ratio: float = 0.2,
        trim_k: int | None = None,
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

        self.trim_ratio = float(trim_ratio)
        self.trim_k = None if trim_k is None else int(trim_k)

        if self.trim_ratio < 0.0:
            raise ValueError("trim_ratio must be non-negative.")
        if self.trim_k is not None and self.trim_k < 0:
            raise ValueError("trim_k must be non-negative.")

        self.current_arrays: ArrayRecord | None = None

    def summary(self) -> None:
        super().summary()
        from flwr.common.logger import log
        from logging import INFO

        log(INFO, "\t├──> Defense settings:")
        log(INFO, "\t│\t├── defense: trimmed_mean")
        log(INFO, "\t│\t├── trim_ratio: %.6f", self.trim_ratio)
        log(
            INFO,
            "\t│\t└── trim_k: %s",
            "auto" if self.trim_k is None else str(self.trim_k),
        )

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Store the current global model before sending it to clients."""
        self.current_arrays = arrays.copy()
        return super().configure_train(server_round, arrays, config, grid)

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        """Aggregate client updates using trimmed mean."""
        valid_replies, _ = self._check_and_log_replies(replies, is_train=True)
        if not valid_replies:
            return None, None

        if self.current_arrays is None:
            raise RuntimeError(
                "Current global model is not available. "
                "Make sure configure_train() was called first."
            )

        array_keys = list(self.current_arrays.keys())
        global_ndarrays = self.current_arrays.to_numpy_ndarrays(keep_input=False)

        client_ndarrays_list: list[list[np.ndarray]] = []
        client_update_norms: list[float] = []

        for msg in valid_replies:
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
                        np.asarray(l, dtype=np.float64) - np.asarray(g, dtype=np.float64)
                    )

            if float_deltas:
                flat_delta = np.concatenate([delta.reshape(-1) for delta in float_deltas], axis=0)
                client_update_norms.append(float(np.linalg.norm(flat_delta)))
            else:
                client_update_norms.append(0.0)

        num_clients = len(client_ndarrays_list)
        if self.trim_k is not None:
            trim_k = int(self.trim_k)
        else:
            trim_k = int(np.floor(self.trim_ratio * num_clients))

        if trim_k < 0:
            raise ValueError("trim_k must be non-negative.")
        if 2 * trim_k >= num_clients:
            raise ValueError(
                f"trim_k={trim_k} is too large for {num_clients} participating clients. "
                f"Need 2 * trim_k < num_clients."
            )

        aggregated_ndarrays: list[np.ndarray] = []

        for tensor_idx, global_arr in enumerate(global_ndarrays):
            g = np.asarray(global_arr)
            tensor_values = [np.asarray(client[tensor_idx]) for client in client_ndarrays_list]

            if not np.issubdtype(g.dtype, np.floating):
                aggregated_ndarrays.append(g)
                continue

            aggregated_tensor = _build_trimmed_mean(
                client_tensors=tensor_values,
                trim_k=trim_k,
            )
            aggregated_ndarrays.append(np.asarray(aggregated_tensor, dtype=g.dtype))

        aggregated_arrays = ArrayRecord(
            {
                key: Array(np.asarray(arr))
                for key, arr in zip(array_keys, aggregated_ndarrays, strict=True)
            }
        )

        reply_contents = [msg.content for msg in valid_replies]
        metrics = (
            self.train_metrics_aggr_fn(reply_contents, self.weighted_by_key)
            if self.train_metrics_aggr_fn is not None
            else MetricRecord()
        )
        if metrics is None:
            metrics = MetricRecord()

        metrics["defense-trim-ratio"] = float(self.trim_ratio)
        metrics["defense-trim-k"] = int(trim_k)
        metrics["defense-total-clients"] = int(num_clients)
        metrics["defense-avg-update-norm"] = (
            float(np.mean(client_update_norms)) if client_update_norms else 0.0
        )
        metrics["defense-max-update-norm"] = (
            float(np.max(client_update_norms)) if client_update_norms else 0.0
        )

        return aggregated_arrays, metrics


class TrimmedMeanDefense(DefenseBase):
    """Defense wrapper that turns FedAvg into TrimmedMeanFedAvg."""

    def apply(self, strategy):
        if not isinstance(strategy, FedAvg):
            raise TypeError(
                f"TrimmedMeanDefense only supports FedAvg, got {type(strategy)!r}."
            )

        trim_ratio = self.get_extra("trim_ratio", self.get_extra("trim-ratio", 0.2))
        trim_k = self.get_extra("trim_k", self.get_extra("trim-k", None))

        return TrimmedMeanFedAvg(
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
            trim_ratio=float(trim_ratio),
            trim_k=None if trim_k is None else int(trim_k),
        )