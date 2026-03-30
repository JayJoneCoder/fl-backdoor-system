"""Norm clipping defense for federated learning."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from flwr.app import Array, ArrayRecord, ConfigRecord, Message, MetricRecord, RecordDict
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg
from flwr.serverapp.strategy.strategy_utils import aggregate_arrayrecords

from .base import DefenseBase, DefenseConfig


def _clip_client_update(
    global_ndarrays: list[np.ndarray],
    local_ndarrays: list[np.ndarray],
    clip_norm: float,
) -> tuple[list[np.ndarray], float, float, bool]:
    """Clip one client's model update by global L2 norm.

    Returns:
        clipped_ndarrays: Clipped local arrays.
        pre_norm: Norm before clipping.
        post_norm: Norm after clipping.
        was_clipped: Whether scaling was applied.
    """
    if len(global_ndarrays) != len(local_ndarrays):
        raise ValueError(
            "global_ndarrays and local_ndarrays must have the same length."
        )

    float_indices: list[int] = []
    float_deltas: list[np.ndarray] = []

    for idx, (g_arr, l_arr) in enumerate(zip(global_ndarrays, local_ndarrays, strict=True)):
        g = np.asarray(g_arr)
        l = np.asarray(l_arr)

        # Only clip floating-point tensors. Integer buffers stay unchanged.
        if np.issubdtype(g.dtype, np.floating) and np.issubdtype(l.dtype, np.floating):
            float_indices.append(idx)
            float_deltas.append(l - g)

    # No float tensors to clip
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

    for idx, (g_arr, l_arr) in enumerate(zip(global_ndarrays, local_ndarrays, strict=True)):
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


class NormClippingFedAvg(FedAvg):
    """FedAvg with server-side norm clipping before aggregation."""

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
        clip_norm: float = 1.0,
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
        self.clip_norm = float(clip_norm)
        if self.clip_norm <= 0.0:
            raise ValueError("clip_norm must be positive.")
        self.current_arrays: ArrayRecord | None = None

    def summary(self) -> None:
        super().summary()
        from flwr.common.logger import log
        from logging import INFO

        log(INFO, "\t├──> Defense settings:")
        log(INFO, "\t│\t├── defense: norm_clipping")
        log(INFO, "\t│\t└── clip_norm: %.6f", self.clip_norm)

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
        """Clip client updates, then aggregate them with standard FedAvg."""
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

        clipped_contents: list[RecordDict] = []
        pre_norms: list[float] = []
        post_norms: list[float] = []
        clipped_clients = 0

        for msg in valid_replies:
            content = RecordDict(msg.content)
            local_arrays = content[self.arrayrecord_key]
            local_ndarrays = local_arrays.to_numpy_ndarrays(keep_input=False)

            clipped_ndarrays, pre_norm, post_norm, was_clipped = _clip_client_update(
                global_ndarrays=global_ndarrays,
                local_ndarrays=local_ndarrays,
                clip_norm=self.clip_norm,
            )

            pre_norms.append(pre_norm)
            post_norms.append(post_norm)
            clipped_clients += int(was_clipped)

            # Preserve the original parameter names.
            clipped_arrays = ArrayRecord(
                {
                    key: Array(np.asarray(arr))
                    for key, arr in zip(array_keys, clipped_ndarrays, strict=True)
                }
            )
            content[self.arrayrecord_key] = clipped_arrays
            clipped_contents.append(content)

        arrays = aggregate_arrayrecords(clipped_contents, self.weighted_by_key)
        metrics = self.train_metrics_aggr_fn(clipped_contents, self.weighted_by_key)

        if metrics is None:
            metrics = MetricRecord()

        total_clients = len(valid_replies)
        metrics["defense-clip-norm"] = float(self.clip_norm)
        metrics["defense-total-clients"] = int(total_clients)
        metrics["defense-clipped-clients"] = int(clipped_clients)
        metrics["defense-clipped-ratio"] = (
            float(clipped_clients / total_clients) if total_clients > 0 else 0.0
        )
        metrics["defense-avg-update-norm"] = (
            float(np.mean(pre_norms)) if pre_norms else 0.0
        )
        metrics["defense-max-update-norm"] = (
            float(np.max(pre_norms)) if pre_norms else 0.0
        )
        metrics["defense-avg-post-norm"] = (
            float(np.mean(post_norms)) if post_norms else 0.0
        )

        return arrays, metrics


class NormClippingDefense(DefenseBase):
    """Defense wrapper that turns FedAvg into NormClippingFedAvg."""

    def apply(self, strategy):
        if not isinstance(strategy, FedAvg):
            raise TypeError(
                f"NormClippingDefense only supports FedAvg, got {type(strategy)!r}."
            )

        clip_norm = self.get_extra("clip_norm", self.get_extra("clip-norm", 1.0))

        return NormClippingFedAvg(
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
            clip_norm=float(clip_norm),
        )