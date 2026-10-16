"""Deterministic IID and label-skewed dataset partitioning."""

from collections.abc import Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, random_split

from fl_backdoor.utils.randomness import derived_seed


def _dataset_targets(dataset) -> torch.Tensor:
    """Return labels without requiring callers to expose a particular dataset type."""
    for attribute in ("targets", "labels"):
        values = getattr(dataset, attribute, None)
        if values is not None:
            targets = torch.as_tensor(values)
            if targets.ndim != 1 or len(targets) != len(dataset):
                raise ValueError("Dataset labels must be a one-dimensional sequence matching the dataset length.")
            return targets

    if isinstance(dataset, Subset):
        targets = _dataset_targets(dataset.dataset)
        return targets[torch.as_tensor(dataset.indices, dtype=torch.long)]

    nested = getattr(dataset, "dataset", None)
    if nested is not None and nested is not dataset:
        return _dataset_targets(nested)

    labels = []
    for index in range(len(dataset)):
        sample = dataset[index]
        if isinstance(sample, Mapping):
            if "label" not in sample:
                raise ValueError("Dataset samples must contain a 'label' field for Dirichlet partitioning.")
            labels.append(sample["label"])
        elif isinstance(sample, (tuple, list)) and len(sample) >= 2:
            labels.append(sample[1])
        else:
            raise ValueError("Dataset samples must expose labels for Dirichlet partitioning.")
    targets = torch.as_tensor(labels)
    if targets.ndim != 1 or len(targets) != len(dataset):
        raise ValueError("Dataset labels must be a one-dimensional sequence matching the dataset length.")
    return targets


def _iid_partition_indices(length: int, num_partitions: int, seed: int) -> list[list[int]]:
    indices = torch.randperm(length, generator=torch.Generator().manual_seed(seed)).tolist()
    per_client = length // num_partitions
    return [
        indices[partition_id * per_client:
                partition_id * per_client + per_client if partition_id < num_partitions - 1 else length]
        for partition_id in range(num_partitions)
    ]


def _dirichlet_partition_indices(dataset, num_partitions: int, alpha: float, seed: int) -> list[list[int]]:
    """Create a complete, deterministic label-skewed partition plan."""
    targets = _dataset_targets(dataset).cpu().numpy()
    rng = np.random.default_rng(derived_seed(seed, "dirichlet", len(dataset), num_partitions))
    partitions: list[list[int]] = [[] for _ in range(num_partitions)]

    for label in np.unique(targets):
        class_indices = np.flatnonzero(targets == label)
        class_indices = rng.permutation(class_indices)
        probabilities = rng.dirichlet(np.full(num_partitions, alpha, dtype=float))
        counts = rng.multinomial(len(class_indices), probabilities)
        start = 0
        for partition_id, count in enumerate(counts):
            stop = start + int(count)
            partitions[partition_id].extend(class_indices[start:stop].tolist())
            start = stop

    # Very small alpha values can produce empty clients. Move samples from the
    # largest clients until every client can still make a train/validation split.
    minimum_size = 2
    while True:
        target = min(range(num_partitions), key=lambda partition_id: len(partitions[partition_id]))
        if len(partitions[target]) >= minimum_size:
            break
        donors = [partition_id for partition_id in range(num_partitions)
                  if len(partitions[partition_id]) > minimum_size]
        if not donors:
            raise ValueError("Dirichlet partitioning could not give every client at least 2 samples.")
        donor = max(donors, key=lambda partition_id: len(partitions[partition_id]))
        moved_index = int(rng.integers(len(partitions[donor])))
        partitions[target].append(partitions[donor].pop(moved_index))

    for partition_id, indices in enumerate(partitions):
        partitions[partition_id] = rng.permutation(indices).tolist()
    return partitions


def partition_loaders(
    dataset,
    partition_id,
    num_partitions,
    batch_size,
    *,
    seed=42,
    shuffle_seed=None,
    partition_mode="iid",
    partition_alpha=0.5,
    partition_seed=None,
):
    if not 0 <= partition_id < num_partitions <= len(dataset):
        raise ValueError("客户端分区编号或数量无效；分区数不能超过样本数。")
    if len(dataset) // num_partitions < 2:
        raise ValueError("每个客户端至少需要 2 个样本，才能划分非空的训练集和验证集；请减少客户端数。")
    if partition_mode not in {"iid", "dirichlet"}:
        raise ValueError("partition-mode must be either 'iid' or 'dirichlet'.")
    if isinstance(partition_alpha, bool) or partition_alpha <= 0:
        raise ValueError("partition-alpha must be positive.")
    if partition_seed is not None and (
        isinstance(partition_seed, bool) or not isinstance(partition_seed, int) or partition_seed < 0
    ):
        raise ValueError("partition-seed must be a non-negative integer.")

    effective_partition_seed = seed if partition_seed is None else partition_seed
    if partition_mode == "iid":
        partition_indices = _iid_partition_indices(len(dataset), num_partitions, effective_partition_seed)
    else:
        partition_indices = _dirichlet_partition_indices(
            dataset, num_partitions, float(partition_alpha), effective_partition_seed,
        )
    subset = Subset(dataset, partition_indices[partition_id])
    n_val = max(1, int(0.2 * len(subset)))
    train, val = random_split(subset, [len(subset) - n_val, n_val], generator=
                              torch.Generator().manual_seed(derived_seed(seed, "split", partition_id)))
    shuffle_seed = derived_seed(seed, "shuffle", partition_id) if shuffle_seed is None else shuffle_seed
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True,
                   generator=torch.Generator().manual_seed(shuffle_seed)),
        DataLoader(val, batch_size=batch_size,
                   generator=torch.Generator().manual_seed(derived_seed(seed, "validation", partition_id))),
    )