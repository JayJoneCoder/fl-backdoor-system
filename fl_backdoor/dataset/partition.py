"""IID partitioning without mutating the process-global Torch RNG."""

import torch
from torch.utils.data import DataLoader, Subset, random_split

from fl_backdoor.utils.randomness import derived_seed


def partition_loaders(dataset, partition_id, num_partitions, batch_size, *, seed=42, shuffle_seed=None):
    if not 0 <= partition_id < num_partitions <= len(dataset):
        raise ValueError("客户端分区编号或数量无效；分区数不能超过样本数。")
    if len(dataset) // num_partitions < 2:
        raise ValueError("每个客户端至少需要 2 个样本，才能划分非空的训练集和验证集；请减少客户端数。")
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed)).tolist()
    per_client = len(dataset) // num_partitions
    start = partition_id * per_client
    end = start + per_client if partition_id < num_partitions - 1 else len(dataset)
    subset = Subset(dataset, indices[start:end])
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