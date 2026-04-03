from __future__ import annotations

from functools import lru_cache
from typing import Iterable

import numpy as np

def normalize_fixed_malicious_clients(fixed_malicious_clients):
    if fixed_malicious_clients is None:
        return ()

    # 字符串，如"0,1,2"
    if isinstance(fixed_malicious_clients, str):
        fixed_malicious_clients = [
            x.strip()
            for x in fixed_malicious_clients.split(",")
            if x.strip() != ""
        ]

    # 转 int
    return tuple(sorted({int(x) for x in fixed_malicious_clients}))


@lru_cache(maxsize=None)
def _select_malicious_clients_cached(
    num_clients: int,
    malicious_ratio: float,
    seed: int,
    server_round: int,
) -> tuple[int, ...]:
    """Cached random selection path only.

    Important: this function intentionally has no list/dict/set arguments,
    so it is always safe for lru_cache.
    """
    num_clients = int(num_clients)
    if num_clients <= 0:
        return tuple()

    ratio = float(malicious_ratio)
    num_malicious = int(round(num_clients * ratio))
    if ratio > 0.0 and num_malicious == 0:
        num_malicious = 1
    num_malicious = min(max(num_malicious, 0), num_clients)

    rng = np.random.default_rng(int(seed) + 10007 * int(server_round))
    client_ids = np.arange(num_clients)
    rng.shuffle(client_ids)
    return tuple(int(cid) for cid in client_ids[:num_malicious])


def select_malicious_clients(
    num_clients: int,
    malicious_ratio: float,
    seed: int = 42,
    *,
    malicious_mode: str = "random",
    fixed_malicious_clients: Iterable[int] | None = None,
    server_round: int = 0,
) -> set[int]:
    """Select malicious client ids.

    Supported modes:
    - random: deterministic per (seed, round)
    - fixed: use user-specified client ids
    """
    mode = str(malicious_mode).lower().strip()

    if mode == "fixed":
        fixed_clients = normalize_fixed_malicious_clients(fixed_malicious_clients)
        return {cid for cid in fixed_clients if 0 <= cid < int(num_clients)}

    if mode != "random":
        raise ValueError("malicious_mode must be 'random' or 'fixed'.")

    return set(
        _select_malicious_clients_cached(
            int(num_clients),
            float(malicious_ratio),
            int(seed),
            int(server_round),
        )
    )


def is_malicious_client(
    cid: int | str,
    num_clients: int,
    malicious_ratio: float,
    seed: int = 42,
    *,
    malicious_mode: str = "random",
    fixed_malicious_clients: Iterable[int] | None = None,
    server_round: int = 0,
) -> bool:
    return int(cid) in select_malicious_clients(
        num_clients=num_clients,
        malicious_ratio=malicious_ratio,
        seed=seed,
        malicious_mode=malicious_mode,
        fixed_malicious_clients=fixed_malicious_clients,
        server_round=server_round,
    )