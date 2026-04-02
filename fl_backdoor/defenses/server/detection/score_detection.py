from __future__ import annotations

import numpy as np
from typing import Any, List, Tuple


def score_based_filter(
    norm_z: np.ndarray,
    cosine: np.ndarray,
    total: int,
    *,
    percentile: float = 80.0,
    weight_norm: float = 1.0,
    weight_cosine: float = 1.0,
    min_kept_clients: int = 2,
    enable_filter: bool = True,
) -> Tuple[List[int], np.ndarray, float, np.ndarray]:
    """
    Returns:
        keep_indices
        score
        threshold
        suspicious_mask
    """

    score = weight_norm * norm_z + weight_cosine * np.clip(1.0 - cosine, 0.0, 2.0)

    threshold = float(np.percentile(score, percentile))
    suspicious = score > threshold

    if enable_filter:
        keep_indices = np.where(~suspicious)[0].tolist()

        if len(keep_indices) < min_kept_clients:
            keep_indices = np.argsort(score)[:min_kept_clients].tolist()
    else:
        keep_indices = list(range(total))

    return keep_indices, score, threshold, suspicious