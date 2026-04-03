from __future__ import annotations

import numpy as np
from typing import Tuple


def score_based_filter(
    score: np.ndarray,
    *,
    percentile: float = 80.0,
) -> Tuple[float, np.ndarray]:
    """
    Pure score thresholding.

    Args:
        score: precomputed anomaly score, higher = more suspicious
        percentile: threshold percentile

    Returns:
        threshold: float
        suspicious_mask: np.ndarray[bool]
    """
    score = np.asarray(score, dtype=np.float64).reshape(-1)

    if score.size == 0:
        return 0.0, np.zeros(0, dtype=bool)

    threshold = float(np.percentile(score, percentile))
    suspicious = score > threshold
    return threshold, suspicious