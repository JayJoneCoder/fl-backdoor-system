"""Feature extraction for detection.

This module centralizes all feature computation used by detection:
- update vectors (delta)
- norm
- robust z-score (norm_z)
- cosine similarity to center
"""

from __future__ import annotations

from typing import Any

import numpy as np

_EPS = 1e-12


def state_dict_to_vector(state_dict: dict[str, Any]) -> np.ndarray:
    """Flatten a model state_dict into a 1D vector."""
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


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity with numerical stability."""
    a_norm = float(np.linalg.norm(a))
    b_norm = float(np.linalg.norm(b))

    if a_norm <= _EPS or b_norm <= _EPS:
        return 1.0

    return float(np.dot(a, b) / (a_norm * b_norm + _EPS))


def extract_features(
    global_vector: np.ndarray,
    local_vectors: list[np.ndarray],
) -> dict[str, np.ndarray]:
    """Extract detection features from client updates.

    Returns a dict containing:
        - deltas
        - norms
        - norm_z
        - cosine
        - center
        - score (default combined score)
    """
    if not local_vectors:
        return {}

    stacked = np.stack(local_vectors, axis=0)

    # =========================
    # Delta
    # =========================
    deltas = stacked - global_vector.reshape(1, -1)

    # =========================
    # Norm
    # =========================
    norms = np.linalg.norm(deltas, axis=1)

    # =========================
    # Robust Z-score (MAD)
    # =========================
    median_norm = float(np.median(norms))
    mad = float(np.median(np.abs(norms - median_norm)))
    robust_scale = 1.4826 * mad + _EPS

    norm_z = np.abs(norms - median_norm) / robust_scale

    # =========================
    # Cosine similarity to center
    # =========================
    center = np.mean(deltas, axis=0)

    cosine = np.array(
        [cosine_similarity(delta, center) for delta in deltas],
        dtype=np.float64,
    )

    # =========================
    # Default score（统一评分）
    # =========================
    score = norm_z + np.clip(1.0 - cosine, 0.0, 2.0)

    return {
        "deltas": deltas,
        "norms": norms,
        "norm_z": norm_z,
        "cosine": cosine,
        "center": center,
        "score": score,
    }