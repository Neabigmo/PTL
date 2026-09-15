"""Leakage-free split-half reproducibility utilities for outcome/audit only."""

from __future__ import annotations

import numpy as np


def split_half_indices(n: int, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if n < 2:
        raise ValueError("split-half reproducibility requires at least two independent groups")
    order = np.random.default_rng(seed).permutation(n)
    midpoint = n // 2
    if midpoint == 0 or midpoint == n:
        raise ValueError("split-half partition is empty")
    left, right = np.sort(order[:midpoint]), np.sort(order[midpoint:])
    if np.intersect1d(left, right).size or len(np.union1d(left, right)) != n:
        raise AssertionError("split-half groups overlap or omit an input group")
    return left, right


def weighted_profile(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.ndim != 2 or len(values) != len(weights) or np.any(weights <= 0):
        raise ValueError("values and positive weights have incompatible shapes")
    return np.average(values, axis=0, weights=weights)


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    denom = np.linalg.norm(left) * np.linalg.norm(right)
    if denom == 0:
        return 1.0 if np.allclose(left, right) else 0.0
    return float(np.clip(np.dot(left, right) / denom, -1.0, 1.0))
