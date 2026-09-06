"""Uncertainty utilities that do not inspect target fidelity labels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EnsembleUQ:
    mean_gene_variance: np.ndarray
    median_gene_variance: np.ndarray
    top_effect_variance: np.ndarray
    cosine_disagreement: np.ndarray
    effect_norm_variance: np.ndarray


def summarize_ensemble(predictions: np.ndarray, top_k: int = 50) -> EnsembleUQ:
    """Summarize [member, item, gene] predictions without target access."""

    values = np.asarray(predictions, dtype=np.float64)
    if values.ndim != 3 or values.shape[0] < 2:
        raise ValueError("ensemble UQ needs at least two [member, item, gene] predictions")
    variance = np.var(values, axis=0, ddof=1)
    mean_prediction = values.mean(axis=0)
    top_k = min(top_k, values.shape[-1])
    top_indices = np.argpartition(np.abs(mean_prediction), -top_k, axis=1)[:, -top_k:]
    top_effect_variance = np.take_along_axis(variance, top_indices, axis=1).mean(axis=1)

    mean_norm = np.linalg.norm(mean_prediction, axis=1)
    member_norm = np.linalg.norm(values, axis=2)
    cosine = np.sum(values * mean_prediction[None, :, :], axis=2)
    cosine = cosine / np.maximum(member_norm * np.maximum(mean_norm[None, :], 1e-12), 1e-12)
    cosine_disagreement = 1.0 - np.mean(cosine, axis=0)
    effect_norm_variance = np.var(member_norm, axis=0, ddof=1)
    return EnsembleUQ(
        mean_gene_variance=variance.mean(axis=1),
        median_gene_variance=np.median(variance, axis=1),
        top_effect_variance=top_effect_variance,
        cosine_disagreement=cosine_disagreement,
        effect_norm_variance=effect_norm_variance,
    )


def empirical_quantile(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Map uncertainty to a predictor-relative empirical quantile."""

    values = np.asarray(values, dtype=float)
    reference = np.asarray(reference, dtype=float)
    reference = np.sort(reference[np.isfinite(reference)])
    if reference.size == 0:
        raise ValueError("reference uncertainty distribution is empty")
    ranks = np.searchsorted(reference, values, side="right")
    return np.clip((ranks - 0.5) / reference.size, 0.0, 1.0)


def confidence_from_uq(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Convert high-uncertainty quantiles to high-is-good confidence."""

    return 1.0 - empirical_quantile(values, reference)
