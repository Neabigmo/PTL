"""Prediction-level selective metrics with environment-first aggregation."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score


def _area(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.trapezoid(y, x)) if hasattr(np, "trapezoid") else float(np.trapz(y, x))


def _coverage_risk(score: np.ndarray, continuous_risk: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-np.asarray(score, dtype=float), kind="mergesort")
    risk = np.asarray(continuous_risk, dtype=float)[order]
    coverage = np.arange(1, len(risk) + 1, dtype=float) / max(len(risk), 1)
    return coverage, np.cumsum(risk) / np.arange(1, len(risk) + 1)


def evaluate_scores(
    y_reliable: np.ndarray,
    score: np.ndarray,
    continuous_risk: np.ndarray | None = None,
) -> dict[str, float]:
    y = np.asarray(y_reliable, dtype=int)
    score = np.clip(np.asarray(score, dtype=float), 1e-6, 1.0 - 1e-6)
    if continuous_risk is None:
        continuous_risk = 1.0 - y
    risk = np.asarray(continuous_risk, dtype=float)
    coverage, curve = _coverage_risk(score, risk)
    oracle_coverage, oracle_curve = _coverage_risk(1.0 - risk, risk)
    # The oracle score is high for low risk, so its curve is the lower envelope.
    aurc = _area(np.r_[0.0, coverage], np.r_[curve[0], curve])
    oracle_aurc = _area(np.r_[0.0, oracle_coverage], np.r_[oracle_curve[0], oracle_curve])

    def at(target: float) -> float:
        index = min(max(int(np.ceil(target * len(curve))) - 1, 0), len(curve) - 1)
        return float(curve[index])

    def ftr_at(target: float) -> float:
        order = np.argsort(-score, kind="mergesort")
        count = max(1, int(np.ceil(target * len(y))))
        return float(1.0 - y[order[:count]].mean())

    result = {
        "n": float(len(y)),
        "aurc": aurc,
        "excess_aurc": aurc - oracle_aurc,
        "risk_at_50": at(0.5),
        "risk_at_80": at(0.8),
        "ftr_at_50": ftr_at(0.5),
        "ftr_at_80": ftr_at(0.8),
        "brier": float(np.mean((score - y) ** 2)),
        "log_loss": float(log_loss(y, score, labels=[0, 1])) if np.unique(y).size > 1 else float("nan"),
        "auroc": float(roc_auc_score(y, score)) if np.unique(y).size > 1 else float("nan"),
        "auprc": float(average_precision_score(y, score)) if np.unique(y).size > 1 else float("nan"),
    }
    realized_order = np.argsort(risk)
    result["spearman_continuous_risk"] = float(
        np.corrcoef(np.argsort(np.argsort(-score)), np.argsort(np.argsort(risk)))[0, 1]
    ) if len(y) > 2 and np.std(score) > 0 and np.std(risk) > 0 else float("nan")
    return result
