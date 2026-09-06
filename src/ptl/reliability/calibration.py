"""Small calibration models for grouped cross-fitting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def _clip_probability(values: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), 1e-5, 1.0 - 1e-5)


def fit_platt(p0: np.ndarray, y: np.ndarray) -> LogisticRegression | float:
    """Fit scalar logistic calibration on training-fold probabilities only."""

    p0 = _clip_probability(p0)
    y = np.asarray(y, dtype=int)
    if np.unique(y).size < 2:
        return float(y.mean())
    model = LogisticRegression(C=10.0, solver="lbfgs", max_iter=1000)
    model.fit(np.log(p0 / (1.0 - p0)).reshape(-1, 1), y)
    return model


def predict_platt(model: LogisticRegression | float, p0: np.ndarray) -> np.ndarray:
    if np.isscalar(model):
        return np.full(len(p0), float(model), dtype=float)
    p0 = _clip_probability(p0)
    logit = np.log(p0 / (1.0 - p0)).reshape(-1, 1)
    return model.predict_proba(logit)[:, 1]


def fit_isotonic(p0: np.ndarray, y: np.ndarray) -> IsotonicRegression | float:
    y = np.asarray(y, dtype=float)
    if np.unique(y).size < 2:
        return float(y.mean())
    model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    model.fit(np.asarray(p0, dtype=float), y)
    return model


def predict_isotonic(model: IsotonicRegression | float, p0: np.ndarray) -> np.ndarray:
    if np.isscalar(model):
        return np.full(len(p0), float(model), dtype=float)
    return np.asarray(model.predict(np.asarray(p0, dtype=float)), dtype=float)


def fit_logistic(X: np.ndarray, y: np.ndarray) -> LogisticRegression | float:
    y = np.asarray(y, dtype=int)
    if np.unique(y).size < 2:
        return float(y.mean())
    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    model.fit(np.asarray(X, dtype=float), y)
    return model


def predict_logistic(model: LogisticRegression | float, X: np.ndarray) -> np.ndarray:
    if np.isscalar(model):
        return np.full(len(X), float(model), dtype=float)
    return model.predict_proba(np.asarray(X, dtype=float))[:, 1]


@dataclass
class ContextualCalibrator:
    """Linear block encoder for p=sigmoid(exp(a(z))*logit(p0)+b(z))."""

    learning_rate: float = 0.03
    epochs: int = 1200
    correction_penalty: float = 0.02
    a_weights: np.ndarray | None = None
    b_weights: np.ndarray | None = None
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None
    a_intercept_: float = 0.0
    b_intercept_: float = 0.0

    def fit(self, X: np.ndarray, p0: np.ndarray, y: np.ndarray) -> "ContextualCalibrator":
        X = np.asarray(X, dtype=float)
        p0 = _clip_probability(p0)
        y = np.asarray(y, dtype=float)
        if X.ndim != 2 or len(X) != len(p0) or len(y) != len(p0):
            raise ValueError("context calibrator inputs must have matching row counts")
        self.mean_ = X.mean(axis=0)
        self.scale_ = np.where(X.std(axis=0) > 1e-8, X.std(axis=0), 1.0)
        Z = (X - self.mean_) / self.scale_
        self.a_weights = np.zeros(Z.shape[1], dtype=float)
        self.b_weights = np.zeros(Z.shape[1], dtype=float)
        self.a_intercept_ = 0.0
        self.b_intercept_ = 0.0
        base_logit = np.log(p0 / (1.0 - p0))
        n = max(len(y), 1)
        for _ in range(self.epochs):
            a = self.a_intercept_ + Z @ self.a_weights
            b = self.b_intercept_ + Z @ self.b_weights
            eta = np.exp(np.clip(a, -4.0, 4.0)) * base_logit + b
            prob = 1.0 / (1.0 + np.exp(-np.clip(eta, -30.0, 30.0)))
            residual = prob - y
            slope = np.exp(np.clip(a, -4.0, 4.0))
            grad_a = (Z.T @ (residual * slope * base_logit)) / n
            grad_b = (Z.T @ residual) / n
            grad_a += self.correction_penalty * self.a_weights
            grad_b += self.correction_penalty * self.b_weights
            self.a_weights -= self.learning_rate * grad_a
            self.b_weights -= self.learning_rate * grad_b
            self.a_intercept_ -= self.learning_rate * float(residual.dot(slope * base_logit) / n)
            self.b_intercept_ -= self.learning_rate * float(residual.mean())
        return self

    def predict(self, X: np.ndarray, p0: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.a_weights is None or self.b_weights is None:
            raise RuntimeError("context calibrator must be fitted before predict")
        X = np.asarray(X, dtype=float)
        p0 = _clip_probability(p0)
        Z = (X - self.mean_) / self.scale_
        a = self.a_intercept_ + Z @ self.a_weights
        b = self.b_intercept_ + Z @ self.b_weights
        eta = np.exp(np.clip(a, -4.0, 4.0)) * np.log(p0 / (1.0 - p0)) + b
        return 1.0 / (1.0 + np.exp(-np.clip(eta, -30.0, 30.0)))
