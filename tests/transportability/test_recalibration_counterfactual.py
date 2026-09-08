import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from scripts.run_formal_v2_recalibration_counterfactual import _aurc, _order_audit
from src.ptl.reliability.calibration import assert_positive_platt_slope


def test_strict_monotonic_recalibration_preserves_comparable_order() -> None:
    base = np.array([0.1, 0.4, 0.7, 0.9])
    recalibrated = np.array([0.2, 0.3, 0.8, 0.99])

    result = _order_audit(base, recalibrated)

    assert result["n_order_disagreements"] == 0
    assert result["rank_order_preserved"] is True
    assert result["n_comparable_pairs"] == 6


def test_aurc_uses_the_confidence_order() -> None:
    risk = np.array([0.1, 0.9, 0.2])
    high_confidence_first = np.array([0.9, 0.1, 0.8])
    low_confidence_first = np.array([0.1, 0.9, 0.2])

    assert _aurc(risk, high_confidence_first) < _aurc(risk, low_confidence_first)


def test_order_audit_reports_ties_without_calling_them_disagreements() -> None:
    result = _order_audit(
        pd.Series([0.1, 0.1, 0.8]).to_numpy(),
        pd.Series([0.2, 0.2, 0.9]).to_numpy(),
    )

    assert result["n_base_tied_pairs"] == 1
    assert result["n_recalibrated_tied_pairs"] == 1
    assert result["n_order_disagreements"] == 0


def test_platt_contract_requires_a_strictly_increasing_map() -> None:
    model = LogisticRegression(C=10.0, solver="lbfgs", max_iter=1000).fit(
        np.array([[-2.0], [-1.0], [1.0], [2.0]]), np.array([0, 0, 1, 1])
    )
    assert assert_positive_platt_slope(model) > 0.0

    decreasing = LogisticRegression(C=10.0, solver="lbfgs", max_iter=1000).fit(
        np.array([[-2.0], [-1.0], [1.0], [2.0]]), np.array([1, 1, 0, 0])
    )
    with pytest.raises(ValueError, match="positive slope"):
        assert_positive_platt_slope(decreasing)
