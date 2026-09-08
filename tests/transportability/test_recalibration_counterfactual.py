import numpy as np
import pandas as pd

from scripts.run_formal_v2_recalibration_counterfactual import _aurc, _order_audit


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
