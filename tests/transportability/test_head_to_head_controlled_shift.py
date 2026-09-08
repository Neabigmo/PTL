from __future__ import annotations

import numpy as np

from scripts.build_head_to_head_controlled_shift import _rank_stats


def test_rank_stats_reports_strict_inversion_and_confidence_tracking() -> None:
    stats = _rank_stats(
        np.asarray([0.1, 0.2, 0.3]),
        np.asarray([0.3, 0.1, 0.2]),
        np.asarray([-0.1, -0.2, -0.3]),
        np.asarray([-0.3, -0.1, -0.2]),
    )
    assert stats["n_comparable_risk_pairs"] == 3
    assert stats["n_risk_inversions"] == 2
    assert stats["risk_inversion_rate"] == 2 / 3
    assert stats["mean_normalized_rank_displacement"] > 0


def test_rank_stats_excludes_tied_pairs() -> None:
    stats = _rank_stats(
        np.asarray([0.1, 0.1, 0.3]),
        np.asarray([0.2, 0.2, 0.3]),
        np.asarray([-0.1, -0.1, -0.3]),
        np.asarray([-0.2, -0.2, -0.3]),
    )
    assert stats["n_comparable_risk_pairs"] == 2
    assert stats["n_risk_inversions"] == 0
