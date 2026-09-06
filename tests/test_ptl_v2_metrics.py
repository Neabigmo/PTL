import numpy as np
import pandas as pd

from ptl.evaluation.bootstrap import paired_hierarchical_bootstrap, summarize_bootstrap_ci
from ptl.evaluation.metrics import evaluate_scores


def test_selective_metrics_have_primary_fields():
    y = np.array([1, 1, 0, 0, 1, 0])
    score = np.array([0.95, 0.85, 0.2, 0.1, 0.7, 0.3])
    result = evaluate_scores(y, score)
    for field in ["aurc", "excess_aurc", "risk_at_50", "risk_at_80", "ftr_at_50", "ftr_at_80", "brier", "log_loss"]:
        assert field in result
        assert np.isfinite(result[field])


def test_paired_hierarchical_bootstrap_keeps_methods_paired():
    frame = pd.DataFrame(
        {
            "predictor": ["a"] * 6,
            "environment_id": ["e1"] * 3 + ["e2"] * 3,
            "biological_instance_id": ["b1", "b2", "b3", "b4", "b5", "b6"],
            "reliable_label": [1, 0, 1, 0, 1, 0],
            "continuous_risk": [0.1, 0.8, 0.2, 0.7, 0.3, 0.6],
            "raw": [0.9, 0.2, 0.8, 0.3, 0.7, 0.4],
            "better": [0.95, 0.1, 0.9, 0.2, 0.85, 0.25],
        }
    )
    draws = paired_hierarchical_bootstrap(frame, ["raw", "better"], n_resamples=5, seed=3)
    summary = summarize_bootstrap_ci(draws)
    assert set(summary["method"]) == {"raw", "better"}
    assert set(summary["metric"]) >= {"aurc", "brier"}
    assert (summary["n_resamples"] == 5).all()
    assert (summary["n_valid"] <= summary["n_resamples"]).all()
