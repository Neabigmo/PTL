from __future__ import annotations

import pandas as pd
import pytest

from src.evaluation.evaluate_phase06 import resolve_phase05_primary_metric


def test_resolve_phase05_primary_metric_prefers_summary_table() -> None:
    phase05 = pd.DataFrame(
        {"mean_cosine_similarity_non_control_test": [0.61]},
        index=["run_a"],
    )
    value, source = resolve_phase05_primary_metric(
        "run_a",
        phase05,
        {"mean_cosine_similarity_non_control_test": 0.42},
    )

    assert value == 0.61
    assert source == "baseline_metrics_by_split"


def test_resolve_phase05_primary_metric_falls_back_to_run_metrics_json() -> None:
    phase05 = pd.DataFrame(
        {"mean_cosine_similarity_non_control_test": [0.61]},
        index=["other_run"],
    )
    value, source = resolve_phase05_primary_metric(
        "run_a",
        phase05,
        {"mean_cosine_similarity_non_control_test": 0.42},
    )

    assert value == 0.42
    assert source == "run_metrics_json"


def test_resolve_phase05_primary_metric_reads_nested_legacy_metrics() -> None:
    value, source = resolve_phase05_primary_metric(
        "run_a",
        pd.DataFrame(),
        {"metrics": {"mean_cosine_similarity_non_control_test": 0.33}},
    )

    assert value == 0.33
    assert source == "run_metrics_json.metrics"


def test_resolve_phase05_primary_metric_raises_when_missing() -> None:
    with pytest.raises(KeyError):
        resolve_phase05_primary_metric("run_a", pd.DataFrame(), {})
