import numpy as np
import pandas as pd

from scripts.run_formal_v2_reliability_shift import (
    _hierarchical_bootstrap_draw,
    _hierarchical_ci,
    crossfit_additive_and_interaction,
)


def _synthetic_frame(interaction: bool) -> pd.DataFrame:
    rows = []
    for environment_index, environment in enumerate(("e1", "e2")):
        for replicate in range(18):
            confidence_bin = replicate % 3
            baseline = 0.1 if environment == "e1" else 0.5
            slope = 0.1 if not interaction or environment == "e1" else 0.4
            rows.append({
                "environment_id": environment,
                "confidence_bin": confidence_bin,
                "continuous_risk": baseline + slope * confidence_bin,
                "biological_instance_id": f"{environment}_{replicate}",
            })
    return pd.DataFrame(rows)


def test_additive_difficulty_only_data_does_not_need_environment_interaction() -> None:
    oof = crossfit_additive_and_interaction(_synthetic_frame(interaction=False), n_splits=5)
    assert abs(float(oof["interaction_improvement"].mean())) < 0.02


def test_environment_confidence_interaction_is_detected() -> None:
    oof = crossfit_additive_and_interaction(_synthetic_frame(interaction=True), n_splits=5)
    assert float(oof["interaction_improvement"].mean()) > 0.01


class _DeterministicRng:
    def integers(self, low: int, high: int, size: int) -> np.ndarray:
        assert (low, high) == (0, 2)
        return np.zeros(size, dtype=int)

    def choice(self, values: list[str], size: int, replace: bool) -> np.ndarray:
        assert replace is True
        return np.asarray(values[:size], dtype=object)


def test_hierarchical_draw_uses_global_positions_and_keeps_duplicate_clusters() -> None:
    oof = pd.DataFrame(
        {
            "environment_id": ["e1", "e1", "e2", "e2"],
            "crossfit_group": ["e1_a", "e1_b", "e2_a", "e2_b"],
            "interaction_improvement": [0.2, 0.2, 0.8, 0.8],
        },
        index=[11, 12, 41, 42],
    )

    positions, clusters = _hierarchical_bootstrap_draw(oof, _DeterministicRng())
    sampled = oof.iloc[positions]

    assert set(sampled["environment_id"]) == {"e1"}
    assert len(set(clusters.tolist())) == 2
    assert all(cluster.split("__bootstrap_cluster_")[0] == environment for cluster, environment in zip(clusters, sampled["environment_id"]))


def test_hierarchical_ci_respects_environment_level_resampling() -> None:
    oof = pd.DataFrame(
        {
            "environment_id": ["e1", "e1", "e2", "e2"],
            "crossfit_group": ["e1_a", "e1_b", "e2_a", "e2_b"],
            "interaction_improvement": [0.2, 0.2, 0.8, 0.8],
        }
    )

    center, lower, upper = _hierarchical_ci(oof, replicates=500, random_state=7)

    assert 0.4 < center < 0.6
    assert lower <= center <= upper


def test_hierarchical_ci_has_expected_null_and_positive_behavior() -> None:
    null_oof = pd.DataFrame(
        {
            "environment_id": ["e1", "e1", "e2", "e2"],
            "crossfit_group": ["e1_a", "e1_b", "e2_a", "e2_b"],
            "interaction_improvement": [0.0, 0.0, 0.0, 0.0],
        }
    )
    positive_oof = null_oof.assign(interaction_improvement=0.1)

    _, null_lower, null_upper = _hierarchical_ci(null_oof, replicates=200, random_state=11)
    _, positive_lower, positive_upper = _hierarchical_ci(positive_oof, replicates=200, random_state=11)

    assert null_lower <= 0.0 <= null_upper
    assert positive_lower > 0.0 and positive_upper >= positive_lower
