import numpy as np
import pandas as pd

from scripts.run_formal_v2_reliability_shift import crossfit_additive_and_interaction


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
