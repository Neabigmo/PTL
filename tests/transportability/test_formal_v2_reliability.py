import numpy as np
import pandas as pd
import pytest

from scripts.run_formal_v2_reliability import (
    FEATURE_COLUMNS,
    scenario_partition,
    validate_deployment_features,
)


def toy_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "biological_instance_id": ["b1", "b2", "b3", "b4", "b5", "b6"],
            "environment_id": ["e1", "e1", "e1", "e2", "e2", "e2"],
            "predictor": ["p1", "p1", "p2", "p1", "p1", "p2"],
            "split": ["validation", "validation", "validation", "validation", "test", "test"],
        }
    )


def test_formal_deployment_features_exclude_outcomes_and_identity() -> None:
    validate_deployment_features(FEATURE_COLUMNS)
    assert "environment_id" not in FEATURE_COLUMNS
    assert "predictor" not in FEATURE_COLUMNS
    assert all("fidelity" not in column for column in FEATURE_COLUMNS)


@pytest.mark.parametrize("scenario,heldout", [("in_domain", ""), ("leave_environment_out", "e2"), ("leave_predictor_out", "p2")])
def test_scenario_partition_is_validation_to_test_and_respects_holdout(scenario: str, heldout: str) -> None:
    frame = toy_rows()
    train, test = scenario_partition(frame, scenario, heldout)
    assert set(frame.loc[train, "split"]) == {"validation"}
    assert set(frame.loc[test, "split"]) == {"test"}
    assert set(frame.loc[train, "biological_instance_id"]).isdisjoint(
        set(frame.loc[test, "biological_instance_id"])
    )
    if scenario == "leave_environment_out":
        assert set(frame.loc[test, "environment_id"]) == {heldout}
        assert heldout not in set(frame.loc[train, "environment_id"])
    if scenario == "leave_predictor_out":
        assert set(frame.loc[test, "predictor"]) == {heldout}
        assert heldout not in set(frame.loc[train, "predictor"])


def test_scenario_partition_rejects_unknown_holdout() -> None:
    with pytest.raises(ValueError, match="unknown held-out"):
        scenario_partition(toy_rows(), "leave_environment_out", "missing")
