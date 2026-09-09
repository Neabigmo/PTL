import numpy as np

from scripts.run_formal_v2_claim_lock_measurement import _synchronized_macro_rows


def test_synchronized_macro_aggregates_matching_draw_indices():
    rows = _synchronized_macro_rows(
        [
            {"metric": "m", "draws": {"ordering_delta_joint_id": np.asarray([0.0, 1.0, 2.0])}},
            {"metric": "m", "draws": {"ordering_delta_joint_id": np.asarray([2.0, 3.0, 4.0])}},
        ],
        group_columns=("metric",),
    )
    result = rows[0]
    assert np.isclose(result["ordering_delta_joint_id_macro"], 2.0)
    assert np.isclose(result["ordering_delta_joint_id_macro_ci_low"], 1.05)
    assert np.isclose(result["ordering_delta_joint_id_macro_ci_high"], 2.95)
    assert result["n_rows_aggregated"] == 2
    assert result["bootstrap_draws"] == 3
