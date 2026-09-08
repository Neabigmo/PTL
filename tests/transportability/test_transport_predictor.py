import pandas as pd
import pytest

from src.ptl.reliability.transport import leave_target_environment_out, source_selection_metrics


DESCRIPTORS = {
    "same_cell_context": 0,
    "same_perturbation_modality": 0,
    "same_platform": 1,
    "same_condition": 0,
    "same_dataset_family": 0,
    "same_context_modality": 0,
    "prediction_geometry_distance": 1.0,
    "uq_distribution_distance": 1.0,
}


def pair(source: str, target: str, gain: float, split: int = 1) -> dict[str, object]:
    return {
        "source_environment_id": source,
        "target_environment_id": target,
        "transfer_gain": gain,
        "split_seed": split,
        **DESCRIPTORS,
    }


def test_strict_target_unseen_excludes_target_as_source_and_self_selection() -> None:
    frame = pd.DataFrame([
        pair("a", "b", 0.2), pair("b", "a", -0.1),
        pair("a", "c", -0.2), pair("c", "a", 0.1),
        pair("b", "c", 0.3), pair("c", "b", -0.2),
    ])
    scored, summary = leave_target_environment_out(frame, probability_threshold=0.5, permutation_repeats=0)
    assert scored["strict_target_unseen"].eq(1).all()
    for target, group in scored.groupby("target_environment_id"):
        assert not group["source_environment_id"].eq(target).any()
    selection = pd.DataFrame(summary["selection"])
    assert not selection["selected_source_environment_id"].eq(selection["target_environment_id"]).any()


def test_fallback_is_real_when_no_candidate_is_accepted() -> None:
    frame = pd.DataFrame([
        pair("a", "b", -0.2), pair("b", "a", -0.1),
        pair("a", "c", -0.3), pair("c", "a", -0.2),
        pair("b", "c", -0.4), pair("c", "b", -0.3),
    ])
    _, summary = leave_target_environment_out(frame, probability_threshold=0.99, permutation_repeats=0)
    selection = pd.DataFrame(summary["selection"])
    assert selection["fallback_to_u_only"].eq(1).all()
    assert summary["fallback_frequency"] == 1.0
    assert summary["selected_source_negative_transfer_rate"] != summary["accepted_pair_negative_transfer_rate"] or pd.isna(summary["selected_source_negative_transfer_rate"])


def test_diagonal_pairs_are_rejected_before_training() -> None:
    frame = pd.DataFrame([pair("a", "a", 0.2), pair("a", "b", 0.1), pair("b", "a", 0.1)])
    with pytest.raises(AssertionError, match="off-diagonal"):
        leave_target_environment_out(frame, permutation_repeats=0)


def test_source_selection_metrics_are_stratified_by_target_and_split() -> None:
    scored = pd.DataFrame([
        {"target_environment_id": "t1", "split_seed": 1, "source_environment_id": "s1", "predicted_gain": 0.3, "positive_transfer_probability": 0.8, "transfer_gain": 0.2},
        {"target_environment_id": "t1", "split_seed": 1, "source_environment_id": "s2", "predicted_gain": -0.1, "positive_transfer_probability": 0.2, "transfer_gain": -0.2},
        {"target_environment_id": "t1", "split_seed": 1, "source_environment_id": "s3", "predicted_gain": 0.1, "positive_transfer_probability": 0.6, "transfer_gain": 0.05},
        {"target_environment_id": "t2", "split_seed": 1, "source_environment_id": "s1", "predicted_gain": -0.2, "positive_transfer_probability": 0.3, "transfer_gain": -0.1},
        {"target_environment_id": "t2", "split_seed": 1, "source_environment_id": "s2", "predicted_gain": 0.2, "positive_transfer_probability": 0.7, "transfer_gain": 0.1},
        {"target_environment_id": "t2", "split_seed": 1, "source_environment_id": "s3", "predicted_gain": 0.05, "positive_transfer_probability": 0.55, "transfer_gain": 0.02},
    ])
    selection = pd.DataFrame([
        {"target_environment_id": "t1", "split_seed": 1, "selected_source_environment_id": "s1", "selected_source_positive": 1, "selection_regret": 0.0},
        {"target_environment_id": "t2", "split_seed": 1, "selected_source_environment_id": "s2", "selected_source_positive": 1, "selection_regret": 0.0},
    ])
    detail, aggregate = source_selection_metrics(scored, selection, permutation_repeats=50)
    assert set(detail["target_environment_id"]) == {"t1", "t2"}
    assert set(detail["split_seed"]) == {1}
    assert detail["selected_source_rank"].tolist() == [1.0, 1.0]
    assert aggregate["target_split_group_count"] == 2.0
    assert 0.0 <= aggregate["macro_spearman_permutation_p_value"] <= 1.0
