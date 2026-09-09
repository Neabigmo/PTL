"""Contract tests for the selected matched-context Claim Lock runner."""

import numpy as np

from scripts.run_formal_v2_claim_lock_replication import (
    _fold_ids,
    _seed_plan,
    _truths_from_lookup,
)


def test_global_folds_are_deterministic_and_balanced() -> None:
    labels = [f"g{index}" for index in range(2392)]
    first = _fold_ids(labels)
    second = _fold_ids(labels)
    assert np.array_equal(first, second)
    assert np.bincount(first).tolist() == [479, 479, 478, 478, 478]


def test_seed_plan_keeps_each_half_disjoint_and_budgeted() -> None:
    context = {
        "groups": {
            "A": np.arange(80, dtype=np.int64),
            "control": np.arange(80, 160, dtype=np.int64),
        }
    }
    budgets = {20: {"A": 20, "control": 20}, 40: {"A": 40, "control": 40}}
    plan = _seed_plan(context, ["A"], budgets, 20260908, "synthetic")
    by_key = dict(zip(plan["virtual_keys"], plan["selected_rows"]))
    for minimum in (20, 40):
        for label in ("A", "control"):
            left = by_key[(minimum, label, 0, minimum)]
            right = by_key[(minimum, label, 1, minimum)]
            assert len(left) == len(right) == minimum
            assert set(left).isdisjoint(set(right))


def test_truth_lookup_is_indexed_by_the_requested_minimum() -> None:
    labels = ["A"]
    lookup = {}
    for minimum, value in ((20, 2.0), (40, 4.0)):
        for half in (0, 1):
            lookup[(minimum, "A", half, minimum)] = np.full(3, value + half, dtype=np.float32)
            lookup[(minimum, "control", half, minimum)] = np.full(3, 1.0, dtype=np.float32)
    primary = _truths_from_lookup(lookup, labels, {"A": 20, "control": 20}, 20)
    sensitivity = _truths_from_lookup(lookup, labels, {"A": 40, "control": 40}, 40)
    assert np.allclose(primary[("nadig_hepg2", 0)], 1.0)
    assert np.allclose(sensitivity[("nadig_hepg2", 0)], 3.0)
