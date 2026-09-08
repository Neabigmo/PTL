"""Unit tests for the outcome-blind replication registry."""

from __future__ import annotations

from scripts.build_reordering_replication_candidate_registry import _pair_record, audit_candidate


def _context(labels: list[str], *, guide_suffix: str = "g1") -> dict:
    return {
        "path": "synthetic.h5ad",
        "shape": [len(labels) * 20 + 10, 100],
        "cell_line_values": ["synthetic"],
        "perturbation_type_values": ["CRISPR"],
        "batch_count": 1,
        "batch_values": ["batch0"],
        "control_cell_count": 10,
        "target_counts": {label: 20 for label in labels},
        "guide_signatures": {label: (f"{label}_{guide_suffix}",) for label in labels},
        "var_count": 100,
    }


def test_pair_record_is_outcome_blind_and_rejects_unresolved_batch_context() -> None:
    labels = [f"GENE_{index}" for index in range(210)]
    candidate = {
        "candidate_id": "synthetic_pair",
        "study": "synthetic",
        "context_a": "A",
        "context_b": "B",
        "path_a": "a.h5ad",
        "path_b": "b.h5ad",
        "readout_modality_expected": "RNA",
        "batch_context_evidence": "not documented",
    }
    record = _pair_record(candidate, _context(labels), _context(labels))
    assert record["exact_shared_perturbation_count"] == 210
    assert record["same_library_evidence"] is True
    assert record["same_readout_modality"] is True
    assert record["outcome_blind"] is True
    assert record["prediction_evaluated"] is False
    assert record["risk_evaluated"] is False
    assert record["eligible"] is False
    assert "batch/context provenance" in record["reason"]


def test_missing_local_metadata_is_recorded_without_risk_evaluation(tmp_path) -> None:
    candidate = {
        "candidate_id": "missing_pair",
        "study": "synthetic",
        "context_a": "A",
        "context_b": "B",
        "path_a": "missing_a.h5ad",
        "path_b": "missing_b.h5ad",
        "readout_modality_expected": "RNA",
        "batch_context_evidence": "not documented",
    }
    record = audit_candidate(tmp_path, candidate, {})
    assert record["eligible"] is False
    assert record["outcome_blind"] is True
    assert record["prediction_evaluated"] is False
    assert record["risk_evaluated"] is False
    assert "metadata files unavailable" in record["reason"]
