"""Unit tests for the outcome-blind replication registry."""

from __future__ import annotations

import scripts.build_reordering_replication_candidate_registry as registry


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
    record = registry._pair_record(candidate, _context(labels), _context(labels))
    assert record["exact_shared_perturbation_count"] == 210
    assert record["same_library_evidence"] is True
    assert record["same_readout_modality"] is True
    assert record["outcome_blind"] is True
    assert record["prediction_evaluated"] is False
    assert record["risk_evaluated"] is False
    assert record["eligible"] is False
    assert record["batch_context_status"] == "unresolved"
    assert "batch/context provenance" in record["reason"]


def test_supported_provenance_is_eligible_and_confounded_is_not(monkeypatch) -> None:
    labels = [f"GENE_{index}" for index in range(210)]
    base = {
        "study": "synthetic",
        "context_a": "A",
        "context_b": "B",
        "path_a": "a.h5ad",
        "path_b": "b.h5ad",
        "readout_modality_expected": "RNA",
        "batch_context_evidence": "predeclared study audit",
    }
    audit = {
        "evidence_source": "synthetic",
        "evidence_type": "unit test",
        "batch_id_context_mapping": "batch nested in context",
        "batch_nested_in_context": True,
        "technical_replicates_exist": True,
        "context_is_intended_biological_factor": True,
        "context_comparison_in_original_study_design": True,
        "audit_notes": "test evidence",
    }
    monkeypatch.setitem(registry.PROVENANCE_AUDITS, "synthetic_supported", {**audit, "batch_context_status": "supported_independent_context"})
    monkeypatch.setitem(registry.PROVENANCE_AUDITS, "synthetic_confounded", {**audit, "batch_context_status": "confounded"})
    left = _context(labels)
    right = _context(labels)
    for candidate_id, expected_status, expected_eligible in [
        ("synthetic_supported", "supported_independent_context", True),
        ("synthetic_confounded", "confounded", False),
    ]:
        record = registry._pair_record({**base, "candidate_id": candidate_id}, left, right)
        assert record["batch_context_status"] == expected_status
        assert record["eligible"] is expected_eligible
        assert record["outcome_blind"] is True
        assert record["prediction_evaluated"] is False
        assert record["risk_evaluated"] is False


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
    record = registry.audit_candidate(tmp_path, candidate, {})
    assert record["eligible"] is False
    assert record["outcome_blind"] is True
    assert record["prediction_evaluated"] is False
    assert record["risk_evaluated"] is False
    assert "metadata files unavailable" in record["reason"]
