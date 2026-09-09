"""Build an outcome-blind registry of candidate matched-context replications.

Only H5AD observation/variable metadata and predeclared study-design evidence
are inspected.  No predictions, responses, risks, inversion statistics, or
target outcomes are loaded.  Batch/context provenance is an evidence-driven
three-state decision, rather than a default rejection of every
context-separated export.
"""

from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path
import re
import sys
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data/raw/scperturb_v1.4"
MIN_SHARED_PERTURBATIONS = 200
MIN_CELLS_PER_PERTURBATION = 20
MIN_GUIDE_SIGNATURE_FRACTION = 0.95
CONTROL_ALIASES = {"control", "ctrl", "non-targeting", "non-targeting control", "ntc"}
PROVENANCE_STATUSES = {
    "supported_independent_context",
    "unresolved",
    "confounded",
}


# These are outcome-blind study-design audits.  They are deliberately kept
# separate from the H5AD outcome metadata so that a candidate can only become
# eligible after the experimental provenance has been checked, never because
# its prediction or risk result looks attractive.
PROVENANCE_AUDITS: dict[str, dict[str, Any]] = {
    "nadig_hepg2_vs_jurkat": {
        "batch_context_status": "supported_independent_context",
        "evidence_source": (
            "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE264667"
        ),
        "evidence_type": "GEO series design and sample metadata",
        "batch_id_context_mapping": (
            "H5AD batch/gem-group IDs are context-local; the GEO design states "
            "that Jurkat and HepG2 screens were conducted in parallel."
        ),
        "batch_nested_in_context": True,
        "technical_replicates_exist": True,
        "technical_replicate_definition": "multiple context-indexed 10x/sample batch units",
        "context_is_intended_biological_factor": True,
        "context_comparison_in_original_study_design": True,
        "audit_notes": (
            "GEO describes two parallel pooled CRISPR screens with scRNA-seq "
            "readout, the same dual-sgRNA library family, low infection rate, "
            "day-7 harvest, and 10x Genomics Chromium 3-prime v3. The local "
            "exports contain one cell line per file and 56/55 context-local "
            "batch IDs; repeated numeric IDs are not treated as shared physical "
            "batches."
        ),
    },
    "replogle_k562_essential_vs_rpe1": {
        "batch_context_status": "confounded",
        "evidence_source": (
            "https://plus.figshare.com/articles/dataset/"
            "_Mapping_information-rich_genotype-phenotype_landscapes_with_genome-scale_Perturb-seq_Replogle_et_al_2022_processed_Perturb-seq_datasets/20029387"
        ),
        "evidence_type": "Figshare dataset design and associated Cell study",
        "batch_id_context_mapping": (
            "The public dataset identifies K562 essential as day 6 and RPE1 "
            "essential as day 7 post-transduction; the timepoint is inseparable "
            "from the context comparison in these exports."
        ),
        "batch_nested_in_context": False,
        "technical_replicates_exist": True,
        "technical_replicate_definition": "multiple gem-group/sample batch units",
        "context_is_intended_biological_factor": True,
        "context_comparison_in_original_study_design": True,
        "audit_notes": (
            "Same study/platform and CRISPRi Perturb-seq family, but the public "
            "processed data explicitly uses different harvest days (K562 day 6, "
            "RPE1 day 7). Without an additional matched-timepoint audit this is "
            "an irreducible context/time confound, so it cannot be headline "
            "replication evidence."
        ),
    },
}


CANDIDATES: tuple[dict[str, Any], ...] = (
    {
        "candidate_id": "nadig_hepg2_vs_jurkat",
        "study": "NadigOConner2024",
        "context_a": "Hep-G2",
        "context_b": "Jurkat",
        "path_a": "data/raw/scperturb_v1.4/NadigOConner2024_hepg2.h5ad",
        "path_b": "data/raw/scperturb_v1.4/NadigOConner2024_jurkat.h5ad",
        "perturbation_modality_expected": "CRISPR",
        "readout_modality_expected": "RNA",
        "same_library_claim": "same study family; exact guide signatures audited from obs metadata",
        "batch_context_evidence": "see predeclared GEO study-design audit",
    },
    {
        "candidate_id": "replogle_k562_essential_vs_rpe1",
        "study": "ReplogleWeissman2022",
        "context_a": "K562",
        "context_b": "RPE1",
        "path_a": "data/raw/scperturb_v1.4/ReplogleWeissman2022_K562_essential.h5ad",
        "path_b": "data/raw/scperturb_v1.4/ReplogleWeissman2022_rpe1.h5ad",
        "perturbation_modality_expected": "CRISPR",
        "readout_modality_expected": "RNA",
        "same_library_claim": "same study family; exact guide signatures audited from obs metadata",
        "batch_context_evidence": "see predeclared Figshare study-design audit",
    },
    {
        "candidate_id": "drepanos_k562_vs_a549",
        "study": "Drepanos2026",
        "context_a": "K562",
        "context_b": "A549",
        "path_a": None,
        "path_b": None,
        "perturbation_modality_expected": "CRISPRko or CRISPRi",
        "readout_modality_expected": "RNA",
        "same_library_claim": "not audited: no local Drepanos H5AD found",
        "batch_context_evidence": "not audited: no local Drepanos H5AD found",
    },
    {
        "candidate_id": "drepanos_k562_ko_vs_crispri",
        "study": "Drepanos2026",
        "context_a": "K562 CRISPRko",
        "context_b": "K562 CRISPRi",
        "path_a": None,
        "path_b": None,
        "perturbation_modality_expected": "CRISPRko vs CRISPRi",
        "readout_modality_expected": "RNA",
        "same_library_claim": "not audited: no local Drepanos H5AD found",
        "batch_context_evidence": "not audited: no local Drepanos H5AD found",
    },
)


def _text(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("").str.strip()


def _is_control(value: str) -> bool:
    return value.casefold() in CONTROL_ALIASES


def _guide_signature(values: pd.Series) -> tuple[str, ...]:
    return tuple(sorted({value for value in _text(values) if value}))


def _load_context(path: Path) -> dict[str, Any]:
    """Load only obs/var metadata from a backed H5AD."""

    dataset = ad.read_h5ad(path, backed="r")
    try:
        # Read only outcome-blind observation metadata needed for the gate.
        # In particular, never materialize expression values or unused obs columns.
        required = {"perturbation", "guide_id", "batch", "cell_line", "perturbation_type"}
        missing = sorted(required.difference(dataset.obs.columns))
        if missing:
            raise ValueError(f"{path.name} is missing metadata columns: {missing}")
        obs = dataset.obs[sorted(required)].copy()
        shape = [int(dataset.n_obs), int(dataset.n_vars)]
        var_names = int(dataset.n_vars)
    finally:
        dataset.file.close()
    obs = obs.copy()
    obs["perturbation"] = _text(obs["perturbation"])
    obs["guide_id"] = _text(obs["guide_id"])
    obs["batch"] = _text(obs["batch"])
    obs["cell_line"] = _text(obs["cell_line"])
    obs["perturbation_type"] = _text(obs["perturbation_type"])
    obs = obs.loc[obs["perturbation"].ne("")].reset_index(drop=True)
    counts = obs["perturbation"].value_counts()
    target_counts = counts.loc[[label for label in counts.index if not _is_control(label)]]
    signatures = {
        label: _guide_signature(obs.loc[obs["perturbation"].eq(label), "guide_id"])
        for label in target_counts.index
    }
    return {
        "path": path.as_posix(),
        "shape": shape,
        "obs": obs,
        "cell_line_values": sorted(value for value in obs["cell_line"].unique() if value),
        "perturbation_type_values": sorted(value for value in obs["perturbation_type"].unique() if value),
        "batch_count": int(obs["batch"].nunique()),
        "batch_values": sorted(value for value in obs["batch"].unique() if value),
        "control_cell_count": int(sum(counts.get(alias, 0) for alias in CONTROL_ALIASES)),
        "target_counts": target_counts.astype(int).to_dict(),
        "guide_signatures": signatures,
        "var_count": var_names,
    }


def _empty_pair_record(candidate: dict[str, Any], reason: str) -> dict[str, Any]:
    provenance = _provenance_details(candidate)
    readout_a = candidate.get("readout_modality_a", candidate.get("readout_modality_expected", "unknown"))
    readout_b = candidate.get("readout_modality_b", candidate.get("readout_modality_expected", "unknown"))
    return {
        "candidate_id": candidate["candidate_id"],
        "study": candidate["study"],
        "context_a": candidate["context_a"],
        "context_b": candidate["context_b"],
        "dataset_a": candidate["path_a"],
        "dataset_b": candidate["path_b"],
        "file_a_available": False,
        "file_b_available": False,
        "exact_shared_perturbation_count": 0,
        "raw_cells_a": 0,
        "raw_cells_b": 0,
        "control_cells_a": 0,
        "control_cells_b": 0,
        "median_min_cells_per_shared_perturbation": np.nan,
        "fraction_shared_ge20_cells": np.nan,
        "fraction_shared_ge40_cells": np.nan,
        "same_perturbation_modality": False,
        "perturbation_modality_a": "",
        "perturbation_modality_b": "",
        "readout_modality_a": readout_a,
        "readout_modality_b": readout_b,
        "same_readout_modality": False,
        "same_library_evidence": False,
        "exact_guide_signature_fraction": np.nan,
        "batch_id_overlap_count": 0,
        "obvious_batch_context_confound": provenance["batch_context_status"] == "confounded",
        "batch_context_status": provenance["batch_context_status"],
        "batch_context_evidence": candidate["batch_context_evidence"],
        "provenance_evidence_source": provenance["evidence_source"],
        "provenance_evidence_type": provenance["evidence_type"],
        "batch_id_context_mapping": provenance["batch_id_context_mapping"],
        "batch_nested_in_context": provenance["batch_nested_in_context"],
        "technical_replicates_exist": provenance["technical_replicates_exist"],
        "context_is_intended_biological_factor": provenance["context_is_intended_biological_factor"],
        "context_comparison_in_original_study_design": provenance["context_comparison_in_original_study_design"],
        "provenance_audit_notes": provenance["audit_notes"],
        "outcome_blind": True,
        "prediction_evaluated": False,
        "risk_evaluated": False,
        "eligible": False,
        "reason": reason,
    }


def _pair_record(candidate: dict[str, Any], left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    provenance = _provenance_details(candidate)
    left_targets = set(left["target_counts"])
    right_targets = set(right["target_counts"])
    shared = sorted(left_targets & right_targets)
    left_counts = pd.Series(left["target_counts"], dtype=float)
    right_counts = pd.Series(right["target_counts"], dtype=float)
    minimum_counts = np.minimum(left_counts.reindex(shared).to_numpy(), right_counts.reindex(shared).to_numpy())
    left_types = set(left["perturbation_type_values"])
    right_types = set(right["perturbation_type_values"])
    same_modality = bool(left_types and right_types and left_types == right_types)
    readout_a = candidate.get("readout_modality_a", candidate.get("readout_modality_expected", "unknown"))
    readout_b = candidate.get("readout_modality_b", candidate.get("readout_modality_expected", "unknown"))
    same_readout = bool(readout_a == readout_b and readout_a not in {None, "", "unknown"})
    signature_matches = [
        left["guide_signatures"].get(label, ()) == right["guide_signatures"].get(label, ())
        for label in shared
    ]
    signature_fraction = float(np.mean(signature_matches)) if signature_matches else np.nan
    same_library = bool(np.isfinite(signature_fraction) and signature_fraction >= MIN_GUIDE_SIGNATURE_FRACTION)
    batch_overlap = len(set(left["batch_values"]) & set(right["batch_values"]))
    batch_context_status = provenance["batch_context_status"]
    if batch_context_status not in PROVENANCE_STATUSES:
        raise ValueError(f"unknown batch/context provenance status: {batch_context_status!r}")
    batch_confound = batch_context_status == "confounded"
    eligible = bool(
        len(shared) >= MIN_SHARED_PERTURBATIONS
        and left["control_cell_count"] > 0
        and right["control_cell_count"] > 0
        and same_modality
        and same_readout
        and same_library
        and float(np.median(minimum_counts)) >= MIN_CELLS_PER_PERTURBATION
        and batch_context_status == "supported_independent_context"
    )
    reasons = []
    if len(shared) < MIN_SHARED_PERTURBATIONS:
        reasons.append(f"shared perturbations {len(shared)} < {MIN_SHARED_PERTURBATIONS}")
    if not left["control_cell_count"] or not right["control_cell_count"]:
        reasons.append("matched controls are unavailable")
    if not same_modality:
        reasons.append("perturbation modality is not identical")
    if not same_readout:
        reasons.append("readout modality is not identical or not documented")
    if not same_library:
        reasons.append("exact guide-signature evidence is below threshold")
    if not shared or float(np.median(minimum_counts)) < MIN_CELLS_PER_PERTURBATION:
        reasons.append("median matched cell count is below threshold")
    if batch_context_status == "unresolved":
        reasons.append("batch/context provenance is unresolved")
    elif batch_context_status == "confounded":
        reasons.append("batch/context provenance is confounded with the intended context")
    return {
        "candidate_id": candidate["candidate_id"],
        "study": candidate["study"],
        "context_a": candidate["context_a"],
        "context_b": candidate["context_b"],
        "dataset_a": left["path"],
        "dataset_b": right["path"],
        "file_a_available": True,
        "file_b_available": True,
        "exact_shared_perturbation_count": len(shared),
        "raw_cells_a": left["shape"][0],
        "raw_cells_b": right["shape"][0],
        "control_cells_a": left["control_cell_count"],
        "control_cells_b": right["control_cell_count"],
        "median_min_cells_per_shared_perturbation": float(np.median(minimum_counts)) if shared else np.nan,
        "fraction_shared_ge20_cells": float(np.mean(minimum_counts >= 20)) if shared else np.nan,
        "fraction_shared_ge40_cells": float(np.mean(minimum_counts >= 40)) if shared else np.nan,
        "same_perturbation_modality": same_modality,
        "perturbation_modality_a": ";".join(sorted(left_types)),
        "perturbation_modality_b": ";".join(sorted(right_types)),
        "readout_modality_a": readout_a,
        "readout_modality_b": readout_b,
        "same_readout_modality": same_readout,
        "same_library_evidence": same_library,
        "exact_guide_signature_fraction": signature_fraction,
        "batch_id_overlap_count": batch_overlap,
        "obvious_batch_context_confound": batch_confound,
        "batch_context_status": batch_context_status,
        "batch_context_evidence": candidate["batch_context_evidence"],
        "provenance_evidence_source": provenance["evidence_source"],
        "provenance_evidence_type": provenance["evidence_type"],
        "batch_id_context_mapping": provenance["batch_id_context_mapping"],
        "batch_nested_in_context": provenance["batch_nested_in_context"],
        "technical_replicates_exist": provenance["technical_replicates_exist"],
        "context_is_intended_biological_factor": provenance["context_is_intended_biological_factor"],
        "context_comparison_in_original_study_design": provenance["context_comparison_in_original_study_design"],
        "provenance_audit_notes": provenance["audit_notes"],
        "outcome_blind": True,
        "prediction_evaluated": False,
        "risk_evaluated": False,
        "eligible": eligible,
        "reason": "eligible by metadata gate" if eligible else "; ".join(reasons),
    }


def _provenance_details(candidate: dict[str, Any]) -> dict[str, Any]:
    """Return a predeclared outcome-blind provenance decision.

    Missing external evidence is ``unresolved``.  It is never silently
    promoted to ``confounded`` and never allowed into headline replication.
    """

    audit = PROVENANCE_AUDITS.get(str(candidate.get("candidate_id", "")), {})
    status = str(audit.get("batch_context_status", "unresolved"))
    if status not in PROVENANCE_STATUSES:
        raise ValueError(f"unknown batch/context provenance status: {status!r}")
    return {
        "batch_context_status": status,
        "evidence_source": str(audit.get("evidence_source", "not audited")),
        "evidence_type": str(audit.get("evidence_type", "no external provenance audit")),
        "batch_id_context_mapping": str(audit.get("batch_id_context_mapping", "not established")),
        "batch_nested_in_context": audit.get("batch_nested_in_context"),
        "technical_replicates_exist": audit.get("technical_replicates_exist"),
        "context_is_intended_biological_factor": audit.get("context_is_intended_biological_factor"),
        "context_comparison_in_original_study_design": audit.get("context_comparison_in_original_study_design"),
        "audit_notes": str(audit.get("audit_notes", "No predeclared study-design evidence was found.")),
    }


def audit_candidate(root: Path, candidate: dict[str, Any], cache: dict[Path, dict[str, Any] | None] | None = None) -> dict[str, Any]:
    path_a = root / candidate["path_a"] if candidate["path_a"] else None
    path_b = root / candidate["path_b"] if candidate["path_b"] else None
    if path_a is None or path_b is None or not path_a.is_file() or not path_b.is_file():
        missing = [str(path) for path in (path_a, path_b) if path is None or not path.is_file()]
        return _empty_pair_record(candidate, "local metadata files unavailable: " + ", ".join(missing))
    cache = {} if cache is None else cache
    for path in (path_a, path_b):
        if path not in cache:
            try:
                cache[path] = _load_context(path)
            except (OSError, ValueError) as exc:
                cache[path] = None
                return _empty_pair_record(candidate, f"local metadata unavailable or incomplete: {exc}")
    left = cache[path_a]
    right = cache[path_b]
    if left is None or right is None:
        return _empty_pair_record(candidate, "local metadata unavailable or incomplete in one or both files")
    return _pair_record(candidate, left, right)


def _study_prefix(path: Path) -> str:
    match = re.match(r"^(.+?20[0-9][0-9])", path.stem)
    return match.group(1) if match else path.stem


def _discover_local_candidates(root: Path) -> list[dict[str, Any]]:
    """Enumerate same-study H5AD pairs without looking at response values."""

    known_pairs = {
        frozenset((candidate.get("path_a"), candidate.get("path_b")))
        for candidate in CANDIDATES
        if candidate.get("path_a") and candidate.get("path_b")
    }
    grouped: dict[str, list[Path]] = {}
    raw_dir = root / "data/raw/scperturb_v1.4"
    for path in sorted(raw_dir.glob("*.h5ad")):
        if path.is_file():
            grouped.setdefault(_study_prefix(path), []).append(path)
    discovered: list[dict[str, Any]] = []
    for study, paths in sorted(grouped.items()):
        if len(paths) < 2:
            continue
        for left, right in combinations(paths, 2):
            rel_left = left.relative_to(root).as_posix()
            rel_right = right.relative_to(root).as_posix()
            if frozenset((rel_left, rel_right)) in known_pairs:
                continue
            readout_left = "protein" if "protein" in left.stem.casefold() else "RNA"
            readout_right = "protein" if "protein" in right.stem.casefold() else "RNA"
            slug = re.sub(r"[^a-z0-9]+", "_", f"{study}_{left.stem}_{right.stem}".casefold()).strip("_")
            discovered.append({
                "candidate_id": f"local_{slug}",
                "study": study,
                "context_a": left.stem,
                "context_b": right.stem,
                "path_a": rel_left,
                "path_b": rel_right,
                "perturbation_modality_expected": "unknown_from_filename",
                "readout_modality_a": readout_left,
                "readout_modality_b": readout_right,
                "same_library_claim": "not prespecified; exact guide signatures audited from obs metadata",
                "batch_context_evidence": "auto-discovered same-study file pair; cross-context batch provenance is not inferred from outcomes",
            })
    return discovered


def build_registry(root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    root = root.resolve()
    candidates = [*CANDIDATES, *_discover_local_candidates(root)]
    cache: dict[Path, dict[str, Any]] = {}
    records = [audit_candidate(root, candidate, cache) for candidate in candidates]
    frame = pd.DataFrame(records)
    eligible = frame.loc[frame["eligible"]].sort_values(
        ["same_library_evidence", "exact_shared_perturbation_count"], ascending=[False, False], kind="stable"
    )
    raw_dir = root / "data/raw/scperturb_v1.4"
    all_local_h5ad = sorted(path.relative_to(root).as_posix() for path in raw_dir.glob("*.h5ad") if path.is_file())
    candidate_columns = [
        "candidate_id", "eligible", "exact_shared_perturbation_count",
        "median_min_cells_per_shared_perturbation", "same_library_evidence",
        "same_readout_modality", "batch_context_status",
        "obvious_batch_context_confound", "provenance_evidence_source", "reason",
    ]
    candidate_summary = frame[candidate_columns].astype(object).where(pd.notna(frame[candidate_columns]), None)
    report = {
        "schema_version": 1,
        "benchmark_id": "ptl_context_v2_outcome_blind_replication_registry_v1",
        "status": "eligible_candidate_found" if not eligible.empty else "no_candidate_passed_outcome_blind_metadata_gate",
        "outcome_blind": True,
        "prediction_evaluated": False,
        "risk_evaluated": False,
        "selection_materialized_before_risk_evaluation": True,
        "eligibility_gate": {
            "exact_shared_perturbations_min": MIN_SHARED_PERTURBATIONS,
            "raw_cell_level_observations_required": True,
            "matched_control_required": True,
            "same_modality_required": True,
            "same_readout_modality_required": True,
            "same_or_demonstrably_matched_library_required": True,
            "median_cells_per_perturbation_min": MIN_CELLS_PER_PERTURBATION,
            "obvious_batch_context_confound_allowed": False,
        },
        "candidate_count": int(len(frame)),
        "eligible_candidate_count": int(len(eligible)),
        "selected_candidate": None if eligible.empty else str(eligible.iloc[0]["candidate_id"]),
        "local_h5ad_inventory": all_local_h5ad,
        "registry_csv": "artifacts/manifests/reordering_replication_candidate_registry.csv",
        "candidate_summary": candidate_summary.to_dict("records"),
        "provenance_audits": {
            candidate_id: audit for candidate_id, audit in PROVENANCE_AUDITS.items()
        },
    }
    return frame, report


def write_replication_boundary(root: Path, report: dict[str, Any]) -> Path:
    """Materialize the claim boundary from the frozen metadata-only registry."""

    output_path = root.resolve() / "artifacts/manifests/formal_v2_claim_lock_replication_boundary.json"
    if report["eligible_candidate_count"]:
        status = "eligible_candidate_frozen_pending_claim_lock"
        boundary = (
            "an outcome-blind metadata gate identified a candidate, but no independent "
            "risk or inversion evaluation is authorized until the exact Claim Lock "
            "protocol is executed once on the frozen candidate"
        )
        selected = report["selected_candidate"]
    else:
        status = "no_valid_independent_matched_context_replication_available"
        boundary = (
            "the outcome-blind registry found no local candidate passing the exact "
            "shared-perturbation, raw-cell, matched-control, modality, library and "
            "batch/context provenance gate; no independent replication claim is made"
        )
        selected = None
    payload = {
        "schema_version": 1,
        "benchmark_id": "ptl_context_v2_frangieh_claim_lock_v1",
        "status": status,
        "boundary": boundary,
        "policy": "do not force a replication claim; report the boundary explicitly and do not claim universality",
        "registry": report["registry_json"],
        "registry_sha256": report["registry_sha256"],
        "selected_candidate": selected,
        "prediction_evaluated": False,
        "risk_evaluated": False,
        "optional_candidate": "GEARS remains optional only if a legal same-predictor/source-outcome contract is established; no new model training was introduced",
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    frame, report = build_registry(args.root)
    output_dir = args.root.resolve() / "artifacts/manifests"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "reordering_replication_candidate_registry.csv"
    json_path = output_dir / "reordering_replication_candidate_registry.json"
    frame.to_csv(csv_path, index=False)
    report["registry_sha256"] = __import__("hashlib").sha256(csv_path.read_bytes()).hexdigest()
    report["registry_json"] = json_path.relative_to(args.root.resolve()).as_posix()
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    boundary_path = write_replication_boundary(args.root, report)
    report["boundary_json"] = boundary_path.relative_to(args.root.resolve()).as_posix()
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
