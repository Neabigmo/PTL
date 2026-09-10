"""Materialize one auditable master table and claim registry."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MASTER_COLUMNS = (
    "dataset", "dataset_split", "evidence_tier", "source_environment_id", "target_environment_id",
    "left_target_environment_id", "right_target_environment_id", "source_target_orientation", "predictor",
    "predictor_information_regime", "regime", "metric", "cell_budget", "cell_budget_label", "universe_mode",
    "label_universe", "n", "observed_D", "observed_D_ci_low", "observed_D_ci_high",
    "measurement_identifiable", "measurement_identifiable_ci_low", "measurement_identifiable_ci_high",
    "joint_identifiable", "joint_identifiable_ci_low", "joint_identifiable_ci_high", "stable_fraction_both",
    "decision_budget_fraction", "decision_budget_ci_low", "decision_budget_ci_high", "retention",
    "retention_ci_low", "retention_ci_high", "regret", "regret_ci_low", "regret_ci_high",
    "normalized_regret", "normalized_regret_ci_low", "normalized_regret_ci_high", "boundary_inversion",
    "boundary_inversion_ci_low", "boundary_inversion_ci_high", "measurement_floor_regret", "joint_floor_regret",
    "excess_regret", "joint_excess_regret", "predictability_model", "predictability_dataset_split",
    "predictability_spearman", "predictability_auroc", "predictability_mae", "calibration_brier",
    "modern_claim_lock_status", "modern_claim_lock_blocker", "checksums", "provenance", "status",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _empty() -> dict:
    return {column: np.nan for column in MASTER_COLUMNS}


def _record_from_atlas(record: dict, manifests: Path) -> dict:
    row = _empty()
    provenance = str(record.get("provenance", ""))
    source = manifests / Path(provenance).name if provenance else None
    row.update({
        "dataset": record.get("dataset"), "evidence_tier": record.get("evidence_tier"),
        "source_environment_id": record.get("source_environment_id"), "target_environment_id": record.get("target_environment_id"),
        "left_target_environment_id": record.get("left_target_environment_id"), "right_target_environment_id": record.get("right_target_environment_id"),
        "predictor": record.get("predictor"), "predictor_information_regime": record.get("predictor_regime"),
        "regime": record.get("availability"), "metric": record.get("metric"), "cell_budget": "full", "cell_budget_label": "full",
        "universe_mode": "matched_fixed" if record.get("dataset") in ("FrangiehIzar2021_RNA", "NadigOConner2024") else np.nan,
        "label_universe": "canonical shared labels" if record.get("status") == "available" else np.nan, "n": record.get("n_perturbations_primary_min"),
        "observed_D": record.get("observed_D"), "observed_D_ci_low": record.get("observed_D_ci_low"), "observed_D_ci_high": record.get("observed_D_ci_high"),
        "measurement_identifiable": record.get("measurement_identifiable"), "measurement_identifiable_ci_low": record.get("measurement_identifiable_ci_low"), "measurement_identifiable_ci_high": record.get("measurement_identifiable_ci_high"),
        "joint_identifiable": record.get("joint_identifiable"), "joint_identifiable_ci_low": record.get("joint_identifiable_ci_low"), "joint_identifiable_ci_high": record.get("joint_identifiable_ci_high"),
        "stable_fraction_both": record.get("stable_fraction_both"), "checksums": _sha256(source) if source and source.is_file() else "unavailable",
        "provenance": provenance, "status": record.get("status", "unavailable"),
    })
    return row


def _atlas_rows(manifests: Path) -> pd.DataFrame:
    path = manifests / "reliability_transport_atlas.csv"
    if not path.is_file():
        return pd.DataFrame(columns=MASTER_COLUMNS)
    return pd.DataFrame([_record_from_atlas(record, manifests) for record in pd.read_csv(path).to_dict("records")])


def _depth_rows(manifests: Path) -> pd.DataFrame:
    path = manifests / "reliability_transport_measurement_depth_matched_fixed_summary.csv"
    if not path.is_file():
        path = manifests / "reliability_transport_measurement_depth_summary.csv"
    if not path.is_file():
        return pd.DataFrame(columns=MASTER_COLUMNS)
    frame = pd.read_csv(path)
    rows = []
    for record in frame.to_dict("records"):
        row = _empty()
        row.update({
            "dataset": "FrangiehIzar2021_RNA", "dataset_split": "within_dataset", "evidence_tier": 1,
            "source_environment_id": record.get("source_environment_id"), "left_target_environment_id": record.get("left_target_environment_id"), "right_target_environment_id": record.get("right_target_environment_id"),
            "predictor": "source_frozen_ensemble", "predictor_information_regime": "strict_source_only_frozen_prediction",
            "regime": "measurement_depth_matched_fixed", "metric": record.get("metric"), "cell_budget": record.get("cell_budget"), "cell_budget_label": record.get("cell_budget_label", record.get("cell_budget")),
            "universe_mode": record.get("universe_mode", "matched_fixed"), "label_universe": "matched fixed universe at highest fixed budget", "n": record.get("n_perturbations_mean"),
            "observed_D": record.get("cross_disagreement"), "observed_D_ci_low": record.get("cross_disagreement_ci_low"), "observed_D_ci_high": record.get("cross_disagreement_ci_high"),
            "measurement_identifiable": record.get("identifiable_divergence"), "measurement_identifiable_ci_low": record.get("identifiable_divergence_ci_low"), "measurement_identifiable_ci_high": record.get("identifiable_divergence_ci_high"),
            "stable_fraction_both": record.get("stable_pair_fraction"), "provenance": path.relative_to(ROOT).as_posix(), "checksums": _sha256(path), "status": "available",
        })
        rows.append(row)
    return pd.DataFrame(rows)


def _decision_rows(manifests: Path) -> pd.DataFrame:
    path = manifests / "reliability_transport_measurement_depth_matched_fixed_decision.csv"
    if not path.is_file():
        return pd.DataFrame(columns=MASTER_COLUMNS)
    frame = pd.read_csv(path)
    numeric = ["retention", "regret", "normalized_regret", "boundary_inversion"]
    group_cols = [c for c in ["source_environment_id", "target_environment_id", "left_target_environment_id", "right_target_environment_id", "metric", "cell_budget_label", "decision_budget_fraction", "universe_mode"] if c in frame.columns]
    if "target_environment_id" not in frame.columns:
        return pd.DataFrame(columns=MASTER_COLUMNS)
    rows = []
    for key, group in frame.groupby(group_cols, sort=True, observed=True):
        rec = dict(zip(group_cols, key if isinstance(key, tuple) else (key,)))
        row = _empty()
        row.update({
            "dataset": "FrangiehIzar2021_RNA", "dataset_split": "within_dataset", "evidence_tier": 1,
            "source_environment_id": rec.get("source_environment_id"), "target_environment_id": rec.get("target_environment_id"),
            "left_target_environment_id": rec.get("left_target_environment_id"), "right_target_environment_id": rec.get("right_target_environment_id"),
            "source_target_orientation": "directed_source_select_target_evaluate", "predictor": "source_frozen_ensemble",
            "predictor_information_regime": "strict_source_only_frozen_prediction", "regime": "decision_transport_directed", "metric": rec.get("metric"),
            "cell_budget_label": rec.get("cell_budget_label"), "universe_mode": rec.get("universe_mode", "matched_fixed"),
            "decision_budget_fraction": rec.get("decision_budget_fraction"), "provenance": path.relative_to(ROOT).as_posix(), "checksums": _sha256(path), "status": "available",
        })
        for column in numeric:
            if column in group:
                row[column] = float(group[column].mean())
                row[f"{column}_ci_low"] = float(group[column].quantile(0.05))
                row[f"{column}_ci_high"] = float(group[column].quantile(0.95))
        for column in ["measurement_floor_regret", "joint_floor_regret", "excess_regret", "joint_excess_regret"]:
            if column in group:
                row[column] = float(group[column].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def _predictability_rows(manifests: Path) -> pd.DataFrame:
    path = manifests / "reliability_transport_predictability.csv"
    if not path.is_file():
        return pd.DataFrame(columns=MASTER_COLUMNS)
    frame = pd.read_csv(path)
    rows = []
    for record in frame.to_dict("records"):
        row = _empty()
        row.update({
            "dataset": record.get("dataset", "FrangiehIzar2021_RNA"), "dataset_split": record.get("dataset_split"), "evidence_tier": 1 if record.get("status") == "executed" else 3,
            "source_environment_id": record.get("source_environment_id"), "left_target_environment_id": record.get("left_target_environment_id"), "right_target_environment_id": record.get("right_target_environment_id"),
            "predictor": record.get("model"), "predictor_information_regime": "fixed_source_only_v2", "regime": "prospective_source_only", "metric": record.get("metric"),
            "cell_budget": record.get("cell_budget_label"), "cell_budget_label": record.get("cell_budget_label"),
            "predictability_model": record.get("model"), "predictability_dataset_split": record.get("dataset_split"), "predictability_spearman": record.get("spearman"), "predictability_auroc": record.get("auroc_high_identifiable"), "predictability_mae": record.get("mean_absolute_error"), "calibration_brier": record.get("calibration_brier"),
            "provenance": path.relative_to(ROOT).as_posix(), "checksums": _sha256(path), "status": record.get("status", "unavailable"),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def _modern_rows(manifests: Path) -> pd.DataFrame:
    path = manifests / "modern_predictor_feasibility.csv"
    if not path.is_file():
        return pd.DataFrame(columns=MASTER_COLUMNS)
    rows = []
    for record in pd.read_csv(path).to_dict("records"):
        row = _empty()
        row.update({"dataset": "FrangiehIzar2021_RNA", "evidence_tier": 3, "predictor": record.get("predictor_id"), "predictor_information_regime": "modern_predictor_audit", "regime": "modern_predictor_feasibility", "modern_claim_lock_status": record.get("claim_lock_eligible"), "modern_claim_lock_blocker": record.get("claim_lock_blocker"), "provenance": path.relative_to(ROOT).as_posix(), "checksums": _sha256(path), "status": "available"})
        rows.append(row)
    return pd.DataFrame(rows)


def build(root: Path = ROOT) -> tuple[pd.DataFrame, dict]:
    root = root.resolve()
    manifests = root / "artifacts/manifests"
    frames = [_atlas_rows(manifests), _depth_rows(manifests), _decision_rows(manifests), _predictability_rows(manifests), _modern_rows(manifests)]
    master = pd.concat(frames, ignore_index=True, sort=False)
    for column in MASTER_COLUMNS:
        if column not in master:
            master[column] = np.nan
    master = master.loc[:, list(MASTER_COLUMNS)].sort_values(["dataset", "regime", "source_environment_id", "target_environment_id", "metric", "cell_budget_label", "predictor"], kind="stable").reset_index(drop=True)
    claims = {
        "schema_version": 2, "status": "locked_to_master_table",
        "selection_policy": "every claim filter resolves to canonical rows with status available and non-unavailable provenance; blocked rows cannot support a positive claim",
        "claims": [
            {"claim_id": "C1_transport_reordering_is_observed", "claim": "Cross-context ordering disagreement is measurable under the prespecified pairwise-order estimand.", "support_filter": {"regime": "canonical_fullsize_claim_lock", "status": "available", "required_columns": ["observed_D", "observed_D_ci_low", "observed_D_ci_high"]}},
            {"claim_id": "C2_measurement_depth_is_a_boundary", "claim": "The identifiable component is reported after U-statistic floor subtraction on a matched fixed universe, with depth-dependent resolution explicit.", "support_filter": {"regime": "measurement_depth_matched_fixed", "universe_mode": "matched_fixed", "status": "available", "required_columns": ["measurement_identifiable", "measurement_identifiable_ci_low", "measurement_identifiable_ci_high"]}},
            {"claim_id": "C3_directed_decision_consequence_is_budget_specific", "claim": "A source ordering has directed source-select/target-evaluate retention and regret consequences at each budget, with target-context floors.", "support_filter": {"regime": "decision_transport_directed", "source_target_orientation": "directed_source_select_target_evaluate", "status": "available", "required_columns": ["decision_budget_fraction", "retention", "regret"]}},
            {"claim_id": "C4_source_only_predictability_is_audited", "claim": "Source-only predictability is reported using held-out outcomes; cross-dataset rows remain explicitly blocked when canonical per-label outcomes are unavailable.", "support_filter": {"regime": "prospective_source_only", "status": "executed", "required_columns": ["predictability_model", "predictability_spearman"]}},
        ],
    }
    return master, claims


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    manifests = args.root.resolve() / "artifacts/manifests"
    master, claims = build(args.root)
    master_path = manifests / "reliability_transport_master.csv"
    claims_path = manifests / "story_claims.json"
    master.to_csv(master_path, index=False)
    claims_path.write_text(json.dumps(claims, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"master_rows": len(master), "claim_count": len(claims["claims"]), "master": master_path.as_posix(), "claims": claims_path.as_posix()}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
