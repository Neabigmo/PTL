"""Build a pair-level evidence atlas without promoting metadata to evidence."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ATLAS_COLUMNS = (
    "dataset", "evidence_tier", "source_environment_id", "target_environment_id",
    "left_target_environment_id", "right_target_environment_id", "predictor", "predictor_regime",
    "metric", "estimand", "observed_value", "observed_value_ci_low", "observed_value_ci_high",
    "observed_D", "observed_D_ci_low", "observed_D_ci_high", "measurement_identifiable",
    "measurement_identifiable_ci_low", "measurement_identifiable_ci_high", "joint_identifiable",
    "joint_identifiable_ci_low", "joint_identifiable_ci_high", "stable_fraction_both", "status",
    "availability", "provenance",
)


def assign_evidence_tier(row: pd.Series) -> int:
    """Assign the metadata tier used by the registry (not an outcome tier)."""
    status = str(row.get("status", ""))
    semantic = str(row.get("semantic_status", ""))
    role = str(row.get("role", ""))
    if status.startswith("ready_") and ("verified" in semantic or "verified" in status):
        return 1 if role == "common_core" else 2
    if status.startswith("ready_"):
        return 2
    return 3


def freeze_registry(registry: pd.DataFrame) -> pd.DataFrame:
    required = {"environment_id", "environment_key", "dataset_id", "role", "semantic_status", "status"}
    missing = required.difference(registry.columns)
    if missing:
        raise ValueError(f"environment registry missing columns: {sorted(missing)}")
    output = registry.copy()
    output["evidence_tier"] = output.apply(assign_evidence_tier, axis=1).astype(int)
    output["evidence_tier_definition"] = output["evidence_tier"].map({
        1: "tier 1: verified common-core metadata anchor; pair-level evidence still required",
        2: "tier 2: verified context expansion; descriptive or independent replication",
        3: "tier 3: registered metadata only; not eligible for the primary claim lock",
    })
    return output.sort_values(["evidence_tier", "environment_id"], kind="stable").reset_index(drop=True)


def _rename_claim_lock(frame: pd.DataFrame, dataset: str, provenance: str, availability: str) -> pd.DataFrame:
    frame = frame.copy()
    frame["dataset"] = dataset
    frame = frame.rename(columns={
        "ordering_cross_disagreement": "observed_D", "ordering_cross_disagreement_ci_low": "observed_D_ci_low", "ordering_cross_disagreement_ci_high": "observed_D_ci_high",
        "ordering_delta_meas_id": "measurement_identifiable", "ordering_delta_meas_id_ci_low": "measurement_identifiable_ci_low", "ordering_delta_meas_id_ci_high": "measurement_identifiable_ci_high",
        "ordering_delta_joint_id": "joint_identifiable", "ordering_delta_joint_id_ci_low": "joint_identifiable_ci_low", "ordering_delta_joint_id_ci_high": "joint_identifiable_ci_high",
    })
    frame["predictor"] = "source_frozen_ensemble"
    frame["predictor_regime"] = "strict_source_only_frozen_prediction"
    frame["estimand"] = "pairwise_order_disagreement_with_U_corrected_identifiable_component"
    frame["status"] = "available"
    frame["availability"] = availability
    frame["provenance"] = provenance
    return frame


def _frangieh_rows(summary: pd.DataFrame, ordering_path: Path | None = None) -> pd.DataFrame:
    frame = _rename_claim_lock(summary, "FrangiehIzar2021_RNA", "artifacts/manifests/formal_v2_claim_lock_measurement_fullsize_summary.csv", "canonical_fullsize_claim_lock")
    if ordering_path is not None and ordering_path.is_file():
        ordering = pd.read_csv(ordering_path)
        cols = ["source_environment_id", "left_target_environment_id", "right_target_environment_id", "metric", "stable_fraction_both"]
        if set(cols).issubset(ordering.columns):
            frame = frame.merge(ordering[cols], on=cols[:4], how="left", suffixes=("", "_ordering"))
            if "stable_fraction_both_ordering" in frame:
                frame["stable_fraction_both"] = frame["stable_fraction_both"].combine_first(frame.pop("stable_fraction_both_ordering"))
    frame["evidence_tier"] = 1
    return frame


def _nadig_rows(summary: pd.DataFrame) -> pd.DataFrame:
    frame = summary.copy().rename(columns={
        "source_context_id": "source_environment_id", "left_target_context_id": "left_target_environment_id", "right_target_context_id": "right_target_environment_id",
    })
    frame = _rename_claim_lock(frame, "NadigOConner2024", "artifacts/manifests/formal_v2_claim_lock_replication_nadig_fullsize.csv", "canonical_nadig_replication")
    frame["evidence_tier"] = 1
    frame["stable_fraction_both"] = np.nan
    return frame


def _tier2_transfer_rows(transfer: pd.DataFrame | None) -> pd.DataFrame:
    if transfer is None or transfer.empty:
        return pd.DataFrame()
    frame = transfer.copy()
    rows: list[dict[str, Any]] = []
    source_col, target_col = "source_environment_id", "target_environment_id"
    for (source, target, baseline), group in frame.groupby([source_col, target_col, "baseline"], sort=True, observed=True):
        value = pd.to_numeric(group["excess_aurc_gain"], errors="coerce")
        if not value.notna().any():
            value = pd.to_numeric(group["transfer_gain"], errors="coerce")
            metric = "transfer_gain"
        else:
            metric = "excess_aurc_gain"
        value = value.dropna()
        if value.empty:
            continue
        rows.append({
            "dataset": "multi_dataset_transfer_surface", "evidence_tier": 2,
            "source_environment_id": source, "target_environment_id": target,
            "predictor": str(baseline), "predictor_regime": "descriptive_observed_transfer_matrix",
            "metric": metric, "estimand": "descriptive_transfer_gain_not_claim_lock",
            "observed_value": float(value.mean()), "observed_value_ci_low": float(value.quantile(0.05)), "observed_value_ci_high": float(value.quantile(0.95)),
            "status": "available", "availability": "tier2_broad_observed_transfer_surface",
            "provenance": "artifacts/manifests/formal_v2_environment_transfer_matrix_all_splits.csv",
        })
    return pd.DataFrame(rows)


def build_atlas(*, registry: pd.DataFrame, frangieh_summary: pd.DataFrame, nadig_summary: pd.DataFrame | None = None,
                ordering_path: Path | None = None, transfer_matrix: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return frozen metadata and a pair-level atlas with explicit tiers."""
    frozen = freeze_registry(registry)
    rows = [_frangieh_rows(frangieh_summary, ordering_path)]
    if nadig_summary is not None and not nadig_summary.empty:
        rows.append(_nadig_rows(nadig_summary))
    tier2 = _tier2_transfer_rows(transfer_matrix)
    if not tier2.empty:
        rows.append(tier2)
    atlas = pd.concat(rows, ignore_index=True, sort=False)
    represented = set(atlas["source_environment_id"].dropna().astype(str))
    metadata_only: list[dict[str, Any]] = []
    for record in frozen.to_dict("records"):
        environment = str(record["environment_key"])
        if environment in represented:
            continue
        metadata_only.append({
            "dataset": record.get("dataset_id"), "evidence_tier": 3, "source_environment_id": environment,
            "predictor": "unavailable", "predictor_regime": "not_run", "status": "unavailable",
            "availability": "tier3_registered_metadata_only", "provenance": "artifacts/manifests/environment_registry.csv",
        })
    if metadata_only:
        atlas = pd.concat([atlas, pd.DataFrame(metadata_only)], ignore_index=True, sort=False)
    for column in ATLAS_COLUMNS:
        if column not in atlas.columns:
            atlas[column] = pd.NA
    atlas = atlas.loc[:, list(ATLAS_COLUMNS)].sort_values(
        ["evidence_tier", "dataset", "source_environment_id", "target_environment_id", "left_target_environment_id", "right_target_environment_id", "predictor", "metric"], kind="stable"
    ).reset_index(drop=True)
    return frozen, atlas


def atlas_report(atlas: pd.DataFrame, registry: pd.DataFrame) -> dict[str, Any]:
    return {
        "schema_version": 2, "status": "executed", "rows": int(len(atlas)), "registry_rows": int(len(registry)),
        "evidence_tiers": {"1": "Frangieh/Nadig pair-level Claim Lock with corrected inference", "2": "broader observed transfer rows; descriptive only", "3": "registered metadata without pair-level evidence"},
        "tier_counts": {str(k): int(v) for k, v in atlas["evidence_tier"].value_counts().sort_index().items()},
        "availability_counts": {str(k): int(v) for k, v in atlas["availability"].value_counts(dropna=False).items()},
        "predictor_axis": sorted(atlas["predictor"].dropna().astype(str).unique().tolist()),
        "unsupported_values_policy": "missing fields remain explicit unavailable values; no unsupported effect, CI, or predictor statistic is imputed",
        "outcome_blindness": "atlas construction reads frozen summaries, transfer matrices and metadata; target outcomes do not select rows",
    }
