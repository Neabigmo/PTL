"""Build a metadata-frozen reliability transportability atlas."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


ATLAS_COLUMNS = (
    "dataset",
    "evidence_tier",
    "source_environment_id",
    "left_target_environment_id",
    "right_target_environment_id",
    "metric",
    "observed_D",
    "observed_D_ci_low",
    "observed_D_ci_high",
    "measurement_identifiable",
    "measurement_identifiable_ci_low",
    "measurement_identifiable_ci_high",
    "joint_identifiable",
    "joint_identifiable_ci_low",
    "joint_identifiable_ci_high",
    "stable_fraction_both",
    "status",
    "availability",
    "provenance",
)


def assign_evidence_tier(row: pd.Series) -> int:
    """Freeze tiers from registry semantics, not from the observed effect."""

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
        1: "tier 1: raw-context verified common-core anchor",
        2: "tier 2: raw-context verified context expansion",
        3: "tier 3: metadata-registered but not eligible for the primary claim lock",
    })
    return output.sort_values(["evidence_tier", "environment_id"], kind="stable").reset_index(drop=True)


def _frangieh_rows(summary: pd.DataFrame, registry: pd.DataFrame) -> pd.DataFrame:
    frame = summary.copy()
    tier_map = registry.set_index("environment_key")["evidence_tier"].to_dict()
    frame["dataset"] = "FrangiehIzar2021_RNA"
    frame["evidence_tier"] = frame["source_environment_id"].map(tier_map).fillna(2).astype(int)
    frame = frame.rename(columns={
        "ordering_cross_disagreement": "observed_D",
        "ordering_cross_disagreement_ci_low": "observed_D_ci_low",
        "ordering_cross_disagreement_ci_high": "observed_D_ci_high",
        "ordering_delta_meas_id": "measurement_identifiable",
        "ordering_delta_meas_id_ci_low": "measurement_identifiable_ci_low",
        "ordering_delta_meas_id_ci_high": "measurement_identifiable_ci_high",
        "ordering_delta_joint_id": "joint_identifiable",
        "ordering_delta_joint_id_ci_low": "joint_identifiable_ci_low",
        "ordering_delta_joint_id_ci_high": "joint_identifiable_ci_high",
    })
    ordering_path = summary.attrs.get("ordering_path")
    if ordering_path:
        ordering = pd.read_csv(ordering_path)
        ordering = ordering[["source_environment_id", "left_target_environment_id", "right_target_environment_id", "metric", "stable_fraction_both"]]
        frame = frame.merge(ordering, on=["source_environment_id", "left_target_environment_id", "right_target_environment_id", "metric"], how="left")
    frame["status"] = "available"
    frame["availability"] = "canonical_fullsize_claim_lock"
    frame["provenance"] = "artifacts/manifests/formal_v2_claim_lock_measurement_fullsize_summary.csv"
    return frame


def _nadig_rows(summary: pd.DataFrame, registry: pd.DataFrame) -> pd.DataFrame:
    frame = summary.copy()
    frame["dataset"] = "Nadig2021"
    frame["evidence_tier"] = 2
    frame = frame.rename(columns={
        "source_context_id": "source_environment_id",
        "left_target_context_id": "left_target_environment_id",
        "right_target_context_id": "right_target_environment_id",
        "ordering_cross_disagreement": "observed_D",
        "ordering_cross_disagreement_ci_low": "observed_D_ci_low",
        "ordering_cross_disagreement_ci_high": "observed_D_ci_high",
        "ordering_delta_meas_id": "measurement_identifiable",
        "ordering_delta_meas_id_ci_low": "measurement_identifiable_ci_low",
        "ordering_delta_meas_id_ci_high": "measurement_identifiable_ci_high",
        "ordering_delta_joint_id": "joint_identifiable",
        "ordering_delta_joint_id_ci_low": "joint_identifiable_ci_low",
        "ordering_delta_joint_id_ci_high": "joint_identifiable_ci_high",
    })
    frame["stable_fraction_both"] = float("nan")
    frame["status"] = "available"
    frame["availability"] = "canonical_nadig_replication"
    frame["provenance"] = "artifacts/manifests/formal_v2_claim_lock_replication_nadig_fullsize.csv"
    return frame


def build_atlas(
    *,
    registry: pd.DataFrame,
    frangieh_summary: pd.DataFrame,
    nadig_summary: pd.DataFrame | None = None,
    ordering_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return frozen registry and a rectangular atlas with explicit status."""

    frozen = freeze_registry(registry)
    frangieh_summary = frangieh_summary.copy()
    if ordering_path is not None:
        frangieh_summary.attrs["ordering_path"] = str(ordering_path)
    rows = [_frangieh_rows(frangieh_summary, frozen)]
    if nadig_summary is not None and not nadig_summary.empty:
        rows.append(_nadig_rows(nadig_summary, frozen))
    atlas = pd.concat(rows, ignore_index=True, sort=False)
    represented = set(atlas["source_environment_id"].dropna().astype(str))
    metadata_only: list[dict[str, Any]] = []
    for record in frozen.to_dict("records"):
        environment = str(record["environment_key"])
        if environment in represented:
            continue
        metadata_only.append({
            "dataset": record.get("dataset_id"),
            "evidence_tier": int(record["evidence_tier"]),
            "source_environment_id": environment,
            "status": "unavailable",
            "availability": "registered_metadata_only",
            "provenance": "artifacts/manifests/environment_registry.csv",
        })
    if metadata_only:
        atlas = pd.concat([atlas, pd.DataFrame(metadata_only)], ignore_index=True, sort=False)
    for column in ATLAS_COLUMNS:
        if column not in atlas.columns:
            atlas[column] = pd.NA
    atlas = atlas.loc[:, list(ATLAS_COLUMNS)].sort_values(
        ["evidence_tier", "dataset", "source_environment_id", "left_target_environment_id", "right_target_environment_id", "metric"],
        kind="stable",
    ).reset_index(drop=True)
    return frozen, atlas


def atlas_report(atlas: pd.DataFrame, registry: pd.DataFrame) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "executed",
        "evidence_tiers": {
            "1": "raw-context verified common-core anchor",
            "2": "raw-context verified context expansion or independent replication",
            "3": "registered metadata surface not eligible for the primary claim lock",
        },
        "rows": int(len(atlas)),
        "registry_rows": int(len(registry)),
        "tier_counts": {str(key): int(value) for key, value in atlas["evidence_tier"].value_counts().sort_index().items()},
        "availability_counts": {str(key): int(value) for key, value in atlas["availability"].value_counts(dropna=False).items()},
        "unsupported_values_policy": "missing fields remain explicit unavailable values; no unsupported effect, CI, or predictor statistic is imputed",
        "outcome_blindness": "atlas construction reads canonical frozen summaries and metadata only; target outcomes are not used to select rows",
    }
