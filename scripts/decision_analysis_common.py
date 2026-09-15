"""Shared contracts for reviewer-facing decision analyses."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


CANONICAL_SUMMARY = "artifacts/manifests/reliability_transport_measurement_depth_matched_fixed_summary.csv"
FULL_DEPTH = "full"


def add_context_ids(frame: pd.DataFrame) -> pd.DataFrame:
    """Add dependence-aware IDs without changing the directed-row surface."""

    out = frame.copy()
    out["source_context_id"] = out["source_environment_id"].astype(str)
    out["unordered_context_pair_id"] = [
        "<->".join(sorted((str(source), str(target))))
        for source, target in zip(out["source_environment_id"], out["target_environment_id"])
    ]
    return out


def directed_endpoint_surface(frame: pd.DataFrame) -> pd.DataFrame:
    """Return only endpoint-source rows with an explicit directed target."""

    required = {"source_environment_id", "left_target_environment_id", "right_target_environment_id"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise RuntimeError(f"directed surface is missing columns: {missing}")
    out = frame.loc[
        frame["source_environment_id"].eq(frame["left_target_environment_id"])
        | frame["source_environment_id"].eq(frame["right_target_environment_id"])
    ].copy()
    out["target_environment_id"] = np.where(
        out["source_environment_id"].eq(out["left_target_environment_id"]),
        out["right_target_environment_id"],
        out["left_target_environment_id"],
    )
    out["transfer_id"] = (
        out["source_environment_id"].astype(str)
        + "->"
        + out["target_environment_id"].astype(str)
    )
    return add_context_ids(out)


def load_canonical_d_adj(root: Path, metrics: tuple[str, ...] | list[str]) -> pd.DataFrame:
    """Load directed full-depth D_adj only from the corrected canonical summary."""

    path = root / CANONICAL_SUMMARY
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, dtype={"cell_budget_label": "string"})
    required = {
        "source_environment_id", "left_target_environment_id", "right_target_environment_id",
        "metric", "cell_budget_label", "identifiable_divergence",
        "identifiable_divergence_ci_low", "identifiable_divergence_ci_high",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise RuntimeError(f"corrected canonical summary is missing columns: {missing}")
    frame = frame.loc[frame["cell_budget_label"].eq(FULL_DEPTH) & frame["metric"].isin(metrics)].copy()
    frame = directed_endpoint_surface(frame)
    frame = frame.rename(columns={
        "identifiable_divergence": "d_meas_id",
        "identifiable_divergence_ci_low": "d_meas_id_ci_low",
        "identifiable_divergence_ci_high": "d_meas_id_ci_high",
    })
    columns = [
        "source_environment_id", "target_environment_id", "transfer_id", "metric",
        "d_meas_id", "d_meas_id_ci_low", "d_meas_id_ci_high",
    ]
    frame = add_context_ids(frame[columns])
    keys = ["source_environment_id", "target_environment_id", "metric"]
    if len(frame) != 18 or frame.duplicated(keys).any():
        raise RuntimeError(f"corrected canonical full-depth D_adj surface must have 18 unique directed rows; found {len(frame)}")
    if frame["source_context_id"].nunique() != 3 or frame["unordered_context_pair_id"].nunique() != 3:
        raise RuntimeError("corrected canonical D_adj surface must contain three source contexts and three unordered pairs")
    pair_direction_counts = frame.groupby("unordered_context_pair_id", sort=True)["transfer_id"].nunique()
    if not pair_direction_counts.eq(2).all():
        raise RuntimeError("corrected canonical D_adj surface must contain both directions for every unordered pair")
    if frame["metric"].nunique() != len(set(metrics)):
        raise RuntimeError("corrected canonical D_adj surface does not contain the requested metric set")
    if not np.isfinite(frame[["d_meas_id", "d_meas_id_ci_low", "d_meas_id_ci_high"]].to_numpy(dtype=float)).all():
        raise RuntimeError("corrected canonical full-depth D_adj contains non-finite values")
    return frame.sort_values(keys, kind="stable").reset_index(drop=True)
