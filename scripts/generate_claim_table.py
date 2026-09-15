"""Materialize the canonical claim table from ordering-estimand summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


SUMMARY_FILES = (
    ("Frangieh", "formal_v2_claim_lock_measurement_fullsize_summary.csv"),
    ("Nadig", "formal_v2_claim_lock_replication_nadig_fullsize.csv"),
    ("Nadig_matched_1255", "formal_v2_claim_lock_replication_nadig_matched1255.csv"),
    ("Nadig_matched_345", "formal_v2_claim_lock_replication_nadig_matched345.csv"),
)


def _first(row: pd.Series, *names: str, default: float = float("nan")) -> Any:
    for name in names:
        if name in row.index and pd.notna(row[name]):
            return row[name]
    return default


def _status(low: float, high: float) -> str:
    if pd.isna(low) or pd.isna(high):
        return "not_available"
    if low > 0:
        return "positive_measurement_identifiable_excess"
    if high < 0:
        return "negative_measurement_identifiable_excess"
    return "inconclusive_ci_crosses_zero"


def build(root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sources: list[str] = []
    ordering_lookup: dict[tuple[str, str, str, str], pd.Series] = {}
    ordering_path = root / "artifacts" / "manifests" / "formal_v2_claim_lock_measurement_fullsize_ordering.csv"
    if ordering_path.is_file():
        ordering = pd.read_csv(ordering_path)
        for _, ordering_row in ordering.iterrows():
            ordering_lookup[tuple(str(ordering_row.get(column, "")) for column in ("source_environment_id", "left_target_environment_id", "right_target_environment_id", "metric"))] = ordering_row
    for dataset, filename in SUMMARY_FILES:
        path = root / "artifacts" / "manifests" / filename
        if not path.is_file():
            continue
        frame = pd.read_csv(path)
        sources.append(path.as_posix())
        for _, row in frame.iterrows():
            lookup_key = tuple(str(_first(row, *names, default="")) for names in (("source_environment_id", "source_context_id"), ("left_target_environment_id", "left_target_context_id"), ("right_target_environment_id", "right_target_context_id"), ("metric",)))
            ordering_row = ordering_lookup.get(lookup_key)
            low = float(_first(row, "ordering_delta_joint_id_ci_low"))
            high = float(_first(row, "ordering_delta_joint_id_ci_high"))
            stable_fraction = (
                ordering_row["stable_fraction_both"]
                if ordering_row is not None and pd.notna(ordering_row.get("stable_fraction_both"))
                else _first(row, "stable_fraction_both")
            )
            rows.append({
                "dataset": dataset,
                "predictor": "source_frozen_strong_linear",
                "source_context": _first(row, "source_environment_id", "source_context_id", default=""),
                "target_contexts": f"{_first(row, 'left_target_environment_id', 'left_target_context_id', default='')}|{_first(row, 'right_target_environment_id', 'right_target_context_id', default='')}",
                "metric": _first(row, "metric", default=""),
                "ordering_estimand": _first(row, "ordering_estimand", default="pairwise_order_disagreement"),
                "observed_reordering": float(_first(row, "ordering_cross_disagreement", "cross_pairwise_disagreement")),
                "observed_reordering_ci_low": float(_first(row, "ordering_cross_disagreement_ci_low")),
                "observed_reordering_ci_high": float(_first(row, "ordering_cross_disagreement_ci_high")),
                "measurement_identifiable_excess": float(_first(row, "ordering_delta_meas_id", "measurement_corrected_ordering_divergence")),
                "measurement_identifiable_ci_low": float(_first(row, "ordering_delta_meas_id_ci_low")),
                "measurement_identifiable_ci_high": float(_first(row, "ordering_delta_meas_id_ci_high")),
                "joint_identifiable_excess": float(_first(row, "ordering_delta_joint_id")),
                "joint_identifiable_ci_low": low,
                "joint_identifiable_ci_high": high,
                "stable_fraction_both": None if pd.isna(stable_fraction) else float(stable_fraction),
                "minimum_strict_support": int(ordering_row["minimum_strict_support"]) if ordering_row is not None and pd.notna(ordering_row.get("minimum_strict_support")) else int(_first(row, "minimum_strict_support", default=8)),
                "rank_displacement_secondary": float(_first(row, "rank_displacement_cross")),
                "rank_displacement_secondary_ci_low": float(_first(row, "rank_displacement_cross_ci_low")),
                "rank_displacement_secondary_ci_high": float(_first(row, "rank_displacement_cross_ci_high")),
                "claim_lock_status": _status(low, high),
                "bootstrap_uncertainty_scope": "raw-cell/pseudoreplicate sampling uncertainty; not all biological variation",
                "source_artifact": path.relative_to(root).as_posix(),
            })
    if not rows:
        raise FileNotFoundError("no canonical ordering summary files are available")
    frame = pd.DataFrame(rows)
    report = {
        "schema_version": 1,
        "purpose": "canonical estimand-separated Claim Lock table",
        "primary_estimand": "pairwise_order_disagreement",
        "secondary_estimand": "normalized_rank_displacement",
        "rows": int(len(frame)),
        "source_artifacts": sources,
        "bootstrap_uncertainty_scope": "raw-cell/pseudoreplicate sampling uncertainty; not all biological variation",
    }
    return frame, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    root = args.root.resolve()
    output_dir = root / "artifacts" / "manifests"
    output_dir.mkdir(parents=True, exist_ok=True)
    frame, report = build(root)
    csv_path = output_dir / "claim_table.csv"
    json_path = output_dir / "claim_table.json"
    frame.to_csv(csv_path, index=False)
    json_frame = frame.astype(object).where(pd.notna(frame), None)
    report["rows_data"] = json_frame.to_dict(orient="records")
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"csv": csv_path.as_posix(), "json": json_path.as_posix(), "rows": len(frame)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
