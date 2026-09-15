"""Run the canonical raw-cell measurement bridge for a frozen predictor family.

The bridge reuses the validated formal-v2 full-size non-parametric resampling
engine and only changes the frozen prediction artifact.  The engine writes an
unordered three-context table; this adapter materializes the six directed
source-to-target transfers only when the predictor was trained in the source
context.  Every family therefore produces the canonical 18-row surface.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_formal_v2_claim_lock_measurement import (  # noqa: E402
    MIN_CELLS_PRIMARY,
    METRICS,
    run as run_measurement,
)
from scripts.run_formal_v2_claim_lock_source_frozen import ENVIRONMENTS  # noqa: E402
from scripts.run_frangieh_source_only_rbf_krr import _endpoint_rows  # noqa: E402


DIRECTED_PAIRS = tuple(
    (source, target)
    for source in ENVIRONMENTS
    for target in ENVIRONMENTS
    if source != target
)


def _canonical_surface(
    root: Path,
    *,
    family: str,
    measurement_report: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert the engine's unordered pair table to the canonical surface."""

    summary = _endpoint_rows(
        pd.read_csv(root / measurement_report["summary_rows"]),
        source_column="source_environment_id",
    )
    # Floor rows retain one record per member pair, whereas summary and
    # ordering rows are already aggregated.  Keep the floor table in its
    # unordered source/pair form and use the aggregated values carried by the
    # summary row for the directed surface; expanding the raw floor rows would
    # incorrectly duplicate each canonical direction three times.
    floor_raw = pd.read_csv(root / measurement_report["floor_rows"])
    floor_summary = (
        floor_raw.groupby(
            [
                "source_environment_id",
                "left_target_environment_id",
                "right_target_environment_id",
                "metric",
            ],
            sort=False,
        )[[
            "n_perturbations",
            "ordering_within_left_u",
            "ordering_within_right_u",
        ]]
        .mean()
    )
    ordering = _endpoint_rows(
        pd.read_csv(root / measurement_report["ordering_rows"]),
        source_column="source_environment_id",
    ).set_index(["source_environment_id", "target_environment_id", "metric"])
    rows: list[dict[str, Any]] = []
    floor_rows: list[dict[str, Any]] = []
    for record in summary.to_dict(orient="records"):
        source = str(record["source_environment_id"])
        target = str(record["target_environment_id"])
        metric = str(record["metric"])
        key = (source, target, metric)
        order = ordering.loc[key]
        floor_key = (
            source,
            str(record["left_target_environment_id"]),
            str(record["right_target_environment_id"]),
            metric,
        )
        floor = floor_summary.loc[floor_key]
        left = str(record["left_target_environment_id"])
        if source == left:
            source_floor = float(floor["ordering_within_left_u"])
            target_floor = float(floor["ordering_within_right_u"])
        else:
            source_floor = float(floor["ordering_within_right_u"])
            target_floor = float(floor["ordering_within_left_u"])
        rows.append({
            "family": family,
            "source_context": source,
            "target_context": target,
            "transfer_id": f"{source}->{target}",
            "metric": metric,
            "n_labels": int(float(record["n_perturbations_primary_min"])),
            "D_cross": float(record["ordering_cross_disagreement"]),
            "D_within_source": source_floor,
            "D_within_target": target_floor,
            "D_adj": float(record["ordering_delta_meas_id"]),
            "ci_lower_90": float(record["ordering_delta_meas_id_ci_low"]),
            "ci_upper_90": float(record["ordering_delta_meas_id_ci_high"]),
            "stable_inversion": float(order["stable_order_inversion_fraction"]),
            "same_context_floor": float(record["ordering_measurement_floor"]),
            "same_context_floor_ci_lower_90": float(record["ordering_measurement_floor_ci_low"]),
            "same_context_floor_ci_upper_90": float(record["ordering_measurement_floor_ci_high"]),
            "minimum_strict_support": int(float(order["minimum_strict_support"])),
            "measurement_resampling": "full_size_nonparametric",
            "target_outcomes_used_for_fit_or_tuning": 0,
        })
        floor_rows.append({
            "family": family,
            "source_context": source,
            "target_context": target,
            "transfer_id": f"{source}->{target}",
            "metric": metric,
            "n_labels": int(float(floor["n_perturbations"])),
            "source_same_context_floor": source_floor,
            "target_same_context_floor": target_floor,
            "same_context_floor": float(record["ordering_measurement_floor"]),
            "ci_lower_90": float(record["ordering_measurement_floor_ci_low"]),
            "ci_upper_90": float(record["ordering_measurement_floor_ci_high"]),
            "measurement_resampling": "full_size_nonparametric",
        })
    surface = pd.DataFrame(rows).sort_values(
        ["source_context", "target_context", "metric"], kind="stable"
    ).reset_index(drop=True)
    same_context = pd.DataFrame(floor_rows).sort_values(
        ["source_context", "target_context", "metric"], kind="stable"
    ).reset_index(drop=True)
    expected = len(DIRECTED_PAIRS) * len(METRICS)
    keys = ["source_context", "target_context", "metric"]
    if len(surface) != expected or surface.duplicated(keys).any():
        raise RuntimeError(f"{family} canonical surface must contain {expected} unique rows")
    if set(zip(surface["source_context"], surface["target_context"])) != set(DIRECTED_PAIRS):
        raise RuntimeError(f"{family} canonical surface has a non-canonical source/target pair")
    if (surface["source_context"] == surface["target_context"]).any():
        raise RuntimeError(f"{family} canonical surface contains a self-transfer")
    if surface["minimum_strict_support"].min() < 8:
        raise RuntimeError(f"{family} stable ordering violates minimum strict support 8")
    return surface, same_context


def run_family(
    root: Path,
    *,
    family: str,
    prediction_path: Path,
    output_stem: str | None = None,
    draws: int = 2000,
) -> dict[str, Any]:
    """Run one frozen family and write its 18-row canonical artifacts."""

    root = root.resolve()
    prediction_path = prediction_path.resolve()
    if not prediction_path.is_file():
        raise FileNotFoundError(prediction_path)
    family_slug = family.strip().lower().replace("-", "_").replace(" ", "_")
    if not family_slug or "/" in family_slug or "\\" in family_slug:
        raise ValueError("family must be a simple name")
    output_stem = output_stem or f"{family_slug}_measurement_fullsize"
    if Path(output_stem).name != output_stem:
        raise ValueError("output_stem must be a simple file stem")
    import scripts.run_formal_v2_claim_lock_measurement as measurement

    previous = measurement.PREDICTION_PATH
    try:
        measurement.PREDICTION_PATH = prediction_path
        measurement_report = run_measurement(
            root,
            seed_chunk=1,
            draws=int(draws),
            output_stem=output_stem,
            resampling_mode="full_size_nonparametric",
        )
    finally:
        measurement.PREDICTION_PATH = previous
    surface, same_context = _canonical_surface(
        root, family=family_slug, measurement_report=measurement_report
    )
    outdir = root / "artifacts/manifests/predictor_families_v2"
    outdir.mkdir(parents=True, exist_ok=True)
    d_adj_path = outdir / f"{family_slug}_d_adj_fullsize.csv"
    floor_path = outdir / f"{family_slug}_same_context_floor.csv"
    report_path = outdir / f"{family_slug}_d_adj_fullsize.json"
    surface.to_csv(d_adj_path, index=False)
    same_context.to_csv(floor_path, index=False)
    report = {
        "schema_version": 1,
        "status": "executed",
        "family": family_slug,
        "prediction_artifact": prediction_path.relative_to(root).as_posix()
        if prediction_path.is_relative_to(root)
        else str(prediction_path),
        "canonical_contract": {
            "rows": int(len(surface)),
            "directed_transfer_count": len(DIRECTED_PAIRS),
            "metric_count": len(METRICS),
            "source_contexts": list(ENVIRONMENTS),
            "minimum_cells": MIN_CELLS_PRIMARY,
            "measurement_seeds": 30,
            "measurement_replicates_per_seed": 2,
            "min_strict_support": 8,
            "target_outcomes_used_for_fit_or_tuning": False,
        },
        "outputs": {
            "d_adj_fullsize": d_adj_path.relative_to(root).as_posix(),
            "same_context_floor": floor_path.relative_to(root).as_posix(),
        },
        "measurement_report": measurement_report,
    }
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + chr(10),
        encoding="utf-8",
    )
    return {**report, "report": report_path.relative_to(root).as_posix()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--family", required=True)
    parser.add_argument("--prediction-path", type=Path, required=True)
    parser.add_argument("--output-stem")
    parser.add_argument("--draws", type=int, default=2000)
    args = parser.parse_args()
    result = run_family(
        args.root,
        family=args.family,
        prediction_path=args.prediction_path,
        output_stem=args.output_stem,
        draws=args.draws,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
