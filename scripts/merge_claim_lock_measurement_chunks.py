"""Merge source-disjoint Frangieh seed chunks and recompute canonical summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_formal_v2_claim_lock_measurement import (  # noqa: E402
    ENVIRONMENTS,
    METRICS,
    MODEL_PAIRS,
    PAIR_ORDER,
    SPLIT_SEEDS,
    SPLIT_SEED,
    _bootstrap_floor,
    _synchronized_macro_rows,
)
from src.evaluation.ordering_estimands import (  # noqa: E402
    crossfit_seed_stable_ordering_summary,
    summarize_fixed_predictor_ordering,
)


INPUT_FIELDS = (
    "cross_left_a", "cross_right_a", "cross_left_b", "cross_right_b",
    "meas_left_a", "meas_left_b", "meas_right_a", "meas_right_b",
    "joint_left_a", "joint_left_b", "joint_right_a", "joint_right_b",
)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_chunk(root: Path, report: dict[str, Any]) -> dict[str, list[dict[str, np.ndarray]]]:
    path = root / report["bootstrap_input_path"]
    keys = report.get("bootstrap_input_keys", [])
    result: dict[str, list[dict[str, np.ndarray]]] = {}
    with np.load(path, allow_pickle=False) as archive:
        for encoded in keys:
            arrays = {field: np.asarray(archive[f"{encoded}__{field}"]) for field in INPUT_FIELDS}
            n_rows = arrays["cross_left_a"].shape[0]
            result[encoded] = [{field: arrays[field][index] for field in INPUT_FIELDS} for index in range(n_rows)]
    return result


def _row_ordering(encoded: str, values: list[dict[str, np.ndarray]]) -> dict[str, Any]:
    source, left, right, metric = encoded.split("__", 3)
    left_risk = np.concatenate([np.stack([v["cross_left_a"] for v in values]), np.stack([v["cross_left_b"] for v in values])], axis=0)
    right_risk = np.concatenate([np.stack([v["cross_right_a"] for v in values]), np.stack([v["cross_right_b"] for v in values])], axis=0)
    result = summarize_fixed_predictor_ordering(left_risk, right_risk)
    seed_left = np.stack([np.stack([v["cross_left_a"] for v in values]), np.stack([v["cross_left_b"] for v in values])], axis=1)
    seed_right = np.stack([np.stack([v["cross_right_a"] for v in values]), np.stack([v["cross_right_b"] for v in values])], axis=1)
    result.update(crossfit_seed_stable_ordering_summary(seed_left, seed_right))
    return {
        "estimand": "pairwise_order_disagreement",
        "source_environment_id": source,
        "left_target_environment_id": left,
        "right_target_environment_id": right,
        "metric": metric,
        "replicate_policy": "30 independent seeds × two independent full-size with-replacement raw-cell pseudoreplicates; fixed source-frozen mean prediction",
        **result,
    }


def merge(root: Path, part_stems: list[str], output_stem: str) -> dict[str, Any]:
    root = root.resolve()
    manifest_dir = root / "artifacts/manifests"
    reports = [_read(manifest_dir / f"{stem}.json") for stem in part_stems]
    seen: list[int] = []
    merged: dict[str, list[dict[str, np.ndarray]]] = {}
    frames: list[pd.DataFrame] = []
    for report in reports:
        if report.get("resampling_mode") != "full_size_nonparametric" or not report.get("streamed_bootstrap_inputs"):
            raise ValueError("every worker must be a streamed full-size measurement chunk")
        seeds = [int(seed) for seed in report.get("split_seeds", [])]
        if set(seeds) & set(seen):
            raise ValueError("worker seed ranges overlap")
        seen.extend(seeds)
        floor_path = root / str(report["summary_rows"]).replace("_summary.csv", "_floors.csv")
        if floor_path.is_file():
            frames.append(pd.read_csv(floor_path))
        chunk = _load_chunk(root, report)
        for key, values in chunk.items():
            merged.setdefault(key, []).extend(values)
    if seen != sorted(seen) or seen != list(SPLIT_SEEDS):
        raise ValueError(f"seed schedule is incomplete or unordered: {seen}")
    if not merged or any(len(values) != 30 for values in merged.values()):
        raise ValueError("merged bootstrap inputs do not contain exactly 30 rows per source/pair/metric")

    summary_rows: list[dict[str, Any]] = []
    macro_inputs: list[dict[str, Any]] = []
    ordering_rows = [_row_ordering(key, values) for key, values in sorted(merged.items())]
    for encoded, values in sorted(merged.items()):
        source, left, right, metric = encoded.split("__", 3)
        stats = _bootstrap_floor(
            values,
            seed=SPLIT_SEED + sum(ord(char) for char in f"{source}|{left}|{right}|{metric}"),
            draws=2000,
            return_draws=True,
        )
        macro_inputs.append({"metric": metric, "draws": stats.pop("_draws")})
        summary_rows.append({
            "estimand": "matched_budget_source_frozen",
            "source_environment_id": source,
            "left_target_environment_id": left,
            "right_target_environment_id": right,
            "metric": metric,
            "n_split_seeds": 30,
            "n_perturbations_primary_min": int(values[0]["cross_left_a"].shape[0]),
            "min_cells_primary": 20,
            "eligibility_min_cells": 20,
            "sensitivity_min_cells": 40,
            "ordering_estimator": "pairwise-order disagreement with U-statistic within-context floors; plugin/V identity retained as a separate diagnostic",
            "rank_displacement_estimator": "normalized rank displacement; secondary magnitude diagnostic only",
            "bootstrap_unit": "matched perturbation label; each draw also selects one of 30 measurement seeds and one unordered model-member pair",
            "measurement_definition": "fixed source-frozen mean prediction versus two independent full-size with-replacement raw-cell truth pseudoreplicates",
            "joint_definition": "source-frozen model member pairs crossed with two independent full-size with-replacement truth pseudoreplicates",
            "metric_entrypoint": "delta_cosine; third_party/systema/evaluation/centroid_accuracy.py-equivalent chunked distance; absolute_effect_rank_agreement",
            "refit_per_metric": 0,
            **stats,
        })
    output_summary = manifest_dir / f"{output_stem}_summary.csv"
    output_floors = manifest_dir / f"{output_stem}_floors.csv"
    output_ordering = manifest_dir / f"{output_stem}_ordering.csv"
    output_macro = manifest_dir / f"{output_stem}_macro.csv"
    output_report = manifest_dir / f"{output_stem}.json"
    pd.DataFrame(summary_rows).to_csv(output_summary, index=False)
    pd.concat(frames, ignore_index=True).to_csv(output_floors, index=False)
    pd.DataFrame(ordering_rows).to_csv(output_ordering, index=False)
    pd.DataFrame(_synchronized_macro_rows(macro_inputs, group_columns=("metric",))).to_csv(output_macro, index=False)
    report = {
        "schema_version": 1,
        "benchmark_id": "ptl_context_v2_frangieh_claim_lock_measurement_v1",
        "status": "full_size_nonparametric_measurement_and_joint_floors_executed",
        "analysis_label": "full_size_nonparametric",
        "resampling_mode": "full_size_nonparametric",
        "raw_source": "data/raw/scperturb_v1.4/FrangiehIzar2021_RNA.h5ad",
        "split_seeds": list(SPLIT_SEEDS),
        "min_cells_primary": 20,
        "eligibility_min_cells": 20,
        "sensitivity_min_cells": 40,
        "floor_rows": output_floors.relative_to(root).as_posix(),
        "ordering_rows": output_ordering.relative_to(root).as_posix(),
        "summary_rows": output_summary.relative_to(root).as_posix(),
        "macro_rows": output_macro.relative_to(root).as_posix(),
        "macro_ci_method": "synchronized within-draw macro of row bootstrap draws",
        "source_chunk_stems": part_stems,
        "ordering_estimand": "tie-aware pairwise-order disagreement; finite within floors use U-statistics",
        "uncertainty_scope": "raw-cell/pseudoreplicate sampling uncertainty; not all biological variation",
    }
    output_report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--part-stems", nargs="+", required=True)
    parser.add_argument("--output-stem", default="formal_v2_claim_lock_measurement_fullsize")
    args = parser.parse_args()
    print(json.dumps(merge(args.root, args.part_stems, args.output_stem), indent=2))


if __name__ == "__main__":
    main()
