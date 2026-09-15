"""Merge source-independent full-size claim-lock replication chunks.

The workers persist the per-seed bootstrap inputs rather than only their
summary statistics.  This makes the final interval reproducible after a
parallel run and prevents a partial chunk from being mistaken for the full
30-seed analysis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_formal_v2_claim_lock_replication import (
    BOOTSTRAP_DRAWS,
    CONTEXTS,
    CANDIDATE_ID,
    METRICS,
    MIN_CELLS_PRIMARY,
    SPLIT_SEEDS,
    SPLIT_SEED,
    _bootstrap_floor,
    _load_registry_decision,
    _synchronized_macro_rows,
    _write_executed_boundary,
)


INPUT_FIELDS = (
    "cross_left_a", "cross_right_a", "cross_left_b", "cross_right_b",
    "meas_left_a", "meas_left_b", "meas_right_a", "meas_right_b",
    "joint_left_a", "joint_left_b", "joint_right_a", "joint_right_b",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_inputs(path: Path) -> dict[tuple[int, str, str], list[dict[str, np.ndarray]]]:
    archive = np.load(path, allow_pickle=False)
    result: dict[tuple[int, str, str], list[dict[str, np.ndarray]]] = {}
    try:
        encoded_keys = sorted({
            "__".join(name.split("__")[:3])
            for name in archive.files
            if name.endswith("__cross_left_a")
        })
        for encoded in encoded_keys:
            minimum_text, source, metric = encoded.split("__", 2)
            key = (int(minimum_text), source, metric)
            arrays = {field: np.asarray(archive[f"{encoded}__{field}"]) for field in INPUT_FIELDS}
            shapes = {array.shape for array in arrays.values()}
            if len(shapes) != 1 or next(iter(shapes))[0] < 1:
                raise ValueError(f"invalid bootstrap input shapes for {encoded}: {shapes}")
            result[key] = [
                {field: arrays[field][index] for field in INPUT_FIELDS}
                for index in range(next(iter(shapes))[0])
            ]
    finally:
        archive.close()
    return result


def _stream_floor(
    root: Path,
    records: list[dict[str, Any]],
    encoded: str,
    *,
    seed: int,
    draws: int,
    ) -> dict[str, Any]:
    """Reconstruct the bounded inputs and use the canonical estimator."""

    inputs: list[dict[str, np.ndarray]] = []
    for record in records:
        with np.load(root / record["path"], allow_pickle=False) as archive:
            arrays = {field: np.asarray(archive[f"{encoded}__{field}"]) for field in INPUT_FIELDS}
            n_items = int(arrays["cross_left_a"].shape[0])
            inputs.extend(
                [{field: arrays[field][index] for field in INPUT_FIELDS} for index in range(n_items)]
            )
    if not inputs:
        raise ValueError(f"no bootstrap inputs found for {encoded}")
    return _bootstrap_floor(inputs, seed=seed, draws=draws, return_draws=True)


def _summary_from_inputs(
    inputs: dict[tuple[int, str, str], list[dict[str, np.ndarray]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (minimum_cells, source, metric), values in sorted(inputs.items()):
        if minimum_cells != MIN_CELLS_PRIMARY or not values:
            continue
        stats = _bootstrap_floor(
            values,
            seed=SPLIT_SEED + sum(ord(char) for char in f"nadig|{minimum_cells}|{source}|{metric}"),
            draws=BOOTSTRAP_DRAWS,
        )
        rows.append({
            "candidate_id": CANDIDATE_ID,
            "estimand": "matched_budget_source_frozen_replication",
            "analysis_label": "primary_min20",
            "source_context_id": source,
            "left_target_context_id": CONTEXTS[0],
            "right_target_context_id": CONTEXTS[1],
            "metric": metric,
            "n_split_seeds": len(values),
            "n_perturbations": int(values[0]["cross_left_a"].shape[0]),
            "eligibility_min_cells": int(minimum_cells),
            "split_seed_start": int(SPLIT_SEEDS[0]),
            "split_seed_end": int(SPLIT_SEEDS[-1]),
            "bootstrap_unit": "matched perturbation label; each draw selects one measurement seed and one unordered model-member pair",
            "measurement_definition": "fixed source-frozen mean prediction versus independent full-size with-replacement raw-cell pseudo-replicates",
            "joint_definition": "source-frozen model member pairs crossed with two independent full-size with-replacement truth pseudoreplicates",
            "metric_entrypoint": "same delta cosine, pinned Systema centroid-accuracy, and absolute-effect-rank implementation as Frangieh Claim Lock",
            "refit_per_metric": 0,
            **stats,
        })
    if len(rows) != len(CONTEXTS) * len(METRICS):
        raise ValueError(f"expected {len(CONTEXTS) * len(METRICS)} primary summary rows, got {len(rows)}")
    return rows


def _summary_from_stream(root: Path, records: list[dict[str, Any]], keys: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    macro_draw_rows: list[dict[str, Any]] = []
    for encoded in sorted(keys):
        minimum_text, source, metric = encoded.split("__", 2)
        minimum_cells = int(minimum_text)
        if minimum_cells != MIN_CELLS_PRIMARY:
            continue
        with np.load(root / records[0]["path"], allow_pickle=False) as archive:
            shape = np.asarray(archive[f"{encoded}__cross_left_a"]).shape
        stats = _stream_floor(
            root,
            records,
            encoded,
            seed=SPLIT_SEED + sum(ord(char) for char in f"nadig|{minimum_cells}|{source}|{metric}"),
            draws=BOOTSTRAP_DRAWS,
        )
        macro_draw_rows.append({
            "analysis_label": "primary_min20",
            "source_context_id": source,
            "metric": metric,
            "draws": stats.pop("_draws"),
        })
        rows.append({
            "candidate_id": CANDIDATE_ID,
            "estimand": "matched_budget_source_frozen_replication",
            "analysis_label": "primary_min20",
            "source_context_id": source,
            "left_target_context_id": CONTEXTS[0],
            "right_target_context_id": CONTEXTS[1],
            "metric": metric,
            "n_split_seeds": int(sum(record["n_items"] for record in records)),
            "n_perturbations": int(shape[1]),
            "eligibility_min_cells": minimum_cells,
            "split_seed_start": int(SPLIT_SEEDS[0]),
            "split_seed_end": int(SPLIT_SEEDS[-1]),
            "bootstrap_unit": "matched perturbation label; each draw selects one measurement seed and one unordered model-member pair",
            "measurement_definition": "fixed source-frozen mean prediction versus independent full-size with-replacement raw-cell pseudo-replicates",
            "joint_definition": "source-frozen model member pair crossed with independent truth halves",
            "metric_entrypoint": "same delta cosine, pinned Systema centroid-accuracy, and absolute-effect-rank implementation as Frangieh Claim Lock",
            "refit_per_metric": 0,
            **stats,
        })
    expected = len(CONTEXTS) * len(METRICS)
    if len(rows) != expected:
        raise ValueError(f"expected {expected} primary summary rows, got {len(rows)}")
    return rows, _synchronized_macro_rows(macro_draw_rows, group_columns=("analysis_label", "metric"))


def _records_for_report(root: Path, report: dict[str, Any]) -> list[dict[str, Any]]:
    raw_records = report.get("bootstrap_input_files", [])
    if not raw_records:
        input_path = report.get("bootstrap_input_path")
        if not input_path or not str(input_path).endswith(".npz"):
            raise ValueError("worker report does not contain streamed bootstrap files")
        raw_records = [{"path": input_path, "split_seeds": report.get("split_seeds", [])}]
    records: list[dict[str, Any]] = []
    for raw_record in raw_records:
        path = str(raw_record["path"])
        with np.load(root / path, allow_pickle=False) as archive:
            cross_names = [name for name in archive.files if name.endswith("__cross_left_a")]
            if not cross_names:
                raise ValueError(f"bootstrap chunk contains no cross-context input: {path}")
            keys = sorted("__".join(name.split("__")[:3]) for name in cross_names)
            n_items = int(np.asarray(archive[f"{keys[0]}__cross_left_a"]).shape[0])
        seeds = [int(seed) for seed in raw_record.get("split_seeds", [])]
        if len(seeds) != n_items:
            raise ValueError(f"bootstrap chunk seed count does not match array rows: {path}")
        records.append({"path": path, "split_seeds": seeds, "keys": keys, "n_items": n_items})
    return records


def merge(root: Path, part_stems: list[str], output_stem: str) -> dict[str, Any]:
    root = root.resolve()
    manifest_dir = root / "artifacts/manifests"
    expected_seeds = list(SPLIT_SEEDS)
    reports: list[dict[str, Any]] = []
    all_floor_frames: list[pd.DataFrame] = []
    input_records: list[dict[str, Any]] = []
    seen_seeds: list[int] = []
    for stem in part_stems:
        report = _read_json(manifest_dir / f"{stem}.json")
        if report.get("resampling_mode") != "full_size_nonparametric":
            raise ValueError(f"{stem} is not a full-size nonparametric worker")
        part_seeds = [int(seed) for seed in report.get("split_seeds", [])]
        if not part_seeds or len(part_seeds) != len(set(part_seeds)):
            raise ValueError(f"{stem} has missing or duplicate worker seeds")
        overlap = sorted(set(seen_seeds) & set(part_seeds))
        if overlap:
            raise ValueError(f"worker seed overlap detected: {overlap}")
        if not set(part_seeds).issubset(set(expected_seeds)):
            raise ValueError(f"{stem} contains seeds outside the declared schedule")
        floor_path = manifest_dir / f"{stem}_floors.csv"
        frame = pd.read_csv(floor_path)
        if frame.empty or set(frame["split_seed"].astype(int)) != set(part_seeds):
            raise ValueError(f"{stem} floor rows do not cover its declared seeds")
        records = _records_for_report(root, report)
        if [seed for record in records for seed in record["split_seeds"]] != part_seeds:
            raise ValueError(f"{stem} bootstrap chunks are not in the declared seed order")
        input_records.extend(records)
        reports.append(report)
        all_floor_frames.append(frame)
        seen_seeds.extend(part_seeds)
    input_records.sort(key=lambda record: record["split_seeds"][0])
    ordered_input_seeds = [seed for record in input_records for seed in record["split_seeds"]]
    if sorted(seen_seeds) != expected_seeds or ordered_input_seeds != expected_seeds:
        raise ValueError(f"worker seeds do not exactly cover the ordered 30-seed schedule: {ordered_input_seeds}")
    key_sets = {tuple(record["keys"]) for record in input_records}
    if len(key_sets) != 1:
        raise ValueError("bootstrap chunks do not share the same metric/source key set")
    keys = list(next(iter(key_sets)))
    summary_rows, macro_rows = _summary_from_stream(root, input_records, keys)
    floors = pd.concat(all_floor_frames, ignore_index=True)
    floors = floors.sort_values(["split_seed", "source_context_id", "metric", "member_pair"]).reset_index(drop=True)
    expected_floor_rows = len(expected_seeds) * len(CONTEXTS) * len(METRICS) * 3
    if len(floors) != expected_floor_rows:
        raise ValueError(f"expected {expected_floor_rows} floor rows, got {len(floors)}")
    if not np.isfinite(floors.select_dtypes(include=[np.number]).to_numpy()).all():
        raise ValueError("merged floor table contains non-finite numeric values")
    output_summary = manifest_dir / f"{output_stem}.csv"
    output_floors = manifest_dir / f"{output_stem}_floors.csv"
    output_macro = manifest_dir / f"{output_stem}_macro.csv"
    output_input_manifest = manifest_dir / f"{output_stem}_bootstrap_inputs.json"
    output_report = manifest_dir / f"{output_stem}.json"
    pd.DataFrame(summary_rows).to_csv(output_summary, index=False)
    pd.DataFrame(macro_rows).to_csv(output_macro, index=False)
    floors.to_csv(output_floors, index=False)
    serialized_records = [
        {"path": record["path"], "split_seeds": record["split_seeds"], "keys": record["keys"]}
        for record in input_records
    ]
    output_input_manifest.write_text(json.dumps({"files": serialized_records, "keys": keys}, indent=2) + "\n", encoding="utf-8")
    registry_report, selected = _load_registry_decision(root)
    base = reports[0]
    report = dict(base)
    report.update({
        "status": "independent_replication_claim_lock_executed",
        "split_seeds": expected_seeds,
        "seed_range_indices": [0, len(expected_seeds)],
        "summary_path": output_summary.relative_to(root).as_posix(),
        "floor_path": output_floors.relative_to(root).as_posix(),
        "macro_path": output_macro.relative_to(root).as_posix(),
        "macro_ci_method": "synchronized within-draw macro of row bootstrap draws",
        "bootstrap_input_path": output_input_manifest.relative_to(root).as_posix(),
        "bootstrap_input_keys": keys,
        "bootstrap_input_files": serialized_records,
        "streamed_bootstrap_inputs": True,
        "parallel_merge_contract": {
            "worker_stems": part_stems,
            "worker_count": len(part_stems),
            "source_disjoint": True,
            "seed_union_verified": expected_seeds,
            "floor_rows_verified": expected_floor_rows,
            "bootstrap_inputs_recombined_before_interval_estimation": True,
            "interval_estimation_mode": "field-wise streaming over seed chunk archives",
        },
    })
    output_report.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    _write_executed_boundary(root, registry_report, selected, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--part-stems", nargs=2, required=True)
    parser.add_argument("--output-stem", default="formal_v2_claim_lock_replication_nadig_fullsize")
    args = parser.parse_args()
    print(json.dumps(merge(args.root, args.part_stems, args.output_stem), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
