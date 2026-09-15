"""Augment an existing fixed-depth shard with member-level risk rows.

The main depth worker historically emitted member-level risks only for the
full-depth surface.  This narrow pass reuses the exact sampling and scoring
contract for fixed budgets, without discarding already-computed item and
decision tables.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_formal_v2_claim_lock_measurement import (  # noqa: E402
    CONDITION_BY_ENVIRONMENT,
    ENVIRONMENTS,
    METRICS,
    PAIR_ORDER,
    RAW_PATH,
    SPLIT_SEEDS,
    _aggregate_seed_plans,
    _load_metadata,
    _load_panel_positions,
    _metric_risks_batch,
    _profile_lookup,
    _seed_plan,
)
from scripts.run_measurement_depth_identifiability import (  # noqa: E402
    DEPTH_ORDER,
    _budget_map,
    _pair_label_set,
)


FIXED_BUDGETS = (10, 20, 40, 80, 160)


def run(
    root: Path = ROOT,
    *,
    output_dir: Path,
    label_start_index: int,
    label_stop_index: int,
    seed_chunk: int = 3,
    seed_start_index: int = 0,
    seed_stop_index: int | None = None,
    risk_output_name: str | None = None,
) -> dict[str, object]:
    root = root.resolve()
    output_dir = output_dir.resolve()
    report_path = output_dir / "reliability_transport_measurement_depth.json"
    if not report_path.is_file():
        raise FileNotFoundError(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("universe_mode") != "matched_fixed":
        raise ValueError("risk augmentation requires a matched-fixed shard")
    payload = np.load(root / "artifacts/source_data/frangieh_source_frozen_predictions.npz", allow_pickle=False)
    all_labels = payload["perturbation_label"].astype(str).tolist()
    labels = all_labels[int(label_start_index):int(label_stop_index)]
    if report.get("label_index_range") != [int(label_start_index), int(label_stop_index)]:
        raise ValueError("CLI label range does not match shard report")
    seed_start_index = int(seed_start_index)
    seed_stop_index = len(SPLIT_SEEDS) if seed_stop_index is None else int(seed_stop_index)
    if not 0 <= seed_start_index < seed_stop_index <= len(SPLIT_SEEDS):
        raise ValueError("CLI seed range is outside the declared schedule")
    split_seeds = tuple(SPLIT_SEEDS[seed_start_index:seed_stop_index])
    predictions = payload["prediction"].astype(np.float32, copy=False)
    panel = payload["evaluation_gene_symbols"].astype(str).tolist()
    metadata = _load_metadata(root, labels)
    metadata_by_row = metadata.set_index("row_index")
    groups = {
        (str(environment), str(label)): group["row_index"].to_numpy(dtype=np.int64)
        for (environment, label), group in metadata.groupby(["environment_key", "perturbation_label"], sort=True)
    }
    required_groups = [(environment, label) for environment in ENVIRONMENTS for label in [*labels, "control"]]
    missing = [key for key in required_groups if key not in groups]
    if missing:
        raise ValueError(f"raw Frangieh groups are missing: {missing[:5]}")
    fixed_metadata = _load_metadata(root, all_labels)
    fixed_groups = {
        (str(environment), str(label)): group["row_index"].to_numpy(dtype=np.int64)
        for (environment, label), group in fixed_metadata.groupby(["environment_key", "perturbation_label"], sort=True)
    }
    fixed_universes = {
        pair: set(_pair_label_set(fixed_groups, all_labels, pair[0], pair[1], max(FIXED_BUDGETS), full=False))
        for pair in PAIR_ORDER
    }
    panel_positions = _load_panel_positions(root, panel)
    label_index = {label: index for index, label in enumerate(labels)}
    rows: list[dict[str, object]] = []
    with h5py.File(root / RAW_PATH.relative_to(ROOT), "r") as handle:
        for budget in FIXED_BUDGETS:
            budget_values = _budget_map(groups, labels, budget, full=False)
            for start in range(0, len(split_seeds), max(1, int(seed_chunk))):
                selected_seeds = split_seeds[start:start + max(1, int(seed_chunk))]
                plans = [_seed_plan(groups, labels, budget_values, seed, replacement=True) for seed in selected_seeds]
                outputs = _aggregate_seed_plans(handle, plans, metadata_by_row, panel_positions, panel_size=len(panel))
                for seed, plan, output in zip(selected_seeds, plans, outputs):
                    profiles = _profile_lookup(plan, output)
                    fixed_risk_cache: dict[tuple[str, str], tuple[dict[str, np.ndarray], dict[str, np.ndarray]]] = {}
                    fixed_label_cache: dict[str, list[str]] = {}
                    for environment in ENVIRONMENTS:
                        environment_labels = [
                            label for label in labels
                            if any(
                                environment in (left_candidate, right_candidate)
                                and (left_candidate, right_candidate, label) in budget_values
                                for left_candidate, right_candidate in PAIR_ORDER
                            )
                        ]
                        if not environment_labels:
                            continue
                        pair_for_label = {
                            label: next(
                                (pair for pair in PAIR_ORDER if environment in pair and (pair[0], pair[1], label) in budget_values),
                                None,
                            )
                            for label in environment_labels
                        }
                        environment_labels = [label for label in environment_labels if pair_for_label[label] is not None]
                        if not environment_labels:
                            continue
                        control_keys = [
                            (left_candidate, right_candidate, "control")
                            for left_candidate, right_candidate in PAIR_ORDER
                            if environment in (left_candidate, right_candidate)
                            and (left_candidate, right_candidate, "control") in budget_values
                        ]
                        control_budget = budget_values[control_keys[0]]
                        truth = np.stack([
                            np.stack([
                                profiles[(environment, label, half, budget_values[pair_for_label[label][0], pair_for_label[label][1], label])]
                                - profiles[(environment, "control", half, control_budget)]
                                for label in environment_labels
                            ])
                            for half in (0, 1)
                        ])
                        fixed_label_cache[environment] = environment_labels
                        for source_index, source in enumerate(ENVIRONMENTS):
                            fixed_risk_cache[(source, environment)] = _metric_risks_batch(
                                predictions[source_index][np.asarray([label_index[label] for label in environment_labels])], truth
                            )
                    for source in ENVIRONMENTS:
                        for left, right in PAIR_ORDER:
                            if source not in (left, right):
                                continue
                            eligible = _pair_label_set(
                                groups, labels, left, right, budget, full=False,
                                fixed_universe=fixed_universes[(left, right)],
                            )
                            if not eligible:
                                continue
                            for metric in METRICS:
                                left_union = fixed_label_cache[left]
                                right_union = fixed_label_cache[right]
                                left_positions = [left_union.index(label) for label in eligible]
                                right_positions = [right_union.index(label) for label in eligible]
                                left_mean = fixed_risk_cache[(source, left)][0][metric][:, left_positions]
                                right_mean = fixed_risk_cache[(source, right)][0][metric][:, right_positions]
                                left_members = fixed_risk_cache[(source, left)][1][metric][:, :, left_positions]
                                right_members = fixed_risk_cache[(source, right)][1][metric][:, :, right_positions]
                                for index, label in enumerate(eligible):
                                    for replicate_index in range(left_mean.shape[0]):
                                        row = {
                                            "source_environment_id": source,
                                            "left_target_environment_id": left,
                                            "right_target_environment_id": right,
                                            "metric": metric,
                                            "split_seed": int(seed),
                                            "perturbation_label": label,
                                            "measurement_replicate": int(replicate_index),
                                            "cell_budget": int(budget),
                                            "cell_budget_label": str(budget),
                                            "cell_budget_order": DEPTH_ORDER[budget],
                                            "universe_mode": "matched_fixed",
                                            "left_risk": float(left_mean[replicate_index, index]),
                                            "right_risk": float(right_mean[replicate_index, index]),
                                        }
                                        for member_index in range(left_members.shape[1]):
                                            row[f"left_member_{member_index}"] = float(left_members[replicate_index, member_index, index])
                                            row[f"right_member_{member_index}"] = float(right_members[replicate_index, member_index, index])
                                        rows.append(row)
    risk_output_name = risk_output_name or "reliability_transport_measurement_depth_risks.csv"
    if Path(risk_output_name).name != risk_output_name:
        raise ValueError("risk_output_name must be a file name, not a path")
    output_path = output_dir / risk_output_name
    pd.DataFrame(rows).to_csv(output_path, index=False)
    if risk_output_name == "reliability_transport_measurement_depth_risks.csv":
        report["checks"]["risk_rows"] = int(len(rows))
        report.setdefault("outputs", {})["risks"] = output_path.relative_to(root).as_posix()
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    result = {
        "status": "executed", "risk_rows": int(len(rows)),
        "seed_index_range": [seed_start_index, seed_stop_index],
        "output": output_path.relative_to(root).as_posix(),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label-start-index", type=int, required=True)
    parser.add_argument("--label-stop-index", type=int, required=True)
    parser.add_argument("--seed-chunk", type=int, default=3)
    parser.add_argument("--seed-start-index", type=int, default=0)
    parser.add_argument("--seed-stop-index", type=int, default=None)
    parser.add_argument("--risk-output-name", default=None)
    args = parser.parse_args()
    run(
        args.root, output_dir=args.output_dir,
        label_start_index=args.label_start_index, label_stop_index=args.label_stop_index,
        seed_chunk=args.seed_chunk, seed_start_index=args.seed_start_index,
        seed_stop_index=args.seed_stop_index, risk_output_name=args.risk_output_name,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
