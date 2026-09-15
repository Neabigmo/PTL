"""Merge disjoint measurement-depth worker outputs into canonical artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_measurement_depth_identifiability import FIXED_CELL_BUDGETS, FULL_LABEL, SPLIT_SEEDS  # noqa: E402
from src.evaluation.measurement_depth import depth_resolution_table, summarize_depth_rows  # noqa: E402


def _key_fingerprint(frame: pd.DataFrame, columns: list[str]) -> str:
    """Hash canonical key rows without materializing a large string column."""

    return hashlib.sha256(pd.util.hash_pandas_object(frame[columns], index=False).values.tobytes()).hexdigest()


def merge(root: Path = ROOT, parts_dir: Path | None = None, draws: int = 2000) -> dict[str, int | str]:
    root = root.resolve()
    parts_dir = (parts_dir or root / "artifacts/manifests/reliability_transport_measurement_depth_parts").resolve()
    part_dirs = sorted(path for path in parts_dir.iterdir() if path.is_dir())
    if not part_dirs:
        raise FileNotFoundError(f"no measurement-depth part directories under {parts_dir}")
    frames: dict[str, list[pd.DataFrame]] = {name: [] for name in ("items", "decision", "coverage")}
    expected_depths = {str(value) for value in FIXED_CELL_BUDGETS} | {FULL_LABEL}
    observed_depths: set[str] = set()
    for part_dir in part_dirs:
        item_path = part_dir / "reliability_transport_measurement_depth_items.csv"
        decision_path = part_dir / "reliability_transport_measurement_depth_decision.csv"
        coverage_path = part_dir / "reliability_transport_measurement_depth_coverage.csv"
        if not all(path.exists() for path in (item_path, decision_path, coverage_path)):
            raise FileNotFoundError(f"incomplete measurement-depth part: {part_dir}")
        item = pd.read_csv(item_path)
        decision = pd.read_csv(decision_path)
        coverage = pd.read_csv(coverage_path)
        depths = set(item["cell_budget_label"].astype(str).unique())
        if len(depths) != 1:
            raise ValueError(f"part must contain one depth, got {depths} in {part_dir}")
        depth = next(iter(depths))
        if depth in observed_depths:
            raise ValueError(f"duplicate depth part: {depth}")
        observed_depths.add(depth)
        if depth not in expected_depths:
            raise ValueError(f"unexpected depth label: {depth}")
        for frame_name, frame in (("items", item), ("decision", decision), ("coverage", coverage)):
            if set(frame["cell_budget_label"].astype(str).unique()) != {depth}:
                raise ValueError(f"{frame_name} depth mismatch in {part_dir}")
            frames[frame_name].append(frame)
    if observed_depths != expected_depths:
        raise ValueError(f"depth coverage mismatch: expected={sorted(expected_depths)} observed={sorted(observed_depths)}")

    items = pd.concat(frames["items"], ignore_index=True).sort_values(
        ["cell_budget_order", "source_environment_id", "left_target_environment_id", "right_target_environment_id", "metric", "split_seed", "perturbation_label"]
    ).reset_index(drop=True)
    decisions = pd.concat(frames["decision"], ignore_index=True).sort_values(
        ["cell_budget_order", "source_environment_id", "left_target_environment_id", "right_target_environment_id", "metric", "split_seed", "decision_budget_fraction"]
    ).reset_index(drop=True)
    coverage = pd.concat(frames["coverage"], ignore_index=True).sort_values(
        ["cell_budget_order", "left_target_environment_id", "right_target_environment_id"]
    ).reset_index(drop=True)
    item_key = ["source_environment_id", "left_target_environment_id", "right_target_environment_id", "metric", "split_seed", "perturbation_label", "cell_budget_label"]
    decision_key = [
        "source_environment_id",
        "left_target_environment_id",
        "right_target_environment_id",
        "metric",
        "split_seed",
        "cell_budget_label",
        "decision_budget_fraction",
    ]
    if items.duplicated(item_key).any():
        raise ValueError("duplicate canonical measurement-depth item keys")
    if decisions.duplicated(decision_key).any():
        raise ValueError("duplicate canonical measurement-depth decision keys")
    if set(items["split_seed"].astype(int)) != set(SPLIT_SEEDS):
        raise ValueError("measurement-depth item seed schedule is incomplete")
    summary = summarize_depth_rows(items, draws=draws, ci_level=0.90)
    resolution = depth_resolution_table(summary)
    output_dir = root / "artifacts/manifests"
    items.to_csv(output_dir / "reliability_transport_measurement_depth_items.csv", index=False)
    summary.to_csv(output_dir / "reliability_transport_measurement_depth_summary.csv", index=False)
    decisions.to_csv(output_dir / "reliability_transport_measurement_depth_decision.csv", index=False)
    coverage.to_csv(output_dir / "reliability_transport_measurement_depth_coverage.csv", index=False)
    resolution.to_csv(output_dir / "reliability_transport_measurement_depth_resolution.csv", index=False)
    report = {
        "schema_version": 1,
        "status": "executed",
        "analysis": "measurement_depth_identifiability",
        "source_frozen_prediction": "artifacts/source_data/frangieh_source_frozen_predictions.npz",
        "raw_source": "data/raw/scperturb_v1.4/FrangiehIzar2021_RNA.h5ad",
        "split_seeds": list(SPLIT_SEEDS),
        "fixed_cell_budgets": list(FIXED_CELL_BUDGETS),
        "full_depth_label": FULL_LABEL,
        "full_depth_contract": "per-label matched target cell count, independently sampled with replacement",
        "decision_budgets": [0.05, 0.10, 0.20, 0.50],
        "bootstrap_draws": int(draws),
        "bootstrap_unit": "measurement seed after perturbation-level exact burden aggregation",
        "metrics": sorted(items["metric"].unique().tolist()),
        "outputs": {
            "items": "artifacts/manifests/reliability_transport_measurement_depth_items.csv",
            "summary": "artifacts/manifests/reliability_transport_measurement_depth_summary.csv",
            "decision": "artifacts/manifests/reliability_transport_measurement_depth_decision.csv",
            "coverage": "artifacts/manifests/reliability_transport_measurement_depth_coverage.csv",
            "resolution": "artifacts/manifests/reliability_transport_measurement_depth_resolution.csv",
        },
        "checks": {
            "part_directories": len(part_dirs),
            "depth_labels": sorted(observed_depths),
            "item_rows": int(len(items)),
            "summary_rows": int(len(summary)),
            "decision_rows": int(len(decisions)),
            "coverage_rows": int(len(coverage)),
            "resolution_rows": int(len(resolution)),
            "item_key_sha256": _key_fingerprint(items, item_key),
            "no_prediction_refit": True,
            "target_outcomes_not_used": True,
        },
    }
    (output_dir / "reliability_transport_measurement_depth.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return {"status": "executed", **report["checks"]}


if __name__ == "__main__":
    merge()
