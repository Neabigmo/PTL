"""Create hierarchical label-by-seed bootstrap draws for directed decisions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.decision_theory import DEFAULT_BUDGET_FRACTIONS, replicate_decision_curve  # noqa: E402


def _draw_group(group: pd.DataFrame, *, draws: int, seed: int) -> list[dict]:
    source = str(group["source_environment_id"].iloc[0])
    left = str(group["left_target_environment_id"].iloc[0])
    right = str(group["right_target_environment_id"].iloc[0])
    if source not in (left, right):
        return []
    target = right if source == left else left
    source_column, target_column = ("left_risk", "right_risk") if source == left else ("right_risk", "left_risk")
    group = group.copy()
    group["perturbation_label"] = group["perturbation_label"].astype(str)
    labels = sorted(group["perturbation_label"].unique())
    seeds = sorted(group["split_seed"].astype(int).unique())
    reps = sorted(group["measurement_replicate"].astype(int).unique())
    if len(labels) < 2 or len(seeds) < 2 or len(reps) < 2:
        return []
    table = group.groupby(["split_seed", "measurement_replicate", "perturbation_label"], sort=True)[[source_column, target_column]].mean()
    arrays = {s: {r: (table.loc[(s, r)].reindex(labels)[source_column].to_numpy(dtype=float), table.loc[(s, r)].reindex(labels)[target_column].to_numpy(dtype=float)) for r in reps} for s in seeds}
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for draw in range(int(draws)):
        seed_indices = rng.integers(0, len(seeds), size=len(seeds))
        label_indices = rng.integers(0, len(labels), size=len(labels))
        selected_curves: list[dict] = []
        floors: list[dict] = []
        for seed_index in seed_indices:
            seed_id = seeds[int(seed_index)]
            source_replicates = np.stack([arrays[seed_id][r][0][label_indices] for r in reps])
            target_replicates = np.stack([arrays[seed_id][r][1][label_indices] for r in reps])
            selected_curves.append({row["budget_fraction"]: row for row in replicate_decision_curve(source_replicates, target_replicates, budgets=DEFAULT_BUDGET_FRACTIONS)})
            floors.append({row["budget_fraction"]: row for row in replicate_decision_curve(target_replicates, target_replicates, budgets=DEFAULT_BUDGET_FRACTIONS, independent_only=True)})
        for budget in DEFAULT_BUDGET_FRACTIONS:
            curve = [item[float(budget)] for item in selected_curves]
            floor = [item[float(budget)] for item in floors]
            record = {
                "source_environment_id": source, "target_environment_id": target,
                "left_target_environment_id": left, "right_target_environment_id": right,
                "metric": str(group["metric"].iloc[0]), "cell_budget": group["cell_budget"].iloc[0],
                "cell_budget_label": group["cell_budget_label"].iloc[0], "cell_budget_order": group["cell_budget_order"].iloc[0],
                "decision_budget_fraction": float(budget), "bootstrap_draw": draw,
                "retention": float(np.mean([x["retention"] for x in curve])),
                "regret": float(np.mean([x["regret"] for x in curve])),
                "normalized_regret": float(np.mean([x["normalized_regret"] for x in curve])),
                "boundary_inversion": float(np.mean([x["boundary_inversion"] for x in curve])),
                "measurement_floor_regret": float(np.mean([x["regret"] for x in floor])),
                "excess_regret": float(np.mean([x["regret"] for x in curve]) - np.mean([x["regret"] for x in floor])),
                "bootstrap_unit": "perturbation_label_then_measurement_seed",
                "status": "executed",
            }
            rows.append(record)
    return rows


def run(
    root: Path = ROOT,
    *,
    draws: int = 1000,
    group_start_index: int = 0,
    group_stop_index: int | None = None,
    output_name: str | None = None,
) -> dict:
    root = root.resolve()
    manifests = root / "artifacts/manifests"
    sources = [
        manifests / "reliability_transport_measurement_depth_matched_fixed_risks.csv",
        manifests / "reliability_transport_measurement_depth_matched_fixed_parts/full/reliability_transport_measurement_depth_risks.csv",
        manifests / "reliability_transport_measurement_depth_parts/full/reliability_transport_measurement_depth_risks.csv",
    ]
    source = next((path for path in sources if path.is_file()), None)
    out = manifests / (output_name or "reliability_transport_measurement_depth_matched_fixed_decision_macro_bootstrap.csv")
    report_path = manifests / "reliability_transport_measurement_depth_decision_macro_bootstrap.json"
    if source is None:
        report = {"schema_version": 1, "status": "blocked_missing_risk_input"}
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return report
    risk = pd.read_csv(source, dtype={"cell_budget_label": "string"})
    required = {"source_environment_id", "left_target_environment_id", "right_target_environment_id", "metric", "split_seed", "perturbation_label", "measurement_replicate", "cell_budget", "cell_budget_label", "cell_budget_order", "left_risk", "right_risk"}
    missing = required.difference(risk.columns)
    if missing:
        raise ValueError(f"risk table missing columns: {sorted(missing)}")
    grouped = list(risk.groupby(["source_environment_id", "left_target_environment_id", "right_target_environment_id", "metric", "cell_budget_label"], sort=True, observed=True))
    start = max(0, int(group_start_index))
    stop = len(grouped) if group_stop_index is None else min(len(grouped), int(group_stop_index))
    if start > stop:
        raise ValueError(f"invalid group range: {start}:{stop}")
    rows: list[dict] = []
    for key, group in grouped[start:stop]:
        rows.extend(_draw_group(group, draws=draws, seed=20260910 + sum((i + 1) * sum(ord(c) for c in str(v)) for i, v in enumerate(key))))
    frame = pd.DataFrame(rows)
    frame.to_csv(out, index=False)
    report = {"schema_version": 1, "status": "executed", "input": source.relative_to(root).as_posix(), "output": out.relative_to(root).as_posix(), "draws": int(draws), "rows": int(len(frame)), "groups_total": len(grouped), "group_start_index": start, "group_stop_index": stop, "bootstrap_unit": "perturbation_label_then_measurement_seed", "orientation": "source selects; target evaluates; target self off-diagonal floor"}
    if group_start_index == 0 and group_stop_index is None and output_name is None:
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--group-start-index", type=int, default=0)
    parser.add_argument("--group-stop-index", type=int, default=None)
    parser.add_argument("--output-name", type=str, default=None)
    args = parser.parse_args()
    run(args.root, draws=args.draws, group_start_index=args.group_start_index, group_stop_index=args.group_stop_index, output_name=args.output_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
