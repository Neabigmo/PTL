"""Build the fixed perturbation-universe measurement-depth analysis.

The legacy depth run used a coverage universe that changed with cell budget.
This merger keeps that analysis intact and creates a separate, auditable
matched-fixed canonical surface.  The fixed universe is frozen at the highest
requested fixed budget (160 cells), while the existing full-depth label shards
are filtered to exactly the same labels before summary and decision analysis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_measurement_depth_identifiability import (  # noqa: E402
    ENVIRONMENTS,
    FULL_LABEL,
    FULL_ORDER,
    PAIR_ORDER,
    SPLIT_SEEDS,
    _append_decision_rows,
    _load_metadata,
    _pair_label_set,
)
from src.evaluation.measurement_depth import (  # noqa: E402
    hierarchical_macro_bootstrap,
    depth_resolution_table,
    summarize_depth_rows,
)


FIXED_BUDGETS = (10, 20, 40, 80, 160)
FULL_BUDGET = 10**9


def _sha256_labels(labels: list[str]) -> str:
    return hashlib.sha256("\n".join(labels).encode("utf-8")).hexdigest()


def _read_fixed_parts(parts_root: Path, all_labels: list[str], fixed: dict[tuple[str, str], set[str]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    item_frames: list[pd.DataFrame] = []
    decision_frames: list[pd.DataFrame] = []
    coverage_frames: list[pd.DataFrame] = []
    risk_frames: list[pd.DataFrame] = []
    # Source-disjoint workers may materialize all fixed budgets in a single
    # shard directory. Prefer that explicit six-way surface when present;
    # legacy budget directories may contain stale intermediates without the
    # completed member-level risk grid.
    shard_root = parts_root / "shards6"
    if shard_root.is_dir():
        part_dirs = sorted(path for path in shard_root.glob("shard_*") if path.is_dir())
    else:
        part_dirs = []
        for budget in FIXED_BUDGETS:
            budget_root = parts_root / f"budget_{budget}"
            if budget_root.is_dir():
                part_dirs.extend(path for path in budget_root.iterdir() if path.is_dir())
        if not part_dirs:
            part_dirs = sorted(path for path in parts_root.glob("shard_*") if path.is_dir())
        else:
            part_dirs = sorted(set(part_dirs))
        if not part_dirs:
            raise FileNotFoundError(f"no matched-fixed parts under {parts_root}")
    loaded_parts: list[tuple[int, int, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]] = []
    seen_ranges: list[tuple[int, int]] = []
    for part_dir in part_dirs:
        report_path = part_dir / "reliability_transport_measurement_depth.json"
        paths = [part_dir / f"reliability_transport_measurement_depth_{name}.csv" for name in ("items", "decision", "coverage", "risks")]
        risk_part_paths = sorted(part_dir.glob("reliability_transport_measurement_depth_risks_seed_*.csv"))
        if not report_path.is_file() or not all(path.is_file() for path in paths[:3]):
            raise FileNotFoundError(f"incomplete matched-fixed part: {part_dir}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        start, stop = map(int, report["label_index_range"])
        if not 0 <= start < stop <= len(all_labels):
            raise ValueError(f"invalid label range in {part_dir}")
        if report.get("universe_mode") != "matched_fixed":
            raise ValueError(f"part was not run under matched_fixed: {part_dir}")
        item = pd.read_csv(paths[0])
        decision = pd.read_csv(paths[1])
        coverage = pd.read_csv(paths[2])
        if risk_part_paths:
            if any(path.stat().st_size <= 2 for path in risk_part_paths):
                raise ValueError(f"empty matched-fixed risk seed part in {part_dir}")
            risks = pd.concat([pd.read_csv(path) for path in risk_part_paths], ignore_index=True, sort=False)
        else:
            if not paths[3].is_file() or paths[3].stat().st_size <= 2:
                raise FileNotFoundError(f"missing matched-fixed risks in {part_dir}")
            risks = pd.read_csv(paths[3])
        for frame in (item, decision, coverage, risks):
            if "universe_mode" not in frame.columns:
                raise ValueError(f"universe_mode missing in {part_dir}")
            if not frame.empty and set(frame["universe_mode"].astype(str).unique()) != {"matched_fixed"}:
                raise ValueError(f"universe mode mismatch in {part_dir}")
        expected = set(all_labels[start:stop]) & set().union(*fixed.values())
        if set(item["perturbation_label"].astype(str).unique()) != expected:
            raise ValueError(f"item labels do not match declared range in {part_dir}")
        if not item.empty and set(item["split_seed"].astype(int).unique()) != set(SPLIT_SEEDS):
            raise ValueError(f"item seed schedule is incomplete in {part_dir}")
        seen_ranges.append((start, stop))
        loaded_parts.append((start, stop, item, decision, coverage, risks))
        seen_ranges.sort()
    cursor = 0
    for start, stop in seen_ranges:
        if start != cursor:
            raise ValueError(f"label range gap/overlap: expected {cursor}, got {start}")
        cursor = stop
    if cursor != len(all_labels):
        raise ValueError(f"label coverage incomplete: {cursor}/{len(all_labels)}")
    for budget in FIXED_BUDGETS:
        for start, stop, item_all, decision_all, coverage_all, risks_all in loaded_parts:
            item = item_all.loc[item_all["cell_budget_label"].astype(str).eq(str(budget))].copy()
            decision = decision_all.loc[decision_all["cell_budget_label"].astype(str).eq(str(budget))].copy()
            coverage = coverage_all.loc[coverage_all["cell_budget_label"].astype(str).eq(str(budget))].copy()
            risks = risks_all.loc[risks_all["cell_budget_label"].astype(str).eq(str(budget))].copy()
            item_frames.append(item)
            decision_frames.append(decision)
            coverage_frames.append(coverage)
            risk_frames.append(risks)
    return (
        pd.concat(item_frames, ignore_index=True),
        pd.concat(decision_frames, ignore_index=True),
        pd.concat(coverage_frames, ignore_index=True),
        pd.concat(risk_frames, ignore_index=True),
    )


def _fixed_universes(root: Path, all_labels: list[str]) -> dict[tuple[str, str], set[str]]:
    metadata = _load_metadata(root, all_labels)
    groups = {
        (str(environment), str(label)): group["row_index"].to_numpy(dtype=np.int64)
        for (environment, label), group in metadata.groupby(["environment_key", "perturbation_label"], sort=True)
    }
    return {
        pair: set(_pair_label_set(groups, all_labels, pair[0], pair[1], max(FIXED_BUDGETS), full=False))
        for pair in PAIR_ORDER
    }


def _filter_to_fixed(frame: pd.DataFrame, fixed: dict[tuple[str, str], set[str]]) -> pd.DataFrame:
    mask = [
        str(label) in fixed[(str(left), str(right))]
        for label, left, right in zip(frame["perturbation_label"], frame["left_target_environment_id"], frame["right_target_environment_id"])
    ]
    output = frame.loc[mask].copy()
    output["universe_mode"] = "matched_fixed"
    return output


def _normalize_budget_labels(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    if "cell_budget_label" in output.columns:
        output["cell_budget_label"] = output["cell_budget_label"].astype(str)
    return output


def _decisions_from_risks(risks: pd.DataFrame, all_labels: list[str]) -> pd.DataFrame:
    label_order = {label: index for index, label in enumerate(all_labels)}
    risks["_label_order"] = risks["perturbation_label"].astype(str).map(label_order)
    member_columns = sorted((column for column in risks.columns if column.startswith("left_member_")), key=lambda c: int(c.rsplit("_", 1)[1]))
    right_member_columns = sorted((column for column in risks.columns if column.startswith("right_member_")), key=lambda c: int(c.rsplit("_", 1)[1]))
    rows: list[dict[str, object]] = []
    group_columns = [
        "source_environment_id", "left_target_environment_id", "right_target_environment_id",
        "metric", "split_seed", "cell_budget", "cell_budget_label", "cell_budget_order",
    ]
    for (source, left, right, metric, seed, cell_budget, cell_budget_label, cell_budget_order), group in risks.groupby(
        group_columns, sort=False
    ):
        if source not in (left, right):
            continue
        group = group.sort_values(["measurement_replicate", "_label_order"])
        replicates = sorted(group["measurement_replicate"].astype(int).unique())
        labels = [label for label, _ in sorted(label_order.items(), key=lambda pair: pair[1]) if label in set(group["perturbation_label"].astype(str))]
        if len(group) != len(replicates) * len(labels) or len(replicates) < 2:
            raise ValueError(f"incomplete matched-fixed full-depth risk grid for {(source, left, right, metric, seed)}")
        left_mean = np.stack([group[group["measurement_replicate"].eq(rep)]["left_risk"].to_numpy(dtype=float) for rep in replicates])
        right_mean = np.stack([group[group["measurement_replicate"].eq(rep)]["right_risk"].to_numpy(dtype=float) for rep in replicates])
        left_members = np.stack([group[group["measurement_replicate"].eq(rep)][member_columns].to_numpy(dtype=float).T for rep in replicates])
        right_members = np.stack([group[group["measurement_replicate"].eq(rep)][right_member_columns].to_numpy(dtype=float).T for rep in replicates])
        _append_decision_rows(
            rows,
            source=str(source), left=str(left), right=str(right), metric=str(metric), split_seed=int(seed),
            cell_budget=int(cell_budget), cell_budget_label=str(cell_budget_label),
            left_risk=left_mean, right_risk=right_mean, left_members=left_members, right_members=right_members,
            universe_mode="matched_fixed",
        )
    return pd.DataFrame(rows)


def _full_decisions(root: Path, fixed: dict[tuple[str, str], set[str]], all_labels: list[str]) -> pd.DataFrame:
    risk_path = root / "artifacts/manifests/reliability_transport_measurement_depth_parts/full/reliability_transport_measurement_depth_risks.csv"
    if not risk_path.is_file():
        raise FileNotFoundError(risk_path)
    risks = _filter_to_fixed(pd.read_csv(risk_path), fixed)
    return _decisions_from_risks(risks, all_labels)


def run(root: Path = ROOT, *, parts_root: Path | None = None, draws: int = 2000) -> dict[str, object]:
    root = root.resolve()
    manifests = root / "artifacts/manifests"
    parts_root = (parts_root or manifests / "reliability_transport_measurement_depth_matched_fixed_parts").resolve()
    payload = np.load(root / "artifacts/source_data/frangieh_source_frozen_predictions.npz", allow_pickle=False)
    all_labels = payload["perturbation_label"].astype(str).tolist()
    fixed = _fixed_universes(root, all_labels)
    fixed_items, _fixed_part_decisions, fixed_coverage, fixed_risks = _read_fixed_parts(parts_root, all_labels, fixed)
    fixed_items = _normalize_budget_labels(fixed_items)
    fixed_coverage = _normalize_budget_labels(fixed_coverage)
    fixed_risks = _normalize_budget_labels(fixed_risks)
    full_items = pd.read_csv(manifests / "reliability_transport_measurement_depth_parts/full/reliability_transport_measurement_depth_items.csv")
    full_items = _normalize_budget_labels(_filter_to_fixed(full_items, fixed))
    full_items["universe_mode"] = "matched_fixed"
    full_risks = pd.read_csv(manifests / "reliability_transport_measurement_depth_parts/full/reliability_transport_measurement_depth_risks.csv")
    full_risks = _normalize_budget_labels(_filter_to_fixed(full_risks, fixed))
    full_risks["universe_mode"] = "matched_fixed"
    risks = pd.concat([fixed_risks, full_risks], ignore_index=True, sort=False)
    # Per-shard decision files are shard-local aggregates: their keys are
    # intentionally identical across disjoint label shards. Recompute the
    # decision boundary once from the merged member-level risks instead of
    # concatenating those non-additive summaries.
    fixed_decisions = _decisions_from_risks(fixed_risks, all_labels)
    full_decisions = _full_decisions(root, fixed, all_labels)
    full_coverage = pd.DataFrame([
        {
            "left_target_environment_id": left, "right_target_environment_id": right, "cell_budget": FULL_BUDGET,
            "cell_budget_label": FULL_LABEL, "cell_budget_order": 6, "n_labels_all": len(all_labels),
            "n_labels_matched": len(fixed[(left, right)]), "coverage_fraction": len(fixed[(left, right)]) / len(all_labels),
            "matched_universe_sha256": _sha256_labels(sorted(fixed[(left, right)])),
            "matched_universe_policy": "fixed universe frozen at 160 cells and reused at full depth; per-label full matched cell count",
            "universe_mode": "matched_fixed",
        }
        for left, right in PAIR_ORDER
    ])
    items = pd.concat([fixed_items, full_items], ignore_index=True, sort=False)
    decisions = pd.concat([fixed_decisions, full_decisions], ignore_index=True, sort=False)
    coverage = pd.concat([fixed_coverage, full_coverage], ignore_index=True, sort=False)
    coverage = coverage.drop_duplicates(["left_target_environment_id", "right_target_environment_id", "cell_budget_label", "universe_mode"], keep="first")
    item_key = ["source_environment_id", "left_target_environment_id", "right_target_environment_id", "metric", "split_seed", "perturbation_label", "cell_budget_label", "universe_mode"]
    decision_key = ["source_environment_id", "target_environment_id", "left_target_environment_id", "right_target_environment_id", "metric", "split_seed", "cell_budget_label", "decision_budget_fraction", "universe_mode"]
    if items.duplicated(item_key).any():
        raise ValueError("duplicate matched-fixed item keys")
    if decisions.duplicated(decision_key).any():
        raise ValueError("duplicate matched-fixed decision keys")
    items = items.sort_values(["cell_budget_order", "source_environment_id", "left_target_environment_id", "right_target_environment_id", "metric", "split_seed", "perturbation_label"]).reset_index(drop=True)
    decisions = decisions.sort_values(["cell_budget_order", "source_environment_id", "target_environment_id", "metric", "split_seed", "decision_budget_fraction"]).reset_index(drop=True)
    coverage = coverage.sort_values(["cell_budget_order", "left_target_environment_id", "right_target_environment_id"]).reset_index(drop=True)
    summary = summarize_depth_rows(items, draws=draws, ci_level=0.90)
    resolution = depth_resolution_table(summary)
    macro = hierarchical_macro_bootstrap(items, draws=draws)
    outputs = {
        "items": manifests / "reliability_transport_measurement_depth_matched_fixed_items.csv",
        "summary": manifests / "reliability_transport_measurement_depth_matched_fixed_summary.csv",
        "decision": manifests / "reliability_transport_measurement_depth_matched_fixed_decision.csv",
        "coverage": manifests / "reliability_transport_measurement_depth_matched_fixed_coverage.csv",
        "resolution": manifests / "reliability_transport_measurement_depth_matched_fixed_resolution.csv",
        "macro_bootstrap": manifests / "reliability_transport_measurement_depth_matched_fixed_macro_bootstrap.csv",
        "risks": manifests / "reliability_transport_measurement_depth_matched_fixed_risks.csv",
    }
    items.to_csv(outputs["items"], index=False)
    summary.to_csv(outputs["summary"], index=False)
    decisions.to_csv(outputs["decision"], index=False)
    coverage.to_csv(outputs["coverage"], index=False)
    resolution.to_csv(outputs["resolution"], index=False)
    macro.to_csv(outputs["macro_bootstrap"], index=False)
    risks.to_csv(outputs["risks"], index=False)
    report = {
        "schema_version": 2,
        "status": "executed",
        "analysis": "measurement_depth_identifiability_matched_fixed",
        "universe_mode": "matched_fixed",
        "fixed_universe_freeze_budget": max(FIXED_BUDGETS),
        "fixed_cell_budgets": list(FIXED_BUDGETS),
        "full_depth_label": FULL_LABEL,
        "decision_budgets": [0.05, 0.10, 0.20, 0.50],
        "bootstrap_draws": int(draws),
        "bootstrap_unit": "perturbation-label resampling nested with measurement-seed resampling; macro draws are explicit",
        "fixed_universe_sha256_by_pair": {f"{left}->{right}": _sha256_labels(sorted(fixed[(left, right)])) for left, right in PAIR_ORDER},
        "outputs": {key: path.relative_to(root).as_posix() for key, path in outputs.items()},
        "checks": {
            "item_rows": int(len(items)), "decision_rows": int(len(decisions)), "coverage_rows": int(len(coverage)),
            "summary_rows": int(len(summary)), "resolution_rows": int(len(resolution)), "macro_rows": int(len(macro)), "risk_rows": int(len(risks)),
            "directed_decision_rows": int(len(decisions)), "all_seeds": sorted(items["split_seed"].astype(int).unique().tolist()),
            "no_prediction_refit": True, "target_outcomes_not_used": True,
        },
    }
    (manifests / "reliability_transport_measurement_depth_matched_fixed.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--parts-root", type=Path, default=None)
    parser.add_argument("--draws", type=int, default=2000)
    args = parser.parse_args()
    run(args.root, parts_root=args.parts_root, draws=args.draws)
