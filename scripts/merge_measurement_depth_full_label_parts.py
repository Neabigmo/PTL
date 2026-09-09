"""Merge disjoint full-depth label shards and rebuild the full-depth decision curve."""

from __future__ import annotations

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
from src.evaluation.measurement_depth import depth_resolution_table, summarize_depth_rows  # noqa: E402


def _key_fingerprint(frame: pd.DataFrame, columns: list[str]) -> str:
    """Hash canonical key rows without materializing a large string column."""

    return hashlib.sha256(pd.util.hash_pandas_object(frame[columns], index=False).values.tobytes()).hexdigest()


def merge(root: Path = ROOT, parts_dir: Path | None = None, draws: int = 2000) -> dict[str, int | str]:
    root = root.resolve()
    full_dir = root / "artifacts/manifests/reliability_transport_measurement_depth_parts/full"
    parts_dir = (parts_dir or full_dir / "label_parts").resolve()
    part_dirs = sorted(path for path in parts_dir.iterdir() if path.is_dir())
    if not part_dirs:
        raise FileNotFoundError(f"no full-depth label parts under {parts_dir}")
    payload = np.load(root / "artifacts/source_data/frangieh_source_frozen_predictions.npz", allow_pickle=False)
    all_labels = payload["perturbation_label"].astype(str).tolist()
    metadata = _load_metadata(root, all_labels)
    groups = {
        (str(environment), str(label)): group["row_index"].to_numpy(dtype=np.int64)
        for (environment, label), group in metadata.groupby(["environment_key", "perturbation_label"], sort=True)
    }
    item_frames: list[pd.DataFrame] = []
    risk_frames: list[pd.DataFrame] = []
    ranges: list[tuple[int, int, str]] = []
    for part_dir in part_dirs:
        item_path = part_dir / "reliability_transport_measurement_depth_items.csv"
        risk_path = part_dir / "reliability_transport_measurement_depth_risks.csv"
        report_path = part_dir / "reliability_transport_measurement_depth.json"
        if not all(path.exists() for path in (item_path, risk_path, report_path)):
            raise FileNotFoundError(f"incomplete full-depth label part: {part_dir}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        start, stop = map(int, report["label_index_range"])
        if not 0 <= start < stop <= len(all_labels):
            raise ValueError(f"invalid label range in {part_dir}: {(start, stop)}")
        item = pd.read_csv(item_path)
        risk = pd.read_csv(risk_path)
        if set(item["cell_budget_label"].astype(str).unique()) != {FULL_LABEL}:
            raise ValueError(f"item part is not full depth: {part_dir}")
        if set(risk["cell_budget_label"].astype(str).unique()) != {FULL_LABEL}:
            raise ValueError(f"risk part is not full depth: {part_dir}")
        expected_labels = set(all_labels[start:stop])
        if set(item["perturbation_label"].astype(str).unique()) != expected_labels:
            raise ValueError(f"item labels do not match declared range in {part_dir}")
        if set(risk["perturbation_label"].astype(str).unique()) != expected_labels:
            raise ValueError(f"risk labels do not match declared range in {part_dir}")
        if set(item["split_seed"].astype(int).unique()) != set(SPLIT_SEEDS):
            raise ValueError(f"item seed schedule is incomplete in {part_dir}")
        if set(risk["split_seed"].astype(int).unique()) != set(SPLIT_SEEDS):
            raise ValueError(f"risk seed schedule is incomplete in {part_dir}")
        ranges.append((start, stop, part_dir.name))
        item_frames.append(item)
        risk_frames.append(risk)
    ranges.sort()
    expected_start = 0
    for start, stop, name in ranges:
        if start != expected_start:
            raise ValueError(f"label range gap or overlap before {name}: expected {expected_start}, got {start}")
        expected_start = stop
    if expected_start != len(all_labels):
        raise ValueError(f"label range coverage stops at {expected_start}, expected {len(all_labels)}")
    items = pd.concat(item_frames, ignore_index=True)
    risks = pd.concat(risk_frames, ignore_index=True)
    item_key = ["source_environment_id", "left_target_environment_id", "right_target_environment_id", "metric", "split_seed", "perturbation_label", "cell_budget_label"]
    risk_key = item_key[:-1] + ["measurement_replicate", "cell_budget_label"]
    if items.duplicated(item_key).any():
        raise ValueError("duplicate full-depth item keys across label parts")
    if risks.duplicated(risk_key).any():
        raise ValueError("duplicate full-depth risk keys across label parts")
    label_order = {label: index for index, label in enumerate(all_labels)}
    items["_label_order"] = items["perturbation_label"].map(label_order)
    items = items.sort_values(["cell_budget_order", "source_environment_id", "left_target_environment_id", "right_target_environment_id", "metric", "split_seed", "_label_order"]).drop(columns="_label_order").reset_index(drop=True)
    risks["_label_order"] = risks["perturbation_label"].map(label_order)
    risks = risks.sort_values(["split_seed", "source_environment_id", "left_target_environment_id", "right_target_environment_id", "metric", "measurement_replicate", "_label_order"]).reset_index(drop=True)

    decision_rows: list[dict[str, object]] = []
    member_columns = sorted(
        [column for column in risks.columns if column.startswith("left_member_")],
        key=lambda column: int(column.rsplit("_", 1)[1]),
    )
    right_member_columns = sorted(
        [column for column in risks.columns if column.startswith("right_member_")],
        key=lambda column: int(column.rsplit("_", 1)[1]),
    )
    if not member_columns or len(member_columns) != len(right_member_columns):
        raise ValueError("full-depth risk member columns are incomplete")
    for (source, left, right, metric, seed), group in risks.groupby(
        ["source_environment_id", "left_target_environment_id", "right_target_environment_id", "metric", "split_seed"], sort=False
    ):
        group = group.sort_values(["measurement_replicate", "_label_order"])
        replicate_values = sorted(group["measurement_replicate"].astype(int).unique())
        label_values = [label for label, _ in sorted(label_order.items(), key=lambda item: item[1]) if label in set(group["perturbation_label"].astype(str))]
        if len(group) != len(replicate_values) * len(label_values) or len(replicate_values) < 2:
            raise ValueError(f"incomplete full-depth risk replicate grid for {(source, left, right, metric, seed)}")
        left_mean = np.stack([
            group[group["measurement_replicate"].eq(replicate)]["left_risk"].to_numpy(dtype=float)
            for replicate in replicate_values
        ])
        right_mean = np.stack([
            group[group["measurement_replicate"].eq(replicate)]["right_risk"].to_numpy(dtype=float)
            for replicate in replicate_values
        ])
        left_members = np.stack([
            group[group["measurement_replicate"].eq(replicate)][member_columns].to_numpy(dtype=float).T
            for replicate in replicate_values
        ])
        right_members = np.stack([
            group[group["measurement_replicate"].eq(replicate)][right_member_columns].to_numpy(dtype=float).T
            for replicate in replicate_values
        ])
        _append_decision_rows(
            decision_rows,
            source=str(source),
            left=str(left),
            right=str(right),
            metric=str(metric),
            split_seed=int(seed),
            cell_budget=10**9,
            cell_budget_label=FULL_LABEL,
            left_risk=left_mean,
            right_risk=right_mean,
            left_members=left_members,
            right_members=right_members,
        )
    decisions = pd.DataFrame(decision_rows)
    risks = risks.drop(columns="_label_order")
    coverage_rows: list[dict[str, object]] = []
    for left, right in PAIR_ORDER:
        eligible = _pair_label_set(groups, all_labels, left, right, 10**9, full=True)
        coverage_rows.append({
            "left_target_environment_id": left,
            "right_target_environment_id": right,
            "cell_budget": 10**9,
            "cell_budget_label": FULL_LABEL,
            "cell_budget_order": FULL_ORDER,
            "n_labels_all": len(all_labels),
            "n_labels_matched": len(eligible),
            "coverage_fraction": float(len(eligible) / len(all_labels)) if all_labels else float("nan"),
            "matched_universe_sha256": hashlib.sha256("\n".join(eligible).encode()).hexdigest(),
            "matched_universe_policy": "all labels with nonempty matched target strata; each label uses its own full matched cell count with replacement",
        })
    coverage = pd.DataFrame(coverage_rows)
    summary = summarize_depth_rows(items, draws=draws, ci_level=0.90)
    resolution = depth_resolution_table(summary)
    full_dir.mkdir(parents=True, exist_ok=True)
    items.to_csv(full_dir / "reliability_transport_measurement_depth_items.csv", index=False)
    risks.to_csv(full_dir / "reliability_transport_measurement_depth_risks.csv", index=False)
    decisions.to_csv(full_dir / "reliability_transport_measurement_depth_decision.csv", index=False)
    coverage.to_csv(full_dir / "reliability_transport_measurement_depth_coverage.csv", index=False)
    summary.to_csv(full_dir / "reliability_transport_measurement_depth_summary.csv", index=False)
    resolution.to_csv(full_dir / "reliability_transport_measurement_depth_resolution.csv", index=False)
    report = {
        "schema_version": 1,
        "status": "executed",
        "analysis": "measurement_depth_identifiability_full_label_shards",
        "split_seeds": list(SPLIT_SEEDS),
        "n_labels": len(all_labels),
        "label_parts": len(part_dirs),
        "full_depth_label": FULL_LABEL,
        "decision_budgets": [0.05, 0.10, 0.20, 0.50],
        "bootstrap_draws": int(draws),
        "outputs": {
            "items": "artifacts/manifests/reliability_transport_measurement_depth_parts/full/reliability_transport_measurement_depth_items.csv",
            "risks": "artifacts/manifests/reliability_transport_measurement_depth_parts/full/reliability_transport_measurement_depth_risks.csv",
            "decision": "artifacts/manifests/reliability_transport_measurement_depth_parts/full/reliability_transport_measurement_depth_decision.csv",
            "coverage": "artifacts/manifests/reliability_transport_measurement_depth_parts/full/reliability_transport_measurement_depth_coverage.csv",
            "summary": "artifacts/manifests/reliability_transport_measurement_depth_parts/full/reliability_transport_measurement_depth_summary.csv",
            "resolution": "artifacts/manifests/reliability_transport_measurement_depth_parts/full/reliability_transport_measurement_depth_resolution.csv",
        },
        "checks": {
            "item_rows": int(len(items)),
            "risk_rows": int(len(risks)),
            "decision_rows": int(len(decisions)),
            "coverage_rows": int(len(coverage)),
            "summary_rows": int(len(summary)),
            "resolution_rows": int(len(resolution)),
            "item_key_sha256": _key_fingerprint(items, item_key),
            "risk_key_sha256": _key_fingerprint(risks, risk_key),
            "no_prediction_refit": True,
            "target_outcomes_not_used": True,
        },
    }
    (full_dir / "reliability_transport_measurement_depth.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return {"status": "executed", **report["checks"]}


if __name__ == "__main__":
    merge()
