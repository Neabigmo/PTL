"""Canonicalize the executed decision bootstrap into a small auditable table.

The full bootstrap is the inferential source for decision intervals.  The
frozen decision table remains the source for point estimates and its 30-seed
quantiles are retained only as descriptive seed-variability fields.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_NAME = "reliability_transport_measurement_depth_matched_fixed_decision_macro_bootstrap.csv"
POINT_NAME = "reliability_transport_measurement_depth_matched_fixed_decision.csv"
SUMMARY_NAME = "reliability_transport_measurement_depth_matched_fixed_decision_macro_bootstrap_summary.csv"
REPORT_NAME = "reliability_transport_measurement_depth_decision_macro_bootstrap.json"
SUMMARY_REPORT_NAME = "reliability_transport_measurement_depth_decision_macro_bootstrap_summary.json"
KEY_COLUMNS = [
    "source_environment_id",
    "target_environment_id",
    "left_target_environment_id",
    "right_target_environment_id",
    "metric",
    "cell_budget_label",
    "decision_budget_fraction",
]
BOOTSTRAP_COLUMNS = [
    "retention",
    "regret",
    "normalized_regret",
    "boundary_inversion",
    "measurement_floor_regret",
    "excess_regret",
]
POINT_ONLY_COLUMNS = ["joint_floor_regret", "joint_excess_regret"]
EXPECTED_BUDGETS = (0.05, 0.10, 0.20, 0.50)
EXPECTED_DRAWS = 2000
EXPECTED_SEEDS = 30


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_revision(root: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    revision = result.stdout.strip()
    return revision or None


def _normalise_keys(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["cell_budget_label"] = frame["cell_budget_label"].astype("string")
    frame["decision_budget_fraction"] = pd.to_numeric(
        frame["decision_budget_fraction"], errors="raise"
    ).round(8)
    return frame


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def _finite(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    values = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        bad = np.argwhere(~np.isfinite(values))[0].tolist()
        raise ValueError(f"{name} contains a non-finite value at row/column {bad}")


def _point_fields(point: pd.DataFrame, group: pd.DataFrame) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for column in BOOTSTRAP_COLUMNS + POINT_ONLY_COLUMNS:
        values[f"{column}_point"] = float(point[column].mean())
    for column in BOOTSTRAP_COLUMNS:
        values[f"{column}_seed_q05"] = float(point[column].quantile(0.05))
        values[f"{column}_seed_q95"] = float(point[column].quantile(0.95))
    values["seed_count"] = int(point["split_seed"].nunique())
    values["point_row_count"] = int(len(point))
    return values


def _bootstrap_fields(group: pd.DataFrame) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for column in BOOTSTRAP_COLUMNS:
        numeric = pd.to_numeric(group[column], errors="raise")
        values[f"{column}_ci_low"] = float(numeric.quantile(0.05))
        values[f"{column}_ci_high"] = float(numeric.quantile(0.95))
    values["draw_count"] = int(group["bootstrap_draw"].nunique())
    values["valid_draw_count"] = int(len(group))
    return values


def build(root: Path = ROOT) -> tuple[pd.DataFrame, dict[str, Any]]:
    root = root.resolve()
    manifests = root / "artifacts/manifests"
    raw_path = manifests / RAW_NAME
    point_path = manifests / POINT_NAME
    bootstrap_report_path = manifests / REPORT_NAME
    summary_path = manifests / SUMMARY_NAME
    summary_report_path = manifests / SUMMARY_REPORT_NAME
    if not raw_path.is_file():
        raise FileNotFoundError(raw_path)
    if not point_path.is_file():
        raise FileNotFoundError(point_path)

    raw = _normalise_keys(pd.read_csv(raw_path, dtype={"cell_budget_label": "string"}, low_memory=False))
    point = _normalise_keys(pd.read_csv(point_path, dtype={"cell_budget_label": "string"}, low_memory=False))
    raw_sha256 = _sha256(raw_path)
    _require_columns(raw, set(KEY_COLUMNS + ["cell_budget", "cell_budget_order", "bootstrap_draw", "bootstrap_unit", "status"] + BOOTSTRAP_COLUMNS), "raw bootstrap")
    _require_columns(point, set(KEY_COLUMNS + ["split_seed", "cell_budget", "cell_budget_order"] + BOOTSTRAP_COLUMNS + POINT_ONLY_COLUMNS), "point decision table")
    _finite(raw, BOOTSTRAP_COLUMNS, "raw bootstrap")
    _finite(point, BOOTSTRAP_COLUMNS + POINT_ONLY_COLUMNS, "point decision table")

    raw_budget_values = tuple(sorted(float(value) for value in raw["decision_budget_fraction"].unique()))
    if raw_budget_values != EXPECTED_BUDGETS:
        raise ValueError(f"raw bootstrap budgets differ from frozen contract: {raw_budget_values}")
    if set(raw["status"].astype(str).unique()) != {"executed"}:
        raise ValueError("raw bootstrap contains a non-executed status")
    if set(raw["bootstrap_unit"].astype(str).unique()) != {"perturbation_label_then_measurement_seed"}:
        raise ValueError("raw bootstrap unit is not the declared perturbation-label then seed unit")

    raw_group_columns = KEY_COLUMNS[:-1]
    raw_group_count = int(raw[raw_group_columns].drop_duplicates().shape[0])
    if raw_group_count != 108:
        raise ValueError(f"expected 108 valid directed groups, observed {raw_group_count}")
    raw_key_counts = raw.groupby(KEY_COLUMNS, sort=True, observed=True).size()
    if not (raw_key_counts == EXPECTED_DRAWS).all():
        raise ValueError("every group-budget key must contain exactly 2,000 draws")
    if raw.duplicated(KEY_COLUMNS + ["bootstrap_draw"]).any():
        raise ValueError("duplicate raw bootstrap group-budget-draw keys")
    if set(raw["bootstrap_draw"].astype(int).unique()) != set(range(EXPECTED_DRAWS)):
        raise ValueError("raw bootstrap draw ids are not exactly 0..1999")

    point_groups = point.groupby(KEY_COLUMNS, sort=True, observed=True)
    point_key_counts = point_groups.size()
    if set(point_key_counts.to_numpy(dtype=int)) != {EXPECTED_SEEDS}:
        raise ValueError("each point-estimate group-budget key must contain exactly 30 seeds")
    raw_keys = set(raw_key_counts.index.tolist())
    point_keys = set(point_key_counts.index.tolist())
    if raw_keys != point_keys:
        raise ValueError(f"raw/point key mismatch: raw_only={len(raw_keys - point_keys)}, point_only={len(point_keys - raw_keys)}")

    point_group_map = {key: group for key, group in point_groups}
    rows: list[dict[str, Any]] = []
    for key, group in raw.groupby(KEY_COLUMNS, sort=True, observed=True):
        point_group = point_group_map[key]
        record = dict(zip(KEY_COLUMNS, key))
        record.update({
            "cell_budget": group["cell_budget"].iloc[0],
            "cell_budget_order": int(group["cell_budget_order"].iloc[0]),
            "universe_mode": "matched_fixed",
        })
        record.update(_point_fields(point_group, group))
        record.update(_bootstrap_fields(group))
        record.update({
            "status": "executed",
            "bootstrap_unit": "perturbation_label_then_measurement_seed",
            "raw_bootstrap_sha256": raw_sha256,
            "point_estimator_source": "reliability_transport_measurement_depth_matched_fixed_decision.csv; mean across frozen 30 split seeds",
            "point_estimator_definition": "frozen decision-table seed mean; not the bootstrap-draw mean",
            "inferential_interval_source": "864000-row perturbation-label then measurement-seed bootstrap; q05/q95 over 2000 draws per group-budget key",
        })
        rows.append(record)

    summary = pd.DataFrame(rows)
    summary = summary.sort_values(KEY_COLUMNS, kind="stable").reset_index(drop=True)
    if len(summary) != 432 or summary.duplicated(KEY_COLUMNS).any():
        raise ValueError(f"summary must contain 432 unique group-budget rows, observed {len(summary)}")
    summary.to_csv(summary_path, index=False)

    bootstrap_report = json.loads(bootstrap_report_path.read_text(encoding="utf-8")) if bootstrap_report_path.is_file() else {}
    risk_rel = bootstrap_report.get("input", "artifacts/manifests/reliability_transport_measurement_depth_matched_fixed_risks.csv")
    risk_path = root / risk_rel
    if not risk_path.is_file():
        raise FileNotFoundError(risk_path)
    summary_report: dict[str, Any] = {
        "schema_version": 1,
        "status": "executed",
        "raw_bootstrap": raw_path.relative_to(root).as_posix(),
        "raw_bootstrap_sha256": raw_sha256,
        "raw_bootstrap_report": bootstrap_report_path.relative_to(root).as_posix(),
        "raw_bootstrap_report_sha256": _sha256(bootstrap_report_path) if bootstrap_report_path.is_file() else None,
        "risk_input": risk_path.relative_to(root).as_posix(),
        "risk_input_sha256": _sha256(risk_path),
        "point_estimate_source": point_path.relative_to(root).as_posix(),
        "point_estimate_source_sha256": _sha256(point_path),
        "summary": summary_path.relative_to(root).as_posix(),
        "summary_sha256": _sha256(summary_path),
        "raw_rows": int(len(raw)),
        "valid_directed_groups": raw_group_count,
        "draws_per_group": EXPECTED_DRAWS,
        "summary_rows": int(len(summary)),
        "summary_keys": int(summary[KEY_COLUMNS].drop_duplicates().shape[0]),
        "seed_count_per_point_key": EXPECTED_SEEDS,
        "budgets": list(EXPECTED_BUDGETS),
        "bootstrap_unit": "perturbation_label_then_measurement_seed",
        "bootstrap_unit_definition": "each draw resamples perturbation labels and measurement seeds while retaining the frozen source/target direction and target self-floor contract",
        "point_estimator_definition": "frozen decision-table mean across 30 split seeds; bootstrap-draw means are not substituted",
        "interval_definition": "q05/q95 over the 2,000 executed draws for each of 432 group-budget keys",
        "execution_host": bootstrap_report.get("execution_host", platform.node()),
        "bootstrap_generation_script": "scripts/build_decision_macro_bootstrap.py",
        "bootstrap_generation_script_sha256": _sha256(root / "scripts/build_decision_macro_bootstrap.py"),
        "bootstrap_generation_commit": bootstrap_report.get("commit") or _git_revision(root),
        "summary_generation_script": Path(__file__).resolve().relative_to(root).as_posix(),
        "summary_generation_script_sha256": _sha256(Path(__file__).resolve()),
        "source_commit_at_summary_generation": _git_revision(root),
        "validation": {
            "raw_unique_group_budget_draw_keys": int(raw[KEY_COLUMNS + ["bootstrap_draw"]].drop_duplicates().shape[0]),
            "raw_duplicate_group_budget_draw_keys": int(raw.duplicated(KEY_COLUMNS + ["bootstrap_draw"]).sum()),
            "raw_all_finite": True,
            "raw_all_status_executed": True,
            "summary_duplicate_group_budget_keys": int(summary.duplicated(KEY_COLUMNS).sum()),
            "summary_all_finite": True,
            "raw_group_count": raw_group_count,
            "raw_rows_expected": 108 * EXPECTED_DRAWS * len(EXPECTED_BUDGETS),
            "summary_rows_expected": 108 * len(EXPECTED_BUDGETS),
        },
    }
    summary_report_path.write_text(json.dumps(summary_report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary, summary_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    summary, report = build(args.root)
    print(json.dumps({"status": report["status"], "summary_rows": len(summary), "summary": report["summary"], "summary_sha256": report["summary_sha256"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
