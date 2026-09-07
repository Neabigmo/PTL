"""Materialize the formal-v2 roster status without claiming missing results."""

from __future__ import annotations

import argparse
import json
import importlib.util
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]


def dependency_status(predictor_id: str) -> tuple[str, str]:
    if predictor_id in {"mean_matching", "strong_linear"}:
        return "success", "formal_v2_predictor_metrics_available"
    if predictor_id == "official_gears":
        if importlib.util.find_spec("gears") is None:
            return "unsupported", "missing_importable_gears_dependency"
        return "pending", "formal_raw_adata_adapter_not_yet_executed"
    if predictor_id in {"cpa", "scgpt"}:
        package = predictor_id
        if importlib.util.find_spec(package) is None:
            return "unsupported", f"missing_optional_dependency:{package}"
        return "pending", "formal_adapter_not_yet_executed"
    if predictor_id == "prescribe":
        if importlib.util.find_spec("prescribe") is None:
            return "unsupported", "optional_predictor_not_installed_compatible_overlap_only"
        return "pending", "compatible_overlap_only_formal_run_not_yet_executed"
    return "pending", "unclassified_roster_entry"


def run(root: Path, output: Path) -> dict[str, Any]:
    roster = yaml.safe_load((root / "configs/model_roster.yaml").read_text(encoding="utf-8"))
    registry = pd.read_csv(root / "artifacts/manifests/environment_registry.csv")
    rows: list[dict[str, Any]] = []
    for predictor in roster["predictors"]:
        status, reason = dependency_status(str(predictor["predictor_id"]))
        for _, environment in registry.iterrows():
            rows.append({
                "predictor_id": predictor["predictor_id"],
                "predictor_family": predictor["family"],
                "environment_id": environment["environment_id"],
                "environment_key": environment["environment_key"],
                "status": status,
                "reason": reason,
                "split_id": roster["uq_policy"]["split_id"],
                "split_seed": roster["uq_policy"]["split_seed"],
                "model_seeds": ",".join(str(value) for value in roster["uq_policy"]["model_seeds"]),
                "scientific_result_claimed": status == "success" and predictor["predictor_id"] in {"mean_matching", "strong_linear"},
            })
    frame = pd.DataFrame(rows).sort_values(["predictor_id", "environment_id"], kind="stable")
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    summary = {
        "schema_version": 1,
        "roster_id": roster["roster_id"],
        "split_id": roster["uq_policy"]["split_id"],
        "rows": int(len(frame)),
        "status_counts": {str(key): int(value) for key, value in frame["status"].value_counts().items()},
        "result_claims": int(frame["scientific_result_claimed"].sum()),
        "status": "formal_v2_roster_status_audited",
    }
    summary_path = output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/manifests/formal_v2_model_status.csv")
    args = parser.parse_args()
    print(json.dumps(run(args.root.resolve(), args.output.resolve()), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
