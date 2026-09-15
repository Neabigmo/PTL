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


def dependency_status(predictor_id: str, root: Path | None = None) -> tuple[str, str, str]:
    """Return scientific support, runtime status, and an evidence reason.

    Missing Python packages are runtime blockers, not evidence that a predictor
    is scientifically incompatible with the benchmark. Keeping those axes
    separate prevents a dependency gap from becoming an unsupported-method
    claim in the roster.
    """

    if predictor_id in {"mean_matching", "strong_linear", "slim_string"}:
        metrics_path = (root or ROOT) / "artifacts/manifests/formal_v2_predictor_metrics.csv"
        if metrics_path.exists():
            metrics = pd.read_csv(metrics_path, usecols=["predictor"])
            if metrics["predictor"].eq(predictor_id).any():
                if predictor_id == "strong_linear":
                    summary_path = metrics_path.with_name("formal_v2_predictor_summary.json")
                    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
                    if summary.get("predictor_version") != "formal_v2_exact_linear_20260907":
                        return "supported", "ready", "exact_linear_run_pending_prior_approximation_present"
                if predictor_id == "slim_string":
                    summary_path = metrics_path.with_name("formal_v2_predictor_summary.json")
                    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
                    versions = summary.get("predictor_versions", {})
                    if versions.get("slim_string") != "formal_v2_slim_string_v0.3_20260907":
                        return "supported", "ready", "slim_embedding_run_pending"
                return "supported", "success", "formal_v2_predictor_metrics_available"
        return "supported", "ready", "formal_v2_predictor_run_not_materialized"
    if predictor_id == "official_gears":
        if importlib.util.find_spec("gears") is None:
            return "supported", "dependency_missing", "missing_importable_gears_dependency"
        return "supported", "ready", "formal_raw_adata_adapter_not_yet_executed"
    if predictor_id in {"cpa", "scgpt"}:
        package = predictor_id
        if importlib.util.find_spec(package) is None:
            return "supported", "dependency_missing", f"missing_optional_dependency:{package}"
        return "supported", "ready", "formal_adapter_not_yet_executed"
    if predictor_id == "prescribe":
        if importlib.util.find_spec("prescribe") is None:
            return "incompatible", "dependency_missing", "optional_non_blocking_compatible_overlap_only"
        return "incompatible", "ready", "compatible_overlap_only_formal_run_not_yet_executed"
    return "unknown", "ready", "unclassified_roster_entry"


def run(root: Path, output: Path) -> dict[str, Any]:
    roster = yaml.safe_load((root / "configs/model_roster.yaml").read_text(encoding="utf-8"))
    registry = pd.read_csv(root / "artifacts/manifests/environment_registry.csv")
    rows: list[dict[str, Any]] = []
    for predictor in roster["predictors"]:
        scientific_support, runtime_status, reason = dependency_status(
            str(predictor["predictor_id"]), root
        )
        for _, environment in registry.iterrows():
            rows.append({
                "predictor_id": predictor["predictor_id"],
                "predictor_family": predictor["family"],
                "environment_id": environment["environment_id"],
                "environment_key": environment["environment_key"],
                "scientific_support": scientific_support,
                "runtime_status": runtime_status,
                "status": runtime_status,
                "reason": reason,
                "split_id": roster["uq_policy"]["split_id"],
                "split_seed": roster["uq_policy"]["split_seed"],
                "model_seeds": ",".join(str(value) for value in roster["uq_policy"]["model_seeds"]),
                "scientific_result_claimed": runtime_status == "success" and scientific_support == "supported",
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
