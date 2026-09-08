"""Evaluate fixed-budget utility from the existing held-out predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("raw_normalized_uq", "u_only_rf", "ptl_rf")
BUDGETS = (0.20, 0.50, 0.80)


def run(root: Path, output_path: Path, summary_path: Path) -> dict[str, Any]:
    frame = pd.read_csv(root / "artifacts/source_data/formal_v2_reliability_predictions.csv")
    frame = frame.loc[frame["scenario"].eq("leave_environment_out")].copy()
    if frame.empty:
        raise ValueError("deployment utility requires leave-environment-out held-out rows")
    rows: list[dict[str, Any]] = []
    for (environment_id, predictor), group in frame.groupby(["environment_id", "predictor"], sort=True):
        for method in METHODS:
            ordered = group.sort_values([method, "biological_instance_id"], ascending=[False, True], kind="stable")
            for budget in BUDGETS:
                n = max(1, int(np.ceil(len(ordered) * budget)))
                selected = ordered.iloc[:n]
                rows.append({
                    "aggregation_level": "environment_predictor",
                    "split_seed": int(group["split_seed"].iloc[0]) if "split_seed" in group.columns else 20260907,
                    "environment_id": environment_id,
                    "predictor": predictor,
                    "method": method,
                    "budget": budget,
                    "n_selected": int(n),
                    "realized_fidelity": float(1.0 - selected["continuous_risk"].mean()),
                    "false_trust_rate": float(1.0 - selected["reliable_label"].mean()),
                    "reliable_perturbation_recovery": float(selected["reliable_label"].mean()),
                    "high_effect_fidelity_fraction": float((selected["fidelity"] >= 0.8).mean()),
                    "budget_specific_retraining": 0,
                })
    result_frame = pd.DataFrame(rows)
    macro_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(20260907)
    for (method, budget), group in result_frame.groupby(["method", "budget"], sort=True):
        environment_values = group.groupby("environment_id", sort=True)["realized_fidelity"].mean().to_numpy(dtype=float)
        false_trust_values = group.groupby("environment_id", sort=True)["false_trust_rate"].mean().to_numpy(dtype=float)
        if len(environment_values) == 0:
            continue
        fidelity_bootstrap = np.asarray([
            rng.choice(environment_values, size=len(environment_values), replace=True).mean()
            for _ in range(2000)
        ])
        false_trust_bootstrap = np.asarray([
            rng.choice(false_trust_values, size=len(false_trust_values), replace=True).mean()
            for _ in range(2000)
        ])
        macro_rows.append({
            "aggregation_level": "environment_macro",
            "split_seed": 20260907,
            "environment_id": "environment_macro",
            "predictor": "macro",
            "method": method,
            "budget": budget,
            "n_selected": int(round(group["n_selected"].mean())),
            "realized_fidelity": float(environment_values.mean()),
            "realized_fidelity_ci_lower": float(np.quantile(fidelity_bootstrap, 0.025)),
            "realized_fidelity_ci_upper": float(np.quantile(fidelity_bootstrap, 0.975)),
            "false_trust_rate": float(false_trust_values.mean()),
            "false_trust_rate_ci_lower": float(np.quantile(false_trust_bootstrap, 0.025)),
            "false_trust_rate_ci_upper": float(np.quantile(false_trust_bootstrap, 0.975)),
            "reliable_perturbation_recovery": float(group["reliable_perturbation_recovery"].mean()),
            "high_effect_fidelity_fraction": float(group["high_effect_fidelity_fraction"].mean()),
            "budget_specific_retraining": 0,
            "environment_count": int(len(environment_values)),
            "uncertainty": "2000 environment-cluster bootstrap replicates",
        })
    result_frame = pd.concat([result_frame, pd.DataFrame(macro_rows)], ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_frame.to_csv(output_path, index=False)
    summary = {
        "schema_version": 1,
        "benchmark_id": "ptl_context_v2",
        "budgets": list(BUDGETS),
        "methods": list(METHODS),
        "scenario": "leave_environment_out",
        "prediction_source": "formal_v2_reliability_predictions.csv",
        "budget_specific_retraining": False,
        "output_path": output_path.relative_to(root).as_posix(),
        "macro_rows": macro_rows,
        "macro_uncertainty": "environment-cluster bootstrap, 2000 replicates",
        "excluded_methods": ["transported_source_gate", "safeguarded_ptl"],
        "status": "fixed_budget_utility_executed_real_predictions",
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    result = run(
        root,
        (args.output or root / "artifacts/manifests/formal_v2_deployment_utility.csv").resolve(),
        (args.summary or root / "artifacts/manifests/formal_v2_deployment_utility.json").resolve(),
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
