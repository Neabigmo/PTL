"""Run the canonical reliability evaluation on every declared split seed.

Each split is written below its own ``results/formal_v2/multisplit/<seed>``
directory.  The script aggregates split-level and environment-level deltas
only after the isolated evaluations exist; it never pools split observations as
independent biological domains.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_formal_v2_reliability import BOOTSTRAP_REPLICATES, run  # noqa: E402


PRIMARY_SEED = 20260907
SCENARIOS = ("in_domain", "leave_environment_out", "leave_predictor_out")


def split_seeds(config_path: Path) -> list[int]:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    seeds = [int(payload["split_seed"]), *(int(value) for value in payload.get("stability_seeds", []))]
    if seeds[0] != PRIMARY_SEED or len(seeds) != len(set(seeds)):
        raise ValueError("the reliability split set must keep primary seed 20260907 and unique seeds")
    return seeds


def paths_for_seed(root: Path, seed: int) -> tuple[Path, Path, Path]:
    if seed == PRIMARY_SEED:
        return (
            root / "artifacts/manifests/biological_instance_registry.csv",
            root / "results/formal_v2/predictors",
            root,
        )
    job_dir = root / "results/formal_v2/multisplit" / str(seed)
    return (
        root / f"artifacts/manifests/biological_instance_registry__split_{seed}.csv",
        job_dir / "predictors",
        job_dir / "reliability",
    )


def aggregate(root: Path, seeds: list[int], output_dir: Path) -> dict[str, Any]:
    split_rows: list[dict[str, Any]] = []
    environment_rows: list[pd.DataFrame] = []
    for seed in seeds:
        _, _, artifact_root = paths_for_seed(root, seed)
        metric_path = artifact_root / "artifacts/manifests/formal_v2_reliability_metrics.csv"
        if not metric_path.is_file():
            raise FileNotFoundError(f"missing split reliability metrics for {seed}: {metric_path}")
        metrics = pd.read_csv(metric_path)
        macro = metrics.loc[metrics["aggregation_level"].eq("environment_macro")].copy()
        pivot = macro.pivot(index="scenario", columns="method", values="aurc")
        required = {"u_only_rf", "ptl_rf"}
        if not required.issubset(pivot.columns):
            raise ValueError(f"split {seed} has no U-only/PTL macro metrics")
        for scenario in SCENARIOS:
            if scenario not in pivot.index:
                raise ValueError(f"split {seed} is missing scenario {scenario}")
            u_only = float(pivot.loc[scenario, "u_only_rf"])
            ptl = float(pivot.loc[scenario, "ptl_rf"])
            delta = ptl - u_only
            split_rows.append({
                "split_seed": seed,
                "scenario": scenario,
                "u_only_aurc": u_only,
                "ptl_aurc": ptl,
                "delta_aurc_ptl_minus_u_only": delta,
                "improved": int(delta < 0.0),
            })
        env = metrics.loc[
            metrics["aggregation_level"].eq("environment_predictor_test")
            & metrics["method"].isin(["u_only_rf", "ptl_rf"])
        ].copy()
        env_pivot = env.pivot_table(
            index=["scenario", "environment_id"], columns="method", values="aurc", aggfunc="mean"
        ).reset_index()
        env_pivot["split_seed"] = seed
        env_pivot["delta_aurc_ptl_minus_u_only"] = env_pivot["ptl_rf"] - env_pivot["u_only_rf"]
        environment_rows.append(env_pivot)

    split_frame = pd.DataFrame(split_rows).sort_values(["scenario", "split_seed"], kind="stable")
    environment_frame = pd.concat(environment_rows, ignore_index=True).sort_values(
        ["scenario", "split_seed", "environment_id"], kind="stable"
    )
    summary_rows: list[dict[str, Any]] = []
    for scenario, group in split_frame.groupby("scenario", sort=True):
        values = group["delta_aurc_ptl_minus_u_only"].to_numpy(dtype=float)
        summary_rows.append({
            "scenario": scenario,
            "n_splits": int(len(values)),
            "mean_delta_aurc_ptl_minus_u_only": float(values.mean()),
            "median_delta_aurc_ptl_minus_u_only": float(np.median(values)),
            "sd_delta_aurc_ptl_minus_u_only": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "sign_agreement_improvement": float(np.mean(values < 0.0)),
            "n_splits_improved": int(np.sum(values < 0.0)),
            "split_level_interval_lower": float(np.min(values)),
            "split_level_interval_upper": float(np.max(values)),
            "environment_level_mean_delta": float(
                environment_frame.loc[environment_frame["scenario"].eq(scenario), "delta_aurc_ptl_minus_u_only"].mean()
            ),
        })
    summary_frame = pd.DataFrame(summary_rows).sort_values("scenario", kind="stable")
    paired = split_frame.pivot(index="split_seed", columns="scenario", values="delta_aurc_ptl_minus_u_only")
    if not {"leave_environment_out", "leave_predictor_out"}.issubset(paired.columns):
        raise ValueError("asymmetry contrast requires both leave-out scenarios")
    paired["delta_env_minus_predictor"] = paired["leave_environment_out"] - paired["leave_predictor_out"]
    contrast = paired["delta_env_minus_predictor"].dropna().to_numpy(dtype=float)
    asymmetry = {
        "contrast": "delta_leave_environment_out_minus_delta_leave_predictor_out",
        "n_splits": int(len(contrast)),
        "mean": float(contrast.mean()),
        "median": float(np.median(contrast)),
        "sd": float(contrast.std(ddof=1)) if len(contrast) > 1 else 0.0,
        "split_level_interval_lower": float(np.min(contrast)),
        "split_level_interval_upper": float(np.max(contrast)),
        "n_negative": int(np.sum(contrast < 0.0)),
        "n_positive": int(np.sum(contrast > 0.0)),
        "interpretation": "negative means environment holdout has a more negative PTL-minus-U-only delta; this contrast is descriptive and not a causal portability claim",
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    split_path = output_dir / "formal_v2_multisplit_reliability.csv"
    summary_path = output_dir / "formal_v2_multisplit_reliability_summary.json"
    environment_path = output_dir / "formal_v2_multisplit_reliability_environment.csv"
    split_frame.to_csv(split_path, index=False)
    environment_frame.to_csv(environment_path, index=False)
    payload = {
        "schema_version": 1,
        "benchmark_id": "ptl_context_v2",
        "split_id": "ptl_biological_instance_split_v1",
        "primary_split_seed": PRIMARY_SEED,
        "split_seeds": seeds,
        "scenarios": list(SCENARIOS),
        "baseline": "u_only_rf",
        "method": "ptl_rf",
        "aggregation": "split_level_summary_with_environment_level_distribution",
        "no_independent_domain_claim": True,
        "split_path": split_path.relative_to(root).as_posix(),
        "environment_path": environment_path.relative_to(root).as_posix(),
        "rows": summary_frame.to_dict("records"),
        "paired_asymmetry_contrast": asymmetry,
        "status": "formal_v2_multisplit_reliability_executed",
    }
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--split-config", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    args = parser.parse_args()
    root = args.root.resolve()
    config = (args.split_config or root / "configs/biological_instance_split.yaml").resolve()
    seeds = split_seeds(config)
    run_records: list[dict[str, Any]] = []
    for seed in seeds:
        manifest, predictor_dir, output_dir = paths_for_seed(root, seed)
        summary_path = output_dir / "artifacts/manifests/formal_v2_reliability_summary.json"
        calibration_path = output_dir / "artifacts/source_data/formal_v2_reliability_calibration_scores.csv"
        if args.resume and summary_path.is_file() and calibration_path.is_file():
            run_records.append({"split_seed": seed, "status": "resumed", "summary": summary_path.relative_to(root).as_posix()})
            continue
        result = run(
            root,
            manifest_path=manifest,
            predictor_dir=predictor_dir,
            split_seed=seed,
            output_dir=output_dir,
            bootstrap_replicates=args.bootstrap_replicates,
        )
        run_records.append({"split_seed": seed, "status": result["status"], "summary": summary_path.relative_to(root).as_posix()})
    aggregate_payload = aggregate(root, seeds, root / "artifacts/manifests")
    aggregate_payload["split_runs"] = run_records
    (root / "artifacts/manifests/formal_v2_multisplit_reliability_summary.json").write_text(
        json.dumps(aggregate_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(aggregate_payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
