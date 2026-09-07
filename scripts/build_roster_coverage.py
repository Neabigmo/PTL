"""Build the explicit predictor-by-environment coverage planning matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return value


def build_matrix(root: Path, benchmark_path: Path, roster_path: Path, output_path: Path, summary_path: Path) -> dict[str, Any]:
    benchmark = load_yaml(benchmark_path)
    roster = load_yaml(roster_path)
    environments = benchmark.get("environments", [])
    predictors = roster.get("predictors", [])
    if len(environments) != 8:
        raise ValueError("Coverage matrix requires the frozen eight-environment benchmark")
    if not predictors:
        raise ValueError("Predictor roster is empty")
    rows: list[dict[str, Any]] = []
    for predictor in predictors:
        for environment in environments:
            rows.append(
                {
                    "predictor_id": predictor["predictor_id"],
                    "predictor_family": predictor["family"],
                    "roster_role": predictor["role"],
                    "predictor_status": predictor["status"],
                    "environment_id": environment["environment_id"],
                    "dataset_id": environment["dataset_id"],
                    "cell_context": environment["cell_context"],
                    "condition": environment["condition"],
                    "environment_status": environment["status"],
                    "coverage_policy": predictor["supported_environment_policy"],
                    "uncertainty_source": predictor["uncertainty_source"],
                    "execution_status": "planned_formal_run",
                }
            )
    frame = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    summary = {
        "schema_version": 2,
        "roster_id": roster["roster_id"],
        "benchmark_id": benchmark["benchmark_id"],
        "predictor_count": len(predictors),
        "environment_count": len(environments),
        "matrix_rows": len(frame),
        "ensemble_seeds": roster["uq_policy"]["ensemble_seeds"],
        "native_uq_preferred": bool(roster["uq_policy"]["native_uq_preferred"]),
        "evaluation_gene_space": roster["uq_policy"]["evaluation_gene_space"],
        "status": "planning_matrix_only_no_scientific_results_claimed",
        "matrix_path": output_path.relative_to(root).as_posix(),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--benchmark", type=Path, default=None)
    parser.add_argument("--roster", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    result = build_matrix(
        root,
        (args.benchmark or root / "configs/context_benchmark.yaml").resolve(),
        (args.roster or root / "configs/model_roster.yaml").resolve(),
        (args.output or root / "artifacts/manifests/predictor_environment_coverage.csv").resolve(),
        (args.summary or root / "artifacts/manifests/predictor_environment_coverage.json").resolve(),
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
