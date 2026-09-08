"""Run the canonical predictor pipeline over the predeclared stability seeds.

Each seed is an independent process with an independent manifest, metrics file,
contract file, and prediction directory.  The primary 20260907 run is never
replaced by a stability job; this runner only orchestrates the existing
canonical scripts and does not change the split fractions, model seeds, or
evaluation protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
PRIMARY_SEED = 20260907


def seeds_from_config(path: Path) -> list[int]:
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    seeds = [int(spec["split_seed"]), *(int(value) for value in spec.get("stability_seeds", []))]
    if seeds[0] != PRIMARY_SEED or len(seeds) != len(set(seeds)):
        raise ValueError("split config must keep primary seed 20260907 and unique stability seeds")
    return seeds


def run_job(root: Path, seed: int, *, resume: bool) -> dict[str, Any]:
    manifest = root / "artifacts/manifests/biological_instance_registry.csv"
    summary = root / "artifacts/manifests/biological_instance_registry.json"
    metrics = root / "artifacts/manifests/formal_v2_predictor_metrics.csv"
    contract = root / "artifacts/manifests/formal_v2_prediction_index.csv"
    if seed != PRIMARY_SEED:
        suffix = f"__split_{seed}"
        manifest = root / f"artifacts/manifests/biological_instance_registry{suffix}.csv"
        summary = root / f"artifacts/manifests/biological_instance_registry{suffix}.json"
        metrics = root / f"artifacts/manifests/formal_v2_predictor_metrics{suffix}.csv"
        contract = root / f"artifacts/manifests/formal_v2_prediction_index{suffix}.csv"
    job_dir = root / "results/formal_v2/multisplit" / str(seed)
    output_dir = root / "results/formal_v2/predictors" if seed == PRIMARY_SEED else job_dir / "predictors"
    job_dir.mkdir(parents=True, exist_ok=True)
    log_path = job_dir / "execution.log"
    required = [manifest, summary, metrics, contract]
    primary_arrays = sorted((root / "results/formal_v2/predictors").glob("*.npz"))
    primary_checksums = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in primary_arrays
    } if seed != PRIMARY_SEED else {}
    if resume and all(path.is_file() for path in required) and len(list(output_dir.glob("*.npz"))) >= 1:
        if seed != PRIMARY_SEED:
            after_checksums = {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted((root / "results/formal_v2/predictors").glob("*.npz"))
            }
            if after_checksums != primary_checksums:
                raise AssertionError(f"resumed stability split {seed} changed canonical primary prediction arrays")
        return {"split_seed": seed, "status": "resumed", "log_path": log_path.relative_to(root).as_posix()}

    env = os.environ.copy()
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[variable] = "1"
    commands = []
    if not manifest.is_file() or not summary.is_file():
        commands.append([
            sys.executable, str(root / "scripts/build_biological_instance_ground_truth.py"),
            "--root", str(root), "--split-seed", str(seed),
        ])
    commands.append([
            sys.executable, str(root / "scripts/run_formal_v2_predictors.py"), "--root", str(root),
            "--output-dir", str(output_dir), "--metrics", str(metrics), "--contract", str(contract),
            "--manifest", str(manifest), "--split-seed", str(seed),
            "--isolated-output-dir",
        ])
    with log_path.open("w", encoding="utf-8") as log:
        for command in commands:
            log.write("$ " + " ".join(command) + "\n")
            log.flush()
            completed = subprocess.run(command, cwd=root, env=env, text=True, stdout=log, stderr=subprocess.STDOUT)
            if completed.returncode:
                raise RuntimeError(f"split seed {seed} failed with exit code {completed.returncode}; see {log_path}")
    if not all(path.is_file() for path in required):
        raise RuntimeError(f"split seed {seed} completed without all required artifacts")
    expected_arrays = len(pd.read_csv(root / "artifacts/manifests/environment_registry.csv")) * 3
    actual_arrays = len(list(output_dir.glob("*.npz")))
    if actual_arrays != expected_arrays:
        raise RuntimeError(f"split seed {seed} wrote {actual_arrays} arrays; expected {expected_arrays} in {output_dir}")
    if seed != PRIMARY_SEED:
        after_checksums = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((root / "results/formal_v2/predictors").glob("*.npz"))
        }
        if after_checksums != primary_checksums:
            raise AssertionError(f"stability split {seed} changed canonical primary prediction arrays")
    return {
        "split_seed": seed,
        "status": "completed",
        "manifest": manifest.relative_to(root).as_posix(),
        "metrics": metrics.relative_to(root).as_posix(),
        "contract": contract.relative_to(root).as_posix(),
        "log_path": log_path.relative_to(root).as_posix(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--split-config", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    config = (args.split_config or root / "configs/biological_instance_split.yaml").resolve()
    jobs = [run_job(root, seed, resume=args.resume) for seed in seeds_from_config(config)]
    result = {
        "schema_version": 1,
        "split_id": "ptl_biological_instance_split_v1",
        "primary_seed": PRIMARY_SEED,
        "seeds": jobs,
        "process_policy": "independent_source_disjoint_seed_jobs; one BLAS thread per job; no nested parallelism",
        "status": "completed" if all(job["status"] in {"completed", "resumed"} for job in jobs) else "failed",
    }
    output = root / "artifacts/manifests/formal_v2_multisplit_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
