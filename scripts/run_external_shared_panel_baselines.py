"""Evaluate simple external controls on the frozen GEARS gene panel.

The public-cohort GEARS pilot intentionally uses a declared 4,096-gene panel.
This companion runner evaluates the same baseline families on exactly that
panel, so a model comparison never mixes full-native and pilot-gene metrics.
It is meant to run only after the GEARS campaign has materialized every cohort
panel; full-native baseline results remain a separately labelled reference.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
MODELS = (
    "control_mean_baseline",
    "global_delta_baseline",
    "perturbation_mean_delta_baseline",
    "cell_context_knn_delta_baseline",
    "ridge_regression_baseline",
)


def run_job(dataset: str, seed: int, split_json: Path, panel: Path, model: str, output_root: Path) -> dict[str, object]:
    output = output_root / model / f"{dataset}__seed{seed}"
    metrics = output / "run_metrics.json"
    if metrics.is_file():
        return {"dataset": dataset, "seed": seed, "model": model, "returncode": 0, "output_dir": str(output), "resumed": True}
    command = [
        PYTHON,
        "src/baselines/run_baseline.py",
        "run-one",
        "--model",
        model,
        "--split-json",
        str(split_json),
        "--output-dir",
        str(output),
        "--seed",
        str(seed),
        "--run-id",
        f"external_condition_holdout_shared_panel__{dataset}__seed{seed}__{model}",
        "--gene-panel",
        str(panel),
    ]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    return {
        "dataset": dataset,
        "seed": seed,
        "model": model,
        "returncode": result.returncode,
        "output_dir": str(output),
        "stdout": result.stdout[-4000:],
        "stderr": result.stderr[-4000:],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--audit",
        type=Path,
        default=ROOT / "results" / "external_controls" / "gears_split_audit.csv",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "results" / "external_controls" / "shared_panel_baselines",
    )
    args = parser.parse_args()

    audit = pd.read_csv(args.audit)
    ready = audit.loc[audit["status"].eq("ready")].copy()
    if ready.empty:
        raise RuntimeError("No ready external condition-holdout rows were found.")
    jobs: list[tuple[str, int, Path, Path, str, Path]] = []
    for row in ready.itertuples(index=False):
        dataset = str(row.dataset_scope)
        panel = ROOT / "artifacts" / "manifests" / f"gears_pilot_gene_space__{dataset}.txt"
        if not panel.is_file():
            raise FileNotFoundError(f"GEARS panel is not yet available for {dataset}: {panel}")
        split_json = Path(str(row.output_path))
        for model in MODELS:
            jobs.append((dataset, int(row.seed), split_json, panel, model, args.output_root))

    events: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(run_job, *job) for job in jobs]
        for future in as_completed(futures):
            event = future.result()
            events.append(event)
            print(f"{event['dataset']} seed={event['seed']} {event['model']}: {event['returncode']}", flush=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "campaign_events.json").write_text(json.dumps(events, indent=2), encoding="utf-8")
    failures = sum(int(event["returncode"]) != 0 for event in events)
    print(f"complete: {len(events)} shared-panel baseline runs; failures={failures}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
