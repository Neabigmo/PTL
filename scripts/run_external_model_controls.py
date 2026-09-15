"""Run the external, condition-disjoint model-control campaign.

All models use identical held-out perturbation conditions within each public
cohort.  Baselines are CPU-parallel; GPU GEARS jobs stay serial to avoid
memory contention.  This runner is intended for a separate remote workspace.
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
DATASETS = (
    "DatlingerBock2021",
    "NormanWeissman2019_filtered",
    "PapalexiSatija2021_eccite_RNA",
    "ReplogleWeissman2022_K562_gwps",
    "ReplogleWeissman2022_K562_essential",
    "ReplogleWeissman2022_rpe1",
    "TianKampmann2021_CRISPRa",
    "TianKampmann2021_CRISPRi",
    "XuCao2023",
)
SEEDS = (0, 1, 2)
BASELINES = (
    "control_mean_baseline",
    "global_delta_baseline",
    "perturbation_mean_delta_baseline",
    "cell_context_knn_delta_baseline",
    "ridge_regression_baseline",
)


def _run(command: list[str]) -> dict[str, object]:
    done = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    return {"command": command, "returncode": done.returncode, "stdout": done.stdout[-4000:], "stderr": done.stderr[-4000:]}


def _baseline_job(dataset: str, seed: int, model: str) -> dict[str, object]:
    split = ROOT / "data" / "processed" / "splits" / f"gears_condition_split__{dataset}__seed{seed}__signature.json"
    output = ROOT / "results" / "external_controls" / "baselines" / model / f"{dataset}__seed{seed}"
    result = _run([
        PYTHON, "src/baselines/run_baseline.py", "run-one", "--model", model,
        "--split-json", str(split), "--output-dir", str(output), "--seed", str(seed),
        "--run-id", f"external_condition_holdout__{dataset}__seed{seed}__{model}",
    ])
    result.update({"kind": "baseline", "dataset": dataset, "seed": seed, "model": model, "output_dir": str(output)})
    return result


def _completed_baseline_event(dataset: str, seed: int, model: str) -> dict[str, object] | None:
    """Reuse a finished independent baseline when resuming a GPU-only retry."""

    output = ROOT / "results" / "external_controls" / "baselines" / model / f"{dataset}__seed{seed}"
    metrics = output / "run_metrics.json"
    if not metrics.is_file():
        return None
    try:
        json.loads(metrics.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return {
        "kind": "baseline",
        "dataset": dataset,
        "seed": seed,
        "model": model,
        "returncode": 0,
        "output_dir": str(output),
        "resumed": True,
    }


def _gears_job(dataset: str, seed: int, epochs: int) -> dict[str, object]:
    template_matrix = ROOT / "results" / "external_controls" / "gears_run_matrix.csv"
    output = ROOT / "results" / "external_controls" / "gears" / f"{dataset}__seed{seed}"
    run_id = f"external_condition_holdout__{dataset}__seed{seed}__gears"
    rows = pd.read_csv(template_matrix)
    selected = rows.loc[(rows["dataset_id"].eq(dataset)) & (rows["seed"].eq(seed))].copy()
    if len(selected) != 1:
        raise ValueError(f"expected one GEARS row for {dataset} seed {seed}")
    selected.loc[:, "run_id"] = run_id
    selected.loc[:, "output_dir"] = str(output)
    selected.loc[:, "prediction_path"] = str(output / "test_predictions.npz")
    selected.loc[:, "metadata_path"] = str(output / "test_metadata.parquet")
    selected.loc[:, "genes_path"] = str(output / "genes.txt")
    selected.loc[:, "log_file"] = str(ROOT / "results" / "logs" / "training" / f"{run_id}.log")
    matrix = ROOT / "results" / "external_controls" / f"gears_run_matrix__{dataset}__seed{seed}.csv"
    selected.to_csv(matrix, index=False)
    result = _run([
        PYTHON, "src/models/run_perturbation_model.py", "run-one", "--run-id", run_id,
        "--run-matrix", str(matrix), "--epochs", str(epochs), "--batch-size", "16",
        "--test-batch-size", "64", "--hidden-size", "64", "--device", "cuda", "--max-genes", "4096",
    ])
    result.update({"kind": "gears", "dataset": dataset, "seed": seed, "model": "gears", "output_dir": str(output)})
    return result


def _collect(events: list[dict[str, object]]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for event in events:
        record = {key: event.get(key) for key in ("kind", "dataset", "seed", "model", "returncode", "output_dir")}
        metrics = Path(str(event["output_dir"])) / "run_metrics.json"
        if metrics.exists():
            try:
                record.update(json.loads(metrics.read_text(encoding="utf-8")))
            except json.JSONDecodeError:
                pass
        records.append(record)
    return pd.DataFrame(records)


def _available_datasets(candidates: tuple[str, ...]) -> tuple[list[str], list[str]]:
    """Select cohorts whose processed signature is present in this workspace.

    The remote external-control workspace intentionally transfers only cohorts
    eligible for the active campaign.  Keeping unavailable candidates out of
    the split builder avoids a missing optional cohort aborting all remaining
    independent controls.
    """

    summary = pd.read_csv(ROOT / "results" / "tables" / "preprocessing_summary.csv")
    available: list[str] = []
    absent: list[str] = []
    for dataset in candidates:
        row = summary.loc[summary["dataset_id"].astype(str).eq(dataset)]
        if row.empty:
            absent.append(dataset)
            continue
        signature = Path(str(row.iloc[0]["signature_path"]))
        if not signature.is_absolute():
            signature = ROOT / signature
        if signature.is_file():
            available.append(dataset)
        else:
            absent.append(dataset)
    return available, absent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--baseline-workers", type=int, default=8)
    args = parser.parse_args()
    output = ROOT / "results" / "external_controls"
    output.mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "logs" / "training").mkdir(parents=True, exist_ok=True)
    available_datasets, absent_datasets = _available_datasets(DATASETS)
    if absent_datasets:
        print("not transferred to this remote campaign: " + ", ".join(absent_datasets), flush=True)
    build = _run([PYTHON, "scripts/build_gears_scientific_pilot_splits.py", "--datasets", *available_datasets, "--seeds", *map(str, SEEDS), "--audit", str(output / "gears_split_audit.csv")])
    if build["returncode"] != 0:
        raise RuntimeError(str(build["stderr"]))
    audit = pd.read_csv(output / "gears_split_audit.csv")
    audit = audit.loc[audit["status"].eq("ready")].copy()
    if audit.empty:
        raise RuntimeError("no public cohorts are eligible for condition-disjoint GEARS evaluation")
    matrix = pd.DataFrame([
        {
            "run_id": f"template__{row.dataset_scope}__seed{int(row.seed)}__gears", "model": "gears",
            "dataset_id": row.dataset_scope, "dataset_scope": row.dataset_scope, "split_family": row.split_family,
            "split_protocol": row.split_protocol, "perturbation_overlap": row.perturbation_overlap, "seed": int(row.seed),
            "split_json_path": row.output_path, "track": row.track, "gene_space_policy": "native",
            "output_dir": "", "prediction_path": "", "metadata_path": "", "genes_path": "", "log_file": "", "status": "planned",
        } for row in audit.itertuples(index=False)
    ])
    matrix.to_csv(output / "gears_run_matrix.csv", index=False)
    events: list[dict[str, object]] = []
    ready_pairs = [(str(row.dataset_scope), int(row.seed)) for row in audit.itertuples(index=False)]
    jobs = [(dataset, seed, model) for dataset, seed in ready_pairs for model in BASELINES]
    pending_jobs = []
    for job in jobs:
        completed = _completed_baseline_event(*job)
        if completed is None:
            pending_jobs.append(job)
        else:
            events.append(completed)
    if events:
        print(f"reusing {len(events)} completed baseline runs", flush=True)
    with ProcessPoolExecutor(max_workers=max(1, args.baseline_workers)) as pool:
        futures = [pool.submit(_baseline_job, *job) for job in pending_jobs]
        for future in as_completed(futures):
            event = future.result(); events.append(event)
            print(f"baseline {event['dataset']} seed={event['seed']} {event['model']}: {event['returncode']}", flush=True)
    for dataset, seed in ready_pairs:
        event = _gears_job(dataset, seed, args.epochs); events.append(event)
        print(f"GEARS {dataset} seed={seed}: {event['returncode']}", flush=True)
    (output / "campaign_events.json").write_text(json.dumps(events, indent=2), encoding="utf-8")
    summary = _collect(events)
    summary.to_csv(output / "external_model_control_summary.csv", index=False)
    failed = int((summary["returncode"] != 0).sum())
    print(f"complete: {len(summary)} runs; failures={failed}", flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
