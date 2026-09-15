from __future__ import annotations

import argparse
import csv
import ctypes
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PYTHON = Path(r"E:\anaconda3\envs\sw_mgli\python.exe")
DEFAULT_MANIFEST = ROOT / "results" / "tables" / "baseline_run_matrix.csv"
DEFAULT_LOG = ROOT / "results" / "logs" / "training" / "phase11_baseline_batch.log"
RUN_BASELINE = ROOT / "src" / "baselines" / "run_baseline.py"

MODEL_COST_ORDER = {
    "control_mean_baseline": 0,
    "global_delta_baseline": 1,
    "perturbation_mean_delta_baseline": 2,
    "cell_context_knn_delta_baseline": 3,
    "ridge_regression_baseline": 4,
}

SPLIT_COST_ORDER = {
    "random_split": 0,
    "unseen_perturbation_split": 1,
    "unseen_combination_split": 2,
    "low_support_split": 3,
    "dataset_heldout_split": 4,
    "external_holdout": 5,
}


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log_line(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{now()}] {message}"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def project_python_processes() -> list[dict[str, str]]:
    current_python_pid = os.getpid()
    command = r"""
$currentPython = CURRENT_PYTHON_PID
$items = Get-Process python,pythonw -ErrorAction SilentlyContinue |
  Where-Object { $_.Id -ne $currentPython } |
  Select-Object @{Name='ProcessId';Expression={$_.Id}},@{Name='Name';Expression={$_.ProcessName}},@{Name='CommandLine';Expression={$_.Path}}
$items |
  ConvertTo-Csv -NoTypeInformation
""".replace("CURRENT_PYTHON_PID", str(current_python_pid))
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", command],
            cwd=str(ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=20,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return []
    if result.returncode != 0 or not result.stdout.strip():
        return []
    rows = list(csv.DictReader(result.stdout.splitlines()))
    return [{"pid": row.get("ProcessId", ""), "name": row.get("Name", ""), "command": row.get("CommandLine", "")} for row in rows]


def kill_project_processes(processes: Iterable[dict[str, str]], log_file: Path) -> None:
    for proc in processes:
        pid = proc.get("pid", "")
        if not pid:
            continue
        command = proc.get("command", "")
        normalized = command.replace("/", "\\")
        root_token = str(ROOT).replace("/", "\\")
        if root_token not in normalized and "src\\baselines\\run_baseline.py" not in normalized:
            log_line(log_file, f"Refusing to stop python process without project command evidence pid={pid}: {command[:240]}")
            continue
        log_line(log_file, f"Stopping stale project python process pid={pid}: {proc.get('command', '')[:240]}")
        subprocess.run(
            ["powershell", "-NoProfile", "-Command", f"Stop-Process -Id {int(pid)} -Force"],
            cwd=str(ROOT),
            timeout=30,
            check=False,
        )


def guard_processes(log_file: Path, kill_zombies: bool) -> None:
    processes = project_python_processes()
    if not processes:
        log_line(log_file, "Process guard: no project python processes detected.")
        return
    if kill_zombies:
        kill_project_processes(processes, log_file)
        time.sleep(2)
        remaining = project_python_processes()
        if remaining:
            raise RuntimeError(f"Project python processes remain after cleanup: {remaining}")
        log_line(log_file, "Process guard: stale project python processes cleared.")
        return
    raise RuntimeError(f"Project python processes are already running; rerun with --kill-zombies if they are stale: {processes}")


def free_memory_gb() -> float | None:
    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatus()
    status.dwLength = ctypes.sizeof(MemoryStatus)
    try:
        ok = ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    except AttributeError:
        return None
    if not ok:
        return None
    return float(status.ullAvailPhys) / (1024**3)


def guard_memory(log_file: Path, min_free_gb: float) -> None:
    if min_free_gb <= 0:
        return
    free_gb = free_memory_gb()
    if free_gb is None:
        log_line(log_file, "Memory guard: unable to read free memory; continuing.")
        return
    log_line(log_file, f"Memory guard: free_physical_memory_gb={free_gb:.3f}.")
    if free_gb < min_free_gb:
        raise RuntimeError(f"Free physical memory {free_gb:.3f} GB is below threshold {min_free_gb:.3f} GB.")


def load_pending(manifest_path: Path, args: argparse.Namespace, exclude_run_ids: set[str] | None = None) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)
    manifest["run_metrics_path"] = manifest["output_dir"].map(lambda value: str(Path(str(value)) / "run_metrics.json"))
    manifest["completed"] = manifest["run_metrics_path"].map(lambda value: Path(value).exists())
    pending = manifest.copy() if args.force_existing else manifest[~manifest["completed"]].copy()
    if args.models:
        pending = pending[pending["model"].astype(str).isin(args.models)]
    if args.split_families:
        pending = pending[pending["split_family"].astype(str).isin(args.split_families)]
    if args.dataset_scopes:
        pending = pending[pending["dataset_scope"].astype(str).isin(args.dataset_scopes)]
    if args.seeds:
        pending = pending[pending["seed"].astype(int).isin(args.seeds)]
    if exclude_run_ids:
        pending = pending[~pending["run_id"].astype(str).isin(exclude_run_ids)]
    if args.schedule == "small-first":
        pending["_split_cost"] = pending["split_family"].map(SPLIT_COST_ORDER).fillna(99).astype(int)
        pending["_model_cost"] = pending["model"].map(MODEL_COST_ORDER).fillna(99).astype(int)
        pending["_gene_cost"] = pd.to_numeric(pending["expected_gene_count"], errors="coerce").fillna(999999).astype(int)
        pending = pending.sort_values(["_split_cost", "_gene_cost", "dataset_scope", "seed", "_model_cost", "run_id"])
        pending = pending.drop(columns=["_split_cost", "_model_cost", "_gene_cost"])
    return pending.reset_index(drop=True)


def build_run_one_command(row: pd.Series, python_path: Path) -> list[str]:
    return [
        str(python_path),
        str(RUN_BASELINE),
        "run-one",
        "--model",
        str(row["model"]),
        "--split-json",
        str(row["split_json_path"]),
        "--output-dir",
        str(row["output_dir"]),
        "--seed",
        str(int(row["seed"])),
        "--run-id",
        str(row["run_id"]),
    ]


def run_single_batch(args: argparse.Namespace, max_runs: int, exclude_run_ids: set[str] | None = None) -> tuple[int, int, int, list[str]]:
    log_file = Path(args.log_file)
    python_path = Path(args.python)
    if not python_path.exists():
        raise FileNotFoundError(python_path)
    pending = load_pending(Path(args.manifest), args, exclude_run_ids=exclude_run_ids)
    total_pending = len(pending)
    selected = pending.head(max_runs).copy()
    log_line(log_file, f"Baseline batch starting: pending_after_filters={total_pending}, selected={len(selected)}, schedule={args.schedule}.")
    if selected.empty:
        return 0, 0, total_pending, []
    completed = 0
    failed = 0
    attempted_run_ids: list[str] = []
    for idx, row in selected.iterrows():
        attempted_run_ids.append(str(row["run_id"]))
        guard_processes(log_file, kill_zombies=args.kill_zombies)
        guard_memory(log_file, min_free_gb=args.min_free_gb)
        command = build_run_one_command(row, python_path)
        log_line(log_file, f"Running {idx + 1}/{len(selected)}: {row['run_id']}")
        start = time.time()
        result = subprocess.run(
            command,
            cwd=str(ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=int(args.timeout_minutes * 60),
            check=False,
        )
        elapsed = time.time() - start
        with log_file.open("a", encoding="utf-8") as handle:
            handle.write(result.stdout)
            if result.stdout and not result.stdout.endswith("\n"):
                handle.write("\n")
        if result.returncode == 0:
            metrics_path = Path(str(row["output_dir"])) / "run_metrics.json"
            if metrics_path.exists():
                completed += 1
                log_line(log_file, f"Completed {row['run_id']} in {elapsed:.1f}s.")
            else:
                failed += 1
                log_line(log_file, f"FAILED {row['run_id']} in {elapsed:.1f}s: command exited 0 but {metrics_path} was not created.")
                if args.stop_on_failure:
                    break
        else:
            failed += 1
            log_line(log_file, f"FAILED {row['run_id']} in {elapsed:.1f}s with exit={result.returncode}.")
            if args.stop_on_failure:
                break
        guard_processes(log_file, kill_zombies=args.kill_zombies)
        guard_memory(log_file, min_free_gb=args.min_free_gb)
        time.sleep(max(0.0, float(args.cooldown_seconds)))
    log_line(log_file, f"Baseline batch finished: completed={completed}, failed={failed}, remaining_estimate={max(total_pending - completed, 0)}.")
    return completed, failed, max(total_pending - completed, 0), attempted_run_ids


def run_batch(args: argparse.Namespace) -> int:
    total_completed = 0
    total_failed = 0
    max_total = int(args.max_total_runs or args.max_runs)
    batch_index = 0
    attempted_in_campaign: set[str] = set()
    while total_completed < max_total:
        batch_index += 1
        remaining_allowance = max_total - total_completed
        batch_size = min(int(args.max_runs), remaining_allowance)
        log_line(Path(args.log_file), f"Campaign batch {batch_index} starting: batch_size={batch_size}, max_total_runs={max_total}.")
        completed, failed, remaining, attempted_run_ids = run_single_batch(
            args,
            max_runs=batch_size,
            exclude_run_ids=attempted_in_campaign,
        )
        attempted_in_campaign.update(attempted_run_ids)
        total_completed += completed
        total_failed += failed
        if failed:
            break
        if completed == 0 or remaining == 0:
            break
        if total_completed < max_total:
            sleep_seconds = max(0.0, float(args.pause_between_batches_seconds))
            log_line(Path(args.log_file), f"Campaign pause: sleeping {sleep_seconds:.1f}s before next batch.")
            time.sleep(sleep_seconds)
    log_line(Path(args.log_file), f"Campaign finished: total_completed={total_completed}, total_failed={total_failed}.")
    return 1 if total_failed else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 11 baseline manifest in small, resumable batches.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--python", default=str(DEFAULT_PYTHON))
    parser.add_argument("--log-file", default=str(DEFAULT_LOG))
    parser.add_argument("--max-runs", type=int, default=5)
    parser.add_argument("--max-total-runs", type=int, default=None)
    parser.add_argument("--timeout-minutes", type=float, default=90.0)
    parser.add_argument("--cooldown-seconds", type=float, default=5.0)
    parser.add_argument("--pause-between-batches-seconds", type=float, default=15.0)
    parser.add_argument("--min-free-gb", type=float, default=2.0)
    parser.add_argument("--schedule", choices=["manifest", "small-first"], default="small-first")
    parser.add_argument("--models", nargs="*")
    parser.add_argument("--split-families", nargs="*")
    parser.add_argument("--dataset-scopes", nargs="*")
    parser.add_argument("--seeds", nargs="*", type=int)
    parser.add_argument("--kill-zombies", action="store_true")
    parser.add_argument("--force-existing", action="store_true", help="Include rows with existing run_metrics.json; run-one overwrites outputs.")
    parser.add_argument("--stop-on-failure", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    raise SystemExit(run_batch(parse_args()))


if __name__ == "__main__":
    main()
