from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

import pandas as pd

from run_baseline_batch import (
    DEFAULT_LOG,
    DEFAULT_MANIFEST,
    DEFAULT_PYTHON,
    ROOT,
    RUN_BASELINE,
    guard_memory,
    guard_processes,
    log_line,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline rows in small split-grouped batches.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--python", default=str(DEFAULT_PYTHON))
    parser.add_argument("--log-file", default=str(DEFAULT_LOG))
    parser.add_argument("--max-groups", type=int, default=1)
    parser.add_argument("--models", nargs="*")
    parser.add_argument("--split-families", nargs="*")
    parser.add_argument("--dataset-scopes", nargs="*")
    parser.add_argument("--seeds", nargs="*", type=int)
    parser.add_argument("--timeout-minutes", type=float, default=120.0)
    parser.add_argument("--pause-between-groups-seconds", type=float, default=30.0)
    parser.add_argument("--min-free-gb", type=float, default=4.0)
    parser.add_argument("--kill-zombies", action="store_true")
    parser.add_argument("--force", action="store_true", help="Run selected rows even when run_metrics.json already exists.")
    return parser.parse_args()


def load_pending(args: argparse.Namespace) -> pd.DataFrame:
    frame = pd.read_csv(args.manifest)
    frame["run_metrics_path"] = frame["output_dir"].map(lambda value: str(Path(str(value)) / "run_metrics.json"))
    frame["completed"] = frame["run_metrics_path"].map(lambda value: Path(value).exists())
    pending = frame.copy() if args.force else frame[~frame["completed"]].copy()
    if args.models:
        pending = pending[pending["model"].astype(str).isin(args.models)]
    if args.split_families:
        pending = pending[pending["split_family"].astype(str).isin(args.split_families)]
    if args.dataset_scopes:
        pending = pending[pending["dataset_scope"].astype(str).isin(args.dataset_scopes)]
    if args.seeds:
        pending = pending[pending["seed"].astype(int).isin(args.seeds)]
    pending = pending.sort_values(["split_family", "dataset_scope", "heldout_target", "seed", "split_json_path", "model", "run_id"])
    return pending.reset_index(drop=True)


def run_group(args: argparse.Namespace, group: pd.DataFrame, group_index: int, group_count: int) -> tuple[int, int]:
    log_file = Path(args.log_file)
    guard_processes(log_file, kill_zombies=args.kill_zombies)
    guard_memory(log_file, min_free_gb=args.min_free_gb)
    with tempfile.NamedTemporaryFile("w", suffix=".csv", prefix="baseline_group_", dir=str(ROOT / "results" / "logs" / "training"), delete=False, encoding="utf-8", newline="") as handle:
        temp_manifest = Path(handle.name)
        group.drop(columns=[column for column in ["run_metrics_path", "completed"] if column in group.columns]).to_csv(handle, index=False)
    log_line(log_file, f"Grouped run {group_index}/{group_count}: rows={len(group)}, split={group['split_json_path'].iloc[0]}")
    import subprocess

    start = time.time()
    result = subprocess.run(
        [str(args.python), str(RUN_BASELINE), "run-all", "--manifest", str(temp_manifest)] + (["--force"] if args.force else []),
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
    completed_mask = group["output_dir"].map(lambda value: (Path(str(value)) / "run_metrics.json").exists())
    completed = int(completed_mask.sum())
    failed = int((~completed_mask).sum())
    if result.returncode != 0:
        failed = max(failed, 1)
    log_line(log_file, f"Grouped run finished in {elapsed:.1f}s: completed={completed}, failed={failed}.")
    guard_processes(log_file, kill_zombies=args.kill_zombies)
    guard_memory(log_file, min_free_gb=args.min_free_gb)
    return completed, failed


def main() -> None:
    args = parse_args()
    pending = load_pending(args)
    log_file = Path(args.log_file)
    if args.max_groups <= 0:
        log_line(log_file, "Grouped campaign: max_groups <= 0, no rows selected.")
        return
    if pending.empty:
        log_line(log_file, "Grouped campaign: no pending rows after filters.")
        return
    groups = []
    for _, group in pending.groupby("split_json_path", sort=False):
        groups.append(group.copy())
        if len(groups) >= args.max_groups:
            break
    total_completed = 0
    total_failed = 0
    for idx, group in enumerate(groups, start=1):
        completed, failed = run_group(args, group, idx, len(groups))
        total_completed += completed
        total_failed += failed
        if failed:
            break
        if idx < len(groups):
            time.sleep(max(0.0, float(args.pause_between_groups_seconds)))
    log_line(log_file, f"Grouped campaign finished: completed={total_completed}, failed={total_failed}.")
    raise SystemExit(1 if total_failed else 0)


if __name__ == "__main__":
    main()
