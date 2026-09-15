#!/usr/bin/env python3
"""
Progress Reporter - Track experiment completion status
"""
import json
import sys
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]
TRACKING_FILE = ROOT / "results" / "logs" / "progress" / "tracking.json"
REGISTRY_FILE = ROOT / "results" / "tables" / "experiment_registry.csv"
BASELINE_MATRIX = ROOT / "results" / "tables" / "baseline_run_matrix.csv"
MANIFEST = ROOT / "results" / "tables" / "perturbation_model_run_matrix.csv"


def load_tracking() -> dict:
    if TRACKING_FILE.exists():
        with open(TRACKING_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {"last_updated": "", "progress": {}}


def save_tracking(data: dict) -> None:
    data["last_updated"] = datetime.now().astimezone().isoformat(timespec="seconds")
    TRACKING_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(TRACKING_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def count_registry_runs() -> dict:
    """Count runs by status from experiment_registry.csv"""
    try:
        import pandas as pd
        if not REGISTRY_FILE.exists():
            return {"total": 0, "completed": 0, "pending": 0, "failed": 0}

        df = pd.read_csv(REGISTRY_FILE)
        total = len(df)
        completed = len(df[df["status"] == "success"])
        cached = len(df[df["status"] == "cached"])
        failed = len(df[df["status"].str.contains("failed|blocked", case=False, na=False)])
        pending = total - completed - cached - failed

        return {
            "total": int(total),
            "completed": int(completed + cached),
            "pending": int(pending),
            "failed": int(failed)
        }
    except Exception as e:
        print(f"Error reading registry: {e}", file=sys.stderr)
        return {"total": 0, "completed": 0, "pending": 0, "failed": 0, "error": str(e)}


def print_progress_bar(percentage: float, width: int = 40) -> str:
    """Create a text progress bar"""
    filled = int(width * percentage / 100)
    bar = "=" * filled + "-" * (width - filled)
    return f"[{bar}] {percentage:.1f}%"


def main() -> None:
    tracking = load_tracking()
    registry_stats = count_registry_runs()

    print(f"\n{'='*60}")
    print(f"  Progress Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    # Baseline runs
    if BASELINE_MATRIX.exists():
        import pandas as pd
        total_baseline = len(pd.read_csv(BASELINE_MATRIX))
    else:
        total_baseline = 525

    completed = registry_stats["completed"]
    pending = total_baseline - completed
    pct = (completed / total_baseline * 100) if total_baseline > 0 else 0

    print(f"Baseline Runs:")
    print(f"  Total:    {total_baseline}")
    print(f"  Completed: {completed}")
    print(f"  Pending:   {pending}")
    print(f"  Failed:    {registry_stats['failed']}")
    print(f"  Progress:  {print_progress_bar(pct)}\n")

    # GEARS runs
    gears_total = 15
    gears_completed = 0
    gears_pct = (gears_completed / gears_total * 100) if gears_total > 0 else 0
    gears_blocked = True

    print(f"GEARS Runs:")
    print(f"  Total:    {gears_total}")
    print(f"  Completed: {gears_completed}")
    print(f"  Pending:   {gears_total - gears_completed}")
    print(f"  Blocked:   {gears_blocked}")
    print(f"  Progress:  {print_progress_bar(gears_pct)}\n")

    # Overall progress
    all_total = total_baseline + gears_total
    all_completed = completed + gears_completed
    all_pct = (all_completed / all_total * 100) if all_total > 0 else 0

    print(f"Overall Progress: {print_progress_bar(all_pct)}")
    print(f"\n{'='*60}\n")

    # Update tracking file
    tracking["progress"] = {
        "baseline_runs": {
            "total": total_baseline,
            "completed": completed,
            "pending": pending,
            "percentage": round(pct, 2)
        },
        "gears_runs": {
            "total": gears_total,
            "completed": gears_completed,
            "pending": gears_total - gears_completed,
            "blocked": gears_blocked
        }
    }
    save_tracking(tracking)


if __name__ == "__main__":
    main()
