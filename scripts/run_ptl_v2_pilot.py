"""Reproducible entry point for the leakage-safe PTL-v2 pilot."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    sys.path.insert(0, str(args.root.resolve() / "src"))
    from ptl.experiments.pilot import run_pilot

    print(run_pilot(args.root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
