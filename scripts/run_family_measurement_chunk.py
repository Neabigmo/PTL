"""Run one source-only predictor family on a declared seed chunk."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_formal_v2_claim_lock_measurement as measurement  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--family", required=True)
    parser.add_argument("--prediction-path", type=Path, required=True)
    parser.add_argument("--output-stem", required=True)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed-start-index", type=int, required=True)
    parser.add_argument("--seed-stop-index", type=int, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    start = int(args.seed_start_index)
    stop = int(args.seed_stop_index)
    if not 0 <= start < stop <= len(measurement.SPLIT_SEEDS):
        raise ValueError("seed range must be within the declared 30-seed schedule")
    prediction_path = args.prediction_path.resolve()
    if not prediction_path.is_file():
        raise FileNotFoundError(prediction_path)
    previous = measurement.PREDICTION_PATH
    try:
        measurement.PREDICTION_PATH = prediction_path
        report = measurement.run(
            root,
            seed_chunk=1,
            draws=int(args.draws),
            output_stem=args.output_stem,
            resampling_mode="full_size_nonparametric",
            split_seeds=tuple(measurement.SPLIT_SEEDS[start:stop]),
            stream_bootstrap_inputs=True,
        )
    finally:
        measurement.PREDICTION_PATH = previous
    report["family"] = args.family
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
