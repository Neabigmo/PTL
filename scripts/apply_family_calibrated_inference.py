"""Replace a predictor family's percentile intervals with calibrated seed-block intervals."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.dadj_inference import infer_d_adj  # noqa: E402

PAIR_ORDER = (
    ("frangieh_melanoma_control", "frangieh_melanoma_coculture"),
    ("frangieh_melanoma_control", "frangieh_melanoma_ifng"),
    ("frangieh_melanoma_coculture", "frangieh_melanoma_ifng"),
)


def _one(payload: tuple[str, str, str, str, list[str]]) -> dict[str, object]:
    source, left, right, metric, paths = payload
    prefix = f"{source}__{left}__{right}__{metric}__"
    arrays: dict[str, list[np.ndarray]] = {field: [] for field in ("cross_left_a", "cross_left_b", "cross_right_a", "cross_right_b")}
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            for field in arrays:
                arrays[field].append(np.asarray(archive[prefix + field], dtype=float))
    left_risk = np.stack((np.concatenate(arrays["cross_left_a"]), np.concatenate(arrays["cross_left_b"])), axis=1)
    right_risk = np.stack((np.concatenate(arrays["cross_right_a"]), np.concatenate(arrays["cross_right_b"])), axis=1)
    seed = 20260917 + sum(map(ord, prefix))
    # The calibrated Student-t interval and one-sided test depend only on the
    # 30 observed seed contributions. Optional bootstrap draws are a
    # sensitivity product and do not enter any inferential result.
    result = infer_d_adj(left_risk, right_risk, rng=np.random.default_rng(seed), bootstrap_draws=0, permutation_draws=0)
    return {"source_context": source, "left": left, "right": right, "metric": metric,
            "D_adj_calibrated": result.estimate, "ci_lower_90_calibrated": result.ci_low,
            "ci_upper_90_calibrated": result.ci_high, "inference_pvalue": result.pvalue,
            "detected_calibrated": result.detected}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True)
    parser.add_argument("--part-stems", nargs="+", required=True)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    family = args.family.strip().lower().replace("-", "_")
    paths = [str(ROOT / "artifacts/manifests" / f"{stem}_bootstrap_inputs.npz") for stem in args.part_stems]
    sample = np.load(paths[0], allow_pickle=False)
    tasks = []
    for key in sample.files:
        if not key.endswith("__cross_left_a"):
            continue
        source, left, right, metric = key.removesuffix("__cross_left_a").split("__", 3)
        tasks.append((source, left, right, metric, paths))
    sample.close()
    with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as pool:
        calibrated = pd.DataFrame(pool.map(_one, tasks))
    surface_path = ROOT / "artifacts/manifests/predictor_families_v2" / f"{family}_d_adj_fullsize.csv"
    surface = pd.read_csv(surface_path)
    rows = []
    for record in surface.to_dict(orient="records"):
        source, target, metric = record["source_context"], record["target_context"], record["metric"]
        left, right = next(pair for pair in PAIR_ORDER if set(pair) == {source, target})
        hit = calibrated.loc[(calibrated.source_context == source) & (calibrated.left == left) &
                             (calibrated.right == right) & (calibrated.metric == metric)]
        if len(hit) != 1:
            raise RuntimeError(f"missing calibrated row for {source}->{target} {metric}")
        row = hit.iloc[0]
        record["D_adj"] = float(row.D_adj_calibrated)
        record["ci_lower_90"] = float(row.ci_lower_90_calibrated)
        record["ci_upper_90"] = float(row.ci_upper_90_calibrated)
        record["inference_pvalue"] = float(row.inference_pvalue)
        record["detected_calibrated"] = bool(row.detected_calibrated)
        record["inference_method"] = "seed_block_empirical_variance_student_t_90pct"
        rows.append(record)
    output = pd.DataFrame(rows)
    output.to_csv(surface_path, index=False)
    calibrated.to_csv(surface_path.with_name(f"{family}_calibrated_inference.csv"), index=False)
    print({"family": family, "rows": len(output), "detected": int(output.detected_calibrated.sum())})


if __name__ == "__main__":
    main()
