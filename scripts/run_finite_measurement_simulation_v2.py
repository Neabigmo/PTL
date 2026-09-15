"""Run the pre-registered finite-measurement ordering simulation.

The simulation has a known latent ordering law and separates three quantities:
the latent context shift, sampling noise at a specified cell depth, and a
tie-aware stable-order rule with minimum strict support eight.  It reports
calibration of the corrected estimator and comparator recovery; it does not
assert universal superiority of any comparator.

The default ``primary`` mode executes the 500-trial preregistered slice at
N=100, tie probability 0.10, noise scale 1.0, all six depths, all five
inversion fractions, and both Gaussian-heteroscedastic and heavy-tailed laws.
``--mode sensitivity`` adds a compact factorial screen over tie/noise/N.  The
summary is intentionally compact; raw trial-level arrays are not retained.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import beta, rankdata

ROOT = Path(__file__).resolve().parents[1]
DEPTHS = (5, 10, 20, 40, 80, 160)
TIES = (0.0, 0.1, 0.3, 0.5)
NOISES = (0.25, 0.5, 1.0, 2.0)
INVERSIONS = (0.0, 0.05, 0.1, 0.2, 0.4)
COUNTS = (50, 100, 250)
LAWS = ("gaussian_heteroscedastic", "student_t_heavy_tailed")
METRICS = (
    "d_adj", "kendall", "spearman", "top10_jaccard", "stable_inversion",
    "floor_adjusted_kendall", "floor_adjusted_spearman", "floor_adjusted_jaccard",
)
MINIMUM_STRICT_SUPPORT = 8
N_REPLICATES = 12


def _rank_distance(left: np.ndarray, right: np.ndarray) -> tuple[float, float, float]:
    left_r = rankdata(left, method="average")
    right_r = rankdata(right, method="average")
    n = len(left)
    n_pairs = n * (n - 1) / 2.0
    left_delta = left[:, None] - left[None, :]
    right_delta = right[:, None] - right[None, :]
    upper = np.triu(np.ones((n, n), dtype=bool), k=1)
    discordant = np.sum(upper & (left_delta * right_delta < 0))
    kendall = float(discordant / n_pairs)
    spearman = float(np.mean((left_r - right_r) ** 2) / (n * n - 1.0) * 3.0)
    k = max(1, int(np.floor(n * 0.1)))
    source_top = set(np.argsort(left, kind="stable")[:k].tolist())
    target_top = set(np.argsort(right, kind="stable")[:k].tolist())
    union = len(source_top | target_top)
    jaccard = float(1.0 - len(source_top & target_top) / union) if union else 0.0
    return kendall, spearman, jaccard


def _pair_probabilities(values: np.ndarray, tie_tolerance: float = 0.0) -> np.ndarray:
    n_reps, n_items = values.shape
    i, j = np.triu_indices(n_items, k=1)
    difference = values[:, i] - values[:, j]
    tolerance = float(tie_tolerance)
    return np.column_stack((
        np.mean(difference < -tolerance, axis=0),
        np.mean(np.abs(difference) <= tolerance, axis=0),
        np.mean(difference > tolerance, axis=0),
    ))


def _d_adj(left: np.ndarray, right: np.ndarray) -> float:
    p_left = _pair_probabilities(left)
    p_right = _pair_probabilities(right)
    cross = np.mean(1.0 - np.sum(p_left * p_right, axis=1))
    def u_floor(p: np.ndarray, n: int) -> float:
        counts = np.rint(p * n).astype(np.int64)
        return float(np.mean(1.0 - np.sum(counts * (counts - 1), axis=1) / (n * (n - 1))))
    return float(cross - 0.5 * (u_floor(p_left, left.shape[0]) + u_floor(p_right, right.shape[0])))


def _stable_inversion(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    p_left = _pair_probabilities(left)
    p_right = _pair_probabilities(right)
    support_left = (p_left[:, 0] + p_left[:, 2]) * left.shape[0]
    support_right = (p_right[:, 0] + p_right[:, 2]) * right.shape[0]
    left_minus = np.rint(p_left[:, 0] * left.shape[0]).astype(int)
    left_plus = np.rint(p_left[:, 2] * left.shape[0]).astype(int)
    right_minus = np.rint(p_right[:, 0] * right.shape[0]).astype(int)
    right_plus = np.rint(p_right[:, 2] * right.shape[0]).astype(int)
    left_lower = beta.ppf(0.025, left_plus + 1, left_minus + 1)
    left_upper = beta.ppf(0.975, left_plus + 1, left_minus + 1)
    right_lower = beta.ppf(0.025, right_plus + 1, right_minus + 1)
    right_upper = beta.ppf(0.975, right_plus + 1, right_minus + 1)
    stable_left = (support_left >= MINIMUM_STRICT_SUPPORT) & ((left_lower > 0.5) | (left_upper < 0.5))
    stable_right = (support_right >= MINIMUM_STRICT_SUPPORT) & ((right_lower > 0.5) | (right_upper < 0.5))
    stable = stable_left & stable_right
    left_sign = np.where(p_left[:, 2] > p_left[:, 0], 1, np.where(p_left[:, 0] > p_left[:, 2], -1, 0))
    right_sign = np.where(p_right[:, 2] > p_right[:, 0], 1, np.where(p_right[:, 0] > p_right[:, 2], -1, 0))
    return (float(np.mean(left_sign[stable] != right_sign[stable])) if np.any(stable) else float("nan"), float(np.mean(stable)))


def _quantize(values: np.ndarray, tie_probability: float) -> np.ndarray:
    if tie_probability <= 0:
        return values
    bins = max(2, int(round(1.0 / tie_probability)))
    lo, hi = float(np.min(values)), float(np.max(values))
    scaled = (values - lo) / max(hi - lo, 1e-12)
    return np.floor(scaled * bins) / bins


def _trial(rng: np.random.Generator, *, n_items: int, depth: int, tie_probability: float, noise_scale: float, inversion_fraction: float, law: str) -> dict[str, float]:
    latent_source = np.sort(rng.normal(size=n_items))
    rho = float(np.cos(np.pi * inversion_fraction))
    if law == "student_t_heavy_tailed":
        innovation = rng.standard_t(df=3, size=n_items)
        innovation /= max(float(np.std(innovation)), 1e-8)
        source_noise = rng.standard_t(df=3, size=(N_REPLICATES, n_items))
        source_noise /= max(float(np.std(source_noise)), 1e-8)
        target_noise = rng.standard_t(df=3, size=(N_REPLICATES, n_items))
        target_noise /= max(float(np.std(target_noise)), 1e-8)
    else:
        innovation = rng.normal(size=n_items)
        source_noise = rng.normal(size=(N_REPLICATES, n_items))
        target_noise = rng.normal(size=(N_REPLICATES, n_items))
    latent_target = rho * latent_source + np.sqrt(max(1.0 - rho * rho, 0.0)) * innovation
    hetero_source = 0.35 + 0.65 * (np.abs(latent_source) / (np.max(np.abs(latent_source)) + 1e-8))
    hetero_target = 0.35 + 0.65 * (np.abs(latent_target) / (np.max(np.abs(latent_target)) + 1e-8))
    sigma = noise_scale / np.sqrt(float(depth))
    measured_source = latent_source[None, :] + sigma * source_noise * hetero_source[None, :]
    measured_target = latent_target[None, :] + sigma * target_noise * hetero_target[None, :]
    measured_source = _quantize(measured_source, tie_probability)
    measured_target = _quantize(measured_target, tie_probability)
    source_mean, target_mean = measured_source.mean(axis=0), measured_target.mean(axis=0)
    kendall, spearman, jaccard = _rank_distance(source_mean, target_mean)
    # Same-context half-split floors use the identical estimator unit as the
    # finite-measurement diagnostic, but never replace the corrected estimate.
    left_half, right_half = measured_source[:N_REPLICATES // 2], measured_target[:N_REPLICATES // 2]
    floor_s = _rank_distance(measured_source[:N_REPLICATES // 2].mean(axis=0), measured_source[N_REPLICATES // 2:].mean(axis=0))
    floor_t = _rank_distance(measured_target[:N_REPLICATES // 2].mean(axis=0), measured_target[N_REPLICATES // 2:].mean(axis=0))
    stable, stable_coverage = _stable_inversion(measured_source, measured_target)
    true_k, true_s, true_j = _rank_distance(latent_source, latent_target)
    return {
        "d_adj": _d_adj(measured_source, measured_target),
        "kendall": kendall, "spearman": spearman, "top10_jaccard": jaccard,
        "stable_inversion": stable,
        "floor_adjusted_kendall": kendall - 0.5 * (floor_s[0] + floor_t[0]),
        "floor_adjusted_spearman": spearman - 0.5 * (floor_s[1] + floor_t[1]),
        "floor_adjusted_jaccard": jaccard - 0.5 * (floor_s[2] + floor_t[2]),
        "true_d_adj": true_k,
        "true_kendall": true_k, "true_spearman": true_s, "true_top10_jaccard": true_j,
        "stable_coverage": stable_coverage,
    }


def _condition(task: tuple[dict[str, Any], int, int]) -> dict[str, Any]:
    spec, trials, seed = task
    rng = np.random.default_rng(seed)
    rows = [_trial(rng, **spec) for _ in range(trials)]
    output: dict[str, Any] = {**spec, "trials": trials}
    for metric in METRICS:
        values = np.asarray([row[metric] for row in rows], dtype=float)
        output[f"{metric}_mean"] = float(np.nanmean(values))
        output[f"{metric}_sd"] = float(np.nanstd(values, ddof=1))
    for metric, true_name in (("d_adj", "true_d_adj"), ("kendall", "true_kendall"), ("spearman", "true_spearman"), ("top10_jaccard", "true_top10_jaccard")):
        estimate = np.asarray([row[metric] for row in rows], dtype=float)
        truth = np.asarray([row[true_name] for row in rows], dtype=float)
        output[f"{metric}_mae_to_latent_truth"] = float(np.nanmean(np.abs(estimate - truth)))
        output[f"{metric}_bias_to_latent_truth"] = float(np.nanmean(estimate - truth))
    output["stable_coverage_mean"] = float(np.nanmean([row["stable_coverage"] for row in rows]))
    return output


def _specs(mode: str) -> list[dict[str, Any]]:
    if mode == "primary":
        return [
            {"n_items": 100, "depth": depth, "tie_probability": 0.1, "noise_scale": 1.0, "inversion_fraction": inversion, "law": law}
            for law in LAWS for depth in DEPTHS for inversion in INVERSIONS
        ]
    if mode == "sensitivity":
        return [
            {"n_items": n, "depth": depth, "tie_probability": tie, "noise_scale": noise, "inversion_fraction": inversion, "law": law}
            for law in LAWS for n in COUNTS for depth in DEPTHS for tie in TIES for noise in NOISES for inversion in INVERSIONS
        ]
    raise ValueError(f"unknown mode {mode}")


def run(*, mode: str, trials: int, workers: int, seed: int) -> dict[str, Any]:
    specs = _specs(mode)
    tasks = [(spec, trials, seed + 7919 * index) for index, spec in enumerate(specs)]
    if workers <= 1:
        rows = [_condition(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(_condition, tasks, chunksize=1))
    outdir = ROOT / "artifacts/manifests/simulation_v2"
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / f"finite_measurement_{mode}.csv"
    pd = __import__("pandas")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    report = {
        "schema_version": 1, "status": "executed", "mode": mode,
        "trials_per_condition": trials, "condition_count": len(rows),
        "workers": workers, "seed": seed, "n_replicates": N_REPLICATES,
        "minimum_strict_support": MINIMUM_STRICT_SUPPORT,
        "estimators": list(METRICS), "generative_laws": list(LAWS),
        "truth_definition": "latent continuous source/target rank distances under a Gaussian-copula inversion law",
        "stable_rule": "Beta(plus+1, minus+1) 95% direction posterior with minimum strict support >=8 is enforced; the canonical exact implementation is shared with src/evaluation/ordering_estimands.py.",
        "interpretation": "Calibration and comparator recovery screen only; no uniform superiority claim is promoted without agreement across both laws and preregistered conditions.",
        "csv": csv_path.relative_to(ROOT).as_posix(),
    }
    json_path = outdir / f"finite_measurement_{mode}.json"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {**report, "json": json_path.relative_to(ROOT).as_posix()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("primary", "sensitivity"), default="primary")
    parser.add_argument("--trials-per-condition", type=int, default=500)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260915)
    args = parser.parse_args()
    if args.trials_per_condition < 10:
        raise ValueError("trials-per-condition must be >=10")
    print(json.dumps(run(mode=args.mode, trials=args.trials_per_condition, workers=args.workers, seed=args.seed), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
