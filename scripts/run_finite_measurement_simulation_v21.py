"""Calibrate finite-measurement ordering estimators against a finite truth law.

This version keeps the old simulation artifacts intact and fixes their main
conceptual limitation: the truth for D_adj is the population pair-state law
induced by finite-depth measurements (including the declared resolution
setting), not a latent continuous Kendall distance. Kendall, Spearman and
top-10% Jaccard receive their own population-mean targets.

The primary factorial is the registered 2 laws x 6 depths x 4 resolution
settings x 4 noise scales x 5 inversion settings grid at N=100. Trial-level
arrays are summarized immediately; no synthetic data are mixed into the
empirical manuscript tables.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import beta, rankdata

ROOT = Path(__file__).resolve().parents[1]
DEPTHS = (5, 10, 20, 40, 80, 160)
RESOLUTIONS = (0.0, 0.1, 0.3, 0.5)
NOISES = (0.25, 0.5, 1.0, 2.0)
INVERSIONS = (0.0, 0.05, 0.1, 0.2, 0.4)
COUNTS = (50, 100, 250)
LAWS = ("gaussian_heteroscedastic", "student_t_heavy_tailed")
METRICS = (
    "d_adj",
    "kendall",
    "spearman",
    "top10_jaccard",
    "stable_inversion",
    "floor_adjusted_kendall",
    "floor_adjusted_spearman",
    "floor_adjusted_jaccard",
)
TRUTH_METRICS = ("d_adj", "kendall", "spearman", "top10_jaccard", "stable_inversion")
MINIMUM_STRICT_SUPPORT = 8
N_REPLICATES = 12
DEFAULT_TRUTH_DRAWS = 1024
# Keep pair-state temporaries bounded when this audit shares a host with
# unrelated scientific jobs. This changes memory use, not the estimator.
# Keep the peak allocation bounded on shared workstations.  Each trial block
# expands to [block, N_REPLICATES, pair_count] in the finite-state estimators;
# block=1 avoids transient multi-megabyte spikes while preserving the exact
# condition/trial protocol.
PAIR_BLOCK = 1
PAIR_STATE_CHUNK = 32


def _latent_pair(
    rng: np.random.Generator,
    *,
    n_items: int,
    inversion_fraction: float,
    law: str,
) -> tuple[np.ndarray, np.ndarray]:
    source = np.sort(rng.normal(size=n_items))
    if law == "student_t_heavy_tailed":
        innovation = rng.standard_t(df=3, size=n_items) / np.sqrt(3.0)
    else:
        innovation = rng.normal(size=n_items)
    rho = float(np.cos(np.pi * inversion_fraction))
    target = rho * source + np.sqrt(max(1.0 - rho * rho, 0.0)) * innovation
    return source.astype(float), target.astype(float)


def _heteroscedastic_scale(values: np.ndarray) -> np.ndarray:
    return 0.35 + 0.65 * (
        np.abs(values) / (np.max(np.abs(values)) + 1e-12)
    )


def _measurement_draws(
    rng: np.random.Generator,
    *,
    latent: np.ndarray,
    depth: int,
    noise_scale: float,
    law: str,
    resolution: float,
    quantization_scale: float,
    n_draws: int,
) -> np.ndarray:
    if law == "student_t_heavy_tailed":
        noise = rng.standard_t(df=3, size=(n_draws, latent.size)) / np.sqrt(3.0)
    else:
        noise = rng.normal(size=(n_draws, latent.size))
    sigma = float(noise_scale) / np.sqrt(float(depth))
    values = latent[None, :] + sigma * noise * _heteroscedastic_scale(latent)[None, :]
    step = float(resolution) * float(quantization_scale)
    if step > 0:
        values = np.rint(values / step) * step
    return values.astype(float)


def _pair_probabilities(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError("values must have shape [replicate, item]")
    pairs = np.column_stack(np.triu_indices(values.shape[1], k=1))
    counts = np.zeros((len(pairs), 3), dtype=np.int64)
    # Stream over measurement draws so truth estimation stays bounded when
    # this audit shares a host with unrelated large scientific processes.
    for start in range(0, values.shape[0], PAIR_STATE_CHUNK):
        difference = (
            values[start : start + PAIR_STATE_CHUNK, pairs[:, 0]]
            - values[start : start + PAIR_STATE_CHUNK, pairs[:, 1]]
        )
        counts[:, 0] += np.sum(difference < 0, axis=0)
        counts[:, 1] += np.sum(difference == 0, axis=0)
        counts[:, 2] += np.sum(difference > 0, axis=0)
    return pairs.astype(np.int64), counts


def _population_truth(
    source_draws: np.ndarray,
    target_draws: np.ndarray,
) -> dict[str, float]:
    _, source_counts = _pair_probabilities(source_draws)
    _, target_counts = _pair_probabilities(target_draws)
    source_prob = source_counts / float(source_draws.shape[0])
    target_prob = target_counts / float(target_draws.shape[0])
    d_adj = 0.5 * float(
        np.mean(np.sum((source_prob - target_prob) ** 2, axis=1))
    )
    source_direction = np.where(
        source_prob[:, 2] > source_prob[:, 0],
        1,
        np.where(source_prob[:, 0] > source_prob[:, 2], -1, 0),
    )
    target_direction = np.where(
        target_prob[:, 2] > target_prob[:, 0],
        1,
        np.where(target_prob[:, 0] > target_prob[:, 2], -1, 0),
    )
    source_strict = np.max(source_prob[:, [0, 2]], axis=1) > 0.5
    target_strict = np.max(target_prob[:, [0, 2]], axis=1) > 0.5
    stable = source_strict & target_strict
    inversion = stable & (source_direction != target_direction)
    population_stable = float(np.mean(inversion[stable])) if np.any(stable) else float("nan")
    rank = _rank_batch(
        source_draws.mean(axis=0, keepdims=True),
        target_draws.mean(axis=0, keepdims=True),
    )
    return {
        "d_adj": d_adj,
        "kendall": float(rank["kendall"][0]),
        "spearman": float(rank["spearman"][0]),
        "top10_jaccard": float(rank["top10_jaccard"][0]),
        "stable_inversion": population_stable,
        "source_pair_tie_rate": float(np.mean(source_prob[:, 1])),
        "target_pair_tie_rate": float(np.mean(target_prob[:, 1])),
        "joint_pair_tie_rate": float(
            np.mean(0.5 * (source_prob[:, 1] + target_prob[:, 1]))
        ),
        "stable_population_coverage": float(np.mean(stable)),
    }


def _rank_batch(left: np.ndarray, right: np.ndarray) -> dict[str, np.ndarray]:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("rank batch arrays must have equal shape [batch, item]")
    n_items = left.shape[1]
    pairs = np.column_stack(np.triu_indices(n_items, k=1))
    left_difference = left[:, pairs[:, 0]] - left[:, pairs[:, 1]]
    right_difference = right[:, pairs[:, 0]] - right[:, pairs[:, 1]]
    kendall = np.mean(left_difference * right_difference < 0, axis=1)
    left_rank = rankdata(left, axis=1, method="average")
    right_rank = rankdata(right, axis=1, method="average")
    spearman = (
        3.0
        * np.mean((left_rank - right_rank) ** 2, axis=1)
        / float(n_items * n_items - 1)
    )
    k = max(1, int(np.floor(n_items * 0.1)))
    left_top = np.argpartition(left, kth=k - 1, axis=1)[:, :k]
    right_top = np.argpartition(right, kth=k - 1, axis=1)[:, :k]
    left_mask = np.zeros(left.shape, dtype=bool)
    right_mask = np.zeros(right.shape, dtype=bool)
    np.put_along_axis(left_mask, left_top, True, axis=1)
    np.put_along_axis(right_mask, right_top, True, axis=1)
    overlap = np.sum(left_mask & right_mask, axis=1)
    jaccard = 1.0 - overlap / float(2 * k - 1)
    return {
        "kendall": kendall.astype(float),
        "spearman": spearman.astype(float),
        "top10_jaccard": jaccard.astype(float),
    }


def _stable_batch(
    left: np.ndarray,
    right: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return stable inversion, coverage, and realized pair-tie rates."""

    batch, n_reps, n_items = left.shape
    pairs = np.column_stack(np.triu_indices(n_items, k=1))
    difference_left = left[:, :, pairs[:, 0]] - left[:, :, pairs[:, 1]]
    difference_right = right[:, :, pairs[:, 0]] - right[:, :, pairs[:, 1]]
    left_counts = np.stack(
        (
            np.sum(difference_left < 0, axis=1),
            np.sum(difference_left == 0, axis=1),
            np.sum(difference_left > 0, axis=1),
        ),
        axis=2,
    )
    right_counts = np.stack(
        (
            np.sum(difference_right < 0, axis=1),
            np.sum(difference_right == 0, axis=1),
            np.sum(difference_right > 0, axis=1),
        ),
        axis=2,
    )
    left_minus = left_counts[:, :, 0]
    left_plus = left_counts[:, :, 2]
    right_minus = right_counts[:, :, 0]
    right_plus = right_counts[:, :, 2]
    left_lower = beta.ppf(0.025, left_plus + 1, left_minus + 1)
    left_upper = beta.ppf(0.975, left_plus + 1, left_minus + 1)
    right_lower = beta.ppf(0.025, right_plus + 1, right_minus + 1)
    right_upper = beta.ppf(0.975, right_plus + 1, right_minus + 1)
    left_stable = (
        (left_minus + left_plus >= MINIMUM_STRICT_SUPPORT)
        & ((left_lower > 0.5) | (left_upper < 0.5))
    )
    right_stable = (
        (right_minus + right_plus >= MINIMUM_STRICT_SUPPORT)
        & ((right_lower > 0.5) | (right_upper < 0.5))
    )
    left_direction = np.where(left_plus > left_minus, 1, np.where(left_minus > left_plus, -1, 0))
    right_direction = np.where(right_plus > right_minus, 1, np.where(right_minus > right_plus, -1, 0))
    stable = left_stable & right_stable
    evaluable = stable & (left_direction != 0) & (right_direction != 0)
    inversion = evaluable & (left_direction != right_direction)
    inversion_rate = np.divide(
        np.sum(inversion, axis=1),
        np.sum(evaluable, axis=1),
        out=np.full(batch, np.nan, dtype=float),
        where=np.sum(evaluable, axis=1) > 0,
    )
    return (
        inversion_rate,
        np.mean(stable, axis=1),
        0.5
        * (
            np.mean(left_counts[:, :, 1], axis=1) / float(n_reps)
            + np.mean(right_counts[:, :, 1], axis=1) / float(n_reps)
        ),
    )


def _d_adj_batch(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    batch, n_reps, n_items = left.shape
    pairs = np.column_stack(np.triu_indices(n_items, k=1))
    left_difference = left[:, :, pairs[:, 0]] - left[:, :, pairs[:, 1]]
    right_difference = right[:, :, pairs[:, 0]] - right[:, :, pairs[:, 1]]
    left_counts = np.stack(
        (
            np.sum(left_difference < 0, axis=1),
            np.sum(left_difference == 0, axis=1),
            np.sum(left_difference > 0, axis=1),
        ),
        axis=2,
    ).astype(float)
    right_counts = np.stack(
        (
            np.sum(right_difference < 0, axis=1),
            np.sum(right_difference == 0, axis=1),
            np.sum(right_difference > 0, axis=1),
        ),
        axis=2,
    ).astype(float)
    left_prob = left_counts / float(n_reps)
    right_prob = right_counts / float(n_reps)
    cross = np.mean(1.0 - np.sum(left_prob * right_prob, axis=2), axis=1)
    left_u = 1.0 - np.sum(left_counts * (left_counts - 1.0), axis=2) / float(
        n_reps * (n_reps - 1)
    )
    right_u = 1.0 - np.sum(right_counts * (right_counts - 1.0), axis=2) / float(
        n_reps * (n_reps - 1)
    )
    d_adj = cross - 0.5 * (np.mean(left_u, axis=1) + np.mean(right_u, axis=1))
    tie_rate = 0.5 * (
        np.mean(left_counts[:, :, 1], axis=1) / float(n_reps)
        + np.mean(right_counts[:, :, 1], axis=1) / float(n_reps)
    )
    return d_adj.astype(float), tie_rate.astype(float)


def _trial_metrics(
    rng: np.random.Generator,
    *,
    source_latent: np.ndarray,
    target_latent: np.ndarray,
    depth: int,
    noise_scale: float,
    law: str,
    resolution: float,
    quantization_scale: float,
    trials: int,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    outputs = {metric: [] for metric in METRICS}
    tie_rates: list[np.ndarray] = []
    stable_coverages: list[np.ndarray] = []
    for start in range(0, trials, PAIR_BLOCK):
        block = min(PAIR_BLOCK, trials - start)
        source = _measurement_draws(
            rng,
            latent=source_latent,
            depth=depth,
            noise_scale=noise_scale,
            law=law,
            resolution=resolution,
            quantization_scale=quantization_scale,
            n_draws=block * N_REPLICATES,
        ).reshape(block, N_REPLICATES, -1)
        target = _measurement_draws(
            rng,
            latent=target_latent,
            depth=depth,
            noise_scale=noise_scale,
            law=law,
            resolution=resolution,
            quantization_scale=quantization_scale,
            n_draws=block * N_REPLICATES,
        ).reshape(block, N_REPLICATES, -1)
        means = _rank_batch(source.mean(axis=1), target.mean(axis=1))
        d_adj, tie_rate = _d_adj_batch(source, target)
        stable, coverage, _ = _stable_batch(source, target)
        floor_s = _rank_batch(
            source[:, : N_REPLICATES // 2].mean(axis=1),
            source[:, N_REPLICATES // 2 :].mean(axis=1),
        )
        floor_t = _rank_batch(
            target[:, : N_REPLICATES // 2].mean(axis=1),
            target[:, N_REPLICATES // 2 :].mean(axis=1),
        )
        outputs["d_adj"].extend(d_adj.tolist())
        outputs["kendall"].extend(means["kendall"].tolist())
        outputs["spearman"].extend(means["spearman"].tolist())
        outputs["top10_jaccard"].extend(means["top10_jaccard"].tolist())
        outputs["stable_inversion"].extend(stable.tolist())
        outputs["floor_adjusted_kendall"].extend(
            (means["kendall"] - 0.5 * (floor_s["kendall"] + floor_t["kendall"])).tolist()
        )
        outputs["floor_adjusted_spearman"].extend(
            (means["spearman"] - 0.5 * (floor_s["spearman"] + floor_t["spearman"])).tolist()
        )
        outputs["floor_adjusted_jaccard"].extend(
            (
                means["top10_jaccard"]
                - 0.5 * (floor_s["top10_jaccard"] + floor_t["top10_jaccard"])
            ).tolist()
        )
        tie_rates.append(tie_rate)
        stable_coverages.append(coverage)
        del source, target
    return (
        {key: np.asarray(value, dtype=float) for key, value in outputs.items()},
        {
            "realized_pair_tie_rate": float(np.mean(np.concatenate(tie_rates))),
            "stable_coverage": float(np.nanmean(np.concatenate(stable_coverages))),
            "stable_tie_rate": float(np.mean(np.concatenate(tie_rates))),
        },
    )


def _condition(task: tuple[dict[str, Any], int, int, int]) -> dict[str, Any]:
    spec, trials, seed, truth_draws = task
    rng = np.random.default_rng(seed)
    source_latent, target_latent = _latent_pair(
        rng,
        n_items=int(spec["n_items"]),
        inversion_fraction=float(spec["inversion_fraction"]),
        law=str(spec["law"]),
    )
    quantization_scale = float(
        np.std(np.concatenate([source_latent, target_latent]))
    )
    truth_rng = np.random.default_rng(seed + 104729)
    truth_source = _measurement_draws(
        truth_rng,
        latent=source_latent,
        depth=int(spec["depth"]),
        noise_scale=float(spec["noise_scale"]),
        law=str(spec["law"]),
        resolution=float(spec["resolution"]),
        quantization_scale=quantization_scale,
        n_draws=truth_draws,
    )
    truth_target = _measurement_draws(
        truth_rng,
        latent=target_latent,
        depth=int(spec["depth"]),
        noise_scale=float(spec["noise_scale"]),
        law=str(spec["law"]),
        resolution=float(spec["resolution"]),
        quantization_scale=quantization_scale,
        n_draws=truth_draws,
    )
    truth = _population_truth(truth_source, truth_target)
    estimates, diagnostics = _trial_metrics(
        rng,
        source_latent=source_latent,
        target_latent=target_latent,
        depth=int(spec["depth"]),
        noise_scale=float(spec["noise_scale"]),
        law=str(spec["law"]),
        resolution=float(spec["resolution"]),
        quantization_scale=quantization_scale,
        trials=trials,
    )
    row: dict[str, Any] = {
        **spec,
        "trials": int(trials),
        "truth_draws": int(truth_draws),
        "quantization_scale": quantization_scale,
        "truth_source_pair_tie_rate": truth["source_pair_tie_rate"],
        "truth_target_pair_tie_rate": truth["target_pair_tie_rate"],
        "truth_pair_tie_rate": truth["joint_pair_tie_rate"],
        "truth_stable_population_coverage": truth["stable_population_coverage"],
        **diagnostics,
    }
    for metric in METRICS:
        values = estimates[metric]
        row[f"{metric}_mean"] = float(np.nanmean(values))
        row[f"{metric}_sd"] = float(np.nanstd(values, ddof=1))
        if metric in TRUTH_METRICS:
            target = truth[metric]
            errors = values - target
            row[f"{metric}_truth"] = float(target)
            row[f"{metric}_mae_to_truth"] = float(np.nanmean(np.abs(errors)))
            row[f"{metric}_bias_to_truth"] = float(np.nanmean(errors))
            row[f"{metric}_rmse_to_truth"] = float(np.sqrt(np.nanmean(errors**2)))
            row[f"{metric}_negative_fraction"] = float(np.mean(values < 0))
    d_adj_values = estimates["d_adj"]
    row["d_adj_empirical_90pct_low"] = float(np.nanquantile(d_adj_values, 0.05))
    row["d_adj_empirical_90pct_high"] = float(np.nanquantile(d_adj_values, 0.95))
    row["d_adj_truth_inside_empirical_90pct"] = bool(
        row["d_adj_empirical_90pct_low"] <= truth["d_adj"] <= row["d_adj_empirical_90pct_high"]
    )
    return row


def _specs(mode: str) -> list[dict[str, Any]]:
    if mode == "primary":
        return [
            {
                "n_items": 100,
                "depth": depth,
                "resolution": resolution,
                "noise_scale": noise,
                "inversion_fraction": inversion,
                "law": law,
            }
            for law in LAWS
            for depth in DEPTHS
            for resolution in RESOLUTIONS
            for noise in NOISES
            for inversion in INVERSIONS
        ]
    if mode == "sensitivity":
        return [
            {
                "n_items": n,
                "depth": depth,
                "resolution": resolution,
                "noise_scale": noise,
                "inversion_fraction": inversion,
                "law": law,
            }
            for law in LAWS
            for n in COUNTS
            for depth in DEPTHS
            for resolution in RESOLUTIONS
            for noise in NOISES
            for inversion in INVERSIONS
        ]
    raise ValueError(f"unknown mode {mode}")


def run(
    *,
    mode: str,
    trials: int,
    workers: int,
    seed: int,
    truth_draws: int,
    shard_index: int = 0,
    shard_count: int = 1,
) -> dict[str, Any]:
    all_specs = _specs(mode)
    if not 1 <= int(shard_count) <= len(all_specs):
        raise ValueError("shard_count must be between 1 and the number of conditions")
    if not 0 <= int(shard_index) < int(shard_count):
        raise ValueError("shard_index must be within shard_count")
    start_index = (len(all_specs) * int(shard_index)) // int(shard_count)
    end_index = (len(all_specs) * (int(shard_index) + 1)) // int(shard_count)
    specs = all_specs[start_index:end_index]
    tasks = [
        (spec, int(trials), int(seed + 7919 * (start_index + index)), int(truth_draws))
        for index, spec in enumerate(specs)
    ]
    if workers <= 1:
        rows = [_condition(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(_condition, tasks, chunksize=1))
    outdir = ROOT / "artifacts/manifests/simulation_v21"
    outdir.mkdir(parents=True, exist_ok=True)
    suffix = "" if int(shard_count) == 1 else f"_part{int(shard_index) + 1:02d}of{int(shard_count):02d}"
    csv_path = outdir / f"finite_measurement_{mode}_v21{suffix}.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    report = {
        "schema_version": 2,
        "status": "executed",
        "mode": mode,
        "trials_per_condition": int(trials),
        "condition_count": len(rows),
        "expected_condition_count": len(all_specs),
        "shard_index": int(shard_index),
        "shard_count": int(shard_count),
        "condition_start_index": int(start_index),
        "condition_end_index_exclusive": int(end_index),
        "workers": int(workers),
        "seed": int(seed),
        "truth_draws_per_condition": int(truth_draws),
        "n_replicates": N_REPLICATES,
        "minimum_strict_support": MINIMUM_STRICT_SUPPORT,
        "estimators": list(METRICS),
        "generative_laws": list(LAWS),
        "resolution_settings": list(RESOLUTIONS),
        "truth_definition": (
            "D_adj truth is one-half the squared L2 distance between the "
            "finite-depth measurement pair-state probabilities, including "
            "the declared quantization resolution. Kendall, Spearman and "
            "top10 Jaccard use the corresponding finite-depth population-mean "
            "ordering targets. Realized pair ties are reported separately."
        ),
        "stable_rule": (
            "Beta(plus+1, minus+1) 95% direction posterior with "
            "minimum strict support >=8; ties remain a third state."
        ),
        "interpretation": (
            "Calibration artifact only. It does not assert comparator "
            "superiority and is not mixed with empirical Frangieh results."
        ),
        "csv": csv_path.relative_to(ROOT).as_posix(),
    }
    json_path = outdir / f"finite_measurement_{mode}_v21{suffix}.json"
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {**report, "json": json_path.relative_to(ROOT).as_posix()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("primary", "sensitivity"), default="primary")
    parser.add_argument("--trials-per-condition", type=int, default=500)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--truth-draws", type=int, default=DEFAULT_TRUTH_DRAWS)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    if args.trials_per_condition < 10:
        raise ValueError("trials-per-condition must be >=10")
    if args.truth_draws < 128:
        raise ValueError("truth-draws must be >=128")
    print(
        json.dumps(
            run(
                mode=args.mode,
                trials=args.trials_per_condition,
                workers=args.workers,
                seed=args.seed,
                truth_draws=args.truth_draws,
                shard_index=args.shard_index,
                shard_count=args.shard_count,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
