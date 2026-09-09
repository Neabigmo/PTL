"""Validate the ordering-identifiability estimands on known synthetic laws.

The simulation keeps the item-pair ordering law categorical, so the target
population quantities are known exactly.  It checks the finite-replicate
plug-in bias, U-statistic centering under a null, recovery under an
alternative, and the fact that a max-floor stress statistic can be negative
even when the identifiable squared-distance divergence is positive.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.ordering_estimands import (
    measurement_corrected_ordering_divergence,
    within_disagreement,
    within_disagreement_u,
)


def _sample_probabilities(
    rng: np.random.Generator,
    population: np.ndarray,
    *,
    n_replicates: int,
    n_pairs: int,
) -> np.ndarray:
    counts = np.asarray([
        rng.multinomial(n_replicates, population)
        for _ in range(n_pairs)
    ])
    return counts.astype(float) / float(n_replicates)


def _trial(
    rng: np.random.Generator,
    population_left: np.ndarray,
    population_right: np.ndarray,
    *,
    n_replicates: int,
    n_pairs: int,
) -> dict[str, float]:
    left = _sample_probabilities(rng, population_left, n_replicates=n_replicates, n_pairs=n_pairs)
    right = _sample_probabilities(rng, population_right, n_replicates=n_replicates, n_pairs=n_pairs)
    cross = 1.0 - np.sum(left * right, axis=1)
    v_left = 1.0 - np.sum(left * left, axis=1)
    v_right = 1.0 - np.sum(right * right, axis=1)
    u_left = 1.0 - np.sum(left * (left * n_replicates - 1.0) / (n_replicates - 1.0), axis=1)
    u_right = 1.0 - np.sum(right * (right * n_replicates - 1.0) / (n_replicates - 1.0), axis=1)
    # Call the public implementation too; this guards against the simulation
    # accidentally validating a second hand-written estimator.
    public_u_left = within_disagreement_u(left, n_replicates=n_replicates)
    public_u_right = within_disagreement_u(right, n_replicates=n_replicates)
    if not np.isclose(public_u_left, float(np.mean(u_left))) or not np.isclose(public_u_right, float(np.mean(u_right))):
        raise AssertionError("synthetic U-statistic implementation mismatch")
    plugin, _ = measurement_corrected_ordering_divergence(left, right)
    return {
        "cross": float(np.mean(cross)),
        "v_floor": float(np.mean(0.5 * (v_left + v_right))),
        "u_floor": float(np.mean(0.5 * (u_left + u_right))),
        "plugin_delta": float(plugin),
        "u_delta": float(np.mean(cross) - 0.5 * (public_u_left + public_u_right)),
        "max_floor_delta": float(np.mean(cross) - max(float(np.mean(u_left)), float(np.mean(u_right)))),
    }


def _render(rows: list[dict[str, float]], path: Path) -> None:
    frame = rows
    labels = [row["scenario"] for row in frame]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    width = 0.18
    for offset, key, label, color in (
        (-1.5 * width, "population_delta", "population identifiable", "#0072B2"),
        (-0.5 * width, "mean_plugin_delta", "plug-in/V", "#D55E00"),
        (0.5 * width, "mean_u_delta", "U-corrected", "#009E73"),
        (1.5 * width, "mean_max_floor_delta", "max-floor stress", "#CC79A7"),
    ):
        ax.bar(x + offset, [row[key] for row in frame], width=width, label=label, color=color)
    ax.axhline(0.0, color="#1f3449", linewidth=0.8)
    ax.set_xticks(x, labels)
    ax.set_ylabel("ordering divergence")
    ax.set_title("Synthetic validation of ordering estimands", loc="left", fontweight="bold")
    ax.legend(frameon=False, ncol=2, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def run(
    root: Path,
    *,
    trials: int = 4000,
    n_replicates: int = 12,
    n_pairs: int = 32,
    seed: int = 20260909,
    with_figure: bool = False,
) -> dict[str, object]:
    if trials < 100 or n_replicates < 2 or n_pairs < 2:
        raise ValueError("synthetic validation needs trials>=100, replicates>=2 and pairs>=2")
    rng = np.random.default_rng(seed)
    scenarios = {
        "finite_null": (np.asarray([0.2, 0.6, 0.2]), np.asarray([0.2, 0.6, 0.2])),
        "known_alternative": (np.asarray([0.7, 0.2, 0.1]), np.asarray([0.65, 0.3, 0.05])),
    }
    population_rows: list[dict[str, float]] = []
    all_rows: list[dict[str, float]] = []
    for scenario, (left, right) in scenarios.items():
        population_delta = 0.5 * float(np.sum((left - right) ** 2))
        population_cross = 1.0 - float(np.dot(left, right))
        population_left = 1.0 - float(np.dot(left, left))
        population_right = 1.0 - float(np.dot(right, right))
        for _ in range(trials):
            result = _trial(rng, left, right, n_replicates=n_replicates, n_pairs=n_pairs)
            result["scenario"] = scenario
            result["population_delta"] = population_delta
            result["population_cross"] = population_cross
            result["population_within_left"] = population_left
            result["population_within_right"] = population_right
            all_rows.append(result)
        subset = [row for row in all_rows if row["scenario"] == scenario]
        population_rows.append({
            "scenario": scenario,
            "population_delta": population_delta,
            "population_cross": population_cross,
            "population_within_left": population_left,
            "population_within_right": population_right,
            "mean_plugin_delta": float(np.mean([row["plugin_delta"] for row in subset])),
            "mean_u_delta": float(np.mean([row["u_delta"] for row in subset])),
            "mean_max_floor_delta": float(np.mean([row["max_floor_delta"] for row in subset])),
            "plugin_bias": float(np.mean([row["plugin_delta"] for row in subset]) - population_delta),
            "u_bias": float(np.mean([row["u_delta"] for row in subset]) - population_delta),
            "trials": int(trials),
            "n_replicates": int(n_replicates),
            "n_pairs": int(n_pairs),
        })
    output_dir = root / "artifacts/manifests"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "ordering_identifiability_synthetic.csv"
    import csv
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(population_rows[0]))
        writer.writeheader()
        writer.writerows(population_rows)
    figure_path = root / "results/figures/iclr_formal/ordering_identifiability_synthetic.png"
    if with_figure:
        _render(population_rows, figure_path)
    payload = {
        "schema_version": 1,
        "status": "synthetic_ordering_estimand_validation_executed",
        "seed": int(seed),
        "trials_per_scenario": int(trials),
        "replicates_per_context": int(n_replicates),
        "pairs_per_trial": int(n_pairs),
        "population_identity": "0.5 * ||pi_left - pi_right||_2^2",
        "finite_sample_null_expectation": "U-corrected delta is centered at zero; plug-in/V delta has positive finite-replicate bias",
        "csv": csv_path.relative_to(root).as_posix(),
        "figure": figure_path.relative_to(root).as_posix() if with_figure else None,
        "rows": population_rows,
    }
    json_path = output_dir / "ordering_identifiability_synthetic.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--trials", type=int, default=4000)
    parser.add_argument("--replicates", type=int, default=12)
    parser.add_argument("--pairs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--with-figure", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.root.resolve(), trials=args.trials, n_replicates=args.replicates, n_pairs=args.pairs, seed=args.seed, with_figure=args.with_figure), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
