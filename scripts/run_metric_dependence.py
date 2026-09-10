"""Cross-fit, tie-aware metric-dependence analysis for transport ordering.

The unit of inference is a pair of perturbation labels. Fifteen seeds are
used for discovery and the disjoint fifteen seeds for validation. Replicates
are averaged within seed before the posterior sign test.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[1]
METRICS = ("delta_cosine", "systema_centroid_accuracy", "absolute_effect_rank_agreement")
STATE_NAMES = ("stable_p<q", "unresolved", "stable_p>q")
_POSTERIOR_CACHE: dict[tuple[int, int], tuple[str, float, float]] = {}


def _posterior_state(values: np.ndarray, tolerance: float = 1e-8) -> tuple[str, float, float, int, int]:
    """Classify risk(p)-risk(q), leaving exact/tolerance ties unresolved."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    lower = int(np.sum(values < -tolerance))
    higher = int(np.sum(values > tolerance))
    ties = int(len(values) - lower - higher)
    if lower + higher == 0:
        return "unresolved", 0.5, 0.5, ties, 0
    cache_key = (lower, higher)
    if cache_key not in _POSTERIOR_CACHE:
        p_lower_cached = float(1.0 - beta.cdf(0.5, 1 + lower, 1 + higher))
        p_higher_cached = float(1.0 - beta.cdf(0.5, 1 + higher, 1 + lower))
        if p_lower_cached >= 0.95 and p_lower_cached > p_higher_cached:
            state_cached = "stable_p<q"
        elif p_higher_cached >= 0.95 and p_higher_cached > p_lower_cached:
            state_cached = "stable_p>q"
        else:
            state_cached = "unresolved"
        _POSTERIOR_CACHE[cache_key] = (state_cached, p_lower_cached, p_higher_cached)
    state, p_lower, p_higher = _POSTERIOR_CACHE[cache_key]
    return state, p_lower, p_higher, ties, lower + higher


def _context_vectors(group: pd.DataFrame, context: str, seeds: list[int]) -> dict[str, np.ndarray]:
    column = "left_risk" if context == str(group["left_target_environment_id"].iloc[0]) else "right_risk"
    reduced = (
        group.loc[group["split_seed"].isin(seeds)]
        .groupby(["perturbation_label", "split_seed"], sort=True)[column]
        .mean().reset_index()
    )
    if reduced.empty:
        return {}
    pivot = reduced.pivot(index="perturbation_label", columns="split_seed", values=column)
    return {str(label): pivot.loc[label].reindex(seeds).to_numpy(dtype=float) for label in pivot.index}


def _pair_records(source: str, target: str, metric: str, group: pd.DataFrame, seeds: list[int]) -> pd.DataFrame:
    source_vectors = _context_vectors(group, source, seeds)
    target_vectors = _context_vectors(group, target, seeds)
    labels = sorted(set(source_vectors) & set(target_vectors))
    rows: list[dict[str, Any]] = []
    for p, q in itertools.combinations(labels, 2):
        source_difference = source_vectors[p][:15] - source_vectors[q][:15]
        target_difference = target_vectors[p][15:] - target_vectors[q][15:]
        src_state, src_p_lower, src_p_higher, src_ties, src_strict = _posterior_state(source_difference)
        tgt_state, tgt_p_lower, tgt_p_higher, tgt_ties, tgt_strict = _posterior_state(target_difference)
        source_margin = float(np.nanmean(np.abs(source_difference))) if np.isfinite(source_difference).any() else float("nan")
        source_mean = float(np.nanmean(source_difference)) if np.isfinite(source_difference).any() else float("nan")
        stable_pair = src_state.startswith("stable_") and tgt_state.startswith("stable_")
        rows.append({
            "source_environment_id": source, "target_environment_id": target, "metric": metric,
            "perturbation_p": p, "perturbation_q": q,
            "source_state_discovery": src_state, "target_state_validation": tgt_state,
            "source_p_lower": src_p_lower, "source_p_higher": src_p_higher,
            "target_p_lower": tgt_p_lower, "target_p_higher": tgt_p_higher,
            "source_tie_count": src_ties, "target_tie_count": tgt_ties,
            "source_strict_count": src_strict, "target_strict_count": tgt_strict,
            "source_pairwise_risk_margin": source_margin, "source_pairwise_risk_difference": source_mean,
            "stable_same": bool(stable_pair and src_state == tgt_state),
            "stable_inversion": bool(stable_pair and src_state != tgt_state),
        })
    return pd.DataFrame(rows)


def run(root: Path = ROOT) -> dict:
    root = root.resolve()
    manifests = root / "artifacts/manifests"
    risk_path = manifests / "reliability_transport_measurement_depth_parts/full/reliability_transport_measurement_depth_risks.csv"
    pair_path = manifests / "reliability_transport_metric_pair_states.csv"
    summary_path = manifests / "reliability_transport_metric_dependence.csv"
    transition_path = manifests / "reliability_transport_metric_transitions.csv"
    report_path = manifests / "reliability_transport_metric_dependence.json"
    if not risk_path.is_file():
        report = {"schema_version": 2, "status": "blocked_missing_full_depth_risks", "input": risk_path.as_posix()}
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return report
    risks = pd.read_csv(risk_path)
    required = {"source_environment_id", "left_target_environment_id", "right_target_environment_id", "metric", "split_seed", "perturbation_label", "left_risk", "right_risk"}
    missing = required.difference(risks.columns)
    if missing:
        raise ValueError(f"risk table missing columns: {sorted(missing)}")
    seeds = sorted(int(v) for v in risks["split_seed"].dropna().unique())
    if len(seeds) < 30:
        raise ValueError(f"metric cross-fit requires 30 disjoint seeds, found {len(seeds)}")
    seeds = seeds[:30]
    pair_frames: list[pd.DataFrame] = []
    for key, group in risks.groupby(["source_environment_id", "left_target_environment_id", "right_target_environment_id", "metric"], sort=True, observed=True):
        source_id, left, right, metric = map(str, key)
        if metric not in METRICS or source_id not in (left, right):
            continue
        for source, target in ((left, right), (right, left)):
            if source != source_id:
                continue
            pair = _pair_records(source, target, metric, group, seeds)
            if not pair.empty:
                pair["left_target_environment_id"] = left
                pair["right_target_environment_id"] = right
                pair["crossfit_split"] = "15_seed_discovery_15_seed_validation"
                pair_frames.append(pair)
    pair_states = pd.concat(pair_frames, ignore_index=True) if pair_frames else pd.DataFrame()
    if pair_states.empty:
        raise RuntimeError("no metric pair states could be constructed")
    pair_states.to_csv(pair_path, index=False)
    summaries: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    for key, group in pair_states.groupby(["source_environment_id", "target_environment_id"], sort=True, observed=True):
        for first, second in itertools.permutations(METRICS, 2):
            a = group.loc[group["metric"].eq(first), ["perturbation_p", "perturbation_q", "source_state_discovery", "target_state_validation", "source_pairwise_risk_margin", "stable_inversion"]].rename(columns={"source_state_discovery": "source_first", "target_state_validation": "target_first", "source_pairwise_risk_margin": "margin_first", "stable_inversion": "inversion_first"})
            b = group.loc[group["metric"].eq(second), ["perturbation_p", "perturbation_q", "source_state_discovery", "target_state_validation", "source_pairwise_risk_margin", "stable_inversion"]].rename(columns={"source_state_discovery": "source_second", "target_state_validation": "target_second", "source_pairwise_risk_margin": "margin_second", "stable_inversion": "inversion_second"})
            merged = a.merge(b, on=["perturbation_p", "perturbation_q"], how="inner")
            if merged.empty:
                continue
            source_stable = merged["source_first"].str.startswith("stable_") & merged["source_second"].str.startswith("stable_")
            target_stable = merged["target_first"].str.startswith("stable_") & merged["target_second"].str.startswith("stable_")
            same = source_stable & target_stable & (merged["source_first"] == merged["source_second"]) & (merged["target_first"] == merged["target_second"])
            inversion = source_stable & target_stable & (merged["source_first"] != merged["source_second"]) & (merged["target_first"] != merged["target_second"])
            inversion_set = set(map(tuple, merged.loc[inversion, ["perturbation_p", "perturbation_q"]].to_numpy()))
            reverse_set = set(map(tuple, merged.loc[merged["inversion_first"] & merged["inversion_second"], ["perturbation_p", "perturbation_q"]].to_numpy()))
            union = inversion_set | reverse_set
            summaries.append({
                "source_environment_id": key[0], "target_environment_id": key[1], "metric_from": first, "metric_to": second,
                "n_perturbation_pairs": int(len(merged)), "stable_same_fraction": float(same.mean()),
                "stable_inversion_fraction": float(inversion.mean()), "stable_to_unresolved_fraction": float((source_stable & ~target_stable).mean()),
                "unresolved_to_stable_fraction": float((~source_stable & target_stable).mean()), "unresolved_both_fraction": float((~source_stable & ~target_stable).mean()),
                "source_pairwise_risk_margin_mean": float(merged["margin_first"].mean()), "source_pairwise_risk_margin_median": float(merged["margin_first"].median()),
                "stable_inversion_set_size": int(len(inversion_set)), "stable_inversion_set_jaccard": float(len(inversion_set & reverse_set) / len(union)) if union else float("nan"),
                "crossfit_split": "15_seed_discovery_15_seed_validation", "state_definition": "tie-aware Beta(1+strict-sign counts) posterior >=0.95; ties excluded; risk p-q", "status": "executed",
            })
            for from_state, to_state in itertools.product(STATE_NAMES, repeat=2):
                mask = merged["source_first"].eq(from_state) & merged["target_second"].eq(to_state)
                transitions.append({
                    "source_environment_id": key[0], "target_environment_id": key[1], "metric_from": first, "metric_to": second,
                    "state_from": from_state, "state_to": to_state, "n_perturbation_pairs": int(len(merged)), "count": int(mask.sum()), "fraction": float(mask.mean()),
                    "state_definition": "tie-aware Beta posterior over source discovery and target validation pairwise risks", "crossfit_split": "15_seed_discovery_15_seed_validation",
                })
    pd.DataFrame(summaries).to_csv(summary_path, index=False)
    pd.DataFrame(transitions).to_csv(transition_path, index=False)
    report = {
        "schema_version": 2, "status": "executed", "ranking_unit": "perturbation-label pair",
        "seed_schedule": {"discovery": seeds[:15], "validation": seeds[15:]},
        "tie_policy": "strict signs only; ties excluded from Beta posterior; stable if posterior direction >=0.95",
        "source_margin_definition": "mean absolute source-context pairwise risk difference, not per-perturbation burden magnitude",
        "outputs": {"pair_states": pair_path.relative_to(root).as_posix(), "pairwise": summary_path.relative_to(root).as_posix(), "transitions": transition_path.relative_to(root).as_posix()},
        "pair_state_rows": int(len(pair_states)), "pairwise_rows": len(summaries), "transition_rows": len(transitions),
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    run(args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
