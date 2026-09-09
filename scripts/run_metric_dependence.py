"""Quantify metric-dependent burden rankings without adding new predictors."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
METRICS = ("delta_cosine", "systema_centroid_accuracy", "absolute_effect_rank_agreement")


def run(root: Path = ROOT) -> dict:
    root = root.resolve()
    manifests = root / "artifacts/manifests"
    items_path = manifests / "reliability_transport_measurement_depth_items.csv"
    summary_path = manifests / "reliability_transport_metric_dependence.csv"
    transition_path = manifests / "reliability_transport_metric_transitions.csv"
    report_path = manifests / "reliability_transport_metric_dependence.json"
    if not items_path.is_file():
        report = {"schema_version": 1, "status": "blocked_missing_depth_items"}
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return report
    items = pd.read_csv(items_path)
    if "cell_budget_label" not in items:
        full = pd.DataFrame()
    else:
        full = items.loc[items["cell_budget_label"].astype(str).eq("full")].copy()
    if full.empty:
        report = {"schema_version": 1, "status": "blocked_missing_full_depth_rows"}
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return report
    group_columns = ["source_environment_id", "left_target_environment_id", "right_target_environment_id"]
    pair_rows: list[dict] = []
    transition_rows: list[dict] = []
    for key, group in full.groupby(group_columns, sort=True, observed=True):
        item = group.groupby(["metric", "perturbation_label"], sort=True)["identifiable_divergence"].mean().unstack("metric")
        item = item.reindex(columns=METRICS)
        signs = np.sign(item)
        for first, second in itertools.combinations(METRICS, 2):
            valid = item[[first, second]].notna().all(axis=1)
            left = item.loc[valid, first].to_numpy(dtype=float)
            right = item.loc[valid, second].to_numpy(dtype=float)
            k = max(1, int(np.floor(0.20 * len(left))))
            first_top = set(np.argsort(-left, kind="stable")[:k])
            second_top = set(np.argsort(-right, kind="stable")[:k])
            first_sign = signs.loc[valid, first].to_numpy(dtype=float)
            second_sign = signs.loc[valid, second].to_numpy(dtype=float)
            pair_rows.append({
                **dict(zip(group_columns, key)),
                "metric_left": first,
                "metric_right": second,
                "n_labels": int(len(left)),
                "spearman_identifiable_burden": float(pd.Series(left).corr(pd.Series(right), method="spearman")) if len(left) >= 3 else float("nan"),
                "top20_jaccard": float(len(first_top & second_top) / len(first_top | second_top)) if first_top | second_top else float("nan"),
                "mean_absolute_margin_left": float(np.mean(np.abs(left))),
                "mean_absolute_margin_right": float(np.mean(np.abs(right))),
                "sign_agreement": float(np.mean(first_sign == second_sign)),
                "stable_overlap": float(np.mean((first_sign != 0) & (second_sign != 0) & (first_sign == second_sign))),
                "status": "executed",
            })
        for first, second in itertools.product(METRICS, repeat=2):
            if first == second:
                continue
            valid = signs[[first, second]].notna().all(axis=1)
            for first_state, second_state in itertools.product(("negative", "unresolved", "positive"), repeat=2):
                first_mask = signs.loc[valid, first].eq(-1 if first_state == "negative" else 1 if first_state == "positive" else 0)
                second_mask = signs.loc[valid, second].eq(-1 if second_state == "negative" else 1 if second_state == "positive" else 0)
                transition_rows.append({
                    **dict(zip(group_columns, key)),
                    "metric_from": first,
                    "metric_to": second,
                    "state_from": first_state,
                    "state_to": second_state,
                    "n_labels": int(len(signs.loc[valid])),
                    "count": int((first_mask & second_mask).sum()),
                    "fraction": float((first_mask & second_mask).mean()) if valid.any() else float("nan"),
                    "state_definition": "sign of full-depth mean identifiable burden; zero is unresolved",
                })
    pd.DataFrame(pair_rows).to_csv(summary_path, index=False)
    pd.DataFrame(transition_rows).to_csv(transition_path, index=False)
    report = {
        "schema_version": 1,
        "status": "executed",
        "ranking_unit": "full-depth mean per-perturbation identifiable burden",
        "top_k_fraction": 0.20,
        "transition_state_definition": "negative/positive sign; zero unresolved",
        "outputs": {"pairwise": summary_path.relative_to(root).as_posix(), "transitions": transition_path.relative_to(root).as_posix()},
        "pairwise_rows": len(pair_rows),
        "transition_rows": len(transition_rows),
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
