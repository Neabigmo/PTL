"""Quantify Reliability Shift from the frozen formal-v2 reliability outputs.

Reliability Shift is treated as a diagnostic decomposition rather than an
additive identity:

* difficulty shift: environments have different mean realized risk;
* ranking shift: confidence ranks risk differently within environments;
* semantic/calibration shift: an environment-dependent confidence--risk
  mapping predicts held-out biological instances beyond an additive
  environment-difficulty null.

All confidence bins are fit from the existing calibration-side scores.  Risk
is used only for evaluation and for the explanatory cross-fitting/bootstrap;
it never enters a deployment feature, transport descriptor, or gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Iterable
import warnings

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_formal_v2_grouping_loss import assign_bins, artifact_paths, declared_seeds  # noqa: E402
from scripts.run_formal_v2_reliability import _aurc_only  # noqa: E402


METHODS = ("raw_normalized_uq", "u_only_rf", "ptl_rf")
CONTROLLED_COMPARISONS = {
    "frangieh_condition": [
        "frangieh_melanoma_control",
        "frangieh_melanoma_coculture",
        "frangieh_melanoma_ifng",
    ],
    "tian_modality": ["tian_neuron_crispra", "tian_neuron_crispri"],
}


def _finite_correlation(left: Iterable[float], right: Iterable[float], method: str) -> float:
    left_array = np.asarray(list(left), dtype=float)
    right_array = np.asarray(list(right), dtype=float)
    mask = np.isfinite(left_array) & np.isfinite(right_array)
    if int(mask.sum()) < 3 or np.std(left_array[mask]) == 0.0 or np.std(right_array[mask]) == 0.0:
        return float("nan")
    if method == "spearman":
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="An input array is constant")
            return float(spearmanr(left_array[mask], right_array[mask]).statistic)
    return float(kendalltau(left_array[mask], right_array[mask], nan_policy="omit").statistic)


def _normalized_aurc(risk: np.ndarray, confidence: np.ndarray) -> float:
    risk = np.asarray(risk, dtype=float)
    confidence = np.asarray(confidence, dtype=float)
    if len(risk) < 2 or not np.isfinite(risk).all() or not np.isfinite(confidence).all():
        return float("nan")
    mean_risk = float(np.mean(risk))
    if mean_risk == 0.0:
        return float("nan")
    return float(_aurc_only(risk, confidence) / mean_risk)


def _design_matrix(
    environments: np.ndarray,
    confidence_bins: np.ndarray,
    environment_levels: list[str],
    bin_levels: list[int],
    *,
    interaction: bool,
) -> np.ndarray:
    env_index = {value: index for index, value in enumerate(environment_levels)}
    bin_index = {value: index for index, value in enumerate(bin_levels)}
    env_one_hot = np.zeros((len(environments), len(environment_levels)), dtype=float)
    bin_one_hot = np.zeros((len(environments), len(bin_levels)), dtype=float)
    for row, (environment, confidence_bin) in enumerate(zip(environments, confidence_bins)):
        env_one_hot[row, env_index[str(environment)]] = 1.0
        bin_one_hot[row, bin_index[int(confidence_bin)]] = 1.0
    if not interaction:
        return np.column_stack([env_one_hot, bin_one_hot])
    environment_bin = (env_one_hot[:, :, None] * bin_one_hot[:, None, :]).reshape(len(environments), -1)
    return np.column_stack([env_one_hot, environment_bin])


def crossfit_additive_and_interaction(
    frame: pd.DataFrame,
    *,
    n_splits: int = 5,
    group_column: str = "biological_instance_id",
    random_state: int = 0,
) -> pd.DataFrame:
    """Return paired out-of-fold errors for additive and interaction models."""

    required = {"continuous_risk", "environment_id", "confidence_bin", group_column}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Reliability Shift cross-fitting missing columns: {missing}")
    work = frame.loc[:, ["continuous_risk", "environment_id", "confidence_bin", group_column]].copy().reset_index(drop=True)
    work["environment_id"] = work["environment_id"].astype(str)
    work["confidence_bin"] = pd.to_numeric(work["confidence_bin"], errors="raise").astype(int)
    work["continuous_risk"] = pd.to_numeric(work["continuous_risk"], errors="raise").astype(float)
    work[group_column] = work[group_column].astype(str)
    group_values = work[group_column].to_numpy()
    unique_groups = np.unique(group_values)
    if len(unique_groups) < 2:
        raise ValueError("Reliability Shift cross-fitting needs at least two held-out biological groups")
    n_splits = min(int(n_splits), len(unique_groups))
    splitter = GroupKFold(n_splits=n_splits)
    environments = work["environment_id"].to_numpy(dtype=str)
    confidence_bins = work["confidence_bin"].to_numpy(dtype=int)
    risks = work["continuous_risk"].to_numpy(dtype=float)
    environment_levels = sorted(pd.unique(environments).tolist())
    bin_levels = sorted(pd.unique(confidence_bins).tolist())
    output = np.full((len(work), 2), np.nan, dtype=float)
    for train_index, test_index in splitter.split(work, risks, groups=group_values):
        additive = Ridge(alpha=1.0).fit(
            _design_matrix(environments[train_index], confidence_bins[train_index], environment_levels, bin_levels, interaction=False),
            risks[train_index],
        )
        interaction = Ridge(alpha=1.0).fit(
            _design_matrix(environments[train_index], confidence_bins[train_index], environment_levels, bin_levels, interaction=True),
            risks[train_index],
        )
        output[test_index, 0] = (risks[test_index] - additive.predict(
            _design_matrix(environments[test_index], confidence_bins[test_index], environment_levels, bin_levels, interaction=False)
        )) ** 2
        output[test_index, 1] = (risks[test_index] - interaction.predict(
            _design_matrix(environments[test_index], confidence_bins[test_index], environment_levels, bin_levels, interaction=True)
        )) ** 2
    result = work[["environment_id", group_column]].copy()
    result["additive_squared_error"] = output[:, 0]
    result["interaction_squared_error"] = output[:, 1]
    result["interaction_improvement"] = result["additive_squared_error"] - result["interaction_squared_error"]
    result["outcome_used_for_evaluation_only"] = 1
    result["crossfit_group"] = work[group_column].to_numpy()
    result["crossfit_group_name"] = group_column
    result["random_state"] = random_state
    return result


def _hierarchical_ci(oof: pd.DataFrame, *, replicates: int, random_state: int) -> tuple[float, float, float]:
    if replicates <= 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(random_state)
    grouped: list[tuple[str, list[str], dict[str, np.ndarray]]] = []
    for environment, env_frame in oof.groupby("environment_id", sort=True):
        instance_ids = env_frame["crossfit_group"].astype(str).to_numpy()
        unique_instances = list(pd.unique(instance_ids))
        positions = {instance: np.flatnonzero(instance_ids == instance) for instance in unique_instances}
        grouped.append((str(environment), unique_instances, positions))
    values = np.empty(replicates, dtype=float)
    for replicate in range(replicates):
        draws: list[np.ndarray] = []
        for _, instances, positions in grouped:
            sampled_instances = rng.choice(instances, size=len(instances), replace=True)
            draws.extend([positions[str(instance)] for instance in sampled_instances])
        indices = np.concatenate(draws) if draws else np.empty(0, dtype=int)
        sampled = oof.iloc[indices]
        values[replicate] = float(sampled["interaction_improvement"].mean()) if len(sampled) else float("nan")
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)), float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))


def _permutation_p_value(frame: pd.DataFrame, observed: float, *, repeats: int, random_state: int, group_column: str) -> float:
    if repeats <= 0 or not np.isfinite(observed):
        return float("nan")
    rng = np.random.default_rng(random_state)
    null_values: list[float] = []
    for _ in range(repeats):
        shuffled = frame.copy()
        shuffled["confidence_bin"] = shuffled.groupby("environment_id", sort=False)["confidence_bin"].transform(
            lambda values: rng.permutation(values.to_numpy())
        )
        oof = crossfit_additive_and_interaction(shuffled, group_column=group_column, random_state=random_state)
        null_values.append(float(oof["interaction_improvement"].mean()))
    null = np.asarray(null_values, dtype=float)
    return float((1 + np.sum(null >= observed)) / (len(null) + 1))


def assign_calibration_bins(test: pd.DataFrame, calibration: pd.DataFrame, method: str, n_bins: int) -> pd.DataFrame:
    """Assign fixed predictor-relative bins without reading test outcomes."""

    frame = test.copy()
    confidence = frame[method].to_numpy(dtype=float)
    calibration_values = calibration.loc[calibration["method"].eq(method), "confidence"].to_numpy(dtype=float)
    if len(calibration_values) == 0:
        raise ValueError(f"missing calibration confidence for {method}")
    frame["confidence_bin"], edges = assign_bins(confidence, calibration_values, n_bins=n_bins)
    frame["confidence_bin_lower"] = frame["confidence_bin"].map(lambda value: float(edges[min(int(value), len(edges) - 1)]))
    frame["confidence_bin_upper"] = frame["confidence_bin"].map(lambda value: float(edges[min(int(value) + 1, len(edges) - 1)]))
    frame["bin_edges_fit_on"] = "calibration_rows_only"
    frame["outcomes_used_for_bin_construction"] = 0
    return frame


def _decomposition_rows(frame: pd.DataFrame, predictor: str, method: str, split_seed: int, *, bootstrap_replicates: int, permutation_repeats: int) -> tuple[list[dict[str, Any]], dict[str, Any], pd.DataFrame]:
    difficulty_rows: list[dict[str, Any]] = []
    for environment, group in frame.groupby("environment_id", sort=True):
        risk = group["continuous_risk"].to_numpy(dtype=float)
        confidence = group["confidence"].to_numpy(dtype=float)
        difficulty_rows.append({
            "split_seed": split_seed,
            "predictor": predictor,
            "method": method,
            "environment_id": str(environment),
            "environment_key": str(group["environment_key"].iloc[0]) if "environment_key" in group.columns else str(environment),
            "n_test_rows": int(len(group)),
            "difficulty_mean_risk": float(risk.mean()),
            "difficulty_risk_variance": float(np.var(risk, ddof=1)) if len(risk) > 1 else 0.0,
            "ranking_spearman_confidence_vs_negative_risk": _finite_correlation(confidence, -risk, "spearman"),
            "ranking_kendall_confidence_vs_negative_risk": _finite_correlation(confidence, -risk, "kendall"),
            "ranking_normalized_aurc": _normalized_aurc(risk, confidence),
            "evaluation_only": 1,
        })
    oof = crossfit_additive_and_interaction(frame, random_state=split_seed)
    semantic = float(oof["interaction_improvement"].mean())
    mean_bootstrap, ci_lower, ci_upper = _hierarchical_ci(oof, replicates=bootstrap_replicates, random_state=split_seed + 101)
    p_value = _permutation_p_value(frame, semantic, repeats=permutation_repeats, random_state=split_seed + 211, group_column="biological_instance_id")
    difficulty = pd.DataFrame(difficulty_rows)
    semantic_row = {
        "split_seed": split_seed,
        "predictor": predictor,
        "method": method,
        "n_test_rows": int(len(frame)),
        "n_environments": int(frame["environment_id"].nunique()),
        "difficulty_mean_risk": float(frame["continuous_risk"].mean()),
        "difficulty_shift_variance": float(difficulty["difficulty_mean_risk"].var(ddof=0)),
        "difficulty_shift_range": float(difficulty["difficulty_mean_risk"].max() - difficulty["difficulty_mean_risk"].min()),
        "ranking_shift_spearman_variance": float(difficulty["ranking_spearman_confidence_vs_negative_risk"].var(ddof=0)),
        "ranking_shift_spearman_range": float(difficulty["ranking_spearman_confidence_vs_negative_risk"].max() - difficulty["ranking_spearman_confidence_vs_negative_risk"].min()),
        "ranking_shift_kendall_variance": float(difficulty["ranking_kendall_confidence_vs_negative_risk"].var(ddof=0)),
        "ranking_shift_normalized_aurc_range": float(difficulty["ranking_normalized_aurc"].max() - difficulty["ranking_normalized_aurc"].min()),
        "additive_null_mse": float(oof["additive_squared_error"].mean()),
        "environment_interaction_mse": float(oof["interaction_squared_error"].mean()),
        "semantic_shift_improvement": semantic,
        "semantic_shift_bootstrap_mean": mean_bootstrap,
        "semantic_shift_ci_lower": ci_lower,
        "semantic_shift_ci_upper": ci_upper,
        "semantic_shift_permutation_p": p_value,
        "semantic_shift_crossfit_group": "biological_instance_id",
        "confidence_bins_fit_on": "calibration_rows_only",
        "outcomes_used_for_model_evaluation_only": 1,
    }
    return difficulty_rows, semantic_row, oof


def _centered_bin_gap(frame: pd.DataFrame, environments: list[str]) -> float:
    selected = frame.loc[frame["environment_id"].isin(environments)].copy()
    selected["centered_risk"] = selected["continuous_risk"] - selected.groupby("environment_id")["continuous_risk"].transform("mean")
    gaps: list[tuple[float, int]] = []
    for _, group in selected.groupby("confidence_bin", sort=True):
        means = group.groupby("environment_id")["centered_risk"].agg(["mean", "count"])
        available = [value for value in environments if value in means.index]
        if len(available) < 2:
            continue
        gaps.append((float(means.loc[available, "mean"].max() - means.loc[available, "mean"].min()), int(group.shape[0])))
    return float(np.average([value for value, _ in gaps], weights=[weight for _, weight in gaps])) if gaps else float("nan")


def _controlled_rows(frame: pd.DataFrame, predictor: str, method: str, split_seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for comparison, environments in CONTROLLED_COMPARISONS.items():
        for left_index, left in enumerate(environments):
            for right in environments[left_index + 1:]:
                pair = frame.loc[frame["environment_key"].isin([left, right])].copy()
                # Use readable environment keys for this local microscope so
                # the pair definition is auditable; the formal encoded IDs
                # remain unchanged in the saved broad-surface artifacts.
                pair["environment_id"] = pair["environment_key"].astype(str)
                shared = set(pair.loc[pair["environment_id"].eq(left), "perturbation_label"].astype(str)).intersection(
                    set(pair.loc[pair["environment_id"].eq(right), "perturbation_label"].astype(str))
                )
                pair = pair.loc[pair["perturbation_label"].astype(str).isin(shared)].copy()
                if len(shared) < 4 or pair["environment_id"].nunique() != 2:
                    rows.append({
                        "split_seed": split_seed,
                        "comparison": comparison,
                        "left_environment_id": left,
                        "right_environment_id": right,
                        "predictor": predictor,
                        "method": method,
                        "n_shared_perturbations": int(len(shared)),
                        "n_rows": int(len(pair)),
                        "status": "insufficient_shared_perturbations_for_controlled_fit",
                        "matched_perturbation_control": "shared perturbation_label",
                        "outcomes_used_for_evaluation_only": 1,
                    })
                    continue
                pair["centered_risk"] = pair["continuous_risk"] - pair.groupby("environment_id")["continuous_risk"].transform("mean")
                left_frame = pair.loc[pair["environment_id"].eq(left)]
                right_frame = pair.loc[pair["environment_id"].eq(right)]
                left_rank = _finite_correlation(left_frame["confidence"], -left_frame["continuous_risk"], "spearman")
                right_rank = _finite_correlation(right_frame["confidence"], -right_frame["continuous_risk"], "spearman")
                oof = crossfit_additive_and_interaction(pair, group_column="perturbation_label", random_state=split_seed)
                rows.append({
                    "split_seed": split_seed,
                    "comparison": comparison,
                    "left_environment_id": left,
                    "right_environment_id": right,
                    "predictor": predictor,
                    "method": method,
                    "n_shared_perturbations": int(len(shared)),
                    "n_rows": int(len(pair)),
                    "difficulty_left_mean_risk": float(left_frame["continuous_risk"].mean()),
                    "difficulty_right_mean_risk": float(right_frame["continuous_risk"].mean()),
                    "difficulty_mean_risk_gap": float(left_frame["continuous_risk"].mean() - right_frame["continuous_risk"].mean()),
                    "ranking_left_spearman": left_rank,
                    "ranking_right_spearman": right_rank,
                    "ranking_spearman_gap": left_rank - right_rank if np.isfinite(left_rank) and np.isfinite(right_rank) else float("nan"),
                    "centered_confidence_mapping_gap": _centered_bin_gap(pair, [left, right]),
                    "additive_null_mse": float(oof["additive_squared_error"].mean()),
                    "interaction_mse": float(oof["interaction_squared_error"].mean()),
                    "interaction_improvement": float(oof["interaction_improvement"].mean()),
                    "status": "executed",
                    "matched_perturbation_control": "shared perturbation_label",
                    "outcomes_used_for_evaluation_only": 1,
                })
    return rows


def run(root: Path, *, n_bins: int = 10, bootstrap_replicates: int = 1000, permutation_repeats: int = 250, seeds: list[int] | None = None) -> dict[str, Any]:
    selected_seeds = seeds or declared_seeds(root)
    difficulty_rows: list[dict[str, Any]] = []
    semantic_rows: list[dict[str, Any]] = []
    controlled_rows: list[dict[str, Any]] = []
    for seed in selected_seeds:
        prediction_path, calibration_path = artifact_paths(root, seed)
        test = pd.read_csv(prediction_path)
        calibration = pd.read_csv(calibration_path)
        test = test.loc[test["scenario"].eq("in_domain")].copy()
        calibration = calibration.loc[calibration["scenario"].eq("in_domain")].copy()
        for predictor in sorted(test["predictor"].astype(str).unique()):
            predictor_frame = test.loc[test["predictor"].astype(str).eq(predictor)].copy()
            for method in METHODS:
                score_frame = assign_calibration_bins(
                    predictor_frame,
                    calibration.loc[calibration["predictor"].astype(str).eq(predictor)],
                    method,
                    n_bins,
                )
                score_frame["confidence"] = pd.to_numeric(score_frame[method], errors="raise")
                difficulty, semantic, _ = _decomposition_rows(
                    score_frame,
                    predictor,
                    method,
                    seed,
                    bootstrap_replicates=bootstrap_replicates,
                    permutation_repeats=permutation_repeats,
                )
                difficulty_rows.extend(difficulty)
                semantic_rows.append(semantic)
                controlled_rows.extend(_controlled_rows(score_frame, predictor, method, seed))

    manifest_dir = root / "artifacts/manifests"
    source_dir = root / "artifacts/source_data"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)
    difficulty_path = manifest_dir / "formal_v2_reliability_shift_decomposition.csv"
    semantic_path = manifest_dir / "formal_v2_reliability_shift_semantic.csv"
    controlled_path = manifest_dir / "formal_v2_controlled_shift.csv"
    decomposition_json_path = manifest_dir / "formal_v2_reliability_shift_decomposition.json"
    controlled_json_path = manifest_dir / "formal_v2_controlled_shift.json"
    summary_path = manifest_dir / "formal_v2_reliability_shift_summary.json"
    difficulty_frame = pd.DataFrame(difficulty_rows).sort_values(["split_seed", "predictor", "method", "environment_id"], kind="stable")
    semantic_frame = pd.DataFrame(semantic_rows).sort_values(["split_seed", "predictor", "method"], kind="stable")
    controlled_frame = pd.DataFrame(controlled_rows).sort_values(["split_seed", "comparison", "predictor", "method"], kind="stable")
    difficulty_frame.to_csv(difficulty_path, index=False)
    semantic_frame.to_csv(semantic_path, index=False)
    controlled_frame.to_csv(controlled_path, index=False)
    decomposition_json_path.write_text(
        json.dumps({"schema_version": 1, "rows": difficulty_frame.to_dict("records")}, indent=2, ensure_ascii=False, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    controlled_json_path.write_text(
        json.dumps({"schema_version": 1, "rows": controlled_frame.to_dict("records")}, indent=2, ensure_ascii=False, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    macro = semantic_frame.groupby(["predictor", "method"], as_index=False).agg(
        n_splits=("split_seed", "nunique"),
        mean_difficulty_shift_variance=("difficulty_shift_variance", "mean"),
        mean_difficulty_shift_range=("difficulty_shift_range", "mean"),
        mean_ranking_shift_spearman_range=("ranking_shift_spearman_range", "mean"),
        mean_ranking_shift_normalized_aurc_range=("ranking_shift_normalized_aurc_range", "mean"),
        mean_additive_null_mse=("additive_null_mse", "mean"),
        mean_environment_interaction_mse=("environment_interaction_mse", "mean"),
        mean_semantic_shift_improvement=("semantic_shift_improvement", "mean"),
        semantic_shift_improvement_sd=("semantic_shift_improvement", "std"),
        semantic_shift_positive_split_fraction=("semantic_shift_improvement", lambda values: float(np.mean(np.asarray(values) > 0.0))),
    )
    payload = {
        "schema_version": 1,
        "benchmark_id": "ptl_context_v2",
        "concept": "Reliability Shift = difficulty shift + ranking shift + semantic/calibration shift (conceptual decomposition, not an additive identity)",
        "split_seeds": selected_seeds,
        "methods": list(METHODS),
        "confidence_binning": "predictor-relative quantile bins fit on calibration-side scores and applied unchanged to test rows",
        "semantic_null": "R = alpha_environment + h(confidence_bin)",
        "semantic_alternative": "R = alpha_environment + h_environment(confidence_bin)",
        "semantic_evaluation": "Ridge models evaluated by biological-instance/group cross-fitting; risk is evaluation-only",
        "bootstrap": "hierarchical environment then biological-instance resampling of paired out-of-fold errors",
        "permutation": "confidence bins permuted within environment to break environment-by-confidence interaction",
        "controlled_comparisons": "Frangieh condition pairs and Tian CRISPRa/CRISPRi, restricted to shared perturbation labels",
        "difficulty_path": difficulty_path.relative_to(root).as_posix(),
        "semantic_path": semantic_path.relative_to(root).as_posix(),
        "controlled_path": controlled_path.relative_to(root).as_posix(),
        "difficulty_json_path": decomposition_json_path.relative_to(root).as_posix(),
        "controlled_json_path": controlled_json_path.relative_to(root).as_posix(),
        "macro": macro.to_dict("records"),
        "status": "formal_v2_reliability_shift_executed",
    }
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--permutation-repeats", type=int, default=250)
    parser.add_argument("--split-seed", action="append", type=int, dest="seeds", default=None)
    args = parser.parse_args()
    payload = run(
        args.root.resolve(),
        n_bins=args.n_bins,
        bootstrap_replicates=args.bootstrap_replicates,
        permutation_repeats=args.permutation_repeats,
        seeds=args.seeds,
    )
    print(json.dumps({"status": payload["status"], "split_seeds": payload["split_seeds"], "macro": payload["macro"]}, indent=2, ensure_ascii=False, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
