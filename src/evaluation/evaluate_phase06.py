from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint

from src.baselines.run_baseline import DataRepository, PreparedSplitData, prepare_split_data, read_gene_list, read_json
from src.evaluation.confidence import ConfidenceContext, compute_model_specific_confidence
from src.evaluation.metrics import (
    EPS,
    TOP_K_VALUES,
    aggregate_metric_frame,
    compute_distributional_metrics_from_context,
    compute_per_signature_metrics,
    fit_embedding_context,
)


RESULTS_DIR = ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
EVAL_LOG_DIR = RESULTS_DIR / "logs" / "evaluation"
PHASE_NAME = "Phase 06"
PHASE05_NAME = "Phase 05"
TAU_VALUES = tuple(round(x, 1) for x in np.arange(0.1, 1.01, 0.1))
PRIMARY_METRIC = "mean_cosine_non_control"
STRESS_FAMILIES = {
    "unseen_perturbation_split",
    "unseen_combination_split",
    "low_support_split",
    "dataset_heldout_split",
    "external_holdout",
}


@dataclass
class SelectedRun:
    run_id: str
    model: str
    split_json_path: Path
    split_family: str
    dataset_scope: str
    heldout_target: str
    seed: int
    gene_space_policy: str
    expected_gene_count: int
    output_dir: Path
    latest_status: str
    latest_timestamp: str


@dataclass
class BundleData:
    metadata: pd.DataFrame
    y_true: np.ndarray
    y_pred: np.ndarray
    genes: list[str]
    run_metrics_payload: dict[str, Any]


class PhaseLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {message}"
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        print(line, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 06 evaluator for baseline benchmark outputs.")
    parser.add_argument("--run-manifest", default=str(TABLES_DIR / "baseline_run_matrix.csv"))
    parser.add_argument("--registry", default=str(TABLES_DIR / "experiment_registry.csv"))
    parser.add_argument("--split-audit", default=str(TABLES_DIR / "split_audit.csv"))
    parser.add_argument("--baseline-metrics", default=str(TABLES_DIR / "baseline_metrics_by_split.csv"))
    parser.add_argument("--results-root", default=str(RESULTS_DIR / "baselines"))
    parser.add_argument("--output-dir", default=str(RESULTS_DIR))
    parser.add_argument("--log-file", default=str(EVAL_LOG_DIR / "phase06_metrics.log"))
    return parser.parse_args()


def tau_label(tau: float) -> str:
    return str(tau).replace(".", "p")


def now_local() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M Asia/Shanghai")


def ensure_dirs(output_dir: Path) -> None:
    (output_dir / "tables").mkdir(parents=True, exist_ok=True)
    (output_dir / "logs" / "evaluation").mkdir(parents=True, exist_ok=True)


def load_latest_phase5_registry(registry_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    registry = pd.read_csv(registry_path)
    phase5 = registry[registry["phase"] == PHASE05_NAME].copy()
    phase5["timestamp"] = pd.to_datetime(phase5["timestamp"])
    phase5 = phase5.sort_values(["run_id", "timestamp"])
    latest = phase5.groupby("run_id", as_index=False).tail(1).reset_index(drop=True)
    return phase5, latest


def build_run_selection(manifest_path: Path, registry_path: Path) -> tuple[list[SelectedRun], pd.DataFrame]:
    manifest = pd.read_csv(manifest_path)
    phase5_rows, latest = load_latest_phase5_registry(registry_path)
    latest_lookup = latest.set_index("run_id")

    audit_rows: list[dict[str, Any]] = []
    selected: list[SelectedRun] = []
    for row in manifest.itertuples(index=False):
        latest_row = latest_lookup.loc[row.run_id]
        status = str(latest_row["status"])
        included = status in {"success", "cached"}
        exclusion_reason = "" if included else f"latest_status={status}"
        audit_rows.append(
            {
                "run_id": row.run_id,
                "phase_source": "manifest",
                "split_family": row.split_family,
                "dataset_scope": row.dataset_scope,
                "heldout_target": row.heldout_target,
                "seed": int(row.seed),
                "latest_registry_timestamp": str(latest_row["timestamp"]),
                "latest_registry_status": status,
                "included_in_phase06": bool(included),
                "exclusion_reason": exclusion_reason,
            }
        )
        if included:
            selected.append(
                SelectedRun(
                    run_id=str(row.run_id),
                    model=str(row.model),
                    split_json_path=Path(row.split_json_path),
                    split_family=str(row.split_family),
                    dataset_scope=str(row.dataset_scope),
                    heldout_target="" if pd.isna(row.heldout_target) else str(row.heldout_target),
                    seed=int(row.seed),
                    gene_space_policy=str(row.gene_space_policy),
                    expected_gene_count=int(row.expected_gene_count),
                    output_dir=Path(row.output_dir),
                    latest_status=status,
                    latest_timestamp=str(latest_row["timestamp"]),
                )
            )

    manifest_run_ids = set(manifest["run_id"])
    smoke_rows = latest[~latest["run_id"].isin(manifest_run_ids)].copy()
    for row in smoke_rows.itertuples(index=False):
        audit_rows.append(
            {
                "run_id": row.run_id,
                "phase_source": "registry_extra",
                "split_family": getattr(row, "split_family", ""),
                "dataset_scope": getattr(row, "dataset", ""),
                "heldout_target": getattr(row, "heldout_target", ""),
                "seed": int(getattr(row, "seed", -1)),
                "latest_registry_timestamp": str(row.timestamp),
                "latest_registry_status": str(row.status),
                "included_in_phase06": False,
                "exclusion_reason": "not_in_manifest_smoke_or_extra",
            }
        )
    audit = pd.DataFrame(audit_rows).sort_values(["phase_source", "run_id"]).reset_index(drop=True)
    return selected, audit


def load_bundle(run: SelectedRun) -> BundleData:
    metadata = pd.read_parquet(run.output_dir / "test_metadata.parquet")
    npz = np.load(run.output_dir / "test_predictions.npz")
    genes = read_gene_list(run.output_dir / "genes.txt")
    payload = json.loads((run.output_dir / "run_metrics.json").read_text(encoding="utf-8"))
    return BundleData(
        metadata=metadata,
        y_true=npz["y_true"].astype(np.float32, copy=False),
        y_pred=npz["y_pred"].astype(np.float32, copy=False),
        genes=genes,
        run_metrics_payload=payload,
    )


def resolve_phase05_primary_metric(
    run_id: str,
    phase05_metrics: pd.DataFrame,
    run_metrics_payload: dict[str, Any],
) -> tuple[float, str]:
    metric_name = "mean_cosine_similarity_non_control_test"
    if metric_name in phase05_metrics.columns and run_id in phase05_metrics.index:
        return float(phase05_metrics.loc[run_id, metric_name]), "baseline_metrics_by_split"
    if metric_name in run_metrics_payload:
        return float(run_metrics_payload[metric_name]), "run_metrics_json"
    nested_metrics = run_metrics_payload.get("metrics", {})
    if isinstance(nested_metrics, dict) and metric_name in nested_metrics:
        return float(nested_metrics[metric_name]), "run_metrics_json.metrics"
    raise KeyError(f"Missing {metric_name} for {run_id} in baseline metrics table and run_metrics.json")


def project_arrays_to_gene_space(bundle: BundleData, target_genes: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if bundle.genes == target_genes:
        return bundle.y_true, bundle.y_pred, bundle.genes
    gene_index = {gene: idx for idx, gene in enumerate(bundle.genes)}
    positions = [gene_index[gene] for gene in target_genes]
    return bundle.y_true[:, positions], bundle.y_pred[:, positions], target_genes


def select_anchor_dataset(run: SelectedRun) -> str:
    if run.split_family in {"dataset_heldout_split", "external_holdout"}:
        return run.heldout_target
    return run.dataset_scope


def compute_anchor_stats(
    model: str,
    dataset_id: str,
    seed: int,
    target_genes: list[str],
    manifest_df: pd.DataFrame,
    bundle_cache: dict[str, BundleData],
) -> dict[str, float]:
    candidates = manifest_df[
        (manifest_df["model"] == model)
        & (manifest_df["split_family"] == "random_split")
        & (manifest_df["dataset_scope"] == dataset_id)
        & (manifest_df["seed"] == seed)
    ]
    if candidates.empty:
        return {"anchor_mean_cosine": float("nan"), "anchor_median_cosine": float("nan")}
    run_id = str(candidates.iloc[0]["run_id"])
    bundle = bundle_cache[run_id]
    y_true, y_pred, _ = project_arrays_to_gene_space(bundle, target_genes)
    non_control_mask = ~bundle.metadata["is_control"].to_numpy(dtype=bool)
    if not non_control_mask.any():
        return {"anchor_mean_cosine": float("nan"), "anchor_median_cosine": float("nan")}
    per_sig = compute_per_signature_metrics(y_true[non_control_mask], y_pred[non_control_mask])
    cosine = per_sig["cosine_similarity"]
    return {
        "anchor_mean_cosine": float(np.mean(cosine)),
        "anchor_median_cosine": float(np.median(cosine)),
    }


def compute_selective_metrics(per_signature_df: pd.DataFrame) -> dict[str, float]:
    non_control = per_signature_df[~per_signature_df["is_control"]].copy()
    if non_control.empty:
        result = {"area_under_selective_risk_curve": float("nan"), "area_under_coverage_curve": float("nan")}
        for tau in TAU_VALUES:
            label = tau_label(tau)
            result[f"coverage_at_{label}"] = float("nan")
            result[f"acceptance_rate_at_{label}"] = float("nan")
            result[f"selective_risk_at_{label}"] = float("nan")
            result[f"abstention_gain_at_{label}"] = float("nan")
            result[f"false_transportability_rate_at_{label}"] = float("nan")
        return result

    base_risk = float(non_control["risk"].mean())
    full_coverage = np.array([1.0], dtype=np.float64)
    full_risk = np.array([base_risk], dtype=np.float64)
    result: dict[str, float] = {}
    coverages = [1.0]
    risks = [base_risk]
    transportable_known = non_control["transportable"].notna()

    for tau in TAU_VALUES:
        label = tau_label(tau)
        accepted = non_control[non_control["confidence"] >= tau].copy()
        coverage = float(len(accepted) / len(non_control))
        result[f"coverage_at_{label}"] = coverage
        result[f"acceptance_rate_at_{label}"] = coverage
        if accepted.empty:
            result[f"selective_risk_at_{label}"] = float("nan")
            result[f"abstention_gain_at_{label}"] = float("nan")
            result[f"false_transportability_rate_at_{label}"] = float("nan")
        else:
            selective_risk = float(accepted["risk"].mean())
            result[f"selective_risk_at_{label}"] = selective_risk
            result[f"abstention_gain_at_{label}"] = base_risk - selective_risk
            if transportable_known.any():
                accepted_known = accepted[accepted["transportable"].notna()]
                if accepted_known.empty:
                    result[f"false_transportability_rate_at_{label}"] = float("nan")
                else:
                    false_rate = float((accepted_known["transportable"].astype(int) == 0).mean())
                    result[f"false_transportability_rate_at_{label}"] = false_rate
            else:
                result[f"false_transportability_rate_at_{label}"] = float("nan")
        coverages.append(coverage)
        risks.append(result[f"selective_risk_at_{label}"] if not math.isnan(result[f"selective_risk_at_{label}"]) else base_risk)

    coverage_array = np.array(coverages, dtype=np.float64)
    risk_array = np.array(risks, dtype=np.float64)
    order = np.argsort(coverage_array)
    trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    result["area_under_selective_risk_curve"] = float(trapezoid(risk_array[order], coverage_array[order]))
    result["area_under_coverage_curve"] = float(trapezoid(np.sort(coverage_array), np.linspace(0.0, 1.0, len(coverage_array))))
    return result


def add_wilson_intervals(per_signature_df: pd.DataFrame, aggregate_row: dict[str, Any]) -> None:
    non_control = per_signature_df[~per_signature_df["is_control"]].copy()
    for tau in TAU_VALUES:
        label = tau_label(tau)
        accepted = non_control[non_control["confidence"] >= tau]
        accepted_n = len(accepted)
        if accepted_n == 0:
            aggregate_row[f"false_transportability_rate_ci_low_at_{label}"] = float("nan")
            aggregate_row[f"false_transportability_rate_ci_high_at_{label}"] = float("nan")
            aggregate_row[f"acceptance_rate_ci_low_at_{label}"] = float("nan")
            aggregate_row[f"acceptance_rate_ci_high_at_{label}"] = float("nan")
            continue
        total_n = len(non_control)
        acc_low, acc_high = proportion_confint(count=accepted_n, nobs=total_n, method="wilson")
        aggregate_row[f"acceptance_rate_ci_low_at_{label}"] = float(acc_low)
        aggregate_row[f"acceptance_rate_ci_high_at_{label}"] = float(acc_high)
        accepted_known = accepted[accepted["transportable"].notna()]
        if accepted_known.empty:
            aggregate_row[f"false_transportability_rate_ci_low_at_{label}"] = float("nan")
            aggregate_row[f"false_transportability_rate_ci_high_at_{label}"] = float("nan")
            continue
        false_count = int((accepted_known["transportable"].astype(int) == 0).sum())
        low, high = proportion_confint(count=false_count, nobs=len(accepted_known), method="wilson")
        aggregate_row[f"false_transportability_rate_ci_low_at_{label}"] = float(low)
        aggregate_row[f"false_transportability_rate_ci_high_at_{label}"] = float(high)


def evaluate_single_run(
    run: SelectedRun,
    repository: DataRepository,
    split_cache: dict[str, PreparedSplitData],
    bundle_cache: dict[str, BundleData],
    manifest_df: pd.DataFrame,
    phase05_metrics: pd.DataFrame,
    embedding_cache: dict[str, Any],
    anchor_cache: dict[tuple[str, str, int, tuple[str, ...]], dict[str, float]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    split_key = str(run.split_json_path)
    if split_key not in split_cache:
        split_cache[split_key] = prepare_split_data(repository, run.split_json_path)
    prepared = split_cache[split_key]
    bundle = bundle_cache[run.run_id]
    if bundle.y_true.shape != bundle.y_pred.shape:
        raise ValueError(f"Prediction shape mismatch for {run.run_id}: {bundle.y_true.shape} vs {bundle.y_pred.shape}")
    if bundle.y_true.shape[0] != len(bundle.metadata):
        raise ValueError(f"Metadata row mismatch for {run.run_id}.")
    if bundle.y_true.shape[1] != len(bundle.genes):
        raise ValueError(f"Gene list mismatch for {run.run_id}.")
    if bundle.y_true.shape[1] != run.expected_gene_count:
        raise ValueError(f"Expected gene count mismatch for {run.run_id}.")

    per_sig_metrics = compute_per_signature_metrics(bundle.y_true, bundle.y_pred)
    per_signature_df = bundle.metadata.copy()
    for key, values in per_sig_metrics.items():
        per_signature_df[key] = values
    per_signature_df["risk"] = 1.0 - per_signature_df["cosine_similarity"]

    confidence_context = ConfidenceContext(
        prepared_data=prepared,
        model_details=bundle.run_metrics_payload.get("model_details", {}),
        reference_gene_space=bundle.genes,
    )
    per_signature_df["confidence"] = compute_model_specific_confidence(run.model, confidence_context)
    per_signature_df["confidence"] = np.clip(per_signature_df["confidence"].astype(float), 0.0, 1.0)
    per_signature_df["transportable"] = pd.Series([pd.NA] * len(per_signature_df), dtype="boolean")
    per_signature_df["anchor_signature_median_cosine"] = float("nan")
    per_signature_df["transport_threshold"] = float("nan")

    anchor_mean = float("nan")
    anchor_median = float("nan")
    random_to_stress_drop = float("nan")
    stress_retention = float("nan")
    if run.split_family in STRESS_FAMILIES:
        anchor_dataset = select_anchor_dataset(run)
        anchor_key = (run.model, anchor_dataset, run.seed, tuple(bundle.genes))
        if anchor_key not in anchor_cache:
            anchor_cache[anchor_key] = compute_anchor_stats(run.model, anchor_dataset, run.seed, bundle.genes, manifest_df, bundle_cache)
        anchor_stats = anchor_cache[anchor_key]
        anchor_mean = anchor_stats["anchor_mean_cosine"]
        anchor_median = anchor_stats["anchor_median_cosine"]
        threshold = 0.8 * anchor_median if not math.isnan(anchor_median) else float("nan")
        non_control_mask = ~per_signature_df["is_control"].to_numpy(dtype=bool)
        if not math.isnan(threshold):
            transportable = per_signature_df.loc[non_control_mask, "cosine_similarity"].to_numpy(dtype=float) >= threshold
            per_signature_df.loc[non_control_mask, "transportable"] = transportable
            per_signature_df.loc[non_control_mask, "anchor_signature_median_cosine"] = anchor_median
            per_signature_df.loc[non_control_mask, "transport_threshold"] = threshold

    non_control_frame = per_signature_df[~per_signature_df["is_control"]]
    control_frame = per_signature_df[per_signature_df["is_control"]]
    phase05_primary_metric, phase05_metric_source = resolve_phase05_primary_metric(
        run.run_id,
        phase05_metrics,
        bundle.run_metrics_payload,
    )
    aggregate_row: dict[str, Any] = {
        "run_id": run.run_id,
        "model": run.model,
        "split_family": run.split_family,
        "dataset_scope": run.dataset_scope,
        "heldout_target": run.heldout_target,
        "seed": run.seed,
        "gene_space_policy": run.gene_space_policy,
        "gene_count": len(bundle.genes),
        "latest_registry_status": run.latest_status,
        "latest_registry_timestamp": run.latest_timestamp,
        "n_test_non_control": int((~bundle.metadata["is_control"]).sum()),
        "n_test_control": int(bundle.metadata["is_control"].sum()),
        "phase05_primary_metric": phase05_primary_metric,
        "phase05_metric_source": phase05_metric_source,
    }
    aggregate_row.update(aggregate_metric_frame(non_control_frame, "non_control"))
    aggregate_row.update(aggregate_metric_frame(control_frame, "control"))
    aggregate_row.update(aggregate_metric_frame(per_signature_df, "overall"))
    aggregate_row["phase05_recomputed_primary_delta"] = (
        aggregate_row["mean_cosine_non_control"] - aggregate_row["phase05_primary_metric"]
    )

    test_non_control_mask = ~bundle.metadata["is_control"].to_numpy(dtype=bool)
    if split_key not in embedding_cache:
        train_non_control = prepared.y_train[~prepared.train_frame["is_control"].to_numpy(dtype=bool)]
        embedding_cache[split_key] = fit_embedding_context(
            train_true=train_non_control,
            test_true=bundle.y_true[test_non_control_mask],
        )
    distributional = compute_distributional_metrics_from_context(
        embedding_cache[split_key],
        bundle.y_pred[test_non_control_mask],
    )
    aggregate_row["distributional_status"] = distributional.status
    aggregate_row["distributional_reason"] = distributional.reason
    aggregate_row["distribution_embedding_dim"] = distributional.embedding_dim
    aggregate_row["distribution_sample_size_true"] = distributional.sample_size_true
    aggregate_row["distribution_sample_size_pred"] = distributional.sample_size_pred
    aggregate_row["mmd_linear"] = distributional.mmd_linear
    aggregate_row["energy_distance"] = distributional.energy_distance
    aggregate_row["sliced_wasserstein_128"] = distributional.sliced_wasserstein_128

    selective = compute_selective_metrics(per_signature_df)
    aggregate_row.update(selective)
    add_wilson_intervals(per_signature_df, aggregate_row)

    aggregate_row["anchor_mean_cosine"] = anchor_mean
    aggregate_row["anchor_signature_median_cosine"] = anchor_median
    if run.split_family in STRESS_FAMILIES and not math.isnan(anchor_mean):
        aggregate_row["random_to_stress_drop"] = anchor_mean - aggregate_row["mean_cosine_non_control"]
        aggregate_row["stress_retention"] = aggregate_row["mean_cosine_non_control"] / max(anchor_mean, EPS)
    else:
        aggregate_row["random_to_stress_drop"] = float("nan")
        aggregate_row["stress_retention"] = float("nan")

    per_signature_df["run_id"] = run.run_id
    per_signature_df["gene_count"] = len(bundle.genes)
    return per_signature_df, aggregate_row


def build_pairwise_comparisons(all_metrics: pd.DataFrame) -> pd.DataFrame:
    compare_metrics = ["mean_cosine_non_control", "random_to_stress_drop"]
    context_columns = ["split_family", "dataset_scope", "heldout_target", "seed"]
    rows: list[dict[str, Any]] = []
    for metric_name in compare_metrics:
        metric_frame = all_metrics.dropna(subset=[metric_name]).copy()
        pivot = metric_frame.pivot_table(index=context_columns, columns="model", values=metric_name)
        models = sorted(pivot.columns.tolist())
        for idx_a, model_a in enumerate(models):
            for model_b in models[idx_a + 1 :]:
                pair = pivot[[model_a, model_b]].dropna()
                if pair.empty:
                    continue
                diffs = pair[model_a] - pair[model_b]
                if len(diffs) >= 2 and not np.allclose(diffs.to_numpy(dtype=float), 0.0):
                    _, p_value = wilcoxon(diffs.to_numpy(dtype=float))
                else:
                    p_value = 1.0
                rows.append(
                    {
                        "metric": metric_name,
                        "model_a": model_a,
                        "model_b": model_b,
                        "n_pairs": int(len(diffs)),
                        "mean_diff": float(diffs.mean()),
                        "median_diff": float(diffs.median()),
                        "p_value_raw": float(p_value),
                    }
                )
    if not rows:
        return pd.DataFrame(columns=["metric", "model_a", "model_b", "n_pairs", "mean_diff", "median_diff", "p_value_raw", "p_value_fdr", "reject_fdr_0_05"])
    result = pd.DataFrame(rows)
    reject, pvals_corrected, _, _ = multipletests(result["p_value_raw"].to_numpy(dtype=float), method="fdr_bh")
    result["p_value_fdr"] = pvals_corrected
    result["reject_fdr_0_05"] = reject
    return result.sort_values(["metric", "p_value_fdr", "model_a", "model_b"]).reset_index(drop=True)


def build_family_summary(all_metrics: pd.DataFrame) -> pd.DataFrame:
    grouped = all_metrics.groupby(["model", "split_family"], dropna=False)
    summary = grouped.agg(
        n_runs=("run_id", "count"),
        mean_cosine_non_control=("mean_cosine_non_control", "mean"),
        median_cosine_non_control=("mean_cosine_non_control", "median"),
        mean_random_to_stress_drop=("random_to_stress_drop", "mean"),
        mean_stress_retention=("stress_retention", "mean"),
        mean_area_under_selective_risk_curve=("area_under_selective_risk_curve", "mean"),
        mean_false_transportability_rate_at_0p5=("false_transportability_rate_at_0p5", "mean"),
        mean_mmd_linear=("mmd_linear", "mean"),
        mean_energy_distance=("energy_distance", "mean"),
        mean_sliced_wasserstein_128=("sliced_wasserstein_128", "mean"),
    ).reset_index()
    return summary.sort_values(["split_family", "mean_cosine_non_control"], ascending=[True, False]).reset_index(drop=True)


def run_phase06(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    ensure_dirs(output_dir)
    logger = PhaseLogger(Path(args.log_file))
    logger.log("Starting Phase 06 evaluation.")

    manifest_df = pd.read_csv(args.run_manifest)
    phase05_metrics = pd.read_csv(args.baseline_metrics)
    phase05_metrics = phase05_metrics.set_index("run_id")

    selected_runs, run_selection_audit = build_run_selection(Path(args.run_manifest), Path(args.registry))
    run_selection_audit.to_csv(output_dir / "tables" / "run_selection_audit.csv", index=False)
    logger.log(
        f"Run selection complete: planned={len(manifest_df)}, included={len(selected_runs)}, excluded={len(run_selection_audit) - len(selected_runs)}."
    )

    repository = DataRepository(ROOT / "results" / "tables" / "preprocessing_summary.csv")
    split_cache: dict[str, PreparedSplitData] = {}
    embedding_cache: dict[str, Any] = {}
    anchor_cache: dict[tuple[str, str, int, tuple[str, ...]], dict[str, float]] = {}
    bundle_cache: dict[str, BundleData] = {}
    for run in selected_runs:
        bundle_cache[run.run_id] = load_bundle(run)

    per_signature_frames: list[pd.DataFrame] = []
    aggregate_rows: list[dict[str, Any]] = []
    current_split_key = ""
    for index, run in enumerate(selected_runs, start=1):
        split_key = str(run.split_json_path)
        if split_key != current_split_key:
            split_cache.clear()
            embedding_cache.clear()
            current_split_key = split_key
        logger.log(f"Evaluating run {index}/{len(selected_runs)}: {run.run_id}")
        per_signature_df, aggregate_row = evaluate_single_run(
            run=run,
            repository=repository,
            split_cache=split_cache,
            bundle_cache=bundle_cache,
            manifest_df=manifest_df,
            phase05_metrics=phase05_metrics,
            embedding_cache=embedding_cache,
            anchor_cache=anchor_cache,
        )
        per_signature_frames.append(per_signature_df)
        aggregate_rows.append(aggregate_row)

    per_signature_metrics = pd.concat(per_signature_frames, ignore_index=True)
    all_metrics = pd.DataFrame(aggregate_rows).sort_values(["split_family", "dataset_scope", "heldout_target", "seed", "model"]).reset_index(drop=True)
    family_summary = build_family_summary(all_metrics)
    pairwise = build_pairwise_comparisons(all_metrics)

    per_signature_metrics.to_parquet(output_dir / "tables" / "per_signature_metrics.parquet", index=False)
    all_metrics.to_csv(output_dir / "tables" / "all_metrics.csv", index=False)
    family_summary.to_csv(output_dir / "tables" / "phase06_summary_by_family.csv", index=False)
    pairwise.to_csv(output_dir / "tables" / "model_pairwise_comparisons.csv", index=False)
    logger.log(
        f"Phase 06 outputs written: per_signature_rows={len(per_signature_metrics)}, run_rows={len(all_metrics)}, family_rows={len(family_summary)}, pairwise_rows={len(pairwise)}."
    )


def main() -> None:
    args = parse_args()
    run_phase06(args)


if __name__ == "__main__":
    main()
