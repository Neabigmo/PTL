"""Run the reproducible formal-v2 simple and bilinear-linear predictors.

The runner consumes only the condition-level ground truth and the frozen
biological-instance split. It writes local prediction arrays under ``results``
and small, tracked contract/metric manifests under ``artifacts/manifests``.
No outcome is used to construct a deployment feature or an uncertainty score.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.evaluation.metrics import safe_rowwise_cosine, rowwise_spearman
from src.ptl.data.ids import prediction_id
from src.ptl.uncertainty.uq import summarize_ensemble


SPLIT_ID = "ptl_biological_instance_split_v1"
PREDICTOR_VERSION = "formal_v2_exact_linear_20260907"
COMPONENT_PATTERN = re.compile(r"[_+|;]+")
METADATA_COLUMNS = {
    "biological_instance_id", "environment_id", "environment_key", "dataset_id",
    "condition", "condition_field", "perturbation_label", "dose", "timepoint",
    "n_reference_groups", "total_cells", "source_reference_keys", "aggregation_rule",
}


def current_commit(root: Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "working-tree"


def load_panel(root: Path) -> list[str]:
    panel = pd.read_csv(root / "artifacts/manifests/evaluation_gene_space.csv")
    required = {"panel_id", "gene_index", "gene_symbol"}
    if required.difference(panel.columns):
        raise ValueError("evaluation gene panel is missing required columns")
    panel = panel.sort_values("gene_index", kind="stable")
    if panel["gene_symbol"].duplicated().any() or not panel["panel_id"].eq("ptl_context_v2_intersection").all():
        raise ValueError("evaluation gene panel is not the frozen context-v2 panel")
    return panel["gene_symbol"].astype(str).tolist()


def load_registry(root: Path) -> pd.DataFrame:
    registry = pd.read_csv(root / "artifacts/manifests/environment_registry.csv")
    required = {"environment_id", "environment_key", "dataset_id"}
    if required.difference(registry.columns):
        raise ValueError("environment registry is missing canonical identity columns")
    if registry["environment_id"].duplicated().any() or registry["environment_key"].duplicated().any():
        raise ValueError("environment registry IDs and keys must be unique")
    return registry


def load_manifest(root: Path) -> pd.DataFrame:
    manifest = pd.read_csv(root / "artifacts/manifests/biological_instance_registry.csv", dtype=str)
    required = {
        "biological_instance_id", "environment_id", "perturbation_label", "dose", "timepoint",
        "split", "split_seed", "model_seeds", "ground_truth_path",
    }
    if required.difference(manifest.columns):
        raise ValueError("biological-instance manifest is missing formal split columns")
    if manifest["biological_instance_id"].duplicated().any():
        raise ValueError("biological-instance manifest contains duplicate IDs")
    if not manifest["split_seed"].eq("20260907").all() or not manifest["model_seeds"].eq("0,1,2").all():
        raise ValueError("formal split seed/model seed contract changed")
    return manifest


def read_ground_truth(root: Path, path: str, genes: list[str] | None = None) -> pd.DataFrame:
    source = root / path
    schema_columns = pq.read_schema(source).names
    columns = [column for column in METADATA_COLUMNS if column in schema_columns]
    if genes is None:
        genes = [column for column in schema_columns if column not in METADATA_COLUMNS]
    columns.extend(gene for gene in genes if gene not in columns)
    frame = pd.read_parquet(source, columns=columns)
    missing = set(genes).difference(frame.columns)
    if missing:
        raise ValueError(f"{source} is missing frozen evaluation genes: {sorted(missing)[:5]}")
    for column in ("dose", "timepoint"):
        if column not in frame.columns:
            frame[column] = ""
    return frame


def components(label: object) -> list[str]:
    value = str(label).strip()
    if not value or value.casefold() in {"control", "ctrl", "non-targeting", "ntc"}:
        return []
    return [part for part in COMPONENT_PATTERN.split(value) if part]


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    weights = np.where(np.isfinite(weights) & (weights > 0), weights, 1.0)
    return (np.asarray(values, dtype=np.float64) * weights[:, None]).sum(axis=0) / weights.sum()


def fit_matching_mean(
    train_labels: np.ndarray,
    train_values: np.ndarray,
    train_weights: np.ndarray,
    query_labels: np.ndarray,
) -> np.ndarray:
    global_mean = weighted_mean(train_values, train_weights)
    exact: dict[str, list[int]] = {}
    by_component: dict[str, list[int]] = {}
    for index, label in enumerate(train_labels.astype(str)):
        exact.setdefault(label, []).append(index)
        for component in components(label):
            by_component.setdefault(component, []).append(index)
    exact_means = {label: weighted_mean(train_values[indexes], train_weights[indexes]) for label, indexes in exact.items()}
    component_means = {
        label: weighted_mean(train_values[indexes], train_weights[indexes])
        for label, indexes in by_component.items()
    }
    predictions = np.empty((len(query_labels), train_values.shape[1]), dtype=np.float32)
    for row, label in enumerate(query_labels.astype(str)):
        if label in exact_means:
            predictions[row] = exact_means[label]
            continue
        means = [component_means[part] for part in components(label) if part in component_means]
        predictions[row] = np.mean(means, axis=0) if means else global_mean
    return predictions


def fit_ahlmann_eltze_bilinear_ridge(
    train_labels: np.ndarray,
    train_values: np.ndarray,
    query_labels: np.ndarray,
    gene_names: list[str] | None = None,
    alpha: float = 0.1,
    n_components: int = 10,
) -> np.ndarray:
    """Fit the publication-style two-sided ridge predictor in native gene space.

    PCA is fit only on the training response matrix. ``G`` contains the first
    ``n_components`` PCA loading coordinates for every native gene, while ``P``
    is formed by averaging the rows of ``G`` for the genes in each perturbation
    label. The fitted map is

    ``W = (G'G + alpha I)^-1 G'(Y-b)P(P'P + alpha I)^-1``

    and queries are reconstructed as ``G W P_query' + b``. Unknown query
    components contribute no embedding, which yields the training centroid for
    an entirely unsupported query rather than leaking evaluation information.
    """

    train_values = np.asarray(train_values, dtype=np.float64)
    if train_values.ndim != 2 or train_values.shape[0] == 0:
        raise ValueError("train_values must be a non-empty 2D array")
    if len(train_labels) != train_values.shape[0]:
        raise ValueError("train_labels and train_values row counts differ")
    if alpha <= 0 or n_components <= 0:
        raise ValueError("alpha and n_components must be positive")
    if gene_names is None:
        gene_names = [str(index) for index in range(train_values.shape[1])]
    gene_names = [str(gene) for gene in gene_names]
    if len(gene_names) != train_values.shape[1] or len(set(gene_names)) != len(gene_names):
        raise ValueError("gene_names must uniquely identify every native response column")

    intercept = train_values.mean(axis=0)
    centered = train_values - intercept
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    rank = min(int(n_components), vt.shape[0], vt.shape[1])
    if rank == 0:
        return np.tile(intercept, (len(query_labels), 1)).astype(np.float32)
    gene_embedding = vt[:rank].T
    gene_index = {gene: index for index, gene in enumerate(gene_names)}

    def perturbation_embedding(label: object) -> np.ndarray:
        known = [gene_embedding[gene_index[part]] for part in components(label) if part in gene_index]
        return np.mean(known, axis=0) if known else np.zeros(rank, dtype=np.float64)

    train_p = np.stack([perturbation_embedding(label) for label in train_labels])
    query_p = np.stack([perturbation_embedding(label) for label in query_labels])
    identity = np.eye(rank, dtype=np.float64)
    left_gram = gene_embedding.T @ gene_embedding + float(alpha) * identity
    right_gram = train_p.T @ train_p + float(alpha) * identity
    left = np.linalg.solve(left_gram, gene_embedding.T @ centered.T @ train_p)
    try:
        weights = np.linalg.solve(right_gram, left.T).T
    except np.linalg.LinAlgError:
        weights = left @ np.linalg.pinv(right_gram)
    return ((gene_embedding @ weights @ query_p.T).T + intercept).astype(np.float32)


def split_frame(frame: pd.DataFrame, manifest: pd.DataFrame, environment_id: str) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    local_manifest = manifest[manifest["environment_id"].eq(environment_id)].copy()
    key = ["environment_id", "perturbation_label", "dose", "timepoint"]
    local_manifest["dose"] = local_manifest["dose"].fillna("")
    local_manifest["timepoint"] = local_manifest["timepoint"].fillna("")
    frame = frame.copy()
    frame = frame.drop(columns=["biological_instance_id"], errors="ignore")
    for column in ("dose", "timepoint"):
        frame[column] = frame[column].fillna("").astype(str)
    joined = frame.merge(local_manifest[key + ["biological_instance_id", "split"]], on=key, how="inner", validate="one_to_one")
    if len(joined) != len(local_manifest):
        raise ValueError(f"ground truth/manifest mismatch for {environment_id}")
    if joined["biological_instance_id"].duplicated().any():
        raise ValueError(f"duplicate biological instances after join for {environment_id}")
    indices = {split: joined.index[joined["split"].eq(split)].to_numpy() for split in ("train", "validation", "test")}
    if any(len(indexes) == 0 for indexes in indices.values()):
        raise ValueError(f"empty formal split for {environment_id}")
    return joined, indices


def metric_row(true_values: np.ndarray, predicted: np.ndarray, labels: np.ndarray, uq: np.ndarray | None) -> dict[str, float]:
    cosine = safe_rowwise_cosine(true_values, predicted)
    spearman = rowwise_spearman(true_values, predicted)
    true_norm = np.linalg.norm(true_values, axis=1)
    pred_norm = np.linalg.norm(predicted, axis=1)
    top_overlaps = []
    for true_row, pred_row in zip(true_values, predicted):
        k = min(50, true_row.shape[0])
        true_top = set(np.argpartition(np.abs(true_row), -k)[-k:].tolist())
        pred_top = set(np.argpartition(np.abs(pred_row), -k)[-k:].tolist())
        top_overlaps.append(len(true_top & pred_top) / max(1, len(true_top | pred_top)))
    row = {
        "n": float(len(true_values)),
        "mean_fidelity": float(np.mean(cosine)),
        "median_fidelity": float(np.median(cosine)),
        "mean_spearman": float(np.mean(spearman)),
        "mean_rmse": float(np.sqrt(np.mean(np.square(predicted - true_values, dtype=np.float64)))),
        "mean_true_norm": float(np.mean(true_norm)),
        "mean_pred_norm": float(np.mean(pred_norm)),
        "mean_top_deg_overlap_at_50": float(np.mean(top_overlaps)),
    }
    if uq is not None:
        row.update({
            "uq_mean_gene_variance": float(np.mean(uq.mean_gene_variance)),
            "uq_median_gene_variance": float(np.mean(uq.median_gene_variance)),
            "uq_top_effect_variance": float(np.mean(uq.top_effect_variance)),
            "uq_cosine_disagreement": float(np.mean(uq.cosine_disagreement)),
            "uq_effect_norm_variance": float(np.mean(uq.effect_norm_variance)),
        })
    return row


def run(root: Path, output_dir: Path, metrics_path: Path, contract_path: Path) -> dict[str, Any]:
    evaluation_genes = load_panel(root)
    registry = load_registry(root)
    manifest = load_manifest(root)
    commit = current_commit(root)
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict[str, Any]] = []
    contract_rows: list[dict[str, Any]] = []
    training_gene_counts_by_dataset: dict[str, int] = {}
    predictor_names = ("mean_matching", "strong_linear")

    for dataset_id, registry_group in registry.groupby("dataset_id", sort=True):
        source = str(manifest.loc[manifest["environment_id"].isin(registry_group["environment_id"]), "ground_truth_path"].iloc[0])
        ground_truth = read_ground_truth(root, source, genes=None)
        native_genes = [column for column in ground_truth.columns if column not in METADATA_COLUMNS]
        missing_evaluation_genes = set(evaluation_genes).difference(native_genes)
        if missing_evaluation_genes:
            raise ValueError(
                f"{source} is missing frozen evaluation genes: {sorted(missing_evaluation_genes)[:5]}"
            )
        training_gene_counts_by_dataset[str(dataset_id)] = len(native_genes)
        evaluation_indices = [native_genes.index(gene) for gene in evaluation_genes]
        for _, registry_row in registry_group.iterrows():
            environment_id = str(registry_row["environment_id"])
            environment_key = str(registry_row["environment_key"])
            joined, indices = split_frame(ground_truth[ground_truth["environment_id"].eq(environment_id)], manifest, environment_id)
            native_values = joined[native_genes].to_numpy(dtype=np.float32, copy=True)
            y = native_values[:, evaluation_indices]
            labels = joined["perturbation_label"].astype(str).to_numpy()
            weights = pd.to_numeric(joined["total_cells"], errors="coerce").fillna(1).to_numpy(dtype=np.float64)
            seed_values = (0, 1, 2)
            saved: dict[str, dict[str, Any]] = {}
            for predictor in predictor_names:
                validation_members: list[np.ndarray] = []
                test_members: list[np.ndarray] = []
                for seed in seed_values:
                    rng = np.random.default_rng(seed + 1009 * (list(registry["environment_id"]).index(environment_id) + 1))
                    train_indices = indices["train"]
                    bootstrap_indices = rng.choice(train_indices, size=len(train_indices), replace=True)
                    train_labels = labels[bootstrap_indices]
                    train_values = y[bootstrap_indices]
                    native_train_values = native_values[bootstrap_indices]
                    train_weights = weights[bootstrap_indices]
                    query_validation = labels[indices["validation"]]
                    query_test = labels[indices["test"]]
                    if predictor == "mean_matching":
                        validation_prediction = fit_matching_mean(train_labels, train_values, train_weights, query_validation)
                        test_prediction = fit_matching_mean(train_labels, train_values, train_weights, query_test)
                        uq_source = "training_bootstrap_disagreement"
                    else:
                        validation_prediction = fit_ahlmann_eltze_bilinear_ridge(
                            train_labels,
                            native_train_values,
                            query_validation,
                            gene_names=native_genes,
                            alpha=0.1,
                            n_components=10,
                        )[:, evaluation_indices]
                        test_prediction = fit_ahlmann_eltze_bilinear_ridge(
                            train_labels,
                            native_train_values,
                            query_test,
                            gene_names=native_genes,
                            alpha=0.1,
                            n_components=10,
                        )[:, evaluation_indices]
                        uq_source = "training_bootstrap_disagreement"
                    validation_members.append(validation_prediction)
                    test_members.append(test_prediction)
                    metric = metric_row(y[indices["test"]], test_prediction, query_test, None)
                    metric.update({
                        "aggregation_level": "environment_predictor_seed",
                        "environment_id": environment_id,
                        "environment_key": environment_key,
                        "dataset_id": dataset_id,
                        "predictor": predictor,
                        "seed": str(seed),
                        "split_id": SPLIT_ID,
                        "uncertainty_source": uq_source,
                        "training_gene_space_policy": "native_predictor_gene_space",
                        "training_gene_count": len(native_genes),
                        "evaluation_gene_space": "ptl_context_v2_intersection",
                        "evaluation_gene_count": len(evaluation_genes),
                    })
                    metric_rows.append(metric)
                validation_stack = np.stack(validation_members, axis=0)
                test_stack = np.stack(test_members, axis=0)
                validation_uq = summarize_ensemble(validation_stack)
                test_uq = summarize_ensemble(test_stack)
                array_path = output_dir / f"{environment_key}__{predictor}.npz"
                np.savez_compressed(
                    array_path,
                    validation_predictions=validation_stack,
                    test_predictions=test_stack,
                    validation_biological_instance_ids=joined.loc[indices["validation"], "biological_instance_id"].astype(str).to_numpy(),
                    test_biological_instance_ids=joined.loc[indices["test"], "biological_instance_id"].astype(str).to_numpy(),
                    model_seeds=np.asarray(seed_values, dtype=np.int64),
                    genes=np.asarray(evaluation_genes, dtype=str),
                )
                saved[predictor] = {"array_path": array_path, "validation_uq": validation_uq, "test_uq": test_uq}
                aggregate = metric_row(y[indices["test"]], test_stack.mean(axis=0), query_test, test_uq)
                aggregate.update({
                    "aggregation_level": "environment_predictor_ensemble_mean",
                    "environment_id": environment_id,
                    "environment_key": environment_key,
                    "dataset_id": dataset_id,
                    "predictor": predictor,
                    "seed": "ensemble_mean",
                    "split_id": SPLIT_ID,
                    "uncertainty_source": uq_source,
                    "training_gene_space_policy": "native_predictor_gene_space",
                    "training_gene_count": len(native_genes),
                    "evaluation_gene_space": "ptl_context_v2_intersection",
                    "evaluation_gene_count": len(evaluation_genes),
                })
                metric_rows.append(aggregate)
                for split_name, split_indices, stack in (
                    ("validation", indices["validation"], validation_stack),
                    ("test", indices["test"], test_stack),
                ):
                    for member_index, seed in enumerate(seed_values):
                        prediction_array = str(array_path.relative_to(root).as_posix())
                        for biological_id in joined.loc[split_indices, "biological_instance_id"].astype(str):
                            contract_rows.append({
                                "prediction_id": prediction_id(biological_id, predictor, SPLIT_ID, seed),
                                "biological_instance_id": biological_id,
                                "environment_id": environment_id,
                                "environment_key": environment_key,
                                "predictor": predictor,
                                "predictor_version": PREDICTOR_VERSION,
                                "predictor_commit": commit,
                                "predictor_split_id": SPLIT_ID,
                                "seed": seed,
                                "split": split_name,
                                "prediction_array_path": prediction_array,
                                "array_key": f"{split_name}_predictions[{member_index}]",
                                "gene_list_path": "artifacts/manifests/evaluation_gene_space.csv",
                                "training_gene_space_policy": "native_predictor_gene_space",
                                "training_gene_count": len(native_genes),
                                "evaluation_gene_space": "ptl_context_v2_intersection",
                                "evaluation_gene_count": len(evaluation_genes),
                                "uncertainty_source": uq_source,
                                "ensemble_member_seeds": "0,1,2",
                            })

    metric_frame = pd.DataFrame(metric_rows)
    metric_frame = metric_frame.sort_values(["aggregation_level", "environment_id", "predictor", "seed"], kind="stable")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metric_frame.to_csv(metrics_path, index=False)
    contract_frame = pd.DataFrame(contract_rows)
    contract_frame = contract_frame.sort_values(["environment_id", "predictor", "split", "seed", "biological_instance_id"], kind="stable")
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    contract_frame.to_csv(contract_path, index=False)
    summary = {
        "schema_version": 1,
        "benchmark_id": "ptl_context_v2",
        "split_id": SPLIT_ID,
        "predictors": list(predictor_names),
        "environments": int(registry["environment_id"].nunique()),
        "model_seeds": list(seed_values),
        "metric_rows": int(len(metric_frame)),
        "contract_rows": int(len(contract_frame)),
        "predictor_version": PREDICTOR_VERSION,
        "metrics_path": metrics_path.relative_to(root).as_posix(),
        "contract_path": contract_path.relative_to(root).as_posix(),
        "evaluation_gene_space": "ptl_context_v2_intersection",
        "evaluation_gene_count": len(evaluation_genes),
        "training_gene_space_policy": "native_predictor_gene_space",
        "training_gene_counts_by_dataset": training_gene_counts_by_dataset,
        "status": "formal_v2_simple_predictors_executed",
    }
    (metrics_path.parent / "formal_v2_predictor_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/formal_v2/predictors")
    parser.add_argument("--metrics", type=Path, default=ROOT / "artifacts/manifests/formal_v2_predictor_metrics.csv")
    parser.add_argument("--contract", type=Path, default=ROOT / "artifacts/manifests/formal_v2_prediction_index.csv")
    args = parser.parse_args()
    result = run(args.root.resolve(), args.output_dir.resolve(), args.metrics.resolve(), args.contract.resolve())
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
