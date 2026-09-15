from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist, pdist
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


EPS = 1e-8
TOP_K_VALUES = (20, 50, 100)
PATHWAY_MIN_GENES = 3


@dataclass
class DistributionalResult:
    status: str
    reason: str
    sample_size_true: int
    sample_size_pred: int
    embedding_dim: int
    mmd_linear: float
    energy_distance: float
    sliced_wasserstein_128: float


@dataclass
class EmbeddingContext:
    scaler: StandardScaler
    pca: PCA
    true_embedding: np.ndarray
    train_embedding_dim: int


def safe_rowwise_cosine(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    true_norm = np.linalg.norm(y_true, axis=1)
    pred_norm = np.linalg.norm(y_pred, axis=1)
    numerator = np.sum(y_true * y_pred, axis=1)
    denominator = true_norm * pred_norm
    cosine = np.zeros_like(numerator, dtype=np.float64)
    valid = denominator > EPS
    cosine[valid] = numerator[valid] / denominator[valid]
    both_zero = (true_norm <= EPS) & (pred_norm <= EPS)
    cosine[both_zero] = 1.0
    return np.clip(cosine, -1.0, 1.0)


def _rowwise_centered_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    true_mean = y_true.mean(axis=1, keepdims=True)
    pred_mean = y_pred.mean(axis=1, keepdims=True)
    true_centered = y_true - true_mean
    pred_centered = y_pred - pred_mean
    numerator = np.sum(true_centered * pred_centered, axis=1)
    true_norm = np.linalg.norm(true_centered, axis=1)
    pred_norm = np.linalg.norm(pred_centered, axis=1)
    denominator = true_norm * pred_norm
    pearson = np.zeros(y_true.shape[0], dtype=np.float64)
    valid = denominator > EPS
    pearson[valid] = numerator[valid] / denominator[valid]
    both_constant = (true_norm <= EPS) & (pred_norm <= EPS)
    same_constant = both_constant & (np.max(np.abs(y_true - y_pred), axis=1) <= EPS)
    pearson[both_constant] = 0.0
    pearson[same_constant] = 1.0
    return np.clip(pearson, -1.0, 1.0)


def _ordinal_rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1, dtype=np.float64)
    return ranks


def _rowwise_rankdata_average(values: np.ndarray) -> np.ndarray:
    """Tie-aware row ranking equivalent to ``scipy.stats.rankdata``.

    The SciPy axis implementation applies a Python-level operation per row,
    which is unnecessarily costly for the repeated 8,229-gene audit.  This
    vectorized implementation preserves average ranks for ties; the ordering
    within an equal-valued tie group is immaterial.  Non-finite inputs use
    SciPy's established ``nan_policy='propagate'``
    behavior as a conservative fallback.
    """

    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("rowwise rankdata expects a two-dimensional array")
    if not np.isfinite(values).all():
        return stats.rankdata(values, axis=1, method="average", nan_policy="propagate")
    n_rows, n_columns = values.shape
    if n_columns == 0:
        return np.empty(values.shape, dtype=np.float64)
    order = np.argsort(values, axis=1, kind="quicksort")
    sorted_values = np.take_along_axis(values, order, axis=1)
    starts = np.ones((n_rows, n_columns), dtype=bool)
    if n_columns > 1:
        starts[:, 1:] = sorted_values[:, 1:] != sorted_values[:, :-1]
    positions = np.broadcast_to(np.arange(n_columns), (n_rows, n_columns))
    group_starts = np.maximum.accumulate(np.where(starts, positions, 0), axis=1)
    ends = np.empty_like(starts)
    if n_columns > 1:
        ends[:, :-1] = starts[:, 1:]
    ends[:, -1] = True
    group_ends = np.minimum.accumulate(
        np.where(ends, positions, n_columns - 1)[:, ::-1], axis=1
    )[:, ::-1]
    sizes = group_ends - group_starts + 1.0
    sorted_ranks = group_starts + 1.0 + (sizes - 1.0) / 2.0
    ranks = np.empty((n_rows, n_columns), dtype=np.float64)
    np.put_along_axis(ranks, order, sorted_ranks, axis=1)
    return ranks


def rowwise_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    true_constant = np.max(np.abs(y_true - y_true[:, :1]), axis=1) <= EPS
    pred_constant = np.max(np.abs(y_pred - y_pred[:, :1]), axis=1) <= EPS
    both_constant = true_constant & pred_constant
    one_constant = true_constant ^ pred_constant
    true_rank = _rowwise_rankdata_average(y_true)
    pred_rank = _rowwise_rankdata_average(y_pred)
    output = _rowwise_centered_pearson(true_rank, pred_rank)
    output[one_constant] = 0.0
    output[both_constant] = 0.0
    same_constant = both_constant & (np.max(np.abs(y_true - y_pred), axis=1) <= EPS)
    output[same_constant] = 1.0
    return np.clip(output, -1.0, 1.0)


def rowwise_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    diff = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
    return np.sqrt(np.mean(np.square(diff), axis=1))


def rowwise_mae(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    diff = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
    return np.mean(np.abs(diff), axis=1)


def top_k_indices(values: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        return np.array([], dtype=np.int64)
    k = min(k, values.shape[0])
    indices = np.argpartition(-values, k - 1)[:k]
    return indices[np.argsort(-values[indices], kind="mergesort")]


def _dcg_binary(ranked_indices: np.ndarray, relevant: set[int], k: int) -> float:
    if k <= 0:
        return 0.0
    gains = np.array([1.0 if int(idx) in relevant else 0.0 for idx in ranked_indices[:k]], dtype=np.float64)
    if gains.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, gains.size + 2, dtype=np.float64))
    return float(np.sum(gains * discounts))


def _average_precision_binary(ranked_indices: np.ndarray, relevant: set[int], k: int) -> float:
    if not relevant or k <= 0:
        return 0.0
    hits = 0
    precision_sum = 0.0
    limit = min(k, len(ranked_indices))
    for rank, idx in enumerate(ranked_indices[:limit], start=1):
        if int(idx) in relevant:
            hits += 1
            precision_sum += hits / rank
    if hits == 0:
        return 0.0
    return float(precision_sum / min(len(relevant), k))


def compute_rank_metrics_for_row(
    true_row: np.ndarray,
    pred_row: np.ndarray,
    ks: Iterable[int] = TOP_K_VALUES,
) -> dict[str, float]:
    abs_true = np.abs(np.asarray(true_row, dtype=np.float64))
    abs_pred = np.abs(np.asarray(pred_row, dtype=np.float64))
    metrics: dict[str, float] = {}
    for k in ks:
        true_top = top_k_indices(abs_true, k)
        pred_top = top_k_indices(abs_pred, k)
        true_set = set(int(idx) for idx in true_top.tolist())
        pred_set = set(int(idx) for idx in pred_top.tolist())
        intersection = true_set & pred_set
        union = true_set | pred_set
        metrics[f"top_deg_overlap_at_{k}"] = float(len(intersection) / len(union)) if union else 0.0

        if intersection:
            direction_matches = 0
            for idx in intersection:
                if np.sign(pred_row[idx]) == np.sign(true_row[idx]):
                    direction_matches += 1
            metrics[f"deg_direction_consistency_at_{k}"] = float(direction_matches / len(intersection))
        else:
            metrics[f"deg_direction_consistency_at_{k}"] = 0.0

        metrics[f"recall_at_{k}"] = float(len(intersection) / min(k, len(true_set))) if true_set else 0.0
        dcg = _dcg_binary(pred_top, true_set, k)
        idcg = _dcg_binary(true_top, true_set, k)
        metrics[f"ndcg_at_{k}"] = float(dcg / max(idcg, EPS)) if idcg > 0 else 0.0
        metrics[f"map_at_{k}"] = _average_precision_binary(pred_top, true_set, k)
    return metrics


def compute_per_signature_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ks: Iterable[int] = TOP_K_VALUES,
) -> dict[str, np.ndarray]:
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    n_rows = y_true.shape[0]
    outputs: dict[str, np.ndarray] = {
        "cosine_similarity": safe_rowwise_cosine(y_true, y_pred),
        "pearson_r": _rowwise_centered_pearson(y_true, y_pred),
        "spearman_r": rowwise_spearman(y_true, y_pred),
        "rmse": rowwise_rmse(y_true, y_pred),
        "mae": rowwise_mae(y_true, y_pred),
        "delta_norm_true": np.linalg.norm(y_true, axis=1).astype(np.float64),
        "delta_norm_pred": np.linalg.norm(y_pred, axis=1).astype(np.float64),
    }
    rank_storage = {f"top_deg_overlap_at_{k}": np.zeros(n_rows, dtype=np.float64) for k in ks}
    rank_storage.update({f"deg_direction_consistency_at_{k}": np.zeros(n_rows, dtype=np.float64) for k in ks})
    rank_storage.update({f"recall_at_{k}": np.zeros(n_rows, dtype=np.float64) for k in ks})
    rank_storage.update({f"ndcg_at_{k}": np.zeros(n_rows, dtype=np.float64) for k in ks})
    rank_storage.update({f"map_at_{k}": np.zeros(n_rows, dtype=np.float64) for k in ks})
    for idx in range(n_rows):
        row_metrics = compute_rank_metrics_for_row(y_true[idx], y_pred[idx], ks=ks)
        for key, value in row_metrics.items():
            rank_storage[key][idx] = value
    outputs.update(rank_storage)
    return outputs


def pathway_level_profiles(matrix: np.ndarray, genes: list[str], gene_sets: dict[str, set[str]], min_genes: int = PATHWAY_MIN_GENES) -> tuple[np.ndarray, list[str]]:
    """Collapse gene-level deltas to signed mean pathway deltas."""
    matrix = np.asarray(matrix, dtype=np.float64)
    gene_to_idx = {gene: idx for idx, gene in enumerate(genes)}
    columns: list[np.ndarray] = []
    names: list[str] = []
    for pathway, members in sorted(gene_sets.items()):
        indices = [gene_to_idx[gene] for gene in members if gene in gene_to_idx]
        if len(indices) < min_genes:
            continue
        columns.append(matrix[:, indices].mean(axis=1))
        names.append(pathway)
    if not columns:
        return np.zeros((matrix.shape[0], 0), dtype=np.float64), []
    return np.vstack(columns).T.astype(np.float64, copy=False), names


def compute_pathway_fidelity_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    genes: list[str],
    gene_sets: dict[str, set[str]],
    min_genes: int = PATHWAY_MIN_GENES,
) -> dict[str, np.ndarray | float | int]:
    """Compute per-signature pathway cosine and direction consistency."""
    true_pathway, pathway_names = pathway_level_profiles(y_true, genes, gene_sets, min_genes=min_genes)
    pred_pathway, _ = pathway_level_profiles(y_pred, genes, gene_sets, min_genes=min_genes)
    if true_pathway.shape[1] == 0:
        n_rows = np.asarray(y_true).shape[0]
        return {
            "pathway_count": 0,
            "pathway_cosine": np.full(n_rows, np.nan, dtype=np.float64),
            "pathway_direction_consistency": np.full(n_rows, np.nan, dtype=np.float64),
            "mean_pathway_cosine": float("nan"),
            "mean_pathway_direction_consistency": float("nan"),
        }
    pathway_cosine = safe_rowwise_cosine(true_pathway, pred_pathway)
    nonzero = np.abs(true_pathway) > EPS
    same_direction = np.sign(true_pathway) == np.sign(pred_pathway)
    direction = np.divide(
        (same_direction & nonzero).sum(axis=1),
        np.maximum(nonzero.sum(axis=1), 1),
        dtype=np.float64,
    )
    return {
        "pathway_count": len(pathway_names),
        "pathway_cosine": pathway_cosine,
        "pathway_direction_consistency": direction,
        "mean_pathway_cosine": float(np.nanmean(pathway_cosine)),
        "mean_pathway_direction_consistency": float(np.nanmean(direction)),
    }


def compute_perturbation_discrimination_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Iterable[str],
    ks: Iterable[int] = (1, 5, 10),
) -> dict[str, float]:
    """Measure whether predicted signatures retrieve the matching true perturbation."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    label_array = np.asarray(list(labels), dtype=object)
    if y_true.shape[0] == 0 or y_true.shape != y_pred.shape:
        return {f"perturbation_retrieval_top{k}": float("nan") for k in ks}
    similarity = 1.0 - cdist(y_pred, y_true, metric="cosine")
    similarity = np.nan_to_num(similarity, nan=-1.0, posinf=1.0, neginf=-1.0)
    order = np.argsort(-similarity, axis=1, kind="mergesort")
    result: dict[str, float] = {}
    for k in ks:
        top = order[:, : min(k, order.shape[1])]
        hit = np.array([label_array[idxs].tolist().count(label_array[i]) > 0 for i, idxs in enumerate(top)], dtype=bool)
        result[f"perturbation_retrieval_top{k}"] = float(hit.mean()) if hit.size else float("nan")
    return result


def aggregate_metric_frame(frame, prefix: str) -> dict[str, float]:
    result: dict[str, float] = {}
    if frame.empty:
        result[f"mean_cosine_{prefix}"] = float("nan")
        result[f"median_cosine_{prefix}"] = float("nan")
        result[f"mean_pearson_{prefix}"] = float("nan")
        result[f"mean_spearman_{prefix}"] = float("nan")
        result[f"rmse_{prefix}_mean"] = float("nan")
        result[f"mae_{prefix}_mean"] = float("nan")
        for k in TOP_K_VALUES:
            result[f"mean_top_deg_overlap_at_{k}_{prefix}"] = float("nan")
            result[f"mean_deg_direction_consistency_at_{k}_{prefix}"] = float("nan")
            result[f"mean_recall_at_{k}_{prefix}"] = float("nan")
            result[f"mean_ndcg_at_{k}_{prefix}"] = float("nan")
            result[f"mean_map_at_{k}_{prefix}"] = float("nan")
        return result
    result[f"mean_cosine_{prefix}"] = float(frame["cosine_similarity"].mean())
    result[f"median_cosine_{prefix}"] = float(frame["cosine_similarity"].median())
    result[f"mean_pearson_{prefix}"] = float(frame["pearson_r"].mean())
    result[f"mean_spearman_{prefix}"] = float(frame["spearman_r"].mean())
    result[f"rmse_{prefix}_mean"] = float(frame["rmse"].mean())
    result[f"mae_{prefix}_mean"] = float(frame["mae"].mean())
    for k in TOP_K_VALUES:
        result[f"mean_top_deg_overlap_at_{k}_{prefix}"] = float(frame[f"top_deg_overlap_at_{k}"].mean())
        result[f"mean_deg_direction_consistency_at_{k}_{prefix}"] = float(frame[f"deg_direction_consistency_at_{k}"].mean())
        result[f"mean_recall_at_{k}_{prefix}"] = float(frame[f"recall_at_{k}"].mean())
        result[f"mean_ndcg_at_{k}_{prefix}"] = float(frame[f"ndcg_at_{k}"].mean())
        result[f"mean_map_at_{k}_{prefix}"] = float(frame[f"map_at_{k}"].mean())
    return result


def _fit_embedding(train_true: np.ndarray, n_components: int) -> tuple[StandardScaler, PCA]:
    scaler = StandardScaler(with_mean=True, with_std=True)
    train_scaled = scaler.fit_transform(train_true.astype(np.float64))
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=0)
    pca.fit(train_scaled)
    return scaler, pca


def _transform_embedding(scaler: StandardScaler, pca: PCA, matrix: np.ndarray) -> np.ndarray:
    scaled = scaler.transform(matrix.astype(np.float64))
    return pca.transform(scaled).astype(np.float64, copy=False)


def _linear_mmd(x_emb: np.ndarray, y_emb: np.ndarray) -> float:
    mean_diff = x_emb.mean(axis=0) - y_emb.mean(axis=0)
    return float(np.dot(mean_diff, mean_diff))


def _multivariate_energy_distance(x_emb: np.ndarray, y_emb: np.ndarray) -> float:
    cross = cdist(x_emb, y_emb, metric="euclidean")
    x_pair = pdist(x_emb, metric="euclidean")
    y_pair = pdist(y_emb, metric="euclidean")
    e_xy = float(np.mean(cross))
    e_xx = float(np.mean(x_pair)) if x_pair.size else 0.0
    e_yy = float(np.mean(y_pair)) if y_pair.size else 0.0
    return float(max(0.0, (2.0 * e_xy) - e_xx - e_yy))


def _sliced_wasserstein(x_emb: np.ndarray, y_emb: np.ndarray, n_slices: int = 128) -> float:
    rng = np.random.default_rng(0)
    dim = x_emb.shape[1]
    directions = rng.normal(size=(n_slices, dim))
    directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), EPS)
    scores = []
    for direction in directions:
        x_proj = x_emb @ direction
        y_proj = y_emb @ direction
        scores.append(float(stats.wasserstein_distance(x_proj, y_proj)))
    return float(np.mean(scores))


def fit_embedding_context(train_true: np.ndarray, test_true: np.ndarray) -> EmbeddingContext | None:
    train_true = np.asarray(train_true, dtype=np.float32)
    test_true = np.asarray(test_true, dtype=np.float32)
    if train_true.shape[0] < 10 or test_true.shape[0] < 5:
        return None
    n_components = int(min(50, train_true.shape[0] - 1, train_true.shape[1]))
    if n_components < 1:
        return None
    scaler, pca = _fit_embedding(train_true, n_components)
    true_embedding = _transform_embedding(scaler, pca, test_true)
    return EmbeddingContext(
        scaler=scaler,
        pca=pca,
        true_embedding=true_embedding,
        train_embedding_dim=n_components,
    )


def compute_distributional_metrics_from_context(
    context: EmbeddingContext | None,
    test_pred: np.ndarray,
) -> DistributionalResult:
    if context is None:
        return DistributionalResult(
            status="unsupported",
            reason="embedding_context_unavailable",
            sample_size_true=0,
            sample_size_pred=0,
            embedding_dim=0,
            mmd_linear=float("nan"),
            energy_distance=float("nan"),
            sliced_wasserstein_128=float("nan"),
        )
    pred_emb = _transform_embedding(context.scaler, context.pca, test_pred)
    return DistributionalResult(
        status="ready",
        reason="",
        sample_size_true=int(context.true_embedding.shape[0]),
        sample_size_pred=int(pred_emb.shape[0]),
        embedding_dim=context.train_embedding_dim,
        mmd_linear=_linear_mmd(context.true_embedding, pred_emb),
        energy_distance=_multivariate_energy_distance(context.true_embedding, pred_emb),
        sliced_wasserstein_128=_sliced_wasserstein(context.true_embedding, pred_emb, n_slices=128),
    )


def compute_distributional_metrics(
    train_true: np.ndarray,
    test_true: np.ndarray,
    test_pred: np.ndarray,
) -> DistributionalResult:
    train_true = np.asarray(train_true, dtype=np.float32)
    test_true = np.asarray(test_true, dtype=np.float32)
    test_pred = np.asarray(test_pred, dtype=np.float32)
    if train_true.shape[0] < 10:
        return DistributionalResult(
            status="unsupported",
            reason="train_non_control_rows_below_10",
            sample_size_true=int(test_true.shape[0]),
            sample_size_pred=int(test_pred.shape[0]),
            embedding_dim=0,
            mmd_linear=float("nan"),
            energy_distance=float("nan"),
            sliced_wasserstein_128=float("nan"),
        )
    if test_true.shape[0] < 5 or test_pred.shape[0] < 5:
        return DistributionalResult(
            status="unsupported",
            reason="test_non_control_rows_below_5",
            sample_size_true=int(test_true.shape[0]),
            sample_size_pred=int(test_pred.shape[0]),
            embedding_dim=0,
            mmd_linear=float("nan"),
            energy_distance=float("nan"),
            sliced_wasserstein_128=float("nan"),
        )
    context = fit_embedding_context(train_true, test_true)
    if context is None:
        return DistributionalResult(
            status="unsupported",
            reason="embedding_dimension_below_1",
            sample_size_true=int(test_true.shape[0]),
            sample_size_pred=int(test_pred.shape[0]),
            embedding_dim=0,
            mmd_linear=float("nan"),
            energy_distance=float("nan"),
            sliced_wasserstein_128=float("nan"),
        )
    return compute_distributional_metrics_from_context(context, test_pred)
