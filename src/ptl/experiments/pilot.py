"""Run the first leakage-safe PTL-v2 pilot from local prediction artifacts.

The pilot deliberately replays existing three-seed baseline prediction arrays
instead of silently claiming that unavailable GEARS outputs exist.  It creates
new v2 contracts, uncertainty summaries, grouped folds, and reliability
comparisons without modifying raw or processed matrices.
"""

from __future__ import annotations

import json
import os
import re
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GroupKFold

from ptl.data.ids import biological_instance_id, environment_id, perturbation_group_id, prediction_id
from ptl.data.contracts import validate_feature_columns
from ptl.evaluation.bootstrap import (
    paired_hierarchical_bootstrap,
    summarize_bootstrap_ci,
    summarize_paired_deltas,
)
from ptl.evaluation.metrics import evaluate_scores
from ptl.reliability.calibration import (
    ContextualCalibrator,
    fit_isotonic,
    fit_logistic,
    fit_platt,
    predict_isotonic,
    predict_logistic,
    predict_platt,
)
from ptl.uncertainty.uq import UQNormalizer, summarize_ensemble


PILOT_DATASETS = {
    "NormanWeissman2019_filtered": {
        "cell_context": "K562",
        "perturbation_modality": "metadata_pending",
        "readout_modality": "RNA",
        "platform": "processed_signature_table",
        "modality": "RNA",
        "condition": "baseline",
    },
    # This is the existing processed K562 surface.  GWPS is the v2 target
    # surface, but it has no compatible processed prediction arrays yet.
    "ReplogleWeissman2022_K562_essential": {
        "cell_context": "K562",
        "perturbation_modality": "metadata_pending",
        "readout_modality": "RNA",
        "platform": "processed_signature_table",
        "modality": "RNA",
        "condition": "essential_legacy_pilot",
    },
    "ReplogleWeissman2022_rpe1": {
        "cell_context": "RPE1",
        "perturbation_modality": "metadata_pending",
        "readout_modality": "RNA",
        "platform": "processed_signature_table",
        "modality": "RNA",
        "condition": "baseline",
    },
}
PILOT_SPLITS = ("random_split", "dataset_heldout_split", "unseen_perturbation_split")
MODEL_DIRS = {
    "mean_global": "global_delta_baseline",
    "mean_matching": "perturbation_mean_delta_baseline",
    "ridge": "ridge_regression_baseline",
}
OOF_MODEL_NAMES = {
    "mean_global": "global_delta_baseline",
    "mean_matching": "perturbation_mean_delta_baseline",
    "ridge": "ridge_regression_baseline",
}
UQ_COLUMNS = [
    "uq_mean_gene_variance",
    "uq_median_gene_variance",
    "uq_top_effect_variance",
    "uq_cosine_disagreement",
    "uq_effect_norm_variance",
]
P_COLUMNS = [
    "prediction_norm",
    "prediction_sparsity",
    "prediction_concentration",
    "prediction_manifold_distance",
]
S_COLUMNS = ["support_cells", "support_signatures", "reference_key_overlap"]
N_COLUMNS = [
    "perturbation_seen_fraction",
    "component_seen_fraction",
    "combination_novelty",
    "perturbation_novelty",
]
CONTEXT_CATEGORICAL_COLUMNS = [
    "cell_context",
    "perturbation_modality",
    "readout_modality",
    "condition",
    "platform",
]
C_COLUMNS = CONTEXT_CATEGORICAL_COLUMNS + ["batch_distance"]
P_BASE_COLUMNS = [column for column in P_COLUMNS if column != "prediction_manifold_distance"]
CONTEXT_NUMERIC_COLUMNS = P_BASE_COLUMNS + S_COLUMNS + N_COLUMNS + ["batch_distance"]
NO_CONTEXT_NUMERIC_COLUMNS = P_BASE_COLUMNS + S_COLUMNS + N_COLUMNS
CONTEXT_COLUMNS = CONTEXT_NUMERIC_COLUMNS + CONTEXT_CATEGORICAL_COLUMNS
ALL_NUMERIC_COLUMNS = UQ_COLUMNS + CONTEXT_NUMERIC_COLUMNS
SIGNATURE_METADATA_COLUMNS = [
    "signature_id", "reference_key", "perturbation_label", "is_control", "n_cells", "batch"
]


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _load_oof_labels(root: Path) -> pd.DataFrame:
    path = root / "results/tables/ptl_oof_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    chunks: list[pd.DataFrame] = []
    usecols = ["signature_id", "model", "split_family", "y_true", "target_risk"]
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=100_000):
        chunk["dataset_id"] = chunk["signature_id"].str.partition("__")[0]
        chunk = chunk[
            chunk["dataset_id"].isin(PILOT_DATASETS)
            & (chunk["split_family"] == "dataset_heldout_split")
            & chunk["model"].isin(OOF_MODEL_NAMES.values())
        ]
        if not chunk.empty:
            chunks.append(chunk)
    if not chunks:
        raise ValueError("no compatible legacy OOF labels found for the pilot")
    labels = pd.concat(chunks, ignore_index=True)
    labels["predictor"] = labels["model"].map({v: k for k, v in OOF_MODEL_NAMES.items()})
    return (
        labels.groupby(["dataset_id", "predictor", "signature_id"], as_index=False)
        .agg(legacy_transportability_label=("y_true", "mean"), legacy_target_risk=("target_risk", "mean"))
    )


def _find_run(root: Path, model_dir: str, dataset_id: str, split_family: str, seed: int) -> Path:
    base = root / "results/baselines" / model_dir
    if split_family == "dataset_heldout_split":
        pattern = f"{split_family}__all_datasets__holdout-{dataset_id}__seed{seed}__signature"
    else:
        pattern = f"{split_family}__{dataset_id}__seed{seed}__signature"
    candidates = sorted(
        base.glob(pattern)
    )
    if len(candidates) != 1:
        raise FileNotFoundError(f"expected one seed directory, found {len(candidates)}: {base}")
    return candidates[0]


def _split_path(root: Path, dataset_id: str, split_family: str, seed: int) -> Path:
    if split_family == "dataset_heldout_split":
        name = f"{split_family}__all_datasets__holdout-{dataset_id}__seed{seed}__signature.json"
    else:
        name = f"{split_family}__{dataset_id}__seed{seed}__signature.json"
    path = root / "data/processed/splits" / name
    if not path.exists():
        raise FileNotFoundError(path)
    return path


@lru_cache(maxsize=None)
def _load_training_metadata(root_text: str, dataset_id: str) -> pd.DataFrame:
    root = Path(root_text)
    path = root / "data/processed" / f"{dataset_id}_delta_signatures.parquet"
    columns = [column for column in SIGNATURE_METADATA_COLUMNS if column != "signature_id"]
    available = set(pq.ParquetFile(path).schema.names)
    requested = [column for column in ["signature_id", *columns] if column in available]
    metadata = pd.read_parquet(path, columns=requested).copy()
    for column in ["signature_id", *columns]:
        if column not in metadata:
            metadata[column] = np.nan
    metadata = metadata[["signature_id", *columns]]
    metadata["signature_id"] = metadata["signature_id"].astype(str)
    return metadata


@lru_cache(maxsize=2)
def _load_all_training_metadata(root_text: str) -> pd.DataFrame:
    root = Path(root_text)
    tables = []
    for path in sorted((root / "data/processed").glob("*_delta_signatures.parquet")):
        dataset_id = path.name.removesuffix("_delta_signatures.parquet")
        tables.append(_load_training_metadata(root_text, dataset_id))
    if not tables:
        raise FileNotFoundError(f"no processed delta-signature metadata under {root / 'data/processed'}")
    return pd.concat(tables, ignore_index=True)


def _components(label: object) -> list[str]:
    text = str(label)
    return [part.strip() for part in re.split(r"[+,;|]", text) if part.strip()]


def _response_sketch(values: np.ndarray, n_blocks: int = 64) -> np.ndarray:
    """Create a compact response-space representation for fold-local PCA."""

    values = np.asarray(values, dtype=np.float32).ravel()
    if not len(values):
        return np.zeros(n_blocks, dtype=np.float32)
    return np.asarray(
        [chunk.mean(dtype=np.float64) for chunk in np.array_split(values, n_blocks)],
        dtype=np.float32,
    )


def _training_features(
    root: Path,
    dataset_id: str,
    split_family: str,
    seed: int,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    train_id_sets = []
    for split_seed in (0, 1, 2):
        split = json.loads(
            _split_path(root, dataset_id, split_family, split_seed).read_text(encoding="utf-8")
        )
        train_ids = split["train_ids"]
        if isinstance(train_ids, dict):
            train_ids = [value for values in train_ids.values() for value in values]
        train_id_sets.append({str(value) for value in train_ids})
    train_ids = set.intersection(*train_id_sets)
    train = (
        _load_all_training_metadata(str(root))
        if split_family == "dataset_heldout_split"
        else _load_training_metadata(str(root), dataset_id)
    )
    train = train[train["signature_id"].isin(train_ids)].copy()
    if train.empty:
        raise ValueError(f"split has no training signatures: {dataset_id}/{split_family}/seed{seed}")

    target = metadata.copy()
    target["perturbation_label"] = target["perturbation_label"].fillna(target["signature_id"])
    train["perturbation_label"] = train["perturbation_label"].fillna(train["signature_id"])
    perturbation_counts = train.groupby("perturbation_label")["signature_id"].nunique()
    cell_support = train.groupby("perturbation_label")["n_cells"].sum(min_count=1).fillna(0.0)
    train_references = set(train["reference_key"].dropna().astype(str))
    train_batches = set(train["batch"].dropna().astype(str))
    train_combinations = set(train["perturbation_label"].astype(str))
    train_components = {
        component
        for label in train["perturbation_label"]
        for component in _components(label)
    }

    labels = target["perturbation_label"].astype(str)
    counts = labels.map(perturbation_counts).fillna(0.0).to_numpy(dtype=float)
    cells = labels.map(cell_support).fillna(0.0).to_numpy(dtype=float)
    components_seen = np.asarray(
        [
            np.mean([component in train_components for component in _components(label)])
            if _components(label)
            else 0.0
            for label in labels
        ],
        dtype=float,
    )
    reference = target["reference_key"].astype(str)
    batch = target["batch"].astype(str)
    max_support = float(counts.max()) if counts.size else 0.0
    return pd.DataFrame(
        {
            "support_cells": cells,
            "support_signatures": counts,
            "reference_key_overlap": reference.isin(train_references).astype(float),
            "perturbation_seen_fraction": (counts > 0).astype(float),
            "component_seen_fraction": components_seen,
            "combination_novelty": np.asarray(
                [float(len(_components(label)) > 1 and label not in train_combinations) for label in labels],
                dtype=float,
            ),
            "perturbation_novelty": 1.0 / np.sqrt(np.maximum(counts, 1.0)),
            "batch_distance": (~batch.isin(train_batches)).astype(float),
            "training_support_max": max(max_support, 1.0),
        },
        index=metadata.index,
    )


def _load_ensemble(root: Path, dataset_id: str, predictor: str, split_family: str) -> pd.DataFrame:
    members: list[np.ndarray] = []
    metadata_by_seed: list[pd.DataFrame] = []
    member_paths: list[Path] = []
    for seed in (0, 1, 2):
        run = _find_run(root, MODEL_DIRS[predictor], dataset_id, split_family, seed)
        pred_path = run / "test_predictions.npz"
        meta_path = run / "test_metadata.parquet"
        if not pred_path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"missing prediction/metadata pair in {run}")
        with np.load(pred_path) as archive:
            members.append(np.asarray(archive["y_pred"], dtype=np.float32))
            target = np.asarray(archive["y_true"], dtype=np.float32)
        current = pd.read_parquet(meta_path).reset_index(drop=True)
        current["signature_id"] = current["signature_id"].astype(str)
        current["__target"] = list(target)
        current["__seed"] = seed
        metadata_by_seed.append(current)
        member_paths.append(pred_path)
    grouped: dict[str, list[tuple[int, np.ndarray, np.ndarray, pd.Series]]] = {}
    for current, member in zip(metadata_by_seed, members):
        for row_index, row in current.iterrows():
            signature = str(row["signature_id"])
            grouped.setdefault(signature, []).append(
                (int(row["__seed"]), np.asarray(member[row_index]), np.asarray(row["__target"]), row)
            )
    records: list[dict[str, object]] = []
    metadata_rows: list[pd.Series] = []
    member_seed_labels: list[str] = []
    uq_values: dict[str, list[float]] = {column: [] for column in UQ_COLUMNS}
    for signature in sorted(grouped):
        group = grouped[signature]
        if len(group) < 2:
            continue
        target = group[0][2]
        if not all(np.allclose(target, item[2], equal_nan=True) for item in group[1:]):
            raise ValueError(f"seed targets are not aligned for {dataset_id}/{predictor}/{signature}")
        values = np.stack([item[1] for item in group], axis=0)
        uq = summarize_ensemble(values[:, None, :])
        mean_prediction = values.mean(axis=0)
        cosine = float(np.dot(mean_prediction, target) / max(
            np.linalg.norm(mean_prediction) * np.linalg.norm(target), 1e-12
        ))
        abs_mean = np.abs(mean_prediction)
        prediction_norm = float(np.linalg.norm(mean_prediction))
        concentration = float(abs_mean.max() / max(prediction_norm, 1e-12))
        metadata_rows.append(group[0][3])
        member_seed_labels.append("|".join(str(item[0]) for item in group))
        records.append({
            "signature_id": signature,
            "prediction_norm": prediction_norm,
            "prediction_sparsity": float((abs_mean < 1e-8).mean()),
            "prediction_concentration": concentration,
            "prediction_response_sketch": _response_sketch(mean_prediction),
            "fidelity_delta_cosine": cosine,
        })
        for name, values_ in zip(UQ_COLUMNS, [
            uq.mean_gene_variance,
            uq.median_gene_variance,
            uq.top_effect_variance,
            uq.cosine_disagreement,
            uq.effect_norm_variance,
        ]):
            uq_values[name].append(float(values_[0]))
    if not metadata_rows:
        raise ValueError(f"no signatures have at least two seed members for {dataset_id}/{predictor}/{split_family}")
    metadata = pd.DataFrame(metadata_rows).reset_index(drop=True)
    record_frame = pd.DataFrame(records)
    training = _training_features(root, dataset_id, split_family, 0, metadata)
    perturbation = metadata["perturbation_label"].fillna(metadata["signature_id"].astype(str))
    environment = environment_id(
        dataset_id,
        PILOT_DATASETS[dataset_id]["cell_context"],
        PILOT_DATASETS[dataset_id]["modality"],
        PILOT_DATASETS[dataset_id]["condition"],
    )
    result = pd.DataFrame(
        {
            "dataset_id": dataset_id,
            "environment_id": environment,
            "predictor": predictor,
            "split_family": split_family,
            "signature_id": metadata["signature_id"].astype(str),
            "perturbation": perturbation.astype(str),
            "perturbation_group_id": [
                perturbation_group_id(environment, value) for value in perturbation
            ],
            **{column: training[column].to_numpy() for column in S_COLUMNS + N_COLUMNS},
            "batch_distance": training["batch_distance"].to_numpy(),
            **{column: record_frame[column].to_numpy() for column in [
                "prediction_norm", "prediction_sparsity", "prediction_concentration",
                "prediction_response_sketch", "fidelity_delta_cosine",
            ]},
            "prediction_array_path": _relative(root, member_paths[0]),
            "gene_list_path": _relative(root, member_paths[0].with_name("genes.txt")),
            "ensemble_member_seeds": member_seed_labels,
            "training_manifest_seeds": "0|1|2_intersection",
        }
    )
    for name in UQ_COLUMNS:
        result[name] = uq_values[name]
    result["biological_instance_id"] = [
        biological_instance_id(environment, value) for value in result["signature_id"]
    ]
    result["prediction_id"] = [
        prediction_id(bio, predictor, split_family, f"ensemble{len(seeds.split('|'))}")
        for bio, seeds in zip(
            result["biological_instance_id"], result["ensemble_member_seeds"]
        )
    ]
    return result


def _add_context_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for field in CONTEXT_CATEGORICAL_COLUMNS:
        frame[field] = frame["dataset_id"].map(
            {dataset: str(details[field]) for dataset, details in PILOT_DATASETS.items()}
        ).fillna("unknown").astype(str)
    frame["fidelity_delta_cosine"] = frame["fidelity_delta_cosine"].clip(-1.0, 1.0)
    frame["continuous_risk"] = (1.0 - frame["fidelity_delta_cosine"]) / 2.0
    frame["uq_quantile"] = np.nan
    frame["degenerate_uq"] = False
    return frame


def _fold_local_one_hot(train_values: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit categorical vocabulary on the fold's training rows only."""

    train_values = np.asarray(train_values, dtype=str)
    values = np.asarray(values, dtype=str)
    if train_values.ndim == 1:
        train_values = train_values.reshape(-1, 1)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    train_parts: list[np.ndarray] = []
    value_parts: list[np.ndarray] = []
    for column in range(train_values.shape[1]):
        categories = sorted(set(train_values[:, column]))
        lookup = {category: index for index, category in enumerate(categories)}
        train_encoded = np.zeros((len(train_values), len(categories)), dtype=float)
        value_encoded = np.zeros((len(values), len(categories)), dtype=float)
        for row, category in enumerate(train_values[:, column]):
            train_encoded[row, lookup[category]] = 1.0
        for row, category in enumerate(values[:, column]):
            index = lookup.get(category)
            if index is not None:
                value_encoded[row, index] = 1.0
        train_parts.append(train_encoded)
        value_parts.append(value_encoded)
    return np.column_stack(train_parts), np.column_stack(value_parts)


def _fold_local_manifold_distance(
    reference: np.ndarray,
    queries: np.ndarray,
    reference_predictors: np.ndarray,
    query_predictors: np.ndarray,
    response_sketches: np.ndarray,
    *,
    exclude_self: bool = False,
) -> np.ndarray:
    """Measure nearest training response distance in a train-only PCA space."""

    result = np.full(len(queries), np.nan, dtype=float)
    for predictor in sorted(np.unique(query_predictors[queries])):
        reference_local = reference[reference_predictors[reference] == predictor]
        query_local = np.flatnonzero(query_predictors[queries] == predictor)
        if not len(reference_local) or not len(query_local):
            continue
        reference_values = np.asarray(response_sketches[reference_local], dtype=float)
        query_values = np.asarray(response_sketches[queries[query_local]], dtype=float)
        center = reference_values.mean(axis=0)
        scale = np.where(reference_values.std(axis=0) > 1e-8, reference_values.std(axis=0), 1.0)
        reference_scaled = (reference_values - center) / scale
        query_scaled = (query_values - center) / scale
        rank = min(8, reference_scaled.shape[0], reference_scaled.shape[1])
        if rank:
            _, _, components = np.linalg.svd(reference_scaled, full_matrices=False)
            reference_latent = reference_scaled @ components[:rank].T
            query_latent = query_scaled @ components[:rank].T
        else:
            reference_latent = reference_scaled
            query_latent = query_scaled
        distances = np.empty(len(query_latent), dtype=float)
        for start in range(0, len(query_latent), 256):
            block = query_latent[start:start + 256]
            pairwise = np.sqrt(
                ((block[:, None, :] - reference_latent[None, :, :]) ** 2).sum(axis=2)
            )
            if exclude_self:
                for row, query_index in enumerate(queries[query_local[start:start + len(block)]]):
                    matches = np.flatnonzero(reference_local == query_index)
                    pairwise[row, matches] = np.inf
            distances[start:start + len(block)] = pairwise.min(axis=1)
        result[query_local] = distances
    if not np.isfinite(result).all():
        fallback = float(np.median(result[np.isfinite(result)])) if np.isfinite(result).any() else 0.0
        result[~np.isfinite(result)] = fallback
    return result


def _fit_one_fold(args: tuple[object, ...]) -> tuple[int, np.ndarray, dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """Fit one grouped outer fold in an independent process.

    The worker receives one fixed train/test split and returns only its OOF
    predictions.  It never mutates shared state; the parent merges results by
    fold after all workers finish.  This keeps the cross-fitting protocol
    unchanged while allowing the expensive model fits to use separate CPUs.
    """

    (
        fold,
        train,
        test,
        y,
        uq_source,
        context_numeric,
        context_categories,
        identity_categories,
        uq_features,
        support_novelty,
        predictors,
        response_sketches,
    ) = args
    fold = int(fold)
    train = np.asarray(train, dtype=int)
    test = np.asarray(test, dtype=int)
    y = np.asarray(y, dtype=int)
    uq_source = np.asarray(uq_source, dtype=float)
    context_numeric = np.asarray(context_numeric, dtype=float)
    context_categories = np.asarray(context_categories, dtype=str)
    identity_categories = np.asarray(identity_categories, dtype=str)
    uq_features = np.asarray(uq_features, dtype=float)
    support_novelty = np.asarray(support_novelty, dtype=float)
    predictors = np.asarray(predictors, dtype=str)
    response_sketches = np.asarray(response_sketches, dtype=float)
    prediction_names = [
        "raw_normalized_uq",
        "platt_logistic",
        "isotonic",
        "support_novelty_only",
        "ptl_rf",
        "ptl_rf_full_id",
        "gbdt_uncalibrated",
        "gbdt_full_id",
        "ptl_context",
        "ptl_full_id",
        "ptl_no_context",
    ]
    predictions = {
        name: np.full(len(test), np.nan, dtype=float)
        for name in prediction_names
    }
    p0_train = np.full(len(train), 0.5, dtype=float)
    p0_test = np.full(len(test), 0.5, dtype=float)
    degenerate_test = np.zeros(len(test), dtype=bool)
    context_cat_train, context_cat_test = _fold_local_one_hot(
        context_categories[train], context_categories[test]
    )
    identity_cat_train, identity_cat_test = _fold_local_one_hot(
        identity_categories[train], identity_categories[test]
    )
    manifold_train = _fold_local_manifold_distance(
        train, train, predictors, predictors, response_sketches, exclude_self=True
    )
    manifold_test = _fold_local_manifold_distance(
        train, test, predictors, predictors, response_sketches
    )
    context_train = np.column_stack([context_numeric[train], manifold_train, context_cat_train])
    context_test = np.column_stack([context_numeric[test], manifold_test, context_cat_test])
    no_context_train = np.column_stack([context_numeric[train][:, :len(NO_CONTEXT_NUMERIC_COLUMNS)], manifold_train])
    no_context_test = np.column_stack([context_numeric[test][:, :len(NO_CONTEXT_NUMERIC_COLUMNS)], manifold_test])
    all_numeric_train = np.column_stack([uq_features[train], context_train])
    all_numeric_test = np.column_stack([uq_features[test], context_test])
    all_numeric_full_train = np.column_stack([all_numeric_train, identity_cat_train])
    all_numeric_full_test = np.column_stack([all_numeric_test, identity_cat_test])
    train_predictors = predictors[train]
    test_predictors = predictors[test]
    for predictor in sorted(np.unique(train_predictors)):
        train_local = np.flatnonzero(train_predictors == predictor)
        test_local = np.flatnonzero(test_predictors == predictor)
        if not len(test_local):
            continue
        reference = uq_source[train[train_local]]
        normalizer = UQNormalizer.fit(reference)
        p0_train[train_local] = normalizer.transform(uq_source[train[train_local]])
        p0_test[test_local] = normalizer.transform(uq_source[test[test_local]])
        degenerate_test[test_local] = normalizer.degenerate
    if np.all(p0_train == 0.5):
        normalizer = UQNormalizer.fit(uq_source[train])
        p0_train = normalizer.transform(uq_source[train])
        p0_test = normalizer.transform(uq_source[test])
        degenerate_test[:] = normalizer.degenerate

    platt_model = fit_platt(p0_train, y[train])
    isotonic_model = fit_isotonic(p0_train, y[train])
    p0_scalar_train = predict_platt(platt_model, p0_train)
    p0_scalar_test = predict_platt(platt_model, p0_test)
    predictions["raw_normalized_uq"] = p0_test
    predictions["platt_logistic"] = p0_scalar_test
    predictions["isotonic"] = predict_isotonic(isotonic_model, p0_test)
    predictions["support_novelty_only"] = predict_logistic(
        fit_logistic(support_novelty[train], y[train]), support_novelty[test]
    )

    if np.unique(y[train]).size < 2:
        constant = float(y[train].mean())
        predictions["ptl_rf"] = np.full(len(test), constant, dtype=float)
        predictions["ptl_rf_full_id"] = np.full(len(test), constant, dtype=float)
        predictions["gbdt_uncalibrated"] = np.full(len(test), constant, dtype=float)
        predictions["gbdt_full_id"] = np.full(len(test), constant, dtype=float)
    else:
        rf = RandomForestClassifier(
            n_estimators=200, min_samples_leaf=10, random_state=17, n_jobs=1
        ).fit(all_numeric_train, y[train])
        gbdt = GradientBoostingClassifier(
            n_estimators=100, max_depth=2, learning_rate=0.04, random_state=17
        ).fit(all_numeric_train, y[train])
        rf_full = RandomForestClassifier(
            n_estimators=200, min_samples_leaf=10, random_state=17, n_jobs=1
        ).fit(all_numeric_full_train, y[train])
        gbdt_full = GradientBoostingClassifier(
            n_estimators=100, max_depth=2, learning_rate=0.04, random_state=17
        ).fit(all_numeric_full_train, y[train])
        predictions["ptl_rf"] = rf.predict_proba(all_numeric_test)[:, 1]
        predictions["ptl_rf_full_id"] = rf_full.predict_proba(all_numeric_full_test)[:, 1]
        predictions["gbdt_uncalibrated"] = gbdt.predict_proba(all_numeric_test)[:, 1]
        predictions["gbdt_full_id"] = gbdt_full.predict_proba(all_numeric_full_test)[:, 1]

    context_model = ContextualCalibrator().fit(context_train, p0_scalar_train, y[train])
    predictions["ptl_context"] = context_model.predict(context_test, p0_scalar_test)
    full_train = np.column_stack([context_train, identity_cat_train])
    full_test = np.column_stack([context_test, identity_cat_test])
    full_model = ContextualCalibrator().fit(full_train, p0_scalar_train, y[train])
    predictions["ptl_full_id"] = full_model.predict(full_test, p0_scalar_test)
    no_context_model = ContextualCalibrator().fit(no_context_train, p0_scalar_train, y[train])
    predictions["ptl_no_context"] = no_context_model.predict(no_context_test, p0_scalar_test)
    return fold, test, predictions, p0_test, degenerate_test, manifold_test


def _fit_fold_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    if "reliable_label" not in frame or frame["reliable_label"].isna().any():
        raise ValueError("reliable_label must be assigned from the random-anchor target before fitting")
    y = frame["reliable_label"].to_numpy(dtype=int)
    risk = frame["continuous_risk"].to_numpy(dtype=float)
    uq_source = frame["uq_mean_gene_variance"].to_numpy(dtype=float)
    context_numeric = frame[CONTEXT_NUMERIC_COLUMNS].to_numpy(dtype=float)
    context_categories = frame[CONTEXT_CATEGORICAL_COLUMNS].astype(str).to_numpy()
    identity_categories = frame[["environment_id", "predictor"]].astype(str).to_numpy()
    uq_features = frame[UQ_COLUMNS].to_numpy(dtype=float)
    support_novelty = frame[S_COLUMNS + N_COLUMNS].to_numpy(dtype=float)
    response_sketches = np.asarray(frame["prediction_response_sketch"].tolist(), dtype=float)
    groups = frame["biological_instance_id"].to_numpy()
    if len(np.unique(groups)) < 5:
        raise ValueError("pilot needs at least five biological-instance groups")
    folds = np.full(len(frame), -1, dtype=int)
    predictions = {
        name: np.full(len(frame), np.nan, dtype=float)
        for name in [
            "raw_normalized_uq",
            "platt_logistic",
            "isotonic",
            "support_novelty_only",
            "ptl_rf",
            "ptl_rf_full_id",
            "gbdt_uncalibrated",
            "gbdt_full_id",
            "ptl_context",
            "ptl_full_id",
            "ptl_no_context",
        ]
    }
    uq_quantile = np.full(len(frame), np.nan, dtype=float)
    degenerate_uq = np.zeros(len(frame), dtype=bool)
    manifold_distance = np.full(len(frame), np.nan, dtype=float)
    splitter = GroupKFold(n_splits=5)
    predictors = frame["predictor"].to_numpy(dtype=str)
    fold_tasks = [
        (
            fold,
            train,
            test,
            y,
            uq_source,
            context_numeric,
            context_categories,
            identity_categories,
            uq_features,
            support_novelty,
            predictors,
            response_sketches,
        )
        for fold, (train, test) in enumerate(splitter.split(context_numeric, y, groups))
    ]
    worker_count = int(os.environ.get("PTL_V2_WORKERS", "5"))
    if not 1 <= worker_count <= len(fold_tasks):
        raise ValueError(f"PTL_V2_WORKERS must be between 1 and {len(fold_tasks)}")
    if worker_count == 1:
        fold_results = [_fit_one_fold(task) for task in fold_tasks]
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            fold_results = list(executor.map(_fit_one_fold, fold_tasks))
    for fold, test, fold_predictions, fold_uq, fold_degenerate, fold_manifold_distance in sorted(
        fold_results, key=lambda result: result[0]
    ):
        folds[test] = fold
        uq_quantile[test] = fold_uq
        degenerate_uq[test] = fold_degenerate
        manifold_distance[test] = fold_manifold_distance
        for name in predictions:
            predictions[name][test] = fold_predictions[name]
    if np.any(folds < 0):
        raise AssertionError("some pilot rows were not assigned to a grouped fold")
    output = frame[[
        "prediction_id", "biological_instance_id", "environment_id", "predictor", "split_family",
        "signature_id", "perturbation_group_id",
    ]].copy()
    output["fold"] = folds
    output["uq_quantile"] = uq_quantile
    output["degenerate_uq"] = degenerate_uq
    output["prediction_manifold_distance"] = manifold_distance
    for name, values in predictions.items():
        output[name] = values
    output["reliable_label"] = y
    output["continuous_risk"] = risk
    return output


def _write_outputs(root: Path, frame: pd.DataFrame, fold_predictions: pd.DataFrame) -> dict[str, Path]:
    source = root / "artifacts/source_data"
    manifests = root / "artifacts/manifests"
    source.mkdir(parents=True, exist_ok=True)
    manifests.mkdir(parents=True, exist_ok=True)
    validate_feature_columns([column for column in frame.columns if column not in {
        "prediction_id", "biological_instance_id", "environment_id", "predictor", "dataset_id",
        "split_family", "signature_id", "perturbation", "perturbation_group_id",
        "prediction_array_path", "gene_list_path", "ensemble_member_seeds",
        "training_manifest_seeds",
        "fidelity_delta_cosine", "continuous_risk", "reliable_label", "legacy_transportability_label",
        "legacy_target_risk", "anchor_fidelity", "transportability_threshold", "transportability_label",
        "legacy_reliable_label", "legacy_threshold", "transportability_tau", "training_support_max",
        "degenerate_uq", "uq_quantile", "anchor_median_fidelity", "prediction_response_sketch",
    }])
    predictions_columns = [
        "prediction_id", "biological_instance_id", "environment_id", "predictor", "split_family",
        "signature_id", "perturbation_group_id", "prediction_array_path",
        "gene_list_path", "ensemble_member_seeds", "training_manifest_seeds", *UQ_COLUMNS,
    ]
    deployment_columns = [
        "prediction_id", "biological_instance_id", "environment_id", "predictor",
        *P_COLUMNS, *S_COLUMNS, *N_COLUMNS, *C_COLUMNS,
    ]
    outcomes_columns = [
        "prediction_id", "biological_instance_id", "environment_id", "predictor", "split_family",
        "signature_id", "perturbation_group_id", "fidelity_delta_cosine", "continuous_risk",
        "anchor_fidelity", "transportability_threshold", "transportability_label", "reliable_label",
        "transportability_tau", "anchor_median_fidelity",
        "legacy_transportability_label", "legacy_target_risk",
    ]
    paths = {
        "predictions": source / "predictions_summary.csv",
        "prediction_uq_manifest": manifests / "prediction_uq_manifest.csv",
        "deployment_features": source / "deployment_features.csv",
        "outcomes": source / "outcomes.csv",
        "folds": source / "folds.csv",
        "fold_summary": manifests / "fold_summary.csv",
        "environment_registry": manifests / "environment_registry.csv",
        "folds_parquet": manifests / "reliability_folds.parquet",
        "rq1": source / "rq1_raw_uq_summary.csv",
        "g4": source / "g4_method_comparison.csv",
        "bootstrap_ci": source / "g4_bootstrap_ci.csv",
        "fig2": source / "figure2_confidence_source.csv",
        "fig3": source / "figure3_selective_source.csv",
        "target_sensitivity": source / "transportability_target_sensitivity.csv",
        "anchor_registry": manifests / "reliability_anchor_registry.csv",
    }
    frame[predictions_columns].drop_duplicates("prediction_id").to_csv(paths["predictions"], index=False)
    prediction_manifest = (
        frame.groupby(["environment_id", "dataset_id", "predictor", "split_family"], as_index=False)
        .agg(
            n_predictions=("prediction_id", "nunique"),
            n_biological_instances=("biological_instance_id", "nunique"),
            ensemble_member_seeds=("ensemble_member_seeds", "first"),
            training_manifest_seeds=("training_manifest_seeds", "first"),
            prediction_array_path=("prediction_array_path", "first"),
            gene_list_path=("gene_list_path", "first"),
            uq_method=("ensemble_member_seeds", lambda values: "aligned_2_or_3_seed_ensemble"),
        )
    )
    prediction_manifest.to_csv(paths["prediction_uq_manifest"], index=False)
    frame[deployment_columns].to_csv(paths["deployment_features"], index=False)
    frame[outcomes_columns].to_csv(paths["outcomes"], index=False)
    fold_table = fold_predictions[["prediction_id", "biological_instance_id", "split_family", "fold"]].drop_duplicates()
    fold_table.to_csv(paths["folds"], index=False)
    fold_table.to_parquet(paths["folds_parquet"], index=False)
    fold_summary = (
        fold_predictions.groupby(["environment_id", "predictor", "split_family", "fold"], as_index=False)
        .agg(
            n_predictions=("prediction_id", "nunique"),
            n_biological_instances=("biological_instance_id", "nunique"),
            reliable_rate=("reliable_label", "mean"),
        )
    )
    fold_summary.to_csv(paths["fold_summary"], index=False)
    environment_registry = pd.DataFrame(
        [
            {
                "environment_id": environment_id(
                    dataset_id, details["cell_context"], details["modality"], details["condition"]
                ),
                "dataset_id": dataset_id,
                "cell_context": details["cell_context"],
                "perturbation_modality": details["perturbation_modality"],
                "readout_modality": details["readout_modality"],
                "platform": details["platform"],
                "condition": details["condition"],
                "role": "pilot_replay",
                "semantic_status": "core_pilot_fields_explicit_metadata_expansion_pending",
                "status": "active_legacy_prediction_surface" if "legacy" in details["condition"] else "active_pilot_surface",
            }
            for dataset_id, details in PILOT_DATASETS.items()
        ]
    )
    environment_registry.to_csv(paths["environment_registry"], index=False)

    sensitivity_rows = []
    anchor_median = frame["transportability_threshold"] / frame["transportability_tau"]
    for tau in (0.6, 0.7, 0.8, 0.9):
        labels = frame["fidelity_delta_cosine"] >= tau * anchor_median
        for keys, group in frame.assign(_sensitivity_label=labels).groupby(
            ["dataset_id", "predictor", "split_family"], sort=True
        ):
            sensitivity_rows.append({
                "dataset_id": keys[0],
                "predictor": keys[1],
                "split_family": keys[2],
                "tau": tau,
                "n": len(group),
                "reliable_rate": group["_sensitivity_label"].mean(),
            })
    pd.DataFrame(sensitivity_rows).to_csv(paths["target_sensitivity"], index=False)

    rq1_rows = []
    working = frame.drop(columns=["uq_quantile", "degenerate_uq"]).merge(
        fold_predictions[["prediction_id", "uq_quantile", "degenerate_uq"]],
        on="prediction_id",
        how="left",
    )
    working["support_bin"] = pd.qcut(working["support_cells"].rank(method="first"), 3, labels=["low", "mid", "high"])
    working["novelty_bin"] = pd.qcut(working["perturbation_novelty"].rank(method="first"), 3, labels=["low", "mid", "high"])
    for keys, group in working.groupby(["environment_id", "predictor", "split_family", "support_bin", "novelty_bin"], observed=True):
        corr = group["uq_quantile"].corr(group["fidelity_delta_cosine"], method="spearman")
        rq1_rows.append({
            "environment_id": keys[0], "predictor": keys[1], "split_family": keys[2],
            "support_bin": keys[3], "novelty_bin": keys[4],
            "n": len(group), "mean_fidelity": group["fidelity_delta_cosine"].mean(),
            "mean_confidence": group["uq_quantile"].mean(), "spearman_confidence_fidelity": corr,
            "degenerate_uq_rate": group["degenerate_uq"].mean(),
        })
    pd.DataFrame(rq1_rows).to_csv(paths["rq1"], index=False)

    method_rows = []
    method_names = [column for column in fold_predictions.columns if column in {
        "raw_normalized_uq", "platt_logistic", "isotonic", "support_novelty_only", "ptl_rf",
        "ptl_rf_full_id", "gbdt_uncalibrated", "gbdt_full_id", "ptl_context",
        "ptl_full_id", "ptl_no_context"
    }]
    for (predictor, environment, split_family), group in fold_predictions.groupby(
        ["predictor", "environment_id", "split_family"]
    ):
        for method in method_names:
            values = evaluate_scores(group["reliable_label"], group[method], group["continuous_risk"])
            method_rows.append({
                "aggregation_level": "predictor_environment_split",
                "predictor": predictor,
                "environment_id": environment,
                "split_family": split_family,
                "method": method,
                **values,
            })
    detail = pd.DataFrame(method_rows)
    macro = detail.groupby("method", as_index=False)[["aurc", "excess_aurc", "risk_at_50", "risk_at_80", "ftr_at_50", "ftr_at_80", "brier", "log_loss", "auroc", "auprc", "spearman_predicted_risk_realized_risk"]].mean(numeric_only=True)
    macro["aggregation_level"] = "environment_macro_average"
    macro["predictor"] = "all_pilot_predictors"
    macro["environment_id"] = "macro"
    macro["split_family"] = "macro"
    macro["n"] = np.nan
    detail = pd.concat([detail, macro[detail.columns]], ignore_index=True)
    detail.to_csv(paths["g4"], index=False)
    bootstrap_draws = paired_hierarchical_bootstrap(
        fold_predictions,
        method_names,
        n_resamples=200,
        seed=17,
    )
    bootstrap_ci = summarize_bootstrap_ci(bootstrap_draws)
    delta_ci = summarize_paired_deltas(bootstrap_draws, "raw_normalized_uq")
    delta_ci["aggregation_level"] = "paired_hierarchical_delta_ci"
    delta_ci["predictor"] = "all_pilot_predictors"
    delta_ci["environment_id"] = "macro"
    delta_ci["split_family"] = "macro"
    bootstrap_ci["aggregation_level"] = "paired_hierarchical_macro_ci"
    bootstrap_ci["baseline_method"] = ""
    bootstrap_ci["resample_level"] = "environment_then_biological_instance"
    bootstrap_ci["predictor"] = "all_pilot_predictors"
    bootstrap_ci["environment_id"] = "macro"
    bootstrap_ci["split_family"] = "macro"
    bootstrap_ci = pd.concat([bootstrap_ci, delta_ci], ignore_index=True, sort=False)
    bootstrap_ci.to_csv(paths["bootstrap_ci"], index=False)
    working[["prediction_id", "environment_id", "predictor", "split_family", "uq_quantile", "degenerate_uq", "fidelity_delta_cosine", "support_cells", "perturbation_novelty"]].to_csv(paths["fig2"], index=False)
    detail.to_csv(paths["fig3"], index=False)
    return paths


def _freeze_reliability_anchor_registry(root: Path, frame: pd.DataFrame) -> pd.DataFrame:
    """Freeze familiar/random fidelity anchors before PTL cross-fitting."""

    path = root / "artifacts/manifests/reliability_anchor_registry.csv"
    anchor = frame.loc[frame["split_family"].eq("random_split")]
    registry = (
        anchor.groupby(["dataset_id", "predictor"], as_index=False)
        .agg(
            anchor_median_fidelity=("fidelity_delta_cosine", "median"),
            n_anchor_predictions=("prediction_id", "nunique"),
        )
        .assign(
            anchor_split_family="random_split",
            anchor_definition="median familiar/random fidelity; frozen before PTL grouped cross-fitting",
            transportability_tau=0.8,
        )
    )
    registry["transportability_threshold"] = (
        registry["transportability_tau"] * registry["anchor_median_fidelity"]
    )
    expected_columns = [
        "dataset_id", "predictor", "anchor_median_fidelity", "n_anchor_predictions",
        "anchor_split_family", "anchor_definition", "transportability_tau",
        "transportability_threshold",
    ]
    registry = registry[expected_columns].sort_values(["dataset_id", "predictor"]).reset_index(drop=True)
    if path.exists():
        frozen = pd.read_csv(path)
        frozen = frozen.reindex(columns=expected_columns).sort_values(["dataset_id", "predictor"]).reset_index(drop=True)
        if list(frozen["dataset_id"].astype(str)) != list(registry["dataset_id"].astype(str)) or list(frozen["predictor"].astype(str)) != list(registry["predictor"].astype(str)):
            raise ValueError("frozen reliability anchor registry does not match the current pilot universe")
        for column in ["anchor_median_fidelity", "transportability_tau", "transportability_threshold"]:
            if not np.allclose(
                frozen[column].to_numpy(dtype=float),
                registry[column].to_numpy(dtype=float),
                atol=1e-12,
                rtol=0.0,
            ):
                raise ValueError(f"frozen reliability anchor changed for column {column}")
        return frozen
    path.parent.mkdir(parents=True, exist_ok=True)
    registry.to_csv(path, index=False)
    return registry


def run_pilot(root: str | Path) -> dict[str, object]:
    root = Path(root).resolve()
    labels = _load_oof_labels(root)
    frames = []
    for dataset_id in PILOT_DATASETS:
        for predictor in MODEL_DIRS:
            for split_family in PILOT_SPLITS:
                frames.append(_load_ensemble(root, dataset_id, predictor, split_family))
    frame = _add_context_features(pd.concat(frames, ignore_index=True))
    frame = frame.merge(labels, on=["dataset_id", "predictor", "signature_id"], how="left")
    anchors = (
        frame.loc[frame["split_family"] == "random_split", [
            "dataset_id", "predictor", "signature_id", "fidelity_delta_cosine"
        ]]
        .rename(columns={"fidelity_delta_cosine": "anchor_fidelity"})
    )
    anchor_registry = _freeze_reliability_anchor_registry(root, frame)
    frame = frame.merge(anchors, on=["dataset_id", "predictor", "signature_id"], how="left")
    frame = frame.merge(
        anchor_registry[["dataset_id", "predictor", "anchor_median_fidelity", "transportability_threshold"]],
        on=["dataset_id", "predictor"],
        how="left",
        validate="many_to_one",
    )
    frame["transportability_tau"] = 0.8
    frame["transportability_label"] = (
        frame["fidelity_delta_cosine"] >= frame["transportability_threshold"]
    ).astype(int)
    frame["reliable_label"] = frame["transportability_label"]
    frame["legacy_reliable_label"] = frame["legacy_transportability_label"]
    frame["legacy_threshold"] = 0.5
    frame["legacy_transportability_label"] = frame["legacy_transportability_label"].fillna(-1.0)
    frame["legacy_target_risk"] = frame["legacy_target_risk"].fillna(np.nan)
    fold_predictions = _fit_fold_predictions(frame)
    frame = frame.merge(
        fold_predictions[["prediction_id", "prediction_manifold_distance"]],
        on="prediction_id",
        how="left",
        validate="one_to_one",
    )
    paths = _write_outputs(root, frame, fold_predictions)
    summary = {
        "pilot_id": "ptl_v2_g4_baseline_replay_2026-09-06",
        "datasets": sorted(frame["dataset_id"].unique().tolist()),
        "predictors": sorted(frame["predictor"].unique().tolist()),
        "rows": int(len(frame)),
        "biological_instances": int(frame["biological_instance_id"].nunique()),
        "folds": 5,
        "cross_fit_workers": int(os.environ.get("PTL_V2_WORKERS", "5")),
        "cross_fit_parallelization": "biological-instance-disjoint grouped cross-fitting with deterministic parent merge",
        "split_families": list(PILOT_SPLITS),
        "transportability_target": "T_i = I[F_i >= 0.8 * median(F_random_anchor)]",
        "transportability_tau_sensitivity": [0.6, 0.7, 0.8, 0.9],
        "gears": "official_gears_scientific_pilot_condition_holdout_completed; see artifacts/manifests/gears_pilot_audit.csv; G2 predictor contract passed independent review",
        "k562_surface": "PTL baseline uses the ReplogleWeissman2022_K562_essential legacy surface; the separate GEARS pilot includes ReplogleWeissman2022_K562_gwps",
        "outputs": {key: _relative(root, value) for key, value in paths.items()},
    }
    summary_path = root / "artifacts/manifests/ptl_v2_g4_pilot_summary.json"
    summary["outputs"]["summary"] = _relative(root, summary_path)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
