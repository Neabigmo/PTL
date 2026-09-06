"""Run the first leakage-safe PTL-v2 pilot from local prediction artifacts.

The pilot deliberately replays existing three-seed baseline prediction arrays
instead of silently claiming that unavailable GEARS outputs exist.  It creates
new v2 contracts, uncertainty summaries, grouped folds, and reliability
comparisons without modifying raw or processed matrices.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GroupKFold

from ptl.data.ids import biological_instance_id, environment_id, prediction_id
from ptl.data.contracts import validate_feature_columns
from ptl.evaluation.bootstrap import paired_hierarchical_bootstrap, summarize_bootstrap_ci
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
from ptl.uncertainty.uq import confidence_from_uq, summarize_ensemble


PILOT_DATASETS = {
    "NormanWeissman2019_filtered": {
        "cell_context": "K562",
        "modality": "RNA",
        "condition": "baseline",
    },
    # This is the existing processed K562 surface.  GWPS is the v2 target
    # surface, but it has no compatible processed prediction arrays yet.
    "ReplogleWeissman2022_K562_essential": {
        "cell_context": "K562",
        "modality": "RNA",
        "condition": "essential_legacy_pilot",
    },
    "ReplogleWeissman2022_rpe1": {
        "cell_context": "RPE1",
        "modality": "RNA",
        "condition": "baseline",
    },
}
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
    "prediction_reference_distance",
]
S_COLUMNS = ["support_cells", "support_signatures", "reference_key_overlap"]
N_COLUMNS = ["perturbation_seen_fraction", "combination_novelty", "perturbation_novelty"]
C_COLUMNS = ["cell_context_code", "modality_code", "condition_code", "batch_distance"]
CONTEXT_COLUMNS = P_COLUMNS + S_COLUMNS + N_COLUMNS + C_COLUMNS
ALL_NUMERIC_COLUMNS = UQ_COLUMNS + CONTEXT_COLUMNS


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


def _find_run(root: Path, model_dir: str, dataset_id: str, seed: int) -> Path:
    base = root / "results/baselines" / model_dir
    candidates = sorted(
        base.glob(f"dataset_heldout_split__all_datasets__holdout-{dataset_id}__seed{seed}__signature")
    )
    if len(candidates) != 1:
        raise FileNotFoundError(f"expected one seed directory, found {len(candidates)}: {base}")
    return candidates[0]


def _load_ensemble(root: Path, dataset_id: str, predictor: str) -> pd.DataFrame:
    members: list[np.ndarray] = []
    metadata: pd.DataFrame | None = None
    member_paths: list[Path] = []
    for seed in (0, 1, 2):
        run = _find_run(root, MODEL_DIRS[predictor], dataset_id, seed)
        pred_path = run / "test_predictions.npz"
        meta_path = run / "test_metadata.parquet"
        if not pred_path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"missing prediction/metadata pair in {run}")
        with np.load(pred_path) as archive:
            members.append(np.asarray(archive["y_pred"], dtype=np.float32))
            target = np.asarray(archive["y_true"], dtype=np.float32)
        current = pd.read_parquet(meta_path).reset_index(drop=True)
        if metadata is None:
            metadata = current
            targets = target
        else:
            if not metadata["signature_id"].astype(str).equals(current["signature_id"].astype(str)):
                raise ValueError(f"seed metadata are not aligned for {dataset_id}/{predictor}")
            if not np.allclose(targets, target, equal_nan=True):
                raise ValueError(f"seed targets are not aligned for {dataset_id}/{predictor}")
        member_paths.append(pred_path)
    assert metadata is not None
    values = np.stack(members, axis=0)
    uq = summarize_ensemble(values)
    mean_prediction = values.mean(axis=0)
    cosine = np.sum(mean_prediction * targets, axis=1)
    cosine = cosine / np.maximum(
        np.linalg.norm(mean_prediction, axis=1) * np.linalg.norm(targets, axis=1), 1e-12
    )
    abs_mean = np.abs(mean_prediction)
    prediction_norm = np.linalg.norm(mean_prediction, axis=1)
    concentration = abs_mean.max(axis=1) / np.maximum(prediction_norm, 1e-12)
    support = metadata.get("n_cells", pd.Series(np.ones(len(metadata))))
    perturbation = metadata["perturbation_label"].fillna(metadata["signature_id"].astype(str))
    support_by_perturbation = perturbation.groupby(perturbation).transform("size").to_numpy()
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
            "signature_id": metadata["signature_id"].astype(str),
            "perturbation": perturbation.astype(str),
            "support_cells": pd.to_numeric(support, errors="coerce").fillna(0).to_numpy(),
            "support_signatures": support_by_perturbation,
            "reference_key_overlap": metadata.get("reference_key", pd.Series([None] * len(metadata))).notna().astype(float),
            "prediction_norm": prediction_norm,
            "prediction_sparsity": (abs_mean < 1e-8).mean(axis=1),
            "prediction_concentration": concentration,
            "prediction_reference_distance": 1.0 - np.clip(concentration, 0.0, 1.0),
            "combination_novelty": perturbation.str.contains(r"\+|,|;", regex=True).astype(float),
            "perturbation_novelty": 1.0 / np.sqrt(np.maximum(support_by_perturbation, 1)),
            "fidelity_delta_cosine": cosine,
            "prediction_array_path": _relative(root, member_paths[0]),
            "gene_list_path": _relative(root, member_paths[0].with_name("genes.txt")),
            "ensemble_member_seeds": "0|1|2",
            **{name: getattr(uq, name.removeprefix("uq_")) for name in []},
        }
    )
    for name, values_ in zip(UQ_COLUMNS, [
        uq.mean_gene_variance,
        uq.median_gene_variance,
        uq.top_effect_variance,
        uq.cosine_disagreement,
        uq.effect_norm_variance,
    ]):
        result[name] = values_
    result["biological_instance_id"] = [
        biological_instance_id(environment, value) for value in result["perturbation"]
    ]
    result["prediction_id"] = [
        prediction_id(bio, predictor, "dataset_heldout_split", "ensemble3")
        for bio in result["biological_instance_id"]
    ]
    return result


def _add_context_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["perturbation_seen_fraction"] = frame.groupby("environment_id")["support_signatures"].transform(
        lambda values: values / max(float(values.max()), 1.0)
    )
    frame["cell_context_code"] = frame["environment_id"].str.contains("rpe1").astype(float) * 2.0
    frame.loc[frame["environment_id"].str.contains("k562"), "cell_context_code"] = 1.0
    frame["modality_code"] = 0.0
    frame["condition_code"] = frame["environment_id"].str.contains("essential").astype(float)
    frame["batch_distance"] = 0.0
    frame["fidelity_delta_cosine"] = frame["fidelity_delta_cosine"].clip(-1.0, 1.0)
    frame["continuous_risk"] = 1.0 - (frame["fidelity_delta_cosine"] + 1.0) / 2.0
    frame["reliable_label"] = (frame["fidelity_delta_cosine"] >= 0.5).astype(int)
    frame["uq_quantile"] = np.nan
    for _, indices in frame.groupby("predictor").groups.items():
        idx = np.asarray(list(indices), dtype=int)
        frame.loc[idx, "uq_quantile"] = confidence_from_uq(
            frame.loc[idx, "uq_mean_gene_variance"].to_numpy(),
            frame.loc[idx, "uq_mean_gene_variance"].to_numpy(),
        )
    return frame


def _fit_fold_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    y = frame["reliable_label"].to_numpy(dtype=int)
    risk = frame["continuous_risk"].to_numpy(dtype=float)
    p0 = np.clip(frame["uq_quantile"].to_numpy(dtype=float), 1e-5, 1.0 - 1e-5)
    context = frame[CONTEXT_COLUMNS].to_numpy(dtype=float)
    all_numeric = frame[ALL_NUMERIC_COLUMNS].to_numpy(dtype=float)
    support_novelty = frame[S_COLUMNS + N_COLUMNS].to_numpy(dtype=float)
    identity = pd.get_dummies(frame[["environment_id", "predictor"]], dtype=float).to_numpy()
    full = np.column_stack([context, identity])
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
            "calibrated_gbdt",
            "ptl_context",
            "ptl_full",
            "ptl_no_id",
        ]
    }
    splitter = GroupKFold(n_splits=5)
    for fold, (train, test) in enumerate(splitter.split(context, y, groups)):
        folds[test] = fold
        predictions["raw_normalized_uq"][test] = p0[test]
        predictions["platt_logistic"][test] = predict_platt(fit_platt(p0[train], y[train]), p0[test])
        predictions["isotonic"][test] = predict_isotonic(fit_isotonic(p0[train], y[train]), p0[test])
        predictions["support_novelty_only"][test] = predict_logistic(
            fit_logistic(support_novelty[train], y[train]), support_novelty[test]
        )

        if np.unique(y[train]).size < 2:
            constant = float(y[train].mean())
            predictions["ptl_rf"][test] = constant
            predictions["calibrated_gbdt"][test] = constant
            predictions["ptl_full"][test] = constant
        else:
            rf = RandomForestClassifier(
                n_estimators=200, min_samples_leaf=10, random_state=17, n_jobs=1
            ).fit(all_numeric[train], y[train])
            gbdt = GradientBoostingClassifier(
                n_estimators=100, max_depth=2, learning_rate=0.04, random_state=17
            ).fit(all_numeric[train], y[train])
            full_model = RandomForestClassifier(
                n_estimators=200, min_samples_leaf=10, random_state=19, n_jobs=1
            ).fit(full[train], y[train])
            predictions["ptl_rf"][test] = rf.predict_proba(all_numeric[test])[:, 1]
            predictions["calibrated_gbdt"][test] = gbdt.predict_proba(all_numeric[test])[:, 1]
            predictions["ptl_full"][test] = full_model.predict_proba(full[test])[:, 1]

        context_model = ContextualCalibrator().fit(context[train], p0[train], y[train])
        predictions["ptl_context"][test] = context_model.predict(context[test], p0[test])
        no_id_model = ContextualCalibrator().fit(context[train], p0[train], y[train])
        predictions["ptl_no_id"][test] = no_id_model.predict(context[test], p0[test])
    if np.any(folds < 0):
        raise AssertionError("some pilot rows were not assigned to a grouped fold")
    output = frame[["prediction_id", "biological_instance_id", "environment_id", "predictor"]].copy()
    output["fold"] = folds
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
        "signature_id", "perturbation", "prediction_array_path", "gene_list_path", "ensemble_member_seeds",
        "fidelity_delta_cosine", "continuous_risk", "reliable_label", "legacy_transportability_label",
        "legacy_target_risk",
    }])
    predictions_columns = [
        "prediction_id", "biological_instance_id", "environment_id", "predictor", "prediction_array_path",
        "gene_list_path", "ensemble_member_seeds", *UQ_COLUMNS,
    ]
    deployment_columns = ["prediction_id", "biological_instance_id", "environment_id", "predictor", *P_COLUMNS, *S_COLUMNS, *N_COLUMNS, *C_COLUMNS]
    outcomes_columns = ["prediction_id", "biological_instance_id", "environment_id", "predictor", "fidelity_delta_cosine", "continuous_risk", "reliable_label", "legacy_transportability_label", "legacy_target_risk"]
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
    }
    frame[predictions_columns].drop_duplicates("prediction_id").to_csv(paths["predictions"], index=False)
    prediction_manifest = (
        frame.groupby(["environment_id", "dataset_id", "predictor"], as_index=False)
        .agg(
            n_predictions=("prediction_id", "nunique"),
            n_biological_instances=("biological_instance_id", "nunique"),
            ensemble_member_seeds=("ensemble_member_seeds", "first"),
            prediction_array_path=("prediction_array_path", "first"),
            gene_list_path=("gene_list_path", "first"),
            uq_method=("ensemble_member_seeds", lambda values: "three_seed_ensemble"),
        )
    )
    prediction_manifest.to_csv(paths["prediction_uq_manifest"], index=False)
    frame[deployment_columns].to_csv(paths["deployment_features"], index=False)
    frame[outcomes_columns].to_csv(paths["outcomes"], index=False)
    fold_table = fold_predictions[["prediction_id", "biological_instance_id", "fold"]].drop_duplicates()
    fold_table.to_csv(paths["folds"], index=False)
    fold_table.to_parquet(paths["folds_parquet"], index=False)
    fold_summary = (
        fold_predictions.groupby(["environment_id", "predictor", "fold"], as_index=False)
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
                "modality": details["modality"],
                "condition": details["condition"],
                "role": "pilot_replay",
                "status": "active_legacy_prediction_surface" if "legacy" in details["condition"] else "active_pilot_surface",
            }
            for dataset_id, details in PILOT_DATASETS.items()
        ]
    )
    environment_registry.to_csv(paths["environment_registry"], index=False)

    rq1_rows = []
    working = frame.copy()
    working["support_bin"] = pd.qcut(working["support_cells"].rank(method="first"), 3, labels=["low", "mid", "high"])
    working["novelty_bin"] = pd.qcut(working["perturbation_novelty"].rank(method="first"), 3, labels=["low", "mid", "high"])
    for keys, group in working.groupby(["environment_id", "predictor", "support_bin", "novelty_bin"], observed=True):
        corr = group["uq_quantile"].corr(group["fidelity_delta_cosine"], method="spearman")
        rq1_rows.append({
            "environment_id": keys[0], "predictor": keys[1], "support_bin": keys[2], "novelty_bin": keys[3],
            "n": len(group), "mean_fidelity": group["fidelity_delta_cosine"].mean(),
            "mean_confidence": group["uq_quantile"].mean(), "spearman_confidence_fidelity": corr,
        })
    pd.DataFrame(rq1_rows).to_csv(paths["rq1"], index=False)

    method_rows = []
    method_names = [column for column in fold_predictions.columns if column in {
        "raw_normalized_uq", "platt_logistic", "isotonic", "support_novelty_only", "ptl_rf",
        "calibrated_gbdt", "ptl_context", "ptl_full", "ptl_no_id"
    }]
    for (predictor, environment), group in fold_predictions.groupby(["predictor", "environment_id"]):
        for method in method_names:
            values = evaluate_scores(group["reliable_label"], group[method], group["continuous_risk"])
            method_rows.append({"aggregation_level": "predictor_environment", "predictor": predictor, "environment_id": environment, "method": method, **values})
    detail = pd.DataFrame(method_rows)
    macro = detail.groupby("method", as_index=False)[["aurc", "excess_aurc", "risk_at_50", "risk_at_80", "ftr_at_50", "ftr_at_80", "brier", "log_loss", "auroc", "auprc", "spearman_continuous_risk"]].mean(numeric_only=True)
    macro["aggregation_level"] = "environment_macro_average"
    macro["predictor"] = "all_pilot_predictors"
    macro["environment_id"] = "macro"
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
    bootstrap_ci.insert(0, "aggregation_level", "paired_hierarchical_macro_ci")
    bootstrap_ci.insert(2, "predictor", "all_pilot_predictors")
    bootstrap_ci.insert(3, "environment_id", "macro")
    bootstrap_ci.to_csv(paths["bootstrap_ci"], index=False)
    frame[["prediction_id", "environment_id", "predictor", "uq_quantile", "fidelity_delta_cosine", "support_cells", "perturbation_novelty"]].to_csv(paths["fig2"], index=False)
    detail.to_csv(paths["fig3"], index=False)
    return paths


def run_pilot(root: str | Path) -> dict[str, object]:
    root = Path(root).resolve()
    labels = _load_oof_labels(root)
    frames = []
    for dataset_id in PILOT_DATASETS:
        for predictor in MODEL_DIRS:
            frames.append(_load_ensemble(root, dataset_id, predictor))
    frame = _add_context_features(pd.concat(frames, ignore_index=True))
    frame = frame.merge(labels, on=["dataset_id", "predictor", "signature_id"], how="left")
    frame["legacy_transportability_label"] = frame["legacy_transportability_label"].fillna(-1.0)
    frame["legacy_target_risk"] = frame["legacy_target_risk"].fillna(np.nan)
    fold_predictions = _fit_fold_predictions(frame)
    paths = _write_outputs(root, frame, fold_predictions)
    summary = {
        "pilot_id": "ptl_v2_g4_baseline_replay_2026-09-06",
        "datasets": sorted(frame["dataset_id"].unique().tolist()),
        "predictors": sorted(frame["predictor"].unique().tolist()),
        "rows": int(len(frame)),
        "biological_instances": int(frame["biological_instance_id"].nunique()),
        "folds": 5,
        "gears": "not_run: official package imports in the project environment, but the v2 data adapter is not yet validated; legacy fallback paths remain excluded",
        "k562_surface": "ReplogleWeissman2022_K562_essential legacy processed surface; GWPS deferred",
        "outputs": {key: _relative(root, value) for key, value in paths.items()},
    }
    summary_path = root / "artifacts/manifests/ptl_v2_g4_pilot_summary.json"
    summary["outputs"]["summary"] = _relative(root, summary_path)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
