#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEEDS = [0, 1, 2]
SIGNATURE_META_COLUMNS = {
    "dataset_id",
    "source_dataset",
    "signature_id",
    "reference_key",
    "perturbation_label",
    "is_control",
    "n_cells",
    "control_label_used",
    "batch",
    "timepoint",
}
CELL_STATE_CANDIDATES = ["cell_context", "predictions", "supervised_name"]
LINEAGE_CANDIDATES = ["class", "supervised_name", "cell_context"]
MISSING_LABELS = {"", "na", "nan", "none", "<na>", "unknown", "null"}
COMBINATION_EXCLUDE = {"MULTI_TARGET"}
COMBINATION_SINGLETON_TOKENS = {"only", "single", "singleton", "ctrl", "control", "neg", "negative"}
COMBINATION_PREFIX_TOKENS = {"tcrlibrary", "library", "grna", "sgrna"}


@dataclass
class DatasetBundle:
    dataset_id: str
    source_dataset: str
    split_role: str
    external_validation_ready: bool
    cell_metadata_path: Path
    qc_manifest_path: Path
    feature_metadata_path: Path
    signature_path: Path
    cell_frame: pd.DataFrame
    signature_frame: pd.DataFrame
    cell_state_col: str | None
    cell_lineage_col: str | None
    signature_state_col: str | None
    signature_lineage_col: str | None
    signature_state_source: str | None
    signature_lineage_source: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build deterministic Phase 04 context-stress splits.")
    parser.add_argument(
        "--summary-path",
        default=str(ROOT / "results" / "tables" / "preprocessing_summary.csv"),
        help="Path to preprocessing summary table",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "data" / "processed" / "splits"),
        help="Directory to store split JSON artifacts",
    )
    parser.add_argument(
        "--audit-path",
        default=str(ROOT / "results" / "tables" / "split_audit.csv"),
        help="Path to split audit CSV",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        help="Deterministic seeds for split generation",
    )
    return parser.parse_args()


def normalize_series(series: pd.Series) -> pd.Series:
    text = series.astype("object").where(~series.isna(), other=pd.NA)
    text = text.astype("string").str.strip()
    lower = text.str.lower()
    return text.mask(lower.isin(MISSING_LABELS), other=pd.NA)


def normalize_scalar(value: Any) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if text.lower() in MISSING_LABELS:
        return None
    return text


def is_combination_label(value: Any) -> bool:
    label = normalize_scalar(value)
    if label is None:
        return False
    if label in COMBINATION_EXCLUDE:
        return False
    if any(token in label for token in ["+", ",", ";", "|"]):
        return True
    if "_" not in label:
        return False
    parts = [part for part in label.split("_") if part]
    normalized_parts: list[str] = []
    for part in parts:
        lower = part.lower()
        if lower in COMBINATION_SINGLETON_TOKENS:
            continue
        if lower in COMBINATION_PREFIX_TOKENS:
            continue
        if lower.startswith("pmj") and lower[3:].isdigit():
            continue
        if lower.isdigit():
            continue
        normalized_parts.append(part)
    return len(normalized_parts) >= 2


def metadata_value_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column not in SIGNATURE_META_COLUMNS and not column.startswith("_")]


def choose_feature_sample(feature_metadata_path: Path, sample_size: int = 256) -> list[str]:
    feature_frame = pd.read_parquet(feature_metadata_path)
    feature_names = sorted(feature_frame.index.astype(str).tolist())
    if not feature_names:
        return []
    if len(feature_names) <= sample_size:
        return feature_names
    indices = np.linspace(0, len(feature_names) - 1, num=sample_size, dtype=int)
    return [feature_names[index] for index in np.unique(indices)]


def safe_fraction(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def load_signature_frame(signature_path: Path, feature_metadata_path: Path) -> pd.DataFrame:
    parquet_columns = pq.ParquetFile(signature_path).schema.names
    feature_sample = [name for name in choose_feature_sample(feature_metadata_path) if name in parquet_columns]
    read_columns = [name for name in sorted(SIGNATURE_META_COLUMNS) if name in parquet_columns] + feature_sample
    frame = pd.read_parquet(signature_path, columns=read_columns)
    frame["perturbation_label"] = frame["perturbation_label"].astype(str)
    frame["reference_key"] = frame["reference_key"].astype(str)
    frame["signature_id"] = frame["signature_id"].astype(str)
    frame["dataset_id"] = frame["dataset_id"].astype(str)
    frame["is_control"] = frame["is_control"].astype(bool)
    frame["group_key"] = frame["reference_key"] + "||" + frame["perturbation_label"]
    frame["is_combination"] = frame["perturbation_label"].map(is_combination_label)

    gene_columns = feature_sample
    if gene_columns:
        matrix = frame[gene_columns].to_numpy(dtype=np.float32, copy=False)
        frame["_diag_gene_mean"] = matrix.mean(axis=1)
        frame["_diag_gene_abs_mean"] = np.abs(matrix).mean(axis=1)
        frame["_diag_gene_std"] = matrix.std(axis=1)
        frame["_diag_gene_l2"] = np.linalg.norm(matrix, axis=1)
        frame["_diag_gene_maxabs"] = np.abs(matrix).max(axis=1)
        frame.drop(columns=gene_columns, inplace=True)
    else:
        for column in ["_diag_gene_mean", "_diag_gene_abs_mean", "_diag_gene_std", "_diag_gene_l2", "_diag_gene_maxabs"]:
            frame[column] = np.nan

    return frame


def load_cell_frame(cell_path: Path, qc_path: Path) -> pd.DataFrame:
    cell_frame = pd.read_parquet(cell_path)
    qc_frame = pd.read_parquet(qc_path, columns=["obs_name", "row_index", "passes_contract_filter", "passes_qc", "keep_for_analysis"])
    merged = cell_frame.merge(
        qc_frame,
        on=["obs_name", "row_index"],
        how="left",
        suffixes=("", "_qc"),
    )
    for column in ["passes_contract_filter", "passes_qc", "keep_for_analysis"]:
        qc_column = f"{column}_qc"
        if qc_column in merged.columns:
            merged[column] = merged[qc_column].fillna(merged[column])
            merged.drop(columns=[qc_column], inplace=True)
    merged["keep_for_analysis"] = merged["keep_for_analysis"].fillna(False).astype(bool)
    merged = merged.loc[merged["keep_for_analysis"]].copy()
    merged["dataset_id"] = merged["dataset_id"].astype(str)
    merged["perturbation_label"] = merged["perturbation_label"].astype(str)
    merged["reference_key"] = merged["reference_key"].astype(str)
    merged["group_key"] = merged["group_key"].astype(str)
    merged["is_control"] = merged["is_control"].astype(bool)
    merged["is_combination"] = merged["perturbation_label"].map(is_combination_label)
    return merged


def choose_direct_label_column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    for column in candidates:
        if column not in frame.columns:
            continue
        values = normalize_series(frame[column])
        if values.dropna().nunique() >= 2:
            return column
    return None


def derive_group_label(
    frame: pd.DataFrame,
    candidates: list[str],
    purity_threshold: float = 0.80,
    min_groups: int = 8,
) -> tuple[str | None, pd.DataFrame]:
    for column in candidates:
        if column not in frame.columns:
            continue
        labels = normalize_series(frame[column])
        eligible = frame.loc[labels.notna(), ["group_key"]].copy()
        if eligible.empty:
            continue
        eligible["label"] = labels.loc[labels.notna()].astype(str)
        counts = eligible.groupby(["group_key", "label"], dropna=False).size().rename("n").reset_index()
        totals = counts.groupby("group_key", dropna=False)["n"].sum().rename("total").reset_index()
        counts = counts.merge(totals, on="group_key", how="left")
        counts["purity"] = counts["n"] / counts["total"]
        top = (
            counts.sort_values(["group_key", "purity", "n", "label"], ascending=[True, False, False, True])
            .groupby("group_key", as_index=False)
            .head(1)
            .copy()
        )
        accepted = top.loc[top["purity"] >= purity_threshold, ["group_key", "label", "purity"]].copy()
        if accepted.empty:
            continue
        if accepted["label"].nunique() < 2:
            continue
        if accepted["group_key"].nunique() < min_groups:
            continue
        accepted.rename(columns={"label": f"derived_{column}_label", "purity": f"derived_{column}_purity"}, inplace=True)
        return column, accepted
    return None, pd.DataFrame()


def load_dataset_bundles(summary_path: Path) -> dict[str, DatasetBundle]:
    summary = pd.read_csv(summary_path)
    bundles: dict[str, DatasetBundle] = {}
    for row in summary.to_dict(orient="records"):
        dataset_id = str(row["dataset_id"])
        cell_metadata_path = Path(row["cell_metadata_path"])
        qc_manifest_path = Path(row["qc_manifest_path"])
        feature_metadata_path = Path(row["feature_metadata_path"])
        signature_path = Path(row["signature_path"])

        cell_frame = load_cell_frame(cell_metadata_path, qc_manifest_path)
        signature_frame = load_signature_frame(signature_path, feature_metadata_path)
        signature_frame["split_role"] = str(row["split_role"])

        cell_state_col = choose_direct_label_column(cell_frame, CELL_STATE_CANDIDATES)
        cell_lineage_col = choose_direct_label_column(cell_frame, LINEAGE_CANDIDATES)

        signature_state_source, signature_state_map = derive_group_label(cell_frame, CELL_STATE_CANDIDATES)
        if signature_state_source:
            label_column = f"derived_{signature_state_source}_label"
            purity_column = f"derived_{signature_state_source}_purity"
            signature_frame = signature_frame.merge(signature_state_map, on="group_key", how="left")
            signature_frame.rename(
                columns={
                    label_column: "signature_state_label",
                    purity_column: "signature_state_purity",
                },
                inplace=True,
            )
        else:
            signature_frame["signature_state_label"] = pd.NA
            signature_frame["signature_state_purity"] = np.nan

        signature_lineage_source, signature_lineage_map = derive_group_label(cell_frame, LINEAGE_CANDIDATES)
        if signature_lineage_source:
            label_column = f"derived_{signature_lineage_source}_label"
            purity_column = f"derived_{signature_lineage_source}_purity"
            signature_frame = signature_frame.merge(signature_lineage_map, on="group_key", how="left")
            signature_frame.rename(
                columns={
                    label_column: "signature_lineage_label",
                    purity_column: "signature_lineage_purity",
                },
                inplace=True,
            )
        else:
            signature_frame["signature_lineage_label"] = pd.NA
            signature_frame["signature_lineage_purity"] = np.nan

        bundles[dataset_id] = DatasetBundle(
            dataset_id=dataset_id,
            source_dataset=str(row["source_dataset"]),
            split_role=str(row["split_role"]),
            external_validation_ready=bool(row.get("external_validation_ready", False)),
            cell_metadata_path=cell_metadata_path,
            qc_manifest_path=qc_manifest_path,
            feature_metadata_path=feature_metadata_path,
            signature_path=signature_path,
            cell_frame=cell_frame,
            signature_frame=signature_frame,
            cell_state_col=cell_state_col,
            cell_lineage_col=cell_lineage_col,
            signature_state_col="signature_state_label" if signature_state_source else None,
            signature_lineage_col="signature_lineage_label" if signature_lineage_source else None,
            signature_state_source=signature_state_source,
            signature_lineage_source=signature_lineage_source,
        )
    return bundles


def ordered_unique(values: pd.Series) -> list[str]:
    normalized = normalize_series(values)
    unique = sorted(normalized.dropna().astype(str).unique().tolist())
    return unique


def random_row_split(frame: pd.DataFrame, seed: int, stratify: pd.Series | None = None) -> tuple[pd.Index, pd.Index, pd.Index]:
    if frame.shape[0] < 3:
        raise ValueError("Need at least 3 rows for train/validation/test split.")
    index = frame.index.to_numpy()
    test_count = max(1, int(round(frame.shape[0] * 0.15)))
    test_count = min(test_count, frame.shape[0] - 2)
    stratify_values = stratify if stratify is not None and stratify.nunique(dropna=False) > 1 and stratify.value_counts().min() >= 2 else None
    train_val_idx, test_idx = train_test_split(
        index,
        test_size=test_count,
        random_state=seed,
        stratify=stratify_values.loc[index] if stratify_values is not None else None,
    )
    train_val = frame.loc[train_val_idx]
    val_count = max(1, int(round(train_val.shape[0] * 0.1765)))
    val_count = min(val_count, train_val.shape[0] - 1)
    stratify_values = (
        stratify.loc[train_val.index]
        if stratify is not None and stratify.loc[train_val.index].nunique(dropna=False) > 1 and stratify.loc[train_val.index].value_counts().min() >= 2
        else None
    )
    train_idx, val_idx = train_test_split(
        train_val.index.to_numpy(),
        test_size=val_count,
        random_state=seed + 1000,
        stratify=stratify_values if stratify_values is not None else None,
    )
    return pd.Index(train_idx), pd.Index(val_idx), pd.Index(test_idx)


def choose_holdout_labels(labels: list[str], seed: int, fraction: float = 0.20) -> list[str]:
    if len(labels) < 2:
        raise ValueError("Need at least two unique labels to hold out one.")
    test_count = max(1, int(round(len(labels) * safe_fraction(fraction))))
    test_count = min(test_count, len(labels) - 1)
    _, test_labels = train_test_split(np.array(sorted(labels)), test_size=test_count, random_state=seed)
    return sorted(test_labels.tolist())


def split_remaining(frame: pd.DataFrame, seed: int) -> tuple[pd.Index, pd.Index]:
    if frame.shape[0] < 2:
        raise ValueError("Need at least two rows to create train/validation from remaining data.")
    stratify = None
    if "is_control" in frame.columns:
        candidate = frame["is_control"].astype(str)
        if candidate.nunique(dropna=False) > 1 and candidate.value_counts().min() >= 2:
            stratify = candidate
    train_idx, val_idx, _ = random_row_split(frame.assign(_dummy=0), seed=seed, stratify=stratify)
    return train_idx, val_idx


def row_splits_from_labels(
    frame: pd.DataFrame,
    label_column: str,
    seed: int,
    include_controls_in_train: bool = True,
) -> tuple[pd.Index, pd.Index, pd.Index, dict[str, Any]]:
    if label_column not in frame.columns:
        raise ValueError(f"Missing label column: {label_column}")
    labels = normalize_series(frame[label_column])
    eligible = frame.loc[(~frame["is_control"]) & labels.notna()].copy()
    if eligible.empty:
        raise ValueError(f"No eligible non-control rows for {label_column}.")
    unique_labels = ordered_unique(eligible[label_column])
    if len(unique_labels) < 2:
        raise ValueError(f"Need at least two unique labels in {label_column}.")
    test_labels = choose_holdout_labels(unique_labels, seed=seed)
    if include_controls_in_train:
        test_mask = labels.isin(test_labels) & (~frame["is_control"])
    else:
        test_mask = labels.isin(test_labels)
    test_idx = frame.index[test_mask]
    remaining = frame.loc[~test_mask].copy()
    train_idx, val_idx = split_remaining(remaining, seed=seed + 2000)
    if include_controls_in_train:
        return train_idx, val_idx, test_idx, {"test_labels": test_labels}
    return train_idx, val_idx, test_idx, {"test_labels": test_labels}


def row_splits_from_mask(
    frame: pd.DataFrame,
    test_mask: pd.Series,
    seed: int,
) -> tuple[pd.Index, pd.Index, pd.Index]:
    test_mask = test_mask.reindex(frame.index).fillna(False).astype(bool)
    if test_mask.sum() == 0:
        raise ValueError("Test mask selects zero rows.")
    if test_mask.sum() >= frame.shape[0]:
        raise ValueError("Test mask selects all rows.")
    remaining = frame.loc[~test_mask].copy()
    train_idx, val_idx = split_remaining(remaining, seed=seed + 3000)
    return train_idx, val_idx, frame.index[test_mask]


def build_random_split(frame: pd.DataFrame, seed: int) -> tuple[pd.Index, pd.Index, pd.Index]:
    stratify = frame["is_control"].astype(str) if "is_control" in frame.columns else None
    return random_row_split(frame, seed=seed, stratify=stratify)


def build_perturbation_split(frame: pd.DataFrame, seed: int) -> tuple[pd.Index, pd.Index, pd.Index, dict[str, Any]]:
    return row_splits_from_labels(frame, "perturbation_label", seed=seed)


def build_combination_split(frame: pd.DataFrame, seed: int) -> tuple[pd.Index, pd.Index, pd.Index, dict[str, Any]]:
    combo_mask = frame["is_combination"] & (~frame["is_control"])
    if combo_mask.sum() == 0:
        raise ValueError("No true combinatorial perturbations are available.")
    train_idx, val_idx, test_idx = row_splits_from_mask(frame, combo_mask, seed=seed)
    combo_labels = sorted(frame.loc[test_idx, "perturbation_label"].astype(str).unique().tolist())
    return train_idx, val_idx, test_idx, {"test_labels": combo_labels}


def low_support_group_keys(signature_frame: pd.DataFrame) -> list[str]:
    non_control = signature_frame.loc[~signature_frame["is_control"]].copy()
    if non_control.empty:
        return []
    sorted_support = non_control["n_cells"].sort_values(kind="mergesort")
    cutoff_position = max(0, int(round(non_control.shape[0] * 0.20)) - 1)
    cutoff_position = min(cutoff_position, non_control.shape[0] - 1)
    cutoff = float(sorted_support.iloc[cutoff_position])
    keys = non_control.loc[non_control["n_cells"] <= cutoff, "group_key"].astype(str).unique().tolist()
    return sorted(keys)


def build_low_support_split(
    frame: pd.DataFrame,
    group_keys: list[str],
    seed: int,
) -> tuple[pd.Index, pd.Index, pd.Index, dict[str, Any]]:
    if not group_keys:
        raise ValueError("No low-support group keys were identified.")
    mask = frame["group_key"].astype(str).isin(group_keys) & (~frame["is_control"])
    train_idx, val_idx, test_idx = row_splits_from_mask(frame, mask, seed=seed)
    return train_idx, val_idx, test_idx, {"test_group_keys": group_keys}


def build_dataset_heldout_split(
    frame: pd.DataFrame,
    heldout_dataset: str,
    seed: int,
) -> tuple[pd.Index, pd.Index, pd.Index]:
    test_mask = frame["dataset_id"].astype(str).eq(heldout_dataset)
    return row_splits_from_mask(frame, test_mask, seed=seed)


def build_external_holdout_split(frame: pd.DataFrame, external_dataset: str, seed: int) -> tuple[pd.Index, pd.Index, pd.Index]:
    test_mask = frame["dataset_id"].astype(str).eq(external_dataset)
    return row_splits_from_mask(frame, test_mask, seed=seed)


def set_overlap(values_a: pd.Series, values_b: pd.Series) -> int:
    set_a = {value for value in normalize_series(values_a).dropna().astype(str).tolist()}
    set_b = {value for value in normalize_series(values_b).dropna().astype(str).tolist()}
    return len(set_a & set_b)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = float(np.linalg.norm(a))
    b_norm = float(np.linalg.norm(b))
    if a_norm == 0.0 or b_norm == 0.0:
        return float("nan")
    cosine = float(np.dot(a, b) / (a_norm * b_norm))
    return float(1.0 - cosine)


def centroid_distance(frame: pd.DataFrame, train_idx: pd.Index, test_idx: pd.Index) -> tuple[float, float]:
    feature_columns = [column for column in frame.columns if column.startswith("_diag_gene_")]
    if not feature_columns:
        return float("nan"), float("nan")
    train_matrix = frame.loc[train_idx, feature_columns].to_numpy(dtype=np.float32, copy=False)
    test_matrix = frame.loc[test_idx, feature_columns].to_numpy(dtype=np.float32, copy=False)
    if train_matrix.size == 0 or test_matrix.size == 0:
        return float("nan"), float("nan")
    train_centroid = train_matrix.mean(axis=0)
    test_centroid = test_matrix.mean(axis=0)
    return float(np.linalg.norm(train_centroid - test_centroid)), cosine_distance(train_centroid, test_centroid)


def count_unique(frame: pd.DataFrame, column: str | None) -> int:
    if column is None or column not in frame.columns:
        return 0
    return int(normalize_series(frame[column]).dropna().nunique())


def build_split_payload_signature(
    frame: pd.DataFrame,
    train_idx: pd.Index,
    val_idx: pd.Index,
    test_idx: pd.Index,
) -> dict[str, list[str]]:
    return {
        "train_ids": frame.loc[train_idx, "signature_id"].astype(str).tolist(),
        "validation_ids": frame.loc[val_idx, "signature_id"].astype(str).tolist(),
        "test_ids": frame.loc[test_idx, "signature_id"].astype(str).tolist(),
    }


def build_split_payload_cell(
    frame: pd.DataFrame,
    train_idx: pd.Index,
    val_idx: pd.Index,
    test_idx: pd.Index,
) -> dict[str, dict[str, list[int]]]:
    def pack(part: pd.DataFrame) -> dict[str, list[int]]:
        payload: dict[str, list[int]] = {}
        for dataset_id, dataset_frame in part.groupby("dataset_id", sort=True):
            payload[str(dataset_id)] = dataset_frame["row_index"].astype(int).tolist()
        return payload

    return {
        "train_ids": pack(frame.loc[train_idx]),
        "validation_ids": pack(frame.loc[val_idx]),
        "test_ids": pack(frame.loc[test_idx]),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def unsupported_payload(
    split_family: str,
    track: str,
    dataset_scope: str,
    seed: int,
    holdout_variable: str,
    unsupported_reason: str,
    bundle_paths: dict[str, str],
    extra_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "split_family": split_family,
        "track": track,
        "dataset_scope": dataset_scope,
        "seed": seed,
        "status": "unsupported",
        "holdout_variable": holdout_variable,
        "unsupported_reason": unsupported_reason,
        "input_paths": bundle_paths,
        "train_ids": [] if track == "signature" else {},
        "validation_ids": [] if track == "signature" else {},
        "test_ids": [] if track == "signature" else {},
    }
    if extra_meta:
        payload.update(extra_meta)
    return payload


def split_filename(split_family: str, dataset_scope: str, seed: int, track: str, heldout_target: str | None = None) -> str:
    parts = [split_family, dataset_scope]
    if heldout_target:
        parts.append(f"holdout-{heldout_target}")
    parts.append(f"seed{seed}")
    parts.append(track)
    safe = "__".join(parts).replace("/", "_")
    return f"{safe}.json"


def build_audit_row(
    payload: dict[str, Any],
    frame: pd.DataFrame | None,
    train_idx: pd.Index | None,
    val_idx: pd.Index | None,
    test_idx: pd.Index | None,
    state_column: str | None,
    lineage_column: str | None,
    output_path: Path,
) -> dict[str, Any]:
    row = {
        "split_family": payload["split_family"],
        "track": payload["track"],
        "dataset_scope": payload["dataset_scope"],
        "seed": payload["seed"],
        "status": payload["status"],
        "holdout_variable": payload["holdout_variable"],
        "heldout_target": payload.get("heldout_target", ""),
        "unsupported_reason": payload.get("unsupported_reason", ""),
        "output_path": str(output_path),
        "train_units": 0,
        "validation_units": 0,
        "test_units": 0,
        "train_perturbations": 0,
        "validation_perturbations": 0,
        "test_perturbations": 0,
        "train_controls": 0,
        "validation_controls": 0,
        "test_controls": 0,
        "train_combinations": 0,
        "validation_combinations": 0,
        "test_combinations": 0,
        "train_cell_states": 0,
        "validation_cell_states": 0,
        "test_cell_states": 0,
        "train_lineages": 0,
        "validation_lineages": 0,
        "test_lineages": 0,
        "train_reference_keys": 0,
        "validation_reference_keys": 0,
        "test_reference_keys": 0,
        "perturbation_overlap_train_validation": 0,
        "perturbation_overlap_train_test": 0,
        "perturbation_overlap_validation_test": 0,
        "dataset_overlap_train_test": 0,
        "reference_key_overlap_train_test": 0,
        "declared_holdout_overlap_train_test": math.nan,
        "declared_holdout_overlap_train_validation": math.nan,
        "declared_holdout_overlap_validation_test": math.nan,
        "train_test_centroid_l2": math.nan,
        "train_test_centroid_cosine": math.nan,
    }
    if frame is None or train_idx is None or val_idx is None or test_idx is None:
        return row

    parts = {
        "train": frame.loc[train_idx],
        "validation": frame.loc[val_idx],
        "test": frame.loc[test_idx],
    }
    for part_name, part_frame in parts.items():
        row[f"{part_name}_units"] = int(part_frame.shape[0])
        row[f"{part_name}_perturbations"] = int(part_frame["perturbation_label"].astype(str).nunique())
        row[f"{part_name}_controls"] = int(part_frame["is_control"].sum())
        row[f"{part_name}_combinations"] = int(part_frame.loc[part_frame["is_combination"], "perturbation_label"].astype(str).nunique())
        row[f"{part_name}_cell_states"] = count_unique(part_frame, state_column)
        row[f"{part_name}_lineages"] = count_unique(part_frame, lineage_column)
        row[f"{part_name}_reference_keys"] = int(part_frame["reference_key"].astype(str).nunique()) if "reference_key" in part_frame.columns else 0

    row["perturbation_overlap_train_validation"] = set_overlap(parts["train"]["perturbation_label"], parts["validation"]["perturbation_label"])
    row["perturbation_overlap_train_test"] = set_overlap(parts["train"]["perturbation_label"], parts["test"]["perturbation_label"])
    row["perturbation_overlap_validation_test"] = set_overlap(parts["validation"]["perturbation_label"], parts["test"]["perturbation_label"])
    row["dataset_overlap_train_test"] = set_overlap(parts["train"]["dataset_id"], parts["test"]["dataset_id"])
    row["reference_key_overlap_train_test"] = set_overlap(parts["train"]["reference_key"], parts["test"]["reference_key"])

    holdout_variable = payload["holdout_variable"]
    if holdout_variable in frame.columns:
        row["declared_holdout_overlap_train_test"] = set_overlap(parts["train"][holdout_variable], parts["test"][holdout_variable])
        row["declared_holdout_overlap_train_validation"] = set_overlap(parts["train"][holdout_variable], parts["validation"][holdout_variable])
        row["declared_holdout_overlap_validation_test"] = set_overlap(parts["validation"][holdout_variable], parts["test"][holdout_variable])
    elif holdout_variable in {"low_support_group", "group_key"} and "group_key" in frame.columns:
        row["declared_holdout_overlap_train_test"] = set_overlap(parts["train"]["group_key"], parts["test"]["group_key"])
        row["declared_holdout_overlap_train_validation"] = set_overlap(parts["train"]["group_key"], parts["validation"]["group_key"])
        row["declared_holdout_overlap_validation_test"] = set_overlap(parts["validation"]["group_key"], parts["test"]["group_key"])

    if payload["track"] == "signature":
        row["train_test_centroid_l2"], row["train_test_centroid_cosine"] = centroid_distance(frame, train_idx, test_idx)

    return row


def bundle_input_paths(bundle: DatasetBundle) -> dict[str, str]:
    return {
        "cell_metadata_path": str(bundle.cell_metadata_path),
        "qc_manifest_path": str(bundle.qc_manifest_path),
        "feature_metadata_path": str(bundle.feature_metadata_path),
        "signature_path": str(bundle.signature_path),
    }


def render_supported_payload(
    split_family: str,
    track: str,
    dataset_scope: str,
    seed: int,
    holdout_variable: str,
    heldout_target: str | None,
    input_paths: dict[str, str],
    split_payload: dict[str, Any],
    extra_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "split_family": split_family,
        "track": track,
        "dataset_scope": dataset_scope,
        "seed": seed,
        "status": "ready",
        "holdout_variable": holdout_variable,
        "heldout_target": heldout_target or "",
        "unsupported_reason": "",
        "input_paths": input_paths,
    }
    payload.update(split_payload)
    if extra_meta:
        payload.update(extra_meta)
    return payload


def make_track_split(
    frame: pd.DataFrame,
    track: str,
    split_family: str,
    dataset_scope: str,
    seed: int,
    holdout_variable: str,
    build_fn,
    output_dir: Path,
    state_column: str | None,
    lineage_column: str | None,
    input_paths: dict[str, str],
    heldout_target: str | None = None,
    extra_meta: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    output_path = output_dir / split_filename(split_family, dataset_scope, seed, track, heldout_target=heldout_target)
    try:
        result = build_fn()
        details: dict[str, Any] = {}
        if len(result) == 3:
            train_idx, val_idx, test_idx = result
        else:
            train_idx, val_idx, test_idx, details = result
        if track == "signature":
            split_payload = build_split_payload_signature(frame, train_idx, val_idx, test_idx)
        else:
            split_payload = build_split_payload_cell(frame, train_idx, val_idx, test_idx)
            split_payload["id_fields"] = ["dataset_id", "row_index", "obs_name"]
            split_payload["cell_reference_note"] = (
                "Resolve obs_name from the staged cell metadata parquet using dataset_id plus row_index. "
                "The QC manifest remains the reconstruction anchor to the backed source matrix."
            )
        payload = render_supported_payload(
            split_family=split_family,
            track=track,
            dataset_scope=dataset_scope,
            seed=seed,
            holdout_variable=holdout_variable,
            heldout_target=heldout_target,
            input_paths=input_paths,
            split_payload=split_payload,
            extra_meta={**(extra_meta or {}), **details},
        )
        write_json(output_path, payload)
        audit = build_audit_row(payload, frame, train_idx, val_idx, test_idx, state_column, lineage_column, output_path)
        return payload, audit
    except Exception as exc:
        payload = unsupported_payload(
            split_family=split_family,
            track=track,
            dataset_scope=dataset_scope,
            seed=seed,
            holdout_variable=holdout_variable,
            unsupported_reason=str(exc),
            bundle_paths=input_paths,
            extra_meta={**(extra_meta or {}), "heldout_target": heldout_target or ""},
        )
        write_json(output_path, payload)
        audit = build_audit_row(payload, None, None, None, None, state_column, lineage_column, output_path)
        return payload, audit


def support_reason_for_direct_column(frame: pd.DataFrame, column: str | None, label_name: str) -> None:
    if column is None:
        raise ValueError(f"No meaningful {label_name} labels were found.")
    unique_labels = ordered_unique(frame[column])
    if len(unique_labels) < 2:
        raise ValueError(f"Need at least two unique {label_name} labels.")


def generate_dataset_family_splits(bundle: DatasetBundle, seed: int, output_dir: Path) -> list[dict[str, Any]]:
    audits: list[dict[str, Any]] = []
    input_paths = bundle_input_paths(bundle)
    dataset_scope = bundle.dataset_id

    signature = bundle.signature_frame.copy()
    cell = bundle.cell_frame.copy()

    shared_low_support_keys = low_support_group_keys(signature)

    family_specs: list[tuple[str, str, Any, Any, str | None, str | None, dict[str, Any] | None]] = [
        ("random_split", "random_row", lambda: build_random_split(signature, seed=seed), lambda: build_random_split(cell, seed=seed), None, None, None),
        (
            "unseen_perturbation_split",
            "perturbation_label",
            lambda: build_perturbation_split(signature, seed=seed),
            lambda: build_perturbation_split(cell, seed=seed),
            None,
            None,
            None,
        ),
        (
            "unseen_combination_split",
            "perturbation_label",
            lambda: build_combination_split(signature, seed=seed),
            lambda: build_combination_split(cell, seed=seed),
            None,
            None,
            {"combination_detection_note": "True combinations use delimiters in perturbation_label and explicitly exclude MULTI_TARGET."},
        ),
        (
            "low_support_split",
            "group_key",
            lambda: build_low_support_split(signature, shared_low_support_keys, seed=seed),
            lambda: build_low_support_split(cell, shared_low_support_keys, seed=seed),
            None,
            None,
            {"low_support_group_keys": shared_low_support_keys},
        ),
    ]

    for split_family, holdout_variable, signature_fn, cell_fn, _, _, extra in family_specs:
        _, audit = make_track_split(
            frame=signature,
            track="signature",
            split_family=split_family,
            dataset_scope=dataset_scope,
            seed=seed,
            holdout_variable=holdout_variable,
            build_fn=signature_fn,
            output_dir=output_dir,
            state_column=bundle.signature_state_col,
            lineage_column=bundle.signature_lineage_col,
            input_paths=input_paths,
            extra_meta=extra,
        )
        audits.append(audit)
        _, audit = make_track_split(
            frame=cell,
            track="cell",
            split_family=split_family,
            dataset_scope=dataset_scope,
            seed=seed,
            holdout_variable=holdout_variable,
            build_fn=cell_fn,
            output_dir=output_dir,
            state_column=bundle.cell_state_col,
            lineage_column=bundle.cell_lineage_col,
            input_paths=input_paths,
            extra_meta=extra,
        )
        audits.append(audit)

    _, audit = make_track_split(
        frame=signature,
        track="signature",
        split_family="unseen_cell_state_split",
        dataset_scope=dataset_scope,
        seed=seed,
        holdout_variable=bundle.signature_state_col or "signature_state_label",
        build_fn=lambda: (
            support_reason_for_direct_column(signature, bundle.signature_state_col, "signature-track state"),
            row_splits_from_labels(signature, bundle.signature_state_col, seed=seed, include_controls_in_train=False),
        )[1],
        output_dir=output_dir,
        state_column=bundle.signature_state_col,
        lineage_column=bundle.signature_lineage_col,
        input_paths=input_paths,
        extra_meta={"label_source": bundle.signature_state_source or ""},
    )
    audits.append(audit)
    _, audit = make_track_split(
        frame=cell,
        track="cell",
        split_family="unseen_cell_state_split",
        dataset_scope=dataset_scope,
        seed=seed,
        holdout_variable=bundle.cell_state_col or "cell_state",
        build_fn=lambda: (
            support_reason_for_direct_column(cell, bundle.cell_state_col, "cell-track state"),
            row_splits_from_labels(cell, bundle.cell_state_col, seed=seed, include_controls_in_train=False),
        )[1],
        output_dir=output_dir,
        state_column=bundle.cell_state_col,
        lineage_column=bundle.cell_lineage_col,
        input_paths=input_paths,
        extra_meta={"label_source": bundle.cell_state_col or ""},
    )
    audits.append(audit)

    _, audit = make_track_split(
        frame=signature,
        track="signature",
        split_family="unseen_cell_type_or_lineage_split",
        dataset_scope=dataset_scope,
        seed=seed,
        holdout_variable=bundle.signature_lineage_col or "signature_lineage_label",
        build_fn=lambda: (
            support_reason_for_direct_column(signature, bundle.signature_lineage_col, "signature-track lineage"),
            row_splits_from_labels(signature, bundle.signature_lineage_col, seed=seed, include_controls_in_train=False),
        )[1],
        output_dir=output_dir,
        state_column=bundle.signature_state_col,
        lineage_column=bundle.signature_lineage_col,
        input_paths=input_paths,
        extra_meta={"label_source": bundle.signature_lineage_source or ""},
    )
    audits.append(audit)
    _, audit = make_track_split(
        frame=cell,
        track="cell",
        split_family="unseen_cell_type_or_lineage_split",
        dataset_scope=dataset_scope,
        seed=seed,
        holdout_variable=bundle.cell_lineage_col or "cell_lineage",
        build_fn=lambda: (
            support_reason_for_direct_column(cell, bundle.cell_lineage_col, "cell-track lineage"),
            row_splits_from_labels(cell, bundle.cell_lineage_col, seed=seed, include_controls_in_train=False),
        )[1],
        output_dir=output_dir,
        state_column=bundle.cell_state_col,
        lineage_column=bundle.cell_lineage_col,
        input_paths=input_paths,
        extra_meta={"label_source": bundle.cell_lineage_col or ""},
    )
    audits.append(audit)

    return audits


def merge_signature_frames(bundles: dict[str, DatasetBundle]) -> pd.DataFrame:
    frames = []
    for bundle in bundles.values():
        frame = bundle.signature_frame.copy()
        frames.append(frame)
    return pd.concat(frames, axis=0, ignore_index=True)


def merge_cell_frames(bundles: dict[str, DatasetBundle]) -> pd.DataFrame:
    return pd.concat([bundle.cell_frame.copy() for bundle in bundles.values()], axis=0, ignore_index=True)


def generate_cross_dataset_splits(bundles: dict[str, DatasetBundle], seeds: list[int], output_dir: Path) -> list[dict[str, Any]]:
    audits: list[dict[str, Any]] = []
    signature = merge_signature_frames(bundles)
    cell = merge_cell_frames(bundles)

    shared_paths = {dataset_id: bundle_input_paths(bundle) for dataset_id, bundle in bundles.items()}
    dataset_scope = "all_datasets"

    for heldout_dataset in sorted(bundles):
        for seed in seeds:
            _, audit = make_track_split(
                frame=signature,
                track="signature",
                split_family="dataset_heldout_split",
                dataset_scope=dataset_scope,
                seed=seed,
                holdout_variable="dataset_id",
                build_fn=lambda heldout=heldout_dataset, random_seed=seed: build_dataset_heldout_split(signature, heldout, random_seed),
                output_dir=output_dir,
                state_column="signature_state_label",
                lineage_column="signature_lineage_label",
                input_paths={dataset_id: paths["signature_path"] for dataset_id, paths in shared_paths.items()},
                heldout_target=heldout_dataset,
                extra_meta={"dataset_members": sorted(bundles)},
            )
            audits.append(audit)
            _, audit = make_track_split(
                frame=cell,
                track="cell",
                split_family="dataset_heldout_split",
                dataset_scope=dataset_scope,
                seed=seed,
                holdout_variable="dataset_id",
                build_fn=lambda heldout=heldout_dataset, random_seed=seed: build_dataset_heldout_split(cell, heldout, random_seed),
                output_dir=output_dir,
                state_column=choose_direct_label_column(cell, CELL_STATE_CANDIDATES),
                lineage_column=choose_direct_label_column(cell, LINEAGE_CANDIDATES),
                input_paths={dataset_id: paths["cell_metadata_path"] for dataset_id, paths in shared_paths.items()},
                heldout_target=heldout_dataset,
                extra_meta={"dataset_members": sorted(bundles)},
            )
            audits.append(audit)

    external_dataset = next(
        (bundle.dataset_id for bundle in bundles.values() if bundle.split_role == "external_validation" and bundle.external_validation_ready),
        None,
    )
    if external_dataset is None:
        raise ValueError("No active external validation dataset is available for external_holdout generation.")

    for seed in seeds:
        _, audit = make_track_split(
            frame=signature,
            track="signature",
            split_family="external_holdout",
            dataset_scope="internal_vs_external",
            seed=seed,
            holdout_variable="dataset_id",
            build_fn=lambda random_seed=seed: build_external_holdout_split(signature, external_dataset, random_seed),
            output_dir=output_dir,
            state_column="signature_state_label",
            lineage_column="signature_lineage_label",
            input_paths={dataset_id: paths["signature_path"] for dataset_id, paths in shared_paths.items()},
            heldout_target=external_dataset,
            extra_meta={"internal_datasets": sorted([dataset_id for dataset_id in bundles if dataset_id != external_dataset])},
        )
        audits.append(audit)
        _, audit = make_track_split(
            frame=cell,
            track="cell",
            split_family="external_holdout",
            dataset_scope="internal_vs_external",
            seed=seed,
            holdout_variable="dataset_id",
            build_fn=lambda random_seed=seed: build_external_holdout_split(cell, external_dataset, random_seed),
            output_dir=output_dir,
            state_column=choose_direct_label_column(cell, CELL_STATE_CANDIDATES),
            lineage_column=choose_direct_label_column(cell, LINEAGE_CANDIDATES),
            input_paths={dataset_id: paths["cell_metadata_path"] for dataset_id, paths in shared_paths.items()},
            heldout_target=external_dataset,
            extra_meta={"internal_datasets": sorted([dataset_id for dataset_id in bundles if dataset_id != external_dataset])},
        )
        audits.append(audit)

    return audits


def write_benchmark_definition(path: Path, audit_frame: pd.DataFrame, bundles: dict[str, DatasetBundle]) -> None:
    support = (
        audit_frame.loc[audit_frame["status"] == "ready", ["split_family", "track", "dataset_scope"]]
        .drop_duplicates()
        .sort_values(["split_family", "track", "dataset_scope"])
    )
    support_lines = "\n".join(
        f"- `{row.split_family}` / `{row.track}` / `{row.dataset_scope}`"
        for row in support.itertuples(index=False)
    )

    bundle_lines = []
    for dataset_id, bundle in sorted(bundles.items()):
        bundle_lines.append(
            f"- `{dataset_id}`: split role `{bundle.split_role}`; cell-state column `{bundle.cell_state_col or 'unsupported'}`; "
            f"cell-lineage column `{bundle.cell_lineage_col or 'unsupported'}`; "
            f"signature-state source `{bundle.signature_state_source or 'unsupported'}`; "
            f"signature-lineage source `{bundle.signature_lineage_source or 'unsupported'}`."
        )

    content = f"""# Benchmark Definition

Last updated: 2026-04-25 Asia/Shanghai

## Purpose

Phase 04 defines the first reproducible context-stress benchmark split system for the Perturbation Transportability Lens project.

- The `signature` track is the canonical modeling interface for Phase 05 baselines.
- The `cell` track is the audit and support layer used for leakage checks, support accounting, and future cell-level extensions.

All split artifacts are generated from the Phase 03 recovery contract only:

- `*_cell_metadata.parquet`
- `*_qc_manifest.parquet`
- `*_delta_signatures.parquet`
- `results/tables/preprocessing_summary.csv`

## Canonical units

- Signature unit: one row in `*_delta_signatures.parquet`
- Cell unit: one retained cell row in `*_cell_metadata.parquet`, referenced back to the backed source matrix through `dataset_id + row_index` and the QC manifest

## Split families

### random_split

Random train / validation / test partition inside a dataset. This is the low-stress reference split.

### unseen_perturbation_split

The test split holds out complete perturbation labels from training. Validation comes from seen perturbations so that the stress remains concentrated on the test set.

### unseen_combination_split

The test split is restricted to true combinatorial perturbations detected from `perturbation_label` delimiters. Placeholder labels such as `MULTI_TARGET` are excluded.

### unseen_cell_state_split

The test split holds out cell-state labels when meaningful state variation exists.

- On the cell track, the preferred state field order is `cell_context`, then `predictions`, then `supervised_name`.
- On the signature track, state labels are derived from dominant per-group cell annotations with an 0.80 purity threshold. If the dominant-label criterion is not satisfied, the split is marked unsupported rather than silently skipped.

### unseen_cell_type_or_lineage_split

The test split holds out lineage labels when available.

- On the cell track, the preferred lineage field order is `class`, then `supervised_name`, then `cell_context`.
- On the signature track, lineage labels are derived from dominant per-group cell annotations with an 0.80 purity threshold.

### dataset_heldout_split

One dataset is held out as test while the remaining datasets are split into train and validation. This is the main dataset-level transport stress family.

### low_support_split

The test split is defined by low-support perturbation groups identified from signature-level backing cell counts (`n_cells`). The same low-support group keys are then projected onto the cell track.

### external_holdout

`GSE284197_screen` is treated as a dedicated external holdout that never mixes with internal random sampling. This role is separate from, and in addition to, its participation in `dataset_heldout_split`.

## Dataset support notes

{chr(10).join(bundle_lines)}

## Why `GSE284197_screen` has two roles

`GSE284197_screen` is both:

1. a standalone external holdout (`external_holdout`)
2. one of the held-out domains in `dataset_heldout_split`

These two roles answer different questions:

- `external_holdout` asks whether a model trained only on the internal scPerturb pool transfers to an independently sourced dataset.
- `dataset_heldout_split` asks whether dataset identity itself behaves like a transport boundary when all currently adopted datasets are treated symmetrically.

## Supported artifacts observed in this run

{support_lines if support_lines else '- No ready split artifacts were recorded.'}

## Unsupported logic

Any infeasible split family is written explicitly as an `unsupported` JSON artifact and an `unsupported` row in `results/tables/split_audit.csv`. This prevents silent simplification and keeps the benchmark surface auditable.
"""
    path.write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary_path)
    output_dir = Path(args.output_dir)
    audit_path = Path(args.audit_path)
    seeds = [int(seed) for seed in args.seeds]

    output_dir.mkdir(parents=True, exist_ok=True)
    bundles = load_dataset_bundles(summary_path)
    all_audits: list[dict[str, Any]] = []

    print(f"[phase04] loaded {len(bundles)} dataset bundles from {summary_path}")
    for dataset_id, bundle in sorted(bundles.items()):
        print(
            f"[phase04] dataset={dataset_id} cell_rows={bundle.cell_frame.shape[0]} "
            f"signature_rows={bundle.signature_frame.shape[0]} "
            f"cell_state={bundle.cell_state_col or 'unsupported'} "
            f"signature_state={bundle.signature_state_source or 'unsupported'}"
        )
        for seed in seeds:
            all_audits.extend(generate_dataset_family_splits(bundle, seed, output_dir))

    all_audits.extend(generate_cross_dataset_splits(bundles, seeds, output_dir))
    audit_frame = pd.DataFrame(all_audits).sort_values(
        ["split_family", "dataset_scope", "heldout_target", "track", "seed"],
        kind="mergesort",
    )
    audit_frame.to_csv(audit_path, index=False)

    benchmark_path = ROOT / "docs" / "benchmark_definition.md"
    write_benchmark_definition(benchmark_path, audit_frame, bundles)

    ready_count = int((audit_frame["status"] == "ready").sum())
    unsupported_count = int((audit_frame["status"] != "ready").sum())
    print(
        f"[phase04] wrote {ready_count} ready artifacts and {unsupported_count} unsupported records. "
        f"audit={audit_path}"
    )


if __name__ == "__main__":
    main()
