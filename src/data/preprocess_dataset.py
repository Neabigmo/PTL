#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import h5py
try:
    import anndata as ad
except ImportError:  # pragma: no cover - exercised when the lightweight H5AD path is used
    ad = None
import numpy as np
import pandas as pd
try:
    import scanpy as sc
except ImportError:  # pragma: no cover - only needed for 10x_h5 input
    sc = None
import yaml
from scipy import sparse


ROOT = Path(__file__).resolve().parents[2]


def _decode_h5_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _read_h5_column(node: Any) -> np.ndarray:
    if isinstance(node, h5py.Group) and "categories" in node and "codes" in node:
        categories = np.asarray([_decode_h5_value(value) for value in node["categories"][:]], dtype=object)
        codes = np.asarray(node["codes"][:], dtype=np.int64)
        output = np.empty(len(codes), dtype=object)
        output[:] = ""
        valid = (codes >= 0) & (codes < len(categories))
        output[valid] = categories[codes[valid]]
        return output
    return np.asarray([_decode_h5_value(value) for value in node[:]])


def _read_h5_frame(group: h5py.Group) -> pd.DataFrame:
    columns = {str(key): _read_h5_column(group[key]) for key in group.keys()}
    return pd.DataFrame(columns)


class _H5SparseMatrix:
    def __init__(self, group: h5py.Group) -> None:
        self.group = group
        self.format = str(group.attrs.get("encoding-type", "csr_matrix")).split("_", 1)[0]
        self.shape = tuple(int(value) for value in group.attrs["shape"])


class _H5BackedAnnData:
    """Minimal backed reader for CSC H5AD files when anndata is unavailable."""

    isbacked = True

    def __init__(self, path: Path) -> None:
        self._handle = h5py.File(path, "r")
        self.X = _H5SparseMatrix(self._handle["X"])
        self.n_obs, self.n_vars = self.X.shape
        self.obs = _read_h5_frame(self._handle["obs"])
        self.var = _read_h5_frame(self._handle["var"])
        if len(self.obs) != self.n_obs or len(self.var) != self.n_vars:
            raise ValueError(f"H5AD metadata shape mismatch in {path}")
        if "cell_barcode" in self.obs:
            self.obs.index = self.obs["cell_barcode"].astype(str)
        else:
            self.obs.index = pd.Index([str(index) for index in range(self.n_obs)])
        if "gene_symbol" in self.var:
            self.var_names = pd.Index(self.var["gene_symbol"].astype(str))
        else:
            self.var_names = pd.Index([str(index) for index in range(self.n_vars)])

    def close(self) -> None:
        self._handle.close()

@dataclass
class DatasetConfig:
    dataset_id: str
    source_dataset: str
    input_path: Path
    input_format: str
    split_role: str
    output_dir: Path
    control_values: list[str]
    obs_fields: dict[str, Any]
    extra_metadata_fields: list[str]
    cell_context_fallbacks: list[str]
    qc: dict[str, Any]
    signature_target_sum: float
    min_cells_per_perturbation: int
    var_name_column: str | None
    write_convenience_adata: bool
    convenience_adata_mode: str
    convenience_var_top_n: int | None
    convenience_var_sort_by: str | None
    label_strategy: str
    label_strategy_kwargs: dict[str, Any]
    contract_filter_query: str | None
    delta_reference_fields: list[str]
    matrix_store_mode: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess a perturbation dataset into staged low-memory artifacts.")
    parser.add_argument("--config", required=True, help="YAML config path")
    return parser.parse_args()


def load_config(path: Path) -> DatasetConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    legacy_mode = raw.get("processed_adata_mode", "none")
    input_path = Path(raw["input_path"])
    output_dir = Path(raw["output_dir"])
    # Historical configs used an absolute checkout path.  Resolve the
    # project-relative data portion when a config is reused from another
    # checkout, while preserving any path that still exists verbatim.
    for original, key in ((input_path, "input_path"), (output_dir, "output_dir")):
        if original.exists():
            continue
        normalized = str(original).replace("\\", "/")
        for marker in ("data/raw/", "data/processed/", "data/external/"):
            if marker in normalized:
                candidate = ROOT / marker.rstrip("/") / normalized.split(marker, 1)[1]
                raw[key] = str(candidate)
                break
    return DatasetConfig(
        dataset_id=raw["dataset_id"],
        source_dataset=raw["source_dataset"],
        input_path=Path(raw["input_path"]),
        input_format=raw.get("input_format", "h5ad"),
        split_role=raw.get("split_role", "train"),
        output_dir=Path(raw["output_dir"]),
        control_values=[str(v).lower() for v in raw.get("control_values", ["control"])],
        obs_fields=raw.get("obs_fields", {}),
        extra_metadata_fields=list(raw.get("extra_metadata_fields", [])),
        cell_context_fallbacks=list(raw.get("cell_context_fallbacks", [])),
        qc=raw.get("qc", {}),
        signature_target_sum=float(raw.get("signature_target_sum", 10000.0)),
        min_cells_per_perturbation=int(raw.get("min_cells_per_perturbation", 10)),
        var_name_column=raw.get("var_name_column"),
        write_convenience_adata=bool(raw.get("write_convenience_adata", raw.get("write_processed_adata", False))),
        convenience_adata_mode=raw.get("convenience_adata_mode", legacy_mode),
        convenience_var_top_n=raw.get("convenience_var_top_n", raw.get("processed_var_top_n")),
        convenience_var_sort_by=raw.get("convenience_var_sort_by", raw.get("processed_var_sort_by")),
        label_strategy=raw.get("label_strategy", "direct"),
        label_strategy_kwargs=raw.get("label_strategy_kwargs", {}),
        contract_filter_query=raw.get("contract_filter_query"),
        delta_reference_fields=list(raw.get("delta_reference_fields", ["batch"])),
        matrix_store_mode=raw.get("matrix_store_mode", "backed_source_with_qc_manifest"),
    )


def make_index_unique(values: Iterable[str]) -> pd.Index:
    counts: dict[str, int] = {}
    unique: list[str] = []
    for value in values:
        base = str(value) if value is not None else "NA"
        seen = counts.get(base, 0)
        if seen == 0:
            unique.append(base)
        else:
            unique.append(f"{base}-{seen}")
        counts[base] = seen + 1
    return pd.Index(unique)


def load_adata(config: DatasetConfig) -> Any:
    if config.input_format == "h5ad":
        if ad is not None:
            return ad.read_h5ad(config.input_path, backed="r")
        return _H5BackedAnnData(config.input_path)
    if config.input_format == "10x_h5":
        if sc is None:
            raise RuntimeError("10x_h5 preprocessing requires scanpy or anndata")
        return sc.read_10x_h5(config.input_path, gex_only=False)
    raise ValueError(f"Unsupported input_format: {config.input_format}")


def resolve_obs_series(obs: pd.DataFrame, field_spec: Any, default_value: str = "NA") -> pd.Series:
    if field_spec is None:
        return pd.Series(default_value, index=obs.index, dtype="object")
    if isinstance(field_spec, str) and field_spec in obs.columns:
        return obs[field_spec].astype("object")
    if isinstance(field_spec, str):
        return pd.Series(field_spec, index=obs.index, dtype="object")
    if isinstance(field_spec, list):
        for candidate in field_spec:
            if candidate in obs.columns:
                return obs[candidate].astype("object")
    return pd.Series(default_value, index=obs.index, dtype="object")


def resolve_cell_context(obs: pd.DataFrame, config: DatasetConfig) -> pd.Series:
    explicit = config.obs_fields.get("cell_context")
    if explicit is not None:
        return resolve_obs_series(obs, explicit, default_value="unknown").fillna("unknown")
    for candidate in config.cell_context_fallbacks:
        if candidate in obs.columns:
            return obs[candidate].astype("object").fillna("unknown")
    return pd.Series("unknown", index=obs.index, dtype="object")


def resolve_perturbation_label(obs: pd.DataFrame, config: DatasetConfig) -> pd.Series:
    if config.label_strategy == "direct":
        return resolve_obs_series(obs, config.obs_fields.get("perturbation"), default_value="unknown").astype(str).fillna("unknown")

    if config.label_strategy == "single_target_with_controls":
        status_field = config.label_strategy_kwargs.get("status_field", config.obs_fields.get("perturbation"))
        single_target_field = config.label_strategy_kwargs["single_target_field"]
        multi_target_label = config.label_strategy_kwargs.get("multi_target_label", "MULTI_TARGET")
        unknown_values = {str(v).lower() for v in config.label_strategy_kwargs.get("unknown_single_target_values", ["NA", "nan"])}

        status = resolve_obs_series(obs, status_field, default_value="unknown").astype(str).fillna("unknown")
        single_target = resolve_obs_series(obs, single_target_field, default_value="NA").astype(str).fillna("NA")
        labels = single_target.copy()
        is_control = status.str.lower().isin(config.control_values)
        invalid_single = single_target.str.lower().isin(unknown_values) | single_target.eq("")
        labels.loc[is_control] = status.loc[is_control]
        labels.loc[~is_control & invalid_single] = multi_target_label
        return labels.astype(str).fillna("unknown")

    raise ValueError(f"Unsupported label_strategy: {config.label_strategy}")


def build_standardized_metadata(adata: ad.AnnData, config: DatasetConfig) -> pd.DataFrame:
    obs = adata.obs.copy()
    metadata = pd.DataFrame(index=obs.index)
    metadata["dataset_id"] = config.dataset_id
    metadata["source_dataset"] = config.source_dataset
    metadata["perturbation_label"] = resolve_perturbation_label(obs, config)
    metadata["is_control"] = metadata["perturbation_label"].str.lower().isin(config.control_values)
    metadata["cell_context"] = resolve_cell_context(obs, config).astype(str).fillna("unknown")
    metadata["batch"] = resolve_obs_series(obs, config.obs_fields.get("batch"), default_value="batch0").astype(str).fillna("batch0")
    metadata["split_role"] = config.split_role
    metadata["obs_name"] = obs.index.astype(str)
    metadata["row_index"] = np.arange(obs.shape[0], dtype=np.int64)

    keep_cols = set(config.extra_metadata_fields)
    keep_cols.update(["ncounts", "ngenes", "percent_mito", "percent_ribo", "guide_id", "gene", "gene_id", "transcript"])
    for key in config.obs_fields.values():
        if isinstance(key, str):
            keep_cols.add(key)
        elif isinstance(key, list):
            keep_cols.update([candidate for candidate in key if isinstance(candidate, str)])
    for column in sorted(keep_cols):
        if column in obs.columns and column not in metadata.columns:
            metadata[column] = obs[column]
    return metadata


def build_qc_mask(metadata: pd.DataFrame, qc: dict[str, Any]) -> pd.Series:
    mask = pd.Series(True, index=metadata.index)
    min_genes = qc.get("min_genes")
    if min_genes is not None and "ngenes" in metadata.columns:
        mask &= pd.to_numeric(metadata["ngenes"], errors="coerce").fillna(0) >= float(min_genes)
    min_counts = qc.get("min_counts")
    if min_counts is not None and "ncounts" in metadata.columns:
        mask &= pd.to_numeric(metadata["ncounts"], errors="coerce").fillna(0) >= float(min_counts)
    max_percent_mito = qc.get("max_percent_mito")
    if max_percent_mito is not None and "percent_mito" in metadata.columns:
        mask &= pd.to_numeric(metadata["percent_mito"], errors="coerce").fillna(0) <= float(max_percent_mito)
    return mask


def build_contract_mask(metadata: pd.DataFrame, config: DatasetConfig) -> pd.Series:
    if not config.contract_filter_query:
        return pd.Series(True, index=metadata.index)
    selected_index = metadata.query(config.contract_filter_query, engine="python").index
    return pd.Series(metadata.index.isin(selected_index), index=metadata.index)


def get_var_frame(adata: ad.AnnData, config: DatasetConfig) -> pd.DataFrame:
    var = adata.var.copy()
    if config.var_name_column and config.var_name_column in var.columns:
        feature_name = var[config.var_name_column].astype(str)
    else:
        feature_name = pd.Index(adata.var_names.astype(str))
    var = var.copy()
    var["feature_name"] = feature_name
    var.index = make_index_unique(var["feature_name"].astype(str))
    var["feature_id"] = adata.var_names.astype(str)
    var["feature_order"] = np.arange(var.shape[0], dtype=np.int64)
    return var


def iter_chunks(adata: ad.AnnData, chunk_size: int):
    if hasattr(adata, "chunked_X"):
        for item in adata.chunked_X(chunk_size):
            if len(item) == 3:
                yield item
            else:
                x_chunk, start = item
                yield x_chunk, start, start + x_chunk.shape[0]
        return
    n_obs = adata.n_obs
    for start in range(0, n_obs, chunk_size):
        end = min(start + chunk_size, n_obs)
        yield adata.X[start:end], start, end


def normalize_chunk(x_chunk: Any, target_sum: float) -> Any:
    if sparse.issparse(x_chunk):
        x_chunk = x_chunk.tocsr().astype(np.float32)
        counts = np.asarray(x_chunk.sum(axis=1)).ravel()
        counts[counts == 0] = 1.0
        x_chunk = x_chunk.multiply(target_sum / counts[:, None]).tocsr()
        x_chunk.data = np.log1p(x_chunk.data)
        return x_chunk
    array = np.asarray(x_chunk, dtype=np.float32)
    counts = array.sum(axis=1)
    counts[counts == 0] = 1.0
    array *= (target_sum / counts)[:, None]
    np.log1p(array, out=array)
    return array


def build_group_metadata(metadata: pd.DataFrame, config: DatasetConfig) -> tuple[pd.DataFrame, np.ndarray]:
    kept = metadata.loc[metadata["keep_for_analysis"]].copy()
    ref_fields = [field for field in config.delta_reference_fields if field in kept.columns]
    if not ref_fields:
        ref_fields = ["dataset_id"]

    group_counts = kept.groupby("group_key", dropna=False).size().rename("n_cells").reset_index()
    group_info = kept[["group_key", "reference_key", "perturbation_label", "is_control"] + ref_fields].drop_duplicates("group_key")
    group_info = group_info.merge(group_counts, on="group_key", how="left")

    eligible = group_info["is_control"] | (group_info["n_cells"] >= config.min_cells_per_perturbation)
    eligible_keys = group_info.loc[eligible, "group_key"].tolist()
    keep_mask = metadata["group_key"].isin(eligible_keys).to_numpy()
    return group_info.loc[eligible].reset_index(drop=True), keep_mask


def compute_profiles(
    adata: ad.AnnData,
    metadata: pd.DataFrame,
    group_info: pd.DataFrame,
    config: DatasetConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if getattr(adata, "isbacked", False) and getattr(adata.X, "format", "") == "csc":
        return compute_profiles_csc_backed(adata, metadata, group_info, config)

    var_frame = get_var_frame(adata, config)
    eligible_keys = group_info["group_key"].tolist()
    key_to_idx = {key: idx for idx, key in enumerate(eligible_keys)}
    sums = np.zeros((len(eligible_keys), adata.n_vars), dtype=np.float64)
    counts = np.zeros(len(eligible_keys), dtype=np.int64)

    full_keep = metadata["group_key"].isin(eligible_keys).to_numpy()
    full_group = metadata["group_key"].astype(str).to_numpy()

    for x_chunk, start, end in iter_chunks(adata, int(config.qc.get("chunk_size", 2048))):
        local_keep = full_keep[start:end]
        if not np.any(local_keep):
            continue
        selected = x_chunk[local_keep]
        normalized = normalize_chunk(selected, config.signature_target_sum)
        chunk_keys = full_group[start:end][local_keep]
        chunk_group_idx = np.fromiter(
            (key_to_idx[key] for key in chunk_keys), dtype=np.int64, count=len(chunk_keys)
        )
        if sparse.issparse(normalized):
            # Aggregate every row in one sparse multiplication. The previous
            # implementation re-sliced the same chunk once per group, which
            # made large sparse screens needlessly slow without changing the
            # resulting pseudobulk sums.
            design = sparse.csr_matrix(
                (np.ones(len(chunk_group_idx), dtype=np.float32),
                 (np.arange(len(chunk_group_idx)), chunk_group_idx)),
                shape=(len(chunk_group_idx), len(eligible_keys)),
            )
            grouped = design.T @ normalized
            sums += grouped.toarray().astype(np.float64, copy=False)
            counts += np.bincount(chunk_group_idx, minlength=len(eligible_keys))
        else:
            for key in np.unique(chunk_keys):
                group_mask = chunk_keys == key
                group_idx = key_to_idx[key]
                sums[group_idx] += normalized[group_mask].sum(axis=0)
                counts[group_idx] += int(group_mask.sum())

    means = np.divide(sums, counts[:, None], out=np.zeros_like(sums), where=counts[:, None] != 0)
    gene_columns = var_frame.index.tolist()

    pseudobulk = group_info.copy()
    pseudobulk["n_cells"] = counts
    pseudobulk.insert(0, "dataset_id", config.dataset_id)
    pseudobulk.insert(1, "source_dataset", config.source_dataset)
    pseudobulk_values = pd.DataFrame(means, index=pseudobulk.index, columns=gene_columns)
    pseudobulk = pd.concat([pseudobulk, pseudobulk_values], axis=1)

    delta_frames: list[pd.DataFrame] = []
    for reference_key, frame in pseudobulk.groupby("reference_key", sort=False):
        control_row = None
        for control_label in config.control_values:
            hit = frame[frame["perturbation_label"].str.lower() == control_label]
            if not hit.empty:
                control_row = hit.iloc[0]
                break
        if control_row is None:
            continue
        control_vector = control_row[gene_columns].to_numpy(dtype=np.float64, copy=False)
        control_label_used = str(control_row["perturbation_label"])
        metadata_columns = [
            "dataset_id",
            "source_dataset",
            "reference_key",
            "perturbation_label",
            "is_control",
            "n_cells",
        ] + [field for field in config.delta_reference_fields if field in frame.columns]
        metadata = frame[metadata_columns].copy()
        metadata.insert(
            2,
            "signature_id",
            [f"{config.dataset_id}__{reference_key}__{label}" for label in frame["perturbation_label"].astype(str)],
        )
        metadata["control_label_used"] = control_label_used
        delta_values = frame[gene_columns].to_numpy(dtype=np.float64, copy=False) - control_vector
        delta_genes = pd.DataFrame(delta_values, columns=gene_columns, index=metadata.index)
        delta_frames.append(pd.concat([metadata, delta_genes], axis=1).reset_index(drop=True))

    delta_signatures = pd.concat(delta_frames, ignore_index=True) if delta_frames else pd.DataFrame()
    return pseudobulk, delta_signatures


def compute_profiles_csc_backed(
    adata: ad.AnnData,
    metadata: pd.DataFrame,
    group_info: pd.DataFrame,
    config: DatasetConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute profiles from a backed CSC matrix in gene blocks.

    H5AD screens are commonly stored as CSC. Row-chunking such a matrix makes
    every chunk repeatedly traverse all gene columns. This path reads each
    gene block once, applies the same library-size/log1p normalization, and
    aggregates rows with one sparse group-design multiplication. It preserves
    the existing signature contract while keeping peak memory independent of
    the full raw expression matrix.
    """
    var_frame = get_var_frame(adata, config)
    gene_columns = var_frame.index.tolist()
    n_cells = int(adata.n_obs)
    n_genes = int(adata.n_vars)
    eligible_keys = group_info["group_key"].tolist()
    key_to_idx = {key: idx for idx, key in enumerate(eligible_keys)}
    keep_mask = metadata["keep_for_analysis"].to_numpy(dtype=bool)
    keep_indices = np.flatnonzero(keep_mask)
    kept_group_idx = np.fromiter(
        (key_to_idx[key] for key in metadata.loc[keep_mask, "group_key"].astype(str)),
        dtype=np.int64,
        count=int(keep_mask.sum()),
    )
    counts = np.bincount(kept_group_idx, minlength=len(eligible_keys)).astype(np.int64, copy=False)

    # The standardized metadata contains the same per-cell library-size field
    # used by the QC contract. For this CSC path it avoids a second full pass
    # over the 740M raw non-zero entries solely to recompute row sums.
    if "ncounts" not in metadata.columns:
        raise ValueError("CSC-backed preprocessing requires ncounts in metadata for normalization")
    cell_counts = pd.to_numeric(metadata["ncounts"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    cell_counts[cell_counts <= 0] = 1.0
    scales = (float(config.signature_target_sum) / cell_counts[keep_indices]).astype(np.float32, copy=False)

    sums = np.zeros((len(eligible_keys), n_genes), dtype=np.float32)
    design = sparse.csr_matrix(
        (np.ones(len(kept_group_idx), dtype=np.float32),
         (np.arange(len(kept_group_idx)), kept_group_idx)),
        shape=(len(kept_group_idx), len(eligible_keys)),
    )
    h5_group = adata.X.group
    data_ds = h5_group["data"]
    indices_ds = h5_group["indices"]
    indptr_ds = h5_group["indptr"]
    gene_chunk_size = int(config.qc.get("gene_chunk_size", 256))

    for gene_start in range(0, n_genes, gene_chunk_size):
        gene_end = min(gene_start + gene_chunk_size, n_genes)
        data_start = int(indptr_ds[gene_start])
        data_end = int(indptr_ds[gene_end])
        indptr = np.asarray(indptr_ds[gene_start:gene_end + 1], dtype=np.int64) - data_start
        data = np.asarray(data_ds[data_start:data_end], dtype=np.float32)
        indices = np.asarray(indices_ds[data_start:data_end], dtype=np.int32)
        block = sparse.csc_matrix((data, indices, indptr), shape=(n_cells, gene_end - gene_start))
        selected = block[keep_indices, :].tocsr()
        selected = selected.multiply(scales[:, None]).tocsr()
        selected.data = np.log1p(selected.data)

        grouped = design.T @ selected
        sums[:, gene_start:gene_end] = grouped.toarray().astype(np.float32, copy=False)

    means = np.divide(
        sums,
        counts[:, None],
        out=np.zeros_like(sums),
        where=counts[:, None] != 0,
    )
    pseudobulk = group_info.copy()
    pseudobulk["n_cells"] = counts
    pseudobulk.insert(0, "dataset_id", config.dataset_id)
    pseudobulk.insert(1, "source_dataset", config.source_dataset)
    pseudobulk_values = pd.DataFrame(means, index=pseudobulk.index, columns=gene_columns)
    pseudobulk = pd.concat([pseudobulk, pseudobulk_values], axis=1)

    delta_frames: list[pd.DataFrame] = []
    for reference_key, frame in pseudobulk.groupby("reference_key", sort=False):
        control_row = None
        for control_label in config.control_values:
            hit = frame[frame["perturbation_label"].str.lower() == control_label]
            if not hit.empty:
                control_row = hit.iloc[0]
                break
        if control_row is None:
            continue
        control_vector = control_row[gene_columns].to_numpy(dtype=np.float32, copy=False)
        control_label_used = str(control_row["perturbation_label"])
        metadata_columns = [
            "dataset_id",
            "source_dataset",
            "reference_key",
            "perturbation_label",
            "is_control",
            "n_cells",
        ] + [field for field in config.delta_reference_fields if field in frame.columns]
        signature_metadata = frame[metadata_columns].copy()
        signature_metadata.insert(
            2,
            "signature_id",
            [f"{config.dataset_id}__{reference_key}__{label}" for label in frame["perturbation_label"].astype(str)],
        )
        signature_metadata["control_label_used"] = control_label_used
        delta_values = frame[gene_columns].to_numpy(dtype=np.float32, copy=False) - control_vector
        delta_genes = pd.DataFrame(delta_values, columns=gene_columns, index=signature_metadata.index)
        delta_frames.append(pd.concat([signature_metadata, delta_genes], axis=1).reset_index(drop=True))

    delta_signatures = pd.concat(delta_frames, ignore_index=True) if delta_frames else pd.DataFrame()
    return pseudobulk, delta_signatures


def choose_convenience_var_idx(adata: ad.AnnData, config: DatasetConfig) -> slice | np.ndarray:
    if not config.convenience_var_top_n:
        return slice(None)
    sort_by = config.convenience_var_sort_by or "mean"
    if sort_by in adata.var.columns:
        scores = pd.to_numeric(adata.var[sort_by], errors="coerce").fillna(0.0).to_numpy()
        order = np.argsort(scores)[::-1][: int(config.convenience_var_top_n)]
        return np.sort(order)
    limit = min(int(config.convenience_var_top_n), adata.n_vars)
    return np.arange(limit)


def write_convenience_copy(adata: ad.AnnData, keep_mask: np.ndarray, output_path: Path, var_idx: slice | np.ndarray) -> None:
    if output_path.exists():
        output_path.unlink()
    view = adata[keep_mask, var_idx]
    view.copy(filename=output_path)


def write_signature_adata(delta_signatures: pd.DataFrame, output_path: Path, reference_fields: list[str]) -> None:
    metadata_columns = [
        "signature_id",
        "dataset_id",
        "source_dataset",
        "reference_key",
        "perturbation_label",
        "is_control",
        "n_cells",
        "control_label_used",
    ] + [field for field in reference_fields if field in delta_signatures.columns]
    gene_columns = [column for column in delta_signatures.columns if column not in metadata_columns]
    obs = delta_signatures[metadata_columns].set_index("signature_id").copy()
    signature_adata = ad.AnnData(
        X=delta_signatures[gene_columns].to_numpy(dtype=np.float32, copy=False),
        obs=obs,
        var=pd.DataFrame(index=gene_columns),
    )
    signature_adata.write_h5ad(output_path, compression="gzip")


def write_matrix_store_manifest(
    adata: ad.AnnData,
    config: DatasetConfig,
    metadata_path: Path,
    qc_manifest_path: Path,
    feature_metadata_path: Path,
    output_path: Path,
) -> None:
    manifest = {
        "dataset_id": config.dataset_id,
        "source_dataset": config.source_dataset,
        "input_path": str(config.input_path),
        "input_format": config.input_format,
        "matrix_store_mode": config.matrix_store_mode,
        "source_matrix_type": type(adata.X).__name__,
        "source_backed": bool(getattr(adata, "isbacked", False)),
        "cell_metadata_path": str(metadata_path),
        "qc_manifest_path": str(qc_manifest_path),
        "feature_metadata_path": str(feature_metadata_path),
        "reconstruction_note": (
            "Open the source matrix in backed mode and subset rows where keep_for_analysis is true "
            "using the row_index and obs_name columns from the QC manifest."
        ),
    }
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def update_summary(summary_path: Path, row: dict[str, Any]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    preferred_columns = [
        "dataset_id",
        "source_dataset",
        "input_path",
        "cell_metadata_path",
        "qc_manifest_path",
        "feature_metadata_path",
        "matrix_store_manifest_path",
        "pseudobulk_path",
        "signature_path",
        "convenience_adata_path",
        "n_cells_input",
        "n_cells_contract",
        "n_cells_kept",
        "n_genes",
        "n_pseudobulk_rows",
        "n_signature_rows",
        "n_unique_perturbations",
        "n_control_cells",
        "split_role",
        "status",
        "matrix_store_mode",
        "convenience_adata_mode",
        "external_validation_ready",
    ]
    frame = pd.DataFrame([row], columns=preferred_columns)
    if summary_path.exists():
        existing = pd.read_csv(summary_path)
        existing = existing.reindex(columns=preferred_columns)
        existing = existing[existing["dataset_id"] != row["dataset_id"]]
        frame = pd.concat([existing, frame], ignore_index=True)
    frame.to_csv(summary_path, index=False)


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config))
    config.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[phase03] loading {config.dataset_id} from {config.input_path}")
    adata = load_adata(config)
    metadata = build_standardized_metadata(adata, config)
    metadata["passes_contract_filter"] = build_contract_mask(metadata, config)
    metadata["passes_qc"] = build_qc_mask(metadata, config.qc)
    metadata["keep_for_analysis"] = metadata["passes_contract_filter"] & metadata["passes_qc"]
    metadata["reference_key"] = ""
    metadata["group_key"] = ""
    kept = metadata.loc[metadata["keep_for_analysis"]].copy()
    if not kept.empty:
        ref_fields = [field for field in config.delta_reference_fields if field in kept.columns]
        if not ref_fields:
            ref_fields = ["dataset_id"]
        kept["reference_key"] = kept[ref_fields].astype(str).agg("||".join, axis=1)
        kept["group_key"] = kept[ref_fields + ["perturbation_label"]].astype(str).agg("||".join, axis=1)
        metadata.loc[kept.index, "reference_key"] = kept["reference_key"]
        metadata.loc[kept.index, "group_key"] = kept["group_key"]
        group_info, eligible_mask = build_group_metadata(metadata, config)
    else:
        group_info = pd.DataFrame()
        eligible_mask = np.zeros(metadata.shape[0], dtype=bool)
    metadata["keep_for_analysis"] = eligible_mask

    var_frame = get_var_frame(adata, config)
    metadata_path = config.output_dir / f"{config.dataset_id}_cell_metadata.parquet"
    qc_manifest_path = config.output_dir / f"{config.dataset_id}_qc_manifest.parquet"
    feature_metadata_path = config.output_dir / f"{config.dataset_id}_feature_metadata.parquet"
    matrix_manifest_path = config.output_dir / f"{config.dataset_id}_matrix_store_manifest.json"
    pseudobulk_path = config.output_dir / f"{config.dataset_id}_pseudobulk.parquet"
    signature_path = config.output_dir / f"{config.dataset_id}_delta_signatures.parquet"

    metadata.to_parquet(metadata_path, index=True)
    qc_manifest = metadata[["obs_name", "row_index", "passes_contract_filter", "passes_qc", "keep_for_analysis", "perturbation_label", "is_control"]].copy()
    qc_manifest.to_parquet(qc_manifest_path, index=True)
    var_frame.to_parquet(feature_metadata_path, index=True)
    write_matrix_store_manifest(adata, config, metadata_path, qc_manifest_path, feature_metadata_path, matrix_manifest_path)

    pseudobulk = pd.DataFrame()
    delta_signatures = pd.DataFrame()
    if group_info.empty:
        pseudobulk = pd.DataFrame(columns=["dataset_id", "source_dataset", "reference_key", "perturbation_label", "is_control", "n_cells"])
        delta_signatures = pd.DataFrame(columns=["dataset_id", "source_dataset", "signature_id", "reference_key", "perturbation_label", "is_control", "n_cells", "control_label_used"])
    else:
        pseudobulk, delta_signatures = compute_profiles(adata, metadata, group_info, config)
    pseudobulk.to_parquet(pseudobulk_path, index=False)
    delta_signatures.to_parquet(signature_path, index=False)

    convenience_path = config.output_dir / f"{config.dataset_id}_convenience_view.h5ad"
    if convenience_path.exists():
        convenience_path.unlink()
    if config.write_convenience_adata:
        print(f"[phase03] writing convenience AnnData to {convenience_path}")
        if config.convenience_adata_mode == "cell":
            write_convenience_copy(adata, metadata["keep_for_analysis"].to_numpy(), convenience_path, choose_convenience_var_idx(adata, config))
        elif config.convenience_adata_mode == "signature":
            write_signature_adata(delta_signatures, convenience_path, config.delta_reference_fields)
        else:
            raise ValueError(f"Unsupported convenience_adata_mode: {config.convenience_adata_mode}")

    summary_row = {
        "dataset_id": config.dataset_id,
        "source_dataset": config.source_dataset,
        "input_path": str(config.input_path),
        "cell_metadata_path": str(metadata_path),
        "qc_manifest_path": str(qc_manifest_path),
        "feature_metadata_path": str(feature_metadata_path),
        "matrix_store_manifest_path": str(matrix_manifest_path),
        "pseudobulk_path": str(pseudobulk_path),
        "signature_path": str(signature_path),
        "convenience_adata_path": str(convenience_path) if config.write_convenience_adata else "",
        "n_cells_input": int(adata.n_obs),
        "n_cells_contract": int(metadata["passes_contract_filter"].sum()),
        "n_cells_kept": int(metadata["keep_for_analysis"].sum()),
        "n_genes": int(adata.n_vars),
        "n_pseudobulk_rows": int(pseudobulk.shape[0]),
        "n_signature_rows": int(delta_signatures.shape[0]),
        "n_unique_perturbations": int(metadata.loc[metadata["keep_for_analysis"], "perturbation_label"].nunique()),
        "n_control_cells": int(metadata.loc[metadata["keep_for_analysis"], "is_control"].sum()),
        "split_role": config.split_role,
        "status": "processed" if int(metadata["keep_for_analysis"].sum()) > 0 else "no_eligible_cells",
        "matrix_store_mode": config.matrix_store_mode,
        "convenience_adata_mode": config.convenience_adata_mode if config.write_convenience_adata else "none",
        "external_validation_ready": bool(config.split_role == "external_validation" and int(metadata["keep_for_analysis"].sum()) > 0),
    }
    update_summary(ROOT / "results" / "tables" / "preprocessing_summary.csv", summary_row)
    print(
        f"[phase03] completed {config.dataset_id}: "
        f"contract={summary_row['n_cells_contract']} kept={summary_row['n_cells_kept']} "
        f"signatures={summary_row['n_signature_rows']}"
    )


if __name__ == "__main__":
    main()
