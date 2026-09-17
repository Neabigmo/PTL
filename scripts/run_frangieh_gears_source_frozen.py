"""Train official GEARS under the strict Frangieh source-frozen contract."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
import random
import sys

import anndata as ad
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.run_perturbation_model import (
    ensure_gears_filter_metadata,
    import_gears_stack,
    invalidate_incompatible_gears_coexpression_cache,
    materialize_official_gears_go_subgraph,
    stage_official_gears_resources,
)

INPUT = ROOT / "data/processed/gears_frangieh_source_frozen"
PROTOCOL = ROOT / "artifacts/source_data/frangieh_source_frozen_predictions.npz"
OUTPUT = ROOT / "artifacts/source_data/gears_frangieh_source_frozen"
NOT_IN_OFFICIAL_GO = {
    "C19orf48", "GAS5", "GSEC", "IDI2-AS1", "LEF1-AS1", "LINC00518",
    "LRRC75A-AS1", "NEAT1", "NUP50-AS1", "PSMB8-AS1", "SLC7A5P1",
    "SNHG6", "ST3GAL6-AS1",
}


def run(source: str, *, epochs: int, batch_size: int, device: str) -> dict[str, object]:
    _, PertData, GEARS = import_gears_stack()
    protocol = np.load(PROTOCOL, allow_pickle=False)
    labels = protocol["perturbation_label"].astype(str)
    fold_ids = protocol["fold_id"].astype(int)
    eligible_protocol = ~np.isin(labels, sorted(NOT_IN_OFFICIAL_GO))
    excluded = labels[~eligible_protocol].tolist()
    labels = labels[eligible_protocol]
    fold_ids = fold_ids[eligible_protocol]
    evaluation_genes = protocol["evaluation_gene_symbols"].astype(str)
    adata_path = INPUT / f"{source}.h5ad"
    adata = ad.read_h5ad(adata_path)
    condition_gene = adata.obs["condition"].astype(str).str.replace("+ctrl", "", regex=False)
    adata = adata[~condition_gene.isin(NOT_IN_OFFICIAL_GO).to_numpy()].copy()
    # pandas 3 may materialize HDF5 strings as ArrowStringArray, which
    # anndata 0.10 cannot write back. GEARS only needs ordinary categories.
    adata.obs.index = pd.Index([str(value) for value in adata.obs.index], dtype=object, name=adata.obs.index.name)
    adata.var.index = pd.Index([str(value) for value in adata.var.index], dtype=object, name=adata.var.index.name)
    for column in adata.obs.columns:
        if pd.api.types.is_string_dtype(adata.obs[column].dtype) or isinstance(adata.obs[column].dtype, pd.CategoricalDtype):
            adata.obs[column] = adata.obs[column].astype(str).astype(object)
    for column in adata.var.columns:
        if pd.api.types.is_string_dtype(adata.var[column].dtype) or isinstance(adata.var[column].dtype, pd.CategoricalDtype):
            adata.var[column] = adata.var[column].astype(str).astype(object)

    cache = OUTPUT / "cache" / source
    cache.mkdir(parents=True, exist_ok=True)
    stage_official_gears_resources(cache)
    pert_data = PertData(str(cache), default_pert_graph=False)
    dataset_name = f"strict_goeligible_{source}"
    dataset_path = cache / dataset_name
    if (dataset_path / "perturb_processed.h5ad").is_file():
        pert_data.load(data_path=str(dataset_path))
    else:
        pert_data.new_data_process(dataset_name=dataset_name, adata=adata, skip_calc_de=False)
        ensure_gears_filter_metadata(pert_data.adata)
        active = {g for c in pert_data.adata.obs["condition"].astype(str) if c != "ctrl" for g in c.split("+") if g != "ctrl"}
        materialize_official_gears_go_subgraph(cache, active, preserve_full_copy=False)
        pert_data.create_dataset_file()

    gene_names = pert_data.adata.var["gene_name"].astype(str).to_numpy()
    gene_index = {g: i for i, g in enumerate(gene_names)}
    missing_genes = [g for g in evaluation_genes if g not in gene_index]
    if missing_genes:
        raise RuntimeError(f"evaluation genes missing from GEARS expression panel: {missing_genes[:10]}")
    evaluation_index = np.asarray([gene_index[g] for g in evaluation_genes], dtype=int)
    prediction = np.full((len(labels), len(evaluation_genes)), np.nan, dtype=np.float32)
    fold_records = []

    for fold in range(5):
        seed = 20260916 + fold
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        split_path = INPUT / f"{source}_splits" / f"fold_{fold}.pkl"
        with split_path.open("rb") as handle:
            split = pickle.load(handle)
        split = {
            key: [value for value in values if str(value).replace("+ctrl", "") not in NOT_IN_OFFICIAL_GO]
            for key, values in split.items()
        }
        filtered_split_path = cache / f"goeligible_fold_{fold}.pkl"
        with filtered_split_path.open("wb") as handle:
            pickle.dump(split, handle)
        pert_data.prepare_split(split="custom", seed=seed, split_dict_path=str(filtered_split_path))
        pert_data.get_dataloader(batch_size=batch_size, test_batch_size=batch_size)
        invalidate_incompatible_gears_coexpression_cache(pert_data, set(gene_names))
        model = GEARS(pert_data, device=device, weight_bias_track=False, proj_name="PTL", exp_name=f"{source}_fold{fold}")
        model.model_initialize(hidden_size=64, uncertainty=False)
        try:
            model.train(epochs=epochs, lr=1e-3, weight_decay=5e-4)
        except (ZeroDivisionError, KeyError) as exc:
            if not hasattr(model, "best_model"):
                raise
            print(f"post-test analysis skipped for fold {fold}: {exc}", flush=True)
        test_indices = np.flatnonzero(fold_ids == fold)
        test_labels = labels[test_indices]
        known = set(map(str, model.pert_list))
        missing = [label for label in test_labels if label not in known]
        if missing:
            raise RuntimeError(f"predeclared GO-eligible universe still missing labels: {missing}")
        raw = model.predict([[label] for label in test_labels])
        for index, label in zip(test_indices, test_labels):
            prediction[index] = np.asarray(raw[label], dtype=np.float32)[evaluation_index]
        fold_records.append({"fold": fold, "train": len(split["train"]), "validation": len(split["val"]), "test": len(test_labels)})

    if not np.isfinite(prediction).all():
        raise RuntimeError("GEARS did not materialize the complete GO-eligible fivefold prediction universe")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    npz_path = OUTPUT / f"{source}.npz"
    np.savez_compressed(npz_path, source_environment=np.asarray(source), perturbation_label=labels,
                        fold_id=fold_ids, prediction=prediction, evaluation_gene_symbols=evaluation_genes)
    report = {"status": "executed", "source_environment": source, "model": "official_GEARS",
              "epochs": epochs, "device": device, "labels": len(labels), "excluded_not_in_go": excluded, "genes": len(evaluation_genes),
              "information_contract": "training, validation, graph construction and tuning use source-context cells only; target cells and outcomes are absent",
              "folds": fold_records, "output": npz_path.relative_to(ROOT).as_posix()}
    (OUTPUT / f"{source}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    run(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
