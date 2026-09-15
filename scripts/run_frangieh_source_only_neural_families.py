"""Fit honest source-only neural predictor families on the Frangieh surface.

This script is deliberately separate from the canonical formal-v2 predictor
runner.  For every source environment and outer perturbation-label fold it
uses only source-condition response and source post-expression information.
The target context is never read while fitting a model.  The output arrays
have the same contract as ``frangieh_source_frozen_predictions.npz`` so that
the downstream transport code can compare families on an identical surface.

The two families are:

``source_only_mlp``
    A direct source-only MLP from a ten-dimensional, train-only PCA embedding
    of perturbation post-expression to the evaluation-gene response.

``source_only_latent_mlp``
    The same source-only input, but with a train-only 32-dimensional PCA of
    the response as the prediction target, followed by reconstruction in gene
    space.  The PCA is refit inside every outer fold.

The small inner validation split selects weight decay from the preregistered
grid.  Three independently bootstrapped members are retained per fold.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_formal_v2_controlled_shift_oof import (  # noqa: E402
    _environment_rows,
)
from scripts.run_formal_v2_predictors import (  # noqa: E402
    METADATA_COLUMNS,
    components,
    load_panel,
    read_training_post_expression,
)


ENVIRONMENTS = (
    "frangieh_melanoma_control",
    "frangieh_melanoma_coculture",
    "frangieh_melanoma_ifng",
)
MODEL_SEEDS = (0, 1, 2)
HIDDEN = (256, 512)
LATENT_DIM = 32
INPUT_DIM = 10
WEIGHT_DECAYS = (1e-5, 1e-4, 1e-3)


class DirectMLP(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, HIDDEN[0]),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(HIDDEN[0], HIDDEN[1]),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(HIDDEN[1], output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _device() -> torch.device:
    requested = os.environ.get("PTL_DEVICE", "cuda")
    if requested.startswith("cuda") and torch.cuda.is_available():
        return torch.device(requested)
    return torch.device("cpu")


def _fit_input_pca(post_by_label: dict[str, np.ndarray], labels: np.ndarray) -> tuple[PCA, dict[str, np.ndarray], np.ndarray]:
    train_post = np.stack([post_by_label[str(label)] for label in labels]).astype(np.float32)
    n_components = min(INPUT_DIM, train_post.shape[0], train_post.shape[1])
    pca = PCA(n_components=n_components, svd_solver="full", random_state=0)
    pca.fit(train_post)
    return pca, post_by_label, train_post


def _query_input(
    pca: PCA,
    post_by_label: dict[str, np.ndarray],
    train_post: np.ndarray,
    train_labels: np.ndarray,
    query_labels: np.ndarray,
) -> np.ndarray:
    global_mean = train_post.mean(axis=0)
    train_set = {str(label) for label in train_labels}
    values: list[np.ndarray] = []
    for label in query_labels.astype(str):
        if label in post_by_label and label in train_set:
            value = post_by_label[label]
        else:
            parts = [post_by_label[p] for p in components(label) if p in post_by_label and p in train_set]
            value = np.mean(parts, axis=0) if parts else global_mean
        values.append(np.asarray(value, dtype=np.float32))
    return pca.transform(np.stack(values)).astype(np.float32)


def _response_scaler(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0).astype(np.float32)
    scale = values.std(axis=0).astype(np.float32)
    scale[~np.isfinite(scale) | (scale < 1e-5)] = 1.0
    return mean, scale


def _fit_pca_target(values: np.ndarray) -> tuple[PCA, np.ndarray]:
    n_components = min(LATENT_DIM, values.shape[0], values.shape[1])
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=0)
    latent = pca.fit_transform(values).astype(np.float32)
    return pca, latent


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _fit_network(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    *,
    weight_decay: float,
    seed: int,
    device: torch.device,
    max_epochs: int,
    patience: int,
) -> tuple[DirectMLP, dict[str, Any]]:
    _seed_everything(seed)
    model = DirectMLP(x_train.shape[1], y_train.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=weight_decay)
    tx = torch.from_numpy(x_train).to(device)
    ty = torch.from_numpy(y_train).to(device)
    vx = torch.from_numpy(x_valid).to(device)
    vy = torch.from_numpy(y_valid).to(device)
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    stale = 0
    best_epoch = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(tx), ty)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            valid_loss = float(torch.nn.functional.mse_loss(model(vx), vy).detach().cpu())
        if valid_loss < best_loss - 1e-7:
            best_loss = valid_loss
            best_epoch = epoch
            stale = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_epoch": best_epoch, "validation_mse": best_loss, "epochs_run": epoch}


def _inner_select_weight_decay(
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    device: torch.device,
    inner_epochs: int,
) -> tuple[float, list[dict[str, Any]]]:
    if len(x) < 5:
        return 1e-4, []
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(x))
    n_valid = max(2, int(round(len(x) * 0.2)))
    valid_idx = order[:n_valid]
    train_idx = order[n_valid:]
    candidates: list[dict[str, Any]] = []
    for index, weight_decay in enumerate(WEIGHT_DECAYS):
        _, info = _fit_network(
            x[train_idx], y[train_idx], x[valid_idx], y[valid_idx],
            weight_decay=weight_decay, seed=seed + index * 997,
            device=device, max_epochs=inner_epochs, patience=min(10, inner_epochs),
        )
        candidates.append({"weight_decay": weight_decay, **info})
    selected = min(candidates, key=lambda row: (row["validation_mse"], row["weight_decay"]))
    return float(selected["weight_decay"]), candidates


def _train_family(
    *,
    family: str,
    frame: pd.DataFrame,
    panel: list[str],
    fold_id: np.ndarray,
    device: torch.device,
    max_epochs: int,
    patience: int,
    inner_epochs: int,
    source_key: str,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    labels = frame["perturbation_label"].astype(str).to_numpy()
    values = frame[panel].to_numpy(dtype=np.float32, copy=True)
    native_genes = [column for column in frame.columns if column not in METADATA_COLUMNS]
    condition_field = str(frame["condition_field"].iloc[0]) if "condition_field" in frame.columns else ""
    condition = str(frame["condition"].iloc[0]) if "condition" in frame.columns else ""
    dataset_id = str(frame["dataset_id"].iloc[0]) if "dataset_id" in frame.columns else ""
    predictions = np.full((len(MODEL_SEEDS), len(labels), len(panel)), np.nan, dtype=np.float32)
    records: list[dict[str, Any]] = []
    for fold in sorted(np.unique(fold_id).tolist()):
        test_idx = np.flatnonzero(fold_id == fold)
        train_idx = np.flatnonzero(fold_id != fold)
        train_labels = labels[train_idx]
        post_by_label, _ = read_training_post_expression(
            ROOT, dataset_id, set(train_labels.tolist()), panel,
            condition_field=condition_field, condition=condition,
        )
        pca_input, post_by_label, train_post = _fit_input_pca(post_by_label, train_labels)
        x_all = _query_input(pca_input, post_by_label, train_post, train_labels, labels)
        x_train_base = x_all[train_idx]
        x_test = x_all[test_idx]
        y_train_base = values[train_idx]
        target_pca: PCA | None = None
        if family == "source_only_latent_mlp":
            target_pca, y_train_model = _fit_pca_target(y_train_base)
        else:
            target_mean, target_scale = _response_scaler(y_train_base)
            y_train_model = ((y_train_base - target_mean) / target_scale).astype(np.float32)
        weight_decay, selection = _inner_select_weight_decay(
            x_train_base, y_train_model,
            seed=20260907 + fold * 101 + sum(ord(c) for c in source_key),
            device=device, inner_epochs=inner_epochs,
        )
        for member, model_seed in enumerate(MODEL_SEEDS):
            rng = np.random.default_rng(model_seed + 1009 * (fold + 1) + 100000 * sum(ord(c) for c in source_key))
            bootstrap_idx = rng.choice(len(train_idx), size=len(train_idx), replace=True)
            x_boot = x_train_base[bootstrap_idx]
            y_boot = y_train_model[bootstrap_idx]
            # A deterministic 15% source-only validation slice is used for
            # early stopping.  It is not used to score any target outcome.
            order = np.arange(len(x_boot))
            valid_n = max(2, int(round(len(order) * 0.15)))
            valid_idx = order[:valid_n]
            fit_idx = order[valid_n:]
            model, info = _fit_network(
                x_boot[fit_idx], y_boot[fit_idx], x_boot[valid_idx], y_boot[valid_idx],
                weight_decay=weight_decay, seed=20260907 + model_seed + 1000 * fold,
                device=device, max_epochs=max_epochs, patience=patience,
            )
            model.eval()
            with torch.no_grad():
                y_pred_model = model(torch.from_numpy(x_test).to(device)).detach().cpu().numpy()
            if family == "source_only_latent_mlp":
                assert target_pca is not None
                y_pred = target_pca.inverse_transform(y_pred_model)
            else:
                y_pred = y_pred_model * target_scale + target_mean
            predictions[member, test_idx] = y_pred.astype(np.float32)
            records.append({
                "family": family, "source_environment": source_key,
                "fold_id": int(fold), "model_seed": int(model_seed),
                "weight_decay": weight_decay, "input_pca_components": int(pca_input.n_components_),
                "target_pca_components": int(target_pca.n_components_) if target_pca is not None else None,
                "fit": info, "inner_selection": selection,
            })
    if not np.isfinite(predictions).all():
        raise RuntimeError(f"{family}/{source_key} left non-finite OOF predictions")
    return predictions, records


def run(args: argparse.Namespace) -> dict[str, Any]:
    panel = load_panel(ROOT)
    canonical = np.load(ROOT / "artifacts/source_data/frangieh_source_frozen_predictions.npz", allow_pickle=True)
    labels = canonical["perturbation_label"].astype(str)
    fold_id = canonical["fold_id"].astype(np.int16)
    device = _device()
    all_records: list[dict[str, Any]] = []
    results: dict[str, Any] = {"device": str(device), "families": {}}
    for family in ("source_only_mlp", "source_only_latent_mlp"):
        # Persist in the same [source, label, member, gene] order as the
        # canonical source-frozen artifact.  _train_family works in
        # [member, label, gene] order because that is convenient for fitting.
        all_predictions = np.empty((len(ENVIRONMENTS), len(labels), len(MODEL_SEEDS), len(panel)), dtype=np.float32)
        for env_index, source_key in enumerate(ENVIRONMENTS):
            frame, _ = _environment_rows(ROOT, source_key)
            frame = frame.set_index(frame["perturbation_label"].astype(str)).loc[labels].reset_index(drop=True)
            prediction, records = _train_family(
                family=family, frame=frame, panel=panel, fold_id=fold_id,
                device=device, max_epochs=args.max_epochs, patience=args.patience,
                inner_epochs=args.inner_epochs, source_key=source_key,
            )
            all_predictions[env_index] = np.transpose(prediction, (1, 0, 2))
            all_records.extend(records)
        output = ROOT / "artifacts/source_data" / f"frangieh_{family}_predictions.npz"
        np.savez_compressed(
            output, source_environment=np.asarray(ENVIRONMENTS), perturbation_label=labels,
            fold_id=fold_id, model_seed=np.asarray(MODEL_SEEDS, dtype=np.int16),
            prediction=all_predictions, evaluation_gene_symbols=np.asarray(panel),
        )
        family_records = [row for row in all_records if row["family"] == family]
        results["families"][family] = {"prediction_path": output.relative_to(ROOT).as_posix(), "records": len(family_records)}
    manifest = ROOT / "artifacts/manifests" / "frangieh_source_only_neural_families_fit.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({
        "status": "executed", "protocol": "source_only_neural_families_v2",
        "source_environments": list(ENVIRONMENTS), "outer_folds": sorted(np.unique(fold_id).astype(int).tolist()),
        "model_seeds": list(MODEL_SEEDS), "families": ["source_only_mlp", "source_only_latent_mlp"],
        "architecture": {"input": "train-only PCA of source post-expression, 10 components", "hidden": list(HIDDEN), "activation": "GELU", "dropout": 0.1, "optimizer": "AdamW", "learning_rate": 1e-3, "max_epochs": args.max_epochs, "patience": args.patience, "weight_decay_grid": list(WEIGHT_DECAYS)},
        "latent_target": {"family": "source_only_latent_mlp", "basis": "outer-fold train-only PCA of source response", "dimension": LATENT_DIM},
        "target_outcomes_used_during_fit": False,
        "gene_count": len(panel), "records": all_records, "run": results,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    results["manifest"] = manifest.relative_to(ROOT).as_posix()
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--inner-epochs", type=int, default=60)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
