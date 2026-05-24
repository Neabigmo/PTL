from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import pickle
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
TRAINING_LOG_DIR = RESULTS_DIR / "logs" / "training"
MANUAL_INTERVENTION = ROOT / "docs" / "manual_intervention_needed.md"
PHASE_NAME = "Phase 11 reinforcement"

GEARS_DATASETS = {
    "NormanWeissman2019_filtered",
    "ReplogleWeissman2022_K562_essential",
    "ReplogleWeissman2022_rpe1",
}

GEARS_SPLIT_FAMILIES = {
    "random_split",
    "unseen_perturbation_split",
    "unseen_combination_split",
    "dataset_heldout_split",
}

REGISTRY_COLUMNS = [
    "run_id",
    "timestamp",
    "phase",
    "dataset",
    "split",
    "model",
    "seed",
    "command",
    "status",
    "main_metric",
    "output_dir",
    "log_file",
    "notes",
]

SIGNATURE_METADATA_COLUMNS = {
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


class RunLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def log(self, message: str) -> None:
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        print(line, flush=True)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def check_gears_dependencies() -> dict[str, Any]:
    required = ["torch", "torch_geometric", "gears"]
    status = {}
    for module in required:
        status[module] = importlib.util.find_spec(module) is not None
    return {
        "available": all(status.values()),
        "modules": status,
        "missing": [name for name, ok in status.items() if not ok],
    }


def append_manual_intervention(title: str, payload: dict[str, Any]) -> None:
    MANUAL_INTERVENTION.parent.mkdir(parents=True, exist_ok=True)
    with MANUAL_INTERVENTION.open("a", encoding="utf-8") as handle:
        handle.write(f"\n\n## {title}\n\n")
        handle.write(f"- Timestamp: {now_iso()}\n")
        for key, value in payload.items():
            handle.write(f"- {key}: {value}\n")


def append_registry(registry_path: Path, row: dict[str, Any]) -> None:
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([{column: row.get(column, "") for column in REGISTRY_COLUMNS}])
    if registry_path.exists():
        frame.to_csv(registry_path, mode="a", header=False, index=False)
    else:
        frame.to_csv(registry_path, index=False)


def import_gears_stack() -> tuple[Any, Any, Any]:
    import anndata as ad
    from gears import GEARS, PertData

    return ad, PertData, GEARS


def normalize_gears_condition(label: Any, is_control: Any = False) -> str:
    text = str(label)
    control_flag = str(is_control).lower() in {"true", "1", "yes"}
    if control_flag or text.lower() in {"control", "ctrl", "non-targeting", "non_targeting", "ntc"}:
        return "ctrl"
    return text


def perturbation_to_gears_list(label: str) -> list[str]:
    condition = normalize_gears_condition(label)
    if condition == "ctrl":
        return []
    return [part for part in condition.split("_") if part]


def build_signature_metadata(test_frame: pd.DataFrame, row: pd.Series) -> pd.DataFrame:
    columns = [column for column in test_frame.columns if column in SIGNATURE_METADATA_COLUMNS]
    metadata = test_frame[columns].copy()
    metadata["model"] = "gears"
    metadata["split_family"] = row["split_family"]
    metadata["seed"] = int(row["seed"])
    metadata["run_id"] = row["run_id"]
    metadata["gene_space_policy"] = row.get("gene_space_policy", "native")
    return metadata


def build_gears_adata_from_prepared(prepared_data: Any, dataset_id: str) -> Any:
    ad, _, _ = import_gears_stack()
    frames = [
        prepared_data.train_frame.copy(),
        prepared_data.val_frame.copy(),
        prepared_data.test_frame.copy(),
    ]
    arrays = [prepared_data.y_train, prepared_data.y_val, prepared_data.y_test]
    roles = ["train", "val", "test"]
    obs_parts: list[pd.DataFrame] = []
    for frame, role in zip(frames, roles):
        obs = frame[["perturbation_label", "is_control"]].copy()
        obs["condition"] = [
            normalize_gears_condition(label, is_control)
            for label, is_control in zip(obs["perturbation_label"], obs["is_control"])
        ]
        obs["cell_type"] = str(dataset_id)
        obs["split_role"] = role
        obs_parts.append(obs[["condition", "cell_type", "split_role", "perturbation_label", "is_control"]])
    obs_all = pd.concat(obs_parts, ignore_index=True)
    x_all = sparse.csr_matrix(np.vstack([np.asarray(array, dtype=np.float32) for array in arrays]))
    var = pd.DataFrame({"gene_name": [str(gene) for gene in prepared_data.gene_columns]}, index=[str(gene) for gene in prepared_data.gene_columns])
    return ad.AnnData(X=x_all, obs=obs_all, var=var)


def custom_condition_split_from_prepared(prepared_data: Any) -> dict[str, list[str]]:
    split: dict[str, list[str]] = {}
    for role, frame in [
        ("train", prepared_data.train_frame),
        ("val", prepared_data.val_frame),
        ("test", prepared_data.test_frame),
    ]:
        conditions = [
            normalize_gears_condition(label, is_control)
            for label, is_control in zip(frame["perturbation_label"], frame["is_control"])
        ]
        split[role] = sorted(set(conditions))
    for role in ("train", "val"):
        if "ctrl" not in split[role]:
            split[role].append("ctrl")
    return split


def ensure_gears_filter_metadata(adata: Any) -> None:
    if "condition_name" not in adata.obs.columns:
        adata.obs["condition_name"] = adata.obs["condition"].astype(str)
    all_gene_indices = list(range(int(adata.n_vars)))
    condition_names = adata.obs[["condition_name", "condition"]].drop_duplicates()["condition_name"].astype(str).tolist()
    if "non_zeros_gene_idx" not in adata.uns:
        adata.uns["non_zeros_gene_idx"] = {condition_name: all_gene_indices for condition_name in condition_names}
    if "rank_genes_groups_cov_all" not in adata.uns:
        fallback_genes = adata.var["gene_name"].astype(str).head(min(20, int(adata.n_vars))).tolist()
        adata.uns["rank_genes_groups_cov_all"] = {condition_name: fallback_genes for condition_name in condition_names}


def self_loop_graph_tensors(size: int, device: str) -> tuple[Any, Any]:
    import torch

    indices = torch.arange(size, dtype=torch.long, device=device)
    edge_index = torch.stack([indices, indices], dim=0)
    edge_weight = torch.ones(size, dtype=torch.float32, device=device)
    return edge_index, edge_weight


def write_standard_outputs(
    row: pd.Series,
    prepared_data: Any,
    predictions: dict[str, np.ndarray],
    command: str,
    metrics: dict[str, Any],
    model_details: dict[str, Any],
) -> None:
    from src.baselines.run_baseline import write_gene_list, write_json

    output_dir = Path(row["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    y_true = np.asarray(prepared_data.y_test, dtype=np.float32)
    y_pred = np.zeros_like(y_true, dtype=np.float32)
    for idx, record in prepared_data.test_frame.reset_index(drop=True).iterrows():
        condition = normalize_gears_condition(record["perturbation_label"], record["is_control"])
        if condition == "ctrl":
            y_pred[idx] = 0.0
            continue
        if condition not in predictions:
            raise KeyError(f"GEARS did not return prediction for condition {condition!r}")
        y_pred[idx] = np.asarray(predictions[condition], dtype=np.float32)

    np.savez_compressed(output_dir / "test_predictions.npz", y_true=y_true, y_pred=y_pred)
    metadata = build_signature_metadata(prepared_data.test_frame, row)
    metadata["gene_count"] = len(prepared_data.gene_columns)
    metadata.to_parquet(output_dir / "test_metadata.parquet", index=False)
    write_gene_list(output_dir / "genes.txt", [str(gene) for gene in prepared_data.gene_columns])
    write_json(
        output_dir / "run_metrics.json",
        {
            "run_id": row["run_id"],
            "model": "gears",
            "split_family": row["split_family"],
            "dataset_scope": row["dataset_scope"],
            "dataset_id": row["dataset_id"],
            "seed": int(row["seed"]),
            "command": command,
            "metrics": metrics,
            "model_details": model_details,
            "gene_space_policy": row.get("gene_space_policy", "native"),
            "gene_count": len(prepared_data.gene_columns),
            "timestamp": now_iso(),
        },
    )


def run_gears_adapter(row: pd.Series, args: argparse.Namespace, command: str, registry_path: Path) -> None:
    from src.baselines.run_baseline import DataRepository, prepare_split_data, summarize_metrics

    _, PertData, GEARS = import_gears_stack()
    output_dir = Path(row["output_dir"])
    log_file = Path(row["log_file"])
    logger = RunLogger(log_file)
    logger.log("Starting GEARS adapter run.")
    logger.log(f"Command: {command}")
    logger.log(f"Run id: {row['run_id']}")
    logger.log(f"Dataset: {row['dataset_id']}; split: {row['split_family']}; seed: {row['seed']}")

    try:
        repository = DataRepository(TABLES_DIR / "preprocessing_summary.csv")
        prepared = prepare_split_data(repository, Path(row["split_json_path"]))
        adata = build_gears_adata_from_prepared(prepared, str(row["dataset_id"]))
        logger.log(f"Built AnnData for GEARS: cells={adata.n_obs}, genes={adata.n_vars}, conditions={adata.obs['condition'].nunique()}.")
        data_path = output_dir / "gears_data"
        data_path.mkdir(parents=True, exist_ok=True)
        pert_data = PertData(str(data_path), default_pert_graph=False)
        pert_data.new_data_process(dataset_name=str(row["run_id"]).lower(), adata=adata, skip_calc_de=True)
        ensure_gears_filter_metadata(pert_data.adata)
        pert_data.create_dataset_file()

        split_dict = custom_condition_split_from_prepared(prepared)
        split_path = output_dir / "gears_custom_split.pkl"
        with split_path.open("wb") as handle:
            pickle.dump(split_dict, handle)
        pert_data.prepare_split(split="custom", seed=int(row["seed"]), split_dict_path=str(split_path))
        train_conditions = len(split_dict.get("train", []))
        batch_size = max(1, min(int(args.batch_size), train_conditions))
        pert_data.get_dataloader(batch_size=batch_size, test_batch_size=max(1, min(int(args.test_batch_size), max(1, len(split_dict.get("test", []))))))

        gears_model = GEARS(pert_data, device=str(args.device), weight_bias_track=False, proj_name="PTL", exp_name=str(row["run_id"]))
        gene_graph, gene_graph_weight = self_loop_graph_tensors(len(pert_data.gene_names), str(args.device))
        pert_graph, pert_graph_weight = self_loop_graph_tensors(len(pert_data.pert_names), str(args.device))
        gears_model.model_initialize(
            hidden_size=int(args.hidden_size),
            G_coexpress=gene_graph,
            G_coexpress_weight=gene_graph_weight,
            G_go=pert_graph,
            G_go_weight=pert_graph_weight,
        )
        logger.log(f"Training GEARS: epochs={args.epochs}, batch_size={batch_size}, device={args.device}.")
        training_warning = ""
        try:
            stderr_path = output_dir / "gears_train_stderr.log"
            with stderr_path.open("a", encoding="utf-8", errors="replace") as stderr_handle:
                with contextlib.redirect_stderr(stderr_handle):
                    gears_model.train(epochs=int(args.epochs), lr=float(args.lr), weight_decay=float(args.weight_decay))
        except ZeroDivisionError as exc:
            if not hasattr(gears_model, "best_model"):
                raise
            training_warning = f"GEARS post-test deeper_analysis skipped: {exc}"
            logger.log(training_warning)

        requested_conditions = [
            normalize_gears_condition(label, is_control)
            for label, is_control in zip(prepared.test_frame["perturbation_label"], prepared.test_frame["is_control"])
        ]
        known_perts = set(str(pert) for pert in gears_model.pert_list)
        prediction_conditions = []
        missing_graph_conditions = []
        for condition in sorted(set(requested_conditions)):
            if condition == "ctrl":
                continue
            pert_list = perturbation_to_gears_list(condition)
            if all(pert in known_perts for pert in pert_list):
                prediction_conditions.append(condition)
            else:
                missing_graph_conditions.append(condition)
        pert_lists = [perturbation_to_gears_list(condition) for condition in prediction_conditions]
        raw_predictions = gears_model.predict(pert_lists) if pert_lists else {}
        predictions = {condition: np.asarray(value, dtype=np.float32) for condition, value in raw_predictions.items()}
        zero_prediction = np.zeros(len(prepared.gene_columns), dtype=np.float32)
        for condition in missing_graph_conditions:
            predictions[condition] = zero_prediction
        if missing_graph_conditions:
            logger.log(f"GEARS perturbation graph missing {len(missing_graph_conditions)} test conditions; wrote zero-vector fallback for output contract.")

        y_pred_for_metrics = np.zeros_like(prepared.y_test, dtype=np.float32)
        for idx, condition in enumerate(requested_conditions):
            if condition != "ctrl":
                y_pred_for_metrics[idx] = predictions[condition]
        metrics = summarize_metrics(prepared.y_test, y_pred_for_metrics, prepared.test_frame["is_control"].to_numpy(dtype=bool))
        metrics["gene_count"] = int(len(prepared.gene_columns))
        model_details = {
            "adapter": "gears_signature_contract",
            "epochs": int(args.epochs),
            "batch_size": int(batch_size),
            "device": str(args.device),
            "n_conditions": int(adata.obs["condition"].nunique()),
            "n_test_conditions": int(len(pert_lists)),
            "n_missing_perturbation_graph_conditions": int(len(missing_graph_conditions)),
            "missing_perturbation_graph_conditions": missing_graph_conditions[:50],
            "custom_split_path": str(split_path),
            "graph_adapter": "self_loop_fallback",
            "training_warning": training_warning,
        }
        write_standard_outputs(row, prepared, predictions, command, metrics, model_details)
        logger.log(f"GEARS metrics: {json.dumps(metrics, ensure_ascii=False)}")
        append_registry(
            registry_path,
            {
                "run_id": row["run_id"],
                "timestamp": now_iso(),
                "phase": PHASE_NAME,
                "dataset": row["dataset_id"],
                "split": row["split_family"],
                "model": "gears",
                "seed": row["seed"],
                "command": command,
                "status": "success",
                "main_metric": metrics.get("mean_cosine_similarity_non_control_test", ""),
                "output_dir": str(output_dir),
                "log_file": str(log_file),
                "notes": "GEARS adapter completed with project signature-level output contract",
            },
        )
        logger.log("GEARS adapter completed.")
    except Exception as exc:  # noqa: BLE001
        output_dir.mkdir(parents=True, exist_ok=True)
        trace = traceback.format_exc()
        logger.log(f"GEARS adapter failed: {exc}")
        logger.log(trace)
        (output_dir / "adapter_failure.json").write_text(
            json.dumps({"timestamp": now_iso(), "run_id": row["run_id"], "error": str(exc), "traceback": trace}, indent=2),
            encoding="utf-8",
        )
        append_registry(
            registry_path,
            {
                "run_id": row["run_id"],
                "timestamp": now_iso(),
                "phase": PHASE_NAME,
                "dataset": row["dataset_id"],
                "split": row["split_family"],
                "model": "gears",
                "seed": row["seed"],
                "command": command,
                "status": "failed",
                "main_metric": "",
                "output_dir": str(output_dir),
                "log_file": str(log_file),
                "notes": str(exc),
            },
        )
        raise


def load_split_candidates(split_audit_path: Path) -> pd.DataFrame:
    audit = pd.read_csv(split_audit_path)
    required = {"split_family", "track", "dataset_scope", "seed", "status", "output_path"}
    missing = required - set(audit.columns)
    if missing:
        raise ValueError(f"Split audit missing required columns for GEARS run matrix: {sorted(missing)}")
    candidates = audit[
        audit["track"].eq("signature")
        & audit["status"].eq("ready")
        & audit["dataset_scope"].astype(str).isin(GEARS_DATASETS)
        & audit["split_family"].astype(str).isin(GEARS_SPLIT_FAMILIES)
    ].copy()
    return candidates.sort_values(["dataset_scope", "split_family", "seed"]).reset_index(drop=True)


def build_gears_run_matrix(split_audit_path: Path, output_path: Path) -> pd.DataFrame:
    candidates = load_split_candidates(split_audit_path)
    rows: list[dict[str, Any]] = []
    for row in candidates.itertuples(index=False):
        dataset = str(row.dataset_scope)
        split_family = str(row.split_family)
        seed = int(row.seed)
        run_id = f"{split_family}__{dataset}__seed{seed}__signature__gears"
        output_dir = RESULTS_DIR / "models" / run_id
        rows.append(
            {
                "run_id": run_id,
                "model": "gears",
                "dataset_id": dataset,
                "dataset_scope": dataset,
                "split_family": split_family,
                "seed": seed,
                "split_json_path": str(row.output_path),
                "track": "signature",
                "gene_space_policy": "native",
                "output_dir": str(output_dir),
                "prediction_path": str(output_dir / "test_predictions.npz"),
                "metadata_path": str(output_dir / "test_metadata.parquet"),
                "genes_path": str(output_dir / "genes.txt"),
                "log_file": str(TRAINING_LOG_DIR / f"{run_id}.log"),
                "status": "planned",
            }
        )
    frame = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return frame


def _write_blocked_outputs(row: pd.Series, dependency_status: dict[str, Any], command: str, registry_path: Path) -> None:
    output_dir = Path(row["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = Path(row["log_file"])
    logger = RunLogger(log_file)
    logger.log("Starting GEARS-compatible adapter run.")
    logger.log(f"Command: {command}")
    logger.log(f"Run id: {row['run_id']}")
    logger.log(f"Dataset: {row['dataset_id']}; split: {row['split_family']}; seed: {row['seed']}")
    logger.log(f"Dependency status: {json.dumps(dependency_status, sort_keys=True)}")
    logger.log("Blocked before training because mandatory GEARS dependencies are unavailable.")
    (output_dir / "blocked_missing_dependency.json").write_text(
        json.dumps(
            {
                "timestamp": now_iso(),
                "run_id": row["run_id"],
                "missing_dependencies": dependency_status["missing"],
                "required_outputs": ["test_predictions.npz", "test_metadata.parquet", "genes.txt"],
                "status": "blocked_missing_dependency",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    append_registry(
        registry_path,
        {
            "run_id": row["run_id"],
            "timestamp": now_iso(),
            "phase": PHASE_NAME,
            "dataset": row["dataset_id"],
            "split": row["split_family"],
            "model": "gears",
            "seed": row["seed"],
            "command": command,
            "status": "blocked_missing_dependency",
            "main_metric": "",
            "output_dir": str(output_dir),
            "log_file": str(log_file),
            "notes": f"Mandatory GEARS run blocked; missing {', '.join(dependency_status['missing'])}",
        },
    )
    append_manual_intervention(
        "GEARS mandatory model blocked",
        {
            "run_id": row["run_id"],
            "dataset": row["dataset_id"],
            "split": row["split_family"],
            "missing_dependencies": ", ".join(dependency_status["missing"]),
            "recommended_action": "Install a project-compatible GEARS stack, including torch_geometric and gears, then rerun src/models/run_perturbation_model.py run-all --model gears.",
            "log_file": str(log_file),
        },
    )


def run_one(args: argparse.Namespace) -> None:
    if args.model != "gears":
        raise ValueError("Only --model gears is currently supported by this adapter.")
    matrix = pd.read_csv(args.run_matrix)
    if args.run_id:
        matrix = matrix[matrix["run_id"].astype(str).eq(args.run_id)]
    if matrix.empty:
        raise ValueError("No GEARS run rows selected.")
    row = matrix.iloc[0]
    deps = check_gears_dependencies()
    command = " ".join(sys.argv)
    if not deps["available"]:
        _write_blocked_outputs(row, deps, command, Path(args.registry))
        raise RuntimeError(f"Mandatory GEARS dependencies are unavailable: {deps['missing']}")
    run_gears_adapter(row, args, command, Path(args.registry))


def run_all(args: argparse.Namespace) -> None:
    matrix_path = Path(args.run_matrix)
    if not matrix_path.exists():
        build_gears_run_matrix(Path(args.split_audit), matrix_path)
    matrix = pd.read_csv(matrix_path)
    if matrix.empty:
        raise ValueError("GEARS run matrix is empty; expected Norman/Replogle K562/RPE1 ready split rows.")
    deps = check_gears_dependencies()
    command = " ".join(sys.argv)
    blocked = 0
    if not deps["available"]:
        for _, row in matrix.iterrows():
            _write_blocked_outputs(row, deps, command, Path(args.registry))
            blocked += 1
        raise RuntimeError(f"GEARS mandatory model blocked for {blocked} runs; missing {deps['missing']}.")
    selected = matrix.copy()
    if getattr(args, "dataset_scopes", None):
        selected = selected[selected["dataset_scope"].astype(str).isin(args.dataset_scopes)]
    if getattr(args, "split_families", None):
        selected = selected[selected["split_family"].astype(str).isin(args.split_families)]
    if getattr(args, "seeds", None):
        selected = selected[selected["seed"].astype(int).isin(args.seeds)]
    if getattr(args, "run_ids", None):
        selected = selected[selected["run_id"].astype(str).isin(args.run_ids)]
    if not getattr(args, "force", False):
        selected = selected[~selected["output_dir"].map(lambda value: (Path(str(value)) / "run_metrics.json").exists())]
    if getattr(args, "max_runs", 0):
        selected = selected.head(int(args.max_runs))
    if selected.empty:
        print("No GEARS rows selected for execution.")
        return
    for row in selected.itertuples(index=False):
        run_gears_adapter(pd.Series(row._asdict()), args, command, Path(args.registry))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run mandatory perturbation-model adapters for the transportability benchmark.")
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check-deps")
    check.add_argument("--model", default="gears", choices=["gears"])

    build = sub.add_parser("build-run-matrix")
    build.add_argument("--split-audit", default=str(TABLES_DIR / "split_audit.csv"))
    build.add_argument("--output", default=str(TABLES_DIR / "perturbation_model_run_matrix.csv"))

    common: dict[str, Any] = {}
    run_one_parser = sub.add_parser("run-one")
    run_one_parser.add_argument("--model", default="gears", choices=["gears"])
    run_one_parser.add_argument("--run-id", default="")
    run_one_parser.add_argument("--run-matrix", default=str(TABLES_DIR / "perturbation_model_run_matrix.csv"))
    run_one_parser.add_argument("--registry", default=str(TABLES_DIR / "experiment_registry.csv"))
    run_one_parser.add_argument("--split-audit", default=str(TABLES_DIR / "split_audit.csv"))
    run_one_parser.add_argument("--epochs", type=int, default=1)
    run_one_parser.add_argument("--batch-size", type=int, default=16)
    run_one_parser.add_argument("--test-batch-size", type=int, default=64)
    run_one_parser.add_argument("--hidden-size", type=int, default=16)
    run_one_parser.add_argument("--lr", type=float, default=0.001)
    run_one_parser.add_argument("--weight-decay", type=float, default=0.0005)
    run_one_parser.add_argument("--device", default="cpu")

    run_all_parser = sub.add_parser("run-all")
    run_all_parser.add_argument("--model", default="gears", choices=["gears"])
    run_all_parser.add_argument("--run-matrix", default=str(TABLES_DIR / "perturbation_model_run_matrix.csv"))
    run_all_parser.add_argument("--registry", default=str(TABLES_DIR / "experiment_registry.csv"))
    run_all_parser.add_argument("--split-audit", default=str(TABLES_DIR / "split_audit.csv"))
    run_all_parser.add_argument("--dataset-scopes", nargs="*")
    run_all_parser.add_argument("--split-families", nargs="*")
    run_all_parser.add_argument("--seeds", nargs="*", type=int)
    run_all_parser.add_argument("--run-ids", nargs="*")
    run_all_parser.add_argument("--max-runs", type=int, default=0)
    run_all_parser.add_argument("--force", action="store_true")
    run_all_parser.add_argument("--epochs", type=int, default=1)
    run_all_parser.add_argument("--batch-size", type=int, default=16)
    run_all_parser.add_argument("--test-batch-size", type=int, default=64)
    run_all_parser.add_argument("--hidden-size", type=int, default=16)
    run_all_parser.add_argument("--lr", type=float, default=0.001)
    run_all_parser.add_argument("--weight-decay", type=float, default=0.0005)
    run_all_parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "check-deps":
        print(json.dumps(check_gears_dependencies(), indent=2, sort_keys=True))
    elif args.command == "build-run-matrix":
        frame = build_gears_run_matrix(Path(args.split_audit), Path(args.output))
        print(f"Wrote {args.output} with {len(frame)} planned GEARS runs.")
    elif args.command == "run-one":
        run_one(args)
    elif args.command == "run-all":
        run_all(args)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
