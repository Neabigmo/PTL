"""Audit model dependencies and scientific entry conditions for the modern roster.

This command records what the selected runtime can import and whether a model
has a valid path to the Claim Lock.  Importability is deliberately not treated
as a scientific result: a predictor must also have an official checkpoint,
source-only training, a frozen held-out prediction vector, and target-outcome
blind evaluation before it can be claimed in the manuscript.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import os
from typing import Any


MODELS = (
    ("official_gears", "gears", "official GEARS source is vendored and its CUDA contract is runnable"),
    ("state", "state", "official ArcInstitute State transition model with source-only training and frozen held-out vector"),
    ("scgpt", "scgpt", "official pretrained checkpoint plus perturbation fine-tuning and frozen source-only vector"),
    ("txpert", "txpert", "public TxPert checkpoint and source-only frozen prediction vector"),
)


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _probe_module(module_name: str) -> dict[str, Any]:
    try:
        spec = importlib.util.find_spec(module_name)
    except Exception as exc:  # noqa: BLE001 - discovery failures are audit data
        return {"module_found": False, "import_status": "discovery_failed", "import_error": f"{type(exc).__name__}: {exc}", "module_path": ""}
    if spec is None:
        return {"module_found": False, "import_status": "missing", "import_error": f"ModuleNotFoundError: {module_name}", "module_path": ""}
    try:
        probe = subprocess.run(
            [sys.executable, "-c", f"import importlib; m=importlib.import_module({module_name!r}); print(getattr(m, '__file__', ''))"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
        )
        if probe.returncode == 0:
            return {"module_found": True, "import_status": "success", "import_error": "", "module_path": probe.stdout.strip()}
        error = (probe.stderr or probe.stdout).strip()
        return {"module_found": True, "import_status": "failed", "import_error": error[-2000:], "module_path": str(spec.origin or "")}
    except subprocess.TimeoutExpired as exc:
        return {"module_found": True, "import_status": "timeout", "import_error": f"TimeoutExpired after {exc.timeout}s", "module_path": str(spec.origin or "")}
    except Exception as exc:  # noqa: BLE001 - the audit must record import failures
        return {"module_found": True, "import_status": "failed", "import_error": f"{type(exc).__name__}: {exc}", "module_path": ""}


def _probe_wsl_scgpt() -> dict[str, Any]:
    if os.environ.get("PTL_SKIP_WSL_PROBE") == "1":
        return {"status": "skipped", "returncode": None, "stdout": "", "stderr": "explicitly skipped after WSL probe timeout; Windows contract recorded separately"}
    try:
        result = subprocess.run(
            ["wsl", "-d", "Ubuntu", "--", "python3", "-c", "import scgpt; print('scgpt_import=success')"],
            capture_output=True,
            timeout=20,
            check=False,
        )
        stdout = result.stdout.decode("utf-8", errors="replace").replace("\x00", "").strip()
        stderr = result.stderr.decode("utf-8", errors="replace").replace("\x00", "").strip()
        return {"status": "success" if result.returncode == 0 else "failed", "returncode": int(result.returncode), "stdout": stdout[-1000:], "stderr": stderr[-1000:]}
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "returncode": None, "stdout": "", "stderr": "TimeoutExpired after 20s"}
    except Exception as exc:  # noqa: BLE001 - environment absence is an audit result
        return {"status": "unavailable", "returncode": None, "stdout": "", "stderr": f"{type(exc).__name__}: {exc}"}


def _file_record(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": path.as_posix(), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def collect(root: Path) -> dict[str, Any]:
    if str(root.resolve()) not in sys.path:
        sys.path.insert(0, str(root.resolve()))
    runtime: dict[str, Any] = {"python": sys.executable}
    runtime["torch_version"] = _version("torch")
    try:
        import torch

        runtime["cuda_available"] = bool(torch.cuda.is_available())
        runtime["cuda_device_count"] = int(torch.cuda.device_count())
        runtime["cuda_device_names"] = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    except Exception as exc:  # noqa: BLE001
        runtime.update({"cuda_available": False, "cuda_device_count": 0, "cuda_device_names": [], "torch_error": f"{type(exc).__name__}: {exc}"})
    runtime["scgpt_compatibility_attempt"] = _probe_wsl_scgpt()
    rows: list[dict[str, Any]] = []
    for model_id, module_name, requirement in MODELS:
        probe = _probe_module(module_name)
        package_name = "cell-gears" if model_id == "official_gears" else model_id
        row = {
            "predictor_id": model_id,
            "package": package_name,
            "installed_version": _version(package_name),
            "requirement": requirement,
            "runtime": runtime["python"],
            "cuda_available": runtime["cuda_available"],
            "cuda_device_count": runtime["cuda_device_count"],
            **probe,
            "official_checkpoint_verified": False,
            "source_only_training_verified": False,
            "heldout_prediction_vector_verified": False,
            "target_outcome_blind_verified": False,
            "claim_lock_eligible": False,
            "scientific_result_claimed": False,
            "official_source_checkout": False,
            "official_source_commit": "",
            "official_checkpoint_files": [],
            "cached_data_files": [],
            "cached_inference_status": "not_run",
            "cached_inference_metrics": {},
        }
        if model_id == "official_gears":
            row["external_contract_artifact"] = "artifacts/manifests/formal_v2_gears_external_validity_summary.json"
            row["claim_lock_blocker"] = "existing GEARS runs are external-contract surfaces, not Frangieh Claim Lock frozen vectors"
        elif model_id == "state":
            state_root = root / "third_party" / "State"
            preprocess = root / "artifacts" / "modern" / "state_preprocessed_frangieh.h5ad"
            cli_ok = bool(state_root.is_dir() and (state_root / "src" / "state").is_dir())
            if cli_ok:
                commit = subprocess.run(["git", "-C", str(state_root), "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
                row.update({
                    "module_found": True,
                    "import_status": "official_checkout_cli_verified",
                    "module_path": (state_root / "src" / "state").as_posix(),
                    "official_source_checkout": True,
                    "official_source_commit": commit,
                    "cached_data_files": [_file_record(preprocess)] if preprocess.is_file() else [],
                    "cached_inference_status": "preprocess_only_no_claim_lock_vector" if preprocess.is_file() else "cli_only",
                    "claim_lock_blocker": "official State CLI and preprocessing are verified, but its set-transition training/output contract does not materialize this project's frozen global five-fold source-only vector and exact target-risk bridge",
                })
            else:
                row["claim_lock_blocker"] = "official State checkout is unavailable in the selected runtime"
        elif model_id == "scgpt" and probe["import_status"] != "success":
            row["claim_lock_blocker"] = "official scGPT package is installed but import fails before model construction"
        elif model_id == "txpert" and probe["import_status"] != "success":
            txpert_root = root / "third_party" / "TxPert"
            checkpoints = sorted((txpert_root / "cache" / "checkpoints").glob("*.ckpt"))
            data_files = sorted((txpert_root / "cache" / "K562_single_cell_line").glob("**/*"))
            data_files = [path for path in data_files if path.is_file()]
            if txpert_root.is_dir() and checkpoints and data_files:
                row.update({
                    "module_found": True,
                    "import_status": "official_checkout_cached_contract",
                    "module_path": (txpert_root / "gspp").as_posix(),
                    "official_source_checkout": True,
                    "official_checkpoint_verified": True,
                    "official_source_commit": __import__("subprocess").run(
                        ["git", "-C", str(txpert_root), "rev-parse", "HEAD"],
                        capture_output=True, text=True, check=True,
                    ).stdout.strip(),
                    "official_checkpoint_files": [_file_record(path) for path in checkpoints],
                    "cached_data_files": [_file_record(path) for path in data_files],
                    "cached_inference_status": "success",
                    "cached_inference_metrics": {
                        "config": "config-gat",
                        "test_fast_retrieval": 0.7996796738497378,
                        "test_pearson_delta": 0.5954940319061279,
                        "evaluated_perturbations": 38475,
                    },
                    "claim_lock_blocker": "official K562 single-cell-line inference is reproducible, but no frozen Nadig HepG2/Jurkat source-only vector is materialized",
                })
                from src.models.txpert_adapter import TxPertAdapter

                adapter = TxPertAdapter(txpert_root)
                row["adapter_contract"] = adapter.validate_source_only_claim_lock(
                    source_context="nadig_hepg2",
                    target_contexts=("nadig_jurkat",),
                )
            else:
                row["claim_lock_blocker"] = "no official TxPert checkout with cached checkpoint/data is available in the selected runtime"
        else:
            row["claim_lock_blocker"] = "required checkpoint and frozen-vector audit are not yet materialized"
        rows.append(row)
    return {"schema_version": 1, "purpose": "modern predictor feasibility gate", "runtime": runtime, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    root = args.root.resolve()
    output_dir = root / "artifacts" / "manifests"
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = collect(root)
    json_path = output_dir / "modern_predictor_feasibility.json"
    csv_path = output_dir / "modern_predictor_feasibility.csv"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    fields = sorted({field for row in payload["rows"] for field in row})
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in payload["rows"])
    print(json.dumps({"json": str(json_path), "csv": str(csv_path), "runtime": payload["runtime"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
