"""Audit duplicate GEARS run families before selecting manuscript evidence.

The repository contains an early one-epoch pilot and a later formal-v2 CUDA
campaign with matching dataset/seed labels but different predictions.  This
audit keeps both families traceable, verifies that they are not pooled as
replicates, and records the formal-v2 family as the sole manuscript source.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PILOT_MANIFEST = ROOT / "artifacts/manifests/gears_pilot_audit.csv"
FORMAL_MANIFEST = ROOT / "artifacts/manifests/formal_v2_gears_external_validity.csv"
FORMAL_ROOT = ROOT / "results/formal_v2/gears_remote"
PILOT_ROOT = ROOT / "results/models"
OUT_CSV = ROOT / "artifacts/manifests/gears_duplicate_run_audit.csv"
OUT_JSON = ROOT / "artifacts/manifests/gears_duplicate_run_audit.json"


def nested_metrics(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload["metrics"]
    details = payload["model_details"]
    return {
        "payload": payload,
        "metrics": metrics,
        "details": details,
        "epochs": int(details["epochs"]),
        "device": str(details["device"]),
        "n_test_rows": int(metrics["n_test_noncontrol"]),
        "cosine": float(metrics["mean_cosine_similarity_non_control_test"]),
        "mse": float(metrics["mse_non_control_test"]),
        "mae": float(metrics["mae_non_control_test"]),
        "gene_count": int(payload["gene_count"]),
        "panel_sha256": str(details["gene_panel_sha256"]),
        "patch_id": str(details["compatibility_patch"]["patch_id"]),
        "patch_hash": str(details["compatibility_patch"]["patch_hash"]),
        "validation_present": bool(
            (path.parent / "validation_predictions.npz").is_file()
            and (path.parent / "validation_metadata.parquet").is_file()
        ),
    }


def main() -> None:
    pilot = pd.read_csv(PILOT_MANIFEST)
    formal = pd.read_csv(FORMAL_MANIFEST)
    keys = ["dataset_id", "seed"]
    pilot = pilot.sort_values(keys).reset_index(drop=True)
    formal = formal.sort_values(keys).reset_index(drop=True)
    if len(pilot) != 9 or len(formal) != 9:
        raise ValueError("Expected exactly 9 pilot and 9 formal-v2 GEARS rows.")
    if set(map(tuple, pilot[keys].to_records(index=False))) != set(
        map(tuple, formal[keys].to_records(index=False))
    ):
        raise ValueError("Pilot and formal-v2 GEARS key sets do not match.")

    rows: list[dict[str, object]] = []
    for (dataset_id, seed), pilot_row in pilot.groupby(keys, sort=True):
        formal_row = formal.loc[
            (formal["dataset_id"] == dataset_id) & (formal["seed"] == seed)
        ].iloc[0]
        pilot_run_id = str(pilot_row.iloc[0]["run_id"])
        formal_run_id = str(formal_row["run_id"])
        pilot_metrics = nested_metrics(PILOT_ROOT / pilot_run_id / "run_metrics.json")
        formal_metrics = nested_metrics(FORMAL_ROOT / formal_run_id / "run_metrics.json")
        same_protocol = all(
            str(pilot_row.iloc[0][column]) == str(formal_row[column])
            for column in ("split_protocol", "perturbation_overlap")
        )
        same_panel = pilot_metrics["panel_sha256"] == formal_metrics["panel_sha256"]
        same_test_rows = pilot_metrics["n_test_rows"] == formal_metrics["n_test_rows"]
        rows.append(
            {
                "dataset_id": dataset_id,
                "seed": int(seed),
                "pilot_run_id": pilot_run_id,
                "formal_v2_run_id": formal_run_id,
                "pilot_output_dir": (PILOT_ROOT / pilot_run_id).relative_to(ROOT).as_posix(),
                "formal_v2_output_dir": (FORMAL_ROOT / formal_run_id).relative_to(ROOT).as_posix(),
                "pilot_epochs": pilot_metrics["epochs"],
                "formal_v2_epochs": formal_metrics["epochs"],
                "pilot_device": pilot_metrics["device"],
                "formal_v2_device": formal_metrics["device"],
                "pilot_test_rows": pilot_metrics["n_test_rows"],
                "formal_v2_test_rows": formal_metrics["n_test_rows"],
                "same_test_rows": int(same_test_rows),
                "same_split_protocol": int(same_protocol),
                "same_gene_panel": int(same_panel),
                "pilot_cosine": pilot_metrics["cosine"],
                "formal_v2_cosine": formal_metrics["cosine"],
                "cosine_delta_formal_minus_pilot": formal_metrics["cosine"] - pilot_metrics["cosine"],
                "pilot_mse": pilot_metrics["mse"],
                "formal_v2_mse": formal_metrics["mse"],
                "pilot_mae": pilot_metrics["mae"],
                "formal_v2_mae": formal_metrics["mae"],
                "formal_v2_validation_present": int(formal_metrics["validation_present"]),
                "canonical_family": "formal_v2",
                "excluded_family": "legacy_one_epoch_pilot",
                "status": "matched_but_not_pooled",
                "reason": (
                    "Same dataset/seed/split/panel labels, but the pilot used one training epoch "
                    "and the formal-v2 campaign used 20 epochs; predictions and metrics differ. "
                    "Formal-v2 is the only GEARS family eligible for Supplementary Tables S10-S11."
                ),
            }
        )

    frame = pd.DataFrame(rows).sort_values(keys).reset_index(drop=True)
    if not frame["same_test_rows"].all() or not frame["same_split_protocol"].all() or not frame["same_gene_panel"].all():
        raise ValueError("Matched GEARS provenance checks failed.")
    if not (frame["pilot_epochs"].eq(1).all() and frame["formal_v2_epochs"].eq(20).all()):
        raise ValueError("Expected one-epoch pilot and 20-epoch formal-v2 runs.")
    if frame["formal_v2_run_id"].duplicated().any():
        raise ValueError("Formal-v2 run IDs must be unique.")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT_CSV, index=False)
    payload = {
        "schema_version": 1,
        "status": "duplicate_gears_families_audited_formal_v2_selected",
        "canonical_family": "formal_v2",
        "excluded_family": "legacy_one_epoch_pilot",
        "matched_rows": int(len(frame)),
        "checks": {
            "same_dataset_seed_keys": True,
            "same_test_row_counts": bool(frame["same_test_rows"].all()),
            "same_split_protocol": bool(frame["same_split_protocol"].all()),
            "same_gene_panels": bool(frame["same_gene_panel"].all()),
            "pilot_epochs": sorted(frame["pilot_epochs"].unique().tolist()),
            "formal_v2_epochs": sorted(frame["formal_v2_epochs"].unique().tolist()),
            "formal_v2_validation_rows": int(frame["formal_v2_validation_present"].sum()),
            "metrics_differ_in_all_rows": bool((frame["cosine_delta_formal_minus_pilot"].abs() > 1e-12).all()),
        },
        "csv_path": OUT_CSV.relative_to(ROOT).as_posix(),
        "rows": frame.to_dict("records"),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8")
    print(json.dumps(payload["checks"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
