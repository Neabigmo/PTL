"""Build a target-free, descriptive profile for the two neural families.

The canonical 18-row transport table remains the cell-resampled bilinear/RBF
surface.  Neural families are evaluated here on the same frozen labels, genes,
contexts, metrics, and source-frozen prediction contract, but with the
deterministic full-depth risk vectors induced by their three source-side
ensemble members.  This keeps the new comparison honest: it is a family
profile and source-side adequacy analysis, not a claim of equivalence or
superiority over the canonical measurement estimator.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_formal_v2_controlled_shift_oof import ENVIRONMENTS  # noqa: E402
from scripts.run_formal_v2_claim_lock_source_frozen import (  # noqa: E402
    _load_common_surface,
    _metric_vectors,
)
from src.evaluation.ordering_estimands import deterministic_pairwise_order_disagreement  # noqa: E402
from src.evaluation.reordering_inference import normalized_rank_displacement  # noqa: E402


METRICS = ("delta_cosine", "systema_centroid_accuracy", "absolute_effect_rank_agreement")
PAIRS = (
    ("frangieh_melanoma_control", "frangieh_melanoma_coculture"),
    ("frangieh_melanoma_control", "frangieh_melanoma_ifng"),
    ("frangieh_melanoma_coculture", "frangieh_melanoma_ifng"),
)
FAMILIES = {
    "bilinear_ridge": "artifacts/source_data/frangieh_source_frozen_predictions.npz",
    "rbf_kernel_ridge_source_only": "artifacts/source_data/frangieh_source_only_rbf_krr_predictions.npz",
    "source_only_mlp": "artifacts/source_data/frangieh_source_only_mlp_predictions.npz",
    "source_only_latent_mlp": "artifacts/source_data/frangieh_source_only_latent_mlp_predictions.npz",
}
OUTPUT_CSV = "artifacts/manifests/neural_family_transport_profiles.csv"
OUTPUT_JSON = "artifacts/manifests/neural_family_transport_profiles.json"


def _load_prediction(root: Path, relative: str) -> dict[str, Any]:
    with np.load(root / relative, allow_pickle=False) as payload:
        prediction = payload["prediction"].astype(np.float32, copy=True)
        return {
            "prediction": prediction,
            "source_environment": payload["source_environment"].astype(str).tolist(),
            "labels": payload["perturbation_label"].astype(str).tolist(),
            "fold_id": payload["fold_id"].astype(np.int16),
            "genes": payload["evaluation_gene_symbols"].astype(str).tolist(),
        }


def build(root: Path = ROOT) -> tuple[Path, Path, dict[str, Any]]:
    root = root.resolve()
    reference = _load_prediction(root, FAMILIES["bilinear_ridge"])
    if reference["prediction"].shape != (3, 243, 3, 8229):
        raise ValueError(f"unexpected canonical prediction shape {reference['prediction'].shape}")
    panel = reference["genes"]
    labels = reference["labels"]
    common, _, truths = _load_common_surface(root, panel)
    if common != labels:
        raise ValueError("profile labels are not aligned with the frozen source surface")
    rows: list[dict[str, Any]] = []
    source_adequacy: dict[str, dict[str, float]] = {}
    prediction_similarity: dict[str, float] = {}
    reference_mean = reference["prediction"].mean(axis=2)
    for family, relative in FAMILIES.items():
        artifact = _load_prediction(root, relative)
        if artifact["prediction"].shape != reference["prediction"].shape:
            raise ValueError(f"{family} prediction shape does not match the frozen surface")
        if artifact["source_environment"] != reference["source_environment"] or artifact["labels"] != labels or artifact["genes"] != panel:
            raise ValueError(f"{family} prediction metadata is not aligned")
        prediction_mean = artifact["prediction"].mean(axis=2)
        per_source: dict[str, float] = {}
        for source_index, source in enumerate(ENVIRONMENTS):
            same = _metric_vectors(truths[source], artifact["prediction"][source_index], labels, panel)
            per_source[source] = float(np.mean([1.0 - np.mean(values) for values in same.values()]))
        source_adequacy[family] = {"macro_source_fidelity": float(np.mean(list(per_source.values())),), **{f"source_fidelity::{k}": v for k, v in per_source.items()}}
        if family != "bilinear_ridge":
            left = reference_mean.reshape(-1, reference_mean.shape[-1])
            right = prediction_mean.reshape(-1, prediction_mean.shape[-1])
            numer = np.sum(left * right, axis=1)
            denom = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
            cosine = np.divide(numer, denom, out=np.zeros_like(numer), where=denom > 0)
            cosine[(np.linalg.norm(left, axis=1) == 0) & (np.linalg.norm(right, axis=1) == 0)] = 1.0
            prediction_similarity[family] = float(np.mean(cosine))
        for source_index, source in enumerate(ENVIRONMENTS):
            metric_risks = {target: _metric_vectors(truths[target], artifact["prediction"][source_index], labels, panel) for target in ENVIRONMENTS}
            for left_context, right_context in PAIRS:
                for direction, target in ((f"{left_context}->{right_context}", right_context), (f"{right_context}->{left_context}", left_context)):
                    for metric in METRICS:
                        left_risk = metric_risks[left_context][metric]
                        right_risk = metric_risks[right_context][metric]
                        rows.append({
                            "predictor_family": family,
                            "source_environment_id": source,
                            "transfer_id": direction,
                            "left_target_environment_id": left_context,
                            "right_target_environment_id": right_context,
                            "target_environment_id": target,
                            "metric": metric,
                            "d_adj_deterministic_profile": deterministic_pairwise_order_disagreement(left_risk, right_risk),
                            "rank_displacement_profile": float(np.nanmean(normalized_rank_displacement(left_risk, right_risk))),
                            "source_mean_risk": float(np.mean(metric_risks[source][metric])),
                            "target_mean_risk": float(np.mean(metric_risks[target][metric])),
                            "source_target_mean_risk_change": float(np.mean(metric_risks[target][metric]) - np.mean(metric_risks[source][metric])),
                            "profile_surface": "source_frozen_deterministic_full_depth",
                        })
    table = pd.DataFrame(rows).sort_values(["predictor_family", "source_environment_id", "transfer_id", "metric"], kind="stable")
    csv_path = root / OUTPUT_CSV
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(csv_path, index=False)
    report = {
        "schema_version": 1,
        "status": "executed",
        "profile_surface": "same frozen Frangieh labels/genes/contexts/metrics; deterministic full-depth source-frozen risk vectors",
        "families": list(FAMILIES),
        "rows_per_family": int(len(table) // len(FAMILIES)),
        "rows_total": int(len(table)),
        "source_adequacy": source_adequacy,
        "prediction_similarity_to_bilinear": prediction_similarity,
        "target_outcomes_used_for_fitting": False,
        "target_outcomes_used_for_family_selection": False,
        "interpretation": "Descriptive profile only; this artifact does not replace the canonical cell-resampled D_adj surface and does not establish superiority, equivalence, or generalization.",
        "inputs": FAMILIES,
        "csv": OUTPUT_CSV,
    }
    json_path = root / OUTPUT_JSON
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return csv_path, json_path, report


if __name__ == "__main__":
    csv_path, json_path, report = build()
    print(csv_path)
    print(json_path)
    print(json.dumps(report, indent=2, ensure_ascii=False))
