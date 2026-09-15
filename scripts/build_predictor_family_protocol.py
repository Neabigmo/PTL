"""Materialize the source-only predictor-family protocol used by PTL.

This is a concise scientific registry, not a new statistical gate.  It makes
the input budget and the allowed evaluation boundary explicit before the
external model adapters write any prediction tensors.  In particular, it
prevents the source-response ridge reference from being described as
information-matched to identity-conditioned methods such as GEARS.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PREDICTION_ARTIFACT = ROOT / "artifacts" / "source_data" / "frangieh_source_frozen_predictions.npz"


MODEL_ROWS = (
    {
        "model": "bilinear_ridge",
        "family": "linear_low_rank",
        "tier": "reference",
        "analysis_layer": "A_native_source_only",
        "native_source_information": "observed source perturbation response; fixed gene embeddings",
        "strict_information_matched": False,
        "adapter_status": "executed_reference",
        "target_control_cells_allowed": False,
    },
    {
        "model": "gears",
        "family": "graph_prior",
        "tier": "core",
        "analysis_layer": "A_native_source_only; B_identity_matched",
        "native_source_information": "source control expression; perturbation identity; fixed GO graph",
        "strict_information_matched": True,
        "adapter_status": "external_cohort_completed; Frangieh_adapter_pending",
        "target_control_cells_allowed": False,
    },
    {
        "model": "biolord",
        "family": "disentangled_latent",
        "tier": "core",
        "analysis_layer": "A_native_source_only; B_identity_matched",
        "native_source_information": "source control expression; perturbation identity; fixed GO graph",
        "strict_information_matched": True,
        "adapter_status": "compatibility_pending",
        "target_control_cells_allowed": False,
    },
    {
        "model": "perturbnet",
        "family": "conditional_generative",
        "tier": "core",
        "analysis_layer": "A_native_source_only; B_identity_matched",
        "native_source_information": "source control expression; perturbation identity; fixed prior representation",
        "strict_information_matched": True,
        "adapter_status": "compatibility_pending",
        "target_control_cells_allowed": False,
    },
    {
        "model": "scouter",
        "family": "semantic_embedding_generator",
        "tier": "core",
        "analysis_layer": "A_native_source_only; B_identity_matched",
        "native_source_information": "source control expression; perturbation identity; fixed gene-text embedding",
        "strict_information_matched": True,
        "adapter_status": "compatibility_pending",
        "target_control_cells_allowed": False,
    },
    {
        "model": "scgpt",
        "family": "foundation_transformer",
        "tier": "extended_shared_gene",
        "analysis_layer": "A_native_source_only; B_shared_gene_sensitivity",
        "native_source_information": "source control expression; perturbation identity; pretrained checkpoint",
        "strict_information_matched": True,
        "adapter_status": "deferred_to_shared_gene_panel",
        "target_control_cells_allowed": False,
    },
    {
        "model": "cpa_ext",
        "family": "context_aware_latent",
        "tier": "rescue_extension",
        "analysis_layer": "context_aware_rescue_only",
        "native_source_information": "source training data; target control cells allowed only in rescue protocol",
        "strict_information_matched": False,
        "adapter_status": "deferred_to_context_aware_rescue",
        "target_control_cells_allowed": True,
    },
)


def build(root: Path) -> tuple[Path, Path]:
    prediction_path = root / "artifacts" / "source_data" / PREDICTION_ARTIFACT.name
    if not prediction_path.exists():
        raise FileNotFoundError(prediction_path)
    with np.load(prediction_path, allow_pickle=False) as payload:
        labels = payload["perturbation_label"].astype(str)
        folds = payload["fold_id"].astype(int)
        sources = payload["source_environment"].astype(str)
        genes = payload["evaluation_gene_symbols"].astype(str)
        seeds = payload["model_seed"].astype(int)
    if len(labels) != len(folds):
        raise ValueError("source-frozen label and fold vectors disagree")
    output_dir = root / "artifacts" / "manifests"
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = output_dir / "predictor_family_protocol.csv"
    pd.DataFrame(MODEL_ROWS).assign(
        target_perturbed_cells_allowed=False,
        target_statistics_allowed=False,
        target_early_stopping_allowed=False,
        source_prediction_frozen_before_target_evaluation=True,
        primary_gene_panel_count=len(genes),
        primary_perturbation_count=len(labels),
        primary_fold_count=len(np.unique(folds)),
    ).to_csv(protocol_path, index=False)
    fold_path = output_dir / "predictor_family_source_only_folds.csv"
    pd.DataFrame(
        [
            {"source_environment": source, "perturbation_label": label, "outer_fold": int(fold)}
            for source in sources for label, fold in zip(labels, folds, strict=True)
        ]
    ).to_csv(fold_path, index=False)
    details_path = output_dir / "predictor_family_protocol.json"
    details_path.write_text(json.dumps({
        "purpose": "Predictor-family generality study; not an accuracy leaderboard.",
        "primary_protocol": "A source-context prediction is materialized before any target outcome is read.",
        "comparison_layers": {
            "A_native_source_only": "Each predictor may use its documented source-side native inputs, but no target information.",
            "B_identity_matched": "Only identity-conditioned predictors using source controls and fixed external priors are directly compared.",
            "context_aware_rescue": "Target control cells may be used only in a separately labelled rescue analysis; target perturbed cells remain excluded.",
        },
        "shared_primary_surface": {
            "source_environments": sources.tolist(),
            "perturbation_count": int(len(labels)),
            "gene_count": int(len(genes)),
            "outer_folds": int(len(np.unique(folds))),
            "reference_model_seeds": seeds.tolist(),
        },
        "prediction_artifact": prediction_path.relative_to(root).as_posix(),
        "protocol_table": protocol_path.relative_to(root).as_posix(),
        "fold_table": fold_path.relative_to(root).as_posix(),
    }, indent=2) + "\n", encoding="utf-8")
    return protocol_path, fold_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    protocol, folds = build(args.root.resolve())
    print(protocol)
    print(folds)


if __name__ == "__main__":
    main()
