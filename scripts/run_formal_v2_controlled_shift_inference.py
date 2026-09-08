"""Run perturbation-level inference for the full-surface Frangieh OOF audit."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_formal_v2_controlled_shift_oof import ENVIRONMENTS, MODEL_SEEDS  # noqa: E402
from scripts.run_formal_v2_reliability_reordering import FRANGIEH_PAIRS  # noqa: E402
from src.evaluation.reordering_inference import (  # noqa: E402
    bootstrap_pairwise_reordering,
    bootstrap_response_rank_association,
    confidence_tracking_permutation_null,
    rank_displacement_summary,
)


BOOTSTRAP_DRAWS = 2000
PERMUTATION_DRAWS = 2000
INFERENCE_SEED = 20260908


def _pair_frame(oof: pd.DataFrame, left_environment: str, right_environment: str) -> pd.DataFrame:
    required = {"environment_key", "perturbation_label", "continuous_risk", "confidence"}
    missing = required.difference(oof.columns)
    if missing:
        raise ValueError(f"OOF predictions are missing required columns: {sorted(missing)}")
    selected = oof.loc[oof["environment_key"].isin([left_environment, right_environment])].copy()
    left = selected.loc[selected["environment_key"].eq(left_environment)].rename(
        columns={"continuous_risk": "risk_left", "confidence": "confidence_left"}
    )
    right = selected.loc[selected["environment_key"].eq(right_environment)].rename(
        columns={"continuous_risk": "risk_right", "confidence": "confidence_right"}
    )
    return left.merge(right, on="perturbation_label", how="inner", validate="one_to_one").sort_values(
        "perturbation_label", kind="stable"
    ).reset_index(drop=True)


def _member_frame(oof: pd.DataFrame, environment_key: str, member_a: int, member_b: int) -> pd.DataFrame:
    columns = [f"member_{member_a}_continuous_risk", f"member_{member_b}_continuous_risk"]
    missing = set(columns).difference(oof.columns)
    if missing:
        raise ValueError(f"OOF predictions are missing member risk columns: {sorted(missing)}")
    frame = oof.loc[oof["environment_key"].eq(environment_key), ["perturbation_label", *columns]].copy()
    frame = frame.rename(columns={columns[0]: "risk_left", columns[1]: "risk_right"})
    if frame["perturbation_label"].duplicated().any():
        raise ValueError(f"duplicate perturbation labels in {environment_key}")
    return frame.sort_values("perturbation_label", kind="stable").reset_index(drop=True)


def run(root: Path, *, bootstrap_draws: int = BOOTSTRAP_DRAWS, permutation_draws: int = PERMUTATION_DRAWS) -> dict[str, Any]:
    manifest_dir = root / "artifacts/manifests"
    oof_path = manifest_dir / "formal_v2_controlled_shift_oof_predictions.csv"
    oof = pd.read_csv(oof_path)
    response_detail = pd.read_csv(manifest_dir / "formal_v2_controlled_shift_oof_perturbations.csv")
    cross_rows: list[dict[str, Any]] = []
    for pair_index, (left_environment, right_environment) in enumerate(FRANGIEH_PAIRS):
        frame = _pair_frame(oof, left_environment, right_environment)
        response_frame = response_detail.loc[
            response_detail["left_environment_id"].eq(left_environment)
            & response_detail["right_environment_id"].eq(right_environment),
            ["perturbation_label", "response_program_shift"],
        ]
        frame = frame.merge(response_frame, on="perturbation_label", how="inner", validate="one_to_one")
        seed = INFERENCE_SEED + 101 * pair_index
        bootstrap = bootstrap_pairwise_reordering(
            frame["risk_left"].to_numpy(),
            frame["risk_right"].to_numpy(),
            frame["confidence_left"].to_numpy(),
            frame["confidence_right"].to_numpy(),
            draws=bootstrap_draws,
            seed=seed,
        )
        null = confidence_tracking_permutation_null(
            frame["risk_left"].to_numpy(),
            frame["risk_right"].to_numpy(),
            frame["confidence_left"].to_numpy(),
            frame["confidence_right"].to_numpy(),
            permutations=permutation_draws,
            seed=seed + 37,
        )
        displacement = rank_displacement_summary(frame["risk_left"].to_numpy(), frame["risk_right"].to_numpy())
        association = bootstrap_response_rank_association(
            frame["response_program_shift"].to_numpy(),
            frame["risk_left"].to_numpy(),
            frame["risk_right"].to_numpy(),
            draws=bootstrap_draws,
            seed=seed + 71,
        )
        cross_rows.append({
            "comparison": "frangieh_condition_oof",
            "left_environment_id": left_environment,
            "right_environment_id": right_environment,
            **bootstrap,
            **displacement,
            **association,
            **null,
            "risk_estimate_unit": "matched perturbation label; pairwise comparisons reconstructed within each draw",
            "confidence_estimate_unit": "matched perturbation label; strict risk inversions",
            "response_rank_association_policy": "computed separately from descriptive response vectors; no outcome-derived feature",
        })

    noise_rows: list[dict[str, Any]] = []
    for environment_index, environment_key in enumerate(ENVIRONMENTS):
        for member_a, member_b in combinations(MODEL_SEEDS, 2):
            frame = _member_frame(oof, environment_key, member_a, member_b)
            seed = INFERENCE_SEED + 1000 + 101 * environment_index + member_a * 11 + member_b
            bootstrap = bootstrap_pairwise_reordering(
                frame["risk_left"].to_numpy(),
                frame["risk_right"].to_numpy(),
                draws=bootstrap_draws,
                seed=seed,
            )
            displacement = rank_displacement_summary(frame["risk_left"].to_numpy(), frame["risk_right"].to_numpy())
            noise_rows.append({
                "environment_id": environment_key,
                "member_a": int(member_a),
                "member_b": int(member_b),
                **bootstrap,
                **displacement,
                "noise_floor_type": "within_context_stochastic_model_member",
                "model_member_contract": "same OOF fold and training labels; independent bootstrap/model seeds",
                "measurement_noise_included": False,
            })

    cross = pd.DataFrame(cross_rows)
    noise = pd.DataFrame(noise_rows)
    cross_path = manifest_dir / "formal_v2_controlled_shift_oof_inference.csv"
    noise_path = manifest_dir / "formal_v2_controlled_shift_noise_floor.csv"
    report_path = manifest_dir / "formal_v2_controlled_shift_inference.json"
    cross.to_csv(cross_path, index=False)
    noise.to_csv(noise_path, index=False)
    payload = {
        "schema_version": 1,
        "benchmark_id": "ptl_context_v2_frangieh_controlled_shift_oof_inference_v1",
        "estimand": "perturbation-level uncertainty for cross-context reliability reordering",
        "bootstrap": {
            "unit": "matched perturbation label",
            "draws": int(bootstrap_draws),
            "seed": int(INFERENCE_SEED),
            "pairwise_reconstruction": True,
        },
        "confidence_permutation_null": {
            "draws": int(permutation_draws),
            "definition": "independent within-environment confidence permutations with observed risks fixed",
        },
        "noise_floor": {
            "type": "within_context_stochastic_model_member",
            "model_seeds": list(MODEL_SEEDS),
            "interpretation": "a model-stochastic instability floor, not a complete measurement-noise ceiling",
        },
        "cross_context_path": cross_path.relative_to(root).as_posix(),
        "noise_floor_path": noise_path.relative_to(root).as_posix(),
        "status": "formal_v2_controlled_shift_scientific_lock_executed",
    }
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--bootstrap-draws", type=int, default=BOOTSTRAP_DRAWS)
    parser.add_argument("--permutation-draws", type=int, default=PERMUTATION_DRAWS)
    args = parser.parse_args()
    print(json.dumps(run(args.root.resolve(), bootstrap_draws=args.bootstrap_draws, permutation_draws=args.permutation_draws), indent=2, ensure_ascii=False, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
