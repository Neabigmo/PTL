"""Assemble the canonical 18-row comparison for source-frozen families."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_frangieh_frozen_family_measurement import _canonical_surface  # noqa: E402


FAMILY_ENGINE_REPORT = {
    "bilinear_ridge": {
        "summary_rows": "artifacts/manifests/formal_v2_claim_lock_measurement_fullsize_summary.csv",
        "floor_rows": "artifacts/manifests/formal_v2_claim_lock_measurement_fullsize_floors.csv",
        "ordering_rows": "artifacts/manifests/formal_v2_claim_lock_measurement_fullsize_ordering.csv",
    }
}
FAMILY_CANONICAL_FILES = {
    "rbf_krr": "artifacts/manifests/frangieh_source_only_rbf_krr_d_adj_fullsize.csv",
    "source_only_mlp": "artifacts/manifests/predictor_families_v2/source_only_mlp_d_adj_fullsize.csv",
    "source_only_latent_mlp": "artifacts/manifests/predictor_families_v2/source_only_latent_mlp_d_adj_fullsize.csv",
    "official_gears": "artifacts/manifests/predictor_families_v2/official_gears_d_adj_fullsize.csv",
}
EXPECTED_ROWS = 18
METRIC_ORDER = (
    "delta_cosine",
    "systema_centroid_accuracy",
    "absolute_effect_rank_agreement",
)


def _standardize(table: pd.DataFrame, family: str) -> pd.DataFrame:
    rename = {
        "source_environment_id": "source_context",
        "target_environment_id": "target_context",
        "cross_disagreement": "D_cross",
        "source_same_context_floor": "D_within_source",
        "target_same_context_floor": "D_within_target",
        "d_adj": "D_adj",
        "d_adj_ci_low": "ci_lower_90",
        "d_adj_ci_high": "ci_upper_90",
    }
    output = table.rename(columns=rename).copy()
    output["family"] = family
    required = {
        "source_context",
        "target_context",
        "metric",
        "D_adj",
        "ci_lower_90",
        "ci_upper_90",
    }
    missing = sorted(required.difference(output.columns))
    if missing:
        raise ValueError(f"{family} canonical surface is missing {missing}")
    output["transfer_id"] = (
        output["source_context"].astype(str)
        + "->"
        + output["target_context"].astype(str)
    )
    if "minimum_strict_support" not in output:
        output["minimum_strict_support"] = 8
    keep = [
        "family",
        "source_context",
        "target_context",
        "transfer_id",
        "metric",
        "D_cross",
        "D_within_source",
        "D_within_target",
        "D_adj",
        "ci_lower_90",
        "ci_upper_90",
        "minimum_strict_support",
        "measurement_resampling",
        "target_outcomes_used_for_fit_or_tuning",
    ]
    for column in keep:
        if column not in output:
            output[column] = np.nan
    return output[keep]


def _load_surfaces() -> pd.DataFrame:
    tables: list[pd.DataFrame] = []
    for family, report in FAMILY_ENGINE_REPORT.items():
        surface, _ = _canonical_surface(ROOT, family=family, measurement_report=report)
        tables.append(surface)
    for family, relative in FAMILY_CANONICAL_FILES.items():
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        tables.append(_standardize(pd.read_csv(path), family))
    combined = pd.concat(tables, ignore_index=True)
    keys = ["family", "source_context", "target_context", "metric"]
    if len(combined) != EXPECTED_ROWS * len(FAMILY_ENGINE_REPORT | FAMILY_CANONICAL_FILES):
        raise RuntimeError("predictor-family surface has an unexpected row count")
    if combined.duplicated(keys).any():
        raise RuntimeError("four-family surface has duplicate family/transfer/metric keys")
    if (combined["source_context"] == combined["target_context"]).any():
        raise RuntimeError("four-family surface contains self-transfers")
    support = pd.to_numeric(combined["minimum_strict_support"], errors="coerce")
    if support.notna().any() and support.min() < 8:
        raise RuntimeError("four-family surface violates minimum strict support 8")
    return combined.sort_values(keys, kind="stable").reset_index(drop=True)


def _sign(value: float) -> str:
    if not np.isfinite(value):
        return "unavailable"
    return "positive" if value > 0 else "negative" if value < 0 else "zero"


def _ci_sign(lower: float, upper: float) -> str:
    if np.isfinite(lower) and lower > 0:
        return "positive"
    if np.isfinite(upper) and upper < 0:
        return "negative"
    return "crosses_zero"


def _compare(combined: pd.DataFrame) -> pd.DataFrame:
    reference = combined.loc[combined["family"] == "bilinear_ridge"].set_index(
        ["source_context", "target_context", "metric"]
    )
    rows: list[dict[str, Any]] = []
    for family in sorted(combined["family"].unique()):
        table = combined.loc[combined["family"] == family].set_index(
            ["source_context", "target_context", "metric"]
        )
        for metric in METRIC_ORDER:
            ref_metric = reference.loc[
                reference.index.get_level_values("metric") == metric
            ]
            fam_metric = table.loc[
                table.index.get_level_values("metric") == metric
            ]
            keys = ref_metric.index.intersection(fam_metric.index)
            left = ref_metric.loc[keys, "D_adj"].to_numpy(float)
            right = fam_metric.loc[keys, "D_adj"].to_numpy(float)
            rho = 1.0 if family == "bilinear_ridge" else float(
                spearmanr(left, right).statistic
            )
            point_agreement = 1.0 if family == "bilinear_ridge" else float(
                np.mean([_sign(a) == _sign(b) for a, b in zip(left, right)])
            )
            if family == "bilinear_ridge":
                ci_agreement = 1.0
            else:
                ci_agreement = float(
                    np.mean(
                        [
                            _ci_sign(
                                ref_metric.loc[key, "ci_lower_90"],
                                ref_metric.loc[key, "ci_upper_90"],
                            )
                            == _ci_sign(
                                fam_metric.loc[key, "ci_lower_90"],
                                fam_metric.loc[key, "ci_upper_90"],
                            )
                            for key in keys
                        ]
                    )
                )
            rows.append(
                {
                    "family": family,
                    "reference_family": "bilinear_ridge",
                    "metric": metric,
                    "rows_compared": int(len(keys)),
                    "spearman_D_adj_vs_reference": rho,
                    "D_adj_point_sign_agreement": point_agreement,
                    "D_adj_CI_sign_agreement": ci_agreement,
                    "mean_D_adj": float(np.mean(right)),
                    "mean_D_adj_difference_vs_reference": float(np.mean(right - left)),
                }
            )
    return pd.DataFrame(rows)


def _source_fidelity() -> pd.DataFrame:
    path = ROOT / "artifacts/manifests/predictor_family_transport_comparison.csv"
    if not path.is_file():
        return pd.DataFrame(
            columns=["family", "source_oof_fidelity_available", "source_oof_fidelity_mean"]
        )
    table = pd.read_csv(path)
    columns = [column for column in table.columns if column.startswith("source_oof_fidelity_")]
    rows = []
    for family, group in table.groupby("predictor_family", sort=True):
        values = pd.to_numeric(group[columns].stack(), errors="coerce").dropna()
        rows.append(
            {
                "family": family,
                "source_oof_fidelity_available": bool(not values.empty),
                "source_oof_fidelity_mean": float(values.mean()) if not values.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


def run() -> dict[str, Any]:
    combined = _load_surfaces()
    comparison = _compare(combined)
    fidelity = _source_fidelity()
    outdir = ROOT / "artifacts/manifests/predictor_families_v2"
    outdir.mkdir(parents=True, exist_ok=True)
    surface_path = outdir / "canonical_four_family_transport.csv"
    comparison_path = outdir / "canonical_four_family_comparison.csv"
    combined.to_csv(surface_path, index=False)
    comparison.to_csv(comparison_path, index=False)
    report = {
        "schema_version": 2,
        "status": "executed",
        "families": sorted(combined["family"].unique().tolist()),
        "canonical_rows_per_family": EXPECTED_ROWS,
        "directed_transfer_count": 6,
        "metric_count": len(METRIC_ORDER),
        "strict_support_required": 8,
        "surface": surface_path.relative_to(ROOT).as_posix(),
        "comparison": comparison_path.relative_to(ROOT).as_posix(),
        "source_fidelity": fidelity.to_dict(orient="records"),
        "scope_note": (
            "This is a canonical transport comparison. Family superiority is "
            "not asserted; source-fidelity fields are reported only when an "
            "existing artifact provides them."
        ),
    }
    json_path = outdir / "canonical_four_family_transport.json"
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {**report, "json": json_path.relative_to(ROOT).as_posix()}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, ensure_ascii=False))
