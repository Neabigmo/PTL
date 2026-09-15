"""Materialize the small set of manuscript claims from canonical artifacts.

This file is intentionally a deterministic summary layer: prose and figure
builders can consume one generated table instead of repeating hand-entered
numbers.  It does not create new estimates or alter any estimator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ptl_figure_style import CONTEXT_ORDER, METRIC_COLORS, METRIC_LABELS  # noqa: E402


def run(root: Path = ROOT) -> dict:
    root = root.resolve()
    manifests = root / "artifacts/manifests"
    atlas = pd.read_csv(manifests / "reliability_transport_atlas.csv")
    atlas = atlas.loc[atlas["dataset"].isin(["FrangiehIzar2021_RNA", "NadigOConner2024"])].copy()
    atlas["claim_family"] = atlas["dataset"].map({"FrangiehIzar2021_RNA": "frangieh_transport", "NadigOConner2024": "nadig_replication"})
    atlas["metric_label"] = atlas["metric"].map(METRIC_LABELS)
    keep = [
        "claim_family", "dataset", "source_environment_id", "left_target_environment_id",
        "right_target_environment_id", "metric", "metric_label", "evidence_tier",
        "observed_D", "observed_D_ci_low", "observed_D_ci_high",
        "measurement_identifiable", "measurement_identifiable_ci_low",
        "measurement_identifiable_ci_high", "joint_identifiable",
        "stable_fraction_both", "status", "availability", "provenance",
    ]
    claims = atlas[keep].sort_values(["claim_family", "source_environment_id", "metric"], kind="stable").reset_index(drop=True)
    claims.insert(0, "claim_id", [f"{family}_{i:02d}" for i, family in enumerate(claims["claim_family"], 1)])
    out_csv = manifests / "paper_claim_summary.csv"
    out_json = manifests / "paper_claim_summary.json"
    claims.to_csv(out_csv, index=False)

    primary = claims.loc[
        (claims["dataset"] == "FrangiehIzar2021_RNA")
        & (claims["source_environment_id"] == "frangieh_melanoma_control")
        & (claims["left_target_environment_id"] == "frangieh_melanoma_control")
        & (claims["right_target_environment_id"] == "frangieh_melanoma_ifng")
    ].set_index("metric")
    decision = pd.read_csv(manifests / "reliability_transport_decision_link.csv")
    decision_report = json.loads((manifests / "reliability_transport_decision_link.json").read_text(encoding="utf-8"))
    decision_overall = decision_report["correlations"]["overall"]
    rho = float(pd.to_numeric(decision["normalized_regret"], errors="coerce").corr(pd.to_numeric(decision["d_meas_id"], errors="coerce"), method="spearman"))
    control = pd.read_csv(manifests / "reviewer_pseudocontext_negative_control.csv")
    control_values = pd.to_numeric(control["measurement_identifiable"], errors="coerce").dropna()
    nadig_means = claims.loc[claims["claim_family"].eq("nadig_replication")].groupby("metric")["measurement_identifiable"].mean()
    stratified_path = manifests / "reviewer_decision_metric_stratified.json"
    stratified = json.loads(stratified_path.read_text(encoding="utf-8")) if stratified_path.is_file() else {}
    stratified_point = float(stratified.get("point_spearman", float("nan")))
    stratified_bootstrap_ci = stratified.get("within_metric_unordered_pair_bootstrap_ci", [float("nan"), float("nan")])
    mean_fidelity = pd.read_csv(manifests / "metric_matched_mean_fidelity_control.csv")
    mean_shift_rho_d = float(mean_fidelity["absolute_mean_risk_shift"].corr(mean_fidelity["d_meas_id"], method="spearman"))
    mean_shift_rho_regret = float(mean_fidelity["absolute_mean_risk_shift"].corr(mean_fidelity["normalized_regret"], method="spearman"))
    payload = {
        "status": "executed",
        "rows": int(len(claims)),
        "frangieh_rows": int((claims["claim_family"] == "frangieh_transport").sum()),
        "nadig_rows": int((claims["claim_family"] == "nadig_replication").sum()),
        "frangieh_shared_labels": 243,
        "decision_link_rows": int(len(decision)),
        "decision_spearman": rho,
        "decision_spearman_ci": [float(decision_overall["unordered_pair_bootstrap_q05"]), float(decision_overall["unordered_pair_bootstrap_q95"])],
        "within_metric_spearman": stratified_point,
        "within_metric_spearman_ci": stratified_bootstrap_ci,
        "mean_fidelity_shift_spearman_d_adj": mean_shift_rho_d,
        "mean_fidelity_shift_spearman_regret": mean_shift_rho_regret,
        "primary_ctrl_ifng_adjusted": {
            metric: float(primary.loc[metric, "measurement_identifiable"])
            for metric in METRIC_COLORS
        },
        "source": "artifacts/manifests/reliability_transport_atlas.csv + reliability_transport_decision_link.csv",
        "outputs": [out_csv.relative_to(root).as_posix(), out_json.relative_to(root).as_posix(), "paper/iclr2027/generated_claims.tex"],
    }
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    tex = root / "paper/iclr2027/generated_claims.tex"
    def val(metric: str, field: str) -> float:
        return float(primary.loc[metric, field])
    lines = [
        "% Generated by scripts/build_paper_claim_summary.py; do not edit by hand.",
        f"\\newcommand{{\\PTLPrimaryDeltaCross}}{{{val('delta_cosine', 'observed_D'):.6f}}}",
        f"\\newcommand{{\\PTLPrimaryDeltaAdj}}{{{val('delta_cosine', 'measurement_identifiable'):.6f}}}",
        f"\\newcommand{{\\PTLPrimaryDeltaAdjCI}}{{[{val('delta_cosine', 'measurement_identifiable_ci_low'):.6f}, {val('delta_cosine', 'measurement_identifiable_ci_high'):.6f}]}}",
        f"\\newcommand{{\\PTLPrimarySystemaCross}}{{{val('systema_centroid_accuracy', 'observed_D'):.6f}}}",
        f"\\newcommand{{\\PTLPrimarySystemaAdj}}{{{val('systema_centroid_accuracy', 'measurement_identifiable'):.6f}}}",
        f"\\newcommand{{\\PTLPrimarySystemaAdjCI}}{{[{val('systema_centroid_accuracy', 'measurement_identifiable_ci_low'):.6f}, {val('systema_centroid_accuracy', 'measurement_identifiable_ci_high'):.6f}]}}",
        f"\\newcommand{{\\PTLPrimaryRankCross}}{{{val('absolute_effect_rank_agreement', 'observed_D'):.6f}}}",
        f"\\newcommand{{\\PTLPrimaryRankAdj}}{{{val('absolute_effect_rank_agreement', 'measurement_identifiable'):.6f}}}",
        f"\\newcommand{{\\PTLPrimaryRankAdjCI}}{{[{val('absolute_effect_rank_agreement', 'measurement_identifiable_ci_low'):.6f}, {val('absolute_effect_rank_agreement', 'measurement_identifiable_ci_high'):.6f}]}}",
        f"\\newcommand{{\\PTLDecisionRho}}{{{rho:.6f}}}",
        f"\\newcommand{{\\PTLDecisionRhoDisplay}}{{{rho:.2f}}}",
        f"\\newcommand{{\\PTLDecisionRhoCI}}{{[{float(decision_overall['unordered_pair_bootstrap_q05']):.2f}, {float(decision_overall['unordered_pair_bootstrap_q95']):.2f}]}}",
        f"\\newcommand{{\\PTLWithinMetricRho}}{{{stratified_point:.2f}}}",
        f"\\newcommand{{\\PTLWithinMetricRhoCI}}{{[{float(stratified_bootstrap_ci[0]):.2f}, {float(stratified_bootstrap_ci[1]):.2f}]}}",
        f"\\newcommand{{\\PTLMeanFidelityShiftRhoD}}{{{mean_shift_rho_d:.2f}}}",
        f"\\newcommand{{\\PTLMeanFidelityShiftRhoRegret}}{{{mean_shift_rho_regret:.2f}}}",
        f"\\newcommand{{\\PTLSameContextRange}}{{{float(control_values.min()):.4f}--{float(control_values.max()):.4f}}}",
        f"\\newcommand{{\\PTLNadigDelta}}{{{float(nadig_means['delta_cosine']):.3f}}}",
        f"\\newcommand{{\\PTLNadigSystema}}{{{float(nadig_means['systema_centroid_accuracy']):.3f}}}",
        f"\\newcommand{{\\PTLNadigRank}}{{{float(nadig_means['absolute_effect_rank_agreement']):.3f}}}",
        "\\newcommand{\\PTLFrangiehLabels}{243}",
    ]
    tex.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    run(parser.parse_args().root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
