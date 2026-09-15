"""Materialize manuscript tables and claim macros from reviewer artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
METRIC_LABELS = {
    "delta_cosine": r"$\Delta$ cosine",
    "systema_centroid_accuracy": "Systema",
    "absolute_effect_rank_agreement": "Abs.-effect rank",
}
TRANSFER_LABELS = {
    "frangieh_melanoma_control->frangieh_melanoma_coculture": r"Ctrl $\to$ Co",
    "frangieh_melanoma_control->frangieh_melanoma_ifng": r"Ctrl $\to$ IFN$\gamma$",
    "frangieh_melanoma_coculture->frangieh_melanoma_control": r"Co $\to$ Ctrl",
    "frangieh_melanoma_coculture->frangieh_melanoma_ifng": r"Co $\to$ IFN$\gamma$",
    "frangieh_melanoma_ifng->frangieh_melanoma_control": r"IFN$\gamma$ $\to$ Ctrl",
    "frangieh_melanoma_ifng->frangieh_melanoma_coculture": r"IFN$\gamma$ $\to$ Co",
}


def _tex(value: Any) -> str:
    return str(value).replace("&", r"\&").replace("%", r"\%")


def _num(value: Any, digits: int = 3) -> str:
    if value is None or not np.isfinite(float(value)):
        return "--"
    return f"{float(value):.{digits}f}"


def _write(path: Path, body: str) -> None:
    path.write_text(body.rstrip() + "\n", encoding="utf-8")


def build(root: Path = ROOT) -> dict[str, Any]:
    root = root.resolve()
    manifests, paper = root / "artifacts/manifests", root / "paper/iclr2027"

    decomp = pd.read_csv(manifests / "reliability_transport_decomposition.csv")
    decomp_cols = ["transfer_id", "metric", "d_tie_cf", "d_direction_cf", "corrected_total_cf", "canonical_d_adj", "cf_minus_canonical", "canonical_reconstruction_error"]
    missing = [column for column in decomp_cols if column not in decomp.columns]
    if missing:
        raise ValueError(f"corrected decomposition is missing columns: {missing}")
    decomp = decomp[decomp_cols].copy()
    decomp["transfer_label"] = decomp["transfer_id"].map(TRANSFER_LABELS).fillna(decomp["transfer_id"])
    decomp["metric_label"] = decomp["metric"].map(METRIC_LABELS).fillna(decomp["metric"])
    lines = [r"\begin{table}[H]", r"\centering", r"\scriptsize", r"\caption{Finite-measurement cross-product decomposition on the matched-fixed full-depth surface. The empirical report contains the tie and directional-bias components; the population strength/flip identity is not treated as an unbiased finite-sample split.}", r"\label{tab:transport-decomposition}", r"\begin{tabular}{llrrrrr}", r"\toprule", r"Transfer & Metric & $D_{\rm tie}^{\rm CF}$ & $D_{\rm dir}^{\rm CF}$ & Sum & $D_{\rm adj}$ & Difference \\", r"\midrule"]
    for _, row in decomp.sort_values(["transfer_id", "metric"]).iterrows():
        lines.append(f"{row.transfer_label} & {row.metric_label} & {_num(row.d_tie_cf)} & {_num(row.d_direction_cf)} & {_num(row.corrected_total_cf)} & {_num(row.canonical_d_adj)} & {_num(row.cf_minus_canonical)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _write(paper / "supplement_transport_decomposition_table.tex", "\n".join(lines))

    comparator = json.loads((manifests / "rank_comparator_benchmark.json").read_text(encoding="utf-8"))
    names = {"kendall_distance": r"Kendall $(1-\tau_b)/2$", "spearman_distance": r"Spearman $(1-\rho)/2$", "top10_jaccard_distance": r"Top-10\% Jaccard distance", "stable_pair_inversion_fraction": "Stable-pair inversion", "d_adj": r"$D_{\rm adj}$"}
    lines = [r"\begin{table}[H]", r"\centering", r"\scriptsize", r"\caption{Rank-comparator benchmark on the same 18 directed transfer--metric rows. The final column is the descriptive Spearman association with normalized regret.}", r"\label{tab:rank-comparator}", r"\begin{tabular}{lrl}", r"\toprule", r"Comparator & $n$ & Spearman with regret \\", r"\midrule"]
    for name in names:
        item = comparator["correlation_with_normalized_regret"][name]
        lines.append(f"{names[name]} & {item['n']} & {_num(item['spearman'], 2)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _write(paper / "supplement_rank_comparator_table.tex", "\n".join(lines))

    incremental = pd.read_csv(manifests / "reviewer_decision_incremental_value.csv")
    lines = [r"\begin{table}[H]", r"\centering", r"\tiny", r"\caption{Prespecified model comparison under leave-one-unordered-context-pair-out validation. $\Delta$MAE is positive when a model improves on the stated reference; intervals are 2,000 unordered-pair block-bootstrap diagnostics. M4 is exploratory because its decomposition features are not a superset of M3.}", r"\label{tab:incremental-decision-value}", r"\resizebox{\linewidth}{!}{%", r"\begin{tabular}{lrrrrrrr}", r"\toprule", r"Model & MAE & RMSE & OOF $\rho$ & $\Delta\mathrm{MAE}_{M0}$ & $\Delta\mathrm{MAE}_{M1}$ & 90\% interval & Validation \\", r"\midrule"]
    for _, row in incremental.iterrows():
        model_label = row["model"].replace("_", " ")
        if row["model"] == "M4_metric_plus_mean_plus_decomposition":
            model_label += " (exploratory)"
        interval = "--" if row["model"] == "M0_metric" else f"[{_num(row.bootstrap_delta_mae_q05)}, {_num(row.bootstrap_delta_mae_q95)}]"
        validation = "ref." if row["model"] == "M0_metric" else "LOPO (3)"
        pvalue = validation
        row["model"] = model_label
        lines.append(f"{_tex(row['model'].replace('_', ' '))} & {_num(row.mae)} & {_num(row.rmse)} & {_num(row.spearman_oof, 2)} & {_num(row.delta_mae_vs_M0)} & {_num(row.delta_mae_vs_M1)} & {interval} & {pvalue} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}"]
    _write(paper / "supplement_incremental_decision_value_table.tex", "\n".join(lines))

    audit = pd.read_csv(manifests / "target_audit_simulation.csv")
    aggregate = audit.groupby("audit_method", as_index=False).agg(mean_upgraded=("target_labels_upgraded", "mean"), false_reuse=("false_reuse", "mean"), correct_reuse=("correct_reuse", "mean"), correct_revise=("correct_revise", "mean"))
    lines = [r"\begin{table}[H]", r"\centering", r"\scriptsize", r"\caption{Empirical target-audit replay. Every candidate receives a 10-cell pilot; the policy allocates additional target cells using only the audit pool. Values average the four decision budgets.}", r"\label{tab:target-audit}", r"\begin{tabular}{lrrrr}", r"\toprule", r"Policy & Mean upgraded & False REUSE & Correct REUSE & Correct REVISE \\", r"\midrule"]
    for _, row in aggregate.sort_values("audit_method").iterrows():
        lines.append(f"{_tex(row.audit_method.replace('_', ' '))} & {_num(row.mean_upgraded, 1)} & {_num(row.false_reuse, 3)} & {_num(row.correct_reuse, 3)} & {_num(row.correct_revise, 3)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _write(paper / "supplement_target_audit_table.tex", "\n".join(lines))

    reconstruction = float(decomp["canonical_reconstruction_error"].abs().max())
    cf_vs_canonical = float(decomp["cf_minus_canonical"].abs().max())
    contrasts = pd.read_csv(manifests / "reviewer_decision_incremental_value_contrasts.csv")
    contrast = {str(row.contrast): row for row in contrasts.itertuples(index=False)}
    c_m1 = contrast["M1-M0"]
    c_m2 = contrast["M2-M0"]
    c_m3 = contrast["M3-M1"]
    m1 = incremental.loc[incremental["model"].eq("M1_metric_plus_mean")].iloc[0]
    m3 = incremental.loc[incremental["model"].eq("M3_metric_plus_mean_plus_d_adj")].iloc[0]
    macros = [
        "% Generated by scripts/build_scientific_upgrade_tables.py; do not edit by hand.",
        f"\\newcommand{{\\PTLDecompositionIdentityError}}{{{reconstruction:.2e}}}",
        f"\\newcommand{{\\PTLDecompositionVsCanonicalMaxDiff}}{{{cf_vs_canonical:.3f}}}",
        f"\\newcommand{{\\PTLComparatorKendallRho}}{{{comparator['correlation_with_normalized_regret']['kendall_distance']['spearman']:.2f}}}",
        f"\\newcommand{{\\PTLComparatorJaccardRho}}{{{comparator['correlation_with_normalized_regret']['top10_jaccard_distance']['spearman']:.2f}}}",
        f"\\newcommand{{\\PTLIncrementalMeanDeltaMAE}}{{{float(c_m1.delta_mae):.3f}}}",
        f"\\newcommand{{\\PTLIncrementalMeanDeltaMAECI}}{{[{float(c_m1.bootstrap_delta_mae_q05):.3f}, {float(c_m1.bootstrap_delta_mae_q95):.3f}]}}",
        f"\\newcommand{{\\PTLIncrementalDAdjMetricOnlyDeltaMAE}}{{{float(c_m2.delta_mae):.3f}}}",
        f"\\newcommand{{\\PTLIncrementalDAdjMetricOnlyDeltaMAECI}}{{[{float(c_m2.bootstrap_delta_mae_q05):.3f}, {float(c_m2.bootstrap_delta_mae_q95):.3f}]}}",
        f"\\newcommand{{\\PTLIncrementalDAdjBeyondMeanDeltaMAE}}{{{float(c_m3.delta_mae):.3f}}}",
        f"\\newcommand{{\\PTLIncrementalDAdjBeyondMeanDeltaMAECI}}{{[{float(c_m3.bootstrap_delta_mae_q05):.3f}, {float(c_m3.bootstrap_delta_mae_q95):.3f}]}}",
        f"\\newcommand{{\\PTLTargetAuditRows}}{{{len(audit)}}}",
        f"\\newcommand{{\\PTLTargetAuditFalseReuse}}{{{audit['false_reuse'].mean():.3f}}}",
    ]
    _write(paper / "generated_scientific_upgrade.tex", "\n".join(macros))
    report = {"status": "executed", "outputs": ["paper/iclr2027/supplement_transport_decomposition_table.tex", "paper/iclr2027/supplement_rank_comparator_table.tex", "paper/iclr2027/supplement_incremental_decision_value_table.tex", "paper/iclr2027/supplement_target_audit_table.tex", "paper/iclr2027/generated_scientific_upgrade.tex"]}
    (manifests / "scientific_upgrade_tables.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(build(args.root), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
