"""Render the canonical predictor-family transport summary for the Supplement."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "artifacts/manifests/predictor_families_v2/canonical_four_family_comparison.csv"
OUTPUT = ROOT / "paper/iclr2027/supplement_predictor_family_canonical_v2_table.tex"
NL = chr(10)
BS = chr(92)


def _fmt(value: float) -> str:
    return "--" if not np.isfinite(value) else f"{value:.3f}"


def main() -> int:
    table = pd.read_csv(INPUT)
    summary = (
        table.groupby("family", as_index=False)
        .agg(
            mean_rho=("spearman_D_adj_vs_reference", "mean"),
            min_point_agreement=("D_adj_point_sign_agreement", "min"),
            min_ci_agreement=("D_adj_CI_sign_agreement", "min"),
            mean_dadj=("mean_D_adj", "mean"),
            mean_delta=("mean_D_adj_difference_vs_reference", "mean"),
            rows=("rows_compared", "min"),
        )
        .sort_values("family")
    )
    family_labels = {
        "bilinear_ridge": "Bilinear ridge",
        "source_only_latent_mlp": "Latent-response MLP",
        "rbf_krr": "RBF kernel ridge",
        "source_only_mlp": "Direct MLP",
        "official_gears": "GEARS (strict source-frozen)",
    }
    lines = []
    for row in summary.to_dict(orient="records"):
        lines.append(
            " & ".join(
                [
                    family_labels.get(row["family"], str(row["family"])),
                    _fmt(float(row["mean_rho"])),
                    _fmt(float(row["min_point_agreement"])),
                    _fmt(float(row["min_ci_agreement"])),
                    _fmt(float(row["mean_dadj"])),
                    _fmt(float(row["mean_delta"])),
                    str(int(row["rows"])),
                ]
            )
            + " "
            + BS * 2
        )
    text = NL.join(
        [
            BS + "begin{table}[t]",
            BS + "centering",
            BS + "caption{" + BS + "textbf{Canonical source-only predictor-family transport.} Each",
            "family is evaluated on the same six directed transfers, three metrics and",
            "full-size raw-cell measurement surface. The correlation and sign columns are",
            "relative to the bilinear-ridge reference and are descriptive; they do not",
            "establish family equivalence or superiority.}",
            BS + "label{tab:predictor-family-canonical-v2}",
            BS + "small",
            BS + "resizebox{" + BS + "linewidth}{!}{%",
            BS + "begin{tabular}{lrrrrrr}",
            BS + "toprule",
            "Family & Mean $" + BS + "rho$ & Min point sign & Min CI sign & Mean $D_{" + BS + "rm adj}$ & $" + BS + "Delta$ mean & Rows " + BS * 2,
            BS + "midrule",
            *lines,
            BS + "bottomrule",
            BS + "end{tabular}",
            "}",
            BS + "end{table}",
            "",
        ]
    )
    OUTPUT.write_text(text, encoding="utf-8")
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
