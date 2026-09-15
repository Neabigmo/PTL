"""Render a compact Supplement table from the corrected v2.1 simulation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "artifacts/manifests/simulation_v21/finite_measurement_primary_v21.csv"
OUTPUT = ROOT / "paper/iclr2027/supplement_finite_measurement_v21_table.tex"

LABELS = {
    "d_adj": r"$D_{\mathrm{adj}}$",
    "kendall": "Kendall distance",
    "spearman": "Spearman distance",
    "top10_jaccard": "Top-10\\% Jaccard distance",
    "stable_inversion": "Stable-pair inversion",
}


def _fmt(value: float) -> str:
    return "--" if not np.isfinite(value) else f"{value:.3f}"


def main() -> int:
    table = pd.read_csv(INPUT)
    rows = []
    for metric, label in LABELS.items():
        mae = pd.to_numeric(table[f"{metric}_mae_to_truth"], errors="coerce")
        bias = pd.to_numeric(table[f"{metric}_bias_to_truth"], errors="coerce")
        rmse = pd.to_numeric(table[f"{metric}_rmse_to_truth"], errors="coerce")
        rows.append(
            " & ".join(
                [
                    label,
                    _fmt(float(mae.mean())),
                    _fmt(float(bias.mean())),
                    _fmt(float(rmse.mean())),
                    _fmt(float(mae.median())),
                ]
            )
            + r" \\"
        )
    resolution = float(table["realized_pair_tie_rate"].mean())
    text = r"""\begin{table}[t]
\centering
\caption{\textbf{Corrected finite-measurement calibration.} Each primary
condition uses a finite-depth pair-state population as the truth for its
matching estimator. Values summarize the condition-level absolute error,
signed bias, RMSE, and median absolute error over the registered primary grid;
the grid contains 960 conditions and 500 trials per condition. The realized
pair-tie rate is reported rather than equated with the resolution setting.}
\label{tab:finite-measurement-v21}
\small
\begin{tabular}{lrrrr}
\toprule
Estimator & Mean MAE & Mean bias & Mean RMSE & Median MAE \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\par\smallskip
\raggedright\footnotesize Mean realized pair-tie rate across conditions:
""" + _fmt(resolution) + r""". The corrected $D_{\mathrm{adj}}$ truth is
not a latent Kendall distance; the previous v2 artifact is retained only for
provenance and is not pooled here.
\end{table}
"""
    OUTPUT.write_text(text, encoding="utf-8")
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
