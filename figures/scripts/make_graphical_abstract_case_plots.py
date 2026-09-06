"""Build real case scatter panels for the CBAC graphical abstract.

The source PPTX contains hand-drawn placeholder scatter plots. This script
replaces those placeholders with compact data-driven panels generated from the
baseline output contract for the three CBAC case examples.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
CASE_TABLE = ROOT / "results" / "tables" / "cbac_case_examples.csv"
OUT_DIR = ROOT / "results" / "figures_cbac" / "graphical_abstract_cases"


CASE_COLORS = {
    "retained_high_transportability": "#1f6aa5",
    "confidence_failure_rejected": "#d46a1f",
    "transfer_boundary_failure": "#b21f24",
}


def _empirical_effect_score(delta: np.ndarray) -> np.ndarray:
    """Return -log10 empirical tail probability by absolute target effect."""
    abs_delta = np.abs(np.asarray(delta, dtype=float))
    order = np.argsort(-abs_delta, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(abs_delta) + 1, dtype=float)
    p = ranks / (len(abs_delta) + 1.0)
    return -np.log10(np.clip(p, 1e-12, 1.0))


def _read_case_arrays(row: pd.Series) -> tuple[np.ndarray, np.ndarray, list[str]]:
    base = ROOT / str(row["output_dir"])
    arrays = np.load(base / "test_predictions.npz")
    metadata = pd.read_parquet(base / "test_metadata.parquet")
    genes = (base / "genes.txt").read_text(encoding="utf-8").splitlines()
    matches = metadata.index[metadata["signature_id"].astype(str).eq(str(row["signature_id"]))].tolist()
    if not matches:
        raise ValueError(f"signature not found in metadata: {row['signature_id']}")
    idx = int(matches[0])
    return arrays["y_true"][idx], arrays["y_pred"][idx], genes


def _plot_case(row: pd.Series, out_path: Path) -> None:
    true_delta, pred_delta, genes = _read_case_arrays(row)
    genes_arr = np.asarray(genes, dtype=str)
    y_score = _empirical_effect_score(true_delta)

    color = CASE_COLORS.get(str(row["case_type"]), "#4c6f91")
    shared = {g for g in str(row["shared_top_genes"]).split(";") if g}
    top_true = set(genes_arr[np.argsort(-np.abs(true_delta), kind="mergesort")[:140]].tolist())
    top_pred = set(genes_arr[np.argsort(-np.abs(pred_delta), kind="mergesort")[:140]].tolist())
    highlighted = np.asarray([g in (top_true | top_pred) for g in genes_arr])
    shared_mask = np.asarray([g in shared for g in genes_arr])

    x_limit = float(np.nanquantile(np.abs(true_delta), 0.995))
    x_limit = max(0.75, min(3.0, x_limit * 1.15))

    fig, ax = plt.subplots(figsize=(1.75, 1.22), dpi=600)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.scatter(true_delta, y_score, s=1.8, c="#9aa0a6", alpha=0.38, linewidths=0, rasterized=True)
    ax.scatter(
        true_delta[highlighted],
        y_score[highlighted],
        s=3.0,
        c=color,
        alpha=0.72,
        linewidths=0,
        rasterized=True,
    )
    if shared_mask.any():
        ax.scatter(
            true_delta[shared_mask],
            y_score[shared_mask],
            s=6.5,
            c=color,
            edgecolors="white",
            linewidths=0.35,
            alpha=0.98,
            rasterized=True,
        )

    ax.axvline(0.0, color="#7d7d7d", linestyle=(0, (2, 2)), linewidth=0.55, alpha=0.85)
    ax.set_xlim(-x_limit, x_limit)
    ax.set_ylim(0, min(4.4, max(2.6, float(np.nanquantile(y_score, 0.995)) * 1.05)))
    ax.set_xlabel("target delta", fontsize=4.4, labelpad=0.6)
    ax.set_ylabel("-log10(rank p)", fontsize=4.4, labelpad=0.6)
    ax.tick_params(axis="both", labelsize=4.0, width=0.42, length=1.8, pad=0.8)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#8f8f8f")
        ax.spines[spine].set_linewidth(0.5)
    ax.grid(False)
    fig.subplots_adjust(left=0.21, right=0.985, top=0.97, bottom=0.24)
    fig.savefig(out_path, transparent=False)
    fig.savefig(out_path.with_suffix(".pdf"), transparent=False)
    fig.savefig(out_path.with_suffix(".svg"), transparent=False)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cases = pd.read_csv(CASE_TABLE)
    for row in cases.itertuples(index=False):
        row_s = pd.Series(row._asdict())
        slug = str(row_s["case_type"]).replace(" ", "_")
        out_path = OUT_DIR / f"{slug}_scatter.png"
        _plot_case(row_s, out_path)
        print(out_path)


if __name__ == "__main__":
    main()
