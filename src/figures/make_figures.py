from __future__ import annotations

import argparse
import math
import sys
import textwrap
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import auc, precision_recall_curve, roc_curve


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
DOCS_DIR = ROOT / "docs"
FIGURES_DIR = RESULTS_DIR / "figures"
NOTES_DIR = DOCS_DIR / "figure_notes"
LOG_FILE = RESULTS_DIR / "logs" / "figures" / "phase09_figures.log"

WHITE = "#FFFFFF"
TEXT = "#2F2F2F"
MUTED = "#6D6D6D"
RULE = "#B7B7B7"
LIGHT_RULE = "#E6E6E6"
HEAT_LOW = "#F7F7F7"
HEAT_HIGH = "#4E79A7"
PRIMARY_METRIC = "mean_cosine_non_control"
CONTROL_MODEL = "control_mean_baseline"

SEMANTIC_PALETTE = {
    "random_split": "#4E79A7",
    "unseen_perturbation_split": "#F28E2B",
    "unseen_combination_split": "#59A14F",
    "low_support_split": "#EDC948",
    "dataset_heldout_split": "#B07AA1",
    "external_holdout": "#E15759",
    "PTL": "#76B7B2",
    "naive_confidence": "#9D9D9D",
}

MODEL_PALETTE = {
    "ridge_regression_baseline": "#4E79A7",
    "global_delta_baseline": "#59A14F",
    "perturbation_mean_delta_baseline": "#F28E2B",
    "cell_context_knn_delta_baseline": "#B07AA1",
    "control_mean_baseline": "#9D9D9D",
}

SPLIT_ORDER = [
    "random_split",
    "unseen_perturbation_split",
    "unseen_combination_split",
    "low_support_split",
    "dataset_heldout_split",
    "external_holdout",
]

REQUIRED_FIGURES = {
    "fig1_png": "fig1_study_design.png",
    "fig1_pdf": "fig1_study_design.pdf",
    "fig1_svg": "fig1_study_design.svg",
    "fig2_png": "fig2_benchmark_composition_audit.png",
    "fig2_pdf": "fig2_benchmark_composition_audit.pdf",
    "fig2_svg": "fig2_benchmark_composition_audit.svg",
    "fig3_png": "fig3_ranking_instability_transfer_decay.png",
    "fig3_pdf": "fig3_ranking_instability_transfer_decay.pdf",
    "fig3_svg": "fig3_ranking_instability_transfer_decay.svg",
    "fig4_png": "fig4_ptl_selective_filtering.png",
    "fig4_pdf": "fig4_ptl_selective_filtering.pdf",
    "fig4_svg": "fig4_ptl_selective_filtering.svg",
    "fig5_png": "fig5_failure_mode_atlas.png",
    "fig5_pdf": "fig5_failure_mode_atlas.pdf",
    "fig5_svg": "fig5_failure_mode_atlas.svg",
    "supp_fig1_png": "supp_fig1_robustness_audit.png",
    "supp_fig1_pdf": "supp_fig1_robustness_audit.pdf",
    "supp_fig1_svg": "supp_fig1_robustness_audit.svg",
}


@dataclass(frozen=True)
class Phase09Inputs:
    main_findings: Path
    transfer_decay: Path
    selective_prediction: Path
    failure_atlas: Path
    narrative: Path
    ptl_predictions: Path | None = None
    ptl_examples: Path | None = None
    all_metrics: Path | None = None
    split_audit: Path | None = None
    data_inventory: Path | None = None
    preprocessing_summary: Path | None = None
    baseline_manifest: Path | None = None


class PhaseLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {message}"
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        print(line, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate publication-style Phase 09 figures.")
    parser.add_argument("--main-findings", default=str(TABLES_DIR / "main_findings.csv"))
    parser.add_argument("--transfer-decay", default=str(TABLES_DIR / "transfer_decay_summary.csv"))
    parser.add_argument("--selective-prediction", default=str(TABLES_DIR / "selective_prediction_summary.csv"))
    parser.add_argument("--failure-atlas", default=str(TABLES_DIR / "failure_mode_atlas.csv"))
    parser.add_argument("--narrative", default=str(DOCS_DIR / "results_narrative.md"))
    parser.add_argument("--ptl-predictions", default=str(TABLES_DIR / "ptl_oof_predictions.csv"))
    parser.add_argument("--ptl-examples", default=str(TABLES_DIR / "ptl_examples.parquet"))
    parser.add_argument("--all-metrics", default=str(TABLES_DIR / "all_metrics.csv"))
    parser.add_argument("--split-audit", default=str(TABLES_DIR / "split_audit.csv"))
    parser.add_argument("--data-inventory", default=str(TABLES_DIR / "data_inventory.csv"))
    parser.add_argument("--preprocessing-summary", default=str(TABLES_DIR / "preprocessing_summary.csv"))
    parser.add_argument("--baseline-manifest", default=str(TABLES_DIR / "baseline_run_matrix.csv"))
    parser.add_argument("--output-dir", default=str(FIGURES_DIR))
    parser.add_argument("--notes-dir", default=str(NOTES_DIR))
    parser.add_argument("--log-file", default=str(LOG_FILE))
    return parser.parse_args()


def figure_paths(output_dir: Path | str) -> dict[str, Path]:
    base = Path(output_dir)
    return {key: base / filename for key, filename in REQUIRED_FIGURES.items()}


def note_paths(notes_dir: Path | str) -> dict[str, Path]:
    base = Path(notes_dir)
    return {
        "fig1": base / "fig1_study_design.md",
        "fig2": base / "fig2_benchmark_composition_audit.md",
        "fig3": base / "fig3_ranking_instability_transfer_decay.md",
        "fig4": base / "fig4_ptl_selective_filtering.md",
        "fig5": base / "fig5_failure_mode_atlas.md",
        "supp_fig1": base / "supp_fig1_robustness_audit.md",
    }


def wrap_label(text: Any, width: int = 20) -> str:
    value = str(text).replace("_", " ").strip()
    if not value:
        return ""
    wrapped = textwrap.wrap(value, width=width, break_long_words=False, break_on_hyphens=False)
    if len(wrapped) == 1 and len(wrapped[0]) > width:
        wrapped = textwrap.wrap(value, width=width, break_long_words=True, break_on_hyphens=True)
    return "\n".join(wrapped)


def pretty_model(model: Any) -> str:
    mapping = {
        "ridge_regression_baseline": "Ridge",
        "global_delta_baseline": "Global delta",
        "perturbation_mean_delta_baseline": "Perturbation mean",
        "cell_context_knn_delta_baseline": "Context kNN",
        "control_mean_baseline": "Control mean",
    }
    value = str(model)
    return mapping.get(value, value.replace("_baseline", "").replace("_", " ").title())


def pretty_split(split: Any) -> str:
    mapping = {
        "random_split": "Random anchor",
        "unseen_perturbation_split": "Held-out perturbation",
        "unseen_combination_split": "Held-out combination",
        "low_support_split": "Low support",
        "dataset_heldout_split": "Dataset heldout",
        "external_holdout": "External holdout",
    }
    value = str(split)
    return mapping.get(value, value.replace("_", " ").title())


def _read_csv(path: Path | None, required: bool = True) -> pd.DataFrame:
    if path is None or not Path(path).exists():
        if required:
            raise FileNotFoundError(path)
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if required and frame.empty:
        raise ValueError(f"Input table is empty: {path}")
    return frame


def _read_parquet(path: Path | None, required: bool = False) -> pd.DataFrame:
    if path is None or not Path(path).exists():
        if required:
            raise FileNotFoundError(path)
        return pd.DataFrame()
    frame = pd.read_parquet(path)
    if required and frame.empty:
        raise ValueError(f"Input table is empty: {path}")
    return frame


def _require_columns(frame: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")


def validate_inputs(inputs: Phase09Inputs) -> dict[str, pd.DataFrame | str]:
    main = _read_csv(inputs.main_findings)
    transfer = _read_csv(inputs.transfer_decay)
    selective = _read_csv(inputs.selective_prediction)
    failure = _read_csv(inputs.failure_atlas)
    predictions = _read_csv(inputs.ptl_predictions, required=False)
    examples = _read_parquet(inputs.ptl_examples, required=False)
    all_metrics = _read_csv(inputs.all_metrics, required=False)
    split_audit = _read_csv(inputs.split_audit, required=False)
    data_inventory = _read_csv(inputs.data_inventory, required=False)
    preprocessing = _read_csv(inputs.preprocessing_summary, required=False)
    baseline_manifest = _read_csv(inputs.baseline_manifest, required=False)

    if not inputs.narrative.exists():
        raise FileNotFoundError(inputs.narrative)
    narrative = inputs.narrative.read_text(encoding="utf-8")
    if not narrative.strip():
        raise ValueError(f"Narrative is empty: {inputs.narrative}")

    _require_columns(main, {"finding_id", "claim", "evidence_table"}, "main findings")
    _require_columns(transfer, {"model", "split_family", PRIMARY_METRIC, "family_rank", "family_winner"}, "transfer decay")
    _require_columns(selective, {"ablation", "estimator", "mean_false_transportability_rate", "mean_selective_risk"}, "selective prediction")
    _require_columns(failure, {"model", "split_family", "failure_mode", "n_signatures", "failure_mode_share", "failure_severity"}, "failure atlas")
    if not predictions.empty:
        _require_columns(predictions, {"ablation", "estimator", "y_true", "score", "target_risk", "split_family", "model"}, "PTL predictions")
    return {
        "main": main,
        "transfer": transfer,
        "selective": selective,
        "failure": failure,
        "predictions": predictions,
        "examples": examples,
        "all_metrics": all_metrics,
        "split_audit": split_audit,
        "data_inventory": data_inventory,
        "preprocessing": preprocessing,
        "baseline_manifest": baseline_manifest,
        "narrative": narrative,
    }


def apply_publication_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": WHITE,
            "axes.facecolor": WHITE,
            "savefig.facecolor": WHITE,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.4,
            "axes.labelsize": 8.8,
            "axes.titlesize": 9.6,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "legend.fontsize": 7.7,
            "axes.edgecolor": RULE,
            "axes.labelcolor": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "text.color": TEXT,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "lines.linewidth": 0.9,
            "patch.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "figure.dpi": 180,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _save_figure(fig: plt.Figure, paths: dict[str, Path], prefix: str) -> list[Path]:
    outputs: list[Path] = []
    for suffix in ("png", "pdf", "svg"):
        path = paths[f"{prefix}_{suffix}"]
        path.parent.mkdir(parents=True, exist_ok=True)
        kwargs: dict[str, Any] = {"bbox_inches": "tight", "facecolor": WHITE}
        if suffix == "png":
            kwargs["dpi"] = 600
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="constrained_layout not applied.*")
            fig.savefig(path, **kwargs)
        outputs.append(path)
    plt.close(fig)
    return outputs


def _panel_label(ax: plt.Axes, label: str, x: float = -0.08, y: float = 1.05) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=10, fontweight="bold", va="top", ha="left")


def _clean_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=2.5, color=RULE)


def _draw_box(ax: plt.Axes, xy: tuple[float, float], wh: tuple[float, float], label: str, color: str, sub: str = "") -> None:
    rect = patches.FancyBboxPatch(
        xy,
        wh[0],
        wh[1],
        boxstyle="round,pad=0.012,rounding_size=0.012",
        facecolor=WHITE,
        edgecolor=color,
        linewidth=0.9,
    )
    ax.add_patch(rect)
    ax.text(xy[0] + wh[0] / 2, xy[1] + wh[1] * 0.60, wrap_label(label, 16), ha="center", va="center", fontsize=6.4, fontweight="bold")
    if sub:
        ax.text(xy[0] + wh[0] / 2, xy[1] + wh[1] * 0.28, wrap_label(sub, 22), ha="center", va="center", fontsize=5.8, color=MUTED)


def _arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str = RULE) -> None:
    ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="-|>", lw=0.7, color=color, shrinkA=2, shrinkB=2))


def _heatmap_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list("ptl_blue", [HEAT_LOW, "#C9D8E8", HEAT_HIGH])


def write_figure_note(
    notes_dir: Path | str,
    fig_id: str,
    title: str,
    message: str,
    input_table: str,
    visual_encoding: str,
    limitation: str,
) -> Path:
    path = note_paths(notes_dir)[fig_id]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                f"# {title}",
                "",
                f"- Message: {message}",
                f"- Input table: {input_table}",
                f"- Visual encoding: {visual_encoding}",
                f"- Limitation: {limitation}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def create_fig1(data: dict[str, Any], output_dir: Path, notes_dir: Path) -> list[Path]:
    paths = figure_paths(output_dir)
    fig = plt.figure(figsize=(7.4, 6.2), constrained_layout=True)
    axes = fig.subplot_mosaic([["A", "B"], ["C", "D"]], gridspec_kw={"wspace": 0.32, "hspace": 0.40})
    for key, ax in axes.items():
        ax.set_axis_off()
        _panel_label(ax, key, x=-0.02, y=1.00)

    ax = axes["A"]
    ax.set_title("Random anchor versus context-stress split", loc="left", pad=3)
    _draw_box(ax, (0.06, 0.64), (0.34, 0.18), "Random anchor", SEMANTIC_PALETTE["random_split"], "overlap-rich")
    _draw_box(ax, (0.56, 0.64), (0.36, 0.18), "Context-stress split", SEMANTIC_PALETTE["external_holdout"], "held-out boundary")
    _arrow(ax, (0.41, 0.73), (0.55, 0.73), RULE)
    y0 = 0.45
    for i, split in enumerate(SPLIT_ORDER[1:]):
        y = y0 - i * 0.075
        ax.scatter(0.10, y, s=45, color=SEMANTIC_PALETTE[split], edgecolor=TEXT, linewidth=0.25)
        ax.text(0.16, y, pretty_split(split), ha="left", va="center", fontsize=7.0)
    ax.text(0.06, 0.05, "Stress families isolate perturbation, combination,\nsupport, dataset, and external transfer.", fontsize=7.0, color=MUTED)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax = axes["B"]
    ax.set_title("Data-to-signature benchmark construction", loc="left", pad=3)
    steps = [
        ("Perturbation data", SEMANTIC_PALETTE["random_split"]),
        ("QC / pseudobulk", SEMANTIC_PALETTE["low_support_split"]),
        ("Delta signatures", SEMANTIC_PALETTE["unseen_combination_split"]),
        ("Predictors", SEMANTIC_PALETTE["dataset_heldout_split"]),
    ]
    for i, (label, color) in enumerate(steps):
        x = 0.04 + i * 0.24
        _draw_box(ax, (x, 0.52), (0.18, 0.18), label, color)
        if i < len(steps) - 1:
            _arrow(ax, (x + 0.18, 0.61), (x + 0.235, 0.61))
    ax.text(0.04, 0.25, "All model comparisons use the same\nsignature-level target surface.", fontsize=7.0, color=MUTED)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax = axes["C"]
    ax.set_title("Model-by-split evaluation surface", loc="left", pad=3)
    models = ["Control", "Global", "Pert. mean", "kNN", "Ridge"]
    for i, split in enumerate(SPLIT_ORDER):
        for j, model in enumerate(models):
            color = SEMANTIC_PALETTE[split] if j == i % len(models) else WHITE
            rect = patches.Rectangle((0.20 + i * 0.105, 0.20 + j * 0.10), 0.075, 0.065, facecolor=color, edgecolor=RULE, linewidth=0.55, alpha=0.82)
            ax.add_patch(rect)
    for i, split in enumerate(SPLIT_ORDER):
        ax.text(0.238 + i * 0.105, 0.13, str(i + 1), ha="center", va="top", fontsize=6.8)
    for j, model in enumerate(models):
        ax.text(0.02, 0.232 + j * 0.10, model, ha="left", va="center", fontsize=6.9)
    ax.text(0.20, 0.82, "Columns 1-6: random and five stress families.", fontsize=6.9, color=MUTED)
    ax.text(0.20, 0.74, "Repeated seeds expose rank instability.", fontsize=6.9, color=MUTED)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax = axes["D"]
    ax.set_title("PTL reliability layer", loc="left", pad=3)
    _draw_box(ax, (0.05, 0.56), (0.24, 0.17), "Model output", SEMANTIC_PALETTE["random_split"], "confidence")
    _draw_box(ax, (0.38, 0.56), (0.24, 0.17), "PTL score", SEMANTIC_PALETTE["PTL"], "context features")
    _draw_box(ax, (0.74, 0.67), (0.18, 0.12), "Keep", SEMANTIC_PALETTE["unseen_combination_split"])
    _draw_box(ax, (0.74, 0.42), (0.18, 0.12), "Abstain", SEMANTIC_PALETTE["external_holdout"])
    _arrow(ax, (0.29, 0.65), (0.38, 0.65), SEMANTIC_PALETTE["PTL"])
    _arrow(ax, (0.62, 0.65), (0.74, 0.72), SEMANTIC_PALETTE["PTL"])
    _arrow(ax, (0.62, 0.62), (0.74, 0.48), SEMANTIC_PALETTE["PTL"])
    ax.text(0.05, 0.20, "A reliability filter is attached\nafter prediction.", fontsize=7.0, color=MUTED)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    outputs = _save_figure(fig, paths, "fig1")
    write_figure_note(notes_dir, "fig1", "Figure 1. Study design", "Context-stress benchmarking is framed as a reliability workflow.", "Study schematic plus Phase 06-08 outputs.", "Flow boxes encode benchmark stages and reliability decisions.", "Schematic panels are conceptual and do not quantify effect sizes.")
    return outputs


def _ordered_splits(values: Iterable[Any]) -> list[str]:
    present = [str(v) for v in values if pd.notna(v)]
    return [s for s in SPLIT_ORDER if s in present] + sorted(s for s in set(present) if s not in SPLIT_ORDER)


def _split_codes(splits: Iterable[Any]) -> tuple[list[int], list[str]]:
    values = [str(s) for s in splits]
    return list(range(1, len(values) + 1)), [f"{i + 1}: {pretty_split(split)}" for i, split in enumerate(values)]


def create_fig2(data: dict[str, Any], output_dir: Path, notes_dir: Path) -> list[Path]:
    paths = figure_paths(output_dir)
    transfer = data["transfer"]
    examples = data["examples"]
    split_audit = data["split_audit"]
    preprocessing = data["preprocessing"]
    all_metrics = data["all_metrics"]

    fig = plt.figure(figsize=(7.4, 9.4), constrained_layout=True)
    axes = fig.subplot_mosaic(
        [["A", "A"], ["B", "C"], ["D", "D"], ["E", "E"]],
        gridspec_kw={"wspace": 0.34, "hspace": 0.58, "height_ratios": [1.0, 1.0, 1.0, 1.25]},
    )
    for label, ax in axes.items():
        _panel_label(ax, label)

    ax = axes["A"]
    ax.set_title("Dataset composition", loc="left", pad=3)
    if not preprocessing.empty and {"dataset_id", "n_signature_rows", "n_unique_perturbations"}.issubset(preprocessing.columns):
        comp = preprocessing.copy()
        comp["dataset"] = comp["dataset_id"].map(lambda x: wrap_label(x, 24))
        comp = comp.sort_values("n_signature_rows", ascending=True)
        ax.barh(comp["dataset"], comp["n_signature_rows"], color=SEMANTIC_PALETTE["random_split"], height=0.55)
        ax.set_xlabel("Signature rows")
        twin = ax.twiny()
        twin.scatter(comp["n_unique_perturbations"], comp["dataset"], color=SEMANTIC_PALETTE["external_holdout"], s=18, zorder=3)
        twin.set_xlabel("Perturbations", color=SEMANTIC_PALETTE["external_holdout"])
        twin.tick_params(axis="x", colors=SEMANTIC_PALETTE["external_holdout"])
        twin.spines["top"].set_color(RULE)
    else:
        ax.text(0.05, 0.5, "Dataset composition table unavailable.", transform=ax.transAxes, fontsize=7)
    _clean_axis(ax)

    ax = axes["B"]
    ax.set_title("Test signature scale", loc="left", pad=3)
    if not split_audit.empty and {"track", "split_family", "test_units"}.issubset(split_audit.columns):
        audit = split_audit[split_audit["track"].astype(str).eq("signature")].copy()
        scale = audit.groupby("split_family", as_index=False)["test_units"].mean()
    elif not all_metrics.empty:
        scale = all_metrics.groupby("split_family", as_index=False)["n_test_non_control"].mean().rename(columns={"n_test_non_control": "test_units"})
    else:
        scale = transfer.groupby("split_family", as_index=False)[PRIMARY_METRIC].size().rename(columns={"size": "test_units"})
    order = _ordered_splits(scale["split_family"])
    scale["split_family"] = pd.Categorical(scale["split_family"], categories=order, ordered=True)
    scale = scale.sort_values("split_family")
    ax.bar(range(len(scale)), scale["test_units"], color=[SEMANTIC_PALETTE.get(str(s), "#CCCCCC") for s in scale["split_family"]], edgecolor=TEXT, linewidth=0.3, width=0.68)
    codes, _legend = _split_codes(scale["split_family"])
    ax.set_xticks(range(len(scale)), [str(code) for code in codes])
    ax.set_ylabel("Mean test units")
    _clean_axis(ax)

    ax = axes["C"]
    ax.set_title("Support distribution", loc="left", pad=3)
    if not examples.empty and {"split_family", "perturbation_train_signature_count"}.issubset(examples.columns):
        box_data = []
        labels = []
        for split in _ordered_splits(examples["split_family"]):
            values = np.log1p(examples.loc[examples["split_family"].eq(split), "perturbation_train_signature_count"].dropna().to_numpy(dtype=float))
            if len(values):
                box_data.append(values)
                labels.append(split)
        bp = ax.boxplot(box_data, patch_artist=True, showfliers=False, widths=0.46)
        for patch, split in zip(bp["boxes"], labels):
            patch.set_facecolor(SEMANTIC_PALETTE.get(split, "#CCCCCC"))
            patch.set_alpha(0.8)
            patch.set_edgecolor(TEXT)
        codes, _legend = _split_codes(labels)
        ax.set_xticks(range(1, len(labels) + 1), [str(code) for code in codes])
        ax.set_ylabel("log1p train signatures")
    else:
        ax.text(0.05, 0.5, "PTL support features unavailable.", transform=ax.transAxes, fontsize=7)
    _clean_axis(ax)

    ax = axes["D"]
    ax.set_title("Train-test context distance", loc="left", pad=3)
    if not split_audit.empty and {"track", "split_family", "train_test_centroid_l2"}.issubset(split_audit.columns):
        dist = split_audit[split_audit["track"].astype(str).eq("signature")].groupby("split_family", as_index=False)["train_test_centroid_l2"].mean()
        dist = dist.dropna()
        order = _ordered_splits(dist["split_family"])
        dist["split_family"] = pd.Categorical(dist["split_family"], categories=order, ordered=True)
        dist = dist.sort_values("split_family")
        ax.plot(range(len(dist)), dist["train_test_centroid_l2"], color=TEXT, marker="o", markersize=3)
        for i, row in enumerate(dist.itertuples(index=False)):
            ax.scatter(i, row.train_test_centroid_l2, color=SEMANTIC_PALETTE.get(str(row.split_family), "#CCCCCC"), s=32, zorder=3, edgecolor=TEXT, linewidth=0.3)
        codes, _legend = _split_codes(dist["split_family"])
        ax.set_xticks(range(len(dist)), [str(code) for code in codes])
        ax.set_ylabel("Centroid L2")
    else:
        ax.text(0.05, 0.5, "Distance audit unavailable.", transform=ax.transAxes, fontsize=7)
    _clean_axis(ax)

    ax = axes["E"]
    ax.set_title("Split audit matrix", loc="left", pad=3)
    if not split_audit.empty:
        audit = split_audit[split_audit.get("track", pd.Series("", index=split_audit.index)).astype(str).eq("signature")].copy()
        cols = [
            "perturbation_overlap_train_test",
            "dataset_overlap_train_test",
            "reference_key_overlap_train_test",
            "declared_holdout_overlap_train_test",
        ]
        cols = [c for c in cols if c in audit.columns]
        matrix = audit.groupby("split_family")[cols].mean().reindex(_ordered_splits(audit["split_family"]))
        im = ax.imshow(matrix.fillna(0).to_numpy(dtype=float), cmap=_heatmap_cmap(), aspect="auto")
        ax.set_yticks(range(len(matrix.index)), [pretty_split(s) for s in matrix.index])
        ax.set_xticks(range(len(cols)), [wrap_label(c.replace("_", " "), 20) for c in cols], rotation=20, ha="right")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f"{matrix.iloc[i, j]:.1f}" if pd.notna(matrix.iloc[i, j]) else "NA", ha="center", va="center", fontsize=7.2)
        fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="Mean audit value")
    else:
        ax.text(0.05, 0.5, "Split audit unavailable.", transform=ax.transAxes, fontsize=7)
    _clean_axis(ax)

    outputs = _save_figure(fig, paths, "fig2")
    write_figure_note(notes_dir, "fig2", "Figure 2. Benchmark composition and audit", "Benchmark scale, support, context distance, and audit signals are shown before model claims.", "preprocessing_summary.csv; split_audit.csv; ptl_examples.parquet; all_metrics.csv", "Bars encode scale, boxplots encode support, and heatmap cells encode split audit means.", "Audit diagnostics describe constructed splits, not independent biological validation.")
    return outputs


def create_fig3(data: dict[str, Any], output_dir: Path, notes_dir: Path) -> list[Path]:
    paths = figure_paths(output_dir)
    transfer = data["transfer"].copy()
    all_metrics = data["all_metrics"].copy()
    transfer = transfer[transfer["model"].ne(CONTROL_MODEL)].copy()
    splits = _ordered_splits(transfer["split_family"])
    models = [m for m in transfer.sort_values("model")["model"].unique()]

    fig = plt.figure(figsize=(7.4, 8.8), constrained_layout=True)
    axes = fig.subplot_mosaic(
        [["A", "A"], ["B", "B"], ["C", "D"]],
        gridspec_kw={"wspace": 0.45, "hspace": 0.58, "height_ratios": [1.35, 1.05, 1.45], "width_ratios": [1.35, 1.0]},
    )
    for label, ax in axes.items():
        _panel_label(ax, label)

    ax = axes["A"]
    ax.set_title("Model fidelity by split family", loc="left", pad=3)
    pivot = transfer.pivot_table(index="model", columns="split_family", values=PRIMARY_METRIC, aggfunc="mean").reindex(models)[splits]
    ranks = transfer.pivot_table(index="model", columns="split_family", values="family_rank", aggfunc="mean").reindex(models)[splits]
    im = ax.imshow(pivot.to_numpy(dtype=float), cmap=_heatmap_cmap(), aspect="auto")
    ax.set_yticks(range(len(pivot.index)), [pretty_model(m) for m in pivot.index])
    ax.set_xticks(range(len(pivot.columns)), [wrap_label(pretty_split(s), 16) for s in pivot.columns], rotation=20, ha="right")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.iloc[i, j]
            rank = ranks.iloc[i, j]
            label = "NA" if pd.isna(value) else f"{value:.2f}\nr{int(rank)}"
            ax.text(j, i, label, ha="center", va="center", fontsize=7.1, color=TEXT)
    fig.colorbar(im, ax=ax, fraction=0.020, pad=0.012, label="Mean non-control cosine")
    _clean_axis(ax)

    ax = axes["B"]
    ax.set_title("Rank shifts", loc="left", pad=3)
    rank_pivot = ranks.T
    for model in rank_pivot.columns:
        y = rank_pivot[model].to_numpy(dtype=float)
        ax.plot(range(len(rank_pivot.index)), y, marker="o", markersize=2.7, color=MODEL_PALETTE.get(model, "#555555"), alpha=0.95, label=pretty_model(model))
    ax.invert_yaxis()
    ax.set_xticks(range(len(rank_pivot.index)), [wrap_label(pretty_split(s), 14) for s in rank_pivot.index], rotation=20, ha="right")
    ax.set_ylabel("Rank")
    ax.set_yticks(sorted(set(rank_pivot.to_numpy().ravel()[~np.isnan(rank_pivot.to_numpy().ravel())].astype(int))))
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), ncols=1, handlelength=1.1, borderaxespad=0)
    _clean_axis(ax)

    ax = axes["C"]
    ax.set_title("Random-to-stress decay", loc="left", pad=3)
    stress = transfer[transfer["split_family"].ne("random_split")].copy()
    stress["drop"] = stress["mean_random_to_stress_drop"].fillna(stress["random_split_mean_cosine"] - stress[PRIMARY_METRIC])
    stress = stress.sort_values("drop", ascending=False).head(9)
    y = np.arange(len(stress))
    ax.hlines(y, 0, stress["drop"], color=LIGHT_RULE, linewidth=0.8)
    ax.scatter(stress["drop"], y, color=[SEMANTIC_PALETTE.get(s, "#999999") for s in stress["split_family"]], s=24, edgecolor=TEXT, linewidth=0.25)
    ax.set_yticks(y, [wrap_label(f"{pretty_model(m)} | {pretty_split(s)}", 18) for m, s in zip(stress["model"], stress["split_family"])])
    ax.tick_params(axis="y", labelsize=6.6)
    ax.set_xlabel("Cosine drop from random anchor")
    ax.axvline(0, color=RULE, lw=0.7)
    _clean_axis(ax)

    ax = axes["D"]
    ax.set_title("Dataset and external transfer zoom", loc="left", pad=3)
    if not all_metrics.empty and {"split_family", "model", PRIMARY_METRIC}.issubset(all_metrics.columns):
        zoom = all_metrics[all_metrics["split_family"].isin(["dataset_heldout_split", "external_holdout"]) & all_metrics["model"].ne(CONTROL_MODEL)].copy()
        x_positions = {"dataset_heldout_split": 0, "external_holdout": 1}
        for model, part in zoom.groupby("model"):
            xs = part["split_family"].map(x_positions).astype(float) + np.linspace(-0.10, 0.10, len(part))
            ax.scatter(xs, part[PRIMARY_METRIC], s=12, alpha=0.55, color=MODEL_PALETTE.get(model, "#555555"), edgecolor="none")
            means = part.groupby("split_family")[PRIMARY_METRIC].mean()
            if len(means) == 2:
                ax.plot([0, 1], [means.get("dataset_heldout_split"), means.get("external_holdout")], color=MODEL_PALETTE.get(model, "#555555"), lw=0.9, label=pretty_model(model))
        ax.set_xticks([0, 1], ["Dataset heldout", "External holdout"])
        ax.set_ylabel("Mean non-control cosine")
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend(handles, labels, loc="best", ncols=2, handlelength=1.0)
    else:
        ax.text(0.05, 0.5, "Run-level metrics unavailable.", transform=ax.transAxes, fontsize=7)
    _clean_axis(ax)

    outputs = _save_figure(fig, paths, "fig3")
    write_figure_note(notes_dir, "fig3", "Figure 3. Ranking instability and transfer decay", "Random-split ranking does not define a universal best model under stress.", "transfer_decay_summary.csv; all_metrics.csv", "Heatmap cells show fidelity and rank; lines and lollipops show rank and fidelity changes.", "The figure compares transparent baselines rather than all possible virtual-cell models.")
    return outputs


def _select_prediction_groups(predictions: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if predictions.empty:
        return {}
    groups: dict[str, pd.DataFrame] = {}
    naive = predictions[predictions["estimator"].eq("naive_confidence")]
    if not naive.empty:
        groups["Naive confidence"] = naive
    preferred = predictions[predictions["ablation"].eq("full_PTL") & predictions["estimator"].eq("random_forest")]
    if preferred.empty:
        preferred = predictions[predictions["ablation"].eq("full_PTL")]
    if preferred.empty:
        preferred = predictions[predictions["estimator"].ne("naive_confidence")]
    if not preferred.empty:
        if preferred["estimator"].nunique() > 1:
            preferred = preferred[preferred["estimator"].eq(preferred["estimator"].iloc[0])]
        groups["Full PTL"] = preferred
    alt = predictions[predictions["ablation"].eq("no_context_distance") & predictions["estimator"].eq("random_forest")]
    if not alt.empty:
        groups["No context distance"] = alt
    return groups


def _selective_curve(frame: pd.DataFrame, coverages: Iterable[float]) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    ordered = frame.sort_values("score", ascending=False).reset_index(drop=True)
    y = ordered["y_true"].to_numpy(dtype=int)
    risk = ordered["target_risk"].to_numpy(dtype=float)
    for coverage in coverages:
        keep = max(1, int(math.ceil(len(ordered) * coverage)))
        selected_y = y[:keep]
        selected_risk = risk[:keep]
        rows.append(
            {
                "coverage": float(coverage),
                "false_transportability_rate": float(np.mean(selected_y == 0)),
                "selective_risk": float(np.mean(selected_risk)),
            }
        )
    return pd.DataFrame(rows)


def create_fig4(data: dict[str, Any], output_dir: Path, notes_dir: Path) -> list[Path]:
    paths = figure_paths(output_dir)
    predictions = data["predictions"].copy()
    selective = data["selective"].copy()
    groups = _select_prediction_groups(predictions)

    fig = plt.figure(figsize=(7.8, 7.2), constrained_layout=True)
    axes = fig.subplot_mosaic(
        [["A", "A"], ["B", "C"], ["D", "E"]],
        gridspec_kw={"wspace": 0.36, "hspace": 0.36, "height_ratios": [0.36, 1.08, 1.02]},
    )
    for label, ax in axes.items():
        _panel_label(ax, label)

    ax = axes["A"]
    ax.set_axis_off()
    ax.set_title("PTL architecture", loc="left", pad=3)
    blocks = [
        ((0.06, 0.72), "Model/split", SEMANTIC_PALETTE["random_split"]),
        ((0.06, 0.55), "Support", SEMANTIC_PALETTE["low_support_split"]),
        ((0.06, 0.38), "Novelty", SEMANTIC_PALETTE["unseen_perturbation_split"]),
        ((0.06, 0.21), "Distance", SEMANTIC_PALETTE["dataset_heldout_split"]),
    ]
    for xy, label, color in blocks:
        _draw_box(ax, xy, (0.24, 0.10), label, color)
        _arrow(ax, (0.30, xy[1] + 0.05), (0.45, 0.51), SEMANTIC_PALETTE["PTL"])
    _draw_box(ax, (0.45, 0.43), (0.25, 0.14), "PTL score", SEMANTIC_PALETTE["PTL"], "model-agnostic")
    _draw_box(ax, (0.78, 0.60), (0.17, 0.10), "Keep", SEMANTIC_PALETTE["unseen_combination_split"])
    _draw_box(ax, (0.78, 0.36), (0.17, 0.10), "Abstain", SEMANTIC_PALETTE["external_holdout"])
    _arrow(ax, (0.70, 0.52), (0.78, 0.65), SEMANTIC_PALETTE["PTL"])
    _arrow(ax, (0.70, 0.50), (0.78, 0.41), SEMANTIC_PALETTE["PTL"])
    ax.text(0.06, 0.03, "Feature groups are computed from local benchmark artifacts.", fontsize=7.0, color=MUTED)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax = axes["B"]
    ax.set_title("ROC curve", loc="left", pad=3)
    if groups:
        for name, frame in groups.items():
            if frame["y_true"].nunique() < 2:
                continue
            fpr, tpr, _ = roc_curve(frame["y_true"].astype(int), frame["score"].astype(float))
            roc_auc = auc(fpr, tpr)
            color = SEMANTIC_PALETTE["PTL"] if "PTL" in name or "context" in name else SEMANTIC_PALETTE["naive_confidence"]
            ls = "--" if name == "No context distance" else "-"
            short_name = name.replace("Naive confidence", "Naive").replace("No context distance", "No context")
            ax.plot(fpr, tpr, label=f"{short_name} ({roc_auc:.2f})", color=color, linestyle=ls)
        ax.plot([0, 1], [0, 1], color=LIGHT_RULE, lw=0.8)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.legend(loc="lower right", handlelength=1.2, fontsize=6.8)
    else:
        ax.text(0.05, 0.5, "PTL prediction scores unavailable.", transform=ax.transAxes, fontsize=7)
    _clean_axis(ax)

    ax = axes["C"]
    ax.set_title("Precision-recall curve", loc="left", pad=3)
    if groups:
        for name, frame in groups.items():
            if frame["y_true"].nunique() < 2:
                continue
            precision, recall, _ = precision_recall_curve(frame["y_true"].astype(int), frame["score"].astype(float))
            ap = auc(recall, precision)
            color = SEMANTIC_PALETTE["PTL"] if "PTL" in name or "context" in name else SEMANTIC_PALETTE["naive_confidence"]
            ls = "--" if name == "No context distance" else "-"
            short_name = name.replace("Naive confidence", "Naive").replace("No context distance", "No context")
            ax.plot(recall, precision, label=f"{short_name} ({ap:.2f})", color=color, linestyle=ls)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.legend(loc="lower left", handlelength=1.2, fontsize=6.8)
    else:
        ax.text(0.05, 0.5, "PTL prediction scores unavailable.", transform=ax.transAxes, fontsize=7)
    _clean_axis(ax)

    ax = axes["D"]
    ax.set_title("False transportability filtering", loc="left", pad=3)
    best = selective.copy()
    best["label"] = best.apply(lambda r: "Naive confidence" if r["estimator"] == "naive_confidence" else f"{r['ablation']} | {r['estimator']}", axis=1)
    best = best.sort_values("mean_false_transportability_rate").head(6)
    colors = [SEMANTIC_PALETTE["naive_confidence"] if "Naive" in label else SEMANTIC_PALETTE["PTL"] for label in best["label"]]
    y = np.arange(len(best))
    ax.barh(y, best["mean_false_transportability_rate"], color=colors, edgecolor=TEXT, linewidth=0.3, height=0.55)
    ax.set_yticks(y, [wrap_label(label, 24) for label in best["label"]])
    ax.set_xlabel("Mean false transportability rate")
    _clean_axis(ax)

    ax = axes["E"]
    ax.set_title("Risk-coverage behavior", loc="left", pad=3)
    if groups:
        for name, frame in groups.items():
            curve = _selective_curve(frame, [0.2, 0.4, 0.6, 0.8, 1.0])
            color = SEMANTIC_PALETTE["PTL"] if "PTL" in name or "context" in name else SEMANTIC_PALETTE["naive_confidence"]
            ls = "--" if name == "No context distance" else "-"
            short_name = name.replace("Naive confidence", "Naive").replace("No context distance", "No context")
            ax.plot(curve["coverage"], curve["selective_risk"], marker="o", markersize=2.8, color=color, linestyle=ls, label=short_name)
        ax.set_xlabel("Coverage")
        ax.set_ylabel("Selected mean risk")
        ax.legend(loc="best", fontsize=6.8)
    else:
        ax.text(0.05, 0.5, "PTL prediction scores unavailable.", transform=ax.transAxes, fontsize=7)
    _clean_axis(ax)

    outputs = _save_figure(fig, paths, "fig4")
    write_figure_note(notes_dir, "fig4", "Figure 4. PTL selective filtering", "PTL improves the primary false-transportability filtering view relative to naive confidence.", "ptl_oof_predictions.csv; selective_prediction_summary.csv", "ROC/PR curves use per-example PTL scores; bars and curves summarize selective prediction.", "Cosine-risk ranking remains a secondary conservative view and is not a claim of uniformly high retained fidelity.")
    return outputs


def _load_representative_cases(examples: pd.DataFrame, manifest: pd.DataFrame, max_cases: int = 2) -> list[dict[str, Any]]:
    if examples.empty or manifest.empty:
        return []
    required = {"run_id", "signature_id", "target_fidelity", "target_risk", "failure_mode"}
    if not required.issubset(examples.columns) or "output_dir" not in manifest.columns:
        return []
    candidates = examples[examples["failure_mode"].astype(str).ne("transportable")].copy()
    if candidates.empty:
        return []
    candidates = candidates.sort_values(["target_risk", "target_fidelity"], ascending=[False, True]).head(12)
    out_map = manifest.set_index("run_id")["output_dir"].to_dict()
    cases: list[dict[str, Any]] = []
    for row in candidates.itertuples(index=False):
        out_dir = Path(str(out_map.get(row.run_id, "")))
        pred_path = out_dir / "test_predictions.npz"
        meta_path = out_dir / "test_metadata.parquet"
        if not pred_path.exists() or not meta_path.exists():
            continue
        meta = pd.read_parquet(meta_path)
        match = meta.index[meta["signature_id"].astype(str).eq(str(row.signature_id))]
        if len(match) == 0:
            continue
        arrays = np.load(pred_path)
        if "y_true" not in arrays.files or "y_pred" not in arrays.files:
            continue
        idx = int(match[0])
        true = arrays["y_true"][idx].astype(float)
        pred = arrays["y_pred"][idx].astype(float)
        if true.size == 0 or pred.size == 0:
            continue
        top = np.argsort(np.abs(true))[-min(250, true.size) :]
        cases.append(
            {
                "title": f"{pretty_model(row.model)} | {pretty_split(row.split_family)}",
                "subtitle": f"{str(row.perturbation_label)[:22]} | cosine {float(row.target_fidelity):.2f}",
                "true": true[top],
                "pred": pred[top],
            }
        )
        if len(cases) >= max_cases:
            break
    return cases


def create_fig5(data: dict[str, Any], output_dir: Path, notes_dir: Path) -> list[Path]:
    paths = figure_paths(output_dir)
    failure = data["failure"].copy()
    examples = data["examples"].copy()
    manifest = data["baseline_manifest"].copy()

    fig = plt.figure(figsize=(7.8, 9.8), constrained_layout=True)
    axes = fig.subplot_mosaic(
        [["A", "A"], ["B", "B"], ["C", "C"], ["D", "D"], ["E", "E"]],
        gridspec_kw={"wspace": 0.34, "hspace": 0.48, "height_ratios": [0.38, 1.12, 1.02, 0.82, 1.45]},
    )
    for label, ax in axes.items():
        _panel_label(ax, label)

    ax = axes["A"]
    ax.set_axis_off()
    ax.set_title("Failure taxonomy", loc="left", pad=3)
    taxonomy = [
        ("Transportable", 0, SEMANTIC_PALETTE["PTL"]),
        ("Below threshold", 1, SEMANTIC_PALETTE["low_support_split"]),
        ("High risk", 2, SEMANTIC_PALETTE["unseen_perturbation_split"]),
        ("Severe failure", 3, SEMANTIC_PALETTE["external_holdout"]),
    ]
    for i, (label, severity, color) in enumerate(taxonomy):
        x = 0.06 + i * 0.23
        _draw_box(ax, (x, 0.36), (0.18, 0.24), label, color)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax = axes["B"]
    ax.set_title("Failure share by model and split", loc="left", pad=3)
    severe = failure.copy()
    severe["weighted_share"] = severe["failure_mode_share"] * severe["failure_severity"]
    matrix = severe.groupby(["model", "split_family"])["weighted_share"].sum().unstack("split_family").reindex(
        [m for m in MODEL_PALETTE if m in severe["model"].unique()]
    )
    columns = _ordered_splits(matrix.columns)
    matrix = matrix.reindex(columns=columns)
    im = ax.imshow(matrix.fillna(0).to_numpy(dtype=float), cmap=LinearSegmentedColormap.from_list("severity", [HEAT_LOW, "#F3C0B9", SEMANTIC_PALETTE["external_holdout"]]), aspect="auto")
    ax.set_yticks(range(len(matrix.index)), [pretty_model(m) for m in matrix.index])
    ax.set_xticks(range(len(matrix.columns)), [wrap_label(pretty_split(s), 16) for s in matrix.columns], rotation=18, ha="right")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix.iloc[i, j]:.2f}" if pd.notna(matrix.iloc[i, j]) else "NA", ha="center", va="center", fontsize=6.9)
    fig.colorbar(im, ax=ax, fraction=0.030, pad=0.02, label="Share x severity")
    _clean_axis(ax)

    ax = axes["C"]
    ax.set_title("Severity composition", loc="left", pad=3)
    comp = failure.groupby(["split_family", "failure_mode"], as_index=False)["n_signatures"].sum()
    pivot = comp.pivot(index="split_family", columns="failure_mode", values="n_signatures").fillna(0).reindex(_ordered_splits(comp["split_family"]))
    pivot = pivot.div(pivot.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    colors = {
        "transportable": SEMANTIC_PALETTE["PTL"],
        "below_transport_threshold": SEMANTIC_PALETTE["low_support_split"],
        "non_transportable": "#CFCFCF",
        "high_risk_failure": SEMANTIC_PALETTE["unseen_perturbation_split"],
        "severe_failure": SEMANTIC_PALETTE["external_holdout"],
    }
    left = np.zeros(len(pivot))
    for col in pivot.columns:
        ax.barh(range(len(pivot)), pivot[col], left=left, color=colors.get(col, "#BBBBBB"), height=0.55, edgecolor=WHITE, linewidth=0.35, label=wrap_label(col, 14))
        left += pivot[col].to_numpy(dtype=float)
    ax.set_yticks(range(len(pivot)), [pretty_split(s) for s in pivot.index])
    ax.set_xlabel("Within-split share")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncols=5, fontsize=6.7, handlelength=1.0, frameon=False)
    _clean_axis(ax)

    ax = axes["D"]
    ax.set_title("Context enrichment", loc="left", pad=3)
    if "context_signal" in failure.columns:
        ctx = failure.groupby("context_signal", as_index=False).agg(n=("n_signatures", "sum"), severity=("failure_severity", "mean"), share=("failure_mode_share", "mean"))
        ctx = ctx.sort_values("n", ascending=True)
        ax.scatter(ctx["share"], range(len(ctx)), s=40 + 18 * ctx["severity"], color=SEMANTIC_PALETTE["external_holdout"], alpha=0.85, edgecolor=TEXT, linewidth=0.3)
        ax.set_yticks(range(len(ctx)), [wrap_label(v, 24) for v in ctx["context_signal"]])
        ax.set_xlabel("Mean mode share")
    else:
        ax.text(0.05, 0.5, "Context signals unavailable.", transform=ax.transAxes, fontsize=7)
    _clean_axis(ax)

    ax = axes["E"]
    ax.set_title("Representative failure cases", loc="left", pad=3)
    cases = _load_representative_cases(examples, manifest)
    if cases:
        for i, case in enumerate(cases):
            inset = ax.inset_axes([0.05 + i * 0.49, 0.08, 0.43, 0.64])
            inset.scatter(case["true"], case["pred"], s=6, alpha=0.45, color=SEMANTIC_PALETTE["external_holdout"], edgecolor="none")
            mn = float(min(np.nanmin(case["true"]), np.nanmin(case["pred"])))
            mx = float(max(np.nanmax(case["true"]), np.nanmax(case["pred"])))
            inset.plot([mn, mx], [mn, mx], color=RULE, lw=0.7)
            inset.set_title(wrap_label(case["title"], 24), fontsize=6.8)
            inset.set_xlabel("True delta", fontsize=6.4)
            inset.set_ylabel("Pred delta", fontsize=6.4)
            inset.tick_params(labelsize=6.2, length=2)
        ax.set_axis_off()
    elif not examples.empty:
        worst = examples[examples["failure_mode"].astype(str).ne("transportable")].sort_values(["target_risk", "target_fidelity"], ascending=[False, True]).head(3)
        ax.set_axis_off()
        for i, row in enumerate(worst.itertuples(index=False)):
            y = 0.78 - i * 0.26
            _draw_box(
                ax,
                (0.05, y),
                (0.88, 0.17),
                f"{pretty_model(row.model)} | {pretty_split(row.split_family)}",
                SEMANTIC_PALETTE.get(str(row.split_family), "#999999"),
                f"risk {float(row.target_risk):.2f}; cosine {float(row.target_fidelity):.2f}",
            )
    else:
        ax.text(0.05, 0.5, "Representative cases unavailable.", transform=ax.transAxes, fontsize=7)

    outputs = _save_figure(fig, paths, "fig5")
    write_figure_note(notes_dir, "fig5", "Figure 5. Failure-mode atlas", "Failures concentrate around novelty, low support, and transfer boundaries.", "failure_mode_atlas.csv; ptl_examples.parquet; baseline prediction arrays when available", "Heatmaps and stacked bars show severity and share; case panels show true-predicted delta structure or scalar fallback cards.", "The atlas is descriptive and does not establish biological mechanism.")
    return outputs


def create_supp_fig1(data: dict[str, Any], output_dir: Path, notes_dir: Path) -> list[Path]:
    paths = figure_paths(output_dir)
    all_metrics = data["all_metrics"].copy()
    transfer = data["transfer"].copy()
    selective = data["selective"].copy()

    fig = plt.figure(figsize=(7.2, 4.6), constrained_layout=True)
    axes = fig.subplot_mosaic([["A", "B"], ["C", "D"]], gridspec_kw={"wspace": 0.30, "hspace": 0.34})
    for label, ax in axes.items():
        _panel_label(ax, label)

    ax = axes["A"]
    ax.set_title("Run matrix completion", loc="left", pad=3)
    if not all_metrics.empty:
        matrix = all_metrics.pivot_table(index="model", columns="split_family", values="run_id", aggfunc="count").reindex(
            [m for m in MODEL_PALETTE if m in all_metrics["model"].unique()]
        )
        matrix = matrix.reindex(columns=_ordered_splits(matrix.columns))
        im = ax.imshow(matrix.fillna(0), cmap=_heatmap_cmap(), aspect="auto")
        ax.set_yticks(range(len(matrix.index)), [pretty_model(m) for m in matrix.index])
        ax.set_xticks(range(len(matrix.columns)), [wrap_label(pretty_split(s), 12) for s in matrix.columns], rotation=30, ha="right")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, str(int(matrix.iloc[i, j])) if pd.notna(matrix.iloc[i, j]) else "0", ha="center", va="center", fontsize=6.4)
        fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="Runs")
    else:
        ax.text(0.05, 0.5, "Run metrics unavailable.", transform=ax.transAxes, fontsize=7)
    _clean_axis(ax)

    ax = axes["B"]
    ax.set_title("Seed-level variation", loc="left", pad=3)
    if not all_metrics.empty and "seed" in all_metrics.columns:
        stress = all_metrics[all_metrics["model"].ne(CONTROL_MODEL)]
        order = _ordered_splits(stress["split_family"])
        data_box = [stress.loc[stress["split_family"].eq(split), PRIMARY_METRIC].dropna().to_numpy(dtype=float) for split in order]
        bp = ax.boxplot(data_box, patch_artist=True, showfliers=False)
        for patch, split in zip(bp["boxes"], order):
            patch.set_facecolor(SEMANTIC_PALETTE.get(split, "#CCCCCC"))
            patch.set_edgecolor(TEXT)
            patch.set_alpha(0.75)
        ax.set_xticks(range(1, len(order) + 1), [wrap_label(pretty_split(s), 10) for s in order], rotation=35, ha="right")
        ax.set_ylabel("Run cosine")
    else:
        ax.text(0.05, 0.5, "Seed metrics unavailable.", transform=ax.transAxes, fontsize=7)
    _clean_axis(ax)

    ax = axes["C"]
    ax.set_title("Evidence tiers", loc="left", pad=3)
    tiers = pd.DataFrame(
        {
            "tier": ["Ranking instability", "PTL filtering", "Failure atlas", "Biological interpretation"],
            "strength": [0.95, 0.82, 0.70, 0.30],
            "color": [SEMANTIC_PALETTE["random_split"], SEMANTIC_PALETTE["PTL"], SEMANTIC_PALETTE["external_holdout"], SEMANTIC_PALETTE["naive_confidence"]],
        }
    )
    ax.barh(range(len(tiers)), tiers["strength"], color=tiers["color"], edgecolor=TEXT, linewidth=0.3)
    ax.set_yticks(range(len(tiers)), tiers["tier"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Relative evidence support")
    _clean_axis(ax)

    ax = axes["D"]
    ax.set_title("PTL ablation audit", loc="left", pad=3)
    best = selective[selective["estimator"].ne("naive_confidence")].sort_values("mean_false_transportability_rate").head(6)
    if not best.empty:
        ax.scatter(best["mean_false_transportability_rate"], best["mean_selective_risk"], s=34, color=SEMANTIC_PALETTE["PTL"], edgecolor=TEXT, linewidth=0.3)
        for row in best.itertuples(index=False):
            ax.text(row.mean_false_transportability_rate, row.mean_selective_risk, wrap_label(row.ablation, 12), fontsize=5.8, ha="left", va="bottom")
        ax.set_xlabel("False transportability")
        ax.set_ylabel("Selective risk")
    else:
        ax.text(0.05, 0.5, "PTL ablations unavailable.", transform=ax.transAxes, fontsize=7)
    _clean_axis(ax)

    outputs = _save_figure(fig, paths, "supp_fig1")
    write_figure_note(notes_dir, "supp_fig1", "Supplementary Figure 1. Robustness audit", "Run completion, seed dispersion, evidence tiers, and PTL ablations support bounded interpretation.", "all_metrics.csv; transfer_decay_summary.csv; selective_prediction_summary.csv", "Heatmaps, boxplots, bars, and scatter points summarize audit dimensions.", "This is a robustness supplement, not a primary claim figure.")
    return outputs


def run_phase09(inputs: Phase09Inputs, output_dir: Path | str, notes_dir: Path | str, log_file: Path | None = None) -> list[Path]:
    apply_publication_style()
    output = Path(output_dir)
    notes = Path(notes_dir)
    output.mkdir(parents=True, exist_ok=True)
    notes.mkdir(parents=True, exist_ok=True)
    logger = PhaseLogger(log_file or LOG_FILE)
    logger.log("Starting Phase 09 figure rebuild with Python/matplotlib-first rendering.")
    logger.log("Skill gate checked through SKILLS_INDEX.md; scientific-visualization and matplotlib skills are active.")

    data = validate_inputs(inputs)
    outputs: list[Path] = []
    for create in (create_fig1, create_fig2, create_fig3, create_fig4, create_fig5, create_supp_fig1):
        created = create(data, output, notes)
        outputs.extend(created)
        logger.log(f"Wrote {len(created)} files for {create.__name__}.")
    logger.log("Phase 09 figure rebuild completed.")
    return outputs


def main() -> None:
    args = parse_args()
    inputs = Phase09Inputs(
        main_findings=Path(args.main_findings),
        transfer_decay=Path(args.transfer_decay),
        selective_prediction=Path(args.selective_prediction),
        failure_atlas=Path(args.failure_atlas),
        narrative=Path(args.narrative),
        ptl_predictions=Path(args.ptl_predictions),
        ptl_examples=Path(args.ptl_examples),
        all_metrics=Path(args.all_metrics),
        split_audit=Path(args.split_audit),
        data_inventory=Path(args.data_inventory),
        preprocessing_summary=Path(args.preprocessing_summary),
        baseline_manifest=Path(args.baseline_manifest),
    )
    run_phase09(inputs=inputs, output_dir=Path(args.output_dir), notes_dir=Path(args.notes_dir), log_file=Path(args.log_file))


if __name__ == "__main__":
    main()
