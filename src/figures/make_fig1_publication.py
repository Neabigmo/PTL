"""
Publication-grade Figure 1: PTL transportability-reliability framework
Nature/Cell level visualization with modular architecture

Author: Generated for Perturbation Transportability paper
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch, Polygon
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
from pathlib import Path

# =============================================================================
# STYLE CONFIGURATION
# =============================================================================

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.linewidth": 1,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "svg.fonttype": "path",
})

# Colorblind-safe palette
COLORS = {
    "blue": "#0077BB",
    "cyan": "#33BBEE",
    "teal": "#009988",
    "orange": "#EE7733",
    "red": "#CC3311",
    "rose": "#EE3377",
    "purple": "#AA3377",
    "grey": "#BBBBBB",
    "light_blue": "#88CCEE",
    "light_orange": "#FBB4AE",
    "success": "#228833",
    "failure": "#CC3311",
    "panel_bg": "#FFFFFF",
    "text": "#1a1a1a",
    "text_secondary": "#666666",
    "grid": "#E5E5E5",
}


# =============================================================================
# DATA LAYER - Mock data generators
# =============================================================================

def generate_panel_A_data():
    """Random-split illusion: overlapping train/test clusters with high score threshold

    Key message: High-scorers cluster in a dashed circle → Model learned training patterns, not generalizable features.
    """
    np.random.seed(42)
    n = 100

    # Train data (in-distribution)
    train_x = np.random.normal(0.35, 0.12, n)
    train_y = np.random.normal(0.55, 0.12, n)

    # Test data (same distribution, low context shift)
    test_x = np.random.normal(0.38, 0.12, n)
    test_y = np.random.normal(0.58, 0.12, n)

    # Higher threshold for high-scorers (0.62 instead of 0.55)
    threshold = 0.62

    # Identify high-scoring points (above threshold)
    train_high = (train_y > threshold) | (train_x > 0.5)
    test_high = (test_y > threshold) | (test_x > 0.5)

    # Cluster center for dashed circle (where high-scorers aggregate)
    cluster_center_x = 0.38
    cluster_center_y = 0.62
    cluster_radius = 0.15  # Smaller dashed circle

    return {
        "train_x": train_x, "train_y": train_y,
        "test_x": test_x, "test_y": test_y,
        "train_high": train_high, "test_high": test_high,
        "threshold": threshold,
        "cluster_center_x": cluster_center_x,
        "cluster_center_y": cluster_center_y,
        "cluster_radius": cluster_radius
    }


def generate_panel_B_data():
    """Context stress ladder levels"""
    levels = [
        {"name": "Random Split", "stress": 0.15, "color": COLORS["blue"]},
        {"name": "Held-out Perturb.", "stress": 0.40, "color": COLORS["cyan"]},
        {"name": "Held-out Comb.", "stress": 0.60, "color": COLORS["orange"]},
        {"name": "Low Support", "stress": 0.75, "color": COLORS["orange"]},
        {"name": "Dataset Holdout", "stress": 0.88, "color": COLORS["red"]},
        {"name": "External Holdout", "stress": 0.98, "color": COLORS["red"]},
    ]
    return levels


def generate_panel_C_data():
    """Performance collapse heatmap data"""
    models = ["GEARS", "CPA", "scGen", "Ridge", "Random"]
    splits = ["Random", "Perturb", "Comb", "Low Sup.", "Dataset", "External"]

    # Simulate realistic decay pattern
    base_perf = [0.42, 0.40, 0.38, 0.35, 0.30, 0.25]
    np.random.seed(42)
    data = []
    for m in models:
        row = []
        for s, base in zip(splits, base_perf):
            noise = np.random.normal(0, 0.05)
            if s == "Random":
                row.append(max(0, base + noise))
            else:
                decay = (splits.index(s)) * 0.07
                row.append(max(0, base - decay + noise))
        data.append(row)

    return {"models": models, "splits": splits, "data": data,
            "highlight": (0, 5), "highlight_val": 0.07}


def generate_panel_D_data():
    """PTL pipeline flow"""
    return {
        "steps": [
            {"name": "VC Model", "desc": "Perturbation\nPrediction", "color": COLORS["blue"]},
            {"name": "PTL Layer", "desc": "Transportability\nAssessment", "color": COLORS["orange"]},
            {"name": "Decision", "desc": "Keep / Abstain", "color": COLORS["purple"]},
        ],
        "features": ["novelty", "support", "uncertainty", "context dist."]
    }


def generate_panel_E_data():
    """Reliability gain comparison"""
    return {
        "categories": ["Naive\n(Confidence)", "PTL\n(Transportability)"],
        "values": [0.504, 0.174],
        "improvement": 0.65,
        "labels": ["False Transportability Rate", "Lower is better"]
    }


def generate_panel_F_data():
    """Failure atlas scatter plot

    User requirement: Low novelty + high support = SAFE (above diagonal line)
                       High novelty + low support = FAILURE (below diagonal line)

    Distribution: More failures in bottom-right, more passes in top-left
    """
    np.random.seed(42)

    # Failure region: bottom-right (high novelty, low support)
    # Cluster around the corner
    n_failure_clusters = 4
    failure_points_per_cluster = 35
    failure_novelty = []
    failure_support = []

    for _ in range(n_failure_clusters):
        # High novelty cluster center (x around 0.7-0.95)
        cx = np.random.uniform(0.7, 0.95)
        # Low support cluster center (y around 0.1-0.35)
        cy = np.random.uniform(0.1, 0.35)
        failure_novelty.extend(np.random.normal(cx, 0.12, failure_points_per_cluster))
        failure_support.extend(np.random.normal(cy, 0.1, failure_points_per_cluster))

    failure_novelty = np.array(failure_novelty)
    failure_support = np.array(failure_support)
    # Clip to valid range
    failure_novelty = np.clip(failure_novelty, 0.5, 1.0)
    failure_support = np.clip(failure_support, 0.0, 0.5)
    severity = np.random.exponential(0.6, len(failure_novelty))
    severity = np.clip(severity, 0, 1)

    # Pass region: top-left (low novelty, high support)
    # Cluster around the corner
    n_pass_clusters = 2
    pass_points_per_cluster = 20
    pass_novelty = []
    pass_support = []

    for _ in range(n_pass_clusters):
        # Low novelty cluster center (x around 0.05-0.25)
        cx = np.random.uniform(0.05, 0.25)
        # High support cluster center (y around 0.75-0.95)
        cy = np.random.uniform(0.75, 0.95)
        pass_novelty.extend(np.random.normal(cx, 0.08, pass_points_per_cluster))
        pass_support.extend(np.random.normal(cy, 0.08, pass_points_per_cluster))

    pass_novelty = np.array(pass_novelty)
    pass_support = np.array(pass_support)
    pass_novelty = np.clip(pass_novelty, 0.0, 0.4)
    pass_support = np.clip(pass_support, 0.6, 1.0)

    return {
        "failure_support": failure_support, "failure_novelty": failure_novelty,
        "failure_severity": severity,
        "pass_support": pass_support, "pass_novelty": pass_novelty,
        "pass_severity": np.zeros(len(pass_novelty))
    }


# =============================================================================
# PLOT LAYER - Panel functions
# =============================================================================

def plot_panel_A(ax, data):
    """Random-split illusion visualization

    Shows high-scoring points (with small red dot inside) clustering together
    because the model learned training data patterns, not generalizable features.
    """
    ax.set_facecolor(COLORS["panel_bg"])

    # Draw ALL train/test points first
    ax.scatter(data["train_x"], data["train_y"],
               c=COLORS["blue"], s=25, alpha=0.5, edgecolors="white", linewidth=0.3)
    ax.scatter(data["test_x"], data["test_y"],
               c=COLORS["cyan"], s=25, alpha=0.5, edgecolors="white", linewidth=0.3)

    # Highlight high-scoring points with small RED dot inside colored dot
    # Train high-scorers
    for x, y in zip(data["train_x"][data["train_high"]], data["train_y"][data["train_high"]]):
        ax.scatter([x], [y], c=COLORS["red"], s=7, marker="o", zorder=10)

    # Test high-scorers
    for x, y in zip(data["test_x"][data["test_high"]], data["test_y"][data["train_high"]]):
        ax.scatter([x], [y], c=COLORS["red"], s=7, marker="o", zorder=10)

    # Draw SMALLER dashed circle where high-scorers cluster
    circle = plt.Circle((data["cluster_center_x"], data["cluster_center_y"]),
                        data["cluster_radius"],
                        fill=False, linestyle="--", linewidth=2.5,
                        color=COLORS["red"], alpha=0.9)
    ax.add_patch(circle)

    # Legend
    ax.scatter([], [], c=COLORS["blue"], s=25, alpha=0.5, edgecolors="white", label="Train")
    ax.scatter([], [], c=COLORS["cyan"], s=25, alpha=0.5, edgecolors="white", label="Test")
    ax.scatter([], [], c=COLORS["red"], s=7, marker="o", label="High Score")
    ax.legend(loc="lower right", frameon=True, fancybox=True, fontsize=7)

    # Clearer annotation explaining the message
    ax.text(0.72, 0.95, "High scorers cluster", transform=ax.transAxes,
            fontsize=8, verticalalignment="top", color=COLORS["red"], fontweight="bold")
    ax.text(0.72, 0.88, "inside dashed circle →", transform=ax.transAxes,
            fontsize=7, verticalalignment="top", color=COLORS["red"])
    ax.text(0.72, 0.81, "Model learned", transform=ax.transAxes,
            fontsize=7, verticalalignment="top", color=COLORS["text"])
    ax.text(0.72, 0.74, "training patterns!", transform=ax.transAxes,
            fontsize=7, verticalalignment="top", color=COLORS["text"])

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Context similarity", fontsize=8)
    ax.set_ylabel("Response similarity", fontsize=8)
    ax.set_title("Random-split illusion", fontweight="bold", pad=12)
    ax.tick_params(labelsize=7)


def plot_panel_B(ax, data):
    """Context stress ladder with labels on LEFT side (bold), scores on RIGHT side of gray bar (black)"""
    ax.set_facecolor(COLORS["panel_bg"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    y_positions = np.linspace(0.88, 0.12, len(data))
    bar_height = 0.10

    for i, (level, y) in enumerate(zip(data, y_positions)):
        # Background bar (gray track)
        bar_width = 0.55
        bar_start = 0.08
        ax.barh(y, bar_width, height=bar_height, left=bar_start,
                color=COLORS["grey"], edgecolor="none", alpha=0.3)

        # Colored fill showing stress level
        fill_width = bar_width * level["stress"]
        ax.barh(y, fill_width, height=bar_height, left=bar_start,
                color=level["color"], edgecolor="none", alpha=0.85)

        # Label on LEFT side of bar (bold, white)
        bar_center_y = y + bar_height / 2
        ax.text(bar_start + 0.02, bar_center_y, level["name"], fontsize=6.5,
                va="center", ha="left", color="white", fontweight="bold")

        # Score on RIGHT side of bar (black)
        score_x = bar_start + bar_width + 0.03
        ax.text(score_x, bar_center_y, f"{level['stress']:.2f}", fontsize=6.5,
                va="center", ha="left", color="black", fontweight="medium")

    # Vertical stress indicator on far right
    ax.axvline(x=0.88, ymin=0.06, ymax=0.96, color=COLORS["red"], linewidth=2)
    ax.text(0.88, 0.02, "Stress →", fontsize=7, ha="center", color=COLORS["red"], fontweight="bold")

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Context-stress ladder", fontweight="bold", pad=12)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)


def plot_panel_C(ax, data):
    """Performance collapse heatmap"""
    ax.set_facecolor(COLORS["panel_bg"])

    df = pd.DataFrame(data["data"], index=data["models"], columns=data["splits"])

    # Create heatmap
    im = ax.imshow(df.values, cmap="RdYlGn", aspect="auto", vmin=0, vmax=0.5)

    # Add text annotations
    for i in range(len(data["models"])):
        for j in range(len(data["splits"])):
            val = df.values[i, j]
            color = "white" if val < 0.15 else COLORS["text"]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color=color, fontweight="bold")

    ax.set_xticks(range(len(data["splits"])))
    ax.set_xticklabels(data["splits"], rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(data["models"])))
    ax.set_yticklabels(data["models"], fontsize=8)

    # Highlight external holdout column with red dashed line
    ax.axvline(x=5 - 0.5, color=COLORS["red"], linewidth=2, linestyle="--")

    # Add collapse annotation using transform for positioning
    # Use figure fraction coordinates
    ax.annotate("-83% Collapse", xy=(0.5, -0.55), xycoords=("axes fraction", "axes fraction"),
                fontsize=11, fontweight="bold", color=COLORS["red"], ha="center",
                xytext=(0.5, -0.55), textcoords=("axes fraction", "axes fraction"),
                arrowprops=dict(arrowstyle="->", color=COLORS["red"], lw=2))

    ax.set_title("Performance collapse", fontweight="bold", pad=18)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02, aspect=20)
    cbar.set_label("Cosine", fontsize=7)
    cbar.ax.tick_params(labelsize=6)


def plot_panel_D(ax, data):
    """PTL transportability layer - triangular arrangement with LARGER boxes"""
    ax.set_facecolor(COLORS["panel_bg"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Triangular layout: 3 boxes arranged as pyramid
    # Top: VC Model (centered)
    # Bottom-left: PTL Layer
    # Bottom-right: Decision

    # LARGER boxes to prevent text overflow
    box_w, box_h = 0.32, 0.26

    # Top box - VC Model (centered at top)
    top_x = 0.5 - box_w/2
    top_y = 0.62

    # Bottom-left box - PTL Layer
    ptl_x = 0.06
    ptl_y = 0.12

    # Bottom-right box - Decision
    dec_x = 0.62
    dec_y = 0.12

    steps = [
        {"name": "VC Model", "desc": "Perturbation\nPrediction", "color": COLORS["blue"], "x": top_x, "y": top_y},
        {"name": "PTL Layer", "desc": "Transportability\nAssessment", "color": COLORS["orange"], "x": ptl_x, "y": ptl_y},
        {"name": "Decision", "desc": "Keep / Abstain", "color": COLORS["purple"], "x": dec_x, "y": dec_y},
    ]

    for step in steps:
        x, y = step["x"], step["y"]

        # Main box background
        rect = FancyBboxPatch((x, y), box_w, box_h,
                              boxstyle="round,pad=0.02",
                              facecolor=step["color"], edgecolor="none",
                              alpha=0.12)
        ax.add_patch(rect)

        rect_outline = FancyBboxPatch((x, y), box_w, box_h,
                                     boxstyle="round,pad=0.02",
                                     facecolor="white", edgecolor=step["color"],
                                     linewidth=2)
        ax.add_patch(rect_outline)

        # Title (larger font)
        ax.text(x + box_w/2, y + box_h*0.68, step["name"], fontsize=11,
                ha="center", va="center", fontweight="bold", color=step["color"])

        # Description
        ax.text(x + box_w/2, y + box_h*0.30, step["desc"], fontsize=9,
                ha="center", va="center", color=COLORS["text_secondary"])

    # Arrows: VC -> PTL (down-left)
    ax.annotate("", xy=(ptl_x + box_w*0.8, ptl_y + box_h),
                xytext=(top_x + box_w*0.3, top_y),
                arrowprops=dict(arrowstyle="->", color=COLORS["text"], lw=1.5,
                                connectionstyle="arc3,rad=0.2"))

    # Arrows: VC -> Decision (down-right)
    ax.annotate("", xy=(dec_x + box_w*0.2, dec_y + box_h),
                xytext=(top_x + box_w*0.7, top_y),
                arrowprops=dict(arrowstyle="->", color=COLORS["text"], lw=1.5,
                                connectionstyle="arc3,rad=-0.2"))

    # Output labels below decision box
    ax.text(dec_x + box_w/2, dec_y - 0.08, "Keep / Abstain", fontsize=8,
            ha="center", color=COLORS["text_secondary"])

    ax.set_title("PTL transportability layer", fontweight="bold", pad=12)


def plot_panel_E(ax, data):
    """Reliability gain bar chart - narrower with diagonal arrow, text moved up"""
    ax.set_facecolor(COLORS["panel_bg"])

    # Narrower bars
    x_pos = [0, 1]
    bar_width = 0.22  # narrower
    bars = ax.bar(x_pos, data["values"], width=bar_width,
                  color=[COLORS["failure"], COLORS["success"]],
                  edgecolor="none", linewidth=0)

    # Value labels above bars
    for bar, val in zip(bars, data["values"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Diagonal arrow from high to low (top-left to bottom-right)
    arrow_x_start = x_pos[0] + bar_width/2 + 0.04
    arrow_y_start = data["values"][0] + 0.05
    arrow_x_end = x_pos[1] + bar_width/2 - 0.04
    arrow_y_end = data["values"][1] + 0.05

    ax.annotate("", xy=(arrow_x_end, arrow_y_end), xytext=(arrow_x_start, arrow_y_start),
                arrowprops=dict(arrowstyle="->", color=COLORS["success"], lw=2.5))

    # Improvement percentage label - ABOVE the arrow
    mid_x = (arrow_x_start + arrow_x_end) / 2
    mid_y = (arrow_y_start + arrow_y_end) / 2
    ax.text(mid_x, mid_y + 0.08, f"-{data['improvement']*100:.0f}%", fontsize=12,
            ha="center", fontweight="bold", color=COLORS["success"])

    ax.set_xticks([p + bar_width/2 for p in x_pos])
    ax.set_xticklabels(["Naive\n(Conf.)", "PTL\n(Transport.)"], fontsize=8)
    ax.set_ylabel("False Transport.", fontsize=8)
    ax.set_ylim(0, 0.60)
    ax.set_title("Reliability gain", fontweight="bold", pad=12)
    ax.tick_params(labelsize=7)

    # Legend note - MOVED UP above the chart
    ax.text(0.5, 0.62, "(Lower is better)", fontsize=7,
            ha="center", style="italic", color=COLORS["text_secondary"])

    ax.spines["left"].set_position(("outward", 8))


def plot_panel_F(ax, data):
    """Failure atlas scatter plot with diagonal boundary

    High novelty (x→1) + Low support (y→0) = FAILURE (bottom-right, red zone)
    Low novelty (x→0) + High support (y→1) = SAFE (top-left, green zone)
    Legend ABOVE colorbar
    """
    ax.set_facecolor(COLORS["panel_bg"])

    # Draw diagonal boundary FIRST
    x = np.linspace(0, 1, 100)
    y = 1 - 0.8 * x  # Diagonal boundary: y = 1 - 0.8x

    # Fill zones with subtle colors
    ax.fill_between(x, 0, y, alpha=0.12, color=COLORS["failure"])  # Failure zone (below)
    ax.fill_between(x, y, 1, alpha=0.12, color=COLORS["success"])   # Safe zone (above)
    ax.plot(x, y, "--", color=COLORS["text"], linewidth=2, alpha=0.8)

    # Failure cases - bottom-right cluster (high novelty, low support)
    scatter_fail = ax.scatter(data["failure_novelty"], data["failure_support"],
                               c=data["failure_severity"], cmap="Reds",
                               s=45, alpha=0.7, edgecolors="white", linewidth=0.3,
                               vmin=0, vmax=1)

    # Pass cases - top-left cluster (low novelty, high support)
    ax.scatter(data["pass_novelty"], data["pass_support"],
               c=COLORS["success"], s=35, alpha=0.7, marker="o",
               edgecolors="white", linewidth=0.3, label="Pass")

    # Zone labels - positioned clearly
    ax.text(0.12, 0.08, "FAILURE\nZone", fontsize=8, color=COLORS["failure"],
            fontweight="bold", ha="left", va="bottom")
    ax.text(0.82, 0.95, "SAFE\nZone", fontsize=8, color=COLORS["success"],
            fontweight="bold", ha="center", va="center")

    # Axis labels
    ax.set_xlabel("Novelty (unseen context)", fontsize=8)
    ax.set_ylabel("Support (# similar cases)", fontsize=8)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.tick_params(labelsize=7)

    # Legend
    ax.legend(loc="lower right", frameon=True, fancybox=True, fontsize=7)

    # Colorbar with label ABOVE (not below)
    cbar = plt.colorbar(scatter_fail, ax=ax, shrink=0.7, pad=0.02, aspect=20)
    cbar.set_label("Severity", fontsize=7, labelpad=12)  # labelpad=12 pushes label up
    cbar.ax.tick_params(labelsize=6)
    # Move tick labels slightly
    cbar.ax.set_xlabel("Severity", fontsize=7)

    ax.set_title("Failure atlas", fontweight="bold", pad=10)


# =============================================================================
# MAIN FIGURE ASSEMBLY
# =============================================================================

def create_figure1(output_dir=None, format="both"):
    """Create complete Figure 1 with all panels"""

    # Generate all data
    data = {
        "A": generate_panel_A_data(),
        "B": generate_panel_B_data(),
        "C": generate_panel_C_data(),
        "D": generate_panel_D_data(),
        "E": generate_panel_E_data(),
        "F": generate_panel_F_data(),
    }

    # Create figure with constrained layout
    fig = plt.figure(figsize=(12, 8.5), facecolor="white")

    # GridSpec layout: 2 rows x 3 columns for better panel visibility
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1.0, 1.0, 1.0],
                  height_ratios=[1.0, 1.0], wspace=0.30, hspace=0.38)

    # Create axes - top row: A, B, C; bottom row: D, E, F
    axes = []
    for row in range(2):
        for col in range(3):
            ax = fig.add_subplot(gs[row, col])
            axes.append(ax)

    # Plot each panel
    plot_functions = [
        plot_panel_A,
        plot_panel_B,
        plot_panel_C,
        plot_panel_D,
        plot_panel_E,
        plot_panel_F,
    ]

    panel_labels = ["A", "B", "C", "D", "E", "F"]

    for i, (ax, plot_fn, label) in enumerate(zip(axes, plot_functions, panel_labels)):
        plot_fn(ax, data[label])

        # Add panel label - adjust position for 2x3 grid
        row = i // 3
        col = i % 3
        if col == 0:
            x_offset = -0.12
        else:
            x_offset = -0.12
        ax.text(x_offset, 1.12, label, transform=ax.transAxes,
                fontsize=14, fontweight="bold", va="top",
                color=COLORS["blue"])

    # Main title
    fig.suptitle("PTL Framework for Transportable Perturbation-Response Predictions",
                 fontsize=14, fontweight="bold", y=1.02)

    # Save outputs
    if output_dir is None:
        output_dir = Path(".")

    if format in ["pdf", "both"]:
        pdf_path = output_dir / "figure1_v2.pdf"
        fig.savefig(pdf_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"[OK] Saved: {pdf_path}")

    if format in ["png", "both"]:
        png_path = output_dir / "figure1.png"
        fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"[OK] Saved: {png_path}")

    if format in ["svg", "both"]:
        svg_path = output_dir / "figure1_v2.svg"
        fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
        print(f"[OK] Saved: {svg_path}")

    plt.close(fig)

    return fig


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate Figure 1")
    parser.add_argument("--output-dir", "-o", type=str, default=".",
                        help="Output directory for figures")
    parser.add_argument("--format", "-f", choices=["pdf", "png", "svg", "both"],
                        default="both", help="Output format(s)")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    create_figure1(output_dir=output_dir, format=args.format)

    print("\n[INFO] Figure 1 generation complete!")
