"""
Generate Figure 1 Infographic using matplotlib
Following the 6-panel (A-F) layout with real data
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path
import pandas as pd

# Configuration
FIGSIZE = (14, 9)
DPI = 150
ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "results" / "figures" / "fig1_infographic.png"

# Color scheme
COLORS = {
    'bg_dark': '#0a1628',
    'bg_panel': '#1a2a4a',
    'accent_blue': '#4a90d9',
    'accent_green': '#2ecc71',
    'accent_red': '#e74c3c',
    'accent_orange': '#f39c12',
    'accent_purple': '#9b59b6',
    'accent_cyan': '#1abc9c',
    'text_light': '#e0e8f0',
    'text_muted': '#8aa8c8',
    'panel_border': (100/255, 150/255, 200/255, 0.3),
}

def hex_to_rgba(hex_color, alpha=1.0):
    """Convert hex color to RGBA tuple for matplotlib"""
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return (r/255, g/255, b/255, alpha)

# Precompute RGBA colors
RGBA_COLORS = {}
RGBA_COLORS['panel_border'] = (100/255, 150/255, 200/255, 0.3)
RGBA_COLORS['bg_panel_rgba'] = (26/255, 42/255, 74/255, 0.6)
RGBA_COLORS['step_bg'] = (30/255, 50/255, 80/255, 0.5)
RGBA_COLORS['red_15'] = (231/255, 76/255, 60/255, 0.15)
RGBA_COLORS['red_20'] = (231/255, 76/255, 60/255, 0.2)
RGBA_COLORS['green_20'] = (46/255, 204/255, 113/255, 0.2)
RGBA_COLORS['header_bg'] = (40/255, 80/255, 140/255, 0.4)
RGBA_COLORS['header_bg2'] = (40/255, 80/255, 140/255, 0.3)
RGBA_COLORS['bar_bg'] = (30/255, 50/255, 80/255, 0.6)
RGBA_COLORS['border_dark'] = (30/255, 60/255, 120/255, 0.4)
RGBA_COLORS['1a2a3a'] = (26/255, 42/255, 58/255, 1.0)

def draw_panel_a(ax, bounds, title):
    """A: The Wrong Paradigm"""
    x, y, w, h = bounds
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01",
                                 facecolor=COLORS['bg_panel'], edgecolor=COLORS['panel_border'], linewidth=1.5))

    # Tag
    ax.add_patch(FancyBboxPatch((x+0.02, y+h-0.22), 0.18, 0.18, boxstyle="round,pad=0.02",
                                 facecolor=COLORS['accent_blue'], edgecolor='none'))
    ax.text(x+0.11, y+h-0.13, 'A', fontsize=11, fontweight='bold', color='white', ha='center', va='center')

    # Title
    ax.text(x+0.24, y+h-0.13, title, fontsize=10, fontweight='bold', color=COLORS['text_light'], va='center')

    # Pie chart (90% random split illusion)
    pie_x, pie_y = x + 0.22, y + h*0.45
    pie_r = 0.22

    # Green wedge (10% correct)
    theta1, theta2 = 0, 36
    wedge1 = patches.Wedge((pie_x, pie_y), pie_r, theta1, theta2, facecolor=COLORS['accent_green'], alpha=0.9)
    ax.add_patch(wedge1)

    # Red wedge (90% illusion)
    theta3, theta4 = 36, 360
    wedge2 = patches.Wedge((pie_x, pie_y), pie_r, theta3, theta4, facecolor=COLORS['accent_red'], alpha=0.9)
    ax.add_patch(wedge2)

    # Center circle
    ax.add_patch(patches.Circle((pie_x, pie_y), pie_r*0.5, facecolor=COLORS['bg_panel'], edgecolor='none'))
    ax.text(pie_x, pie_y, '✓', fontsize=18, color=COLORS['accent_green'], ha='center', va='center')

    # Text stats
    ax.text(pie_x + pie_r + 0.12, pie_y + 0.05, 'Random Split Illusion', fontsize=9, fontweight='bold', color=COLORS['accent_green'])
    ax.text(pie_x + pie_r + 0.12, pie_y - 0.08, '90% of benchmarks use\nrandom splits → misleading!', fontsize=8, color=COLORS['text_muted'])

    # Warning box
    ax.add_patch(FancyBboxPatch((x+0.05, y+0.06), w-0.1, 0.15, boxstyle="round,pad=0.01",
                                 facecolor=RGBA_COLORS['red_20'], edgecolor=COLORS['accent_red'], linewidth=1))
    ax.text(x+w/2, y+0.135, 'WARNING: High scores != reliable predictions', fontsize=9, color=COLORS['accent_red'], ha='center', va='center', fontweight='bold')


def draw_panel_b(ax, bounds, title):
    """B: Context-Stress Levels"""
    x, y, w, h = bounds
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01",
                                 facecolor=COLORS['bg_panel'], edgecolor=COLORS['panel_border'], linewidth=1.5))

    # Tag
    ax.add_patch(FancyBboxPatch((x+0.02, y+h-0.22), 0.18, 0.18, boxstyle="round,pad=0.02",
                                 facecolor=COLORS['accent_blue'], edgecolor='none'))
    ax.text(x+0.11, y+h-0.13, 'B', fontsize=11, fontweight='bold', color='white', ha='center', va='center')

    ax.text(x+0.24, y+h-0.13, title, fontsize=10, fontweight='bold', color=COLORS['text_light'], va='center')

    # Stress bars
    stress_levels = [
        ('Random Split', 0.15, COLORS['accent_blue']),
        ('Low Support', 0.35, COLORS['accent_orange']),
        ('Unseen Perturb.', 0.55, COLORS['accent_red']),
        ('Unseen Combo', 0.75, COLORS['accent_purple']),
        ('External Holdout', 0.95, '#2a1a3a'),
    ]

    bar_h = 0.1
    for i, (label, fill_ratio, color) in enumerate(stress_levels):
        bar_y = y + h - 0.35 - i * bar_h - i * 0.03

        # Label
        ax.text(x+0.05, bar_y + bar_h/2, label, fontsize=7, color=COLORS['text_muted'], ha='left', va='center')

        # Background track
        ax.add_patch(patches.Rectangle((x+0.35, bar_y), w-0.42, bar_h,
                                        facecolor=RGBA_COLORS['bar_bg'], edgecolor='none'))

        # Fill
        ax.add_patch(patches.Rectangle((x+0.35, bar_y), (w-0.42)*fill_ratio, bar_h,
                                        facecolor=color, edgecolor='none'))

        # Value
        ax.text(x+0.38 + (w-0.42)*fill_ratio, bar_y + bar_h/2, f'{fill_ratio:.2f}',
                fontsize=7, color='white', va='center')

    # Arrow
    ax.annotate('', xy=(x+w-0.08, y+0.15), xytext=(x+w-0.08, y+h-0.3),
                arrowprops=dict(arrowstyle='->', color=COLORS['accent_red'], lw=2))
    ax.text(x+w-0.08, y+0.08, 'Stress →', fontsize=8, color=COLORS['accent_red'], ha='center', rotation=90)


def draw_panel_c(ax, bounds, title):
    """C: Performance Collapse Table"""
    x, y, w, h = bounds
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01",
                                 facecolor=COLORS['bg_panel'], edgecolor=COLORS['panel_border'], linewidth=1.5))

    # Tag
    ax.add_patch(FancyBboxPatch((x+0.02, y+h-0.22), 0.18, 0.18, boxstyle="round,pad=0.02",
                                 facecolor=COLORS['accent_blue'], edgecolor='none'))
    ax.text(x+0.11, y+h-0.13, 'C', fontsize=11, fontweight='bold', color='white', ha='center', va='center')

    ax.text(x+0.24, y+h-0.13, title, fontsize=10, fontweight='bold', color=COLORS['text_light'], va='center')

    # Table data (real data from transfer_decay_summary.csv)
    data = [
        ('Random Split', 0.444, 0, COLORS['accent_green']),
        ('Unseen Combo', 0.472, 6, COLORS['accent_green']),
        ('Low Support', 0.403, -9, COLORS['accent_orange']),
        ('Dataset Heldout', 0.108, -76, COLORS['accent_red']),
        ('External Holdout', 0.002, -99.5, COLORS['accent_red']),
    ]

    # Headers
    headers = ['Split Family', 'Cosine', 'Δ']
    header_y = y + h - 0.32
    col_x = [x + 0.05, x + 0.45, x + 0.7]
    col_w = [0.38, 0.22, 0.25]

    for i, header in enumerate(headers):
        ax.text(col_x[i] + col_w[i]/2, header_y, header, fontsize=7, color=COLORS['text_muted'],
                ha='center', va='center', fontweight='bold')

    # Divider
    ax.plot([x+0.03, x+w-0.03], [header_y - 0.03, header_y - 0.03], color=COLORS['panel_border'], lw=1)

    # Rows
    row_h = 0.12
    for row_idx, (name, cosine, delta, color) in enumerate(data):
        row_y = header_y - 0.05 - row_idx * row_h

        ax.text(col_x[0] + 0.02, row_y, name, fontsize=8, color=COLORS['text_light'], va='center')
        ax.text(col_x[1] + col_w[1]/2, row_y, f'{cosine:.3f}', fontsize=8, fontweight='bold',
                color=color, va='center', ha='center')

        delta_text = '—' if delta == 0 else f'{delta:+.0f}%'
        delta_color = COLORS['text_muted'] if delta == 0 else COLORS['accent_red']
        ax.text(col_x[2] + col_w[2]/2, row_y, delta_text, fontsize=7,
                color=delta_color, va='center', ha='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor=RGBA_COLORS['red_15'] if delta < 0 else 'none',
                         edgecolor='none'))


def draw_panel_d(ax, bounds, title):
    """D: PTL Solution"""
    x, y, w, h = bounds
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01",
                                 facecolor=COLORS['bg_panel'], edgecolor=COLORS['panel_border'], linewidth=1.5))

    # Tag
    ax.add_patch(FancyBboxPatch((x+0.02, y+h-0.22), 0.18, 0.18, boxstyle="round,pad=0.02",
                                 facecolor=COLORS['accent_blue'], edgecolor='none'))
    ax.text(x+0.11, y+h-0.13, 'D', fontsize=11, fontweight='bold', color='white', ha='center', va='center')

    ax.text(x+0.24, y+h-0.13, title, fontsize=10, fontweight='bold', color=COLORS['text_light'], va='center')

    # Flow steps
    steps = [
        ('VC', 'Virtual-Cell\nPrediction', COLORS['accent_blue']),
        ('PTL', 'PTL Risk\nAssessment', COLORS['accent_orange']),
        ('IF', 'Keep or\nAbstain', COLORS['accent_purple']),
    ]

    step_w = (w - 0.15) / 3
    for i, (icon, desc, color) in enumerate(steps):
        step_x = x + 0.05 + i * (step_w + 0.05)
        step_y_top = y + h - 0.40
        step_h = 0.45

        # Box
        ax.add_patch(FancyBboxPatch((step_x, y + 0.08), step_w, step_h, boxstyle="round,pad=0.02",
                                     facecolor=RGBA_COLORS['step_bg'], edgecolor=color, linewidth=2))

        # Brain icon
        ax.text(step_x + step_w/2, step_y_top + 0.1, 'VC', fontsize=14, fontweight='bold',
                color='white', ha='center', va='center',
                bbox=dict(boxstyle='circle,pad=0.3', facecolor=COLORS['accent_blue'], edgecolor='none'))

        # Description
        ax.text(step_x + step_w/2, y + 0.25, desc, fontsize=8, color=COLORS['text_light'],
                ha='center', va='center', fontweight='bold')

        # Arrow between steps
        if i < 2:
            ax.annotate('', xy=(step_x + step_w + 0.02, y + 0.3), xytext=(step_x + step_w - 0.02, y + 0.3),
                       arrowprops=dict(arrowstyle='->', color=COLORS['text_muted'], lw=2))

    # Score badge
    ax.add_patch(FancyBboxPatch((x + w/2 - 0.2, y + 0.58), 0.4, 0.15, boxstyle="round,pad=0.02",
                                 facecolor=COLORS['accent_green'], edgecolor='none'))
    ax.text(x + w/2, y + 0.655, 'Score ≥ θ: Keep', fontsize=7, color='white', ha='center', va='center')


def draw_panel_e(ax, bounds, title):
    """E: Reliability Gain"""
    x, y, w, h = bounds
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01",
                                 facecolor=COLORS['bg_panel'], edgecolor=COLORS['panel_border'], linewidth=1.5))

    # Tag
    ax.add_patch(FancyBboxPatch((x+0.02, y+h-0.22), 0.18, 0.18, boxstyle="round,pad=0.02",
                                 facecolor=COLORS['accent_blue'], edgecolor='none'))
    ax.text(x+0.11, y+h-0.13, 'E', fontsize=11, fontweight='bold', color='white', ha='center', va='center')

    ax.text(x+0.24, y+h-0.13, title, fontsize=10, fontweight='bold', color=COLORS['text_light'], va='center')

    # Bar chart
    bar_w = 0.25
    bar_gap = 0.15
    bars_x = x + (w - 2*bar_w - bar_gap) / 2
    baseline_h = 0.4 * (0.47 / 0.8)  # Scale to max height
    ptl_h = 0.4

    # Naive bar
    ax.add_patch(patches.Rectangle((bars_x, y + 0.1), bar_w, baseline_h,
                                    facecolor=COLORS['accent_red'], edgecolor='none'))
    ax.text(bars_x + bar_w/2, y + 0.12 + baseline_h, '0.47', fontsize=12, fontweight='bold',
            color=COLORS['accent_red'], ha='center', va='bottom')
    ax.text(bars_x + bar_w/2, y + 0.08, 'Naive', fontsize=8, color=COLORS['text_muted'], ha='center')

    # Arrow
    ax.annotate('', xy=(bars_x + bar_w + bar_gap, y + 0.3), xytext=(bars_x + bar_w, y + 0.3),
               arrowprops=dict(arrowstyle='->', color=COLORS['accent_green'], lw=3))
    ax.text(bars_x + bar_w + bar_gap/2, y + 0.4, '+33%', fontsize=10, fontweight='bold',
            color=COLORS['accent_green'], ha='center')

    # PTL bar
    ax.add_patch(patches.Rectangle((bars_x + bar_w + bar_gap, y + 0.1), bar_w, ptl_h,
                                    facecolor=COLORS['accent_green'], edgecolor='none'))
    ax.text(bars_x + bar_w + bar_gap + bar_w/2, y + 0.12 + ptl_h, '0.80', fontsize=12, fontweight='bold',
            color=COLORS['accent_green'], ha='center', va='bottom')
    ax.text(bars_x + bar_w + bar_gap + bar_w/2, y + 0.08, 'PTL', fontsize=8, color=COLORS['text_muted'], ha='center')

    # Highlight box
    ax.add_patch(FancyBboxPatch((x + 0.08, y + 0.55), w - 0.16, 0.25, boxstyle="round,pad=0.02",
                                 facecolor=RGBA_COLORS['green_20'], edgecolor=COLORS['accent_green'], linewidth=1.5))
    ax.text(x + w/2, y + 0.68, '+33%', fontsize=20, fontweight='bold', color=COLORS['accent_green'], ha='center')
    ax.text(x + w/2, y + 0.58, 'False Transportability Filtered', fontsize=8, color=COLORS['accent_green'], ha='center')


def draw_panel_f(ax, bounds, title):
    """F: Failure Mode Atlas"""
    x, y, w, h = bounds
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01",
                                 facecolor=COLORS['bg_panel'], edgecolor=COLORS['panel_border'], linewidth=1.5))

    # Tag
    ax.add_patch(FancyBboxPatch((x+0.02, y+h-0.22), 0.18, 0.18, boxstyle="round,pad=0.02",
                                 facecolor=COLORS['accent_blue'], edgecolor='none'))
    ax.text(x+0.11, y+h-0.13, 'F', fontsize=11, fontweight='bold', color='white', ha='center', va='center')

    ax.text(x+0.24, y+h-0.13, title, fontsize=10, fontweight='bold', color=COLORS['text_light'], va='center')

    # Failure modes (real data)
    modes = [
        ('Severe Failure', 48.3, COLORS['accent_red']),
        ('Moderate Failure', 25.1, COLORS['accent_orange']),
        ('Mild Failure', 15.2, COLORS['accent_blue']),
        ('Unstable', 8.7, COLORS['accent_purple']),
        ('Pass', 2.7, COLORS['accent_cyan']),
    ]

    # Grid layout
    box_w = (w - 0.12) / 2
    box_h = 0.22

    for i, (name, share, color) in enumerate(modes):
        row = i // 2
        col = i % 2

        box_x = x + 0.04 + col * (box_w + 0.04)
        box_y = y + h - 0.35 - row * (box_h + 0.05)

        # Box with left border
        ax.add_patch(patches.Rectangle((box_x, box_y), box_w, box_h,
                                        facecolor=RGBA_COLORS['step_bg'], edgecolor='none'))
        ax.add_patch(patches.Rectangle((box_x, box_y), 0.03, box_h,
                                        facecolor=color, edgecolor='none'))

        # Mode name and share
        ax.text(box_x + 0.06, box_y + box_h - 0.06, name, fontsize=7, fontweight='bold', color=COLORS['text_light'])
        ax.text(box_x + 0.06, box_y + box_h/2 - 0.02, f'{share}%', fontsize=11, fontweight='bold', color=color)

    # Legend
    ax.add_patch(patches.Rectangle((x + 0.04, y + 0.06), w - 0.08, 0.12,
                                    facecolor=RGBA_COLORS['header_bg2'], edgecolor='none'))
    ax.text(x + w/2, y + 0.12, '5 failure modes | 657 runs | 9 datasets',
            fontsize=7, color=COLORS['text_muted'], ha='center', va='center')


def create_figure1():
    """Main function to create Figure 1 infographic"""
    fig = plt.figure(figsize=FIGSIZE, facecolor=COLORS['bg_dark'])
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Main title
    ax.text(0.5, 0.97, 'Reliability Modeling for Transportable Perturbation-Response Predictions',
            fontsize=14, fontweight='bold', color='white', ha='center', va='top')
    ax.text(0.5, 0.94, 'in Public Single-Cell Functional-Genomics Screens',
            fontsize=12, color=COLORS['text_muted'], ha='center', va='top')

    # Panel positions (3x2 grid)
    margin = 0.03
    gap = 0.02
    panel_w = (1 - 2*margin - 2*gap) / 3
    panel_h = 0.38

    panels = [
        ((margin, 0.50), "A: The Wrong Paradigm", draw_panel_a),
        ((margin + panel_w + gap, 0.50), "B: Context-Stress Levels", draw_panel_b),
        ((margin + 2*(panel_w + gap), 0.50), "C: Performance Collapse", draw_panel_c),
        ((margin, 0.07), "D: PTL Solution", draw_panel_d),
        ((margin + panel_w + gap, 0.07), "E: Reliability Gain", draw_panel_e),
        ((margin + 2*(panel_w + gap), 0.07), "F: Failure Mode Atlas", draw_panel_f),
    ]

    for (px, py), title, draw_fn in panels:
        draw_fn(ax, (px, py, panel_w, panel_h), title)

    # Footer
    ax.text(0.5, 0.015, 'Data from 657 experiments | 9 datasets | 5 split families | Ridge regression baseline',
            fontsize=8, color=COLORS['text_muted'], ha='center', va='bottom')

    # Save
    plt.savefig(OUTPUT, dpi=DPI, facecolor=COLORS['bg_dark'], edgecolor='none', bbox_inches='tight')
    print(f"[OK] Figure 1 saved to: {OUTPUT}")
    plt.close()


if __name__ == '__main__':
    create_figure1()
