from __future__ import annotations

import argparse
from pathlib import Path

from figlib.style import apply_style
from make_fig1_framework import make as make_fig1
from make_fig2_benchmark_audit import make as make_fig2
from make_fig3_rank_instability import make as make_fig3
from make_fig4_ptl_filtering import make as make_fig4
from make_fig5_failure_atlas import make as make_fig5
from make_supplementary import make as make_supplementary

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "results" / "figures_methods"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build data-driven PTL Methods manuscript figures.")
    parser.add_argument("--output-dir", default=str(OUTPUT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    apply_style()
    make_fig3(output_dir)
    make_fig4(output_dir)
    make_fig2(output_dir)
    make_fig5(output_dir)
    make_fig1(output_dir)
    make_supplementary(output_dir)
    print(f"Wrote PTL Methods figures to {output_dir}")


if __name__ == "__main__":
    main()

