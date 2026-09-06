from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from .palettes import WHITE


def save_figure(
    fig: plt.Figure,
    output_dir: Path,
    stem: str,
    formats: tuple[str, ...] = ("png", "pdf", "svg", "tiff"),
    dpi: int = 600,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for ext in formats:
        path = output_dir / f"{stem}.{ext}"
        kwargs = {"bbox_inches": "tight", "facecolor": WHITE}
        if ext in {"png", "tif", "tiff"}:
            kwargs["dpi"] = dpi
        fig.savefig(path, **kwargs)
        paths.append(path)
    plt.close(fig)
    return paths


def save_panel(fig: plt.Figure, output_dir: Path, stem: str) -> list[Path]:
    return save_figure(fig, output_dir / "panels", stem, formats=("pdf", "svg", "png"), dpi=600)

