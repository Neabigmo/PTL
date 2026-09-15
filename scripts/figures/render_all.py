"""Reference-style render entry point for all six PTL figures."""

from __future__ import annotations

from pathlib import Path

from scripts.make_reliability_transportability_figures import run
from scripts.figures.data_contracts import validate_canonical_inputs
from scripts.figures.figure_data_manifest import write_data_map


def render_all(root: Path) -> dict:
    result = validate_canonical_inputs(root)
    if result["status"] != "PASS":
        raise RuntimeError("Figure data mismatch: " + ", ".join(result["failures"]))
    write_data_map(root)
    return run(root)


if __name__ == "__main__":
    render_all(Path(__file__).resolve().parents[2])
