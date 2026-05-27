from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "figures" / "scripts"


def main() -> None:
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    runpy.run_path(str(SCRIPTS / "make_all_figures.py"), run_name="__main__")


if __name__ == "__main__":
    main()

