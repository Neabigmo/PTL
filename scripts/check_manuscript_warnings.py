"""Fail fast on LaTeX citation, reference, and rerun warnings."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


FORBIDDEN_PATTERNS = (
    re.compile(r"undefined citation", re.IGNORECASE),
    re.compile(r"there were undefined citations", re.IGNORECASE),
    re.compile(r"reference .* undefined", re.IGNORECASE),
    re.compile(r"label\(s\) may have changed", re.IGNORECASE),
)


def find_warnings(log_path: Path) -> list[str]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    return [line.strip() for line in text.splitlines() if any(pattern.search(line) for pattern in FORBIDDEN_PATTERNS)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    args = parser.parse_args()
    if not args.log.is_file():
        print(f"missing LaTeX log: {args.log}")
        return 2
    warnings = find_warnings(args.log)
    if warnings:
        print("forbidden manuscript warnings:")
        print("\n".join(warnings))
        return 1
    print(f"manuscript warning check passed: {args.log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
