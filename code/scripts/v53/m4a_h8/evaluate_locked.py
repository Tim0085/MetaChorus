from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from metachorus_v53.m4a_h8.locked_evaluation import run_locked_evaluation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True, type=Path)
    parser.add_argument("--h5-result-root", required=True, type=Path)
    parser.add_argument("--private-search-mount-root", required=True, type=Path)
    parser.add_argument("--acm-label-source", required=True, type=Path)
    parser.add_argument("--freebase-label-source", required=True, type=Path)
    parser.add_argument("--acm-label-source-root", required=True, type=Path)
    parser.add_argument("--freebase-label-source-root", required=True, type=Path)
    args = parser.parse_args()
    value = run_locked_evaluation(**vars(args))
    print(value["status"])
    print(value["identity"])


if __name__ == "__main__":
    main()
