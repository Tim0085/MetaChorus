from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from metachorus_v53.m6f_m7f_fast.cache import build_sanitized_cache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=("dblp", "imdb", "aminer"))
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-parent", required=True, type=Path)
    args = parser.parse_args()
    print(build_sanitized_cache(dataset=args.dataset, source_root=args.source_root, output_parent=args.output_parent))


if __name__ == "__main__":
    main()
