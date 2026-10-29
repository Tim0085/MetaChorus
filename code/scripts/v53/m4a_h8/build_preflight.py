from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from metachorus_v53.m4a_h8.preflight import build_preflight


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--taskbook", required=True, type=Path)
    parser.add_argument("--parent-package", required=True, type=Path)
    parser.add_argument("--h7-result-root", required=True, type=Path)
    parser.add_argument("--h5-result-root", required=True, type=Path)
    parser.add_argument("--result-root", required=True, type=Path)
    args = parser.parse_args()
    if args.repo_root.resolve(strict=True) != REPO_ROOT:
        parser.error("--repo-root must be the frozen H8 worktree containing this script")
    try:
        value = build_preflight(**vars(args))
    except RuntimeError as error:
        if str(error) == "H8_RESULT_ROOT_NOT_FRESH":
            print("H8_RESULT_ROOT_NOT_FRESH", file=sys.stderr)
            raise SystemExit(1) from None
        raise
    print(value["status"])
    print(value["identity"])


if __name__ == "__main__":
    main()
