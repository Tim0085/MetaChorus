from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT)]

from metachorus_v53.m7f_h3.preflight import build_preflight


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--taskbook", required=True, type=Path)
    parser.add_argument("--parent-aminer-package", required=True, type=Path)
    parser.add_argument("--parent-h1-package", required=True, type=Path)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--seed-role-root", required=True, type=Path)
    parser.add_argument("--result-root", required=True, type=Path)
    args = parser.parse_args()
    if args.repo_root.resolve() != REPO_ROOT.resolve():
        parser.error("--repo-root must be this frozen source worktree")
    value = build_preflight(**vars(args))
    print({"status": value["status"], "identity": value["identity"]})


if __name__ == "__main__":
    main()
