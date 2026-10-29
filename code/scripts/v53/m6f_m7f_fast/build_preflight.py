from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from metachorus_v53.m6f_m7f_fast.preflight import build_preflight


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--taskbook", required=True, type=Path)
    parser.add_argument("--parent-package", required=True, type=Path)
    parser.add_argument("--cache-parent", required=True, type=Path)
    parser.add_argument("--seed-and-role-root", required=True, type=Path)
    parser.add_argument("--result-root", required=True, type=Path)
    args = parser.parse_args()
    if args.repo_root.resolve() != REPO_ROOT.resolve():
        parser.error("--repo-root must be this frozen source worktree")
    print(
        build_preflight(
            repo_root=args.repo_root,
            taskbook=args.taskbook,
            parent_package=args.parent_package,
            cache_roots={name: args.cache_parent / name for name in ("dblp", "imdb", "aminer")},
            seed_and_role_root=args.seed_and_role_root,
            result_root=args.result_root,
        )
    )


if __name__ == "__main__":
    main()
