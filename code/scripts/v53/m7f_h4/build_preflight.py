from __future__ import annotations

import argparse
from pathlib import Path

from metachorus_v53.m7f_h4.preflight import build_preflight


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the append-only M7F-H4 preflight")
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--taskbook", required=True)
    parser.add_argument("--parent-package", required=True)
    parser.add_argument("--imdb-source-package", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--seed-role-root", required=True)
    parser.add_argument("--result-root", required=True)
    args = parser.parse_args()
    result = build_preflight(
        repo_root=Path(args.repo_root),
        taskbook=Path(args.taskbook),
        parent_package=Path(args.parent_package),
        imdb_source_package=Path(args.imdb_source_package),
        cache_root=Path(args.cache_root),
        seed_role_root=Path(args.seed_role_root),
        result_root=Path(args.result_root),
    )
    print({"status": result["status"], "identity": result["identity"]})


if __name__ == "__main__":
    main()
