from __future__ import annotations

import argparse
from pathlib import Path

from metachorus_v53.m8_f1.preflight import build_preflight


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the append-only M8-F1 preflight")
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--taskbook", required=True)
    parser.add_argument("--parent-package", required=True)
    parser.add_argument("--h8-support-package", required=True)
    parser.add_argument("--h1-support-package", required=True)
    parser.add_argument("--h2-support-package", required=True)
    parser.add_argument("--cache-parent", required=True)
    parser.add_argument("--seed-role-root", required=True)
    parser.add_argument("--result-root", required=True)
    args = parser.parse_args()
    result = build_preflight(
        repo_root=Path(args.repo_root),
        taskbook=Path(args.taskbook),
        parent_package=Path(args.parent_package),
        h8_support_package=Path(args.h8_support_package),
        h1_support_package=Path(args.h1_support_package),
        h2_support_package=Path(args.h2_support_package),
        cache_parent=Path(args.cache_parent),
        seed_role_root=Path(args.seed_role_root),
        result_root=Path(args.result_root),
    )
    print({"status": result["status"], "identity": result["identity"]})


if __name__ == "__main__":
    main()
