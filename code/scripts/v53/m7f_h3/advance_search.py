from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT)]

from metachorus_v53.m7f_h3.common import read_csv, read_json
from metachorus_v53.m7f_h3.planning import load_dblp_cache
from metachorus_v53.m7f_h3.stages import advance_search


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True, type=Path)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--logical-stage", required=True, choices=("R1", "R2", "R3", "R4"))
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--seal", required=True, type=Path)
    parser.add_argument("--role-manifest", required=True, type=Path)
    parser.add_argument("--role-seal", required=True, type=Path)
    parser.add_argument("--private-mount-audit", required=True, type=Path)
    parser.add_argument("--private-search-mount-root", required=True, type=Path)
    args = parser.parse_args()
    value = advance_search(
        result_root=args.result_root,
        cache=load_dblp_cache(args.cache_root),
        rows=read_csv(args.plan),
        seal=read_json(args.seal),
        logical_stage=args.logical_stage,
        role_manifest_path=args.role_manifest,
        role_seal_path=args.role_seal,
        private_mount_audit_path=args.private_mount_audit,
        private_search_mount_root=args.private_search_mount_root,
    )
    print(value)


if __name__ == "__main__":
    main()
