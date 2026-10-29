from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from metachorus_v53.m6f_m7f_fast.labels import build_seed_and_label_role_seals


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True, type=Path)
    parser.add_argument("--private-mount-root", required=True, type=Path)
    parser.add_argument("--dblp-label-source", required=True, type=Path)
    parser.add_argument("--imdb-label-source", required=True, type=Path)
    parser.add_argument("--aminer-label-source", required=True, type=Path)
    args = parser.parse_args()
    print(
        build_seed_and_label_role_seals(
            result_root=args.result_root,
            private_mount_root=args.private_mount_root,
            label_sources={
                "dblp": args.dblp_label_source,
                "imdb": args.imdb_label_source,
                "aminer": args.aminer_label_source,
            },
        )
    )


if __name__ == "__main__":
    main()
