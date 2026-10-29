from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from metachorus_v53.m6f_m7f_fast.stages import run_preliminary_evaluation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--role-manifest", type=Path, required=True)
    parser.add_argument("--private-search-mount-root", type=Path, required=True)
    for dataset in ("dblp", "imdb", "aminer"):
        parser.add_argument(f"--{dataset}-label-source", type=Path, required=True)
        parser.add_argument(f"--{dataset}-label-source-root", type=Path, required=True)
    args = parser.parse_args()
    audit = run_preliminary_evaluation(
        result_root=args.result_root,
        role_manifest_path=args.role_manifest,
        private_search_mount_root=args.private_search_mount_root,
        label_sources={name: getattr(args, f"{name}_label_source") for name in ("dblp", "imdb", "aminer")},
        label_source_roots={name: getattr(args, f"{name}_label_source_root") for name in ("dblp", "imdb", "aminer")},
    )
    print({"status": audit["status"], "identity": audit["identity"]})


if __name__ == "__main__":
    main()
