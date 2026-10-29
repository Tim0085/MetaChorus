from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT)]

from metachorus_v53.m7f_h3.common import read_csv, read_json
from metachorus_v53.m7f_h3.training import ENGINE_STAGE, run_dynamic_worker


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--result-root", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--seal", required=True, type=Path)
    parser.add_argument("--logical-stage", required=True, choices=tuple(ENGINE_STAGE))
    parser.add_argument("--worker-index", required=True, type=int, choices=range(4))
    parser.add_argument("--worker-count", type=int, default=4, choices=(4,))
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-tree", required=True)
    parser.add_argument("--no-enforce-target-gpu", action="store_true")
    args = parser.parse_args()
    if args.repo_root.resolve() != REPO_ROOT.resolve():
        parser.error("--repo-root must be the installed frozen source")
    value = run_dynamic_worker(
        rows=read_csv(args.plan),
        seal=read_json(args.seal),
        logical_stage=args.logical_stage,
        worker_index=args.worker_index,
        worker_count=args.worker_count,
        repo_root=args.repo_root,
        cache_root=args.cache_root,
        result_root=args.result_root,
        source_commit=args.source_commit,
        source_tree=args.source_tree,
        enforce_target_gpu=not args.no_enforce_target_gpu,
    )
    print({"status": value["status"], "identity": value["identity"], "worker_index": args.worker_index})


if __name__ == "__main__":
    main()
