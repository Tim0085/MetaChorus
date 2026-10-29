from __future__ import annotations

import argparse
from pathlib import Path

from metachorus_v53.m8_f3r_rerun_2g.common import read_csv, read_json
from metachorus_v53.m8_f3r_rerun_2g.training import run_dynamic_worker


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--freebase-cache", required=True)
    parser.add_argument("--dblp-cache", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--seal", required=True)
    parser.add_argument("--schedule")
    parser.add_argument("--stage", choices=("CANARY", "EVALUATION"), required=True)
    parser.add_argument("--worker-index", choices=range(2), type=int, required=True)
    parser.add_argument("--worker-count", choices=(2,), type=int, default=2)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-tree", required=True)
    args = parser.parse_args()
    receipt = run_dynamic_worker(
        rows=read_csv(args.plan),
        seal=read_json(args.seal),
        stage=args.stage,
        worker_index=args.worker_index,
        worker_count=args.worker_count,
        repo_root=args.repo_root,
        cache_roots={"freebase": args.freebase_cache, "dblp": args.dblp_cache},
        result_root=args.result_root,
        source_commit=args.source_commit,
        source_tree=args.source_tree,
        schedule=read_csv(args.schedule) if args.schedule else (),
        enforce_target_gpu=True,
    )
    print({"status": receipt["status"], "identity": receipt["identity"]})


if __name__ == "__main__":
    main()
