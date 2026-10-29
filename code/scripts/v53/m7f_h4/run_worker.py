from __future__ import annotations

import argparse

from metachorus_v53.m7f_h4.training import run_dynamic_worker


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one M7F-H4 dynamic queue worker")
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--seal", required=True)
    parser.add_argument("--logical-stage", choices=("CANARY", "CONFIRMATION"), required=True)
    parser.add_argument("--worker-index", type=int, required=True)
    parser.add_argument("--worker-count", type=int, default=4)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-tree", required=True)
    parser.add_argument("--launch-nonce", required=True)
    parser.add_argument("--cpu-test-double", action="store_true")
    args = parser.parse_args()
    value = run_dynamic_worker(
        repo_root=args.repo_root,
        cache_root=args.cache_root,
        result_root=args.result_root,
        plan_path=args.plan,
        seal_path=args.seal,
        logical_stage=args.logical_stage,
        worker_index=args.worker_index,
        worker_count=args.worker_count,
        source_commit=args.source_commit,
        source_tree=args.source_tree,
        launch_nonce=args.launch_nonce,
        enforce_target_gpu=not args.cpu_test_double,
    )
    print({"status": value["status"], "identity": value["identity"]})


if __name__ == "__main__":
    main()
