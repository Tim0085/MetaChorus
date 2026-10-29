from __future__ import annotations

import argparse
import json
from pathlib import Path

from metachorus_v53.m4a_h5.stages import run_plan_shard


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one deterministic H5 GPU shard")
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--h1-result-root", required=True)
    parser.add_argument("--h4-result-root", required=True)
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument(
        "--stage", required=True, choices=("R2", "R3", "THIRD_SEED_R3_AUDIT", "R4")
    )
    parser.add_argument("--shard-index", required=True, type=int)
    parser.add_argument("--shard-count", required=True, type=int)
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Use the single append-only engineering retry after a preserved pre-metric failure",
    )
    args = parser.parse_args()
    result = run_plan_shard(
        plan_path=Path(args.plan),
        stage=args.stage,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        repo_root=Path(args.repo_root),
        cache_root=Path(args.cache_root),
        h1_result_root=Path(args.h1_result_root),
        h4_result_root=Path(args.h4_result_root),
        h5_result_root=Path(args.result_root),
        enforce_target_gpu=True,
        retry_failed=args.retry_failed,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
