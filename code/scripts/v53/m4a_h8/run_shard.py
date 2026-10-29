from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from metachorus_v53.m4a_h8.stages import run_plan_shard


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--result-root", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--stage", required=True, choices=("CANARY", "S_EVAL"))
    parser.add_argument("--shard-index", required=True, type=int)
    parser.add_argument("--shard-count", default=2, type=int)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--skip-target-gpu-check", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.repo_root.resolve(strict=True) != REPO_ROOT:
        parser.error("--repo-root must be the frozen H8 worktree containing this script")
    value = run_plan_shard(
        repo_root=args.repo_root,
        cache_root=args.cache_root,
        result_root=args.result_root,
        plan_path=args.plan,
        stage=args.stage,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        retry_failed=args.retry_failed,
        enforce_target_gpu=not args.skip_target_gpu_check,
    )
    print(value)


if __name__ == "__main__":
    main()
