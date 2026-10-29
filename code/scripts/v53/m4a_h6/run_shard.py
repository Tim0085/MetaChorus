from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from metachorus_v53.m4a_h6.stages import run_plan_shard


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--h5-result-root", required=True, type=Path)
    parser.add_argument("--h6-result-root", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--stage", required=True, choices=("CANARY", "THIRD_SEED_R3_AUDIT", "R4"))
    parser.add_argument("--shard-index", required=True, type=int)
    parser.add_argument("--shard-count", required=True, type=int)
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    summary = run_plan_shard(
        plan_path=args.plan,
        stage=args.stage,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        repo_root=args.repo_root,
        cache_root=args.cache_root,
        h5_result_root=args.h5_result_root,
        h6_result_root=args.h6_result_root,
        retry_failed=args.retry_failed,
        enforce_target_gpu=True,
    )
    print(summary)


if __name__ == "__main__":
    main()
