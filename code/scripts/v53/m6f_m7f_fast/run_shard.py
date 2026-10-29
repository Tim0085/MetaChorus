from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from metachorus_v53.m6f_m7f_fast.stages import run_plan_shard


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--cache-parent", required=True, type=Path)
    parser.add_argument("--result-root", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--seal", required=True, type=Path)
    parser.add_argument("--stage", required=True, choices=("CANARY", "SEARCH", "PRELIMINARY"))
    parser.add_argument("--expected-seal-status", required=True)
    parser.add_argument("--shard-index", required=True, type=int, choices=(0, 1))
    parser.add_argument("--shard-count", type=int, default=2, choices=(2,))
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--local-test-no-gpu", action="store_true")
    args = parser.parse_args()
    print(
        run_plan_shard(
            repo_root=args.repo_root,
            cache_parent=args.cache_parent,
            result_root=args.result_root,
            plan_path=args.plan,
            seal_path=args.seal,
            stage=args.stage,
            expected_seal_status=args.expected_seal_status,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            retry_failed=args.retry_failed,
            enforce_target_gpu=not args.local_test_no_gpu,
        )
    )


if __name__ == "__main__":
    main()
