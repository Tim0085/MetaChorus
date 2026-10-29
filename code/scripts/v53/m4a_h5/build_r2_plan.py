from __future__ import annotations

import argparse
import json
from pathlib import Path

from metachorus_v53.m4a_h5.planning import build_r2_plan


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the H5 R2 no-pruning plan")
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--h1-result-root", required=True)
    parser.add_argument("--h4-result-root", required=True)
    args = parser.parse_args()
    rows, reuse, budget = build_r2_plan(
        result_root=Path(args.result_root),
        h1_result_root=Path(args.h1_result_root),
        h4_result_root=Path(args.h4_result_root),
    )
    print(
        json.dumps(
            {
                "r2_identity_count": len(rows),
                "exact_h4_reuse_count": len(reuse),
                "budget": budget,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
