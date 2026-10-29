from __future__ import annotations

import argparse
import json
from pathlib import Path

from metachorus_v53.m4a_h5.stages import (
    evaluate_r2,
    evaluate_r3,
    evaluate_r4,
    evaluate_third_seed,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Close, evaluate, and gate one H5 stage")
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--h1-result-root", required=True)
    parser.add_argument("--h2-result-root", required=True)
    parser.add_argument("--h4-result-root", required=True)
    parser.add_argument("--private-label-mount-root", required=True)
    parser.add_argument("--stage", required=True, choices=("R2", "R3", "THIRD_SEED_R3_AUDIT", "R4"))
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    common = {
        "result_root": Path(args.result_root),
        "h4_result_root": Path(args.h4_result_root),
        "private_mount_root": Path(args.private_label_mount_root),
        "workers": args.workers,
    }
    if args.stage == "R2":
        result = evaluate_r2(
            **common,
            h1_result_root=Path(args.h1_result_root),
            h2_result_root=Path(args.h2_result_root),
        )
    elif args.stage == "R3":
        result = evaluate_r3(
            **common,
            h1_result_root=Path(args.h1_result_root),
            h2_result_root=Path(args.h2_result_root),
        )
    elif args.stage == "THIRD_SEED_R3_AUDIT":
        result = evaluate_third_seed(**common)
    else:
        result = evaluate_r4(**common)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
