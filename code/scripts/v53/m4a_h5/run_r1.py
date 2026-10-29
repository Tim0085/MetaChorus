from __future__ import annotations

import argparse
import json
from pathlib import Path

from metachorus_v53.m4a_h5.r1 import run_r1


def main() -> None:
    parser = argparse.ArgumentParser(description="Run H5 R1 evaluation-only pool construction")
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--h1-result-root", required=True)
    parser.add_argument("--h2-result-root", required=True)
    parser.add_argument("--private-label-mount-root", required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    result = run_r1(
        result_root=Path(args.result_root),
        h1_result_root=Path(args.h1_result_root),
        h2_result_root=Path(args.h2_result_root),
        private_mount_root=Path(args.private_label_mount_root),
        workers=args.workers,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
