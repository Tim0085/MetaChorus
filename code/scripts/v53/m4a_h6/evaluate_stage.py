from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from metachorus_v53.m4a_h6.stages import evaluate_r4_and_decide, evaluate_third_seed_and_gate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True, type=Path)
    parser.add_argument("--h5-result-root", required=True, type=Path)
    parser.add_argument("--h2-result-root", type=Path)
    parser.add_argument("--private-label-mount-root", required=True, type=Path)
    parser.add_argument("--stage", required=True, choices=("THIRD_SEED_R3_AUDIT", "R4"))
    parser.add_argument("--workers", default=1, type=int)
    args = parser.parse_args()
    if args.stage == "THIRD_SEED_R3_AUDIT":
        if args.h2_result_root is None:
            parser.error("--h2-result-root is required for the conditional post-reliability fidelity gate")
        decision = evaluate_third_seed_and_gate(
            result_root=args.result_root,
            h5_result_root=args.h5_result_root,
            h2_result_root=args.h2_result_root,
            private_mount_root=args.private_label_mount_root,
            workers=args.workers,
        )
    else:
        decision = evaluate_r4_and_decide(
            result_root=args.result_root,
            h5_result_root=args.h5_result_root,
            private_mount_root=args.private_label_mount_root,
            workers=args.workers,
        )
    print(decision["status"])
    print(decision["identity"])


if __name__ == "__main__":
    main()
