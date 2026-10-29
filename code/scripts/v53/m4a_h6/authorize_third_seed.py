from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from metachorus_v53.m4a_h6.stages import authorize_third_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True, type=Path)
    parser.add_argument("--h5-result-root", required=True, type=Path)
    args = parser.parse_args()
    value = authorize_third_seed(result_root=args.result_root, h5_result_root=args.h5_result_root)
    print(value["seal"]["status"])
    print(value["seal"]["identity"])


if __name__ == "__main__":
    main()
