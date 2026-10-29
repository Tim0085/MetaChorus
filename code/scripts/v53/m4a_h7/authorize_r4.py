from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from metachorus_v53.m4a_h7.stages import authorize_r4


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True, type=Path)
    args = parser.parse_args()
    value = authorize_r4(result_root=args.result_root)
    print(value["seal"]["status"])
    print(value["seal"]["identity"])


if __name__ == "__main__":
    main()
