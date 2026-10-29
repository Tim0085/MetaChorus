from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from metachorus_v53.m4a_h7.preflight import (
    H7PreflightInputs,
    H7PreflightResultRootNotFresh,
    build_preflight,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--taskbook", required=True, type=Path)
    parser.add_argument("--parent-package", required=True, type=Path)
    parser.add_argument("--h6-result-root", required=True, type=Path)
    parser.add_argument("--h5-result-root", required=True, type=Path)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--private-label-mount-root", required=True, type=Path)
    parser.add_argument("--result-root", required=True, type=Path)
    args = parser.parse_args()
    try:
        payload = build_preflight(H7PreflightInputs(**vars(args)))
    except H7PreflightResultRootNotFresh:
        print("H7_RESULT_ROOT_NOT_FRESH", file=sys.stderr)
        raise SystemExit(1) from None
    print(payload["status"])
    print(payload["identity"])


if __name__ == "__main__":
    main()
