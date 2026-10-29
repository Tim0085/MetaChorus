from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT)]

from metachorus_v53.m7f_h3.common import read_csv, read_json
from metachorus_v53.m7f_h3.stages import close_canaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True, type=Path)
    args = parser.parse_args()
    result = args.result_root.resolve(strict=True)
    value = close_canaries(
        result_root=result,
        rows=read_csv(result / "DBLP_H3_CANARY_PLAN.csv"),
        seal=read_json(result / "DBLP_H3_CANARY_PLAN_SEAL.json"),
    )
    print({"status": value["status"], "identity": value["identity"]})


if __name__ == "__main__":
    main()
