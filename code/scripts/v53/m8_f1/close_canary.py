from __future__ import annotations

import argparse
from pathlib import Path

from metachorus_v53.m8_f1.common import read_csv, read_json
from metachorus_v53.m8_f1.evaluation import close_canaries


def main() -> None:
    parser = argparse.ArgumentParser(description="Close the four M8-F1 engineering canaries")
    parser.add_argument("--result-root", required=True)
    args = parser.parse_args()
    result = Path(args.result_root).resolve(strict=True)
    value = close_canaries(
        result_root=result,
        rows=read_csv(result / "M8_F1_CANARY_PLAN.csv"),
        seal=read_json(result / "M8_F1_CANARY_PLAN_SEAL.json"),
    )
    print({"status": value["status"], "identity": value["identity"]})


if __name__ == "__main__":
    main()
