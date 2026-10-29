from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from metachorus_v53.m6f_m7f_fast.stages import close_preliminary_training_registry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    args = parser.parse_args()
    rows, seal = close_preliminary_training_registry(args.result_root)
    print({"status": seal["status"], "record_count": len(rows), "identity": seal["identity"]})


if __name__ == "__main__":
    main()
