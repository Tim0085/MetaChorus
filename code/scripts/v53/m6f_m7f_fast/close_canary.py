from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from metachorus_v53.m6f_m7f_fast.stages import authorize_r1_budget, close_canaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-tree", required=True)
    parser.add_argument("--cache-parent", required=True, type=Path)
    args = parser.parse_args()
    results, gate = close_canaries(args.result_root)
    _, seal = authorize_r1_budget(
        result_root=args.result_root,
        cache_parent=args.cache_parent,
        source_commit=args.source_commit,
        source_tree=args.source_tree,
    )
    print({"canary_results": results["identity"], "canary_gate": gate["identity"], "r1_plan": seal["identity"]})


if __name__ == "__main__":
    main()
