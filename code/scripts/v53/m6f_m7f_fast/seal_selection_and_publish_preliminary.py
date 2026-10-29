from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from metachorus_v53.m6f_m7f_fast.common import read_csv
from metachorus_v53.m6f_m7f_fast.planning import load_cache_resolution
from metachorus_v53.m6f_m7f_fast.stages import publish_preliminary_plan, seal_final_search_selection


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--cache-parent", type=Path, required=True)
    parser.add_argument("--final-metrics", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-tree", required=True)
    args = parser.parse_args()
    caches = {name: load_cache_resolution(args.cache_parent / name) for name in ("dblp", "imdb", "aminer")}
    selected, selection_seal = seal_final_search_selection(
        result_root=args.result_root,
        cache_resolutions=caches,
        final_metric_rows=read_csv(args.final_metrics),
    )
    logical, physical, plan_seal = publish_preliminary_plan(
        result_root=args.result_root,
        cache_resolutions=caches,
        source_commit=args.source_commit,
        source_tree=args.source_tree,
    )
    print({
        "status": "PASS",
        "selected_dataset_count": len(selected),
        "selection_seal_identity": selection_seal["identity"],
        "logical_row_count": len(logical),
        "physical_run_count": len(physical),
        "preliminary_plan_identity": plan_seal["identity"],
    })


if __name__ == "__main__":
    main()
