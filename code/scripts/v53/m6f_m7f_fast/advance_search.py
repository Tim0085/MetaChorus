from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from metachorus_v53.m6f_m7f_fast.planning import load_cache_resolution
from metachorus_v53.m6f_m7f_fast.stages import _dataset_execution_status, evaluate_and_plan_next_search_stage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--cache-parent", type=Path, required=True)
    parser.add_argument("--role-manifest", type=Path, required=True)
    parser.add_argument("--role-seal", type=Path, required=True)
    parser.add_argument("--private-mount-audit", type=Path, required=True)
    parser.add_argument("--private-search-mount-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--seal", type=Path, required=True)
    parser.add_argument("--expected-seal-status", required=True)
    parser.add_argument("--endpoint", type=int, choices=(300, 1000, 2000, 3000), required=True)
    args = parser.parse_args()
    _, active_datasets, _ = _dataset_execution_status(args.result_root.resolve())
    caches = {name: load_cache_resolution(args.cache_parent / name) for name in active_datasets}
    metrics, next_rows, next_seal = evaluate_and_plan_next_search_stage(
        result_root=args.result_root,
        cache_resolutions=caches,
        role_manifest_path=args.role_manifest,
        role_seal_path=args.role_seal,
        private_mount_audit_path=args.private_mount_audit,
        private_search_mount_root=args.private_search_mount_root,
        plan_path=args.plan,
        seal_path=args.seal,
        expected_seal_status=args.expected_seal_status,
        endpoint=args.endpoint,
    )
    print({
        "status": "PASS",
        "metrics_count": len(metrics),
        "next_plan_count": len(next_rows) if next_rows is not None else 0,
        "next_plan_identity": next_seal["identity"] if next_seal is not None else None,
    })


if __name__ == "__main__":
    main()
