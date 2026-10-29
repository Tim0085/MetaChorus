from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT)]

from metachorus_v53.m7f_h3.stages import evaluate_preliminary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True, type=Path)
    parser.add_argument("--corrected-five-dataset-table", required=True, type=Path)
    parser.add_argument("--role-manifest", required=True, type=Path)
    parser.add_argument("--private-search-mount-root", required=True, type=Path)
    parser.add_argument("--label-source", required=True, type=Path)
    parser.add_argument("--label-source-root", required=True, type=Path)
    args = parser.parse_args()
    value = evaluate_preliminary(
        result_root=args.result_root,
        corrected_five_dataset_table_path=args.corrected_five_dataset_table,
        role_manifest_path=args.role_manifest,
        private_search_mount_root=args.private_search_mount_root,
        label_source=args.label_source,
        label_source_root=args.label_source_root,
    )
    print({"status": value["status"], "identity": value["identity"]})


if __name__ == "__main__":
    main()
