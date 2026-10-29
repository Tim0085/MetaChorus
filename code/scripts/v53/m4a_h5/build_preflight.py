from __future__ import annotations

import argparse
import json
from pathlib import Path

from metachorus_v53.m4a_h5.preflight import H5PreflightInputs, build_preflight


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the append-only M4A-H5 preflight")
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--workspace-root", required=True)
    parser.add_argument("--taskbook", required=True)
    parser.add_argument("--parent-package", required=True)
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--private-label-mount-root", required=True)
    parser.add_argument("--acm-label-source", required=True)
    parser.add_argument("--freebase-label-source", required=True)
    args = parser.parse_args()
    result = build_preflight(
        H5PreflightInputs(
            repo_root=Path(args.repo_root),
            workspace_root=Path(args.workspace_root),
            taskbook=Path(args.taskbook),
            parent_package=Path(args.parent_package),
            result_root=Path(args.result_root),
            private_label_mount_root=Path(args.private_label_mount_root),
            acm_label_source=Path(args.acm_label_source),
            freebase_label_source=Path(args.freebase_label_source),
        )
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
