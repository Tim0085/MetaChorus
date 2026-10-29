from __future__ import annotations

import argparse
from pathlib import Path

from metachorus_v53.m8_f3r_rerun_2g.constants import PARENT_PACKAGES
from metachorus_v53.m8_f3r_rerun_2g.preflight import build_preflight


def _parents(values: list[str]) -> dict[str, Path]:
    output = {}
    for value in values:
        key, separator, path = value.partition("=")
        if not separator or key in output:
            raise ValueError("--parent must be one unique ROLE=PATH entry")
        output[key] = Path(path)
    if set(output) != set(PARENT_PACKAGES):
        raise ValueError("--parent entries must cover the exact frozen parent roles")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--taskbook", required=True)
    parser.add_argument("--parent", action="append", required=True)
    parser.add_argument("--freebase-cache", required=True)
    parser.add_argument("--dblp-cache", required=True)
    parser.add_argument("--freebase-role-manifest", required=True)
    parser.add_argument("--freebase-mount-audit", required=True)
    parser.add_argument("--dblp-role-root", required=True)
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--estimated-remote-storage-bytes", type=int, default=0)
    args = parser.parse_args()
    value = build_preflight(
        repo_root=args.repo_root,
        taskbook=args.taskbook,
        parent_packages=_parents(args.parent),
        cache_roots={"freebase": args.freebase_cache, "dblp": args.dblp_cache},
        freebase_role_manifest=args.freebase_role_manifest,
        freebase_mount_audit=args.freebase_mount_audit,
        dblp_role_root=args.dblp_role_root,
        result_root=args.result_root,
        estimated_remote_storage_bytes=args.estimated_remote_storage_bytes,
    )
    print({"status": value["status"], "identity": value["identity"]})


if __name__ == "__main__":
    main()
