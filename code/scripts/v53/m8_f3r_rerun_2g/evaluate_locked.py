from __future__ import annotations

import argparse

from metachorus_v53.m8_f3r_rerun_2g.evaluation import evaluate_locked


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--freebase-role-manifest", required=True)
    parser.add_argument("--dblp-role-manifest", required=True)
    parser.add_argument("--freebase-private-search-root", required=True)
    parser.add_argument("--dblp-private-search-root", required=True)
    parser.add_argument("--freebase-label-source", required=True)
    parser.add_argument("--freebase-label-root", required=True)
    parser.add_argument("--dblp-label-source", required=True)
    parser.add_argument("--dblp-label-root", required=True)
    args = parser.parse_args()
    value = evaluate_locked(
        result_root=args.result_root,
        freebase_role_manifest_path=args.freebase_role_manifest,
        dblp_role_manifest_path=args.dblp_role_manifest,
        private_search_mount_roots={
            "freebase": args.freebase_private_search_root,
            "dblp": args.dblp_private_search_root,
        },
        label_sources={
            "freebase": args.freebase_label_source,
            "dblp": args.dblp_label_source,
        },
        label_source_roots={
            "freebase": args.freebase_label_root,
            "dblp": args.dblp_label_root,
        },
    )
    print({"status": value["status"], "identity": value["identity"]})


if __name__ == "__main__":
    main()
