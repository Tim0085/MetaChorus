from __future__ import annotations

import argparse

from metachorus_v53.m8_f1.evaluation import evaluate_locked_final10


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the once-opened IMDb/AMiner locked M8-F1 evaluation")
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--role-manifest", required=True)
    parser.add_argument("--seed-manifest", required=True)
    parser.add_argument("--private-search-mount-root", required=True)
    parser.add_argument("--imdb-label-source", required=True)
    parser.add_argument("--imdb-label-source-root", required=True)
    parser.add_argument("--aminer-label-source", required=True)
    parser.add_argument("--aminer-label-source-root", required=True)
    args = parser.parse_args()
    value = evaluate_locked_final10(
        result_root=args.result_root,
        role_manifest_path=args.role_manifest,
        seed_manifest_path=args.seed_manifest,
        private_search_mount_root=args.private_search_mount_root,
        label_sources={"imdb": args.imdb_label_source, "aminer": args.aminer_label_source},
        label_source_roots={
            "imdb": args.imdb_label_source_root,
            "aminer": args.aminer_label_source_root,
        },
    )
    print({"status": value["status"], "identity": value["identity"]})


if __name__ == "__main__":
    main()
