from __future__ import annotations

import argparse

from metachorus_v53.m7f_h4.evaluation import evaluate_locked_confirmation


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the once-opened locked M7F-H4 DBLP evaluation")
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--role-manifest", required=True)
    parser.add_argument("--seed-manifest", required=True)
    parser.add_argument("--private-search-mount-root", required=True)
    parser.add_argument("--label-source", required=True)
    parser.add_argument("--label-source-root", required=True)
    args = parser.parse_args()
    value = evaluate_locked_confirmation(
        result_root=args.result_root,
        role_manifest_path=args.role_manifest,
        seed_manifest_path=args.seed_manifest,
        private_search_mount_root=args.private_search_mount_root,
        label_source=args.label_source,
        label_source_root=args.label_source_root,
    )
    print({"status": value["status"], "identity": value["identity"]})


if __name__ == "__main__":
    main()
