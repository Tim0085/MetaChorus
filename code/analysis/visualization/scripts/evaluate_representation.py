from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from m9_common import atomic_write_json, canonical_hash, load_json, sha256_file


def _load_roles(entry: Mapping[str, Any], node_count: int) -> dict[str, Any]:
    import numpy as np

    label_path = Path(entry["label_source"]).resolve()
    mount_path = Path(entry["private_search_mount"]).resolve()
    if entry.get("evaluation_label_loader") == "h8":
        source_root = Path(entry["source_root"]).resolve()
        sys.path.insert(0, str(source_root))
        sys.path.insert(0, str(source_root / "src"))
        try:
            from metachorus_v53.m4a_h5.labels import load_acm_labels, load_freebase_labels

            labels = (
                load_acm_labels(label_path)
                if str(entry["final_set_id"]).startswith("set_")
                else load_freebase_labels(label_path)
            )
        finally:
            sys.path.pop(0)
            sys.path.pop(0)
    else:
        labels = np.load(label_path, allow_pickle=False)
    mount = np.load(mount_path, allow_pickle=False)
    if labels.ndim != 1 or len(labels) != node_count:
        raise ValueError("label source shape mismatch")
    pool_indices = np.asarray(mount["global_search_pool_indices"], dtype=np.int64)
    pool_labels = np.asarray(mount["global_search_pool_labels"], dtype=np.int64)
    if not np.array_equal(labels[pool_indices], pool_labels):
        raise ValueError("private search pool labels do not match frozen label source")
    mask = np.ones(node_count, dtype=bool)
    mask[pool_indices] = False
    locked_indices = np.flatnonzero(mask)
    if len(pool_indices) + len(locked_indices) != node_count:
        raise ValueError("fit/report roles do not cover all target nodes")
    return {
        "labels": labels,
        "pool_indices": pool_indices,
        "pool_labels": pool_labels,
        "locked_indices": locked_indices,
        "locked_labels": labels[locked_indices],
        "label_sha256": sha256_file(label_path),
        "mount_sha256": sha256_file(mount_path),
        "role_manifest_sha256": sha256_file(entry["evaluation_role_manifest"]),
    }


def _kmeans_seeds(entry: Mapping[str, Any]) -> tuple[int, ...]:
    source_root = Path(entry["source_root"]).resolve()
    sys.path.insert(0, str(source_root))
    sys.path.insert(0, str(source_root / "src"))
    try:
        if entry.get("evaluation_seed_provider") == "h8":
            from metachorus_v53.m4a_h8.locked_evaluation import derive_kmeans_seeds

            seeds = tuple(int(value) for value in derive_kmeans_seeds())
        else:
            from metachorus_v53.m6f_m7f_fast.constants import PRELIM_KMEANS_SEEDS

            seeds = tuple(int(value) for value in PRELIM_KMEANS_SEEDS)
    finally:
        sys.path.pop(0)
        sys.path.pop(0)
    if len(seeds) != 20 or len(set(seeds)) != 20:
        raise RuntimeError("frozen KMeans seed contract must contain 20 unique seeds")
    return seeds


def evaluate_representation(
    representation_path: str | Path,
    *,
    dataset: str,
    entry: Mapping[str, Any],
) -> dict[str, Any]:
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        adjusted_rand_score,
        f1_score,
        normalized_mutual_info_score,
    )

    representation_path = Path(representation_path).resolve()
    features = np.asarray(
        np.load(representation_path, allow_pickle=False), dtype=np.float64
    )
    if features.ndim != 2 or not np.isfinite(features).all():
        raise ValueError("representation must be a finite two-dimensional array")
    roles = _load_roles(entry, len(features))
    classifier = LogisticRegression(
        C=1.0,
        penalty="l2",
        solver="lbfgs",
        max_iter=5000,
        fit_intercept=True,
        class_weight=None,
    )
    classifier.fit(features[roles["pool_indices"]], roles["pool_labels"])
    prediction = classifier.predict(features[roles["locked_indices"]])
    micro = float(f1_score(roles["locked_labels"], prediction, average="micro"))
    macro = float(f1_score(roles["locked_labels"], prediction, average="macro"))
    clustering_rows = []
    for seed in _kmeans_seeds(entry):
        assignment = KMeans(
            n_clusters=int(entry["class_count"]),
            random_state=seed,
            n_init=10,
            algorithm="lloyd",
        ).fit_predict(features)
        report_assignment = assignment[roles["locked_indices"]]
        clustering_rows.append({
            "kmeans_seed": seed,
            "NMI": float(normalized_mutual_info_score(roles["locked_labels"], report_assignment)),
            "ARI": float(adjusted_rand_score(roles["locked_labels"], report_assignment)),
            "assignment_sha256": canonical_hash(assignment.tolist()),
        })
    nmi = float(np.mean([row["NMI"] for row in clustering_rows]))
    ari = float(np.mean([row["ARI"] for row in clustering_rows]))
    value = {
        "schema_version": "metachorus.m9-vis-p1.single-representation-evaluation.v1",
        "status": "PASS",
        "dataset": dataset,
        "representation_path": str(representation_path),
        "representation_sha256": sha256_file(representation_path),
        "node_count": int(len(features)),
        "Micro_F1": micro,
        "Macro_F1": macro,
        "NMI": nmi,
        "ARI": ari,
        "F_cls": (micro + macro) / 2.0,
        "F_clu": (nmi + ari) / 2.0,
        "F_joint": (micro + macro + nmi + ari) / 4.0,
        "classification_fit_role": "GLOBAL_SEARCH_POOL",
        "classification_metric_role": entry["classification_metric_role"],
        "clustering_fit_role": "ALL_TARGET_NODES_UNLABELED",
        "clustering_metric_role": entry["classification_metric_role"],
        "classification_fit_count": 1,
        "kmeans_repeat_count": len(clustering_rows),
        "kmeans_n_init": 10,
        "kmeans_algorithm": "lloyd",
        "labels_used_during_training": False,
        "labels_used_during_clustering_fit": False,
        "raw_labels_persisted": False,
        "label_source_sha256": roles["label_sha256"],
        "private_search_mount_sha256": roles["mount_sha256"],
        "evaluation_role_manifest_sha256": roles["role_manifest_sha256"],
        "clustering_rows": clustering_rows,
    }
    value["evaluation_identity"] = canonical_hash(value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--representation", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--sources", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    sources = load_json(args.sources)
    entry = sources["datasets"][args.dataset]
    result = evaluate_representation(
        args.representation, dataset=args.dataset, entry=entry
    )
    atomic_write_json(args.output, result)
    print({"status": "PASS", "dataset": args.dataset, "evaluation_identity": result["evaluation_identity"]})


if __name__ == "__main__":
    main()
