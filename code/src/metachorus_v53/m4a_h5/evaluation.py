"""H5 search-only classification and clustering evaluation."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Mapping, Sequence

import numpy as np
import yaml
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, f1_score, normalized_mutual_info_score

from metachorus_v53.audit.common import (
    atomic_write_json,
    atomic_write_text,
    canonical_json_hash,
    sha256_file,
)

from .constants import (
    CLASSIFICATION_SPLIT_COUNT,
    CLASSIFIER_CONTRACT,
    CLUSTER_SEARCH_SEED_COUNT,
    KMEANS_CONTRACT,
    TARGET_NODE_COUNTS,
)
from .labels import load_private_search_mount


def _features(value: np.ndarray, dataset: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError("representation must be a finite two-dimensional matrix")
    if dataset not in TARGET_NODE_COUNTS or array.shape[0] != TARGET_NODE_COUNTS[dataset]:
        raise ValueError(f"{dataset} representation is not in canonical target-node order/size")
    return array


def build_classifier() -> LogisticRegression:
    return LogisticRegression(
        C=float(CLASSIFIER_CONTRACT["C"]),
        penalty=str(CLASSIFIER_CONTRACT["penalty"]),
        solver=str(CLASSIFIER_CONTRACT["solver"]),
        max_iter=int(CLASSIFIER_CONTRACT["max_iter"]),
        fit_intercept=bool(CLASSIFIER_CONTRACT["fit_intercept"]),
        class_weight=CLASSIFIER_CONTRACT["class_weight"],
    )


def evaluate_classification(
    representation: np.ndarray,
    *,
    dataset: str,
    private_mount_root: Path,
    split_seeds: Sequence[int],
    label_manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    features = _features(representation, dataset)
    if len(split_seeds) != CLASSIFICATION_SPLIT_COUNT:
        raise ValueError("H5 classification requires exactly five frozen within-pool splits")
    mount = load_private_search_mount(
        private_mount_root, dataset, expected_manifest=label_manifest
    )
    observed_seeds = mount["split_seeds"].astype(np.uint64, copy=False).tolist()
    if [int(value) for value in split_seeds] != [int(value) for value in observed_seeds]:
        raise ValueError("classification split seeds do not match the sealed private mount")
    pool = mount["global_search_pool_indices"].astype(np.int64, copy=False)
    labels = mount["global_search_pool_labels"].astype(np.int64, copy=False)
    train_positions = mount["search_train_pool_positions"].astype(np.int64, copy=False)
    validation_positions = mount["search_validation_pool_positions"].astype(np.int64, copy=False)
    if train_positions.shape[0] != CLASSIFICATION_SPLIT_COUNT or validation_positions.shape[0] != CLASSIFICATION_SPLIT_COUNT:
        raise RuntimeError("private mount does not contain five within-pool splits")
    output: list[dict[str, Any]] = []
    for split_index, split_seed in enumerate(split_seeds):
        train_pos = train_positions[split_index]
        validation_pos = validation_positions[split_index]
        if (
            train_pos.size == 0
            or validation_pos.size == 0
            or train_pos.min() < 0
            or validation_pos.min() < 0
            or train_pos.max() >= pool.size
            or validation_pos.max() >= pool.size
        ):
            raise ValueError("within-pool split positions are invalid")
        if np.intersect1d(train_pos, validation_pos).size:
            raise ValueError("classification train/validation positions overlap")
        if np.union1d(train_pos, validation_pos).size != pool.size:
            raise ValueError("classification split does not cover the global search pool")
        train = pool[train_pos]
        validation = pool[validation_pos]
        classifier = build_classifier()
        classifier.fit(features[train], labels[train_pos])
        predictions = classifier.predict(features[validation])
        micro = float(f1_score(labels[validation_pos], predictions, average="micro", zero_division=0))
        macro = float(f1_score(labels[validation_pos], predictions, average="macro", zero_division=0))
        output.append(
            {
                "classification_split_index": split_index,
                "classification_split_seed": int(split_seed),
                "train_count": int(train.size),
                "validation_count": int(validation.size),
                "micro_f1": micro,
                "macro_f1": macro,
                "F_cls": (micro + macro) / 2.0,
                "classifier": "logistic_regression",
                "feature_preprocessing": "none",
                "evaluation_role": "GLOBAL_SEARCH_POOL_ONLY",
                "h5_locked_report_access": False,
            }
        )
    return output


def _assignment_sha256(assignments: np.ndarray) -> str:
    values = np.asarray(assignments, dtype="<i4")
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def assignment_stability(assignments: Sequence[np.ndarray]) -> tuple[float, float]:
    if len(assignments) != CLUSTER_SEARCH_SEED_COUNT:
        raise ValueError("exactly twenty clustering assignments are required")
    pairwise = [
        float(adjusted_rand_score(assignments[left], assignments[right]))
        for left in range(len(assignments))
        for right in range(left + 1, len(assignments))
    ]
    return mean(pairwise), float(np.percentile(np.asarray(pairwise, dtype=np.float64), 10))


def evaluate_clustering(
    representation: np.ndarray,
    *,
    dataset: str,
    private_mount_root: Path,
    kmeans_seeds: Sequence[int],
    label_manifest: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    features = _features(representation, dataset)
    if len(kmeans_seeds) != CLUSTER_SEARCH_SEED_COUNT or len(set(kmeans_seeds)) != CLUSTER_SEARCH_SEED_COUNT:
        raise ValueError("H5 clustering requires exactly twenty unique frozen seeds")
    expected_seeds = [int(value) for value in label_manifest["seed_bank"]["clustering_search_seeds"]]
    if [int(value) for value in kmeans_seeds] != expected_seeds:
        raise ValueError("clustering seeds do not match the sealed H5 seed bank")
    mount = load_private_search_mount(
        private_mount_root, dataset, expected_manifest=label_manifest
    )
    pool = mount["global_search_pool_indices"].astype(np.int64, copy=False)
    labels = mount["global_search_pool_labels"].astype(np.int64, copy=False)
    if pool.size == 0 or pool.min() < 0 or pool.max() >= features.shape[0]:
        raise ValueError("global-search-pool indices exceed canonical target-node representations")
    class_count = int(np.unique(labels).size)
    rows: list[dict[str, Any]] = []
    assignments: list[np.ndarray] = []
    for index, seed in enumerate(kmeans_seeds):
        clusters = KMeans(
            n_clusters=class_count,
            n_init=int(KMEANS_CONTRACT["n_init"]),
            algorithm=str(KMEANS_CONTRACT["algorithm"]),
            random_state=int(seed),
        ).fit_predict(features)
        clusters = np.asarray(clusters, dtype=np.int64)
        assignments.append(clusters)
        observed = clusters[pool]
        nmi = float(normalized_mutual_info_score(labels, observed))
        ari = float(adjusted_rand_score(labels, observed))
        rows.append(
            {
                "kmeans_seed_index": index,
                "kmeans_seed": int(seed),
                "global_search_pool_count": int(pool.size),
                "nmi": nmi,
                "ari": ari,
                "F_clu": (nmi + ari) / 2.0,
                "assignment_sha256": _assignment_sha256(clusters),
                "kmeans_algorithm": "lloyd",
                "kmeans_n_init": 10,
                "fit_all_target_nodes_without_labels": True,
                "evaluation_role": "ENTIRE_GLOBAL_SEARCH_POOL",
                "h5_locked_report_access": False,
            }
        )
    values = [float(row["F_clu"]) for row in rows]
    sample_sd = stdev(values)
    standard_error = sample_sd / math.sqrt(CLUSTER_SEARCH_SEED_COUNT)
    stability_mean, stability_p10 = assignment_stability(assignments)
    summary = {
        "repeat_count": CLUSTER_SEARCH_SEED_COUNT,
        "mean_F_clu": mean(values),
        "sample_sd_F_clu": sample_sd,
        "SE_clu": standard_error,
        "LCB_clu": mean(values) - standard_error,
        "assignment_stability_mean": stability_mean,
        "assignment_stability_p10": stability_p10,
        "assignment_hash_count": len({row["assignment_sha256"] for row in rows}),
        "global_search_pool_count": int(pool.size),
        "h5_locked_report_access": False,
    }
    return rows, summary


def evaluator_contract() -> dict[str, Any]:
    return {
        "schema_version": "metachorus.v53.m4a-h5.search-evaluator-contract.v1",
        "status": "FROZEN_BEFORE_ANY_H5_EVALUATION",
        "classification": dict(CLASSIFIER_CONTRACT),
        "clustering": dict(KMEANS_CONTRACT),
        "classification_repeat_count": CLASSIFICATION_SPLIT_COUNT,
        "clustering_repeat_count": CLUSTER_SEARCH_SEED_COUNT,
        "classification_score": "F_cls=(MicroF1+MacroF1)/2",
        "clustering_score": "F_clu=(NMI+ARI)/2",
        "clustering_lcb": "mean(F_clu)-sample_sd(F_clu)/sqrt(20)",
        "label_roles": {
            "classification_fit": "SEARCH_TRAIN within GLOBAL_SEARCH_POOL",
            "classification_metric": "SEARCH_VALIDATION within GLOBAL_SEARCH_POOL",
            "clustering_fit": "all_target_nodes_without_labels",
            "clustering_metric": "entire_GLOBAL_SEARCH_POOL",
            "forbidden": ["H5_LOCKED_REPORT", "full_label_array", "future_S_eval"],
        },
        "raw_sd_or_naive_se_is_primary_gate": False,
        "taskwise_continuous_weight_search": False,
    }


def write_evaluator_contract(result_root: Path) -> Path:
    value = evaluator_contract()
    path = result_root / "TC_LPMSO_H5_SEARCH_EVALUATOR_CONTRACT.yaml"
    text = yaml.safe_dump(value, allow_unicode=True, sort_keys=True)
    atomic_write_text(path, text)
    return path


def seal_evaluator(
    *,
    result_root: Path,
    label_manifest_path: Path,
    private_mount_audit_path: Path,
) -> dict[str, Any]:
    contract_path = result_root / "TC_LPMSO_H5_SEARCH_EVALUATOR_CONTRACT.yaml"
    basis = {
        "contract_sha256": sha256_file(contract_path),
        "label_manifest_sha256": sha256_file(label_manifest_path),
        "private_mount_audit_sha256": sha256_file(private_mount_audit_path),
        "sealed_before_any_h5_evaluation": True,
        "h5_locked_report_access": False,
        "future_s_eval_access": False,
    }
    payload = {
        "schema_version": "metachorus.v53.m4a-h5.search-evaluator-seal.v1",
        "status": "SEALED_BEFORE_H5_EVALUATION",
        "basis": basis,
        "identity": canonical_json_hash(basis),
    }
    atomic_write_json(result_root / "TC_LPMSO_H5_SEARCH_EVALUATOR_SEAL.json", payload)
    return payload
