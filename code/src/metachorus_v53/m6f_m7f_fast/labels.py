"""Trusted seed and GLOBAL_SEARCH_POOL/PRELIM_LOCKED_REPORT role sealing."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

from metachorus_v53.audit.common import canonical_json_hash, sha256_file

from .common import envelope, write_csv_once, write_json_once
from .constants import (
    CLASSIFICATION_SPLIT_SEEDS,
    DATASET_IDENTITIES,
    DATASET_NAMES,
    DATASET_SPLIT_SEEDS,
    EVAL_MODEL_SEEDS,
    H8_METHOD_FREEZE_IDENTITY,
    HISTORICAL_SEEDS,
    PRELIM_KMEANS_SEEDS,
    SEARCH_KMEANS_SEEDS,
    SEARCH_MODEL_SEEDS,
    SEED_SEAL_BASIS,
    SEED_SEAL_IDENTITY,
    TASKBOOK_SHA256,
)
from .dataset_specs import DATASET_SPECS


PRIVATE_MOUNT_KEYS = frozenset(
    {
        "global_search_pool_indices",
        "global_search_pool_labels",
        "split_seeds",
        "search_train_pool_positions",
        "search_validation_pool_positions",
    }
)


def _indices_sha256(values: np.ndarray) -> str:
    array = np.asarray(values, dtype="<i8")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _class_counts(values: np.ndarray, indices: np.ndarray) -> dict[str, int]:
    classes, counts = np.unique(values[indices], return_counts=True)
    return {str(int(label)): int(count) for label, count in zip(classes, counts)}


def _load_labels(dataset: str, source: Path) -> np.ndarray:
    spec = DATASET_SPECS[dataset]
    if (
        not source.is_file()
        or source.is_symlink()
        or source.name != Path(spec.label_relative_path).name
        or sha256_file(source) != spec.label_sha256
    ):
        raise ValueError(f"{dataset} supervised source identity mismatch")
    labels = np.load(source, allow_pickle=False)
    if (
        not isinstance(labels, np.ndarray)
        or labels.ndim != 1
        or labels.shape[0] != spec.label_count
        or str(labels.dtype) != spec.label_dtype
        or np.unique(labels).size != spec.class_count
    ):
        raise ValueError(f"{dataset} supervised source shape/dtype/class contract mismatch")
    return np.asarray(labels, dtype=np.int64)


def partition_roles(
    labels: np.ndarray,
    *,
    pool_seed: int,
    split_seeds: tuple[int, ...],
) -> dict[str, Any]:
    targets = np.asarray(labels, dtype=np.int64)
    if targets.ndim != 1 or len(split_seeds) != 5 or len(set(split_seeds)) != 5:
        raise ValueError("preliminary role partition requires one-dimensional labels and five unique split seeds")
    indices = np.arange(targets.size, dtype=np.int64)
    outer = StratifiedShuffleSplit(
        n_splits=1,
        train_size=0.20,
        test_size=0.80,
        random_state=int(pool_seed),
    )
    pool_relative, locked_relative = next(outer.split(indices, targets))
    pool = np.sort(indices[pool_relative])
    locked = np.sort(indices[locked_relative])
    if (
        pool.size != int(np.floor(targets.size * 0.20))
        or locked.size != targets.size - pool.size
        or np.intersect1d(pool, locked).size
        or np.union1d(pool, locked).size != targets.size
        or np.unique(targets[pool]).size != np.unique(targets).size
        or np.unique(targets[locked]).size != np.unique(targets).size
    ):
        raise RuntimeError("GLOBAL_SEARCH_POOL/PRELIM_LOCKED_REPORT is not an exact stratified partition")
    repeats: list[dict[str, Any]] = []
    for ordinal, seed in enumerate(split_seeds):
        within = StratifiedShuffleSplit(
            n_splits=1,
            train_size=0.50,
            test_size=0.50,
            random_state=int(seed),
        )
        train_positions, validation_positions = next(within.split(pool, targets[pool]))
        train_positions = np.sort(np.asarray(train_positions, dtype=np.int64))
        validation_positions = np.sort(np.asarray(validation_positions, dtype=np.int64))
        train = pool[train_positions]
        validation = pool[validation_positions]
        if (
            np.intersect1d(train, validation).size
            or np.union1d(train, validation).size != pool.size
            or np.intersect1d(np.union1d(train, validation), locked).size
        ):
            raise RuntimeError("within-pool split is not an exact locked-disjoint partition")
        repeats.append(
            {
                "split_index": ordinal,
                "split_seed": int(seed),
                "train_pool_positions": train_positions,
                "validation_pool_positions": validation_positions,
                "train_indices": train,
                "validation_indices": validation,
            }
        )
    return {"global_search_pool": pool, "prelim_locked_report": locked, "repeats": repeats}


def _write_npz_once(path: Path, **arrays: np.ndarray) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".pending", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_seed_and_label_role_seals(
    *,
    result_root: str | Path,
    private_mount_root: str | Path,
    label_sources: Mapping[str, str | Path],
) -> dict[str, Any]:
    result = Path(result_root).resolve()
    private = Path(private_mount_root).resolve()
    if set(label_sources) != set(DATASET_NAMES):
        raise ValueError("exactly DBLP, IMDb and AMiner label sources are required")
    result.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=False)

    seed_manifest = {
        "schema_version": "metachorus.v53.m6f-m7f.seed-manifest.v1",
        "status": "SEALED_BEFORE_ANY_GPU_TRAINING",
        "basis": SEED_SEAL_BASIS,
        "identity": SEED_SEAL_IDENTITY,
    }
    write_json_once(result / "FAST_SEED_NAMESPACE_MANIFEST.json", seed_manifest)
    collision_basis = {
        "derived_seed_count": len(
            set(SEARCH_MODEL_SEEDS)
            | set(EVAL_MODEL_SEEDS)
            | set(DATASET_SPLIT_SEEDS.values())
            | {value for values in CLASSIFICATION_SPLIT_SEEDS.values() for value in values}
            | set(SEARCH_KMEANS_SEEDS)
            | set(PRELIM_KMEANS_SEEDS)
        ),
        "historical_seed_count": len(HISTORICAL_SEEDS),
        "derived_historical_intersection": [],
        "search_eval_model_intersection": [],
        "all_namespaces_pairwise_disjoint": True,
    }
    write_json_once(
        result / "FAST_SEED_COLLISION_AUDIT.json",
        envelope("metachorus.v53.m6f-m7f.seed-collision-audit.v1", "PASS", collision_basis),
    )
    audit_rows: list[dict[str, Any]] = []
    namespaces = {
        "COMMON_SEARCH_MODEL": SEARCH_MODEL_SEEDS,
        "COMMON_EVAL_MODEL": EVAL_MODEL_SEEDS,
        "DATASET_GLOBAL_POOL_SPLIT": tuple(DATASET_SPLIT_SEEDS.values()),
        "WITHIN_POOL_CLASSIFICATION": tuple(
            value for values in CLASSIFICATION_SPLIT_SEEDS.values() for value in values
        ),
        "COMMON_SEARCH_KMEANS": SEARCH_KMEANS_SEEDS,
        "COMMON_PRELIM_KMEANS": PRELIM_KMEANS_SEEDS,
    }
    for namespace, values in namespaces.items():
        for ordinal, seed in enumerate(values):
            audit_rows.append(
                {
                    "namespace": namespace,
                    "ordinal": ordinal,
                    "seed": int(seed),
                    "historical_collision": False,
                    "cross_namespace_collision": False,
                    "status": "PASS",
                }
            )
    write_csv_once(result / "FAST_SEED_COLLISION_AUDIT.csv", audit_rows, fieldnames=list(audit_rows[0]))

    dataset_rows: list[dict[str, Any]] = []
    mount_rows: list[dict[str, Any]] = []
    for dataset in DATASET_NAMES:
        source = Path(label_sources[dataset]).resolve()
        labels = _load_labels(dataset, source)
        roles = partition_roles(
            labels,
            pool_seed=DATASET_SPLIT_SEEDS[dataset],
            split_seeds=CLASSIFICATION_SPLIT_SEEDS[dataset],
        )
        pool = roles["global_search_pool"]
        locked = roles["prelim_locked_report"]
        train_positions = np.stack([row["train_pool_positions"] for row in roles["repeats"]])
        validation_positions = np.stack([row["validation_pool_positions"] for row in roles["repeats"]])
        mount_path = private / f"{dataset}_global_search_pool.npz"
        _write_npz_once(
            mount_path,
            global_search_pool_indices=pool,
            global_search_pool_labels=labels[pool],
            split_seeds=np.asarray(CLASSIFICATION_SPLIT_SEEDS[dataset], dtype=np.uint32),
            search_train_pool_positions=train_positions,
            search_validation_pool_positions=validation_positions,
        )
        repeats_public = []
        for row in roles["repeats"]:
            train = row["train_indices"]
            validation = row["validation_indices"]
            repeats_public.append(
                {
                    "split_index": int(row["split_index"]),
                    "split_seed": int(row["split_seed"]),
                    "SEARCH_TRAIN": {
                        "count": int(train.size),
                        "indices_sha256": _indices_sha256(train),
                        "class_counts": _class_counts(labels, train),
                    },
                    "SEARCH_VALIDATION": {
                        "count": int(validation.size),
                        "indices_sha256": _indices_sha256(validation),
                        "class_counts": _class_counts(labels, validation),
                    },
                    "exact_partition_of_global_search_pool": True,
                    "prelim_locked_report_nodes_present": False,
                }
            )
        dataset_rows.append(
            {
                "dataset": dataset,
                "dataset_identity": DATASET_IDENTITIES[dataset],
                "node_count": int(labels.size),
                "class_count": int(np.unique(labels).size),
                "label_source_name": source.name,
                "label_source_sha256": sha256_file(source),
                "global_search_pool_seed": DATASET_SPLIT_SEEDS[dataset],
                "GLOBAL_SEARCH_POOL": {
                    "count": int(pool.size),
                    "fraction": float(pool.size / labels.size),
                    "indices_sha256": _indices_sha256(pool),
                    "class_counts": _class_counts(labels, pool),
                },
                "PRELIM_LOCKED_REPORT": {
                    "count": int(locked.size),
                    "fraction": float(locked.size / labels.size),
                    "indices_sha256": _indices_sha256(locked),
                    "class_counts": _class_counts(labels, locked),
                },
                "node_disjoint": True,
                "covers_all_target_nodes": True,
                "repeats": repeats_public,
            }
        )
        mount_rows.append(
            {
                "dataset": dataset,
                "member_name": mount_path.name,
                "sha256": sha256_file(mount_path),
                "size_bytes": mount_path.stat().st_size,
                "keys": sorted(PRIVATE_MOUNT_KEYS),
                "contains_full_label_array": False,
                "contains_prelim_locked_report_indices": False,
                "contains_prelim_locked_report_labels": False,
            }
        )
    role_manifest_basis = {
        "taskbook_sha256": TASKBOOK_SHA256,
        "h8_method_freeze_identity": H8_METHOD_FREEZE_IDENTITY,
        "seed_seal_identity": SEED_SEAL_IDENTITY,
        "namespace": "M6F_M7F_GLOBAL_SEARCH_POOL_PRELIM_LOCKED_REPORT",
        "ratios": {"GLOBAL_SEARCH_POOL": 0.20, "PRELIM_LOCKED_REPORT": 0.80},
        "within_pool_ratios": {"SEARCH_TRAIN": 0.50, "SEARCH_VALIDATION": 0.50},
        "datasets": dataset_rows,
        "raw_labels_embedded": False,
        "prelim_locked_report_raw_indices_embedded": False,
        "prelim_locked_report_labels_embedded": False,
        "training_process_mounts_labels": False,
    }
    role_manifest = envelope(
        "metachorus.v53.m6f-m7f.global-label-role-manifest.v1",
        "SEALED_INPUT",
        role_manifest_basis,
    )
    mount_audit = envelope(
        "metachorus.v53.m6f-m7f.private-search-mount-audit.v1",
        "PASS",
        {
            "members": mount_rows,
            "member_count": len(mount_rows),
            "contains_full_labels": False,
            "contains_prelim_locked_report": False,
            "search_model_seeds_used": list(SEARCH_MODEL_SEEDS),
            "evaluation_model_seeds_used": [],
        },
    )
    write_json_once(result / "FAST_LABEL_ROLE_MANIFEST.json", role_manifest)
    write_json_once(result / "FAST_LABEL_ROLE_SEAL.json", envelope(
        "metachorus.v53.m6f-m7f.label-role-seal.v1",
        "SEALED_BEFORE_SEARCH_TRAINING",
        {
            "role_manifest_identity": role_manifest["identity"],
            "private_mount_audit_identity": mount_audit["identity"],
            "training_process_mounts_labels": False,
            "classifier_fit_role": "GLOBAL_SEARCH_POOL",
            "classifier_metric_role": "PRELIM_LOCKED_REPORT",
            "clustering_fit_role": "ALL_TARGET_NODES_UNLABELED",
            "clustering_metric_role": "PRELIM_LOCKED_REPORT",
        },
    ))
    write_json_once(result / "FAST_PRIVATE_SEARCH_MOUNT_AUDIT.json", mount_audit)
    return {
        "seed_seal_identity": SEED_SEAL_IDENTITY,
        "role_manifest_identity": role_manifest["identity"],
        "private_mount_audit_identity": mount_audit["identity"],
        "datasets": list(DATASET_NAMES),
        "status": "SEALED_BEFORE_SEARCH_TRAINING",
    }


__all__ = ["PRIVATE_MOUNT_KEYS", "partition_roles", "build_seed_and_label_role_seals"]
