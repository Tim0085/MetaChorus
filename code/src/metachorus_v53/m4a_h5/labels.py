"""Frozen global 20/80 label roles and search-only physical mounts for H5."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy import io as scipy_io
from sklearn.model_selection import StratifiedShuffleSplit

from metachorus_v53.audit.common import (
    atomic_write_json,
    canonical_json_hash,
    read_json,
    sha256_file,
)

from .constants import (
    CLASSIFICATION_SPLIT_COUNT,
    CLUSTER_SEARCH_SEED_COUNT,
    EXPECTED_CLASS_COUNTS,
    FREEBASE_EXPLORATION_NAMESPACE,
    FUTURE_S_EVAL_MODEL_SEEDS,
    GLOBAL_POOL_NAMESPACE,
    GLOBAL_SEARCH_POOL_FRACTION,
    PARENT_PACKAGE_SHA256,
    TASKBOOK_SHA256,
    TARGET_NODE_COUNTS,
    WITHIN_POOL_TRAIN_FRACTION,
)


PRIVATE_MOUNT_KEYS = {
    "global_search_pool_indices",
    "global_search_pool_labels",
    "split_seeds",
    "search_train_pool_positions",
    "search_validation_pool_positions",
}


def _canonical_seed(namespace: str, *, dataset: str, role: str, index: int, collision: int = 0) -> int:
    payload = {
        "namespace": namespace,
        "taskbook_sha256": TASKBOOK_SHA256,
        "parent_package_sha256": PARENT_PACKAGE_SHA256,
        "dataset": dataset,
        "role": role,
        "index": int(index),
        "collision": int(collision),
    }
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], "big", signed=False)


def derive_seed_bank(datasets: tuple[str, ...] = ("acm", "freebase")) -> dict[str, Any]:
    """Derive every H5 search seed from the two frozen hashes and namespaces."""

    used: set[int] = set(FUTURE_S_EVAL_MODEL_SEEDS)

    def allocate(namespace: str, dataset: str, role: str, count: int) -> list[int]:
        output: list[int] = []
        for index in range(count):
            collision = 0
            value = _canonical_seed(
                namespace,
                dataset=dataset,
                role=role,
                index=index,
                collision=collision,
            )
            while value in used:
                collision += 1
                value = _canonical_seed(
                    namespace,
                    dataset=dataset,
                    role=role,
                    index=index,
                    collision=collision,
                )
            used.add(value)
            output.append(value)
        return output

    per_dataset: dict[str, Any] = {}
    for dataset in datasets:
        per_dataset[dataset] = {
            "global_search_pool_seed": allocate(
                GLOBAL_POOL_NAMESPACE, dataset, "global_search_pool", 1
            )[0],
            "within_pool_classification_split_seeds": allocate(
                GLOBAL_POOL_NAMESPACE,
                dataset,
                "within_pool_classification_split",
                CLASSIFICATION_SPLIT_COUNT,
            ),
        }
    clustering = allocate(GLOBAL_POOL_NAMESPACE, "all", "clustering_search", CLUSTER_SEARCH_SEED_COUNT)
    exploration = allocate(
        FREEBASE_EXPLORATION_NAMESPACE,
        "freebase",
        "four_band_exploration",
        1,
    )[0]
    flat = [
        value
        for row in per_dataset.values()
        for value in (
            [int(row["global_search_pool_seed"])]
            + [int(item) for item in row["within_pool_classification_split_seeds"]]
        )
    ] + clustering + [exploration]
    if len(flat) != len(set(flat)):
        raise RuntimeError("H5 derived seed bank contains a collision")
    if set(flat) & set(FUTURE_S_EVAL_MODEL_SEEDS):
        raise RuntimeError("H5 search seed bank overlaps future S_eval seeds")
    return {
        "namespace": GLOBAL_POOL_NAMESPACE,
        "per_dataset": per_dataset,
        "clustering_search_seeds": clustering,
        "freebase_exploration_seed": exploration,
        "future_s_eval_seeds_used": False,
    }


def load_acm_labels(acm_mat: str | Path) -> np.ndarray:
    raw = scipy_io.loadmat(Path(acm_mat))["PvsC"]
    conf_ids = [0, 1, 9, 10, 13]
    label_ids = [0, 1, 2, 2, 1]
    selected = (raw[:, conf_ids].sum(1) != 0).A1.nonzero()[0]
    filtered = raw[selected]
    paper_indices, conference_indices = filtered.nonzero()
    labels = np.zeros(len(selected), dtype=np.int64)
    for conference, label in zip(conf_ids, label_ids):
        labels[paper_indices[conference_indices == conference]] = label
    return _validated_labels("acm", labels)


def load_freebase_labels(label_file: str | Path) -> np.ndarray:
    labels = np.load(Path(label_file), allow_pickle=False).astype(np.int64, copy=False)
    return _validated_labels("freebase", labels)


def _validated_labels(dataset: str, value: np.ndarray) -> np.ndarray:
    labels = np.asarray(value, dtype=np.int64)
    if labels.ndim != 1 or labels.shape[0] != TARGET_NODE_COUNTS[dataset]:
        raise ValueError(f"{dataset} labels do not match canonical target-node order/size")
    classes, counts = np.unique(labels, return_counts=True)
    observed = {int(label): int(count) for label, count in zip(classes, counts)}
    if observed != EXPECTED_CLASS_COUNTS[dataset]:
        raise ValueError(f"{dataset} label distribution mismatch: {observed}")
    if min(observed.values()) < 10:
        raise ValueError(f"{dataset} class counts cannot support the frozen stratified protocol")
    return labels


def _indices_sha256(values: np.ndarray) -> str:
    array = np.asarray(values, dtype="<i8")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _class_counts(labels: np.ndarray, indices: np.ndarray) -> dict[str, int]:
    classes, counts = np.unique(labels[indices], return_counts=True)
    return {str(int(label)): int(count) for label, count in zip(classes, counts)}


def global_search_roles(
    labels: np.ndarray,
    *,
    pool_seed: int,
    split_seeds: list[int] | tuple[int, ...],
) -> dict[str, Any]:
    targets = np.asarray(labels, dtype=np.int64)
    if len(split_seeds) != CLASSIFICATION_SPLIT_COUNT or len(set(split_seeds)) != CLASSIFICATION_SPLIT_COUNT:
        raise ValueError("H5 requires exactly five unique frozen classification split seeds")
    indices = np.arange(targets.size, dtype=np.int64)
    outer = StratifiedShuffleSplit(
        n_splits=1,
        train_size=GLOBAL_SEARCH_POOL_FRACTION,
        test_size=1.0 - GLOBAL_SEARCH_POOL_FRACTION,
        random_state=int(pool_seed),
    )
    pool_relative, locked_relative = next(outer.split(indices, targets))
    pool = np.sort(indices[pool_relative])
    locked = np.sort(indices[locked_relative])
    expected_pool_count = int(np.floor(targets.size * GLOBAL_SEARCH_POOL_FRACTION))
    if pool.size != expected_pool_count or locked.size != targets.size - expected_pool_count:
        raise RuntimeError("global label-role partition has an unexpected deterministic size")
    if np.intersect1d(pool, locked).size or np.union1d(pool, locked).size != indices.size:
        raise RuntimeError("global search pool and H5 locked report are not an exact partition")
    if np.unique(targets[pool]).size != np.unique(targets).size:
        raise RuntimeError("global search pool omitted a class")
    if np.unique(targets[locked]).size != np.unique(targets).size:
        raise RuntimeError("H5 locked report omitted a class")
    repeats: list[dict[str, Any]] = []
    for split_index, split_seed in enumerate(split_seeds):
        within = StratifiedShuffleSplit(
            n_splits=1,
            train_size=WITHIN_POOL_TRAIN_FRACTION,
            test_size=1.0 - WITHIN_POOL_TRAIN_FRACTION,
            random_state=int(split_seed),
        )
        train_positions, validation_positions = next(within.split(pool, targets[pool]))
        train_positions = np.sort(np.asarray(train_positions, dtype=np.int64))
        validation_positions = np.sort(np.asarray(validation_positions, dtype=np.int64))
        train = pool[train_positions]
        validation = pool[validation_positions]
        if np.intersect1d(train, validation).size or np.union1d(train, validation).size != pool.size:
            raise RuntimeError("within-pool split is not an exact 50/50 partition")
        if np.intersect1d(np.union1d(train, validation), locked).size:
            raise RuntimeError("H5 locked report leaked into a search repeat")
        repeats.append(
            {
                "split_index": split_index,
                "split_seed": int(split_seed),
                "train_pool_positions": train_positions,
                "validation_pool_positions": validation_positions,
                "train_indices": train,
                "validation_indices": validation,
            }
        )
    return {
        "global_search_pool": pool,
        "h5_locked_report": locked,
        "repeats": repeats,
    }


def build_global_label_role_artifacts(
    *,
    result_root: Path,
    private_mount_root: Path,
    dataset_labels: Mapping[str, np.ndarray],
    label_source_files: Mapping[str, Path],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Publish public hashes and a physically search-only private mount."""

    result_root.mkdir(parents=True, exist_ok=True)
    private_mount_root.mkdir(parents=True, exist_ok=False)
    seed_bank = derive_seed_bank(tuple(sorted(dataset_labels)))
    datasets: list[dict[str, Any]] = []
    mount_rows: list[dict[str, Any]] = []
    for dataset in sorted(dataset_labels):
        labels = _validated_labels(dataset, dataset_labels[dataset])
        seeds = seed_bank["per_dataset"][dataset]
        roles = global_search_roles(
            labels,
            pool_seed=int(seeds["global_search_pool_seed"]),
            split_seeds=[int(value) for value in seeds["within_pool_classification_split_seeds"]],
        )
        pool = roles["global_search_pool"]
        locked = roles["h5_locked_report"]
        train_positions = np.stack(
            [row["train_pool_positions"] for row in roles["repeats"]], axis=0
        )
        validation_positions = np.stack(
            [row["validation_pool_positions"] for row in roles["repeats"]], axis=0
        )
        mount_path = private_mount_root / f"{dataset}_global_search_pool.npz"
        np.savez_compressed(
            mount_path,
            global_search_pool_indices=pool,
            global_search_pool_labels=labels[pool],
            split_seeds=np.asarray(
                seeds["within_pool_classification_split_seeds"], dtype=np.uint32
            ),
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
                    "h5_locked_report_nodes_present": False,
                }
            )
        datasets.append(
            {
                "dataset": dataset,
                "node_count": int(labels.size),
                "class_count": int(np.unique(labels).size),
                "label_source_name": Path(label_source_files[dataset]).name,
                "label_source_sha256": sha256_file(Path(label_source_files[dataset])),
                "global_search_pool_seed": int(seeds["global_search_pool_seed"]),
                "GLOBAL_SEARCH_POOL": {
                    "count": int(pool.size),
                    "fraction": float(pool.size / labels.size),
                    "indices_sha256": _indices_sha256(pool),
                    "class_counts": _class_counts(labels, pool),
                },
                "H5_LOCKED_REPORT": {
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
                "contains_h5_locked_report_indices": False,
                "contains_h5_locked_report_labels": False,
            }
        )
    manifest = {
        "schema_version": "metachorus.v53.m4a-h5.global-label-role-manifest.v1",
        "status": "SEALED_INPUT",
        "taskbook_sha256": TASKBOOK_SHA256,
        "parent_package_sha256": PARENT_PACKAGE_SHA256,
        "namespace": GLOBAL_POOL_NAMESPACE,
        "ratios": {"GLOBAL_SEARCH_POOL": 0.20, "H5_LOCKED_REPORT": 0.80},
        "within_pool_ratios": {"SEARCH_TRAIN": 0.50, "SEARCH_VALIDATION": 0.50},
        "seed_bank": seed_bank,
        "datasets": datasets,
        "raw_labels_embedded": False,
        "h5_locked_report_raw_indices_embedded": False,
        "h5_locked_report_labels_embedded": False,
        "historical_disclosure": "labels were used before H5; H5_LOCKED_REPORT is frozen from H5 onward",
    }
    audit = {
        "schema_version": "metachorus.v53.m4a-h5.private-search-mount-audit.v1",
        "status": "PASS",
        "member_count": len(mount_rows),
        "members": mount_rows,
        "contains_full_labels": False,
        "contains_h5_locked_report": False,
        "future_s_eval_seeds_used": False,
    }
    atomic_write_json(result_root / "TC_LPMSO_H5_GLOBAL_LABEL_ROLE_MANIFEST.json", manifest)
    atomic_write_json(result_root / "TC_LPMSO_H5_PRIVATE_SEARCH_MOUNT_AUDIT.json", audit)
    return manifest, audit


def _manifest_dataset(manifest: Mapping[str, Any], dataset: str) -> Mapping[str, Any]:
    rows = [row for row in manifest.get("datasets", []) if str(row.get("dataset")) == dataset]
    if len(rows) != 1:
        raise RuntimeError(f"label-role manifest lacks one canonical {dataset} row")
    return rows[0]


def load_private_search_mount(
    private_mount_root: Path,
    dataset: str,
    *,
    expected_manifest: Mapping[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    if dataset not in TARGET_NODE_COUNTS:
        raise ValueError(f"unknown H5 dataset: {dataset}")
    path = private_mount_root / f"{dataset}_global_search_pool.npz"
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != PRIVATE_MOUNT_KEYS:
            raise RuntimeError(f"search-only mount has unexpected members: {sorted(payload.files)}")
        result = {name: np.asarray(payload[name]).copy() for name in PRIVATE_MOUNT_KEYS}
    pool = result["global_search_pool_indices"].astype(np.int64, copy=False)
    labels = result["global_search_pool_labels"].astype(np.int64, copy=False)
    if pool.ndim != 1 or labels.shape != pool.shape or np.unique(pool).size != pool.size:
        raise RuntimeError("invalid global-search-pool mount identity")
    if pool.size == 0 or pool.min() < 0 or pool.max() >= TARGET_NODE_COUNTS[dataset]:
        raise RuntimeError("global-search-pool mount exceeds canonical target-node bounds")
    split_seeds = result["split_seeds"]
    train_positions = result["search_train_pool_positions"]
    validation_positions = result["search_validation_pool_positions"]
    if split_seeds.shape != (CLASSIFICATION_SPLIT_COUNT,) or len(set(split_seeds.tolist())) != CLASSIFICATION_SPLIT_COUNT:
        raise RuntimeError("private mount does not contain five unique split seeds")
    if train_positions.ndim != 2 or validation_positions.ndim != 2:
        raise RuntimeError("private mount split positions must be two-dimensional")
    if train_positions.shape[0] != CLASSIFICATION_SPLIT_COUNT or validation_positions.shape[0] != CLASSIFICATION_SPLIT_COUNT:
        raise RuntimeError("private mount split count differs from the frozen protocol")
    for split_index in range(CLASSIFICATION_SPLIT_COUNT):
        train = np.asarray(train_positions[split_index], dtype=np.int64)
        validation = np.asarray(validation_positions[split_index], dtype=np.int64)
        if (
            train.size == 0
            or validation.size == 0
            or train.min() < 0
            or validation.min() < 0
            or train.max() >= pool.size
            or validation.max() >= pool.size
            or np.intersect1d(train, validation).size
            or np.union1d(train, validation).size != pool.size
        ):
            raise RuntimeError(f"private mount split {split_index} is not an exact pool partition")
    observed_classes, observed_counts = np.unique(labels, return_counts=True)
    if set(observed_classes.tolist()) != set(EXPECTED_CLASS_COUNTS[dataset]):
        raise RuntimeError("private mount omitted a canonical class")
    if expected_manifest is not None:
        row = _manifest_dataset(expected_manifest, dataset)
        pool_manifest = row["GLOBAL_SEARCH_POOL"]
        observed_class_counts = {
            str(int(label)): int(count)
            for label, count in zip(observed_classes, observed_counts)
        }
        if (
            int(row["node_count"]) != TARGET_NODE_COUNTS[dataset]
            or int(pool_manifest["count"]) != pool.size
            or str(pool_manifest["indices_sha256"]) != _indices_sha256(pool)
            or dict(pool_manifest["class_counts"]) != observed_class_counts
        ):
            raise RuntimeError(f"{dataset} private mount differs from its sealed public manifest")
        repeats = row.get("repeats", [])
        if len(repeats) != CLASSIFICATION_SPLIT_COUNT:
            raise RuntimeError("sealed manifest lacks five classification repeats")
        for split_index, repeat in enumerate(repeats):
            train = pool[np.asarray(train_positions[split_index], dtype=np.int64)]
            validation = pool[np.asarray(validation_positions[split_index], dtype=np.int64)]
            if (
                int(repeat["split_index"]) != split_index
                or int(repeat["split_seed"]) != int(split_seeds[split_index])
                or int(repeat["SEARCH_TRAIN"]["count"]) != train.size
                or str(repeat["SEARCH_TRAIN"]["indices_sha256"]) != _indices_sha256(train)
                or int(repeat["SEARCH_VALIDATION"]["count"]) != validation.size
                or str(repeat["SEARCH_VALIDATION"]["indices_sha256"]) != _indices_sha256(validation)
            ):
                raise RuntimeError(f"{dataset} private split {split_index} differs from the seal")
    return result


def verify_global_label_role_seal(
    *,
    result_root: Path,
    private_mount_root: Path,
) -> dict[str, Any]:
    manifest_path = result_root / "TC_LPMSO_H5_GLOBAL_LABEL_ROLE_MANIFEST.json"
    audit_path = result_root / "TC_LPMSO_H5_PRIVATE_SEARCH_MOUNT_AUDIT.json"
    contract_path = result_root / "TC_LPMSO_H5_SEARCH_EVALUATOR_CONTRACT.yaml"
    label_seal = read_json(result_root / "TC_LPMSO_H5_GLOBAL_LABEL_ROLE_SEAL.json")
    evaluator_seal = read_json(result_root / "TC_LPMSO_H5_SEARCH_EVALUATOR_SEAL.json")
    manifest = read_json(manifest_path)
    audit = read_json(audit_path)
    if label_seal.get("status") != "SEALED_BEFORE_H5_EVALUATION" or evaluator_seal.get("status") != "SEALED_BEFORE_H5_EVALUATION":
        raise RuntimeError("H5 label/evaluator seals are not active")
    if label_seal.get("identity") != canonical_json_hash(label_seal.get("basis", {})):
        raise RuntimeError("H5 global label-role seal identity mismatch")
    if evaluator_seal.get("identity") != canonical_json_hash(evaluator_seal.get("basis", {})):
        raise RuntimeError("H5 evaluator seal identity mismatch")
    label_basis = label_seal["basis"]
    evaluator_basis = evaluator_seal["basis"]
    expected_hashes = {
        "manifest": sha256_file(manifest_path),
        "audit": sha256_file(audit_path),
        "contract": sha256_file(contract_path),
    }
    if (
        label_basis.get("manifest_sha256") != expected_hashes["manifest"]
        or label_basis.get("private_mount_audit_sha256") != expected_hashes["audit"]
        or label_basis.get("evaluator_contract_sha256") != expected_hashes["contract"]
        or evaluator_basis.get("label_manifest_sha256") != expected_hashes["manifest"]
        or evaluator_basis.get("private_mount_audit_sha256") != expected_hashes["audit"]
        or evaluator_basis.get("contract_sha256") != expected_hashes["contract"]
    ):
        raise RuntimeError("H5 label/evaluator seal dependency hash mismatch")
    if manifest.get("taskbook_sha256") != TASKBOOK_SHA256 or manifest.get("parent_package_sha256") != PARENT_PACKAGE_SHA256:
        raise RuntimeError("H5 label-role manifest is bound to the wrong frozen inputs")
    expected_names = {f"{dataset}_global_search_pool.npz" for dataset in TARGET_NODE_COUNTS}
    members = sorted(private_mount_root.glob("*_global_search_pool.npz"))
    if {path.name for path in members} != expected_names or len(list(private_mount_root.iterdir())) != len(expected_names):
        raise RuntimeError("private search mount contains missing or unexpected members")
    audit_members = {str(row["member_name"]): row for row in audit.get("members", [])}
    if audit.get("status") != "PASS" or int(audit.get("member_count", -1)) != len(expected_names) or set(audit_members) != expected_names:
        raise RuntimeError("private search mount audit is incomplete")
    member_identity_rows = []
    for path in members:
        row = audit_members[path.name]
        observed_sha = sha256_file(path)
        if observed_sha != row.get("sha256") or path.stat().st_size != int(row.get("size_bytes", -1)):
            raise RuntimeError(f"private search mount member changed after seal: {path.name}")
        if set(row.get("keys", [])) != PRIVATE_MOUNT_KEYS:
            raise RuntimeError(f"private search mount key audit mismatch: {path.name}")
        member_identity_rows.append({"name": path.name, "sha256": observed_sha})
        load_private_search_mount(
            private_mount_root,
            path.name.split("_", 1)[0],
            expected_manifest=manifest,
        )
    if (
        int(label_basis.get("private_member_count", -1)) != len(expected_names)
        or label_basis.get("private_member_identity") != canonical_json_hash(member_identity_rows)
    ):
        raise RuntimeError("private search mount aggregate identity mismatch")
    return manifest


def seal_global_label_roles(
    *,
    result_root: Path,
    evaluator_contract_path: Path,
    private_mount_root: Path,
) -> dict[str, Any]:
    manifest_path = result_root / "TC_LPMSO_H5_GLOBAL_LABEL_ROLE_MANIFEST.json"
    audit_path = result_root / "TC_LPMSO_H5_PRIVATE_SEARCH_MOUNT_AUDIT.json"
    members = sorted(private_mount_root.glob("*_global_search_pool.npz"))
    expected_names = {f"{dataset}_global_search_pool.npz" for dataset in TARGET_NODE_COUNTS}
    if {path.name for path in members} != expected_names or len(list(private_mount_root.iterdir())) != len(expected_names):
        raise RuntimeError("private search mount must contain exactly one member per dataset")
    basis = {
        "manifest_sha256": sha256_file(manifest_path),
        "private_mount_audit_sha256": sha256_file(audit_path),
        "evaluator_contract_sha256": sha256_file(evaluator_contract_path),
        "private_member_count": len(members),
        "private_member_identity": canonical_json_hash(
            [{"name": path.name, "sha256": sha256_file(path)} for path in members]
        ),
        "one_global_search_pool_per_dataset": True,
        "one_h5_locked_report_per_dataset": True,
        "sealed_before_any_h5_evaluation": True,
        "training_process_mounts_labels": False,
        "search_evaluator_mount_contains_locked_report": False,
        "future_s_eval_seeds_used": False,
    }
    payload = {
        "schema_version": "metachorus.v53.m4a-h5.global-label-role-seal.v1",
        "status": "SEALED_BEFORE_H5_EVALUATION",
        "basis": basis,
        "identity": canonical_json_hash(basis),
    }
    atomic_write_json(result_root / "TC_LPMSO_H5_GLOBAL_LABEL_ROLE_SEAL.json", payload)
    return payload
