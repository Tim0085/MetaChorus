"""Post-training search and preliminary evaluators with explicit label roles."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, f1_score, normalized_mutual_info_score

from metachorus_v53.audit.common import canonical_json_hash, read_json, sha256_file

from .constants import (
    CLASSIFIER_CONTRACT,
    CLASSIFICATION_SPLIT_SEEDS,
    DATASET_NAMES,
    KMEANS_CONTRACT,
    PRELIM_KMEANS_SEEDS,
    SEARCH_KMEANS_SEEDS,
)
from .dataset_specs import DATASET_SPECS
from .labels import PRIVATE_MOUNT_KEYS, _load_labels, partition_roles
from .protocol import metric_values


def _load_search_mount(root: Path, dataset: str, manifest_row: Mapping[str, Any]) -> dict[str, np.ndarray]:
    source = root / f"{dataset}_global_search_pool.npz"
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(source)
    with np.load(source, allow_pickle=False) as payload:
        if set(payload.files) != set(PRIVATE_MOUNT_KEYS):
            raise ValueError("private search mount keys are not exact")
        values = {key: np.asarray(payload[key]).copy() for key in payload.files}
    pool = np.asarray(values["global_search_pool_indices"], dtype=np.int64)
    labels = np.asarray(values["global_search_pool_labels"], dtype=np.int64)
    seeds = np.asarray(values["split_seeds"], dtype=np.uint32)
    train = np.asarray(values["search_train_pool_positions"], dtype=np.int64)
    validation = np.asarray(values["search_validation_pool_positions"], dtype=np.int64)
    if (
        pool.ndim != 1
        or labels.shape != pool.shape
        or seeds.shape != (5,)
        or tuple(map(int, seeds)) != CLASSIFICATION_SPLIT_SEEDS[dataset]
        or train.ndim != 2
        or validation.ndim != 2
        or train.shape[0] != 5
        or validation.shape[0] != 5
        or hashlib.sha256(np.asarray(pool, dtype="<i8").tobytes(order="C")).hexdigest()
        != manifest_row["GLOBAL_SEARCH_POOL"]["indices_sha256"]
    ):
        raise ValueError("private search mount identity mismatch")
    for split in range(5):
        if (
            train.shape[1] + validation.shape[1] != pool.size
            or train[split].size == 0
            or validation[split].size == 0
            or train[split].min() < 0
            or validation[split].min() < 0
            or train[split].max() >= pool.size
            or validation[split].max() >= pool.size
            or np.intersect1d(train[split], validation[split]).size
            or np.union1d(train[split], validation[split]).size != pool.size
        ):
            raise ValueError("private search split is not an exact pool partition")
    return {
        "pool_indices": pool,
        "pool_labels": labels,
        "split_seeds": seeds,
        "train_positions": train,
        "validation_positions": validation,
    }


def _classifier() -> LogisticRegression:
    return LogisticRegression(
        C=float(CLASSIFIER_CONTRACT["C"]),
        penalty=str(CLASSIFIER_CONTRACT["penalty"]),
        solver=str(CLASSIFIER_CONTRACT["solver"]),
        max_iter=int(CLASSIFIER_CONTRACT["max_iter"]),
        fit_intercept=bool(CLASSIFIER_CONTRACT["fit_intercept"]),
        class_weight=CLASSIFIER_CONTRACT["class_weight"],
    )


def evaluate_search_representation(
    features: np.ndarray,
    *,
    dataset: str,
    search_mount: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    values = np.asarray(features, dtype=np.float64)
    pool = np.asarray(search_mount["pool_indices"], dtype=np.int64)
    labels = np.asarray(search_mount["pool_labels"], dtype=np.int64)
    train = np.asarray(search_mount["train_positions"], dtype=np.int64)
    validation = np.asarray(search_mount["validation_positions"], dtype=np.int64)
    if (
        dataset not in DATASET_SPECS
        or values.ndim != 2
        or values.shape[0] != DATASET_SPECS[dataset].label_count
        or not np.isfinite(values).all()
        or labels.shape != pool.shape
    ):
        raise ValueError("search representation/role contract mismatch")
    classification_rows: list[dict[str, Any]] = []
    for split_index in range(5):
        model = _classifier()
        model.fit(values[pool[train[split_index]]], labels[train[split_index]])
        predicted = model.predict(values[pool[validation[split_index]]])
        classification_rows.append(
            {
                "split_index": split_index,
                "split_seed": int(CLASSIFICATION_SPLIT_SEEDS[dataset][split_index]),
                "Micro_F1": float(f1_score(labels[validation[split_index]], predicted, average="micro")),
                "Macro_F1": float(f1_score(labels[validation[split_index]], predicted, average="macro")),
                "fit_role": "GLOBAL_SEARCH_POOL.SEARCH_TRAIN",
                "metric_role": "GLOBAL_SEARCH_POOL.SEARCH_VALIDATION",
                "prelim_locked_report_access": False,
            }
        )
    cluster_rows, cluster_summary = _evaluate_clustering(
        values,
        dataset=dataset,
        metric_indices=pool,
        metric_labels=labels,
        seeds=SEARCH_KMEANS_SEEDS,
        metric_role="GLOBAL_SEARCH_POOL",
    )
    micro = float(np.mean([row["Micro_F1"] for row in classification_rows]))
    macro = float(np.mean([row["Macro_F1"] for row in classification_rows]))
    nmi = float(np.mean([row["NMI"] for row in cluster_rows]))
    ari = float(np.mean([row["ARI"] for row in cluster_rows]))
    return {
        "status": "PASS",
        "dataset": dataset,
        "classification_repeats": classification_rows,
        "clustering_repeats": cluster_rows,
        "assignment_stability_mean": float(cluster_summary["assignment_stability_mean"]),
        "metrics": metric_values(micro, macro, nmi, ari),
        "prelim_locked_report_access": False,
        "raw_labels_persisted": False,
    }


def open_search_mount(
    *,
    role_manifest_path: str | Path,
    role_seal_path: str | Path,
    private_mount_audit_path: str | Path,
    private_search_mount_root: str | Path,
    dataset: str,
) -> dict[str, np.ndarray]:
    """Open only the search-safe 20% mount; never a raw label source."""

    manifest = read_json(role_manifest_path)
    seal = read_json(role_seal_path)
    audit = read_json(private_mount_audit_path)
    if manifest.get("status") != "SEALED_INPUT" or manifest.get("basis", {}).get("training_process_mounts_labels") is not False:
        raise ValueError("search evaluator role manifest is not sealed")
    if (
        seal.get("status") != "SEALED_BEFORE_SEARCH_TRAINING"
        or seal.get("identity") != canonical_json_hash(seal.get("basis", {}))
        or seal.get("basis", {}).get("role_manifest_identity") != manifest.get("identity")
        or audit.get("status") != "PASS"
        or audit.get("identity") != seal.get("basis", {}).get("private_mount_audit_identity")
    ):
        raise ValueError("search evaluator role/audit identity chain is invalid")
    rows = [row for row in manifest.get("basis", {}).get("datasets", []) if row.get("dataset") == dataset]
    if len(rows) != 1:
        raise ValueError("search evaluator dataset role is missing or duplicated")
    root = Path(private_search_mount_root).resolve(strict=True)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("private search mount root is not regular")
    members = [row for row in audit.get("basis", {}).get("members", []) if row.get("dataset") == dataset]
    source = root / f"{dataset}_global_search_pool.npz"
    if (
        len(members) != 1
        or members[0].get("member_name") != source.name
        or source.stat().st_size != int(members[0].get("size_bytes", -1))
        or sha256_file(source) != members[0].get("sha256")
        or set(members[0].get("keys", [])) != set(PRIVATE_MOUNT_KEYS)
    ):
        raise ValueError("search-safe private mount member changed")
    return _load_search_mount(root, dataset, rows[0])


def _evaluate_clustering(
    features: np.ndarray,
    *,
    dataset: str,
    metric_indices: np.ndarray,
    metric_labels: np.ndarray,
    seeds: Sequence[int],
    metric_role: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    values = np.asarray(features, dtype=np.float64)
    indices = np.asarray(metric_indices, dtype=np.int64)
    labels = np.asarray(metric_labels, dtype=np.int64)
    expected = SEARCH_KMEANS_SEEDS if metric_role == "GLOBAL_SEARCH_POOL" else PRELIM_KMEANS_SEEDS
    if (
        dataset not in DATASET_SPECS
        or tuple(map(int, seeds)) != expected
        or indices.ndim != 1
        or labels.shape != indices.shape
        or indices.size == 0
        or indices.min() < 0
        or indices.max() >= values.shape[0]
    ):
        raise ValueError("KMeans role/seed contract mismatch")
    rows: list[dict[str, Any]] = []
    assignments: list[np.ndarray] = []
    for repeat, seed in enumerate(seeds):
        assignment = KMeans(
            n_clusters=DATASET_SPECS[dataset].class_count,
            n_init=int(KMEANS_CONTRACT["n_init"]),
            algorithm=str(KMEANS_CONTRACT["algorithm"]),
            random_state=int(seed),
        ).fit_predict(values)
        assignment = np.asarray(assignment, dtype=np.int64)
        assignments.append(assignment)
        observed = assignment[indices]
        rows.append(
            {
                "repeat_index": repeat,
                "kmeans_seed": int(seed),
                "NMI": float(normalized_mutual_info_score(labels, observed)),
                "ARI": float(adjusted_rand_score(labels, observed)),
                "assignment_sha256": hashlib.sha256(np.asarray(assignment, dtype="<i4").tobytes(order="C")).hexdigest(),
                "fit_role": "ALL_TARGET_NODES_UNLABELED",
                "metric_role": metric_role,
                "labels_used_during_fit": False,
                "fit_node_count": int(values.shape[0]),
                "metric_node_count": int(indices.size),
                "status": "PASS",
            }
        )
    pairwise = [
        adjusted_rand_score(assignments[left], assignments[right])
        for left in range(len(assignments))
        for right in range(left + 1, len(assignments))
    ]
    return rows, {
        "NMI": float(np.mean([row["NMI"] for row in rows])),
        "ARI": float(np.mean([row["ARI"] for row in rows])),
        "assignment_stability_mean": float(np.mean(pairwise)),
        "assignment_stability_p10": float(np.percentile(pairwise, 10)),
    }


class PreliminaryLabelSession:
    """Load each supervised source once after the physical run registry is sealed."""

    def __init__(
        self,
        *,
        role_manifest_path: str | Path,
        private_search_mount_root: str | Path,
        label_sources: Mapping[str, str | Path],
        label_source_roots: Mapping[str, str | Path],
        training_registry_seal: Mapping[str, Any],
    ) -> None:
        if (
            training_registry_seal.get("status") != "SEALED_BEFORE_PRELIMINARY_EVALUATION"
            or training_registry_seal.get("basis", {}).get("all_status_pass") is not True
            or training_registry_seal.get("basis", {}).get("labels_mounted") is not False
            or training_registry_seal.get("basis", {}).get("preliminary_evaluation_started") is not False
        ):
            raise RuntimeError("preliminary labels forbidden before exact training registry seal")
        self.manifest = read_json(role_manifest_path)
        if (
            self.manifest.get("status") != "SEALED_INPUT"
            or self.manifest.get("identity") != canonical_json_hash(self.manifest.get("basis", {}))
        ):
            raise ValueError("label role manifest is not sealed")
        self.private_root = Path(private_search_mount_root).resolve(strict=True)
        self.sources = {key: Path(value) for key, value in label_sources.items()}
        self.source_roots = {key: Path(value).resolve(strict=True) for key, value in label_source_roots.items()}
        declared_active = training_registry_seal.get("basis", {}).get("active_datasets", DATASET_NAMES)
        self.active_datasets = tuple(str(name) for name in declared_active)
        if (
            not self.active_datasets
            or any(name not in DATASET_NAMES for name in self.active_datasets)
            or set(self.sources) != set(self.source_roots)
            or not set(self.active_datasets).issubset(self.sources)
            or not set(self.sources).issubset(DATASET_NAMES)
        ):
            raise ValueError("preliminary evaluator label sources differ from the active dataset subset")
        self._cache: dict[str, dict[str, np.ndarray]] = {}
        self.access_count = 0
        self.per_dataset_access_count = {name: 0 for name in DATASET_NAMES}

    def open_dataset(self, dataset: str) -> dict[str, np.ndarray]:
        if dataset not in self.active_datasets:
            raise RuntimeError("preliminary evaluator cannot open a budget-blocked dataset")
        if dataset in self._cache:
            return self._cache[dataset]
        source = self.sources[dataset]
        root = self.source_roots[dataset]
        if source.is_symlink() or not source.is_file() or root.is_symlink() or not root.is_dir():
            raise ValueError("supervised source/root is not regular")
        try:
            source.resolve(strict=True).relative_to(root)
        except ValueError as error:
            raise ValueError("supervised source escapes authorized root") from error
        rows = [row for row in self.manifest["basis"]["datasets"] if row["dataset"] == dataset]
        if len(rows) != 1 or source.name != rows[0]["label_source_name"] or sha256_file(source) != rows[0]["label_source_sha256"]:
            raise ValueError("supervised source differs from sealed role manifest")
        labels = _load_labels(dataset, source)
        roles = partition_roles(
            labels,
            pool_seed=int(rows[0]["global_search_pool_seed"]),
            split_seeds=CLASSIFICATION_SPLIT_SEEDS[dataset],
        )
        search = _load_search_mount(self.private_root, dataset, rows[0])
        pool = np.asarray(roles["global_search_pool"], dtype=np.int64)
        locked = np.asarray(roles["prelim_locked_report"], dtype=np.int64)
        if not np.array_equal(pool, search["pool_indices"]) or not np.array_equal(labels[pool], search["pool_labels"]):
            raise ValueError("search mount and role manifest do not share the same pool")
        value = {
            "pool_indices": pool,
            "pool_labels": labels[pool],
            "locked_indices": locked,
            "locked_labels": labels[locked],
        }
        self.access_count += 1
        self.per_dataset_access_count[dataset] += 1
        self._cache[dataset] = value
        return value


def evaluate_preliminary_representation(
    features: np.ndarray,
    *,
    dataset: str,
    roles: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    values = np.asarray(features, dtype=np.float64)
    pool = np.asarray(roles["pool_indices"], dtype=np.int64)
    locked = np.asarray(roles["locked_indices"], dtype=np.int64)
    pool_labels = np.asarray(roles["pool_labels"], dtype=np.int64)
    locked_labels = np.asarray(roles["locked_labels"], dtype=np.int64)
    if values.shape[0] != pool.size + locked.size or np.intersect1d(pool, locked).size:
        raise ValueError("preliminary role partition is not an exact node cover")
    model = _classifier()
    model.fit(values[pool], pool_labels)
    predicted = model.predict(values[locked])
    micro = float(f1_score(locked_labels, predicted, average="micro"))
    macro = float(f1_score(locked_labels, predicted, average="macro"))
    cluster_rows, summary = _evaluate_clustering(
        values,
        dataset=dataset,
        metric_indices=locked,
        metric_labels=locked_labels,
        seeds=PRELIM_KMEANS_SEEDS,
        metric_role="PRELIM_LOCKED_REPORT",
    )
    metrics = metric_values(micro, macro, float(summary["NMI"]), float(summary["ARI"]))
    return {
        "status": "PASS",
        "dataset": dataset,
        "metrics": metrics,
        "clustering_repeats": cluster_rows,
        "assignment_stability_mean": summary["assignment_stability_mean"],
        "assignment_stability_p10": summary["assignment_stability_p10"],
        "classification_fit_role": "GLOBAL_SEARCH_POOL",
        "classification_metric_role": "PRELIM_LOCKED_REPORT",
        "clustering_fit_role": "ALL_TARGET_NODES_UNLABELED",
        "clustering_metric_role": "PRELIM_LOCKED_REPORT",
        "raw_labels_persisted": False,
        "confirmatory_claim_allowed": False,
    }


__all__ = [
    "evaluate_search_representation",
    "open_search_mount",
    "PreliminaryLabelSession",
    "evaluate_preliminary_representation",
]
