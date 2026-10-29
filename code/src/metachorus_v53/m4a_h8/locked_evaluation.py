"""Independent post-training evaluator for the frozen H5 locked report."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, f1_score, normalized_mutual_info_score

from metachorus_v53.audit.common import canonical_json_hash, read_json, sha256_file
from metachorus_v53.m4a_h5.labels import (
    global_search_roles,
    load_acm_labels,
    load_freebase_labels,
    load_private_search_mount,
)

from .common import (
    append_jsonl_once,
    envelope,
    read_csv,
    validate_envelope,
    write_csv_once,
    write_json_once,
)
from .constants import (
    BOOTSTRAP_RESAMPLES,
    CLASSIFIER_CONTRACT,
    DEVELOPMENT_MODEL_SEEDS,
    EXPECTED_SCIENCE_RUNS,
    FROZEN_CLASS_COUNTS,
    H5_BANK_SEEDS,
    H5_CLASSIFICATION_SPLIT_SEEDS,
    H5_CLUSTER_SEARCH_SEEDS,
    KMEANS_CONTRACT,
    PARENT_PACKAGE_SHA256,
    ROLE_MANIFEST_SHA256,
    ROLE_SEAL_SHA256,
    S_EVAL_MODEL_SEEDS,
    TASKBOOK_SHA256,
)
from .stages import close_science_registry


CLASSIFICATION_FIELDS = (
    "physical_run_id",
    "dataset",
    "role",
    "candidate_set_id",
    "candidate_set_hash",
    "model_seed",
    "Micro_F1",
    "Macro_F1",
    "F_cls",
    "fit_role",
    "test_role",
    "classifier_fit_count",
    "locked_access_count_for_dataset",
    "status",
)

CLUSTERING_LONG_FIELDS = (
    "physical_run_id",
    "dataset",
    "candidate_set_hash",
    "model_seed",
    "kmeans_repeat_index",
    "kmeans_seed",
    "NMI",
    "ARI",
    "assignment_sha256",
    "fit_role",
    "metric_role",
    "labels_used_during_fit",
    "fit_node_count",
    "metric_node_count",
    "status",
)

CLUSTERING_SUMMARY_FIELDS = (
    "physical_run_id",
    "dataset",
    "role",
    "candidate_set_id",
    "candidate_set_hash",
    "model_seed",
    "NMI",
    "ARI",
    "F_clu",
    "kmeans_repeat_count",
    "assignment_stability_mean",
    "assignment_stability_p10",
    "assignment_hash_count",
    "fit_role",
    "metric_role",
    "status",
)

LOCKED_METRIC_FIELDS = (
    "physical_run_id",
    "dataset",
    "role",
    "candidate_set_id",
    "candidate_set_hash",
    "model_seed",
    "Micro_F1",
    "Macro_F1",
    "NMI",
    "ARI",
    "F_cls",
    "F_clu",
    "F_joint",
    "gpu_uuid",
    "runtime_domain_identity",
    "training_seconds",
    "status",
)


def _indices_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes(order="C")).hexdigest()


def _canonical_seed(index: int, collision: int = 0) -> int:
    payload = {
        "namespace": "TC_LPMSO_H8_LOCKED_KMEANS",
        "taskbook_sha256": TASKBOOK_SHA256,
        "parent_package_sha256": PARENT_PACKAGE_SHA256,
        "h5_role_seal_sha256": ROLE_SEAL_SHA256,
        "index": int(index),
        "collision": int(collision),
    }
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).digest()
    return int.from_bytes(digest[:4], "big", signed=False)


def derive_kmeans_seeds() -> tuple[int, ...]:
    forbidden = set(S_EVAL_MODEL_SEEDS) | set(DEVELOPMENT_MODEL_SEEDS)
    forbidden |= set(H5_CLUSTER_SEARCH_SEEDS) | set(H5_CLASSIFICATION_SPLIT_SEEDS)
    forbidden |= set(H5_BANK_SEEDS)
    output: list[int] = []
    for index in range(int(KMEANS_CONTRACT["repeat_count"])):
        collision = 0
        value = _canonical_seed(index, collision)
        while value in forbidden or value in output:
            collision += 1
            value = _canonical_seed(index, collision)
        output.append(value)
    if len(output) != 20 or len(set(output)) != 20:
        raise RuntimeError("H8 KMeans seed derivation failed")
    return tuple(output)


def _validate_training_gate(result_root: Path) -> tuple[list[dict[str, str]], dict[str, Any]]:
    seal = close_science_registry(result_root)
    seal = validate_envelope(
        seal,
        schema_version="metachorus.v53.m4a-h8.training-registry-seal.v1",
        status="SEALED_BEFORE_LOCKED_EVALUATION",
    )
    rows = read_csv(result_root / "TC_LPMSO_H8_RUN_REGISTRY.csv")
    artifacts = read_csv(result_root / "TC_LPMSO_H8_CHECKPOINT_REGISTRY.csv")
    if (
        len(rows) != EXPECTED_SCIENCE_RUNS
        or int(seal["basis"].get("run_count", -1)) != EXPECTED_SCIENCE_RUNS
        or seal["basis"].get("all_status_pass") is not True
        or seal["basis"].get("all_model_seeds_consumed") is not True
        or seal["basis"].get("labels_mounted") is not False
        or seal["basis"].get("h5_locked_report_access") is not False
        or seal["basis"].get("locked_evaluation_started") is not False
        or sha256_file(result_root / "TC_LPMSO_H8_RUN_REGISTRY.csv")
        != seal["basis"].get("run_registry_sha256")
        or sha256_file(result_root / "TC_LPMSO_H8_CHECKPOINT_REGISTRY.csv")
        != seal["basis"].get("artifact_registry_sha256")
    ):
        raise RuntimeError("H8 locked evaluation forbidden before exact 50-run registry seal")
    representations = [row for row in artifacts if row["artifact_kind"] == "REPRESENTATION"]
    if len(representations) != EXPECTED_SCIENCE_RUNS:
        raise RuntimeError("H8 representation registry is incomplete")
    return rows, {str(row["physical_run_id"]): row for row in representations}


def prepare_locked_evaluator(result_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    training_seal = close_science_registry(result_root)
    role_seal = validate_envelope(
        read_json(result_root / "TC_LPMSO_H8_LOCKED_ROLE_SEAL.json"),
        schema_version="metachorus.v53.m4a-h8.locked-role-seal.v1",
        status="SEALED_BEFORE_ANY_H8_TRAINING",
    )
    seeds = derive_kmeans_seeds()
    seed_basis = {
        "namespace": "TC_LPMSO_H8_LOCKED_KMEANS",
        "taskbook_sha256": TASKBOOK_SHA256,
        "parent_package_sha256": PARENT_PACKAGE_SHA256,
        "h5_role_seal_sha256": ROLE_SEAL_SHA256,
        "seeds": list(seeds),
        "seed_count": len(seeds),
        "unique": True,
        "disjoint_from_model_search_split_bank_seeds": True,
        "derived_before_locked_evaluation": True,
    }
    seed_seal = envelope(
        "metachorus.v53.m4a-h8.kmeans-seed-seal.v1",
        "SEALED_BEFORE_LOCKED_EVALUATION",
        seed_basis,
    )
    write_json_once(
        result_root / "TC_LPMSO_H8_LOCKED_KMEANS_SEED_NAMESPACE.json",
        envelope(
            "metachorus.v53.m4a-h8.kmeans-seed-namespace.v1",
            "SEALED_BEFORE_LOCKED_EVALUATION",
            seed_basis,
        ),
    )
    write_json_once(result_root / "TC_LPMSO_H8_LOCKED_KMEANS_SEED_SEAL.json", seed_seal)
    evaluator_basis = {
        "training_registry_seal_identity": training_seal["identity"],
        "locked_role_seal_identity": role_seal["identity"],
        "kmeans_seed_seal_identity": seed_seal["identity"],
        "classification_contract": CLASSIFIER_CONTRACT,
        "classification_fit_role": "GLOBAL_SEARCH_POOL",
        "classification_test_role": "H5_LOCKED_REPORT",
        "clustering_contract": KMEANS_CONTRACT,
        "clustering_fit_role": "ALL_TARGET_NODES_UNLABELED",
        "clustering_metric_role": "H5_LOCKED_REPORT",
        "locked_access_count_before_evaluation": 0,
        "hyperparameter_search": False,
        "training_process_mounted_labels": False,
    }
    evaluator_seal = envelope(
        "metachorus.v53.m4a-h8.locked-evaluator-seal.v1",
        "SEALED_BEFORE_FIRST_LOCKED_ACCESS",
        evaluator_basis,
    )
    write_json_once(result_root / "TC_LPMSO_H8_LOCKED_EVALUATOR_SEAL.json", evaluator_seal)
    write_json_once(
        result_root / "M4A_H8_LOCKED_ACCESS_INTENT.json",
        envelope(
            "metachorus.v53.m4a-h8.locked-access-intent.v1",
            "AUTHORIZED_AFTER_TRAINING_SEAL",
            {
                "training_registry_seal_identity": training_seal["identity"],
                "locked_evaluator_seal_identity": evaluator_seal["identity"],
                "authorized_datasets": ["acm", "freebase"],
                "authorized_accesses_per_dataset": 1,
                "access_count_before_evaluation": 0,
                "classification_fit_role": "GLOBAL_SEARCH_POOL",
                "classification_test_role": "H5_LOCKED_REPORT",
                "clustering_fit_role": "ALL_TARGET_NODES_UNLABELED",
                "clustering_metric_role": "H5_LOCKED_REPORT",
                "raw_labels_may_not_be_persisted": True,
            },
        ),
    )
    return seed_seal, evaluator_seal


class LockedLabelSession:
    """Read each locked dataset once, only after the training registry is sealed."""

    def __init__(
        self,
        *,
        result_root: Path,
        h5_result_root: Path,
        private_search_mount_root: Path,
        label_sources: Mapping[str, Path],
        label_source_roots: Mapping[str, Path],
    ):
        self.result_root = result_root
        if (
            h5_result_root.is_symlink()
            or not h5_result_root.is_dir()
            or private_search_mount_root.is_symlink()
            or not private_search_mount_root.is_dir()
        ):
            raise ValueError("H8 locked evaluator roots must be regular directories")
        self.h5_result_root = h5_result_root.resolve(strict=True)
        self.private_search_mount_root = private_search_mount_root.resolve(strict=True)
        self.label_sources = {key: Path(value) for key, value in label_sources.items()}
        self.label_source_roots: dict[str, Path] = {}
        for key, value in label_source_roots.items():
            root = Path(value)
            if root.is_symlink() or not root.is_dir():
                raise ValueError("H8 locked label source root is not an authorized directory")
            self.label_source_roots[key] = root.resolve(strict=True)
        self.training_rows, self.representations = _validate_training_gate(result_root)
        prepare_locked_evaluator(result_root)
        self.manifest_path = self.h5_result_root / "TC_LPMSO_H5_GLOBAL_LABEL_ROLE_MANIFEST.json"
        if (
            self.manifest_path.is_symlink()
            or not self.manifest_path.is_file()
            or sha256_file(self.manifest_path) != ROLE_MANIFEST_SHA256
        ):
            raise ValueError("H8 H5 role manifest hash mismatch")
        self.manifest = read_json(self.manifest_path)
        self._cache: dict[str, dict[str, np.ndarray]] = {}
        self.access_count = 0
        self.per_dataset_access_count = {"acm": 0, "freebase": 0}

    def _manifest_row(self, dataset: str) -> Mapping[str, Any]:
        rows = [
            row for row in self.manifest.get("datasets", []) if row.get("dataset") == dataset
        ]
        if len(rows) != 1:
            raise ValueError("H8 role manifest dataset row mismatch")
        return rows[0]

    def open_dataset(self, dataset: str) -> dict[str, np.ndarray]:
        if dataset in self._cache:
            return self._cache[dataset]
        if dataset not in {"acm", "freebase"}:
            raise ValueError(dataset)
        source = self.label_sources.get(dataset)
        if source is None or source.is_symlink() or not source.is_file():
            raise FileNotFoundError(f"H8 locked label source unavailable for {dataset}")
        allowed_root = self.label_source_roots.get(dataset)
        if allowed_root is None or allowed_root.is_symlink() or not allowed_root.is_dir():
            raise ValueError("H8 locked label source root is not an authorized directory")
        try:
            source.resolve(strict=True).relative_to(allowed_root)
        except ValueError as error:
            raise ValueError("H8 locked label source escapes its authorized root") from error
        manifest_row = self._manifest_row(dataset)
        if (
            source.name != manifest_row.get("label_source_name")
            or sha256_file(source) != manifest_row.get("label_source_sha256")
        ):
            raise ValueError("H8 locked label source identity mismatch")
        search = load_private_search_mount(
            self.private_search_mount_root,
            dataset,
            expected_manifest=self.manifest,
        )
        labels = load_acm_labels(source) if dataset == "acm" else load_freebase_labels(source)
        seeds = self.manifest["seed_bank"]["per_dataset"][dataset]
        roles = global_search_roles(
            labels,
            pool_seed=int(seeds["global_search_pool_seed"]),
            split_seeds=[int(value) for value in seeds["within_pool_classification_split_seeds"]],
        )
        pool = np.asarray(roles["global_search_pool"], dtype=np.int64)
        locked = np.asarray(roles["h5_locked_report"], dtype=np.int64)
        if (
            not np.array_equal(pool, search["global_search_pool_indices"])
            or not np.array_equal(labels[pool], search["global_search_pool_labels"])
            or _indices_sha256(pool)
            != manifest_row["GLOBAL_SEARCH_POOL"]["indices_sha256"]
            or _indices_sha256(locked)
            != manifest_row["H5_LOCKED_REPORT"]["indices_sha256"]
            or np.intersect1d(pool, locked).size
            or np.union1d(pool, locked).size != labels.size
        ):
            raise ValueError("H8 locked/search role partition identity mismatch")
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


def _representation(
    result_root: Path, artifact: Mapping[str, Any], *, expected_run_id: str
) -> np.ndarray:
    if artifact.get("physical_run_id") != expected_run_id:
        raise ValueError("H8 representation registry identity mismatch")
    relative = str(artifact["relative_path"])
    if "\\" in relative or relative.startswith("/") or ".." in Path(relative).parts:
        raise ValueError("H8 representation registry path is unsafe")
    path = result_root.joinpath(*relative.split("/"))
    root = result_root.resolve(strict=True)
    if path.is_symlink() or not path.is_file():
        raise ValueError("H8 representation is not a regular file")
    try:
        path.resolve(strict=True).relative_to(root)
    except ValueError as error:
        raise ValueError("H8 representation escapes result root") from error
    if sha256_file(path) != artifact["sha256"] or path.stat().st_size != int(
        artifact["size_bytes"]
    ):
        raise ValueError("H8 representation hash/size mismatch")
    array = np.load(path, allow_pickle=False)
    if array.ndim != 2 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("H8 representation is not finite two-dimensional data")
    return np.asarray(array, dtype=np.float64)


def evaluate_locked_classification(
    features: np.ndarray,
    *,
    pool_indices: np.ndarray,
    pool_labels: np.ndarray,
    locked_indices: np.ndarray,
    locked_labels: np.ndarray,
) -> dict[str, float | int | str]:
    """Fit once on GLOBAL_SEARCH_POOL and score only H5_LOCKED_REPORT."""

    values = np.asarray(features, dtype=np.float64)
    pool = np.asarray(pool_indices, dtype=np.int64)
    locked = np.asarray(locked_indices, dtype=np.int64)
    pool_truth = np.asarray(pool_labels, dtype=np.int64)
    locked_truth = np.asarray(locked_labels, dtype=np.int64)
    if (
        values.ndim != 2
        or values.size == 0
        or not np.isfinite(values).all()
        or pool.size == 0
        or locked.size == 0
        or np.intersect1d(pool, locked).size
        or pool.min() < 0
        or locked.min() < 0
        or pool.max() >= values.shape[0]
        or locked.max() >= values.shape[0]
        or pool_truth.ndim != 1
        or locked_truth.ndim != 1
        or pool_truth.size != pool.size
        or locked_truth.size != locked.size
    ):
        raise ValueError("H8 locked classification partition is invalid")
    classifier = LogisticRegression(
        C=float(CLASSIFIER_CONTRACT["C"]),
        penalty=str(CLASSIFIER_CONTRACT["penalty"]),
        solver=str(CLASSIFIER_CONTRACT["solver"]),
        max_iter=int(CLASSIFIER_CONTRACT["max_iter"]),
        fit_intercept=bool(CLASSIFIER_CONTRACT["fit_intercept"]),
        class_weight=CLASSIFIER_CONTRACT["class_weight"],
    )
    classifier.fit(values[pool], pool_truth)
    predicted = classifier.predict(values[locked])
    micro = float(f1_score(locked_truth, predicted, average="micro"))
    macro = float(f1_score(locked_truth, predicted, average="macro"))
    return {
        "Micro_F1": micro,
        "Macro_F1": macro,
        "F_cls": (micro + macro) / 2.0,
        "classifier_fit_count": 1,
        "fit_role": "GLOBAL_SEARCH_POOL",
        "test_role": "H5_LOCKED_REPORT",
    }


def evaluate_locked_clustering(
    features: np.ndarray,
    *,
    dataset: str,
    locked_indices: np.ndarray,
    locked_labels: np.ndarray,
    kmeans_seeds: Sequence[int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fit KMeans on every unlabeled target node and score only locked nodes."""

    values = np.asarray(features, dtype=np.float64)
    locked = np.asarray(locked_indices, dtype=np.int64)
    if dataset not in FROZEN_CLASS_COUNTS:
        raise ValueError("H8 locked clustering dataset is not frozen")
    class_count = int(FROZEN_CLASS_COUNTS[dataset])
    truth = np.asarray(locked_labels, dtype=np.int64)
    expected_kmeans_seeds = derive_kmeans_seeds()
    if (
        values.ndim != 2
        or values.size == 0
        or not np.isfinite(values).all()
        or locked.size == 0
        or locked.min() < 0
        or locked.max() >= values.shape[0]
        or int(class_count) <= 1
        or truth.ndim != 1
        or truth.size != locked.size
        or len(kmeans_seeds) != int(KMEANS_CONTRACT["repeat_count"])
        or len(set(int(seed) for seed in kmeans_seeds)) != len(kmeans_seeds)
        or tuple(int(seed) for seed in kmeans_seeds) != expected_kmeans_seeds
    ):
        raise ValueError("H8 locked clustering contract is invalid")
    rows: list[dict[str, Any]] = []
    assignments: list[np.ndarray] = []
    for repeat_index, kmeans_seed in enumerate(kmeans_seeds):
        assignment = KMeans(
            n_clusters=int(class_count),
            n_init=int(KMEANS_CONTRACT["n_init"]),
            algorithm=str(KMEANS_CONTRACT["algorithm"]),
            random_state=int(kmeans_seed),
        ).fit_predict(values)
        assignment = np.asarray(assignment, dtype=np.int64)
        assignments.append(assignment)
        observed = assignment[locked]
        nmi = float(normalized_mutual_info_score(truth, observed))
        ari = float(adjusted_rand_score(truth, observed))
        rows.append(
            {
                "kmeans_repeat_index": repeat_index,
                "kmeans_seed": int(kmeans_seed),
                "NMI": nmi,
                "ARI": ari,
                "assignment_sha256": hashlib.sha256(
                    np.asarray(assignment, dtype="<i4").tobytes(order="C")
                ).hexdigest(),
                "fit_role": "ALL_TARGET_NODES_UNLABELED",
                "metric_role": "H5_LOCKED_REPORT",
                "labels_used_during_fit": False,
                "fit_node_count": int(values.shape[0]),
                "metric_node_count": int(locked.size),
                "status": "PASS",
            }
        )
    pairwise_stability = np.asarray(
        [
            adjusted_rand_score(assignments[left], assignments[right])
            for left in range(len(assignments))
            for right in range(left + 1, len(assignments))
        ],
        dtype=np.float64,
    )
    nmi_mean = float(np.mean([row["NMI"] for row in rows]))
    ari_mean = float(np.mean([row["ARI"] for row in rows]))
    return rows, {
        "NMI": nmi_mean,
        "ARI": ari_mean,
        "F_clu": (nmi_mean + ari_mean) / 2.0,
        "kmeans_repeat_count": len(rows),
        "assignment_stability_mean": float(np.mean(pairwise_stability)),
        "assignment_stability_p10": float(np.percentile(pairwise_stability, 10)),
        "assignment_hash_count": len({row["assignment_sha256"] for row in rows}),
        "fit_role": "ALL_TARGET_NODES_UNLABELED",
        "metric_role": "H5_LOCKED_REPORT",
        "status": "PASS",
    }


def run_locked_evaluation(
    *,
    result_root: Path,
    h5_result_root: Path,
    private_search_mount_root: Path,
    acm_label_source: Path,
    freebase_label_source: Path,
    acm_label_source_root: Path,
    freebase_label_source_root: Path,
) -> dict[str, Any]:
    session = LockedLabelSession(
        result_root=result_root,
        h5_result_root=h5_result_root,
        private_search_mount_root=private_search_mount_root,
        label_sources={"acm": acm_label_source, "freebase": freebase_label_source},
        label_source_roots={
            "acm": acm_label_source_root,
            "freebase": freebase_label_source_root,
        },
    )
    kmeans_seeds = derive_kmeans_seeds()
    classification_rows: list[dict[str, Any]] = []
    clustering_long_rows: list[dict[str, Any]] = []
    clustering_summary_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for row in session.training_rows:
        run_id = str(row["physical_run_id"])
        dataset = str(row["dataset"])
        features = _representation(
            result_root,
            session.representations[run_id],
            expected_run_id=run_id,
        )
        labels = session.open_dataset(dataset)
        if features.shape[0] != labels["pool_indices"].size + labels["locked_indices"].size:
            raise ValueError("H8 representation target-node count mismatch")
        classification = evaluate_locked_classification(
            features,
            pool_indices=labels["pool_indices"],
            pool_labels=labels["pool_labels"],
            locked_indices=labels["locked_indices"],
            locked_labels=labels["locked_labels"],
        )
        micro = float(classification["Micro_F1"])
        macro = float(classification["Macro_F1"])
        f_cls = float(classification["F_cls"])
        classification_rows.append(
            {
                "physical_run_id": run_id,
                "dataset": dataset,
                "role": str(row["role"]),
                "candidate_set_id": str(row["candidate_set_id"]),
                "candidate_set_hash": str(row["candidate_set_hash"]),
                "model_seed": int(row["model_seed"]),
                "Micro_F1": micro,
                "Macro_F1": macro,
                "F_cls": f_cls,
                "fit_role": classification["fit_role"],
                "test_role": classification["test_role"],
                "classifier_fit_count": classification["classifier_fit_count"],
                "locked_access_count_for_dataset": session.per_dataset_access_count[dataset],
                "status": "PASS",
            }
        )
        clustering_rows, clustering_summary = evaluate_locked_clustering(
            features,
            dataset=dataset,
            locked_indices=labels["locked_indices"],
            locked_labels=labels["locked_labels"],
            kmeans_seeds=kmeans_seeds,
        )
        for clustering_row in clustering_rows:
            clustering_long_rows.append(
                {
                    "physical_run_id": run_id,
                    "dataset": dataset,
                    "candidate_set_hash": str(row["candidate_set_hash"]),
                    "model_seed": int(row["model_seed"]),
                    **clustering_row,
                }
            )
        nmi_mean = float(clustering_summary["NMI"])
        ari_mean = float(clustering_summary["ARI"])
        f_clu = float(clustering_summary["F_clu"])
        clustering_summary_rows.append(
            {
                "physical_run_id": run_id,
                "dataset": dataset,
                "role": str(row["role"]),
                "candidate_set_id": str(row["candidate_set_id"]),
                "candidate_set_hash": str(row["candidate_set_hash"]),
                "model_seed": int(row["model_seed"]),
                "NMI": nmi_mean,
                "ARI": ari_mean,
                "F_clu": f_clu,
                "kmeans_repeat_count": clustering_summary["kmeans_repeat_count"],
                "assignment_stability_mean": clustering_summary[
                    "assignment_stability_mean"
                ],
                "assignment_stability_p10": clustering_summary[
                    "assignment_stability_p10"
                ],
                "assignment_hash_count": clustering_summary["assignment_hash_count"],
                "fit_role": clustering_summary["fit_role"],
                "metric_role": clustering_summary["metric_role"],
                "status": clustering_summary["status"],
            }
        )
        metric_rows.append(
            {
                "physical_run_id": run_id,
                "dataset": dataset,
                "role": str(row["role"]),
                "candidate_set_id": str(row["candidate_set_id"]),
                "candidate_set_hash": str(row["candidate_set_hash"]),
                "model_seed": int(row["model_seed"]),
                "Micro_F1": micro,
                "Macro_F1": macro,
                "NMI": nmi_mean,
                "ARI": ari_mean,
                "F_cls": f_cls,
                "F_clu": f_clu,
                "F_joint": (f_cls + f_clu) / 2.0,
                "gpu_uuid": str(row["gpu_uuid"]),
                "runtime_domain_identity": str(row["runtime_domain_identity"]),
                "training_seconds": float(row["model_total_seconds"]),
                "status": "PASS",
            }
        )
    if (
        len(metric_rows) != EXPECTED_SCIENCE_RUNS
        or len(classification_rows) != EXPECTED_SCIENCE_RUNS
        or len(clustering_summary_rows) != EXPECTED_SCIENCE_RUNS
        or len(clustering_long_rows)
        != EXPECTED_SCIENCE_RUNS * int(KMEANS_CONTRACT["repeat_count"])
        or session.access_count != 2
        or session.per_dataset_access_count != {"acm": 1, "freebase": 1}
    ):
        raise RuntimeError("H8 locked evaluation output/access count mismatch")
    write_csv_once(
        result_root / "TC_LPMSO_H8_LOCKED_CLASSIFICATION_METRICS.csv",
        classification_rows,
        fieldnames=CLASSIFICATION_FIELDS,
    )
    write_csv_once(
        result_root / "TC_LPMSO_H8_LOCKED_CLUSTERING_METRICS_LONG.csv",
        clustering_long_rows,
        fieldnames=CLUSTERING_LONG_FIELDS,
    )
    write_csv_once(
        result_root / "TC_LPMSO_H8_LOCKED_CLUSTERING_METRICS_SUMMARY.csv",
        clustering_summary_rows,
        fieldnames=CLUSTERING_SUMMARY_FIELDS,
    )
    write_csv_once(
        result_root / "TC_LPMSO_H8_LOCKED_METRICS_BY_MODEL_SEED.csv",
        metric_rows,
        fieldnames=LOCKED_METRIC_FIELDS,
    )
    access_basis = {
        "training_registry_seal_identity": read_json(
            result_root / "TC_LPMSO_H8_TRAINING_REGISTRY_SEAL.json"
        )["identity"],
        "locked_evaluator_seal_identity": read_json(
            result_root / "TC_LPMSO_H8_LOCKED_EVALUATOR_SEAL.json"
        )["identity"],
        "first_access_stage": "POST_TRAINING_LOCKED_EVALUATION",
        "h5_locked_report_access_count": session.access_count,
        "per_dataset_access_count": session.per_dataset_access_count,
        "classification_fit_role": "GLOBAL_SEARCH_POOL",
        "classification_test_role": "H5_LOCKED_REPORT",
        "clustering_fit_role": "ALL_TARGET_NODES_UNLABELED",
        "clustering_metric_role": "H5_LOCKED_REPORT",
        "raw_labels_persisted": False,
        "locked_indices_persisted": False,
        "hyperparameter_search": False,
        "metric_row_count": len(metric_rows),
        "bootstrap_resamples_reserved_for_decision": BOOTSTRAP_RESAMPLES,
    }
    access = envelope(
        "metachorus.v53.m4a-h8.locked-access-audit.v1", "PASS", access_basis
    )
    write_json_once(result_root / "M4A_H8_LOCKED_ACCESS_AUDIT.json", access)
    append_jsonl_once(
        result_root / "M4A_H8_EXECUTION_LOG.jsonl",
        {
            "event_id": "H8_LOCKED_EVALUATION_COMPLETE",
            "status": "PASS",
            "locked_access_audit_identity": access["identity"],
            "metric_row_count": len(metric_rows),
            "h5_locked_report_access_count": session.access_count,
            "raw_labels_persisted": False,
        },
    )
    return access
