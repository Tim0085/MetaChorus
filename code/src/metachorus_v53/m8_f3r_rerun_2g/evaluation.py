"""Registry-sealed locked evaluation for the Freebase and DBLP reruns."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from metachorus_v53.m4a_h5.labels import (
    global_search_roles,
    load_freebase_labels,
    load_private_search_mount,
)
from metachorus_v53.m4a_h8.constants import ROLE_MANIFEST_SHA256
from metachorus_v53.m4a_h8.locked_evaluation import (
    derive_kmeans_seeds,
    evaluate_locked_classification,
    evaluate_locked_clustering,
)
from metachorus_v53.m6f_m7f_fast.evaluation import (
    PreliminaryLabelSession,
    evaluate_preliminary_representation,
)

from .common import (
    canonical_json_hash,
    envelope,
    read_csv,
    read_json,
    sha256_file,
    write_csv_once,
    write_json_once,
)
from .constants import (
    ABLATION_DATASETS,
    EVALUATION_CONTRACTS,
    EVALUATION_MODEL_SEEDS,
    FREEBASE_H8_KMEANS_SEED_IDENTITY,
    FROZEN_SETS,
    MAX_ATTEMPTS_PER_IDENTITY,
    METRIC_FIELDS,
    TASKBOOK_SHA256,
    TASKBOOK_STAGE,
)
from .lifecycle import publish_json_state, publish_snapshot
from .science import (
    metric_rows_for_variants,
    recompute_paired_deltas,
    recompute_summary,
    validate_metric_matrix,
    validate_paired_deltas,
    validate_physical_metric_roles,
    validate_summary,
)
from .training import completed_marker, validate_plan


RUN_REGISTRY_FIELDS = (
    "dataset",
    "physical_run_id",
    "set_id",
    "set_hash",
    "model_seed",
    "start_update",
    "endpoint_update",
    "attempt_index",
    "GPU_UUID",
    "gpu_pci_bus_id",
    "runtime_domain",
    "training_seconds",
    "representation_relative_path",
    "representation_sha256",
    "representation_size_bytes",
    "checkpoint_relative_path",
    "checkpoint_sha256",
    "checkpoint_size_bytes",
    "marker_relative_path",
    "marker_identity",
    "result_source",
    "labels_mounted_during_training",
    "status",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fields(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    if not rows:
        raise ValueError("cannot publish an empty governed CSV")
    output: list[str] = []
    for row in rows:
        for key in row:
            if key not in output:
                output.append(str(key))
    return output


def _inside(root: Path, relative: str) -> Path:
    value = Path(relative)
    if value.is_absolute() or ".." in value.parts:
        raise ValueError("governed artifact path is unsafe")
    path = root / value
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(path)
    try:
        path.resolve(strict=True).relative_to(root.resolve(strict=True))
    except ValueError as error:
        raise ValueError("governed artifact escapes the result root") from error
    return path


def _new_run_registry_row(
    *, result: Path, matrix_row: Mapping[str, Any], marker_path: Path, marker: Mapping[str, Any]
) -> dict[str, Any]:
    basis = marker["basis"]
    if (
        basis.get("dataset") != matrix_row.get("dataset")
        or basis.get("set_hash") != matrix_row.get("set_hash")
        or int(basis.get("model_seed", -1)) != int(matrix_row.get("model_seed", -2))
        or int(basis.get("endpoint_update", -1)) != int(matrix_row.get("endpoint_update", -2))
    ):
        raise ValueError("completion marker differs from its evaluation-matrix identity")
    representation = _inside(result, str(basis["representation_relative_path"]))
    checkpoint = _inside(result, str(basis["checkpoint_relative_path"]))
    if (
        sha256_file(representation) != basis.get("representation_sha256")
        or representation.stat().st_size != int(basis.get("representation_size_bytes", -1))
        or sha256_file(checkpoint) != basis.get("checkpoint_sha256")
        or checkpoint.stat().st_size != int(basis.get("checkpoint_size_bytes", -1))
    ):
        raise ValueError("completion marker artifacts changed before registry sealing")
    return {
        "dataset": str(matrix_row["dataset"]),
        "physical_run_id": str(matrix_row["physical_run_id"]),
        "set_id": str(matrix_row["set_id"]),
        "set_hash": str(matrix_row["set_hash"]),
        "model_seed": int(matrix_row["model_seed"]),
        "start_update": 0,
        "endpoint_update": int(matrix_row["endpoint_update"]),
        "attempt_index": int(basis["attempt_index"]),
        "GPU_UUID": str(basis["gpu_uuid"]),
        "gpu_pci_bus_id": str(basis["gpu_pci_bus_id"]),
        "runtime_domain": str(basis["runtime_domain"]),
        "training_seconds": float(basis["model_total_seconds"]),
        "representation_relative_path": str(basis["representation_relative_path"]),
        "representation_sha256": str(basis["representation_sha256"]),
        "representation_size_bytes": int(basis["representation_size_bytes"]),
        "checkpoint_relative_path": str(basis["checkpoint_relative_path"]),
        "checkpoint_sha256": str(basis["checkpoint_sha256"]),
        "checkpoint_size_bytes": int(basis["checkpoint_size_bytes"]),
        "marker_relative_path": marker_path.resolve(strict=True).relative_to(result).as_posix(),
        "marker_identity": str(marker["identity"]),
        "result_source": "NEW_RUN",
        "labels_mounted_during_training": False,
        "status": "PASS",
    }


def _reuse_registry_row(result: Path, matrix_row: Mapping[str, Any], evidence: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "dataset",
        "physical_run_id",
        "set_id",
        "set_hash",
        "model_seed",
        "representation_relative_path",
        "representation_sha256",
        "representation_size_bytes",
        "checkpoint_relative_path",
        "checkpoint_sha256",
        "checkpoint_size_bytes",
        "marker_relative_path",
        "marker_identity",
        "GPU_UUID",
        "gpu_pci_bus_id",
        "runtime_domain",
        "training_seconds",
    }
    if not required.issubset(evidence):
        raise ValueError("exact-reuse evidence lacks physical provenance")
    for field in ("dataset", "physical_run_id", "set_id", "set_hash", "model_seed"):
        if str(evidence[field]) != str(matrix_row[field]):
            raise ValueError("exact-reuse evidence differs from its frozen matrix row")
    representation = _inside(result, str(evidence["representation_relative_path"]))
    checkpoint = _inside(result, str(evidence["checkpoint_relative_path"]))
    marker = _inside(result, str(evidence["marker_relative_path"]))
    marker_value = read_json(marker)
    marker_basis = marker_value.get("basis", {})
    if (
        sha256_file(representation) != evidence["representation_sha256"]
        or representation.stat().st_size != int(evidence["representation_size_bytes"])
        or sha256_file(checkpoint) != evidence["checkpoint_sha256"]
        or checkpoint.stat().st_size != int(evidence["checkpoint_size_bytes"])
        or marker_value.get("schema_version")
        != "metachorus.v53.m8-f3r-rerun-2g.run-complete.v1"
        or marker_value.get("status") != "PASS"
        or marker_value.get("identity") != evidence["marker_identity"]
        or marker_value.get("identity") != canonical_json_hash(marker_basis)
        or marker_basis.get("logical_stage") != "EVALUATION"
        or marker_basis.get("physical_run_id") != matrix_row.get("physical_run_id")
        or marker_basis.get("dataset") != matrix_row.get("dataset")
        or marker_basis.get("set_hash") != matrix_row.get("set_hash")
        or int(marker_basis.get("model_seed", -1)) != int(matrix_row.get("model_seed", -2))
        or int(marker_basis.get("start_update", -1)) != 0
        or int(marker_basis.get("endpoint_update", -1))
        != int(matrix_row.get("endpoint_update", -2))
        or int(marker_basis.get("attempt_index", -1))
        not in range(MAX_ATTEMPTS_PER_IDENTITY)
        or marker_basis.get("labels_mounted_during_training") is not False
        or int(marker_basis.get("downstream_evaluations_during_training", -1)) != 0
        or marker_basis.get("one_process_per_gpu") is not True
        or marker_basis.get("ddp_used") is not False
        or marker_basis.get("data_parallel_used") is not False
        or marker_basis.get("torchrun_used") is not False
    ):
        raise ValueError("exact-reuse artifact identity changed")
    return {
        **{field: evidence[field] for field in RUN_REGISTRY_FIELDS if field in evidence},
        "dataset": str(matrix_row["dataset"]),
        "physical_run_id": str(matrix_row["physical_run_id"]),
        "set_id": str(matrix_row["set_id"]),
        "set_hash": str(matrix_row["set_hash"]),
        "model_seed": int(matrix_row["model_seed"]),
        "start_update": 0,
        "endpoint_update": int(matrix_row["endpoint_update"]),
        "result_source": "EXACT_REUSE",
        "labels_mounted_during_training": False,
        "status": "PASS",
    }


def seal_training_registry(
    *,
    result_root: str | Path,
    reuse_evidence: Sequence[Mapping[str, Any]] = (),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    result = Path(result_root).resolve(strict=True)
    matrix_rows = read_csv(result / "ABLATION_EVALUATION_MATRIX.csv")
    matrix_seal = read_json(result / "ABLATION_EVALUATION_MATRIX_SEAL.json")
    matrix = validate_plan(matrix_rows, matrix_seal, stage="EVALUATION")
    reuse = {
        (str(row.get("dataset", "")), str(row.get("set_hash", "")), int(row.get("model_seed", -1))): row
        for row in reuse_evidence
    }
    if len(reuse) != len(reuse_evidence):
        raise ValueError("exact-reuse evidence contains duplicate scientific identities")
    registry: list[dict[str, Any]] = []
    for row in matrix:
        status = str(row["status"])
        if status == "BLOCKED":
            raise RuntimeError("cannot unlock labels while an evaluation identity is blocked")
        if status == "NEW_RUN_REQUIRED":
            completed = completed_marker(result, row, "EVALUATION")
            if completed is None:
                raise RuntimeError("cannot unlock labels before every new physical run completes")
            registry.append(
                _new_run_registry_row(
                    result=result,
                    matrix_row=row,
                    marker_path=completed[0],
                    marker=completed[1],
                )
            )
        else:
            evidence = reuse.get((str(row["dataset"]), str(row["set_hash"]), int(row["model_seed"])))
            if evidence is None:
                raise RuntimeError("exact-reuse matrix row lacks staged governed evidence")
            registry.append(_reuse_registry_row(result, row, evidence))
    identities = [(row["dataset"], row["set_hash"], row["model_seed"]) for row in registry]
    if len(identities) != len(set(identities)) or len(registry) != len(matrix):
        raise ValueError("physical training registry is not an exact matrix cover")
    write_csv_once(result / "ABLATION_RUN_REGISTRY_MINIMAL.csv", registry, RUN_REGISTRY_FIELDS)
    basis = {
        "row_count": len(registry),
        "rows_identity": canonical_json_hash(registry),
        "evaluation_matrix_seal_identity": matrix_seal["identity"],
        "active_datasets": list(ABLATION_DATASETS),
        "all_status_pass": True,
        "labels_mounted": False,
        "locked_evaluation_started": False,
        "preliminary_evaluation_started": False,
        "sealed_at_utc": utc_now(),
    }
    seal = envelope(
        "metachorus.v53.m8-f3r-rerun-2g.training-registry-seal.v1",
        "SEALED_BEFORE_PRELIMINARY_EVALUATION",
        basis,
    )
    write_json_once(result / "M8_F3R_RERUN_2G_TRAINING_REGISTRY_SEAL.json", seal)
    return registry, seal


class FreebaseLockedLabelSession:
    """Open the Freebase supervised source once, after registry sealing."""

    def __init__(
        self,
        *,
        role_manifest_path: str | Path,
        private_search_mount_root: str | Path,
        label_source: str | Path,
        label_source_root: str | Path,
        training_registry_seal: Mapping[str, Any],
    ) -> None:
        if (
            training_registry_seal.get("status") != "SEALED_BEFORE_PRELIMINARY_EVALUATION"
            or training_registry_seal.get("basis", {}).get("all_status_pass") is not True
            or training_registry_seal.get("basis", {}).get("labels_mounted") is not False
            or training_registry_seal.get("basis", {}).get("locked_evaluation_started") is not False
        ):
            raise RuntimeError("Freebase labels are forbidden before the exact training registry seal")
        role_raw = Path(role_manifest_path).absolute()
        if role_raw.is_symlink():
            raise ValueError("Freebase role manifest cannot be a symlink")
        role = role_raw.resolve(strict=True)
        if sha256_file(role) != ROLE_MANIFEST_SHA256:
            raise ValueError("Freebase role manifest identity changed")
        self.manifest = read_json(role)
        rows = [row for row in self.manifest.get("datasets", []) if row.get("dataset") == "freebase"]
        if (
            self.manifest.get("status") != "SEALED_INPUT"
            or self.manifest.get("raw_labels_embedded") is not False
            or self.manifest.get("h5_locked_report_labels_embedded") is not False
            or len(rows) != 1
        ):
            raise ValueError("Freebase label-role manifest is invalid")
        self.row = rows[0]
        mount_raw = Path(private_search_mount_root).absolute()
        if mount_raw.is_symlink():
            raise ValueError("Freebase private-search root cannot be a symlink")
        self.mount_root = mount_raw.resolve(strict=True)
        self.source = Path(label_source)
        self.source_root = Path(label_source_root).resolve(strict=True)
        self._roles: dict[str, np.ndarray] | None = None
        self.access_count = 0

    def open(self) -> dict[str, np.ndarray]:
        if self._roles is not None:
            return self._roles
        if (
            self.source.is_symlink()
            or not self.source.is_file()
            or self.source_root.is_symlink()
            or not self.source_root.is_dir()
        ):
            raise ValueError("Freebase supervised source/root is not regular")
        try:
            self.source.resolve(strict=True).relative_to(self.source_root)
        except ValueError as error:
            raise ValueError("Freebase supervised source escapes its authorized root") from error
        if (
            self.source.name != self.row["label_source_name"]
            or sha256_file(self.source) != self.row["label_source_sha256"]
        ):
            raise ValueError("Freebase supervised source differs from the sealed manifest")
        labels = load_freebase_labels(self.source)
        split_seeds = [int(row["split_seed"]) for row in self.row.get("repeats", [])]
        roles = global_search_roles(
            labels,
            pool_seed=int(self.row["global_search_pool_seed"]),
            split_seeds=split_seeds,
        )
        search = load_private_search_mount(
            self.mount_root, "freebase", expected_manifest=self.manifest
        )
        pool = np.asarray(roles["global_search_pool"], dtype=np.int64)
        locked = np.asarray(roles["h5_locked_report"], dtype=np.int64)
        if (
            not np.array_equal(pool, search["global_search_pool_indices"])
            or not np.array_equal(labels[pool], search["global_search_pool_labels"])
        ):
            raise ValueError("Freebase search mount and locked-role partition do not share one pool")
        self._roles = {
            "pool_indices": pool,
            "pool_labels": labels[pool],
            "locked_indices": locked,
            "locked_labels": labels[locked],
        }
        self.access_count += 1
        return self._roles


def _representation(result: Path, row: Mapping[str, Any]) -> np.ndarray:
    path = _inside(result, str(row["representation_relative_path"]))
    if (
        sha256_file(path) != row["representation_sha256"]
        or path.stat().st_size != int(row["representation_size_bytes"])
    ):
        raise ValueError("sealed representation identity changed before evaluation")
    values = np.load(path, allow_pickle=False)
    if values.ndim != 2 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("sealed representation is not finite two-dimensional data")
    return np.asarray(values, dtype=np.float64)


def _physical_metric_row(
    registry_row: Mapping[str, Any], metrics: Mapping[str, Any], *, evaluated_at_utc: str
) -> dict[str, Any]:
    values = metrics["metrics"]
    return {
        "dataset": str(registry_row["dataset"]),
        "physical_run_id": str(registry_row["physical_run_id"]),
        "set_id": str(registry_row["set_id"]),
        "set_hash": str(registry_row["set_hash"]),
        "model_seed": int(registry_row["model_seed"]),
        "Micro_F1": float(values["Micro_F1"]),
        "Macro_F1": float(values["Macro_F1"]),
        "NMI": float(values["NMI"]),
        "ARI": float(values["ARI"]),
        "assignment_stability": float(metrics["assignment_stability_mean"]),
        "GPU_UUID": str(registry_row["GPU_UUID"]),
        "runtime_domain": str(registry_row["runtime_domain"]),
        "training_seconds": float(registry_row["training_seconds"]),
        "result_source": str(registry_row["result_source"]),
        "evaluated_at_utc": evaluated_at_utc,
        "classification_fit_role": str(metrics["classification_fit_role"]),
        "classification_metric_role": str(metrics["classification_metric_role"]),
        "clustering_fit_role": str(metrics["clustering_fit_role"]),
        "clustering_metric_role": str(metrics["clustering_metric_role"]),
        "raw_labels_persisted": False,
        "status": "PASS",
    }


def evaluate_locked(
    *,
    result_root: str | Path,
    freebase_role_manifest_path: str | Path,
    dblp_role_manifest_path: str | Path,
    private_search_mount_roots: Mapping[str, str | Path],
    label_sources: Mapping[str, str | Path],
    label_source_roots: Mapping[str, str | Path],
    reuse_evidence: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    if set(private_search_mount_roots) != set(ABLATION_DATASETS) or set(label_sources) != set(ABLATION_DATASETS) or set(label_source_roots) != set(ABLATION_DATASETS):
        raise ValueError("locked evaluator inputs must cover exactly Freebase and DBLP")
    for dataset in ABLATION_DATASETS:
        if Path(private_search_mount_roots[dataset]).absolute().is_symlink():
            raise ValueError(f"private-search root cannot be a symlink: {dataset}")
    result = Path(result_root).resolve(strict=True)
    registry, registry_seal = seal_training_registry(
        result_root=result, reuse_evidence=reuse_evidence
    )
    started = envelope(
        "metachorus.v53.m8-f3r-rerun-2g.locked-evaluation-started.v1",
        "STARTED_AFTER_TRAINING_REGISTRY_SEAL",
        {
            "training_registry_seal_identity": registry_seal["identity"],
            "physical_run_count": len(registry),
            "label_access_count_before_start": 0,
            "started_at_utc": utc_now(),
        },
    )
    write_json_once(result / "M8_F3R_RERUN_2G_LOCKED_EVALUATION_STARTED.json", started)

    selections = read_csv(result / "ABLATION_SELECTIONS_ALL.csv")
    projection_rows: dict[str, list[dict[str, Any]]] = {}
    for dataset in ABLATION_DATASETS:
        manual = FROZEN_SETS[dataset]["MANUAL"]
        projection_rows[dataset] = [
            dict(row)
            for row in selections
            if str(row["dataset"]).casefold() == dataset
        ] + [
            {
                "dataset": dataset,
                "ablation_variant": "MANUAL",
                "set_id": manual["set_id"],
                "set_hash": manual["set_hash"],
            }
        ]

    freebase_session = FreebaseLockedLabelSession(
        role_manifest_path=freebase_role_manifest_path,
        private_search_mount_root=private_search_mount_roots["freebase"],
        label_source=label_sources["freebase"],
        label_source_root=label_source_roots["freebase"],
        training_registry_seal=registry_seal,
    )
    freebase_roles = freebase_session.open()
    kmeans_seeds = derive_kmeans_seeds()
    seed_basis = {
        "derived_before_locked_evaluation": True,
        "disjoint_from_model_search_split_bank_seeds": True,
        "h5_role_seal_sha256": "7074834d03d47105e7c9c6cfbcf79e6d7ab983e0a51e9cc207bd25ccf76f9ae2",
        "namespace": "TC_LPMSO_H8_LOCKED_KMEANS",
        "parent_package_sha256": "204ea4f2ab3123644a7448e0d2742417f78b2083898a56e3747465ca5b1d1346",
        "seed_count": 20,
        "seeds": list(kmeans_seeds),
        "taskbook_sha256": "b0f5651eff3c8eeb01d216a07bb3fa916b5d789b1c9c25857624ec3c2b7d63fd",
        "unique": True,
    }
    if canonical_json_hash(seed_basis) != FREEBASE_H8_KMEANS_SEED_IDENTITY:
        raise RuntimeError("Freebase H8 KMeans seed namespace identity changed")

    freebase_physical = []
    evaluated_at = utc_now()
    for row in registry:
        if str(row["dataset"]) != "freebase":
            continue
        features = _representation(result, row)
        classification = evaluate_locked_classification(
            features,
            pool_indices=freebase_roles["pool_indices"],
            pool_labels=freebase_roles["pool_labels"],
            locked_indices=freebase_roles["locked_indices"],
            locked_labels=freebase_roles["locked_labels"],
        )
        _, clustering = evaluate_locked_clustering(
            features,
            dataset="freebase",
            locked_indices=freebase_roles["locked_indices"],
            locked_labels=freebase_roles["locked_labels"],
            kmeans_seeds=kmeans_seeds,
        )
        metrics = {
            "metrics": {
                "Micro_F1": classification["Micro_F1"],
                "Macro_F1": classification["Macro_F1"],
                "NMI": clustering["NMI"],
                "ARI": clustering["ARI"],
            },
            "assignment_stability_mean": clustering["assignment_stability_mean"],
            "classification_fit_role": "GLOBAL_SEARCH_POOL",
            "classification_metric_role": "H5_LOCKED_REPORT",
            "clustering_fit_role": "ALL_TARGET_NODES_UNLABELED",
            "clustering_metric_role": "H5_LOCKED_REPORT",
        }
        freebase_physical.append(
            _physical_metric_row(row, metrics, evaluated_at_utc=evaluated_at)
        )
    validate_physical_metric_roles(freebase_physical)
    freebase_metric_rows = metric_rows_for_variants(
        freebase_physical, projection_rows["freebase"]
    )
    write_csv_once(
        result / "ABLATION_PHYSICAL_METRICS_FREEBASE.csv",
        freebase_physical,
        _fields(freebase_physical),
    )
    write_csv_once(
        result / "ABLATION_METRICS_BY_SEED_FREEBASE_TEMP.csv",
        freebase_metric_rows,
        METRIC_FIELDS,
    )
    current_state = read_json(result / "ABLATION_EXECUTION_STATUS.json")
    freebase_state = envelope(
        "metachorus.v53.m8-f3r-rerun-2g.execution-status.v1",
        "FREEBASE_COMPLETE",
        {
            **dict(current_state.get("basis", {})),
            "stage": TASKBOOK_STAGE,
            "taskbook_sha256": TASKBOOK_SHA256,
            "terminal_status": "FREEBASE_COMPLETE",
            "freebase_complete": True,
            "dblp_complete": False,
            "freebase_metric_rows": len(freebase_metric_rows),
            "freebase_label_access_count": freebase_session.access_count,
        },
    )
    publish_json_state(result / "ABLATION_EXECUTION_STATUS.json", freebase_state)
    freebase_snapshot = publish_snapshot(
        result,
        prefix="SNAPSHOT_1_FREEBASE",
        member_names=(
            "ABLATION_PHYSICAL_METRICS_FREEBASE.csv",
            "ABLATION_METRICS_BY_SEED_FREEBASE_TEMP.csv",
            "ABLATION_SELECTIONS_ALL.csv",
            "ABLATION_SELECTION_SEAL.json",
            "ABLATION_RUN_REGISTRY_MINIMAL.csv",
            "M8_F3R_RERUN_2G_TRAINING_REGISTRY_SEAL.json",
            "M8_F3R_RERUN_2G_LOCKED_EVALUATION_STARTED.json",
            "M8_F3R_RERUN_2G_GIT_AND_CONFIG_IDENTITY.json",
            "ABLATION_EXECUTION_STATUS.json",
        ),
    )

    dblp_session = PreliminaryLabelSession(
        role_manifest_path=dblp_role_manifest_path,
        private_search_mount_root=private_search_mount_roots["dblp"],
        label_sources={"dblp": label_sources["dblp"]},
        label_source_roots={"dblp": label_source_roots["dblp"]},
        training_registry_seal=registry_seal,
    )
    dblp_roles = dblp_session.open_dataset("dblp")
    dblp_physical = []
    evaluated_at = utc_now()
    for row in registry:
        if str(row["dataset"]) != "dblp":
            continue
        features = _representation(result, row)
        metrics = evaluate_preliminary_representation(
            features, dataset="dblp", roles=dblp_roles
        )
        dblp_physical.append(
            _physical_metric_row(row, metrics, evaluated_at_utc=evaluated_at)
        )
    validate_physical_metric_roles(dblp_physical)
    dblp_metric_rows = metric_rows_for_variants(
        dblp_physical, projection_rows["dblp"]
    )
    write_csv_once(
        result / "ABLATION_PHYSICAL_METRICS_DBLP.csv",
        dblp_physical,
        _fields(dblp_physical),
    )
    write_csv_once(
        result / "ABLATION_METRICS_BY_SEED_DBLP_TEMP.csv",
        dblp_metric_rows,
        METRIC_FIELDS,
    )
    physical_metrics = [*freebase_physical, *dblp_physical]
    write_csv_once(
        result / "ABLATION_PHYSICAL_METRICS.csv",
        physical_metrics,
        _fields(physical_metrics),
    )
    physical_seal = envelope(
        "metachorus.v53.m8-f3r-rerun-2g.physical-metrics-seal.v1",
        "PASS",
        {
            "row_count": len(physical_metrics),
            "rows_identity": canonical_json_hash(physical_metrics),
            "training_registry_seal_identity": registry_seal["identity"],
            "freebase_label_access_count": freebase_session.access_count,
            "dblp_label_access_count": dblp_session.per_dataset_access_count["dblp"],
            "raw_labels_persisted": False,
        },
    )
    write_json_once(result / "ABLATION_PHYSICAL_METRICS_SEAL.json", physical_seal)

    metric_rows = [*freebase_metric_rows, *dblp_metric_rows]
    validate_metric_matrix(metric_rows, require_manual=True)
    summary = recompute_summary(metric_rows)
    full_deltas = recompute_paired_deltas(metric_rows, reference_variant="FULL_FROZEN")
    manual_deltas = recompute_paired_deltas(metric_rows, reference_variant="MANUAL")
    validate_summary(summary, metric_rows)
    validate_paired_deltas(full_deltas, metric_rows, reference_variant="FULL_FROZEN")
    validate_paired_deltas(manual_deltas, metric_rows, reference_variant="MANUAL")
    write_csv_once(result / "ABLATION_METRICS_BY_SEED.csv", metric_rows, METRIC_FIELDS)
    write_csv_once(result / "ABLATION_SUMMARY.csv", summary, _fields(summary))
    write_csv_once(result / "ABLATION_PAIRED_DELTAS_VS_FULL.csv", full_deltas, _fields(full_deltas))
    write_csv_once(result / "ABLATION_PAIRED_DELTAS_VS_MANUAL.csv", manual_deltas, _fields(manual_deltas))

    access_basis = {
        "training_registry_seal_identity": registry_seal["identity"],
        "physical_metrics_seal_identity": physical_seal["identity"],
        "freebase_label_access_count": freebase_session.access_count,
        "dblp_label_access_count": dblp_session.per_dataset_access_count["dblp"],
        "labels_mounted_during_training": False,
        "raw_labels_persisted": False,
        "metric_rows": len(metric_rows),
        "metric_rows_identity": canonical_json_hash(metric_rows),
        "summary_rows_identity": canonical_json_hash(summary),
        "full_delta_rows_identity": canonical_json_hash(full_deltas),
        "manual_delta_rows_identity": canonical_json_hash(manual_deltas),
        "freebase_snapshot_identity": freebase_snapshot["snapshot_identity"],
    }
    if access_basis["freebase_label_access_count"] != 1 or access_basis["dblp_label_access_count"] != 1:
        raise RuntimeError("each ablation dataset label source must be opened exactly once")
    receipt = envelope(
        "metachorus.v53.m8-f3r-rerun-2g.locked-evaluation-complete.v1",
        "PASS",
        access_basis,
    )
    write_json_once(result / "M8_F3R_RERUN_2G_LOCKED_ACCESS_AUDIT.json", receipt)
    dblp_state = envelope(
        "metachorus.v53.m8-f3r-rerun-2g.execution-status.v1",
        "DBLP_COMPLETE",
        {
            **dict(freebase_state["basis"]),
            "terminal_status": "DBLP_COMPLETE",
            "dblp_complete": True,
            "dblp_metric_rows": len(dblp_metric_rows),
            "dblp_label_access_count": dblp_session.per_dataset_access_count["dblp"],
            "freebase_snapshot_identity": freebase_snapshot["snapshot_identity"],
            "physical_metrics_seal_identity": physical_seal["identity"],
            "locked_access_audit_identity": receipt["identity"],
        },
    )
    publish_json_state(result / "ABLATION_EXECUTION_STATUS.json", dblp_state)
    publish_snapshot(
        result,
        prefix="SNAPSHOT_2_DBLP",
        member_names=(
            "ABLATION_PHYSICAL_METRICS_FREEBASE.csv",
            "ABLATION_PHYSICAL_METRICS_DBLP.csv",
            "ABLATION_PHYSICAL_METRICS.csv",
            "ABLATION_PHYSICAL_METRICS_SEAL.json",
            "ABLATION_METRICS_BY_SEED_FREEBASE_TEMP.csv",
            "ABLATION_METRICS_BY_SEED_DBLP_TEMP.csv",
            "ABLATION_METRICS_BY_SEED.csv",
            "ABLATION_SUMMARY.csv",
            "ABLATION_PAIRED_DELTAS_VS_FULL.csv",
            "ABLATION_PAIRED_DELTAS_VS_MANUAL.csv",
            "ABLATION_SELECTIONS_ALL.csv",
            "ABLATION_SELECTION_SEAL.json",
            "ABLATION_RUN_REGISTRY_MINIMAL.csv",
            "M8_F3R_RERUN_2G_TRAINING_REGISTRY_SEAL.json",
            "M8_F3R_RERUN_2G_LOCKED_EVALUATION_STARTED.json",
            "M8_F3R_RERUN_2G_LOCKED_ACCESS_AUDIT.json",
            "M8_F3R_RERUN_2G_GIT_AND_CONFIG_IDENTITY.json",
            "ABLATION_EXECUTION_STATUS.json",
        ),
    )
    return receipt


__all__ = [
    "RUN_REGISTRY_FIELDS",
    "seal_training_registry",
    "FreebaseLockedLabelSession",
    "evaluate_locked",
]
