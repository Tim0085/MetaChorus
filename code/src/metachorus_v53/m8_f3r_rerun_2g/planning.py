"""Sealed evaluation matrix, resource forecast, and two-GPU group schedule."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Mapping, Sequence

from .common import canonical_json_hash, envelope
from .constants import (
    ABLATION_DATASETS,
    ENDPOINT_UPDATE,
    EVALUATION_MODEL_SEEDS,
    FROZEN_SETS,
    HISTORICAL_FULL_RUN_SECONDS,
    RESOURCE_LIMITS,
    RUN_CAPS,
    TRAINER,
    VIEW_PIPELINE,
)
from .selection import validate_selection_rows

EXACT_REUSE_FIELDS = (
    "dataset_identity", "set_hash", "model_seed", "start_update", "endpoint_update", "trainer",
    "view_pipeline", "labels_mounted_during_training", "classification_fit_role", "classification_metric_role",
    "clustering_fit_role", "clustering_metric_role", "kmeans_random_state_identity", "kmeans_n_init", "kmeans_algorithm",
)


def unique_selected_sets(selection_rows: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    validate_selection_rows(selection_rows)
    grouped: dict[str, dict[str, dict[str, Any]]] = {dataset: {} for dataset in ABLATION_DATASETS}
    for row in selection_rows:
        dataset = str(row["dataset"]).casefold()
        grouped[dataset].setdefault(
            str(row["set_hash"]),
            {"dataset": dataset, "set_id": str(row["set_id"]), "set_hash": str(row["set_hash"]), "path_ids": str(row["path_ids"])},
        )
    for dataset in ABLATION_DATASETS:
        manual = FROZEN_SETS[dataset]["MANUAL"]
        grouped[dataset].setdefault(
            str(manual["set_hash"]),
            {
                "dataset": dataset,
                "set_id": str(manual["set_id"]),
                "set_hash": str(manual["set_hash"]),
                "path_ids": json.dumps(list(manual["path_ids"]), ensure_ascii=False, separators=(",", ":")),
            },
        )
    counts = {dataset: len(values) for dataset, values in grouped.items()}
    if any(count > RUN_CAPS["maximum_unique_sets_per_dataset_including_manual"] for count in counts.values()) or sum(counts.values()) > RUN_CAPS["maximum_unique_sets_total"]:
        raise RuntimeError("BLOCKED_UNIQUE_SET_CAP_EXCEEDED")
    return {dataset: sorted(values.values(), key=lambda row: row["set_hash"]) for dataset, values in grouped.items()}


def exact_reuse_matches(required: Mapping[str, Any], existing: Mapping[str, Any]) -> bool:
    def normalize(field: str, value: Any) -> Any:
        if field in {"model_seed", "start_update", "endpoint_update", "kmeans_n_init"}:
            return int(value)
        if field == "labels_mounted_during_training":
            if isinstance(value, bool):
                return value
            return str(value).strip().casefold() in {"true", "1", "yes"}
        return str(value)
    return all(field in required and field in existing and normalize(field, required[field]) == normalize(field, existing[field]) for field in EXACT_REUSE_FIELDS)


def build_evaluation_matrix(
    selection_rows: Sequence[Mapping[str, Any]],
    *,
    dataset_identities: Mapping[str, str],
    evaluation_contracts: Mapping[str, Mapping[str, Any]],
    existing_rows: Sequence[Mapping[str, Any]] = (),
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sets = unique_selected_sets(selection_rows)
    existing = defaultdict(list)
    for row in existing_rows:
        existing[(str(row.get("dataset", "")).casefold(), str(row.get("set_hash", "")), int(row.get("model_seed", -1)))].append(dict(row))
    reuse_registry: list[dict[str, Any]] = []
    matrix: list[dict[str, Any]] = []
    for dataset in ABLATION_DATASETS:
        if dataset not in dataset_identities or dataset not in evaluation_contracts:
            raise ValueError(f"missing frozen identity or evaluator contract: {dataset}")
        contract = dict(evaluation_contracts[dataset])
        for set_row in sets[dataset]:
            for seed in EVALUATION_MODEL_SEEDS:
                required = {
                    "dataset_identity": dataset_identities[dataset],
                    "set_hash": set_row["set_hash"],
                    "model_seed": seed,
                    "start_update": 0,
                    "endpoint_update": ENDPOINT_UPDATE,
                    "trainer": TRAINER,
                    "view_pipeline": VIEW_PIPELINE,
                    "labels_mounted_during_training": False,
                    **contract,
                }
                matches = [row for row in existing[(dataset, set_row["set_hash"], seed)] if exact_reuse_matches(required, row)]
                if len(matches) > 1:
                    raise ValueError("multiple exact reuse rows claim the same scientific identity")
                status = "EXACT_REUSE" if matches else "NEW_RUN_REQUIRED"
                physical = f"M8-F3R-2G-{dataset.upper()}-{set_row['set_hash'][:12]}-E{EVALUATION_MODEL_SEEDS.index(seed):02d}"
                matrix.append(
                    {
                        "dataset": dataset,
                        "dataset_identity": dataset_identities[dataset],
                        "set_id": set_row["set_id"],
                        "set_hash": set_row["set_hash"],
                        "path_ids": set_row["path_ids"],
                        "model_seed": seed,
                        "start_update": 0,
                        "endpoint_update": ENDPOINT_UPDATE,
                        "trainer": TRAINER,
                        "view_pipeline": VIEW_PIPELINE,
                        "labels_mounted": False,
                        "downstream_evaluations_during_training": 0,
                        "physical_run_id": physical,
                        "status": status,
                    }
                )
                reuse_registry.append(
                    {
                        **required,
                        "physical_run_id": physical,
                        "matched_existing_run_id": str(matches[0].get("physical_run_id", "")) if matches else "",
                        "matched_existing_marker_identity": str(matches[0].get("marker_identity", "")) if matches else "",
                        "status": status,
                    }
                )
    identities = [(row["dataset"], row["set_hash"], row["model_seed"]) for row in matrix]
    if len(identities) != len(set(identities)):
        raise AssertionError("evaluation matrix contains duplicate identities")
    return matrix, reuse_registry


def seal_matrix(matrix: Sequence[Mapping[str, Any]], *, selection_seal_identity: str, sealed_at_utc: str) -> dict[str, Any]:
    allowed = {"EXACT_REUSE", "NEW_RUN_REQUIRED", "BLOCKED"}
    if not matrix or any(str(row.get("status", "")) not in allowed for row in matrix):
        raise ValueError("evaluation matrix status is invalid")
    basis = {
        "row_count": len(matrix),
        "rows_identity": canonical_json_hash(list(matrix)),
        "selection_seal_identity": selection_seal_identity,
        "datasets": list(ABLATION_DATASETS),
        "evaluation_model_seeds": list(EVALUATION_MODEL_SEEDS),
        "endpoint_update": ENDPOINT_UPDATE,
        "labels_mounted_during_training": False,
        "evaluation_roles_frozen": True,
        "sealed_at_utc": sealed_at_utc,
    }
    return envelope("metachorus.v53.m8-f3r-rerun-2g.evaluation-matrix-seal.v1", "EVALUATION_MATRIX_SEALED", basis)


def build_two_gpu_group_schedule(matrix: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in matrix:
        if str(row.get("status")) == "NEW_RUN_REQUIRED":
            groups[(str(row["dataset"]), int(row["model_seed"]))].append(row)
    loads = [0.0, 0.0]
    result = []
    estimated = [
        (identity, rows, len(rows) * HISTORICAL_FULL_RUN_SECONDS[identity[0]])
        for identity, rows in groups.items()
    ]
    for (dataset, seed), rows, seconds in sorted(estimated, key=lambda item: (-item[2], item[0])):
        gpu = min(range(2), key=lambda index: (loads[index], index))
        loads[gpu] += seconds
        result.append(
            {
                "dataset": dataset,
                "model_seed": seed,
                "group_id": f"{dataset}__{seed}",
                "gpu_slot": gpu,
                "run_count": len(rows),
                "physical_run_ids": json.dumps([str(row["physical_run_id"]) for row in rows], separators=(",", ":")),
                "estimated_group_seconds": seconds,
                "estimated_gpu_load_after_assignment_seconds": loads[gpu],
                "scheduling_algorithm": "LONGEST_PROCESSING_TIME_FIRST",
                "group_may_split_across_gpus": False,
                "status": "PLANNED",
            }
        )
    if any(len({row["gpu_slot"] for row in result if row["dataset"] == dataset and row["model_seed"] == seed}) > 1 for dataset, seed in groups):
        raise AssertionError("dataset/model-seed group was split across GPUs")
    return result


def resource_forecast(matrix: Sequence[Mapping[str, Any]], schedule: Sequence[Mapping[str, Any]], *, search_continuation_runs: int = 0, estimated_remote_storage_bytes: int = 0) -> dict[str, Any]:
    fresh = [row for row in matrix if str(row.get("status")) == "NEW_RUN_REQUIRED"]
    seconds = sum(HISTORICAL_FULL_RUN_SECONDS[str(row["dataset"])] for row in fresh)
    gpu_loads = [sum(float(row["estimated_group_seconds"]) for row in schedule if int(row["gpu_slot"]) == index) for index in range(2)]
    basis = {
        "new_full_training_runs": len(fresh),
        "search_continuation_runs": int(search_continuation_runs),
        "estimated_gpu_hours": seconds / 3600.0,
        "ideal_two_gpu_critical_path_hours": seconds / 7200.0,
        "scheduled_gpu_load_seconds": gpu_loads,
        "scheduled_two_gpu_critical_path_hours": max(gpu_loads, default=0.0) / 3600.0,
        "estimated_remote_storage_bytes": int(estimated_remote_storage_bytes),
        "hard_gpu_hours": RESOURCE_LIMITS["hard_gpu_hours"],
        "hard_critical_path_hours": RESOURCE_LIMITS["hard_critical_path_hours"],
        "hard_remote_storage_bytes": RESOURCE_LIMITS["hard_remote_storage_bytes"],
    }
    hard_failures = []
    if basis["new_full_training_runs"] > RUN_CAPS["maximum_new_full_training_runs"]:
        hard_failures.append("NEW_FULL_TRAINING_RUN_CAP")
    if basis["search_continuation_runs"] > RUN_CAPS["maximum_search_continuation_runs"]:
        hard_failures.append("SEARCH_CONTINUATION_RUN_CAP")
    if basis["estimated_gpu_hours"] > RESOURCE_LIMITS["hard_gpu_hours"]:
        hard_failures.append("HARD_GPU_HOURS")
    if basis["scheduled_two_gpu_critical_path_hours"] > RESOURCE_LIMITS["hard_critical_path_hours"]:
        hard_failures.append("HARD_CRITICAL_PATH")
    if basis["estimated_remote_storage_bytes"] > RESOURCE_LIMITS["hard_remote_storage_bytes"]:
        hard_failures.append("HARD_REMOTE_STORAGE")
    basis["hard_failures"] = hard_failures
    return envelope(
        "metachorus.v53.m8-f3r-rerun-2g.resource-forecast.v1",
        "PASS" if not hard_failures else "BLOCKED_RESOURCE_FORECAST",
        basis,
    )


def build_canary_plan() -> list[dict[str, Any]]:
    rows = []
    for ordinal, (gpu, dataset, role) in enumerate(
        ((0, "freebase", "MANUAL"), (0, "dblp", "FULL_FROZEN"), (1, "freebase", "FULL_FROZEN"), (1, "dblp", "MANUAL"))
    ):
        set_row = FROZEN_SETS[dataset][role]
        rows.append(
            {
                "canary_ordinal": ordinal,
                "gpu_slot": gpu,
                "dataset": dataset,
                "path_set_role": role,
                "set_id": set_row["set_id"],
                "set_hash": set_row["set_hash"],
                "path_ids": json.dumps(list(set_row["path_ids"]), ensure_ascii=False, separators=(",", ":")),
                "model_seed": EVALUATION_MODEL_SEEDS[0],
                "physical_run_id": f"M8-F3R-2G-CANARY-{dataset.upper()}-{set_row['set_hash'][:12]}-C{ordinal:02d}",
                "start_update": 0,
                "endpoint_update": 20,
                "labels_mounted": False,
                "downstream_evaluations_during_training": 0,
                "scientific_result": False,
                "formal_run_restarts_from_zero": True,
                "status": "AUTHORIZED_ENGINEERING_CANARY",
            }
        )
    return rows


__all__ = [name for name in globals() if not name.startswith("_")]
