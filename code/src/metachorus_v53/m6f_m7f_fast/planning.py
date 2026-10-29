"""Exact search and preliminary evaluation planning over frozen caches."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from metachorus_v53.audit.common import canonical_json_hash, read_json

from .common import envelope
from .constants import (
    DATASET_NAMES,
    EVAL_MODEL_SEEDS,
    MAX_GPUS,
    PRELIM_ENDPOINT,
    SEARCH_ENDPOINTS,
    SEARCH_MODEL_SEEDS,
)


SEARCH_STAGE = "R1_ALL_SETS_0000_TO_0300"


def _canonical_serialized_plan_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact typed row recovered after the CSV round trip."""

    value = dict(row)
    for field in (
        "candidate_set_ordinal",
        "physical_run_ordinal",
        "model_seed",
        "model_seed_ordinal",
        "continuation_from_update",
        "endpoint_update",
        "attempt_index",
        "downstream_evaluations_during_training",
        "worst_case_gpu_edge_index_bytes",
    ):
        if field in value:
            value[field] = int(value[field])
    for field in ("path_ids", "checkpoint_schedule", "logical_roles"):
        if field in value:
            value[field] = list(value[field])
    for field in (
        "labels_mounted",
        "eligible_for_selector_development",
        "eligible_for_preliminary_scientific_result",
        "training_from_zero",
        "search_checkpoint_reused",
        "search_optimizer_reused",
        "search_rng_state_reused",
    ):
        if field in value:
            value[field] = bool(value[field])
    return value


def load_cache_resolution(cache_root: str | Path) -> dict[str, Any]:
    root = Path(cache_root).resolve()
    complete = read_json(root / "SANITIZED_CACHE_COMPLETE.json")
    manifest = read_json(root / "SANITIZED_CACHE_MANIFEST.json")
    path_library = read_json(root / "candidates" / "path_library.json")
    all_sets = read_json(root / "candidates" / "all_sets.json")
    fixed = read_json(root / "candidates" / "fixed_cardinality_sets.json")
    dataset = str(complete["dataset"])
    if (
        dataset not in DATASET_NAMES
        or complete.get("status") != "COMPLETE"
        or complete.get("cache_identity") != manifest.get("cache_identity")
        or path_library.get("dataset") != dataset
        or all_sets.get("dataset") != dataset
        or fixed.get("dataset") != dataset
        or path_library.get("path_library_identity") != manifest["identity_basis"].get("path_library_identity")
        or all_sets.get("all_sets_identity") != manifest["identity_basis"].get("all_sets_identity")
    ):
        raise ValueError(f"invalid frozen cache resolution: {root}")
    manual = [row for row in fixed.get("sets", []) if row.get("path_set_role") == "MANUAL"]
    if len(manual) != 1 or fixed.get("current_automatic", {}).get("status") != "NOT_AVAILABLE_EXACT_IDENTITY":
        raise ValueError(f"{dataset} baseline registry is not exact")
    return {
        "dataset": dataset,
        "cache_root": root,
        "cache_identity": str(complete["cache_identity"]),
        "path_library_identity": str(path_library["path_library_identity"]),
        "all_sets_identity": str(all_sets["all_sets_identity"]),
        "path_library": path_library,
        "all_sets": all_sets,
        "manual": manual[0],
        "current_automatic": dict(fixed["current_automatic"]),
    }


def physical_run_id(dataset: str, candidate_ordinal: int, seed_ordinal: int, *, prefix: str = "M6F") -> str:
    token = {"dblp": "DBLP", "imdb": "IMDB", "aminer": "AMINER"}[dataset]
    kind = "Q" if prefix == "M6F" else "E"
    return f"{prefix}-FAST-{token}-C{int(candidate_ordinal):03d}-{kind}{int(seed_ordinal):02d}"


def canary_run_id(dataset: str, candidate_ordinal: int, seed_ordinal: int = 0) -> str:
    token = {"dblp": "DBLP", "imdb": "IMDB", "aminer": "AMINER"}[dataset]
    return f"M6F-FAST-CANARY-{token}-C{int(candidate_ordinal):03d}-Q{int(seed_ordinal):02d}"


def build_search_plan(cache_rows: Mapping[str, Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if set(cache_rows) != set(DATASET_NAMES):
        raise ValueError("search plan requires exactly three frozen dataset caches")
    rows: list[dict[str, Any]] = []
    for dataset in DATASET_NAMES:
        value = cache_rows[dataset]
        for candidate in value["all_sets"]["sets"]:
            for seed_ordinal, seed in enumerate(SEARCH_MODEL_SEEDS):
                rows.append(
                    {
                        "stage": SEARCH_STAGE,
                        "physical_run_id": physical_run_id(dataset, int(candidate["candidate_set_ordinal"]), seed_ordinal),
                        "dataset": dataset,
                        "cache_identity": value["cache_identity"],
                        "path_library_identity": value["path_library_identity"],
                        "candidate_set_id": candidate["candidate_set_id"],
                        "candidate_set_hash": candidate["candidate_set_hash"],
                        "candidate_set_ordinal": int(candidate["candidate_set_ordinal"]),
                        "path_ids": list(candidate["path_ids"]),
                        "model_seed": int(seed),
                        "model_seed_ordinal": seed_ordinal,
                        "continuation_from_update": 0,
                        "endpoint_update": SEARCH_ENDPOINTS[0],
                        "checkpoint_schedule": list(SEARCH_ENDPOINTS),
                        "execution_action": "TRAIN_FROM_ZERO_TO_0300",
                        "labels_mounted": False,
                        "downstream_evaluations_during_training": 0,
                        "eligible_for_selector_development": True,
                        "eligible_for_preliminary_scientific_result": False,
                        "attempt_index": 0,
                        "status": "PENDING_GPU",
                    }
                )
    expected = sum(len(cache_rows[name]["all_sets"]["sets"]) for name in DATASET_NAMES) * 3
    if len(rows) != expected or len({row["physical_run_id"] for row in rows}) != expected:
        raise RuntimeError("search plan is not an exact unique set×seed matrix")
    seal_basis = {
        "stage": SEARCH_STAGE,
        "row_count": len(rows),
        "rows_identity": canonical_json_hash([_canonical_serialized_plan_row(row) for row in rows]),
        "per_dataset_rows": {name: sum(row["dataset"] == name for row in rows) for name in DATASET_NAMES},
        "search_model_seeds": list(SEARCH_MODEL_SEEDS),
        "labels_mounted": False,
        "training_from_zero": True,
        "endpoint_update": SEARCH_ENDPOINTS[0],
        "run_level_parallel_only": True,
        "maximum_gpu_count": MAX_GPUS,
        "formal_preliminary_evaluation_started": False,
    }
    seal = envelope(
        "metachorus.v53.m6f-m7f.search-plan-seal.v1",
        "SEALED_BEFORE_SEARCH_GPU_TRAINING",
        seal_basis,
    )
    sealed = [{**row, "plan_seal_identity": seal["identity"]} for row in rows]
    return sealed, seal


def build_canary_plan(
    cache_rows: Mapping[str, Mapping[str, Any]],
    *,
    source_commit: str,
    source_tree: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select one deterministic worst-case structural set per dataset.

    The engineering canary uses the set with the largest summed frozen
    projection edge-index footprint.  Ties are broken by the candidate hash;
    no labels or scientific metrics are consulted.
    """

    if set(cache_rows) != set(DATASET_NAMES):
        raise ValueError("canary plan requires exactly three frozen dataset caches")
    rows: list[dict[str, Any]] = []
    for dataset in DATASET_NAMES:
        value = cache_rows[dataset]
        path_bytes = {
            str(row["path_id"]): int(row["gpu_edge_index_bytes"])
            for row in value["path_library"]["paths"]
        }
        candidates = list(value["all_sets"]["sets"])
        if not candidates:
            raise ValueError(f"{dataset} has no canary candidate")
        candidate = sorted(
            candidates,
            key=lambda row: (
                -sum(path_bytes[str(path_id)] for path_id in row["path_ids"]),
                str(row["candidate_set_hash"]),
            ),
        )[0]
        rows.append(
            {
                "stage": "CANARY",
                "physical_run_id": canary_run_id(dataset, int(candidate["candidate_set_ordinal"]), 0),
                "dataset": dataset,
                "cache_identity": value["cache_identity"],
                "path_library_identity": value["path_library_identity"],
                "candidate_set_id": candidate["candidate_set_id"],
                "candidate_set_hash": candidate["candidate_set_hash"],
                "candidate_set_ordinal": int(candidate["candidate_set_ordinal"]),
                "path_ids": list(candidate["path_ids"]),
                "model_seed": int(SEARCH_MODEL_SEEDS[0]),
                "model_seed_ordinal": 0,
                "continuation_from_update": 0,
                "endpoint_update": 20,
                "checkpoint_schedule": [20],
                "execution_action": "ENGINEERING_CANARY_FROM_ZERO_TO_0020",
                "worst_case_gpu_edge_index_bytes": sum(
                    path_bytes[str(path_id)] for path_id in candidate["path_ids"]
                ),
                "labels_mounted": False,
                "downstream_evaluations_during_training": 0,
                "eligible_for_selector_development": False,
                "eligible_for_preliminary_scientific_result": False,
                "attempt_index": 0,
                "source_commit": str(source_commit),
                "source_tree": str(source_tree),
                "status": "PENDING_GPU",
            }
        )
    seal_basis = {
        "row_count": len(rows),
        "rows_identity": canonical_json_hash([_canonical_serialized_plan_row(row) for row in rows]),
        "datasets": list(DATASET_NAMES),
        "selection_rule": "MAX_SUM_FROZEN_PROJECTION_EDGE_INDEX_BYTES_THEN_HASH",
        "engineering_seed": int(SEARCH_MODEL_SEEDS[0]),
        "endpoint_update": 20,
        "labels_mounted": False,
        "scientific_metrics_allowed": False,
        "formal_search_authorized": False,
        "maximum_gpu_count": MAX_GPUS,
        "source_commit": str(source_commit),
        "source_tree": str(source_tree),
    }
    seal = envelope(
        "metachorus.v53.m6f-m7f.canary-plan-seal.v1",
        "SEALED_BEFORE_ENGINEERING_CANARIES",
        seal_basis,
    )
    return [{**row, "plan_seal_identity": seal["identity"]} for row in rows], seal


def deterministic_explore_indices(set_count: int, *, dataset_identity: str) -> list[int]:
    """Select four deterministic ordinal bands without reading labels or metrics."""

    target = min(40, math.ceil(0.20 * set_count))
    if target <= 0:
        return []
    bands = np.array_split(np.arange(set_count, dtype=np.int64), 4)
    result: list[int] = []
    for band_index, band in enumerate(bands):
        quota = target // 4 + int(band_index < target % 4)
        ranked = sorted(
            map(int, band),
            key=lambda ordinal: canonical_json_hash(
                {
                    "dataset_identity": dataset_identity,
                    "namespace": "M6F_R1_STRUCTURAL_EXPLORATION",
                    "candidate_set_ordinal": ordinal,
                }
            ),
        )
        result.extend(ranked[:quota])
    if len(result) != target or len(set(result)) != target:
        raise RuntimeError("deterministic exploration sampler violated its budget")
    return sorted(result)


def build_preliminary_matrix(
    selected_rows: Mapping[str, Mapping[str, Any]],
    *,
    dataset_statuses: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Build logical role rows and hash-deduplicated physical runs."""

    if dataset_statuses is None:
        if set(selected_rows) != set(DATASET_NAMES):
            raise ValueError("preliminary matrix requires a sealed selection per dataset")
        active_datasets = tuple(DATASET_NAMES)
        blocked_datasets: tuple[str, ...] = ()
    else:
        if set(dataset_statuses) != set(DATASET_NAMES):
            raise ValueError("preliminary matrix dataset statuses are incomplete")
        blocked_datasets = tuple(
            dataset
            for dataset in DATASET_NAMES
            if dataset_statuses[dataset].get("status") == "FAST_SEARCH_BUDGET_BLOCKED"
        )
        active_datasets = tuple(dataset for dataset in DATASET_NAMES if dataset not in blocked_datasets)
        if not active_datasets or set(selected_rows) != set(active_datasets):
            raise ValueError("preliminary selections differ from the active dataset subset")
    logical: list[dict[str, Any]] = []
    for dataset in DATASET_NAMES:
        if dataset in blocked_datasets:
            for role in ("FINAL_FROZEN_METHOD", "MANUAL", "CURRENT_AUTOMATIC"):
                for seed_ordinal, seed in enumerate(EVAL_MODEL_SEEDS):
                    logical.append(
                        {
                            "dataset": dataset,
                            "logical_role": role,
                            "availability": "FAST_SEARCH_BUDGET_BLOCKED",
                            "candidate_set_id": "",
                            "candidate_set_hash": "",
                            "candidate_set_ordinal": "",
                            "path_ids": [],
                            "model_seed": int(seed),
                            "model_seed_ordinal": seed_ordinal,
                            "training_from_zero": False,
                            "endpoint_update": 0,
                            "status": "FAST_SEARCH_BUDGET_BLOCKED",
                        }
                    )
            continue
        selection = selected_rows[dataset]
        roles: list[tuple[str, Mapping[str, Any] | None, str]] = [
            ("FINAL_FROZEN_METHOD", selection["selected_set"], "AVAILABLE"),
            ("MANUAL", selection["manual_set"], "AVAILABLE"),
        ]
        current = selection.get("current_automatic")
        if isinstance(current, Mapping) and current.get("status") == "EXACT_IDENTITY_RESOLVED":
            roles.append(("CURRENT_AUTOMATIC", current, "AVAILABLE"))
        else:
            roles.append(("CURRENT_AUTOMATIC", None, "NOT_AVAILABLE_EXACT_IDENTITY"))
        for role, candidate, availability in roles:
            for seed_ordinal, seed in enumerate(EVAL_MODEL_SEEDS):
                logical.append(
                    {
                        "dataset": dataset,
                        "logical_role": role,
                        "availability": availability,
                        "candidate_set_id": "" if candidate is None else candidate["candidate_set_id"],
                        "candidate_set_hash": "" if candidate is None else candidate["candidate_set_hash"],
                        "candidate_set_ordinal": "" if candidate is None else int(candidate["candidate_set_ordinal"]),
                        "path_ids": [] if candidate is None else list(candidate["path_ids"]),
                        "model_seed": int(seed),
                        "model_seed_ordinal": seed_ordinal,
                        "training_from_zero": candidate is not None,
                        "endpoint_update": PRELIM_ENDPOINT if candidate is not None else 0,
                        "status": "PENDING_GPU" if candidate is not None else "NOT_AVAILABLE_EXACT_IDENTITY",
                    }
                )
    if len(logical) != 27:
        raise RuntimeError("preliminary logical matrix must contain exactly 27 role rows")
    physical_by_identity: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in logical:
        if row["availability"] != "AVAILABLE":
            continue
        identity = (str(row["dataset"]), str(row["candidate_set_hash"]), int(row["model_seed"]))
        value = physical_by_identity.setdefault(
            identity,
            {
                "dataset": row["dataset"],
                "candidate_set_id": row["candidate_set_id"],
                "candidate_set_hash": row["candidate_set_hash"],
                "candidate_set_ordinal": row["candidate_set_ordinal"],
                "path_ids": row["path_ids"],
                "model_seed": row["model_seed"],
                "model_seed_ordinal": row["model_seed_ordinal"],
                "logical_roles": [],
                "continuation_from_update": 0,
                "endpoint_update": PRELIM_ENDPOINT,
                "execution_action": "FRESH_PRELIMINARY_TRAIN_FROM_ZERO_TO_3000",
                "search_checkpoint_reused": False,
                "search_optimizer_reused": False,
                "search_rng_state_reused": False,
                "labels_mounted": False,
                "downstream_evaluations_during_training": 0,
                "status": "PENDING_GPU",
            },
        )
        value["logical_roles"].append(row["logical_role"])
    physical: list[dict[str, Any]] = []
    for ordinal, value in enumerate(sorted(physical_by_identity.values(), key=lambda row: (row["dataset"], row["candidate_set_hash"], row["model_seed"]))):
        dataset = str(value["dataset"])
        physical.append(
            {
                **value,
                "physical_run_id": physical_run_id(dataset, int(value["candidate_set_ordinal"]), int(value["model_seed_ordinal"]), prefix="M7F"),
                "physical_run_ordinal": ordinal,
            }
        )
    seal_basis = {
        "logical_row_count": len(logical),
        "physical_row_count": len(physical),
        "logical_rows_identity": canonical_json_hash(logical),
        "physical_rows_identity": canonical_json_hash(physical),
        "evaluation_model_seeds": list(EVAL_MODEL_SEEDS),
        "active_datasets": list(active_datasets),
        "blocked_datasets": list(blocked_datasets),
        "physical_deduplication_key": ["dataset", "candidate_set_hash", "model_seed"],
        "fresh_from_zero": True,
        "search_state_reuse": False,
        "labels_mounted": False,
        "maximum_gpu_count": MAX_GPUS,
    }
    return logical, physical, envelope(
        "metachorus.v53.m6f-m7f.preliminary-matrix-seal.v1",
        "SEALED_AFTER_SELECTION_BEFORE_PRELIMINARY_GPU_TRAINING",
        seal_basis,
    )



import numpy as np


__all__ = [
    "SEARCH_STAGE",
    "load_cache_resolution",
    "physical_run_id",
    "canary_run_id",
    "build_search_plan",
    "build_canary_plan",
    "deterministic_explore_indices",
    "build_preliminary_matrix",
]
