"""Frozen DBLP ten-seed paired confirmation planning."""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from .common import canonical_json_hash, envelope
from .constants import (
    ALL_SETS_IDENTITY,
    CACHE_IDENTITY,
    CANARY_ENDPOINT,
    CONFIRMATION_CHECKPOINT_SCHEDULE,
    CONFIRMATION_ENDPOINT,
    CONFIRMATION_MODEL_SEEDS,
    CONFIRMATION_ROLES,
    DATASET,
    DATASET_IDENTITY,
    EXPECTED_GPU_COUNT,
    LABEL_ROLE_MANIFEST_IDENTITY,
    MANUAL_PATH_IDS,
    MANUAL_SET_HASH,
    MANUAL_SET_ID,
    MANUAL_SET_ORDINAL,
    METHOD_FREEZE_IDENTITY,
    PATH_LIBRARY_IDENTITY,
    SELECTED_PATH_IDS,
    SELECTED_SET_HASH,
    SELECTED_SET_ID,
    SELECTED_SET_ORDINAL,
)
from metachorus_v53.m7f_h3.planning import (
    load_dblp_cache as _load_h3_dblp_cache,
    resolve_all_sets as _resolve_h3_all_sets,
)


PLAN_SCHEMA = "metachorus.v53.m7f-h4.dblp-plan-seal.v1"
STAGE_STATUS = {
    "CANARY": "SEALED_BEFORE_ENGINEERING_CANARIES",
    "CONFIRMATION": "SEALED_BEFORE_10SEED_CONFIRMATION_GPU_TRAINING",
}


def canonical_plan_row(row: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(row)
    for field in (
        "candidate_set_ordinal",
        "model_seed",
        "model_seed_ordinal",
        "start_update",
        "continuation_from_update",
        "endpoint_update",
        "attempt_index",
        "preferred_gpu_slot",
        "queue_ordinal",
        "downstream_evaluations_during_training",
    ):
        if field in value and value[field] not in (None, ""):
            value[field] = int(value[field])
    for field in ("path_ids", "checkpoint_schedule"):
        if field in value and isinstance(value[field], str):
            value[field] = json.loads(value[field])
        if field in value:
            value[field] = list(value[field])
    for field in (
        "labels_mounted",
        "training_from_zero",
        "search_checkpoint_reused",
        "eligible_for_scientific_result",
        "eligible_for_selector_development",
    ):
        if field in value and not isinstance(value[field], bool):
            token = str(value[field]).strip().casefold()
            if token not in {"true", "false"}:
                raise ValueError(f"invalid boolean plan field {field}")
            value[field] = token == "true"
    return value


def _unsealed(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        canonical_plan_row({key: item for key, item in row.items() if key != "plan_seal_identity"})
        for row in rows
    ]


def load_dblp_cache(cache_root: str) -> dict[str, Any]:
    cache = _load_h3_dblp_cache(cache_root)
    if (
        cache.get("cache_identity") != CACHE_IDENTITY
        or cache.get("path_library_identity") != PATH_LIBRARY_IDENTITY
        or cache.get("all_sets_identity") != ALL_SETS_IDENTITY
    ):
        raise ValueError("M7F-H4 DBLP cache identity differs from the frozen H3 parent")
    return cache


def resolve_all_sets(cache: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = _resolve_h3_all_sets(cache)
    selected = [row for row in rows if row["candidate_set_hash"] == SELECTED_SET_HASH]
    manual = [row for row in rows if row["candidate_set_hash"] == MANUAL_SET_HASH]
    if (
        len(selected) != 1
        or selected[0]["candidate_set_id"] != SELECTED_SET_ID
        or int(selected[0]["candidate_set_ordinal"]) != SELECTED_SET_ORDINAL
        or tuple(selected[0]["path_ids"]) != SELECTED_PATH_IDS
        or len(manual) != 1
        or manual[0]["candidate_set_id"] != MANUAL_SET_ID
        or int(manual[0]["candidate_set_ordinal"]) != MANUAL_SET_ORDINAL
        or tuple(manual[0]["path_ids"]) != MANUAL_PATH_IDS
    ):
        raise ValueError("M7F-H4 selected or Manual set identity is not exact")
    return rows


def _candidate(cache: Mapping[str, Any], role: str) -> dict[str, Any]:
    expected = {
        "DBLP_SELECTED": (SELECTED_SET_ID, SELECTED_SET_HASH, SELECTED_SET_ORDINAL, SELECTED_PATH_IDS),
        "DBLP_MANUAL": (MANUAL_SET_ID, MANUAL_SET_HASH, MANUAL_SET_ORDINAL, MANUAL_PATH_IDS),
    }[role]
    matches = [
        dict(row)
        for row in cache["all_sets"]["sets"]
        if row.get("candidate_set_id") == expected[0]
        and row.get("candidate_set_hash") == expected[1]
    ]
    if (
        len(matches) != 1
        or int(matches[0].get("candidate_set_ordinal", -1)) != expected[2]
        or tuple(matches[0].get("path_ids", [])) != expected[3]
    ):
        raise ValueError(f"M7F-H4 frozen {role} set is not uniquely resolved")
    return matches[0]


def pair_gpu_slot(seed_ordinal: int) -> int:
    return int(seed_ordinal) % EXPECTED_GPU_COUNT


def pair_runtime_domain(seed_ordinal: int) -> str:
    return f"M7F-H4-DBLP-PAIR-GPU{pair_gpu_slot(seed_ordinal)}"


def _physical_run_id(role: str, candidate_ordinal: int, seed_ordinal: int, *, canary: bool) -> str:
    arm = {"DBLP_SELECTED": "SELECTED", "DBLP_MANUAL": "MANUAL"}[role]
    phase = "CANARY" if canary else "CONFIRM"
    return f"M7F-H4-{phase}-DBLP-{arm}-C{candidate_ordinal:03d}-F{seed_ordinal:02d}"


def _training_config_identity(
    *, role: str, candidate: Mapping[str, Any], model_seed: int, endpoint: int
) -> str:
    return canonical_json_hash(
        {
            "dataset": DATASET,
            "dataset_identity": DATASET_IDENTITY,
            "cache_identity": CACHE_IDENTITY,
            "path_library_identity": PATH_LIBRARY_IDENTITY,
            "all_sets_identity": ALL_SETS_IDENTITY,
            "method_freeze_identity": METHOD_FREEZE_IDENTITY,
            "logical_role": role,
            "candidate_set_id": candidate["candidate_set_id"],
            "candidate_set_hash": candidate["candidate_set_hash"],
            "path_ids": list(candidate["path_ids"]),
            "model_seed": int(model_seed),
            "start_update": 0,
            "endpoint_update": int(endpoint),
            "label_role_identity": LABEL_ROLE_MANIFEST_IDENTITY,
            "labels_mounted": False,
        }
    )


def _row(
    cache: Mapping[str, Any], *, role: str, seed_ordinal: int, canary: bool, queue_ordinal: int
) -> dict[str, Any]:
    candidate = _candidate(cache, role)
    seed = int(CONFIRMATION_MODEL_SEEDS[seed_ordinal])
    endpoint = CANARY_ENDPOINT if canary else CONFIRMATION_ENDPOINT
    stage = "CANARY" if canary else "CONFIRMATION"
    return {
        "stage": "CANARY" if canary else "PRELIMINARY",
        "logical_stage": stage,
        "logical_role": role,
        "physical_run_id": _physical_run_id(
            role, int(candidate["candidate_set_ordinal"]), seed_ordinal, canary=canary
        ),
        "dataset": DATASET,
        "dataset_identity": DATASET_IDENTITY,
        "cache_identity": cache["cache_identity"],
        "path_library_identity": cache["path_library_identity"],
        "all_sets_identity": cache["all_sets_identity"],
        "candidate_set_id": candidate["candidate_set_id"],
        "candidate_set_hash": candidate["candidate_set_hash"],
        "candidate_set_ordinal": int(candidate["candidate_set_ordinal"]),
        "path_ids": list(candidate["path_ids"]),
        "model_seed": seed,
        "model_seed_ordinal": int(seed_ordinal),
        "start_update": 0,
        "continuation_from_update": 0,
        "endpoint_update": endpoint,
        "checkpoint_schedule": [endpoint] if canary else list(CONFIRMATION_CHECKPOINT_SCHEDULE),
        "execution_action": (
            "ENGINEERING_CANARY_FROM_ZERO_TO_0020"
            if canary
            else "FRESH_10SEED_CONFIRMATION_FROM_ZERO_TO_3000"
        ),
        "training_config_identity": _training_config_identity(
            role=role, candidate=candidate, model_seed=seed, endpoint=endpoint
        ),
        "label_role_identity": LABEL_ROLE_MANIFEST_IDENTITY,
        "planned_gpu_pair_group": f"DBLP-H4-SEED-{seed}",
        "planned_runtime_domain": pair_runtime_domain(seed_ordinal),
        "preferred_gpu_slot": pair_gpu_slot(seed_ordinal),
        "queue_ordinal": int(queue_ordinal),
        "training_from_zero": True,
        "search_checkpoint_reused": False,
        "labels_mounted": False,
        "downstream_evaluations_during_training": 0,
        "eligible_for_scientific_result": not canary,
        "eligible_for_selector_development": False,
        "attempt_index": 0,
        "status": "PENDING_GPU",
    }


def _seal_plan(logical_stage: str, rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    normalized = _unsealed(rows)
    basis = {
        "logical_stage": logical_stage,
        "dataset": DATASET,
        "row_count": len(normalized),
        "rows_identity": canonical_json_hash(normalized),
        "model_seeds": sorted({int(row["model_seed"]) for row in normalized}),
        "logical_roles": sorted({str(row["logical_role"]) for row in normalized}),
        "start_update": 0,
        "endpoint_update": CANARY_ENDPOINT if logical_stage == "CANARY" else CONFIRMATION_ENDPOINT,
        "pair_same_seed_same_gpu_domain": True,
        "training_from_zero": True,
        "labels_mounted": False,
        "downstream_evaluations_during_training": 0,
        "run_level_parallel_only": True,
        "maximum_gpu_count": EXPECTED_GPU_COUNT,
        "one_process_per_gpu": True,
        "ddp_used": False,
        "data_parallel_used": False,
        "torchrun_used": False,
        "further_search": False,
        "algorithm_retuning": False,
    }
    seal = envelope(PLAN_SCHEMA, STAGE_STATUS[logical_stage], basis)
    sealed_rows = [{**row, "plan_seal_identity": seal["identity"]} for row in normalized]
    validate_plan(sealed_rows, seal, logical_stage=logical_stage)
    return sealed_rows, seal


def build_canary_plan(cache: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    specs = (
        ("DBLP_SELECTED", 0),
        ("DBLP_MANUAL", 0),
        ("DBLP_SELECTED", 1),
        ("DBLP_MANUAL", 1),
    )
    rows = [
        _row(cache, role=role, seed_ordinal=seed_ordinal, canary=True, queue_ordinal=ordinal)
        for ordinal, (role, seed_ordinal) in enumerate(specs)
    ]
    
    for ordinal, row in enumerate(rows):
        row["preferred_gpu_slot"] = ordinal
        row["planned_runtime_domain"] = f"M7F-H4-CANARY-GPU{ordinal}"
    return _seal_plan("CANARY", rows)


def build_confirmation_plan(cache: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed_ordinal, _seed in enumerate(CONFIRMATION_MODEL_SEEDS):
        for role in CONFIRMATION_ROLES:
            rows.append(
                _row(
                    cache,
                    role=role,
                    seed_ordinal=seed_ordinal,
                    canary=False,
                    queue_ordinal=len(rows),
                )
            )
    return _seal_plan("CONFIRMATION", rows)


def validate_plan(
    rows: Sequence[Mapping[str, Any]], seal: Mapping[str, Any], *, logical_stage: str
) -> list[dict[str, Any]]:
    if logical_stage not in STAGE_STATUS:
        raise ValueError("unsupported M7F-H4 logical stage")
    normalized = [canonical_plan_row(row) for row in rows]
    basis = seal.get("basis", {})
    expected_count = 4 if logical_stage == "CANARY" else 20
    expected_endpoint = CANARY_ENDPOINT if logical_stage == "CANARY" else CONFIRMATION_ENDPOINT
    expected_seeds = set(CONFIRMATION_MODEL_SEEDS[:2] if logical_stage == "CANARY" else CONFIRMATION_MODEL_SEEDS)
    if (
        seal.get("schema_version") != PLAN_SCHEMA
        or seal.get("status") != STAGE_STATUS[logical_stage]
        or seal.get("identity") != canonical_json_hash(basis)
        or basis.get("logical_stage") != logical_stage
        or basis.get("dataset") != DATASET
        or int(basis.get("row_count", -1)) != expected_count
        or basis.get("rows_identity") != canonical_json_hash(_unsealed(normalized))
        or set(map(int, basis.get("model_seeds", []))) != expected_seeds
        or set(map(str, basis.get("logical_roles", []))) != set(CONFIRMATION_ROLES)
        or int(basis.get("start_update", -1)) != 0
        or int(basis.get("endpoint_update", -1)) != expected_endpoint
        or basis.get("pair_same_seed_same_gpu_domain") is not True
        or basis.get("training_from_zero") is not True
        or basis.get("labels_mounted") is not False
        or basis.get("further_search") is not False
        or basis.get("algorithm_retuning") is not False
        or len(normalized) != expected_count
        or len({str(row.get("physical_run_id")) for row in normalized}) != expected_count
        or any(row.get("plan_seal_identity") != seal.get("identity") for row in normalized)
        or any(row.get("logical_stage") != logical_stage for row in normalized)
        or any(row.get("dataset") != DATASET for row in normalized)
        or any(int(row.get("start_update", -1)) != 0 for row in normalized)
        or any(int(row.get("continuation_from_update", -1)) != 0 for row in normalized)
        or any(int(row.get("endpoint_update", -1)) != expected_endpoint for row in normalized)
        or any(row.get("training_from_zero") is not True for row in normalized)
        or any(row.get("search_checkpoint_reused") is not False for row in normalized)
        or any(row.get("labels_mounted") is not False for row in normalized)
        or any(int(row.get("downstream_evaluations_during_training", -1)) != 0 for row in normalized)
        or any(int(row.get("attempt_index", -1)) not in {0, 1} for row in normalized)
    ):
        raise ValueError("M7F-H4 run plan/seal identity is invalid")
    for seed_ordinal, seed in enumerate(CONFIRMATION_MODEL_SEEDS):
        pair = [row for row in normalized if int(row["model_seed"]) == int(seed)]
        if logical_stage == "CANARY" and seed_ordinal >= 2:
            if pair:
                raise ValueError("M7F-H4 canary uses an unauthorized confirmation seed")
            continue
        if (
            len(pair) != 2
            or {str(row["logical_role"]) for row in pair} != set(CONFIRMATION_ROLES)
            or len({int(row["preferred_gpu_slot"]) for row in pair}) != (2 if logical_stage == "CANARY" else 1)
            or (
                logical_stage == "CONFIRMATION"
                and len({str(row["planned_runtime_domain"]) for row in pair}) != 1
            )
        ):
            raise ValueError("M7F-H4 same-seed selected/Manual pair is incomplete or crosses its runtime domain")
    for role, expected_hash, expected_id, expected_paths in (
        ("DBLP_SELECTED", SELECTED_SET_HASH, SELECTED_SET_ID, SELECTED_PATH_IDS),
        ("DBLP_MANUAL", MANUAL_SET_HASH, MANUAL_SET_ID, MANUAL_PATH_IDS),
    ):
        cohort = [row for row in normalized if row["logical_role"] == role]
        expected_role_count = 2 if logical_stage == "CANARY" else 10
        if (
            len(cohort) != expected_role_count
            or {row["candidate_set_hash"] for row in cohort} != {expected_hash}
            or {row["candidate_set_id"] for row in cohort} != {expected_id}
            or any(tuple(row["path_ids"]) != expected_paths for row in cohort)
        ):
            raise ValueError(f"M7F-H4 {role} cohort identity is invalid")
    return normalized


__all__ = [
    "PLAN_SCHEMA",
    "STAGE_STATUS",
    "canonical_plan_row",
    "load_dblp_cache",
    "resolve_all_sets",
    "pair_gpu_slot",
    "pair_runtime_domain",
    "build_canary_plan",
    "build_confirmation_plan",
    "validate_plan",
]
