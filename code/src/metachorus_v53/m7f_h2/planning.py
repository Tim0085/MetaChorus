"""Frozen AMiner-only planning for the M7F-H2 campaign."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from metachorus_v53.audit.common import canonical_json_hash
from metachorus_v53.m6f_m7f_fast.planning import (
    canary_run_id,
    load_cache_resolution,
    physical_run_id,
)
from metachorus_v53.m6f_m7f_fast.training import validate_run_marker

from .common import envelope, read_json, sha256_file
from .constants import (
    ALL_SETS_IDENTITY,
    CACHE_IDENTITY,
    CANARY_ENDPOINT,
    DATASET,
    DATASET_IDENTITY,
    EVALUATION_MODEL_SEEDS,
    EXPECTED_GPU_COUNT,
    LEGAL_PATH_COUNT,
    LEGAL_SET_COUNT,
    LONGEST_CANARY_SET_HASH,
    LONGEST_CANARY_SET_ID,
    MANUAL_SET_HASH,
    MANUAL_SET_ID,
    PATH_LIBRARY_IDENTITY,
    PRELIM_ENDPOINT,
    SEARCH_ENDPOINTS,
    SEARCH_MODEL_SEEDS,
    SET_CARDINALITY,
    SHORTEST_CANARY_SET_HASH,
    SHORTEST_CANARY_SET_ID,
)


PLAN_SCHEMA = "metachorus.v53.m7f-h2.aminer-plan-seal.v1"
STAGE_STATUS = {
    "CANARY": "SEALED_BEFORE_ENGINEERING_CANARIES",
    "R1": "SEALED_BEFORE_R1_GPU_TRAINING",
    "R2": "SEALED_BEFORE_SEARCH_CONTINUATION",
    "R3": "SEALED_BEFORE_SEARCH_CONTINUATION",
    "R4": "SEALED_BEFORE_SEARCH_CONTINUATION",
    "PRELIMINARY": "SEALED_BEFORE_PRELIMINARY_GPU_TRAINING",
}


def canonical_plan_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Recover the exact typed plan identity after a CSV round trip."""

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
        "queue_ordinal",
        "preferred_gpu_slot",
    ):
        if field in value and value[field] not in (None, ""):
            value[field] = int(value[field])
    for field in ("path_ids", "checkpoint_schedule", "logical_roles"):
        if field in value and isinstance(value[field], str):
            value[field] = json.loads(value[field])
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
        if field in value and not isinstance(value[field], bool):
            token = str(value[field]).strip().casefold()
            if token not in {"true", "false"}:
                raise ValueError(f"invalid boolean plan field {field}")
            value[field] = token == "true"
    return value


def _unsealed(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        canonical_plan_row({key: value for key, value in row.items() if key != "plan_seal_identity"})
        for row in rows
    ]


def _seal_plan(
    logical_stage: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    endpoint_update: int,
    continuation_from_update: int,
    extra_basis: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if logical_stage not in STAGE_STATUS:
        raise ValueError(f"unsupported H2 logical stage: {logical_stage}")
    normalized = _unsealed(rows)
    basis = {
        "logical_stage": logical_stage,
        "dataset": DATASET,
        "row_count": len(normalized),
        "rows_identity": canonical_json_hash(normalized),
        "continuation_from_update": int(continuation_from_update),
        "endpoint_update": int(endpoint_update),
        "search_model_seeds": list(SEARCH_MODEL_SEEDS) if logical_stage in {"R1", "R2", "R3", "R4"} else [],
        "canary_model_seeds": sorted({int(row["model_seed"]) for row in normalized}) if logical_stage == "CANARY" else [],
        "evaluation_model_seeds": list(EVALUATION_MODEL_SEEDS) if logical_stage == "PRELIMINARY" else [],
        "labels_mounted": False,
        "prelim_locked_report_access": False,
        "run_level_parallel_only": True,
        "maximum_gpu_count": EXPECTED_GPU_COUNT,
        "one_process_per_gpu": True,
        "ddp_used": False,
        "data_parallel_used": False,
        "torchrun_used": False,
        **dict(extra_basis or {}),
    }
    seal = envelope(PLAN_SCHEMA, STAGE_STATUS[logical_stage], basis)
    return [{**row, "plan_seal_identity": seal["identity"]} for row in normalized], seal


def validate_plan(
    rows: Sequence[Mapping[str, Any]],
    seal: Mapping[str, Any],
    *,
    logical_stage: str,
) -> list[dict[str, Any]]:
    if logical_stage not in STAGE_STATUS:
        raise ValueError(f"unsupported H2 logical stage: {logical_stage}")
    normalized = [canonical_plan_row(row) for row in rows]
    basis = seal.get("basis", {})
    unsealed = _unsealed(normalized)
    endpoint = int(basis.get("endpoint_update", -1))
    continuation = int(basis.get("continuation_from_update", -1))
    row_endpoints = {int(row.get("endpoint_update", -1)) for row in normalized}
    row_starts = {int(row.get("continuation_from_update", -1)) for row in normalized}
    row_seeds = {int(row.get("model_seed", -1)) for row in normalized}
    expected_search_seeds = set(SEARCH_MODEL_SEEDS)
    expected_eval_seeds = set(EVALUATION_MODEL_SEEDS)
    if (
        seal.get("schema_version") != PLAN_SCHEMA
        or seal.get("status") != STAGE_STATUS[logical_stage]
        or seal.get("identity") != canonical_json_hash(basis)
        or basis.get("logical_stage") != logical_stage
        or basis.get("dataset") != DATASET
        or int(basis.get("row_count", -1)) != len(normalized)
        or basis.get("rows_identity") != canonical_json_hash(unsealed)
        or basis.get("labels_mounted") is not False
        or basis.get("prelim_locked_report_access") is not False
        or basis.get("run_level_parallel_only") is not True
        or int(basis.get("maximum_gpu_count", -1)) != EXPECTED_GPU_COUNT
        or basis.get("one_process_per_gpu") is not True
        or basis.get("ddp_used") is not False
        or basis.get("data_parallel_used") is not False
        or basis.get("torchrun_used") is not False
        or (normalized and row_endpoints != {endpoint})
        or (normalized and row_starts != {continuation})
        or any(row.get("plan_seal_identity") != seal.get("identity") for row in normalized)
        or any(row.get("dataset") != DATASET for row in normalized)
        or any(row.get("logical_stage") != logical_stage for row in normalized)
        or any(row.get("labels_mounted") is not False for row in normalized)
        or any(int(row.get("downstream_evaluations_during_training", -1)) != 0 for row in normalized)
        or any(int(row.get("attempt_index", -1)) not in {0, 1} for row in normalized)
    ):
        raise ValueError("M7F-H2 run plan/seal identity is invalid")
    if logical_stage in {"R1", "R2", "R3", "R4"}:
        if (
            row_seeds != expected_search_seeds
            or set(map(int, basis.get("search_model_seeds", []))) != expected_search_seeds
            or basis.get("canary_model_seeds") != []
            or basis.get("evaluation_model_seeds") != []
        ):
            raise ValueError("M7F-H2 search plan does not use the frozen three-seed namespace")
        for digest in {str(row["candidate_set_hash"]) for row in normalized}:
            cohort = [row for row in normalized if str(row["candidate_set_hash"]) == digest]
            if len(cohort) != 3 or {int(row["model_seed"]) for row in cohort} != expected_search_seeds:
                raise ValueError("M7F-H2 search candidate lacks exactly three frozen seeds")
    elif logical_stage == "CANARY":
        if len(normalized) != 4 or set(map(int, basis.get("canary_model_seeds", []))) != row_seeds:
            raise ValueError("M7F-H2 canary plan is not the frozen four-run matrix")
    elif normalized:
        roles = {str(row.get("logical_role", "")) for row in normalized}
        if (
            row_seeds != expected_eval_seeds
            or set(map(int, basis.get("evaluation_model_seeds", []))) != expected_eval_seeds
            or roles != {"FINAL_FROZEN_METHOD", "MANUAL"}
        ):
            raise ValueError("M7F-H2 preliminary plan is not the paired three-seed matrix")
        for seed in expected_eval_seeds:
            cohort = [row for row in normalized if int(row["model_seed"]) == seed]
            if len(cohort) != 2 or {str(row["logical_role"]) for row in cohort} != roles:
                raise ValueError("M7F-H2 preliminary seed is not paired")
            if len({int(row["preferred_gpu_slot"]) for row in cohort}) != 1:
                raise ValueError("M7F-H2 preliminary pair is not pinned to one GPU slot")
    return normalized


def load_aminer_cache(cache_root: str | Path) -> dict[str, Any]:
    value = load_cache_resolution(cache_root)
    paths = list(value["path_library"].get("paths", []))
    sets = list(value["all_sets"].get("sets", []))
    if (
        value.get("dataset") != DATASET
        or value.get("cache_identity") != CACHE_IDENTITY
        or value.get("path_library_identity") != PATH_LIBRARY_IDENTITY
        or value.get("all_sets_identity") != ALL_SETS_IDENTITY
        or len(paths) != LEGAL_PATH_COUNT
        or len(sets) != LEGAL_SET_COUNT
        or len({str(row.get("candidate_set_hash")) for row in sets}) != LEGAL_SET_COUNT
        or any(len(row.get("path_ids", [])) != SET_CARDINALITY for row in sets)
        or value["manual"].get("candidate_set_id") != MANUAL_SET_ID
        or value["manual"].get("candidate_set_hash") != MANUAL_SET_HASH
    ):
        raise ValueError("AMiner H2 cache/candidate identity mismatch")
    return value


def resolve_all_sets(cache: Mapping[str, Any]) -> list[dict[str, Any]]:
    path_map = {str(row["path_id"]): row for row in cache["path_library"]["paths"]}
    rows: list[dict[str, Any]] = []
    for candidate in cache["all_sets"]["sets"]:
        path_ids = list(map(str, candidate["path_ids"]))
        if len(path_ids) != SET_CARDINALITY or any(path_id not in path_map for path_id in path_ids):
            raise ValueError("AMiner candidate path resolution failed")
        rows.append(
            {
                "dataset": DATASET,
                "dataset_identity": DATASET_IDENTITY,
                "cache_identity": cache["cache_identity"],
                "path_library_identity": cache["path_library_identity"],
                "all_sets_identity": cache["all_sets_identity"],
                "candidate_set_id": candidate["candidate_set_id"],
                "candidate_set_hash": candidate["candidate_set_hash"],
                "candidate_set_ordinal": int(candidate["candidate_set_ordinal"]),
                "path_ids": path_ids,
                "path_count": len(path_ids),
                "projection_hashes": [str(path_map[path_id]["semantic_projection_hash"]) for path_id in path_ids],
                "projection_arrays": [str(path_map[path_id]["array"]) for path_id in path_ids],
                "status": "PASS",
            }
        )
    rows.sort(key=lambda row: int(row["candidate_set_ordinal"]))
    if len(rows) != LEGAL_SET_COUNT or len({row["candidate_set_hash"] for row in rows}) != LEGAL_SET_COUNT:
        raise RuntimeError("AMiner all-set resolution is not exactly 91 unique sets")
    return rows


def _candidate(cache: Mapping[str, Any], *, set_id: str, set_hash: str) -> dict[str, Any]:
    matches = [
        dict(row)
        for row in cache["all_sets"]["sets"]
        if row.get("candidate_set_id") == set_id and row.get("candidate_set_hash") == set_hash
    ]
    if len(matches) != 1:
        raise ValueError(f"frozen candidate identity not found: {set_id}")
    return matches[0]


def _base_search_row(
    cache: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    seed: int,
    seed_ordinal: int,
    endpoint: int,
) -> dict[str, Any]:
    return {
        "stage": "R1_ALL_SETS_0000_TO_0300" if endpoint == SEARCH_ENDPOINTS[0] else "SEARCH",
        "logical_stage": f"R{SEARCH_ENDPOINTS.index(endpoint) + 1}",
        "physical_run_id": physical_run_id(DATASET, int(candidate["candidate_set_ordinal"]), seed_ordinal),
        "dataset": DATASET,
        "cache_identity": cache["cache_identity"],
        "path_library_identity": cache["path_library_identity"],
        "all_sets_identity": cache["all_sets_identity"],
        "candidate_set_id": candidate["candidate_set_id"],
        "candidate_set_hash": candidate["candidate_set_hash"],
        "candidate_set_ordinal": int(candidate["candidate_set_ordinal"]),
        "path_ids": list(candidate["path_ids"]),
        "model_seed": int(seed),
        "model_seed_ordinal": int(seed_ordinal),
        "continuation_from_update": 0 if endpoint == SEARCH_ENDPOINTS[0] else SEARCH_ENDPOINTS[SEARCH_ENDPOINTS.index(endpoint) - 1],
        "endpoint_update": int(endpoint),
        "checkpoint_schedule": list(SEARCH_ENDPOINTS),
        "execution_action": "TRAIN_FROM_ZERO_TO_0300" if endpoint == 300 else "STRICT_CONTINUATION",
        "labels_mounted": False,
        "downstream_evaluations_during_training": 0,
        "eligible_for_selector_development": True,
        "eligible_for_preliminary_scientific_result": False,
        "attempt_index": 0,
        "status": "PENDING_GPU" if endpoint == 300 else "AUTHORIZED",
    }


def build_canary_plan(cache: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    specs = (
        (MANUAL_SET_ID, MANUAL_SET_HASH, SEARCH_MODEL_SEEDS[0], 0, 0),
        (SHORTEST_CANARY_SET_ID, SHORTEST_CANARY_SET_HASH, SEARCH_MODEL_SEEDS[0], 0, 1),
        (LONGEST_CANARY_SET_ID, LONGEST_CANARY_SET_HASH, SEARCH_MODEL_SEEDS[0], 0, 2),
        (MANUAL_SET_ID, MANUAL_SET_HASH, SEARCH_MODEL_SEEDS[1], 1, 3),
    )
    rows: list[dict[str, Any]] = []
    for set_id, set_hash, seed, seed_ordinal, gpu_slot in specs:
        candidate = _candidate(cache, set_id=set_id, set_hash=set_hash)
        rows.append(
            {
                "stage": "CANARY",
                "logical_stage": "CANARY",
                "physical_run_id": canary_run_id(DATASET, int(candidate["candidate_set_ordinal"]), seed_ordinal),
                "dataset": DATASET,
                "cache_identity": cache["cache_identity"],
                "path_library_identity": cache["path_library_identity"],
                "all_sets_identity": cache["all_sets_identity"],
                "candidate_set_id": candidate["candidate_set_id"],
                "candidate_set_hash": candidate["candidate_set_hash"],
                "candidate_set_ordinal": int(candidate["candidate_set_ordinal"]),
                "path_ids": list(candidate["path_ids"]),
                "model_seed": int(seed),
                "model_seed_ordinal": int(seed_ordinal),
                "continuation_from_update": 0,
                "endpoint_update": CANARY_ENDPOINT,
                "checkpoint_schedule": [CANARY_ENDPOINT],
                "execution_action": "ENGINEERING_CANARY_FROM_ZERO_TO_0020",
                "preferred_gpu_slot": gpu_slot,
                "queue_ordinal": gpu_slot,
                "labels_mounted": False,
                "downstream_evaluations_during_training": 0,
                "eligible_for_selector_development": False,
                "eligible_for_preliminary_scientific_result": False,
                "attempt_index": 0,
                "status": "PENDING_GPU",
            }
        )
    return _seal_plan("CANARY", rows, endpoint_update=CANARY_ENDPOINT, continuation_from_update=0)


def build_r1_plan(cache: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = [
        {
            **_base_search_row(cache, candidate, seed=seed, seed_ordinal=seed_ordinal, endpoint=300),
            "queue_ordinal": ordinal * len(SEARCH_MODEL_SEEDS) + seed_ordinal,
        }
        for ordinal, candidate in enumerate(cache["all_sets"]["sets"])
        for seed_ordinal, seed in enumerate(SEARCH_MODEL_SEEDS)
    ]
    if len(rows) != 273 or len({row["physical_run_id"] for row in rows}) != 273:
        raise RuntimeError("AMiner H2 R1 must contain exactly 273 unique physical runs")
    return _seal_plan(
        "R1",
        rows,
        endpoint_update=300,
        continuation_from_update=0,
        extra_basis={"legal_set_count": LEGAL_SET_COUNT, "all_sets_identity": cache["all_sets_identity"]},
    )


def _completion_marker(result: Path, row: Mapping[str, Any], endpoint: int) -> tuple[Path, int]:
    run_parent = result / "search_runs" / str(row["physical_run_id"])
    name = f"M6F_M7F_SEARCH_{endpoint:06d}_COMPLETE.json"
    markers = [
        attempt / name
        for attempt in sorted(run_parent.glob("attempt_*"))
        if (attempt / name).is_file() and not (attempt / name).is_symlink()
    ]
    if len(markers) != 1:
        raise RuntimeError(f"H2 continuation source is not unique: {row['physical_run_id']}")
    return markers[0], int(markers[0].parent.name.removeprefix("attempt_"))


def build_continuation_plan(
    *,
    prior_rows: Sequence[Mapping[str, Any]],
    selected_candidate_hashes: set[str],
    result_root: str | Path,
    next_endpoint: int,
    prior_seal_identity: str,
    selection_seal: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if next_endpoint not in SEARCH_ENDPOINTS[1:]:
        raise ValueError("invalid H2 continuation endpoint")
    prior_endpoint = SEARCH_ENDPOINTS[SEARCH_ENDPOINTS.index(next_endpoint) - 1]
    logical_stage = f"R{SEARCH_ENDPOINTS.index(next_endpoint) + 1}"
    normalized = [canonical_plan_row(row) for row in prior_rows]
    selected = set(map(str, selected_candidate_hashes))
    selection_basis = selection_seal.get("basis", {})
    expected_selection_status = {
        1000: "SEALED_AFTER_R1_BEFORE_R2",
        2000: "SEALED_AFTER_R2_BEFORE_R3",
        3000: "SEALED_AFTER_R3_BEFORE_R4",
    }[next_endpoint]
    if (
        not selected
        or re.fullmatch(r"[0-9a-f]{64}", str(prior_seal_identity)) is None
        or selection_seal.get("status") != expected_selection_status
        or selection_seal.get("identity") != canonical_json_hash(selection_basis)
        or selection_basis.get("dataset") != DATASET
        or set(map(str, selection_basis.get("selected_candidate_hashes", []))) != selected
        or int(selection_basis.get("selected_candidate_set_count", -1)) != len(selected)
    ):
        raise ValueError("H2 continuation lacks a sealed candidate selection")
    if {str(row.get("plan_seal_identity")) for row in normalized} != {prior_seal_identity}:
        raise ValueError("H2 continuation source plan seal differs")
    available = {str(row["candidate_set_hash"]) for row in normalized}
    if not selected.issubset(available):
        raise ValueError("H2 continuation selection references an unknown candidate")
    for digest in selected:
        cohort = [row for row in normalized if row["candidate_set_hash"] == digest]
        if len(cohort) != 3 or {int(row["model_seed"]) for row in cohort} != set(SEARCH_MODEL_SEEDS):
            raise ValueError("H2 continuation candidate lacks three frozen search seeds")
    result = Path(result_root).resolve()
    rows: list[dict[str, Any]] = []
    for row in normalized:
        if str(row["candidate_set_hash"]) not in selected:
            continue
        if int(row.get("endpoint_update", -1)) != prior_endpoint:
            raise ValueError("H2 continuation source endpoint differs")
        marker_path, attempt_index = _completion_marker(result, row, prior_endpoint)
        marker = read_json(marker_path)
        basis = validate_run_marker(marker, row, stage="SEARCH", endpoint=prior_endpoint)
        run_root = marker_path.parent
        config_path = run_root / "RESOLVED_CONFIG.json"
        if config_path.is_symlink() or not config_path.is_file():
            raise RuntimeError("H2 continuation source lacks a governed config")
        config = read_json(config_path)
        config_identity = canonical_json_hash(config)
        config_seal_identity = str(config.get("plan_seal_hash", ""))
        expected_config_seal_identity = str(
            row.get("run_config_plan_seal_identity", prior_seal_identity)
        )
        checkpoint = result / Path(*str(basis["checkpoint_relative_path"]).split("/"))
        expected = run_root / "checkpoints" / f"update_{prior_endpoint:06d}.pt"
        if (
            basis.get("run_identity") != config_identity
            or re.fullmatch(r"[0-9a-f]{64}", config_seal_identity) is None
            or config_seal_identity != expected_config_seal_identity
            or (
                "run_config_identity" in row
                and str(row["run_config_identity"]) != config_identity
            )
            or checkpoint.is_symlink()
            or not checkpoint.is_file()
            or checkpoint.resolve(strict=True) != expected.resolve(strict=True)
            or sha256_file(checkpoint) != basis.get("checkpoint_sha256")
        ):
            raise ValueError("H2 strict continuation source identity is invalid")
        rows.append(
            {
                **{key: value for key, value in row.items() if key != "plan_seal_identity"},
                "stage": "SEARCH",
                "logical_stage": logical_stage,
                "attempt_index": attempt_index,
                "continuation_from_update": prior_endpoint,
                "endpoint_update": next_endpoint,
                "execution_action": f"STRICT_CONTINUATION_{prior_endpoint:04d}_TO_{next_endpoint:04d}",
                "resume_checkpoint_path": str(checkpoint.resolve()),
                "resume_checkpoint_sha256": sha256_file(checkpoint),
                "run_config_plan_seal_identity": config_seal_identity,
                "run_config_identity": config_identity,
                "queue_ordinal": len(rows),
                "status": "AUTHORIZED",
            }
        )
    return _seal_plan(
        logical_stage,
        rows,
        endpoint_update=next_endpoint,
        continuation_from_update=prior_endpoint,
        extra_basis={
            "prior_plan_seal_identity": prior_seal_identity,
            "candidate_selection_identity": selection_seal["identity"],
            "selected_candidate_set_count": len(selected),
        },
    )


def build_preliminary_plan(
    cache: Mapping[str, Any], selected_set: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if selected_set.get("candidate_set_hash") == MANUAL_SET_HASH:
        return _seal_plan(
            "PRELIMINARY",
            [],
            endpoint_update=PRELIM_ENDPOINT,
            continuation_from_update=0,
            extra_basis={"manual_physical_run_reuse": True, "selected_set_hash": MANUAL_SET_HASH},
        )
    candidate = _candidate(
        cache,
        set_id=str(selected_set["candidate_set_id"]),
        set_hash=str(selected_set["candidate_set_hash"]),
    )
    manual = _candidate(cache, set_id=MANUAL_SET_ID, set_hash=MANUAL_SET_HASH)
    rows = []
    for seed_ordinal, seed in enumerate(EVALUATION_MODEL_SEEDS):
        for role_ordinal, (logical_role, arm) in enumerate(
            (("FINAL_FROZEN_METHOD", candidate), ("MANUAL", manual))
        ):
            rows.append({
                "stage": "PRELIMINARY",
                "logical_stage": "PRELIMINARY",
                "logical_role": logical_role,
                "physical_run_id": physical_run_id(DATASET, int(arm["candidate_set_ordinal"]), seed_ordinal, prefix="M7F"),
                "dataset": DATASET,
                "cache_identity": cache["cache_identity"],
                "path_library_identity": cache["path_library_identity"],
                "all_sets_identity": cache["all_sets_identity"],
                "candidate_set_id": arm["candidate_set_id"],
                "candidate_set_hash": arm["candidate_set_hash"],
                "candidate_set_ordinal": int(arm["candidate_set_ordinal"]),
                "path_ids": list(arm["path_ids"]),
                "model_seed": int(seed),
                "model_seed_ordinal": seed_ordinal,
                "continuation_from_update": 0,
                "endpoint_update": PRELIM_ENDPOINT,
                "checkpoint_schedule": [1000, 2000, 3000],
                "execution_action": "FRESH_PRELIMINARY_TRAIN_FROM_ZERO_TO_3000",
                "training_from_zero": True,
                "search_checkpoint_reused": False,
                "search_optimizer_reused": False,
                "search_rng_state_reused": False,
                "labels_mounted": False,
                "downstream_evaluations_during_training": 0,
                "eligible_for_selector_development": False,
                "eligible_for_preliminary_scientific_result": True,
                "attempt_index": 0,
                "preferred_gpu_slot": seed_ordinal,
                "queue_ordinal": seed_ordinal * 2 + role_ordinal,
                "status": "PENDING_GPU",
            })
    return _seal_plan(
        "PRELIMINARY",
        rows,
        endpoint_update=PRELIM_ENDPOINT,
        continuation_from_update=0,
        extra_basis={
            "manual_physical_run_reuse": False,
            "manual_rerun_required_for_same_gpu_pairing": True,
            "same_gpu_uuid_within_evaluation_seed_required": True,
            "selected_set_hash": candidate["candidate_set_hash"],
            "manual_set_hash": manual["candidate_set_hash"],
        },
    )


__all__ = [
    "PLAN_SCHEMA",
    "STAGE_STATUS",
    "canonical_plan_row",
    "validate_plan",
    "load_aminer_cache",
    "resolve_all_sets",
    "build_canary_plan",
    "build_r1_plan",
    "build_continuation_plan",
    "build_preliminary_plan",
]
