"""Frozen DBLP budgeted planning for the M7F-H3 campaign."""

from __future__ import annotations

import json
import re
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

from metachorus_v53.m6f_m7f_fast.planning import canary_run_id, load_cache_resolution, physical_run_id

from .common import canonical_json_hash, envelope, read_json, sha256_file
from .constants import (
    ALL_SETS_IDENTITY, CACHE_IDENTITY, CANARY_ENDPOINT, DATASET, DATASET_IDENTITY,
    EVALUATION_MODEL_SEEDS, EXPECTED_GPU_COUNT, LEGAL_PATH_COUNT, LEGAL_SET_COUNT,
    LONGEST_CANARY_SET_HASH, LONGEST_CANARY_SET_ID, MANUAL_SET_HASH, MANUAL_SET_ID,
    PATH_LIBRARY_IDENTITY, PRELIM_ENDPOINT, R1_MODEL_SEEDS, R1_POOL_SIZE,
    R3_SHORTLIST_SIZE, SEARCH_ENDPOINTS, SEARCH_MODEL_SEEDS, SET_CARDINALITY,
    SHORTEST_CANARY_SET_HASH, SHORTEST_CANARY_SET_ID,
)


PLAN_SCHEMA = "metachorus.v53.m7f-h3.dblp-plan-seal.v1"
STAGE_STATUS = {
    "CANARY": "SEALED_BEFORE_ENGINEERING_CANARIES",
    "R1": "SEALED_BEFORE_R1_GPU_TRAINING",
    "R2": "SEALED_AFTER_R1_BEFORE_R2",
    "R3": "SEALED_AFTER_R2_BEFORE_R3",
    "R4": "SEALED_AFTER_R3_BEFORE_R4",
    "PRELIMINARY": "SEALED_BEFORE_PRELIMINARY_GPU_TRAINING",
}


def canonical_plan_row(row: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(row)
    for field in (
        "candidate_set_ordinal", "physical_run_ordinal", "model_seed", "model_seed_ordinal",
        "continuation_from_update", "endpoint_update", "attempt_index",
        "downstream_evaluations_during_training", "queue_ordinal", "preferred_gpu_slot",
    ):
        if field in value and value[field] not in (None, ""):
            value[field] = int(value[field])
    for field in ("path_ids", "checkpoint_schedule", "logical_roles"):
        if field in value and isinstance(value[field], str):
            value[field] = json.loads(value[field])
        if field in value:
            value[field] = list(value[field])
    for field in (
        "labels_mounted", "eligible_for_selector_development",
        "eligible_for_preliminary_scientific_result", "training_from_zero",
        "search_checkpoint_reused", "search_optimizer_reused", "search_rng_state_reused",
    ):
        if field in value and not isinstance(value[field], bool):
            token = str(value[field]).strip().casefold()
            if token not in {"true", "false"}:
                raise ValueError(f"invalid boolean plan field {field}")
            value[field] = token == "true"
    return value


def _unsealed(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [canonical_plan_row({key: value for key, value in row.items() if key != "plan_seal_identity"}) for row in rows]


def _seal_plan(
    logical_stage: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    endpoint_update: int,
    extra_basis: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if logical_stage not in STAGE_STATUS:
        raise ValueError(f"unsupported H3 logical stage: {logical_stage}")
    normalized = _unsealed(rows)
    basis = {
        "logical_stage": logical_stage,
        "dataset": DATASET,
        "row_count": len(normalized),
        "rows_identity": canonical_json_hash(normalized),
        "continuation_from_updates": sorted({int(row.get("continuation_from_update", 0)) for row in normalized}),
        "endpoint_update": int(endpoint_update),
        "search_model_seeds": sorted({int(row["model_seed"]) for row in normalized}) if logical_stage in {"R1", "R2", "R3", "R4"} else [],
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


def _expected_stage_boundary(logical_stage: str, row: Mapping[str, Any]) -> bool:
    seed = int(row.get("model_seed", -1))
    start = int(row.get("continuation_from_update", -1))
    endpoint = int(row.get("endpoint_update", -1))
    if logical_stage == "CANARY":
        return start == 0 and endpoint == CANARY_ENDPOINT
    if logical_stage == "R1":
        return seed == R1_MODEL_SEEDS[0] and start == 0 and endpoint == 300
    if logical_stage == "R2":
        expected = 300 if seed == R1_MODEL_SEEDS[0] else 0
        return start == expected and endpoint == 1000
    if logical_stage == "R3":
        return start == 1000 and endpoint == 2000
    if logical_stage == "R4":
        return start == 2000 and endpoint == 3000
    return start == 0 and endpoint == PRELIM_ENDPOINT


def validate_plan(rows: Sequence[Mapping[str, Any]], seal: Mapping[str, Any], *, logical_stage: str) -> list[dict[str, Any]]:
    if logical_stage not in STAGE_STATUS:
        raise ValueError(f"unsupported H3 logical stage: {logical_stage}")
    normalized = [canonical_plan_row(row) for row in rows]
    basis = seal.get("basis", {})
    unsealed = _unsealed(normalized)
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
        or int(basis.get("endpoint_update", -1)) != ({"CANARY": 20, "R1": 300, "R2": 1000, "R3": 2000, "R4": 3000, "PRELIMINARY": 3000}[logical_stage])
        or basis.get("continuation_from_updates") != sorted({int(row.get("continuation_from_update", 0)) for row in normalized})
        or basis.get("run_level_parallel_only") is not True
        or int(basis.get("maximum_gpu_count", -1)) != EXPECTED_GPU_COUNT
        or basis.get("one_process_per_gpu") is not True
        or basis.get("ddp_used") is not False
        or basis.get("data_parallel_used") is not False
        or basis.get("torchrun_used") is not False
        or any(row.get("plan_seal_identity") != seal.get("identity") for row in normalized)
        or any(row.get("dataset") != DATASET or row.get("logical_stage") != logical_stage for row in normalized)
        or any(row.get("labels_mounted") is not False for row in normalized)
        or any(int(row.get("downstream_evaluations_during_training", -1)) != 0 for row in normalized)
        or any(int(row.get("attempt_index", -1)) not in {0, 1} for row in normalized)
        or any(not _expected_stage_boundary(logical_stage, row) for row in normalized)
    ):
        raise ValueError("M7F-H3 run plan/seal identity is invalid")
    seeds = {int(row["model_seed"]) for row in normalized}
    candidate_hashes = {str(row["candidate_set_hash"]) for row in normalized}
    expected_search_seeds = (
        set(R1_MODEL_SEEDS)
        if logical_stage == "R1"
        else set(SEARCH_MODEL_SEEDS) if logical_stage in {"R2", "R3", "R4"} else set()
    )
    if (
        set(map(int, basis.get("search_model_seeds", []))) != expected_search_seeds
        or set(map(int, basis.get("evaluation_model_seeds", [])))
        != (set(EVALUATION_MODEL_SEEDS) if logical_stage == "PRELIMINARY" else set())
        or len({str(row["physical_run_id"]) for row in normalized}) != len(normalized)
    ):
        raise ValueError("M7F-H3 plan seed or physical-run identity is invalid")
    if logical_stage == "CANARY":
        if len(normalized) != 4 or seeds != {R1_MODEL_SEEDS[0]} or len(candidate_hashes) != 4:
            raise ValueError("H3 canary plan is not the exact four-run matrix")
    elif logical_stage == "R1":
        if len(normalized) != LEGAL_SET_COUNT or seeds != set(R1_MODEL_SEEDS) or len(candidate_hashes) != LEGAL_SET_COUNT:
            raise ValueError("H3 R1 is not exactly 220 sets by one frozen seed")
    elif logical_stage in {"R2", "R3"}:
        if len(normalized) != R1_POOL_SIZE * 3 or seeds != set(SEARCH_MODEL_SEEDS) or len(candidate_hashes) != R1_POOL_SIZE:
            raise ValueError("H3 R2/R3 is not exactly 24 sets by three seeds")
    elif logical_stage == "R4":
        if len(normalized) != R3_SHORTLIST_SIZE * 3 or seeds != set(SEARCH_MODEL_SEEDS) or len(candidate_hashes) != R3_SHORTLIST_SIZE:
            raise ValueError("H3 R4 is not exactly eight sets by three seeds")
    elif normalized:
        if seeds != set(EVALUATION_MODEL_SEEDS) or len(candidate_hashes) not in {1, 2}:
            raise ValueError("H3 preliminary plan is not the exact three-seed paired/deduplicated matrix")
        roles = {str(row.get("logical_role")) for row in normalized}
        expected_roles = {"MANUAL"} if len(candidate_hashes) == 1 else {"BUDGETED_SELECTED_SET", "MANUAL"}
        if roles != expected_roles:
            raise ValueError("H3 preliminary roles do not match the selected/Manual deduplication policy")
        for seed in EVALUATION_MODEL_SEEDS:
            cohort = [row for row in normalized if int(row["model_seed"]) == seed]
            if len(cohort) != len(candidate_hashes) or len({int(row["preferred_gpu_slot"]) for row in cohort}) != 1:
                raise ValueError("H3 preliminary seed pair is incomplete or crosses GPU slots")
    for digest in candidate_hashes:
        cohort = [row for row in normalized if str(row["candidate_set_hash"]) == digest]
        if logical_stage in {"R2", "R3", "R4"} and {int(row["model_seed"]) for row in cohort} != set(SEARCH_MODEL_SEEDS):
            raise ValueError("H3 refinement candidate lacks three frozen search seeds")
    return normalized


def load_dblp_cache(cache_root: str | Path) -> dict[str, Any]:
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
        or any(
            len(row.get("path_ids", [])) != SET_CARDINALITY
            or len(set(map(str, row.get("path_ids", [])))) != SET_CARDINALITY
            for row in sets
        )
        or value["manual"].get("candidate_set_id") != MANUAL_SET_ID
        or value["manual"].get("candidate_set_hash") != MANUAL_SET_HASH
        or len(value["manual"].get("path_ids", [])) != SET_CARDINALITY
        or len(set(map(str, value["manual"].get("path_ids", [])))) != SET_CARDINALITY
        or value.get("current_automatic", {}).get("status") != "NOT_AVAILABLE_EXACT_IDENTITY"
    ):
        raise ValueError("DBLP H3 cache/candidate identity mismatch")
    return value


def resolve_all_sets(cache: Mapping[str, Any]) -> list[dict[str, Any]]:
    path_map = {str(row["path_id"]): row for row in cache["path_library"]["paths"]}
    rows = []
    for candidate in cache["all_sets"]["sets"]:
        path_ids = list(map(str, candidate["path_ids"]))
        if (
            len(path_ids) != SET_CARDINALITY
            or len(set(path_ids)) != SET_CARDINALITY
            or any(path_id not in path_map for path_id in path_ids)
        ):
            raise ValueError("DBLP candidate path resolution failed")
        hop_count = sum(len(path_map[path_id]["relation_sequence"]) for path_id in path_ids)
        rows.append({
            "dataset": DATASET, "dataset_identity": DATASET_IDENTITY,
            "cache_identity": cache["cache_identity"], "path_library_identity": cache["path_library_identity"],
            "all_sets_identity": cache["all_sets_identity"], "candidate_set_id": candidate["candidate_set_id"],
            "candidate_set_hash": candidate["candidate_set_hash"],
            "candidate_set_ordinal": int(candidate["candidate_set_ordinal"]), "path_ids": path_ids,
            "path_count": len(path_ids), "total_hop_count": hop_count,
            "projection_hashes": [str(path_map[path_id]["semantic_projection_hash"]) for path_id in path_ids],
            "projection_arrays": [str(path_map[path_id]["array"]) for path_id in path_ids], "status": "PASS",
        })
    rows.sort(key=lambda row: int(row["candidate_set_ordinal"]))
    if len(rows) != LEGAL_SET_COUNT or len({row["candidate_set_hash"] for row in rows}) != LEGAL_SET_COUNT:
        raise RuntimeError("DBLP all-set resolution is not exactly 220 unique sets")
    return rows


def _candidate(cache: Mapping[str, Any], *, set_id: str | None = None, set_hash: str) -> dict[str, Any]:
    matches = [dict(row) for row in cache["all_sets"]["sets"] if row.get("candidate_set_hash") == set_hash and (set_id is None or row.get("candidate_set_id") == set_id)]
    if len(matches) != 1:
        raise ValueError(f"frozen candidate identity not found: {set_id or set_hash}")
    return matches[0]


def _base_search_row(cache: Mapping[str, Any], candidate: Mapping[str, Any], *, seed: int, seed_ordinal: int, logical_stage: str, start: int, endpoint: int) -> dict[str, Any]:
    fresh = start == 0
    return {
        "stage": "R1_ALL_SETS_0000_TO_0300" if logical_stage == "R1" else "SEARCH",
        "logical_stage": logical_stage,
        "physical_run_id": physical_run_id(DATASET, int(candidate["candidate_set_ordinal"]), seed_ordinal),
        "dataset": DATASET, "cache_identity": cache["cache_identity"],
        "path_library_identity": cache["path_library_identity"], "all_sets_identity": cache["all_sets_identity"],
        "candidate_set_id": candidate["candidate_set_id"], "candidate_set_hash": candidate["candidate_set_hash"],
        "candidate_set_ordinal": int(candidate["candidate_set_ordinal"]), "path_ids": list(candidate["path_ids"]),
        "model_seed": int(seed), "model_seed_ordinal": int(seed_ordinal),
        "continuation_from_update": int(start), "endpoint_update": int(endpoint),
        "checkpoint_schedule": list(SEARCH_ENDPOINTS),
        "execution_action": f"TRAIN_FROM_ZERO_TO_{endpoint:04d}" if fresh else f"STRICT_CONTINUATION_{start:04d}_TO_{endpoint:04d}",
        "training_from_zero": fresh, "search_checkpoint_reused": not fresh,
        "labels_mounted": False, "downstream_evaluations_during_training": 0,
        "eligible_for_selector_development": True, "eligible_for_preliminary_scientific_result": False,
        "attempt_index": 0, "status": "PENDING_GPU" if fresh else "AUTHORIZED",
    }


def build_canary_plan(cache: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    resolved = resolve_all_sets(cache)
    median_hops = statistics.median(int(row["total_hop_count"]) for row in resolved)
    median_row = min((row for row in resolved if int(row["total_hop_count"]) == median_hops), key=lambda row: str(row["candidate_set_hash"]))
    specs = (
        (MANUAL_SET_ID, MANUAL_SET_HASH, 0),
        (SHORTEST_CANARY_SET_ID, SHORTEST_CANARY_SET_HASH, 1),
        (LONGEST_CANARY_SET_ID, LONGEST_CANARY_SET_HASH, 2),
        (str(median_row["candidate_set_id"]), str(median_row["candidate_set_hash"]), 3),
    )
    rows = []
    for set_id, set_hash, gpu_slot in specs:
        candidate = _candidate(cache, set_id=set_id, set_hash=set_hash)
        rows.append({
            "stage": "CANARY", "logical_stage": "CANARY",
            "physical_run_id": canary_run_id(DATASET, int(candidate["candidate_set_ordinal"]), 0),
            "dataset": DATASET, "cache_identity": cache["cache_identity"],
            "path_library_identity": cache["path_library_identity"], "all_sets_identity": cache["all_sets_identity"],
            "candidate_set_id": candidate["candidate_set_id"], "candidate_set_hash": candidate["candidate_set_hash"],
            "candidate_set_ordinal": int(candidate["candidate_set_ordinal"]), "path_ids": list(candidate["path_ids"]),
            "model_seed": R1_MODEL_SEEDS[0], "model_seed_ordinal": 0,
            "continuation_from_update": 0, "endpoint_update": CANARY_ENDPOINT,
            "checkpoint_schedule": [CANARY_ENDPOINT], "execution_action": "ENGINEERING_CANARY_FROM_ZERO_TO_0020",
            "preferred_gpu_slot": gpu_slot, "queue_ordinal": gpu_slot,
            "labels_mounted": False, "downstream_evaluations_during_training": 0,
            "eligible_for_selector_development": False, "eligible_for_preliminary_scientific_result": False,
            "attempt_index": 0, "status": "PENDING_GPU",
        })
    return _seal_plan("CANARY", rows, endpoint_update=CANARY_ENDPOINT, extra_basis={"median_total_hop_count": median_hops, "median_canary_set_hash": median_row["candidate_set_hash"]})


def build_r1_plan(cache: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = [
        {**_base_search_row(cache, candidate, seed=R1_MODEL_SEEDS[0], seed_ordinal=0, logical_stage="R1", start=0, endpoint=300), "queue_ordinal": ordinal}
        for ordinal, candidate in enumerate(cache["all_sets"]["sets"])
    ]
    if len(rows) != LEGAL_SET_COUNT or len({row["physical_run_id"] for row in rows}) != LEGAL_SET_COUNT:
        raise RuntimeError("DBLP H3 R1 must contain exactly 220 unique one-seed runs")
    return _seal_plan("R1", rows, endpoint_update=300, extra_basis={
        "legal_set_count": LEGAL_SET_COUNT, "r1_model_seed_count": 1,
        "all_sets_identity": cache["all_sets_identity"],
        "evidence_level": "BUDGET_LIMITED_ONE_SEED_FULL_SPACE_SCREEN",
    })


def _outer_marker(result: Path, row: Mapping[str, Any], logical_stage: str) -> tuple[Path, dict[str, Any]]:
    run_parent = result / "search_runs" / str(row["physical_run_id"])
    matches = [attempt / f"DBLP_H3_{logical_stage}_COMPLETE.json" for attempt in sorted(run_parent.glob("attempt_*")) if (attempt / f"DBLP_H3_{logical_stage}_COMPLETE.json").is_file() and not (attempt / f"DBLP_H3_{logical_stage}_COMPLETE.json").is_symlink()]
    if len(matches) != 1:
        raise RuntimeError(f"H3 continuation source is not unique: {row['physical_run_id']} {logical_stage}")
    value = read_json(matches[0])
    basis = value.get("basis", {})
    if (
        value.get("schema_version") != "metachorus.v53.m7f-h3.run-complete.v1"
        or value.get("status") != "PASS" or value.get("identity") != canonical_json_hash(basis)
        or basis.get("logical_stage") != logical_stage or basis.get("physical_run_id") != row.get("physical_run_id")
        or basis.get("plan_seal_identity") != row.get("plan_seal_identity")
        or int(basis.get("endpoint_update", -1)) != int(row.get("endpoint_update", -2))
    ):
        raise ValueError("H3 continuation outer marker identity is invalid")
    return matches[0], value


def build_continuation_plan(
    *, prior_rows: Sequence[Mapping[str, Any]], selected_candidate_hashes: set[str],
    result_root: str | Path, next_endpoint: int, prior_seal_identity: str,
    selection_seal: Mapping[str, Any], cache: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if next_endpoint not in SEARCH_ENDPOINTS[1:]:
        raise ValueError("invalid H3 continuation endpoint")
    prior_endpoint = SEARCH_ENDPOINTS[SEARCH_ENDPOINTS.index(next_endpoint) - 1]
    prior_stage = f"R{SEARCH_ENDPOINTS.index(prior_endpoint) + 1}"
    logical_stage = f"R{SEARCH_ENDPOINTS.index(next_endpoint) + 1}"
    normalized = [canonical_plan_row(row) for row in prior_rows]
    selected = set(map(str, selected_candidate_hashes))
    expected_count = R1_POOL_SIZE if next_endpoint in {1000, 2000} else R3_SHORTLIST_SIZE
    selection_basis = selection_seal.get("basis", {})
    expected_selection_status = {
        1000: "SEALED_AFTER_R1_BEFORE_R2",
        2000: "SEALED_AFTER_R2_BEFORE_R3",
        3000: "SEALED_AFTER_R3_BEFORE_R4",
    }[next_endpoint]
    if (
        len(selected) != expected_count or MANUAL_SET_HASH not in selected
        or re.fullmatch(r"[0-9a-f]{64}", str(prior_seal_identity)) is None
        or selection_seal.get("status") != expected_selection_status
        or selection_seal.get("identity") != canonical_json_hash(selection_basis)
        or set(map(str, selection_basis.get("selected_candidate_hashes", []))) != selected
        or int(selection_basis.get("selected_candidate_set_count", -1)) != len(selected)
        or {str(row.get("plan_seal_identity")) for row in normalized} != {prior_seal_identity}
    ):
        raise ValueError("H3 continuation lacks a sealed exact candidate selection")
    by_hash = {str(row["candidate_set_hash"]): row for row in cache["all_sets"]["sets"]}
    if not selected.issubset(by_hash):
        raise ValueError("H3 continuation selection references an unknown candidate")
    result = Path(result_root).resolve()
    prior_by_key = {(str(row["candidate_set_hash"]), int(row["model_seed"])): row for row in normalized}
    rows = []
    for digest in sorted(selected):
        candidate = by_hash[digest]
        for seed_ordinal, seed in enumerate(SEARCH_MODEL_SEEDS):
            if next_endpoint == 1000 and seed != R1_MODEL_SEEDS[0]:
                row = _base_search_row(cache, candidate, seed=seed, seed_ordinal=seed_ordinal, logical_stage=logical_stage, start=0, endpoint=next_endpoint)
            else:
                prior = prior_by_key.get((digest, int(seed)))
                if prior is None or int(prior["endpoint_update"]) != prior_endpoint:
                    raise ValueError("H3 continuation source matrix is incomplete")
                marker_path, outer = _outer_marker(result, prior, prior_stage)
                run_root = marker_path.parent
                engine_path = result / Path(*str(outer["basis"]["engine_marker_relative_path"]).split("/"))
                engine = read_json(engine_path)
                engine_basis = engine.get("basis", {})
                config_path = run_root / "RESOLVED_CONFIG.json"
                checkpoint = run_root / "checkpoints" / f"update_{prior_endpoint:06d}.pt"
                if config_path.is_symlink() or not config_path.is_file() or checkpoint.is_symlink() or not checkpoint.is_file():
                    raise RuntimeError("H3 continuation source lacks config/checkpoint")
                config = read_json(config_path)
                config_identity = canonical_json_hash(config)
                config_seal = str(config.get("plan_seal_hash", ""))
                expected_config_seal = str(prior.get("run_config_plan_seal_identity", prior_seal_identity))
                if (
                    engine.get("status") != "PASS" or engine.get("identity") != canonical_json_hash(engine_basis)
                    or engine_basis.get("run_identity") != config_identity
                    or engine_basis.get("checkpoint_sha256") != sha256_file(checkpoint)
                    or config_seal != expected_config_seal
                ):
                    raise ValueError("H3 strict continuation source identity is invalid")
                row = {
                    **{key: value for key, value in prior.items() if key != "plan_seal_identity"},
                    "logical_stage": logical_stage, "stage": "SEARCH", "continuation_from_update": prior_endpoint,
                    "endpoint_update": next_endpoint, "execution_action": f"STRICT_CONTINUATION_{prior_endpoint:04d}_TO_{next_endpoint:04d}",
                    "training_from_zero": False, "search_checkpoint_reused": True,
                    "resume_checkpoint_path": str(checkpoint.resolve()), "resume_checkpoint_sha256": sha256_file(checkpoint),
                    "run_config_plan_seal_identity": config_seal, "run_config_identity": config_identity,
                    "status": "AUTHORIZED",
                }
            row["queue_ordinal"] = len(rows)
            rows.append(row)
    return _seal_plan(logical_stage, rows, endpoint_update=next_endpoint, extra_basis={
        "prior_plan_seal_identity": prior_seal_identity,
        "candidate_selection_identity": selection_seal["identity"],
        "selected_candidate_set_count": len(selected),
        "mixed_r2_training_policy": "R1_SEED_CONTINUES_0300_TO_1000;OTHER_TWO_SEEDS_FRESH_0000_TO_1000" if next_endpoint == 1000 else "ALL_THREE_SEEDS_STRICT_CONTINUATION",
    })


def build_preliminary_plan(cache: Mapping[str, Any], selected_set: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected = _candidate(cache, set_id=str(selected_set["candidate_set_id"]), set_hash=str(selected_set["candidate_set_hash"]))
    manual = _candidate(cache, set_id=MANUAL_SET_ID, set_hash=MANUAL_SET_HASH)
    arms = [("BUDGETED_SELECTED_SET", selected)]
    if selected["candidate_set_hash"] != MANUAL_SET_HASH:
        arms.append(("MANUAL", manual))
    else:
        arms = [("MANUAL", manual)]
    rows = []
    for seed_ordinal, seed in enumerate(EVALUATION_MODEL_SEEDS):
        for role_ordinal, (logical_role, arm) in enumerate(arms):
            rows.append({
                "stage": "PRELIMINARY", "logical_stage": "PRELIMINARY", "logical_role": logical_role,
                "physical_run_id": physical_run_id(DATASET, int(arm["candidate_set_ordinal"]), seed_ordinal, prefix="M7F"),
                "dataset": DATASET, "cache_identity": cache["cache_identity"],
                "path_library_identity": cache["path_library_identity"], "all_sets_identity": cache["all_sets_identity"],
                "candidate_set_id": arm["candidate_set_id"], "candidate_set_hash": arm["candidate_set_hash"],
                "candidate_set_ordinal": int(arm["candidate_set_ordinal"]), "path_ids": list(arm["path_ids"]),
                "model_seed": int(seed), "model_seed_ordinal": seed_ordinal,
                "continuation_from_update": 0, "endpoint_update": PRELIM_ENDPOINT,
                "checkpoint_schedule": [1000, 2000, 3000], "execution_action": "FRESH_PRELIMINARY_TRAIN_FROM_ZERO_TO_3000",
                "training_from_zero": True, "search_checkpoint_reused": False,
                "search_optimizer_reused": False, "search_rng_state_reused": False,
                "labels_mounted": False, "downstream_evaluations_during_training": 0,
                "eligible_for_selector_development": False, "eligible_for_preliminary_scientific_result": True,
                "attempt_index": 0, "preferred_gpu_slot": seed_ordinal,
                "queue_ordinal": seed_ordinal * len(arms) + role_ordinal, "status": "PENDING_GPU",
            })
    return _seal_plan("PRELIMINARY", rows, endpoint_update=PRELIM_ENDPOINT, extra_basis={
        "manual_physical_run_deduplicated": selected["candidate_set_hash"] == MANUAL_SET_HASH,
        "same_gpu_uuid_within_evaluation_seed_required": len(arms) == 2,
        "selected_set_hash": selected["candidate_set_hash"], "manual_set_hash": MANUAL_SET_HASH,
        "search_checkpoint_reuse": False,
    })


__all__ = [
    "PLAN_SCHEMA", "STAGE_STATUS", "canonical_plan_row", "validate_plan", "load_dblp_cache",
    "resolve_all_sets", "build_canary_plan", "build_r1_plan", "build_continuation_plan", "build_preliminary_plan",
]
