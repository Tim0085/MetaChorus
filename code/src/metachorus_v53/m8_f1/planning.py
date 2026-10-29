"""Frozen IMDb/AMiner ten-seed completion planning for M8-F1."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from metachorus_v53.m6f_m7f_fast.planning import load_cache_resolution as _load_cache_resolution

from .common import canonical_json_hash, envelope
from .constants import (
    CANARY_ENDPOINT,
    CONFIRMATION_MODEL_SEEDS,
    DATASETS,
    DATASET_SPECS,
    EXPECTED_GPU_COUNT,
    FINAL10_CHECKPOINT_SCHEDULE,
    FINAL10_ENDPOINT,
    LABEL_ROLE_MANIFEST_IDENTITY,
    METHOD_FREEZE_IDENTITY,
)


PLAN_SCHEMA = "metachorus.v53.m8-f1.final10-plan-seal.v1"
STAGE_STATUS = {
    "CANARY": "SEALED_BEFORE_ENGINEERING_CANARIES",
    "FINAL10": "SEALED_BEFORE_FINAL10_GPU_TRAINING",
}


def canonical_plan_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Recover the exact typed row after a CSV round trip."""

    value = dict(row)
    for field in (
        "candidate_set_ordinal",
        "model_seed",
        "model_seed_ordinal",
        "target_node_count",
        "class_count",
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
    for field in ("path_ids", "checkpoint_schedule", "representation_schedule"):
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
        "dynamic_queue_assignment",
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


def load_dataset_cache(dataset: str, cache_root: str | Path) -> dict[str, Any]:
    """Load and bind one frozen cache to the exact M8-F1 dataset identity."""

    if dataset not in DATASETS:
        raise ValueError(f"M8-F1 does not authorize dataset {dataset!r}")
    value = _load_cache_resolution(cache_root)
    spec = DATASET_SPECS[dataset]
    if (
        value.get("dataset") != dataset
        or value.get("cache_identity") != spec["cache_identity"]
        or value.get("path_library_identity") != spec["path_library_identity"]
        or value.get("all_sets_identity") != spec["all_sets_identity"]
    ):
        raise ValueError(f"M8-F1 {dataset} frozen cache identity changed")
    return value


def load_cache_map(cache_parent: str | Path) -> dict[str, dict[str, Any]]:
    root = Path(cache_parent).resolve(strict=True)
    return {dataset: load_dataset_cache(dataset, root / dataset) for dataset in DATASETS}


def resolve_manual_set(cache: Mapping[str, Any], dataset: str) -> dict[str, Any]:
    """Resolve the one authorized Manual set from the frozen all-set bank."""

    if dataset not in DATASETS or cache.get("dataset") != dataset:
        raise ValueError("M8-F1 cache/dataset mismatch")
    spec = DATASET_SPECS[dataset]
    matches = [
        dict(row)
        for row in cache.get("all_sets", {}).get("sets", [])
        if row.get("candidate_set_id") == spec["manual_set_id"]
        and row.get("candidate_set_hash") == spec["manual_set_hash"]
    ]
    if (
        len(matches) != 1
        or int(matches[0].get("candidate_set_ordinal", -1)) != spec["manual_set_ordinal"]
        or tuple(matches[0].get("path_ids", [])) != tuple(spec["manual_path_ids"])
    ):
        raise ValueError(f"M8-F1 {dataset} Manual set identity is not exact")
    return matches[0]


def resolve_all_sets(cache: Mapping[str, Any], dataset: str) -> list[dict[str, Any]]:
    """Validate the bank and return its rows for the public resolution artifact."""

    manual = resolve_manual_set(cache, dataset)
    rows = [dict(row) for row in cache.get("all_sets", {}).get("sets", [])]
    spec = DATASET_SPECS[dataset]
    if (
        len(rows) != int(spec["legal_set_count"])
        or len({row.get("candidate_set_hash") for row in rows}) != len(rows)
        or manual not in rows
    ):
        raise ValueError(f"M8-F1 {dataset} candidate bank is not exact")
    return rows


def _physical_run_id(dataset: str, seed_ordinal: int, *, canary: bool) -> str:
    token = dataset.upper()
    phase = "CANARY" if canary else "FINAL10"
    return f"M8-F1-{phase}-{token}-MANUAL-C000-F{seed_ordinal:02d}"


def _training_config_identity(
    *, dataset: str, candidate: Mapping[str, Any], model_seed: int, endpoint: int
) -> str:
    spec = DATASET_SPECS[dataset]
    return canonical_json_hash(
        {
            "dataset": dataset,
            "dataset_identity": spec["dataset_identity"],
            "cache_identity": spec["cache_identity"],
            "path_library_identity": spec["path_library_identity"],
            "all_sets_identity": spec["all_sets_identity"],
            "method_freeze_identity": METHOD_FREEZE_IDENTITY,
            "logical_role": "MANUAL",
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
    caches: Mapping[str, Mapping[str, Any]],
    *,
    dataset: str,
    seed_ordinal: int,
    canary: bool,
    queue_ordinal: int,
    preferred_gpu_slot: int,
) -> dict[str, Any]:
    cache = caches[dataset]
    candidate = resolve_manual_set(cache, dataset)
    spec = DATASET_SPECS[dataset]
    seed = int(CONFIRMATION_MODEL_SEEDS[seed_ordinal])
    endpoint = CANARY_ENDPOINT if canary else FINAL10_ENDPOINT
    logical_stage = "CANARY" if canary else "FINAL10"
    return {
        "stage": logical_stage,
        "logical_stage": logical_stage,
        "logical_role": "MANUAL",
        "final_method_role": "MANUAL_FALLBACK",
        "physical_run_id": _physical_run_id(dataset, seed_ordinal, canary=canary),
        "dataset": dataset,
        "dataset_identity": spec["dataset_identity"],
        "cache_identity": cache["cache_identity"],
        "path_library_identity": cache["path_library_identity"],
        "all_sets_identity": cache["all_sets_identity"],
        "target_node_type": spec["target_node_type"],
        "target_node_count": spec["target_node_count"],
        "class_count": spec["class_count"],
        "candidate_set_id": candidate["candidate_set_id"],
        "candidate_set_hash": candidate["candidate_set_hash"],
        "candidate_set_ordinal": int(candidate["candidate_set_ordinal"]),
        "path_ids": list(candidate["path_ids"]),
        "model_seed": seed,
        "model_seed_ordinal": int(seed_ordinal),
        "start_update": 0,
        "continuation_from_update": 0,
        "endpoint_update": endpoint,
        "checkpoint_schedule": [endpoint] if canary else list(FINAL10_CHECKPOINT_SCHEDULE),
        "representation_schedule": [endpoint],
        "execution_action": (
            "ENGINEERING_CANARY_FROM_ZERO_TO_0020"
            if canary
            else "FRESH_FINAL10_FROM_ZERO_TO_3000"
        ),
        "training_config_identity": _training_config_identity(
            dataset=dataset, candidate=candidate, model_seed=seed, endpoint=endpoint
        ),
        "label_role_identity": LABEL_ROLE_MANIFEST_IDENTITY,
        "planned_runtime_domain": (
            f"M8-F1-CANARY-GPU{preferred_gpu_slot}" if canary else "M8-F1-DYNAMIC-4GPU-QUEUE"
        ),
        "preferred_gpu_slot": int(preferred_gpu_slot),
        "dynamic_queue_assignment": not canary,
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


def _seal_plan(
    logical_stage: str, rows: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    normalized = _unsealed(rows)
    basis = {
        "logical_stage": logical_stage,
        "datasets": list(DATASETS),
        "row_count": len(normalized),
        "rows_identity": canonical_json_hash(normalized),
        "model_seeds": sorted({int(row["model_seed"]) for row in normalized}),
        "logical_roles": ["MANUAL"],
        "manual_set_hashes": {
            dataset: DATASET_SPECS[dataset]["manual_set_hash"] for dataset in DATASETS
        },
        "start_update": 0,
        "endpoint_update": CANARY_ENDPOINT if logical_stage == "CANARY" else FINAL10_ENDPOINT,
        "training_from_zero": True,
        "labels_mounted": False,
        "downstream_evaluations_during_training": 0,
        "dynamic_four_worker_queue": logical_stage == "FINAL10",
        "canary_fixed_gpu_mapping": logical_stage == "CANARY",
        "run_level_parallel_only": True,
        "maximum_gpu_count": EXPECTED_GPU_COUNT,
        "one_process_per_gpu": True,
        "ddp_used": False,
        "data_parallel_used": False,
        "torchrun_used": False,
        "further_search": False,
        "algorithm_retuning": False,
        "additional_model_seeds": False,
    }
    seal = envelope(PLAN_SCHEMA, STAGE_STATUS[logical_stage], basis)
    sealed_rows = [{**row, "plan_seal_identity": seal["identity"]} for row in normalized]
    validate_plan(sealed_rows, seal, logical_stage=logical_stage)
    return sealed_rows, seal


def build_canary_plan(
    caches: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if set(caches) != set(DATASETS):
        raise ValueError("M8-F1 canary plan requires IMDb and AMiner caches")
    specs = (
        ("imdb", 0, 0),
        ("aminer", 0, 1),
        ("imdb", 1, 2),
        ("aminer", 1, 3),
    )
    rows = [
        _row(
            caches,
            dataset=dataset,
            seed_ordinal=seed_ordinal,
            canary=True,
            queue_ordinal=ordinal,
            preferred_gpu_slot=gpu_slot,
        )
        for ordinal, (dataset, seed_ordinal, gpu_slot) in enumerate(specs)
    ]
    return _seal_plan("CANARY", rows)


def build_final10_plan(
    caches: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if set(caches) != set(DATASETS):
        raise ValueError("M8-F1 final10 plan requires IMDb and AMiner caches")
    rows: list[dict[str, Any]] = []
    for seed_ordinal, _seed in enumerate(CONFIRMATION_MODEL_SEEDS):
        for dataset in DATASETS:
            rows.append(
                _row(
                    caches,
                    dataset=dataset,
                    seed_ordinal=seed_ordinal,
                    canary=False,
                    queue_ordinal=len(rows),
                    preferred_gpu_slot=-1,
                )
            )
    return _seal_plan("FINAL10", rows)


def build_confirmation_plan(
    caches: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Backward-compatible alias for callers migrating from H4."""

    return build_final10_plan(caches)


def validate_plan(
    rows: Sequence[Mapping[str, Any]], seal: Mapping[str, Any], *, logical_stage: str
) -> list[dict[str, Any]]:
    if logical_stage not in STAGE_STATUS:
        raise ValueError("unsupported M8-F1 logical stage")
    normalized = [canonical_plan_row(row) for row in rows]
    basis = seal.get("basis", {})
    expected_count = 4 if logical_stage == "CANARY" else 20
    expected_endpoint = CANARY_ENDPOINT if logical_stage == "CANARY" else FINAL10_ENDPOINT
    expected_seeds = set(
        CONFIRMATION_MODEL_SEEDS[:2] if logical_stage == "CANARY" else CONFIRMATION_MODEL_SEEDS
    )
    expected_checkpoint_schedule = (
        [CANARY_ENDPOINT] if logical_stage == "CANARY" else list(FINAL10_CHECKPOINT_SCHEDULE)
    )
    expected_action = (
        "ENGINEERING_CANARY_FROM_ZERO_TO_0020"
        if logical_stage == "CANARY"
        else "FRESH_FINAL10_FROM_ZERO_TO_3000"
    )
    if (
        seal.get("schema_version") != PLAN_SCHEMA
        or seal.get("status") != STAGE_STATUS[logical_stage]
        or seal.get("identity") != canonical_json_hash(basis)
        or basis.get("logical_stage") != logical_stage
        or tuple(basis.get("datasets", [])) != DATASETS
        or int(basis.get("row_count", -1)) != expected_count
        or basis.get("rows_identity") != canonical_json_hash(_unsealed(normalized))
        or set(map(int, basis.get("model_seeds", []))) != expected_seeds
        or set(map(str, basis.get("logical_roles", []))) != {"MANUAL"}
        or basis.get("manual_set_hashes")
        != {dataset: DATASET_SPECS[dataset]["manual_set_hash"] for dataset in DATASETS}
        or int(basis.get("start_update", -1)) != 0
        or int(basis.get("endpoint_update", -1)) != expected_endpoint
        or basis.get("training_from_zero") is not True
        or basis.get("labels_mounted") is not False
        or int(basis.get("downstream_evaluations_during_training", -1)) != 0
        or basis.get("run_level_parallel_only") is not True
        or int(basis.get("maximum_gpu_count", -1)) != EXPECTED_GPU_COUNT
        or basis.get("one_process_per_gpu") is not True
        or basis.get("ddp_used") is not False
        or basis.get("data_parallel_used") is not False
        or basis.get("torchrun_used") is not False
        or basis.get("further_search") is not False
        or basis.get("algorithm_retuning") is not False
        or basis.get("additional_model_seeds") is not False
        or len(normalized) != expected_count
        or len({str(row.get("physical_run_id")) for row in normalized}) != expected_count
        or len({int(row.get("queue_ordinal", -1)) for row in normalized}) != expected_count
        or {int(row.get("queue_ordinal", -1)) for row in normalized} != set(range(expected_count))
        or any(row.get("plan_seal_identity") != seal.get("identity") for row in normalized)
        or any(row.get("logical_stage") != logical_stage for row in normalized)
        or any(row.get("dataset") not in DATASETS for row in normalized)
        or any(row.get("logical_role") != "MANUAL" for row in normalized)
        or any(row.get("final_method_role") != "MANUAL_FALLBACK" for row in normalized)
        or any(int(row.get("start_update", -1)) != 0 for row in normalized)
        or any(int(row.get("continuation_from_update", -1)) != 0 for row in normalized)
        or any(int(row.get("endpoint_update", -1)) != expected_endpoint for row in normalized)
        or any(row.get("training_from_zero") is not True for row in normalized)
        or any(row.get("search_checkpoint_reused") is not False for row in normalized)
        or any(row.get("labels_mounted") is not False for row in normalized)
        or any(int(row.get("downstream_evaluations_during_training", -1)) != 0 for row in normalized)
        or any(int(row.get("attempt_index", -1)) != 0 for row in normalized)
        or any(list(row.get("checkpoint_schedule", [])) != expected_checkpoint_schedule for row in normalized)
        or any(list(row.get("representation_schedule", [])) != [expected_endpoint] for row in normalized)
        or any(row.get("execution_action") != expected_action for row in normalized)
        or any(row.get("label_role_identity") != LABEL_ROLE_MANIFEST_IDENTITY for row in normalized)
        or any(row.get("eligible_for_scientific_result") is not (logical_stage == "FINAL10") for row in normalized)
        or any(row.get("eligible_for_selector_development") is not False for row in normalized)
        or any(row.get("status") != "PENDING_GPU" for row in normalized)
    ):
        raise ValueError("M8-F1 run plan/seal identity is invalid")

    for dataset in DATASETS:
        spec = DATASET_SPECS[dataset]
        cohort = [row for row in normalized if row["dataset"] == dataset]
        expected_dataset_count = 2 if logical_stage == "CANARY" else 10
        if (
            len(cohort) != expected_dataset_count
            or {int(row["model_seed"]) for row in cohort} != expected_seeds
            or {row["dataset_identity"] for row in cohort} != {spec["dataset_identity"]}
            or {row["cache_identity"] for row in cohort} != {spec["cache_identity"]}
            or {row["candidate_set_hash"] for row in cohort} != {spec["manual_set_hash"]}
            or {row["candidate_set_id"] for row in cohort} != {spec["manual_set_id"]}
            or {int(row["candidate_set_ordinal"]) for row in cohort}
            != {int(spec["manual_set_ordinal"])}
            or {row["path_library_identity"] for row in cohort}
            != {spec["path_library_identity"]}
            or {row["all_sets_identity"] for row in cohort}
            != {spec["all_sets_identity"]}
            or {row["target_node_type"] for row in cohort} != {spec["target_node_type"]}
            or {int(row["target_node_count"]) for row in cohort}
            != {int(spec["target_node_count"])}
            or {int(row["class_count"]) for row in cohort} != {int(spec["class_count"])}
            or any(tuple(row["path_ids"]) != tuple(spec["manual_path_ids"]) for row in cohort)
            or any(
                row.get("training_config_identity")
                != _training_config_identity(
                    dataset=dataset,
                    candidate=row,
                    model_seed=int(row["model_seed"]),
                    endpoint=expected_endpoint,
                )
                for row in cohort
            )
        ):
            raise ValueError(f"M8-F1 {dataset} Manual cohort identity is invalid")

    if logical_stage == "CANARY":
        expected = [
            ("imdb", int(CONFIRMATION_MODEL_SEEDS[0]), 0),
            ("aminer", int(CONFIRMATION_MODEL_SEEDS[0]), 1),
            ("imdb", int(CONFIRMATION_MODEL_SEEDS[1]), 2),
            ("aminer", int(CONFIRMATION_MODEL_SEEDS[1]), 3),
        ]
        observed = [
            (str(row["dataset"]), int(row["model_seed"]), int(row["preferred_gpu_slot"]))
            for row in sorted(normalized, key=lambda row: int(row["queue_ordinal"]))
        ]
        if observed != expected or basis.get("canary_fixed_gpu_mapping") is not True:
            raise ValueError("M8-F1 canary GPU/dataset/seed mapping is invalid")
    elif (
        basis.get("dynamic_four_worker_queue") is not True
        or any(row.get("dynamic_queue_assignment") is not True for row in normalized)
        or any(int(row.get("preferred_gpu_slot", 0)) != -1 for row in normalized)
    ):
        raise ValueError("M8-F1 FINAL10 must use the dynamic four-worker queue")
    for row in normalized:
        seed_ordinal = int(row.get("model_seed_ordinal", -1))
        if (
            seed_ordinal not in range(len(CONFIRMATION_MODEL_SEEDS))
            or int(row["model_seed"]) != int(CONFIRMATION_MODEL_SEEDS[seed_ordinal])
            or row["physical_run_id"]
            != _physical_run_id(str(row["dataset"]), seed_ordinal, canary=logical_stage == "CANARY")
            or row.get("planned_runtime_domain")
            != (
                f"M8-F1-CANARY-GPU{int(row['preferred_gpu_slot'])}"
                if logical_stage == "CANARY"
                else "M8-F1-DYNAMIC-4GPU-QUEUE"
            )
            or row.get("dynamic_queue_assignment") is not (logical_stage == "FINAL10")
        ):
            raise ValueError("M8-F1 row seed/run/runtime identity is invalid")
    return normalized


__all__ = [
    "PLAN_SCHEMA",
    "STAGE_STATUS",
    "canonical_plan_row",
    "load_dataset_cache",
    "load_cache_map",
    "resolve_manual_set",
    "resolve_all_sets",
    "build_canary_plan",
    "build_final10_plan",
    "build_confirmation_plan",
    "validate_plan",
]
