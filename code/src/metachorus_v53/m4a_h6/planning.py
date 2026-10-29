"""H6 third-seed and conditional R4 sealed execution plans."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Mapping

from metachorus_v53.audit.common import canonical_json_hash, read_json, sha256_file

from .common import envelope, read_csv, safe_relative, seal_rows, write_csv_once, write_json_once
from .constants import (
    ALL_DEVELOPMENT_MODEL_SEEDS,
    CANARY_UPDATES,
    CONTROLS,
    EXPECTED_POOL_COUNTS,
    EXPECTED_POOL_TOTAL,
    FREEBASE_CANARY_HASH,
    FREEBASE_CANARY_ID,
    FREEBASE_CANARY_ORDINAL,
    H5_R3_CHECKPOINT_REGISTRY_SHA256,
    MAX_R4_CONTINUATIONS,
    MAX_THIRD_SEED_RUNS,
    SEARCH_MODEL_SEEDS,
    THIRD_DEVELOPMENT_MODEL_SEED,
)
from .continuation import h6_physical_run_id, validate_stage_marker


PLAN_FIELDS = (
    "stage",
    "physical_run_id",
    "dataset",
    "candidate_set_id",
    "candidate_set_hash",
    "candidate_set_ordinal",
    "model_seed",
    "parent_source",
    "parent_physical_run_id",
    "parent_attempt_relative_path",
    "parent_checkpoint_relative_path",
    "parent_checkpoint_sha256",
    "parent_representation_relative_path",
    "parent_representation_sha256",
    "template_config_sha256",
    "path_ids_json",
    "continuation_from_update",
    "endpoint_update",
    "continuation_mode",
    "execution_action",
    "roles_json",
    "labels_mounted",
    "downstream_evaluations_during_training",
    "gpu_preference",
    "run_role",
    "attempt_index",
    "status",
    "plan_seal_sha256",
)


def _attempt_from_artifact(relative_path: str) -> str:
    path = safe_relative(relative_path)
    if (
        len(path.parts) != 5
        or path.parts[0] != "runs"
        or not path.parts[1]
        or not path.parts[2].startswith("attempt_")
        or path.parts[3] not in {"checkpoints", "representations"}
    ):
        raise ValueError(f"non-canonical parent artifact path: {relative_path}")
    return path.parent.parent.as_posix()


def _gpu_preference(dataset: str, candidate_hash: str, ordinal: int) -> int:
    for name, (_, control_hash) in CONTROLS[dataset].items():
        if candidate_hash == control_hash:
            return 0 if name == "manual" else 1
    return int(candidate_hash[:8], 16) % 2


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if str(value).casefold() in {"true", "1"}:
        return True
    if str(value).casefold() in {"false", "0"}:
        return False
    raise ValueError(f"invalid boolean value: {value!r}")


def _valid_sha256(value: Any) -> bool:
    return re.fullmatch(r"[0-9a-f]{64}", str(value)) is not None


def _canonical_plan_row(row: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(row)
    value.pop("plan_seal_sha256", None)
    for field in (
        "candidate_set_ordinal",
        "model_seed",
        "continuation_from_update",
        "endpoint_update",
        "downstream_evaluations",
        "downstream_evaluations_during_training",
        "gpu_preference",
        "attempt_index",
    ):
        if field in value:
            value[field] = int(value[field])
    for field in (
        "labels_mounted",
        "eligible_for_scientific_result",
        "eligible_for_selector_development",
        "reusable_for_science",
    ):
        if field in value:
            value[field] = _bool(value[field])
    return value


def _validated_envelope(
    path: Path,
    *,
    schema_version: str,
    status: str,
) -> dict[str, Any]:
    value = read_json(path)
    basis = value.get("basis")
    if (
        value.get("schema_version") != schema_version
        or value.get("status") != status
        or not isinstance(basis, Mapping)
        or value.get("identity") != canonical_json_hash(basis)
    ):
        raise RuntimeError(f"invalid sealed H6 artifact: {path.name}")
    return value


def _canonical_pool_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "dataset": str(row["dataset"]),
        "candidate_set_id": str(row["candidate_set_id"]),
        "candidate_set_hash": str(row["candidate_set_hash"]),
        "candidate_set_ordinal": int(row["candidate_set_ordinal"]),
        "roles_json": str(row["roles_json"]),
        "is_manual": _bool(row["is_manual"]),
        "is_current_automatic": _bool(row["is_current_automatic"]),
        "truth_fields_read": _bool(row["truth_fields_read"]),
        "h5_locked_report_access": _bool(row["h5_locked_report_access"]),
        "s_eval_access": _bool(row["s_eval_access"]),
    }


def validate_canary_plan_rows(rows: list[Mapping[str, Any]]) -> None:
    if len(rows) != 2 or {str(row.get("dataset")) for row in rows} != {"acm", "freebase"}:
        raise RuntimeError("H6 canary plan must contain exactly one ACM and one Freebase run")
    expected_candidates = {
        "acm": (*CONTROLS["acm"]["manual"], 0),
        "freebase": (FREEBASE_CANARY_ID, FREEBASE_CANARY_HASH, FREEBASE_CANARY_ORDINAL),
    }
    if {int(row.get("gpu_preference", -1)) for row in rows} != {0, 1}:
        raise RuntimeError("H6 canary plan must reserve one run per GPU")
    for row in rows:
        dataset = str(row.get("dataset"))
        candidate_id, candidate_hash, ordinal = expected_candidates[dataset]
        expected_run_id = h6_physical_run_id(
            "CANARY", dataset, ordinal, THIRD_DEVELOPMENT_MODEL_SEED
        )
        parent = safe_relative(str(row.get("parent_attempt_relative_path", "")))
        path_ids = json.loads(str(row.get("path_ids_json", "")))
        if (
            len(parent.parts) != 3
            or parent.parts[0] != "runs"
            or parent.parts[1] != str(row.get("parent_physical_run_id"))
            or re.fullmatch(r"attempt_\d{2}", parent.parts[2]) is None
            or str(row.get("stage")) != "CANARY"
            or str(row.get("physical_run_id")) != expected_run_id
            or str(row.get("candidate_set_id")) != candidate_id
            or str(row.get("candidate_set_hash")) != candidate_hash
            or int(row.get("candidate_set_ordinal", -1)) != ordinal
            or int(row.get("model_seed", -1)) != THIRD_DEVELOPMENT_MODEL_SEED
            or str(row.get("parent_source")) != "H5_R3_TEMPLATE_CONFIG_ONLY"
            or int(row.get("continuation_from_update", -1)) != 0
            or int(row.get("endpoint_update", -1)) != CANARY_UPDATES
            or str(row.get("continuation_mode")) != "FROM_ZERO_ENGINEERING_CANARY"
            or str(row.get("execution_action")) != "RUN_ENGINEERING_CANARY"
            or str(row.get("run_role")) != "M4A_H6_ENGINEERING_CANARY"
            or _bool(row.get("eligible_for_scientific_result"))
            or _bool(row.get("eligible_for_selector_development"))
            or _bool(row.get("labels_mounted"))
            or int(row.get("downstream_evaluations", -1)) != 0
            or _bool(row.get("reusable_for_science"))
            or int(row.get("attempt_index", -1)) != 0
            or str(row.get("status")) != "PENDING_GPU"
            or not isinstance(path_ids, list)
            or not path_ids
            or not all(isinstance(value, str) and value for value in path_ids)
        ):
            raise RuntimeError(f"invalid H6 canary plan row: {row.get('physical_run_id')}")


def validate_canary_gate(result_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    results = _validated_envelope(
        result_root / "M4A_H6_CANARY_RESULTS.json",
        schema_version="metachorus.v53.m4a-h6.canary-results.v1",
        status="PASS",
    )
    decision = _validated_envelope(
        result_root / "M4A_H6_CANARY_GATE_DECISION.json",
        schema_version="metachorus.v53.m4a-h6.canary-gate-decision.v1",
        status="H6_CANARY_PASS",
    )
    records = results["basis"].get("records", [])
    if (
        int(results["basis"].get("run_count", -1)) != 2
        or len(records) != 2
        or {str(row.get("dataset")) for row in records} != {"acm", "freebase"}
        or any(row.get("status") != "PASS" for row in records)
        or results["basis"].get("scientific_metrics_observed") is not False
        or results["basis"].get("fanout_started_before_canary_pass") is not False
        or results["basis"].get("labels_mounted") is not False
        or results["basis"].get("h5_locked_report_access") is not False
        or results["basis"].get("s_eval_started") is not False
        or decision["basis"].get("canary_results_identity") != results["identity"]
        or decision["basis"].get("fanout_authorized") is not True
        or decision["basis"].get("fanout_started_before_decision") is not False
        or decision["basis"].get("scientific_metrics_observed") is not False
        or decision["basis"].get("labels_mounted") is not False
        or decision["basis"].get("h5_locked_report_access") is not False
        or decision["basis"].get("s_eval_started") is not False
    ):
        raise RuntimeError("H6 canary gate lineage or engineering-only boundary is invalid")
    plan_path = result_root / "M4A_H6_CANARY_RUN_PLAN.csv"
    plan = read_csv(plan_path)
    plan_seal = _validated_envelope(
        result_root / "M4A_H6_CANARY_PLAN_SEAL.json",
        schema_version="metachorus.v53.m4a-h6.execution-plan-seal.v1",
        status="SEALED_BEFORE_GPU_TRAINING",
    )
    canonical_rows = [_canonical_plan_row(row) for row in plan]
    validate_canary_plan_rows(plan)
    if (
        len(plan) != 2
        or len({str(row["physical_run_id"]) for row in plan}) != 2
        or plan_seal["basis"].get("stage") != "CANARY"
        or int(plan_seal["basis"].get("row_count", -1)) != 2
        or plan_seal["basis"].get("rows_identity") != canonical_json_hash(canonical_rows)
        or plan_seal["basis"].get("labels_mounted") is not False
        or plan_seal["basis"].get("run_level_parallel_only") is not True
        or int(plan_seal["basis"].get("maximum_gpu_count", -1)) != 2
        or plan_seal["basis"].get("h5_locked_report_access") is not False
        or plan_seal["basis"].get("s_eval_started") is not False
        or any(str(row.get("plan_seal_sha256")) != plan_seal["identity"] for row in plan)
    ):
        raise RuntimeError("H6 canary plan seal is invalid")
    records_by_id = {str(row["physical_run_id"]): row for row in records}
    if len(records_by_id) != 2 or set(records_by_id) != {
        str(row["physical_run_id"]) for row in plan
    }:
        raise RuntimeError("H6 canary results do not match the sealed plan")
    for row in plan:
        run_id = str(row["physical_run_id"])
        attempts = [
            attempt
            for attempt in sorted((result_root / "canary_runs" / run_id).glob("attempt_*"))
            if (attempt / "H6_CANARY_000020_COMPLETE.json").is_file()
        ]
        if len(attempts) != 1:
            raise RuntimeError(f"H6 canary marker count mismatch: {run_id}")
        marker = validate_stage_marker(
            attempts[0] / "H6_CANARY_000020_COMPLETE.json",
            plan_row=row,
            endpoint_update=20,
            run_root=attempts[0],
        )
        record = records_by_id[run_id]
        if (
            record.get("stage_marker_identity") != marker["identity"]
            or record.get("checkpoint_sha256") != marker["checkpoint_sha256"]
            or record.get("representation_sha256") != marker["representation_sha256"]
            or int(record.get("successful_attempt_index", -1))
            != int(attempts[0].name.rsplit("_", 1)[-1])
        ):
            raise RuntimeError(f"H6 canary result/marker lineage mismatch: {run_id}")
    return results, decision


def validate_pool_seal(
    result_root: Path,
    h5_result_root: Path | None = None,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    pool_path = result_root / "TC_LPMSO_H6_R3_UNCERTAINTY_POOL.csv"
    components_path = result_root / "TC_LPMSO_H6_R3_UNCERTAINTY_POOL_COMPONENTS.csv"
    pool = read_csv(pool_path)
    seal = _validated_envelope(
        result_root / "TC_LPMSO_H6_R3_UNCERTAINTY_POOL_SEAL.json",
        schema_version="metachorus.v53.m4a-h6.r3-uncertainty-pool-seal.v1",
        status="SEALED_BEFORE_THIRD_SEED_TRAINING",
    )
    canonical_pool = [_canonical_pool_row(row) for row in pool]
    counts = {
        dataset: sum(row["dataset"] == dataset for row in canonical_pool)
        for dataset in ("acm", "freebase")
    }
    identities = {
        (row["dataset"], row["candidate_set_id"], row["candidate_set_hash"], row["candidate_set_ordinal"])
        for row in canonical_pool
    }
    hash_identities = {(row["dataset"], row["candidate_set_hash"]) for row in canonical_pool}
    id_identities = {(row["dataset"], row["candidate_set_id"]) for row in canonical_pool}
    ordinal_identities = {
        (row["dataset"], row["candidate_set_ordinal"]) for row in canonical_pool
    }
    basis = seal["basis"]
    if (
        len(pool) != EXPECTED_POOL_TOTAL
        or len(identities) != EXPECTED_POOL_TOTAL
        or len(hash_identities) != EXPECTED_POOL_TOTAL
        or len(id_identities) != EXPECTED_POOL_TOTAL
        or len(ordinal_identities) != EXPECTED_POOL_TOTAL
        or counts != EXPECTED_POOL_COUNTS
        or any(
            row["truth_fields_read"]
            or row["h5_locked_report_access"]
            or row["s_eval_access"]
            for row in canonical_pool
        )
        or basis.get("pool_sha256") != sha256_file(pool_path)
        or basis.get("components_sha256") != sha256_file(components_path)
        or basis.get("pool_identity") != canonical_json_hash(canonical_pool)
        or int(basis.get("acm_count", -1)) != EXPECTED_POOL_COUNTS["acm"]
        or int(basis.get("freebase_count", -1)) != EXPECTED_POOL_COUNTS["freebase"]
        or int(basis.get("total_count", -1)) != EXPECTED_POOL_TOTAL
        or basis.get("truth_artifacts_opened_before_seal") is not False
        or basis.get("h5_locked_report_access") is not False
        or basis.get("s_eval_started") is not False
        or basis.get("third_seed_started") is not False
    ):
        raise RuntimeError("H6 uncertainty pool seal does not bind the frozen 10/75/85 pool")
    if h5_result_root is not None:
        source_relative = safe_relative(str(basis.get("source_artifact", "")))
        if len(source_relative.parts) != 1:
            raise RuntimeError("H6 pool source artifact must be a top-level H5 file")
        source = h5_result_root.joinpath(*source_relative.parts)
        if not source.is_file() or sha256_file(source) != str(basis.get("source_sha256")):
            raise RuntimeError("H6 pool seal no longer binds the immutable H5 R3 metrics")
    return pool, seal


def validate_third_seed_plan_rows(
    rows: list[Mapping[str, Any]],
    *,
    pool_rows: list[Mapping[str, Any]] | None = None,
) -> None:
    if len(rows) != MAX_THIRD_SEED_RUNS:
        raise RuntimeError("H6 third-seed plan is not exactly 85 runs")
    run_ids = {str(row["physical_run_id"]) for row in rows}
    candidate_keys = {
        (str(row["dataset"]), str(row["candidate_set_hash"])) for row in rows
    }
    candidate_ids = {
        (str(row["dataset"]), str(row["candidate_set_id"])) for row in rows
    }
    candidate_ordinals = {
        (str(row["dataset"]), int(row["candidate_set_ordinal"])) for row in rows
    }
    counts = {
        dataset: sum(str(row["dataset"]) == dataset for row in rows)
        for dataset in ("acm", "freebase")
    }
    if (
        len(run_ids) != EXPECTED_POOL_TOTAL
        or len(candidate_keys) != EXPECTED_POOL_TOTAL
        or len(candidate_ids) != EXPECTED_POOL_TOTAL
        or len(candidate_ordinals) != EXPECTED_POOL_TOTAL
    ):
        raise RuntimeError("H6 third-seed plan contains duplicate identities")
    if counts != EXPECTED_POOL_COUNTS:
        raise RuntimeError(f"H6 third-seed dataset counts drifted: {counts}")
    if pool_rows is not None:
        planned = {
            (
                str(row["dataset"]),
                str(row["candidate_set_id"]),
                str(row["candidate_set_hash"]),
                int(row["candidate_set_ordinal"]),
            )
            for row in rows
        }
        frozen = {
            (
                str(row["dataset"]),
                str(row["candidate_set_id"]),
                str(row["candidate_set_hash"]),
                int(row["candidate_set_ordinal"]),
            )
            for row in pool_rows
        }
        if planned != frozen or len(frozen) != EXPECTED_POOL_TOTAL:
            raise RuntimeError("H6 third-seed plan no longer matches the frozen pool")
    for row in rows:
        expected_run_id = h6_physical_run_id(
            "THIRD_SEED_R3_AUDIT",
            str(row["dataset"]),
            int(row["candidate_set_ordinal"]),
            THIRD_DEVELOPMENT_MODEL_SEED,
        )
        path_ids = json.loads(str(row["path_ids_json"]))
        if (
            str(row["physical_run_id"]) != expected_run_id
            or str(row.get("stage")) != "THIRD_SEED_R3_AUDIT"
            or not str(row.get("candidate_set_id", ""))
            or not _valid_sha256(row.get("candidate_set_hash"))
            or int(row["model_seed"]) != THIRD_DEVELOPMENT_MODEL_SEED
            or str(row["parent_source"]) != "H5_R3_TEMPLATE_CONFIG_ONLY"
            or str(row["parent_checkpoint_relative_path"]) != ""
            or str(row["parent_checkpoint_sha256"]) != ""
            or str(row["parent_representation_relative_path"]) != ""
            or str(row["parent_representation_sha256"]) != ""
            or int(row["continuation_from_update"]) != 0
            or int(row["endpoint_update"]) != 2000
            or str(row["continuation_mode"]) != "FROM_ZERO_THIRD_DEVELOPMENT_SEED"
            or str(row["execution_action"]) != "TRAIN_H6_THIRD_SEED_FROM_ZERO"
            or str(row["run_role"]) != "M4A_H6_THIRD_SEED_R3_AUDIT"
            or _bool(row["labels_mounted"])
            or int(row["downstream_evaluations_during_training"]) != 0
            or int(row.get("attempt_index", 0)) != 0
            or str(row.get("status", "PENDING_GPU")) != "PENDING_GPU"
            or not isinstance(path_ids, list)
            or not path_ids
            or not all(isinstance(value, str) and value for value in path_ids)
        ):
            raise RuntimeError(f"invalid H6 third-seed plan row: {row['physical_run_id']}")


def validate_r4_plan_rows(rows: list[Mapping[str, Any]]) -> None:
    if not rows or len(rows) > MAX_R4_CONTINUATIONS:
        raise RuntimeError("H6 R4 plan must contain between 1 and 48 continuations")
    run_ids = {str(row["physical_run_id"]) for row in rows}
    if len(run_ids) != len(rows):
        raise RuntimeError("H6 R4 plan contains duplicate physical identities")
    by_candidate: dict[tuple[str, str], set[int]] = {}
    for row in rows:
        key = (str(row["dataset"]), str(row["candidate_set_hash"]))
        by_candidate.setdefault(key, set()).add(int(row["model_seed"]))
        expected_run_id = h6_physical_run_id(
            "R4", str(row["dataset"]), int(row["candidate_set_ordinal"]), int(row["model_seed"])
        )
        if (
            str(row["physical_run_id"]) != expected_run_id
            or str(row.get("stage")) != "R4"
            or not str(row.get("candidate_set_id", ""))
            or not _valid_sha256(row.get("candidate_set_hash"))
            or int(row["continuation_from_update"]) != 2000
            or int(row["endpoint_update"]) != 3000
            or str(row["continuation_mode"]) != "STRICT_CONTINUATION"
            or str(row["execution_action"]) != "CONTINUE_THREE_SEEDS_TO_R4"
            or str(row["run_role"]) != "M4A_H6_R4_SEARCH"
            or _bool(row["labels_mounted"])
            or int(row["downstream_evaluations_during_training"]) != 0
            or len(str(row["parent_checkpoint_sha256"])) != 64
            or len(str(row["parent_representation_sha256"])) != 64
            or int(row.get("attempt_index", 0)) != 0
            or str(row.get("status", "PENDING_GPU")) != "PENDING_GPU"
        ):
            raise RuntimeError(f"invalid H6 R4 plan row: {row['physical_run_id']}")
    if any(seeds != set(ALL_DEVELOPMENT_MODEL_SEEDS) for seeds in by_candidate.values()):
        raise RuntimeError("every H6 R4 candidate must continue all three development seeds")


def _validate_parent_artifact(
    *,
    root: Path,
    parent: Mapping[str, Any],
    relative_field: str,
    sha_field: str,
    expected_directory: str,
) -> Path:
    relative = safe_relative(str(parent[relative_field]))
    if (
        len(relative.parts) != 5
        or relative.parts[0] != "runs"
        or relative.parts[1] != str(parent["physical_run_id"])
        or not re.fullmatch(r"attempt_\d{2}", relative.parts[2])
        or relative.parts[3] != expected_directory
        or relative.parts[4] != "update_002000." + ("pt" if expected_directory == "checkpoints" else "npy")
    ):
        raise RuntimeError(f"non-canonical H6 R4 parent artifact: {relative.as_posix()}")
    path = root.joinpath(*relative.parts)
    if not path.is_file() or path.is_symlink() or sha256_file(path) != str(parent[sha_field]):
        raise RuntimeError(f"missing or changed H6 R4 parent artifact: {relative.as_posix()}")
    return path


def validate_third_seed_parent_lineage(
    rows: list[Mapping[str, Any]],
    *,
    h5_result_root: Path,
    h6_result_root: Path,
) -> None:
    registry_path = h5_result_root / "TC_LPMSO_H5_R3_CHECKPOINT_REGISTRY.csv"
    if sha256_file(registry_path) != H5_R3_CHECKPOINT_REGISTRY_SHA256:
        raise RuntimeError("immutable H5 R3 checkpoint registry changed")
    registry_rows = read_csv(registry_path)
    registry = {
        (str(row["dataset"]), str(row["candidate_set_hash"]), int(row["model_seed"])): row
        for row in registry_rows
    }
    if len(registry) != len(registry_rows):
        raise RuntimeError("H5 R3 checkpoint registry contains duplicate identities")
    path_rows = read_csv(h6_result_root / "M4A_H6_PATH_SET_RESOLUTION_AUDIT.csv")
    path_by_hash = {str(row["candidate_set_hash"]): row for row in path_rows}
    if len(path_by_hash) != EXPECTED_POOL_TOTAL:
        raise RuntimeError("H6 path-set audit identity count drifted")
    for row in rows:
        key = (
            str(row["dataset"]),
            str(row["candidate_set_hash"]),
            SEARCH_MODEL_SEEDS[0],
        )
        parent = registry.get(key)
        path_row = path_by_hash.get(str(row["candidate_set_hash"]))
        if parent is None or path_row is None:
            raise RuntimeError(f"missing H6 third-seed template lineage: {key[:2]}")
        attempt = _attempt_from_artifact(str(parent["checkpoint_relative_path"]))
        config = h5_result_root / attempt / "RESOLVED_CONFIG.json"
        if (
            int(parent["endpoint_update"]) != 2000
            or str(parent["status"]) != "PASS"
            or str(parent["candidate_set_id"]) != str(row["candidate_set_id"])
            or int(parent["candidate_set_ordinal"]) != int(row["candidate_set_ordinal"])
            or str(row["parent_physical_run_id"]) != str(parent["physical_run_id"])
            or str(row["parent_attempt_relative_path"]) != attempt
            or str(path_row["candidate_set_id"]) != str(row["candidate_set_id"])
            or int(path_row["candidate_set_ordinal"]) != int(row["candidate_set_ordinal"])
            or str(path_row["path_ids_json"]) != str(row["path_ids_json"])
            or not config.is_file()
            or config.is_symlink()
            or sha256_file(config) != str(row["template_config_sha256"])
        ):
            raise RuntimeError(f"H6 third-seed template lineage mismatch: {row['physical_run_id']}")


def validate_r4_parent_lineage(
    rows: list[Mapping[str, Any]],
    *,
    h5_result_root: Path,
    h6_result_root: Path,
) -> None:
    h5_registry_path = h5_result_root / "TC_LPMSO_H5_R3_CHECKPOINT_REGISTRY.csv"
    if sha256_file(h5_registry_path) != H5_R3_CHECKPOINT_REGISTRY_SHA256:
        raise RuntimeError("immutable H5 R3 checkpoint registry changed")
    h6_registry_path = h6_result_root / "TC_LPMSO_H6_THIRD_SEED_CHECKPOINT_REGISTRY.csv"
    h6_seal = _validated_envelope(
        h6_result_root / "TC_LPMSO_H6_THIRD_SEED_REGISTRY_SEAL.json",
        schema_version="metachorus.v53.m4a-h6.stage-registry-seal.v1",
        status="SEALED_BEFORE_STAGE_EVALUATION",
    )
    if (
        h6_seal["basis"].get("stage") != "THIRD_SEED_R3_AUDIT"
        or int(h6_seal["basis"].get("endpoint_update", -1)) != 2000
        or int(h6_seal["basis"].get("checkpoint_count", -1)) != EXPECTED_POOL_TOTAL
        or h6_seal["basis"].get("checkpoint_registry_sha256") != sha256_file(h6_registry_path)
        or h6_seal["basis"].get("all_status_pass") is not True
    ):
        raise RuntimeError("H6 third-seed registry seal is invalid for R4 continuation")
    h5_rows = read_csv(h5_registry_path)
    h6_rows = read_csv(h6_registry_path)
    h5_registry = {
        (str(row["dataset"]), str(row["candidate_set_hash"]), int(row["model_seed"])): row
        for row in h5_rows
    }
    h6_registry = {
        (str(row["dataset"]), str(row["candidate_set_hash"]), int(row["model_seed"])): row
        for row in h6_rows
    }
    if len(h5_registry) != len(h5_rows) or len(h6_registry) != len(h6_rows):
        raise RuntimeError("H6 R4 parent registries contain duplicate identities")
    path_rows = read_csv(h6_result_root / "M4A_H6_PATH_SET_RESOLUTION_AUDIT.csv")
    path_by_hash = {str(row["candidate_set_hash"]): row for row in path_rows}
    if len(path_by_hash) != EXPECTED_POOL_TOTAL:
        raise RuntimeError("H6 path-set audit identity count drifted")
    for row in rows:
        seed = int(row["model_seed"])
        parent_root = h5_result_root if seed in SEARCH_MODEL_SEEDS else h6_result_root
        registry = h5_registry if seed in SEARCH_MODEL_SEEDS else h6_registry
        expected_source = "H5_RESULT_ROOT" if seed in SEARCH_MODEL_SEEDS else "H6_RESULT_ROOT"
        key = (str(row["dataset"]), str(row["candidate_set_hash"]), seed)
        parent = registry.get(key)
        path_row = path_by_hash.get(str(row["candidate_set_hash"]))
        if parent is None or path_row is None:
            raise RuntimeError(f"missing exact H6 R4 parent: {key}")
        checkpoint = _validate_parent_artifact(
            root=parent_root,
            parent=parent,
            relative_field="checkpoint_relative_path",
            sha_field="checkpoint_sha256",
            expected_directory="checkpoints",
        )
        _validate_parent_artifact(
            root=parent_root,
            parent=parent,
            relative_field="representation_relative_path",
            sha_field="representation_sha256",
            expected_directory="representations",
        )
        attempt = _attempt_from_artifact(str(parent["checkpoint_relative_path"]))
        config = parent_root / attempt / "RESOLVED_CONFIG.json"
        if (
            str(row["parent_source"]) != expected_source
            or int(parent["endpoint_update"]) != 2000
            or str(parent["status"]) != "PASS"
            or str(parent["candidate_set_id"]) != str(row["candidate_set_id"])
            or int(parent["candidate_set_ordinal"]) != int(row["candidate_set_ordinal"])
            or str(row["parent_physical_run_id"]) != str(parent["physical_run_id"])
            or str(row["parent_attempt_relative_path"]) != attempt
            or str(row["parent_checkpoint_relative_path"]) != str(parent["checkpoint_relative_path"])
            or str(row["parent_checkpoint_sha256"]) != str(parent["checkpoint_sha256"])
            or str(row["parent_representation_relative_path"])
            != str(parent["representation_relative_path"])
            or str(row["parent_representation_sha256"]) != str(parent["representation_sha256"])
            or checkpoint.parent.parent != config.parent
            or not config.is_file()
            or config.is_symlink()
            or sha256_file(config) != str(row["template_config_sha256"])
            or str(path_row["candidate_set_id"]) != str(row["candidate_set_id"])
            or int(path_row["candidate_set_ordinal"]) != int(row["candidate_set_ordinal"])
            or str(path_row["path_ids_json"]) != str(row["path_ids_json"])
        ):
            raise RuntimeError(f"H6 R4 parent lineage mismatch: {row['physical_run_id']}")


def build_third_seed_plan(result_root: Path, h5_result_root: Path) -> dict[str, Any]:
    _, canary = validate_canary_gate(result_root)
    pool, pool_seal = validate_pool_seal(result_root, h5_result_root)
    path_rows = read_csv(result_root / "M4A_H6_PATH_SET_RESOLUTION_AUDIT.csv")
    if len(path_rows) != EXPECTED_POOL_TOTAL:
        raise RuntimeError("H6 path-set audit no longer covers the complete frozen pool")
    path_by_hash = {str(row["candidate_set_hash"]): row for row in path_rows}
    if len(path_by_hash) != EXPECTED_POOL_TOTAL or set(path_by_hash) != {
        str(row["candidate_set_hash"]) for row in pool
    }:
        raise RuntimeError("H6 path-set audit identity mismatch")
    h5_registry = read_csv(h5_result_root / "TC_LPMSO_H5_R3_CHECKPOINT_REGISTRY.csv")
    template_rows = [row for row in h5_registry if int(row["model_seed"]) == SEARCH_MODEL_SEEDS[0]]
    templates = {str(row["candidate_set_hash"]): row for row in template_rows}
    if len(templates) != len(template_rows):
        raise RuntimeError("H5 R3 template registry contains duplicate candidate identities")
    rows: list[dict[str, Any]] = []
    for candidate in pool:
        dataset = str(candidate["dataset"])
        candidate_hash = str(candidate["candidate_set_hash"])
        ordinal = int(candidate["candidate_set_ordinal"])
        template = templates.get(candidate_hash)
        if template is None:
            raise ValueError(f"missing H5 R3 template for {candidate_hash}")
        attempt = _attempt_from_artifact(str(template["checkpoint_relative_path"]))
        config = h5_result_root / attempt / "RESOLVED_CONFIG.json"
        if not config.is_file():
            raise FileNotFoundError(config)
        rows.append(
            {
                "stage": "THIRD_SEED_R3_AUDIT",
                "physical_run_id": h6_physical_run_id(
                    "THIRD_SEED_R3_AUDIT",
                    dataset,
                    ordinal,
                    THIRD_DEVELOPMENT_MODEL_SEED,
                ),
                "dataset": dataset,
                "candidate_set_id": str(candidate["candidate_set_id"]),
                "candidate_set_hash": candidate_hash,
                "candidate_set_ordinal": ordinal,
                "model_seed": THIRD_DEVELOPMENT_MODEL_SEED,
                "parent_source": "H5_R3_TEMPLATE_CONFIG_ONLY",
                "parent_physical_run_id": str(template["physical_run_id"]),
                "parent_attempt_relative_path": attempt,
                "parent_checkpoint_relative_path": "",
                "parent_checkpoint_sha256": "",
                "parent_representation_relative_path": "",
                "parent_representation_sha256": "",
                "template_config_sha256": sha256_file(config),
                "path_ids_json": str(path_by_hash[candidate_hash]["path_ids_json"]),
                "continuation_from_update": 0,
                "endpoint_update": 2000,
                "continuation_mode": "FROM_ZERO_THIRD_DEVELOPMENT_SEED",
                "execution_action": "TRAIN_H6_THIRD_SEED_FROM_ZERO",
                "roles_json": str(candidate["roles_json"]),
                "labels_mounted": False,
                "downstream_evaluations_during_training": 0,
                "gpu_preference": _gpu_preference(dataset, candidate_hash, ordinal),
                "run_role": "M4A_H6_THIRD_SEED_R3_AUDIT",
                "attempt_index": 0,
                "status": "PENDING_GPU",
            }
        )
    validate_third_seed_plan_rows(rows, pool_rows=pool)
    validate_third_seed_parent_lineage(
        rows,
        h5_result_root=h5_result_root,
        h6_result_root=result_root,
    )
    sealed, seal = seal_rows(rows, stage="THIRD_SEED_R3_AUDIT")
    validate_third_seed_plan_rows(sealed, pool_rows=pool)
    plan_path = write_csv_once(
        result_root / "TC_LPMSO_H6_THIRD_SEED_RUN_PLAN.csv",
        sealed,
        fieldnames=PLAN_FIELDS,
    )
    write_json_once(result_root / "TC_LPMSO_H6_THIRD_SEED_PLAN_SEAL.json", seal)
    role_basis = {
        "old_h5_third_seed_started": False,
        "h6_third_seed": THIRD_DEVELOPMENT_MODEL_SEED,
        "h6_run_role": "M4A_H6_THIRD_SEED_R3_AUDIT",
        "from_zero": True,
        "old_search_checkpoints_reused": False,
        "old_optimizer_states_reused": False,
        "pool_seal_identity": pool_seal["identity"],
        "plan_sha256": sha256_file(plan_path),
        "plan_seal_identity": seal["identity"],
        "labels_mounted": False,
        "h5_locked_report_access": False,
        "s_eval_started": False,
    }
    role = envelope(
        "metachorus.v53.m4a-h6.third-seed-role-supersession.v1",
        "H6_THIRD_SEED_ROLE_ACTIVE",
        role_basis,
    )
    write_json_once(result_root / "TC_LPMSO_H6_THIRD_SEED_ROLE_SUPERSESSION.json", role)
    return {"plan": plan_path, "seal": seal, "role": role}


def build_r4_plan(result_root: Path, h5_result_root: Path) -> dict[str, Any]:
    gate = _validated_envelope(
        result_root / "TC_LPMSO_H6_THREE_SEED_R3_GATE_DECISION.json",
        schema_version="metachorus.v53.m4a-h6.three-seed-r3-gate-decision.v1",
        status="TC_LPMSO_H6_THREE_SEED_R3_PASS",
    )
    promotion_path = result_root / "TC_LPMSO_H6_R4_PROMOTION_TABLE.csv"
    pareto_path = result_root / "TC_LPMSO_H6_R3_PARETO_FRONTS.csv"
    promotion_seal = _validated_envelope(
        result_root / "TC_LPMSO_H6_R4_PROMOTION_SEAL.json",
        schema_version="metachorus.v53.m4a-h6.r4-promotion-seal.v1",
        status="SEALED_BEFORE_R4_TRAINING",
    )
    promotion = read_csv(promotion_path)
    promotion_basis = promotion_seal["basis"]
    promotion_keys = {(str(row["dataset"]), str(row["candidate_set_hash"])) for row in promotion}
    if (
        len(promotion_keys) != len(promotion)
        or promotion_basis.get("three_seed_gate_identity") != gate["identity"]
        or promotion_basis.get("r3_pareto_sha256") != sha256_file(pareto_path)
        or promotion_basis.get("promotion_table_sha256") != sha256_file(promotion_path)
        or int(promotion_basis.get("acm_promotion_slots", -1)) != 2
        or int(promotion_basis.get("freebase_promotion_slots", -1)) != 10
        or promotion_basis.get("controls_outside_slots") is not True
        or int(promotion_basis.get("promotion_row_count", -1)) != len(promotion)
    ):
        raise RuntimeError("H6 R4 promotion seal does not bind the gate and promotion table")
    for dataset in ("acm", "freebase"):
        cohort = [row for row in promotion if str(row["dataset"]) == dataset]
        slot_count = sum(_bool(row["consumes_promotion_slot"]) for row in cohort)
        hashes = {str(row["candidate_set_hash"]) for row in cohort}
        controls = {candidate_hash for _, candidate_hash in CONTROLS[dataset].values()}
        if slot_count != {"acm": 2, "freebase": 10}[dataset] or not controls.issubset(hashes):
            raise RuntimeError(f"H6 R4 promotion membership drifted for {dataset}")
    path_rows = read_csv(result_root / "M4A_H6_PATH_SET_RESOLUTION_AUDIT.csv")
    path_by_hash = {str(row["candidate_set_hash"]): row for row in path_rows}
    h5_registry_rows = read_csv(h5_result_root / "TC_LPMSO_H5_R3_CHECKPOINT_REGISTRY.csv")
    h5_registry = {
        (
            str(row["dataset"]),
            str(row["candidate_set_hash"]),
            int(row["model_seed"]),
        ): row
        for row in h5_registry_rows
    }
    h6_registry_rows = read_csv(result_root / "TC_LPMSO_H6_THIRD_SEED_CHECKPOINT_REGISTRY.csv")
    h6_registry = {
        (
            str(row["dataset"]),
            str(row["candidate_set_hash"]),
            int(row["model_seed"]),
        ): row
        for row in h6_registry_rows
    }
    if len(h5_registry) != len(h5_registry_rows) or len(h6_registry) != len(h6_registry_rows):
        raise RuntimeError("H6 R4 parent registries contain duplicate identities")
    rows: list[dict[str, Any]] = []
    for candidate in promotion:
        dataset = str(candidate["dataset"])
        candidate_hash = str(candidate["candidate_set_hash"])
        ordinal = int(candidate["candidate_set_ordinal"])
        for seed in ALL_DEVELOPMENT_MODEL_SEEDS:
            source = h5_registry if seed in SEARCH_MODEL_SEEDS else h6_registry
            parent = source.get((dataset, candidate_hash, seed))
            if parent is None or int(parent["endpoint_update"]) != 2000 or parent["status"] != "PASS":
                raise ValueError(f"missing exact R4 parent state: {dataset}/{candidate_hash}/{seed}")
            parent_root = "H5_RESULT_ROOT" if seed in SEARCH_MODEL_SEEDS else "H6_RESULT_ROOT"
            source_root = h5_result_root if parent_root == "H5_RESULT_ROOT" else result_root
            checkpoint = _validate_parent_artifact(
                root=source_root,
                parent=parent,
                relative_field="checkpoint_relative_path",
                sha_field="checkpoint_sha256",
                expected_directory="checkpoints",
            )
            _validate_parent_artifact(
                root=source_root,
                parent=parent,
                relative_field="representation_relative_path",
                sha_field="representation_sha256",
                expected_directory="representations",
            )
            attempt = _attempt_from_artifact(str(parent["checkpoint_relative_path"]))
            config_root = h5_result_root if parent_root == "H5_RESULT_ROOT" else result_root
            config = config_root / attempt / "RESOLVED_CONFIG.json"
            if checkpoint.parent.parent != config.parent or not config.is_file() or config.is_symlink():
                raise RuntimeError(f"missing canonical H6 R4 parent config: {attempt}")
            rows.append(
                {
                    "stage": "R4",
                    "physical_run_id": h6_physical_run_id("R4", dataset, ordinal, seed),
                    "dataset": dataset,
                    "candidate_set_id": str(candidate["candidate_set_id"]),
                    "candidate_set_hash": candidate_hash,
                    "candidate_set_ordinal": ordinal,
                    "model_seed": seed,
                    "parent_source": parent_root,
                    "parent_physical_run_id": str(parent["physical_run_id"]),
                    "parent_attempt_relative_path": attempt,
                    "parent_checkpoint_relative_path": str(parent["checkpoint_relative_path"]),
                    "parent_checkpoint_sha256": str(parent["checkpoint_sha256"]),
                    "parent_representation_relative_path": str(parent["representation_relative_path"]),
                    "parent_representation_sha256": str(parent["representation_sha256"]),
                    "template_config_sha256": sha256_file(config),
                    "path_ids_json": str(path_by_hash[candidate_hash]["path_ids_json"]),
                    "continuation_from_update": 2000,
                    "endpoint_update": 3000,
                    "continuation_mode": "STRICT_CONTINUATION",
                    "execution_action": "CONTINUE_THREE_SEEDS_TO_R4",
                    "roles_json": str(candidate["roles_json"]),
                    "labels_mounted": False,
                    "downstream_evaluations_during_training": 0,
                    "gpu_preference": _gpu_preference(dataset, candidate_hash, ordinal),
                    "run_role": "M4A_H6_R4_SEARCH",
                    "attempt_index": 0,
                    "status": "PENDING_GPU",
                }
            )
    validate_r4_plan_rows(rows)
    validate_r4_parent_lineage(
        rows,
        h5_result_root=h5_result_root,
        h6_result_root=result_root,
    )
    sealed, seal = seal_rows(rows, stage="R4")
    validate_r4_plan_rows(sealed)
    path = write_csv_once(
        result_root / "TC_LPMSO_H6_R4_RUN_PLAN.csv",
        sealed,
        fieldnames=PLAN_FIELDS,
    )
    write_json_once(result_root / "TC_LPMSO_H6_R4_PLAN_SEAL.json", seal)
    return {"plan": path, "seal": seal}
