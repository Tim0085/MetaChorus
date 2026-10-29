"""Append-only execution stages for the M6F search and M7F preliminary run."""

from __future__ import annotations

import csv
import json
import math
import os
import re
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from metachorus_v53.audit.common import canonical_json_hash, read_json, sha256_file

from .common import envelope, read_csv, write_csv_once, write_json_once, write_text_once
from .constants import (
    CANARY_ENDPOINT,
    COMMON_ROOT_CAUSE_PAUSE_THRESHOLD,
    DATASET_NAMES,
    EVAL_MODEL_SEEDS,
    MAX_ATTEMPTS_PER_IDENTITY,
    MAX_GPUS,
    PRELIM_ENDPOINT,
    RESOURCE_LIMITS,
    SEARCH_ENDPOINTS,
    SEARCH_MODEL_SEEDS,
)
from .evaluation import (
    PreliminaryLabelSession,
    evaluate_preliminary_representation,
    evaluate_search_representation,
    open_search_mount,
)
from .planning import build_preliminary_matrix, physical_run_id
from .protocol import (
    METRICS,
    aggregate_objectives,
    close_search_selection,
    derive_r1_high_recall_pool,
    derive_robust_shortlist,
    manual_safety_gate,
    select_unique_robust_set,
    preliminary_decision,
)
from .training import run_plan_row, validate_run_marker


def _json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    parsed = json.loads(str(value))
    if not isinstance(parsed, list):
        raise ValueError("expected a JSON list")
    return parsed


def _canonical_plan_row(row: Mapping[str, Any]) -> dict[str, Any]:
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
        if field in value and value[field] != "":
            value[field] = int(value[field])
    for field in ("path_ids", "checkpoint_schedule", "logical_roles"):
        if field in value and not isinstance(value[field], list):
            value[field] = _json_list(value[field])
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
            token = value[field]
            if isinstance(token, bool):
                continue
            if str(token).casefold() not in {"true", "false"}:
                raise ValueError(f"invalid plan boolean: {field}")
            value[field] = str(token).casefold() == "true"
    return value


def _unsealed_rows_identity(rows: Sequence[Mapping[str, Any]]) -> str:
    return canonical_json_hash(
        [
            {key: value for key, value in row.items() if key != "plan_seal_identity"}
            for row in rows
        ]
    )


def validate_plan(
    plan_path: str | Path,
    seal_path: str | Path,
    *,
    expected_status: str,
    expected_stage: str | Sequence[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = [_canonical_plan_row(row) for row in read_csv(plan_path)]
    seal = read_json(seal_path)
    basis = seal.get("basis", {})
    expected_stages = {expected_stage} if isinstance(expected_stage, str) else set(expected_stage)
    if (
        seal.get("status") != expected_status
        or seal.get("identity") != canonical_json_hash(basis)
        or int(basis.get("row_count", -1)) != len(rows)
        or basis.get("rows_identity") != _unsealed_rows_identity(rows)
        or any(row.get("stage") not in expected_stages for row in rows)
        or any(row.get("labels_mounted") is not False for row in rows)
        or any(int(row.get("downstream_evaluations_during_training", -1)) != 0 for row in rows)
        or any(row.get("plan_seal_identity") != seal.get("identity") for row in rows)
    ):
        raise ValueError("fast plan/seal validation failed")
    return rows, seal


def authorize_search_after_canary(
    result_root: str | Path,
    *,
    source_commit: str,
    source_tree: str,
    active_datasets: Sequence[str] | None = None,
    budget_gate: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    result = Path(result_root).resolve()
    _, gate = close_canaries(result)
    sealed_budget = read_json(result / "FAST_R1_MEASURED_BUDGET_GATE.json")
    budget_basis = sealed_budget.get("basis", {})
    if (
        sealed_budget.get("schema_version") != "metachorus.v53.m6f-m7f.r1-measured-budget-gate.v1"
        or sealed_budget.get("status") not in {"PASS", "PASS_WITH_SOFT_RESOURCE_WARNING", "PARTIAL_PASS_WITH_BLOCKED_DATASETS"}
        or sealed_budget.get("identity") != canonical_json_hash(budget_basis)
        or budget_gate is None
        or str(budget_gate.get("identity")) != str(sealed_budget["identity"])
    ):
        raise ValueError("authorized R1 lacks its exact measured budget gate")
    draft, draft_seal = validate_plan(
        result / "FAST_SEARCH_EXPERIMENT_MATRIX.csv",
        result / "FAST_SEARCH_EXPERIMENT_MATRIX_SEAL.json",
        expected_status="SEALED_BEFORE_SEARCH_GPU_TRAINING",
        expected_stage="R1_ALL_SETS_0000_TO_0300",
    )
    active = tuple(DATASET_NAMES if active_datasets is None else active_datasets)
    if (
        not active
        or len(set(active)) != len(active)
        or any(name not in DATASET_NAMES for name in active)
        or tuple(budget_basis.get("active_datasets", [])) != active
    ):
        raise ValueError("authorized R1 datasets are invalid")
    rows = [
        {
            **row,
            "stage": "SEARCH",
            "source_commit": str(source_commit),
            "source_tree": str(source_tree),
            "status": "AUTHORIZED",
            "plan_seal_identity": "",
        }
        for row in draft
        if str(row["dataset"]) in active
    ]
    unsealed = [{key: value for key, value in row.items() if key != "plan_seal_identity"} for row in rows]
    seal = envelope(
        "metachorus.v53.m6f-m7f.r1-authorized-plan-seal.v1",
        "SEALED_AFTER_CANARY_BEFORE_R1_GPU_TRAINING",
        {
            "row_count": len(rows),
            "rows_identity": canonical_json_hash(unsealed),
            "draft_plan_seal_identity": draft_seal["identity"],
            "canary_gate_identity": gate["identity"],
            "source_commit": str(source_commit),
            "source_tree": str(source_tree),
            "active_datasets": list(active),
            "blocked_datasets": [name for name in DATASET_NAMES if name not in active],
            "dataset_budget_gate_identity": "" if budget_gate is None else str(budget_gate["identity"]),
            "labels_mounted": False,
            "prelim_locked_report_access": False,
        },
    )
    sealed = [{**row, "plan_seal_identity": seal["identity"]} for row in rows]
    write_csv_once(result / "FAST_SEARCH_R1_RUN_PLAN.csv", sealed, fieldnames=list(sealed[0]))
    write_json_once(result / "FAST_SEARCH_R1_RUN_PLAN_SEAL.json", seal)
    return sealed, seal


def _root_cause(error: BaseException) -> tuple[str, str]:
    message = str(error).casefold()
    message = re.sub(r"[a-z]:[\\/][^\s]+|/(?:[^\s/]+/)+[^\s]+", "<path>", message)
    message = re.sub(r"\b[0-9a-f]{32,64}\b", "<hash>", message)
    message = re.sub(r"\b\d+\b", "<n>", message)
    signature = canonical_json_hash({"kind": type(error).__name__, "message": message})
    return type(error).__name__, canonical_json_hash({"signature": signature})


def _failure_path(result_root: Path, stage: str, dataset: str, run_id: str, attempt: int) -> Path:
    return result_root / "failed_attempts" / dataset / stage.casefold() / f"{run_id}__attempt_{attempt:02d}.json"


def _dataset_pause(result_root: Path, stage: str, dataset: str) -> Path:
    return result_root / f"FAST_{dataset.upper()}_{stage}_COMMON_ROOT_CAUSE_PAUSE.json"


def _write_dataset_pause_once(
    path: Path,
    *,
    stage: str,
    dataset: str,
    root_cause_identity: str,
    affected_physical_run_ids: Sequence[str],
) -> None:
    value = envelope(
        "metachorus.v53.m6f-m7f.dataset-common-root-cause-pause.v1",
        "IMMEDIATE_DATASET_COMMON_ROOT_CAUSE_PAUSE",
        {
            "stage": stage,
            "dataset": dataset,
            "root_cause_identity": root_cause_identity,
            "affected_physical_run_ids": sorted(set(affected_physical_run_ids)),
            "fanout_must_stop_for_dataset": True,
            "other_datasets_may_continue": True,
            "completed_runs_preserved": True,
        },
    )
    payload = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    handle, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            pass
    finally:
        temporary.unlink(missing_ok=True)
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("dataset pause receipt is not a regular file")
    existing = read_json(path)
    basis = existing.get("basis", {})
    existing_affected = basis.get("affected_physical_run_ids", [])
    canonical_affected = (
        sorted(set(map(str, existing_affected)))
        if isinstance(existing_affected, list)
        else []
    )
    actual_affected: set[str] = set()
    failure_root = path.parent / "failed_attempts" / dataset / stage.casefold()
    for receipt_path in failure_root.glob("*__attempt_*.json"):
        receipt = read_json(receipt_path)
        receipt_basis = receipt.get("basis", {})
        if (
            receipt.get("schema_version") == "metachorus.v53.m6f-m7f.failed-attempt.v1"
            and receipt.get("status") == "FAILED_PRESERVED"
            and receipt.get("identity") == canonical_json_hash(receipt_basis)
            and receipt_basis.get("stage") == stage
            and receipt_basis.get("dataset") == dataset
            and receipt_basis.get("root_cause_identity") == root_cause_identity
        ):
            actual_affected.add(str(receipt_basis.get("physical_run_id", "")))
    if (
        existing.get("schema_version") != value["schema_version"]
        or existing.get("status") != value["status"]
        or existing.get("identity") != canonical_json_hash(basis)
        or basis.get("stage") != stage
        or basis.get("dataset") != dataset
        or basis.get("root_cause_identity") != root_cause_identity
        or basis.get("fanout_must_stop_for_dataset") is not True
        or basis.get("other_datasets_may_continue") is not True
        or basis.get("completed_runs_preserved") is not True
        or canonical_affected != existing_affected
        or len(canonical_affected) < COMMON_ROOT_CAUSE_PAUSE_THRESHOLD
        or not set(canonical_affected).issubset(actual_affected)
    ):
        raise RuntimeError("existing dataset pause receipt is invalid")


def run_plan_shard(
    *,
    repo_root: str | Path,
    cache_parent: str | Path,
    result_root: str | Path,
    plan_path: str | Path,
    seal_path: str | Path,
    stage: str,
    expected_seal_status: str,
    shard_index: int,
    shard_count: int = 2,
    retry_failed: bool = False,
    enforce_target_gpu: bool = True,
) -> dict[str, Any]:
    if shard_count != 2 or shard_index not in {0, 1}:
        raise ValueError("fast execution uses exactly two independent single-GPU shards")
    expected_stages = (stage, "R1_ALL_SETS_0000_TO_0300") if stage == "SEARCH" else stage
    rows, _ = validate_plan(
        plan_path,
        seal_path,
        expected_status=expected_seal_status,
        expected_stage=expected_stages,
    )
    selected = [row for index, row in enumerate(rows) if index % 2 == shard_index]
    result = Path(result_root).resolve()
    completed: list[str] = []
    failed: list[str] = []
    paused: list[str] = []
    for planned in selected:
        failed_attempt_count, unrecovered_count = _failure_budget_counts(result)
        if failed_attempt_count >= int(RESOURCE_LIMITS["hard_failed_attempts"]):
            raise RuntimeError("FAST_SEARCH_BUDGET_BLOCKED: FAILED_ATTEMPTS")
        if unrecovered_count >= int(RESOURCE_LIMITS["hard_unrecovered_identities"]) and not retry_failed:
            raise RuntimeError("FAST_SEARCH_BUDGET_BLOCKED: UNRECOVERED_IDENTITIES")
        row = dict(planned)
        dataset = str(row["dataset"])
        run_id = str(row["physical_run_id"])
        pause = _dataset_pause(result, stage, dataset)
        if pause.is_file():
            paused.append(run_id)
            continue
        existing = [
            _failure_path(result, stage, dataset, run_id, attempt)
            for attempt in range(MAX_ATTEMPTS_PER_IDENTITY)
            if _failure_path(result, stage, dataset, run_id, attempt).is_file()
        ]
        if retry_failed:
            if len(existing) != 1 or int(read_json(existing[0]).get("basis", {}).get("attempt_index", -1)) != 0:
                continue
            row["attempt_index"] = 1
        elif existing:
            failed.append(run_id)
            continue
        else:
            row["attempt_index"] = 0
        try:
            run_plan_row(
                plan_row=row,
                stage=stage,
                repo_root=repo_root,
                cache_root=Path(cache_parent) / dataset,
                result_root=result,
                source_commit=str(row["source_commit"]),
                source_tree=str(row["source_tree"]),
                enforce_target_gpu=enforce_target_gpu,
            )
            completed.append(run_id)
        except Exception as error:
            attempt = int(row["attempt_index"])
            kind, root = _root_cause(error)
            failure = envelope(
                "metachorus.v53.m6f-m7f.failed-attempt.v1",
                "FAILED_PRESERVED",
                {
                    "stage": stage,
                    "dataset": dataset,
                    "physical_run_id": run_id,
                    "attempt_index": attempt,
                    "failure_kind": kind,
                    "root_cause_identity": root,
                    "scientific_metrics_observed": False,
                    "labels_mounted": False,
                    "prelim_locked_report_access": False,
                },
            )
            write_json_once(_failure_path(result, stage, dataset, run_id, attempt), failure)
            failed.append(run_id)
            receipts = [
                read_json(path)
                for path in (result / "failed_attempts" / dataset / stage.casefold()).glob("*__attempt_*.json")
            ]
            affected = sorted(
                {
                    str(value["basis"]["physical_run_id"])
                    for value in receipts
                    if value.get("basis", {}).get("root_cause_identity") == root
                }
            )
            if len(affected) >= COMMON_ROOT_CAUSE_PAUSE_THRESHOLD:
                _write_dataset_pause_once(
                    pause,
                    stage=stage,
                    dataset=dataset,
                    root_cause_identity=root,
                    affected_physical_run_ids=affected,
                )
    summary = {
        "stage": stage,
        "shard_index": shard_index,
        "planned_count": len(selected),
        "completed_count": len(completed),
        "failed_count": len(failed),
        "completed_physical_run_ids": completed,
        "failed_physical_run_ids": failed,
        "paused_physical_run_ids": paused,
        "labels_mounted": False,
    }
    if failed:
        raise RuntimeError(f"fast shard preserved failures: {failed}")
    if paused:
        raise RuntimeError(f"fast shard stopped identities behind dataset pause: {paused}")
    return summary


def close_canaries(result_root: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    result = Path(result_root).resolve()
    rows, seal = validate_plan(
        result / "FAST_CANARY_RUN_PLAN.csv",
        result / "FAST_CANARY_PLAN_SEAL.json",
        expected_status="SEALED_BEFORE_ENGINEERING_CANARIES",
        expected_stage="CANARY",
    )
    records: list[dict[str, Any]] = []
    for row in rows:
        run_parent = result / "canary_runs" / str(row["physical_run_id"])
        attempts = sorted(run_parent.glob("attempt_*")) if run_parent.is_dir() else []
        markers = [path / "M6F_M7F_CANARY_000020_COMPLETE.json" for path in attempts if (path / "M6F_M7F_CANARY_000020_COMPLETE.json").is_file()]
        if len(markers) != 1:
            raise RuntimeError(f"canary completion is not unique: {row['physical_run_id']}")
        marker = read_json(markers[0])
        attempt_index = int(markers[0].parent.name.removeprefix("attempt_"))
        basis = validate_run_marker(
            marker,
            row,
            stage="CANARY",
            endpoint=20,
            expected_attempt_index=attempt_index,
        )
        audit = basis.get("canary_engineering_audit", {})
        if (
            audit.get("status") != "PASS"
            or audit.get("checkpoint_write_pass") is not True
            or audit.get("checkpoint_read_pass") is not True
            or audit.get("checkpoint_restored_in_fresh_session") is not True
            or audit.get("path_resolution_pass") is not True
            or audit.get("optimizer_step_pass") is not True
        ):
            raise ValueError("fast canary engineering audit failed")
        records.append(
            {
                "dataset": row["dataset"],
                "physical_run_id": row["physical_run_id"],
                "candidate_set_hash": row["candidate_set_hash"],
                "model_seed": row["model_seed"],
                "model_total_seconds": float(basis["model_total_seconds"]),
                "peak_gpu_reserved_bytes": int(basis["peak_gpu_reserved_bytes"]),
                "run_output_bytes": sum(
                    path.stat().st_size
                    for path in markers[0].parent.rglob("*")
                    if path.is_file() and not path.is_symlink()
                ),
                "marker_identity": marker["identity"],
                "status": "PASS",
            }
        )
    if len(records) != 3 or {str(row["dataset"]) for row in records} != set(DATASET_NAMES):
        raise RuntimeError("exactly one canary per dataset is required")
    results = envelope(
        "metachorus.v53.m6f-m7f.canary-results.v1",
        "PASS",
        {
            "plan_seal_identity": seal["identity"],
            "run_count": len(records),
            "records": records,
            "all_checkpoint_roundtrips_pass": True,
            "scientific_metrics_observed": False,
            "labels_mounted": False,
        },
    )
    write_json_once(result / "FAST_CANARY_RESULTS.json", results)
    gate = envelope(
        "metachorus.v53.m6f-m7f.canary-gate.v1",
        "FAST_CANARY_PASS",
        {
            "canary_results_identity": results["identity"],
            "fanout_authorized": True,
            "scientific_metrics_observed": False,
            "labels_mounted": False,
        },
    )
    write_json_once(result / "FAST_CANARY_GATE_DECISION.json", gate)
    return results, gate


def _cache_tree_bytes(root: Path) -> int:
    total = 0
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("fast cache tree contains a symlink")
        if path.is_file():
            total += path.stat().st_size
    return total


def _two_gpu_makespan_seconds(durations: Sequence[float]) -> float:
    """Return a deterministic LPT upper estimate for two run-level GPU slots."""

    loads = [0.0 for _ in range(MAX_GPUS)]
    for duration in sorted((float(value) for value in durations), reverse=True):
        if duration < 0.0 or not math.isfinite(duration):
            raise ValueError("resource duration is invalid")
        slot = min(range(MAX_GPUS), key=lambda index: (loads[index], index))
        loads[slot] += duration
    return max(loads, default=0.0)


def _resource_reasons(
    *,
    gpu_hours: float,
    critical_hours: float,
    storage_bytes: int,
    failed_attempts: int = 0,
    unrecovered_identities: int = 0,
) -> tuple[list[str], list[str]]:
    soft: list[str] = []
    hard: list[str] = []
    checks = (
        ("GPU_HOURS", gpu_hours, "soft_gpu_hours", "hard_gpu_hours"),
        ("CRITICAL_PATH", critical_hours, "soft_critical_path_hours", "hard_critical_path_hours"),
        ("REMOTE_STORAGE", storage_bytes, "soft_remote_storage_bytes", "hard_remote_storage_bytes"),
    )
    for name, value, soft_key, hard_key in checks:
        if value > RESOURCE_LIMITS[hard_key]:
            hard.append(name)
        elif value > RESOURCE_LIMITS[soft_key]:
            soft.append(name)
    if int(failed_attempts) >= int(RESOURCE_LIMITS["hard_failed_attempts"]):
        hard.append("FAILED_ATTEMPTS")
    if int(unrecovered_identities) >= int(RESOURCE_LIMITS["hard_unrecovered_identities"]):
        hard.append("UNRECOVERED_IDENTITIES")
    return soft, hard


def _failure_budget_counts(result: Path) -> tuple[int, int]:
    receipts = []
    root = result / "failed_attempts"
    if root.exists():
        for path in sorted(root.rglob("*__attempt_*.json")):
            if path.is_symlink() or not path.is_file():
                raise ValueError("fast failure registry contains a non-regular member")
            value = read_json(path)
            basis = value.get("basis", {})
            if (
                value.get("schema_version") != "metachorus.v53.m6f-m7f.failed-attempt.v1"
                or value.get("status") != "FAILED_PRESERVED"
                or value.get("identity") != canonical_json_hash(basis)
            ):
                raise ValueError("fast failure receipt identity is invalid")
            receipts.append(value)
    failed_ids = {str(value["basis"]["physical_run_id"]) for value in receipts}
    completed_ids = {
        str(read_json(path).get("basis", {}).get("physical_run_id", ""))
        for base in ("canary_runs", "search_runs", "preliminary_runs")
        for path in (result / base).glob("*/attempt_*/M6F_M7F_*_COMPLETE.json")
        if path.is_file() and not path.is_symlink()
    }
    return len(receipts), len(failed_ids - completed_ids)


def _dataset_campaign_forecast(
    *,
    dataset: str,
    canary_record: Mapping[str, Any],
    r1_run_count: int,
    observed_cache_bytes: int,
    observed_failed_attempts: int = 0,
    observed_unrecovered_identities: int = 0,
) -> dict[str, Any]:
    """Forecast one dataset so a hard resource gate cannot stop its peers."""

    if dataset not in DATASET_NAMES:
        raise ValueError("resource forecast dataset is invalid")
    if r1_run_count <= 0 or r1_run_count % len(SEARCH_MODEL_SEEDS):
        raise ValueError("R1 matrix is not an exact candidate-set x three-seed matrix")
    candidate_count = r1_run_count // len(SEARCH_MODEL_SEEDS)
    high_recall_upper = min(
        candidate_count,
        min(80, math.ceil(0.40 * candidate_count))
        + min(40, math.ceil(0.20 * candidate_count))
        + 1,
    )
    r3_shortlist_upper = high_recall_upper
    per_update = float(canary_record["model_total_seconds"]) / float(CANARY_ENDPOINT)
    if per_update <= 0.0 or not math.isfinite(per_update):
        raise ValueError("canary seconds-per-update is invalid")
    canary_bytes = int(canary_record.get("run_output_bytes", 0))
    if canary_bytes <= 0 or int(observed_cache_bytes) <= 0:
        raise ValueError("dataset resource observations are incomplete")

    continuation_runs = high_recall_upper * len(SEARCH_MODEL_SEEDS)
    r4_runs = r3_shortlist_upper * len(SEARCH_MODEL_SEEDS)
    stage_durations = {
        "CANARY": [float(canary_record["model_total_seconds"])],
        "R1_0000_TO_0300": [SEARCH_ENDPOINTS[0] * per_update] * r1_run_count,
        "R2_0300_TO_1000": [
            (SEARCH_ENDPOINTS[1] - SEARCH_ENDPOINTS[0]) * per_update
        ] * continuation_runs,
        "R3_1000_TO_2000": [
            (SEARCH_ENDPOINTS[2] - SEARCH_ENDPOINTS[1]) * per_update
        ] * continuation_runs,
        "R4_2000_TO_3000": [
            (SEARCH_ENDPOINTS[3] - SEARCH_ENDPOINTS[2]) * per_update
        ] * r4_runs,
        "PRELIMINARY_0000_TO_3000": [
            PRELIM_ENDPOINT * per_update
        ] * (2 * len(EVAL_MODEL_SEEDS)),
    }
    stage_gpu_seconds = {name: sum(values) for name, values in stage_durations.items()}
    stage_critical_seconds = {
        name: _two_gpu_makespan_seconds(values)
        for name, values in stage_durations.items()
    }
    estimated_output_bytes = canary_bytes * (
        1
        + r1_run_count
        + high_recall_upper * len(SEARCH_MODEL_SEEDS) * 2
        + r3_shortlist_upper * len(SEARCH_MODEL_SEEDS)
        + 2 * len(EVAL_MODEL_SEEDS) * 3
    )
    storage_bytes = int(observed_cache_bytes) + estimated_output_bytes
    gpu_hours = sum(stage_gpu_seconds.values()) / 3600.0
    critical_hours = sum(stage_critical_seconds.values()) / 3600.0
    soft, hard = _resource_reasons(
        gpu_hours=gpu_hours,
        critical_hours=critical_hours,
        storage_bytes=storage_bytes,
        failed_attempts=observed_failed_attempts,
        unrecovered_identities=observed_unrecovered_identities,
    )
    return {
        "dataset": dataset,
        "seconds_per_update": per_update,
        "candidate_set_count": candidate_count,
        "r1_run_count": r1_run_count,
        "high_recall_set_upper_bound": high_recall_upper,
        "r3_shortlist_set_upper_bound": r3_shortlist_upper,
        "preliminary_physical_run_upper_bound": 2 * len(EVAL_MODEL_SEEDS),
        "stage_gpu_seconds": stage_gpu_seconds,
        "stage_critical_path_seconds": stage_critical_seconds,
        "estimated_full_campaign_gpu_hours": gpu_hours,
        "estimated_full_campaign_critical_path_hours": critical_hours,
        "observed_cache_bytes": int(observed_cache_bytes),
        "observed_canary_output_bytes": canary_bytes,
        "estimated_full_campaign_output_bytes": estimated_output_bytes,
        "estimated_full_campaign_remote_storage_bytes": storage_bytes,
        "soft_warning_reasons": soft,
        "hard_limit_reasons": hard,
        "status": "FAST_SEARCH_BUDGET_BLOCKED" if hard else "AUTHORIZED_WITH_SOFT_RESOURCE_WARNING" if soft else "AUTHORIZED",
    }


def _full_campaign_forecast(
    *,
    canary_records: Sequence[Mapping[str, Any]],
    r1_run_count_by_dataset: Mapping[str, int],
    observed_cache_bytes: int,
    observed_cache_bytes_by_dataset: Mapping[str, int] | None = None,
    observed_failed_attempts: int = 0,
    observed_unrecovered_identities: int = 0,
) -> dict[str, Any]:
    """Estimate the training budget for search and preliminary evaluation."""

    records = {str(row["dataset"]): row for row in canary_records}
    if set(records) != set(DATASET_NAMES):
        raise ValueError("resource forecast requires one canary record per dataset")
    seconds_per_update: dict[str, float] = {}
    candidate_counts: dict[str, int] = {}
    high_recall_upper: dict[str, int] = {}
    r3_shortlist_upper: dict[str, int] = {}
    for dataset in DATASET_NAMES:
        run_count = int(r1_run_count_by_dataset[dataset])
        if run_count <= 0 or run_count % len(SEARCH_MODEL_SEEDS):
            raise ValueError("R1 matrix is not an exact candidate-set x three-seed matrix")
        candidate_count = run_count // len(SEARCH_MODEL_SEEDS)
        candidate_counts[dataset] = candidate_count
        control_count = 1  
        high_recall_upper[dataset] = min(
            candidate_count,
            min(80, math.ceil(0.40 * candidate_count))
            + min(40, math.ceil(0.20 * candidate_count))
            + control_count,
        )
        
        
        r3_shortlist_upper[dataset] = high_recall_upper[dataset]
        value = float(records[dataset]["model_total_seconds"]) / float(CANARY_ENDPOINT)
        if value <= 0.0 or not math.isfinite(value):
            raise ValueError("canary seconds-per-update is invalid")
        seconds_per_update[dataset] = value

    stage_durations: dict[str, list[float]] = {
        "CANARY": [float(records[dataset]["model_total_seconds"]) for dataset in DATASET_NAMES],
        "R1_0000_TO_0300": [],
        "R2_0300_TO_1000": [],
        "R3_1000_TO_2000": [],
        "R4_2000_TO_3000": [],
        "PRELIMINARY_0000_TO_3000": [],
    }
    for dataset in DATASET_NAMES:
        per_update = seconds_per_update[dataset]
        stage_durations["R1_0000_TO_0300"].extend(
            [SEARCH_ENDPOINTS[0] * per_update] * int(r1_run_count_by_dataset[dataset])
        )
        continuation_runs = high_recall_upper[dataset] * len(SEARCH_MODEL_SEEDS)
        stage_durations["R2_0300_TO_1000"].extend(
            [(SEARCH_ENDPOINTS[1] - SEARCH_ENDPOINTS[0]) * per_update] * continuation_runs
        )
        stage_durations["R3_1000_TO_2000"].extend(
            [(SEARCH_ENDPOINTS[2] - SEARCH_ENDPOINTS[1]) * per_update] * continuation_runs
        )
        r4_runs = r3_shortlist_upper[dataset] * len(SEARCH_MODEL_SEEDS)
        stage_durations["R4_2000_TO_3000"].extend(
            [(SEARCH_ENDPOINTS[3] - SEARCH_ENDPOINTS[2]) * per_update] * r4_runs
        )
        stage_durations["PRELIMINARY_0000_TO_3000"].extend(
            [PRELIM_ENDPOINT * per_update] * (2 * len(EVAL_MODEL_SEEDS))
        )
    stage_gpu_seconds = {name: sum(values) for name, values in stage_durations.items()}
    stage_critical_seconds = {
        name: _two_gpu_makespan_seconds(values)
        for name, values in stage_durations.items()
    }
    total_seconds = sum(stage_gpu_seconds.values())
    critical_seconds = sum(stage_critical_seconds.values())

    canary_bytes = {
        dataset: int(records[dataset].get("run_output_bytes", 0))
        for dataset in DATASET_NAMES
    }
    if any(value <= 0 for value in canary_bytes.values()):
        raise ValueError("canary output-size observation is missing")
    estimated_output_bytes = sum(canary_bytes.values())
    for dataset in DATASET_NAMES:
        
        
        
        estimated_output_bytes += canary_bytes[dataset] * (
            int(r1_run_count_by_dataset[dataset])
            + high_recall_upper[dataset] * len(SEARCH_MODEL_SEEDS) * 2
            + r3_shortlist_upper[dataset] * len(SEARCH_MODEL_SEEDS)
            + 2 * len(EVAL_MODEL_SEEDS) * 3
        )
    storage_bytes = int(observed_cache_bytes) + estimated_output_bytes
    gpu_hours = total_seconds / 3600.0
    critical_hours = critical_seconds / 3600.0
    soft, hard = _resource_reasons(
        gpu_hours=gpu_hours,
        critical_hours=critical_hours,
        storage_bytes=storage_bytes,
        failed_attempts=observed_failed_attempts,
        unrecovered_identities=observed_unrecovered_identities,
    )
    per_dataset_forecasts: dict[str, Any] = {}
    if observed_cache_bytes_by_dataset is not None:
        if set(observed_cache_bytes_by_dataset) != set(DATASET_NAMES):
            raise ValueError("dataset cache byte observations are incomplete")
        per_dataset_forecasts = {
            dataset: _dataset_campaign_forecast(
                dataset=dataset,
                canary_record=records[dataset],
                r1_run_count=int(r1_run_count_by_dataset[dataset]),
                observed_cache_bytes=int(observed_cache_bytes_by_dataset[dataset]),
                observed_failed_attempts=observed_failed_attempts,
                observed_unrecovered_identities=observed_unrecovered_identities,
            )
            for dataset in DATASET_NAMES
        }
    return {
        "seconds_per_update_by_dataset": seconds_per_update,
        "candidate_set_count_by_dataset": candidate_counts,
        "r1_run_count_by_dataset": {key: int(value) for key, value in r1_run_count_by_dataset.items()},
        "high_recall_set_upper_bound_by_dataset": high_recall_upper,
        "r3_shortlist_set_upper_bound_by_dataset": r3_shortlist_upper,
        "preliminary_physical_run_upper_bound_by_dataset": {
            dataset: 2 * len(EVAL_MODEL_SEEDS) for dataset in DATASET_NAMES
        },
        "stage_gpu_seconds": stage_gpu_seconds,
        "stage_critical_path_seconds": stage_critical_seconds,
        "estimated_full_campaign_gpu_hours": gpu_hours,
        "estimated_full_campaign_critical_path_hours": critical_hours,
        "observed_cache_bytes": int(observed_cache_bytes),
        "observed_canary_output_bytes_by_dataset": canary_bytes,
        "estimated_full_campaign_output_bytes": estimated_output_bytes,
        "estimated_full_campaign_remote_storage_bytes": storage_bytes,
        "observed_failed_attempts": int(observed_failed_attempts),
        "observed_unrecovered_identities": int(observed_unrecovered_identities),
        "soft_warning_reasons": soft,
        "hard_limit_reasons": hard,
        "per_dataset_forecasts": per_dataset_forecasts,
    }


def authorize_r1_budget(
    *,
    result_root: str | Path,
    cache_parent: str | Path,
    source_commit: str,
    source_tree: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Use measured canaries to gate R1 against the full campaign forecast."""

    result = Path(result_root).resolve()
    canary_results, _ = close_canaries(result)
    canary_records = canary_results["basis"]["records"]
    draft, draft_seal = validate_plan(
        result / "FAST_SEARCH_EXPERIMENT_MATRIX.csv",
        result / "FAST_SEARCH_EXPERIMENT_MATRIX_SEAL.json",
        expected_status="SEALED_BEFORE_SEARCH_GPU_TRAINING",
        expected_stage="R1_ALL_SETS_0000_TO_0300",
    )
    dataset_rows = {dataset: sum(row["dataset"] == dataset for row in draft) for dataset in DATASET_NAMES}
    cache_root = Path(cache_parent).resolve(strict=True)
    cache_bytes_by_dataset = {
        dataset: _cache_tree_bytes(cache_root / dataset)
        for dataset in DATASET_NAMES
    }
    cache_bytes = sum(cache_bytes_by_dataset.values())
    failed_attempts, unrecovered = _failure_budget_counts(result)
    forecast = _full_campaign_forecast(
        canary_records=canary_records,
        r1_run_count_by_dataset=dataset_rows,
        observed_cache_bytes=cache_bytes,
        observed_cache_bytes_by_dataset=cache_bytes_by_dataset,
        observed_failed_attempts=failed_attempts,
        observed_unrecovered_identities=unrecovered,
    )
    dataset_forecasts = forecast["per_dataset_forecasts"]
    blocked_datasets = [
        dataset
        for dataset in DATASET_NAMES
        if dataset_forecasts[dataset]["hard_limit_reasons"]
    ]
    active_datasets = [dataset for dataset in DATASET_NAMES if dataset not in blocked_datasets]
    status_records = [
        {
            "dataset": dataset,
            "status": dataset_forecasts[dataset]["status"],
            "hard_limit_reasons": list(dataset_forecasts[dataset]["hard_limit_reasons"]),
            "soft_warning_reasons": list(dataset_forecasts[dataset]["soft_warning_reasons"]),
            "estimated_gpu_hours": float(dataset_forecasts[dataset]["estimated_full_campaign_gpu_hours"]),
            "estimated_critical_path_hours": float(dataset_forecasts[dataset]["estimated_full_campaign_critical_path_hours"]),
            "estimated_remote_storage_bytes": int(dataset_forecasts[dataset]["estimated_full_campaign_remote_storage_bytes"]),
            "candidate_set_count": int(dataset_forecasts[dataset]["candidate_set_count"]),
            "r1_run_count": int(dataset_forecasts[dataset]["r1_run_count"]),
        }
        for dataset in DATASET_NAMES
    ]
    dataset_status = envelope(
        "metachorus.v53.m6f-m7f.dataset-execution-status.v1",
        "PARTIAL" if blocked_datasets and active_datasets else "BLOCKED" if blocked_datasets else "PASS",
        {
            "records": status_records,
            "active_datasets": active_datasets,
            "blocked_datasets": blocked_datasets,
            "blocked_datasets_preserve_full_frozen_candidate_banks": True,
            "other_datasets_may_continue": True,
            "labels_mounted": False,
        },
    )
    write_json_once(result / "FAST_DATASET_EXECUTION_STATUS.json", dataset_status)
    basis = {
        "canary_results_identity": canary_results["identity"],
        "draft_plan_seal_identity": draft_seal["identity"],
        "r1_total_run_count": len(draft),
        **forecast,
        "resource_limits": RESOURCE_LIMITS,
        "active_datasets": active_datasets,
        "blocked_datasets": blocked_datasets,
        "dataset_execution_status_identity": dataset_status["identity"],
        "formal_r1_authorized": bool(active_datasets),
        "all_datasets_authorized": not blocked_datasets,
    }
    budget = envelope(
        "metachorus.v53.m6f-m7f.r1-measured-budget-gate.v1",
        (
            "FAST_SEARCH_BUDGET_BLOCKED"
            if not active_datasets
            else "PARTIAL_PASS_WITH_BLOCKED_DATASETS"
            if blocked_datasets
            else "PASS_WITH_SOFT_RESOURCE_WARNING"
            if forecast["soft_warning_reasons"]
            else "PASS"
        ),
        basis,
    )
    write_json_once(result / "FAST_R1_MEASURED_BUDGET_GATE.json", budget)
    if not active_datasets:
        raise RuntimeError("FAST_SEARCH_BUDGET_BLOCKED: ALL_DATASETS")
    return authorize_search_after_canary(
        result,
        source_commit=source_commit,
        source_tree=source_tree,
        active_datasets=active_datasets,
        budget_gate=budget,
    )


def _representation(result: Path, marker: Mapping[str, Any]) -> np.ndarray:
    relative = str(marker["basis"]["representation_relative_path"])
    source = result / Path(*relative.split("/"))
    if source.is_symlink() or not source.is_file() or sha256_file(source) != marker["basis"]["representation_sha256"]:
        raise ValueError("fast representation identity mismatch")
    value = np.load(source, allow_pickle=False)
    if value.ndim != 2 or not np.isfinite(value).all():
        raise ValueError("fast representation is invalid")
    return value


def _dataset_execution_status(result: Path) -> tuple[dict[str, Any], tuple[str, ...], tuple[str, ...]]:
    value = read_json(result / "FAST_DATASET_EXECUTION_STATUS.json")
    basis = value.get("basis", {})
    active = tuple(str(name) for name in basis.get("active_datasets", []))
    blocked = tuple(str(name) for name in basis.get("blocked_datasets", []))
    records = basis.get("records", [])
    if (
        value.get("schema_version") != "metachorus.v53.m6f-m7f.dataset-execution-status.v1"
        or value.get("status") not in {"PASS", "PARTIAL", "BLOCKED"}
        or value.get("identity") != canonical_json_hash(basis)
        or set(active) | set(blocked) != set(DATASET_NAMES)
        or set(active) & set(blocked)
        or len(active) != len(set(active))
        or len(blocked) != len(set(blocked))
        or len(records) != len(DATASET_NAMES)
        or {str(row.get("dataset")) for row in records} != set(DATASET_NAMES)
        or any(name not in DATASET_NAMES for name in (*active, *blocked))
        or any(
            str(row.get("status")) == "FAST_SEARCH_BUDGET_BLOCKED"
            for row in records
            if str(row.get("dataset")) in active
        )
        or any(
            str(row.get("status")) != "FAST_SEARCH_BUDGET_BLOCKED"
            for row in records
            if str(row.get("dataset")) in blocked
        )
    ):
        raise ValueError("dataset execution status is invalid")
    return value, active, blocked


def evaluate_search_endpoint(
    *,
    result_root: str | Path,
    role_manifest_path: str | Path,
    role_seal_path: str | Path,
    private_mount_audit_path: str | Path,
    private_search_mount_root: str | Path,
    plan_rows: Sequence[Mapping[str, Any]],
    endpoint: int,
) -> list[dict[str, Any]]:
    if endpoint not in SEARCH_ENDPOINTS:
        raise ValueError("search evaluation endpoint is outside the frozen schedule")
    result = Path(result_root).resolve()
    active_datasets = tuple(name for name in DATASET_NAMES if any(str(row["dataset"]) == name for row in plan_rows))
    if not active_datasets:
        raise ValueError("search evaluation plan has no active dataset")
    mount_by_dataset = {
        dataset: open_search_mount(
            role_manifest_path=role_manifest_path,
            role_seal_path=role_seal_path,
            private_mount_audit_path=private_mount_audit_path,
            private_search_mount_root=private_search_mount_root,
            dataset=dataset,
        )
        for dataset in active_datasets
    }
    output: list[dict[str, Any]] = []
    for planned in plan_rows:
        row = _canonical_plan_row(planned)
        run_parent = result / "search_runs" / str(row["physical_run_id"])
        markers = [path / f"M6F_M7F_SEARCH_{endpoint:06d}_COMPLETE.json" for path in sorted(run_parent.glob("attempt_*")) if (path / f"M6F_M7F_SEARCH_{endpoint:06d}_COMPLETE.json").is_file()]
        if len(markers) != 1:
            raise RuntimeError(f"search endpoint completion is not unique: {row['physical_run_id']}")
        marker = read_json(markers[0])
        attempt_index = int(markers[0].parent.name.removeprefix("attempt_"))
        validate_run_marker(
            marker,
            row,
            stage="SEARCH",
            endpoint=endpoint,
            expected_attempt_index=attempt_index,
        )
        metrics = evaluate_search_representation(
            _representation(result, marker),
            dataset=str(row["dataset"]),
            search_mount=mount_by_dataset[str(row["dataset"])],
        )
        output.append(
            {
                "physical_run_id": row["physical_run_id"],
                "dataset": row["dataset"],
                "candidate_set_id": row["candidate_set_id"],
                "candidate_set_hash": row["candidate_set_hash"],
                "candidate_set_ordinal": row["candidate_set_ordinal"],
                "model_seed": row["model_seed"],
                "endpoint_update": endpoint,
                "marker_identity": marker["identity"],
                "representation_sha256": marker["basis"]["representation_sha256"],
                "model_total_seconds": float(marker["basis"]["model_total_seconds"]),
                "peak_gpu_reserved_bytes": int(marker["basis"].get("peak_gpu_reserved_bytes", 0)),
                **metrics["metrics"],
                "assignment_stability_mean": float(metrics["assignment_stability_mean"]),
                "prelim_locked_report_access": False,
                "status": "PASS",
            }
        )
    return output


def build_continuation_plan(
    *,
    prior_rows: Sequence[Mapping[str, Any]],
    selected_candidate_hashes: Mapping[str, set[str]],
    result_root: str | Path,
    next_endpoint: int,
    prior_seal_identity: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if next_endpoint not in SEARCH_ENDPOINTS[1:]:
        raise ValueError("invalid fast continuation endpoint")
    prior_endpoint = SEARCH_ENDPOINTS[SEARCH_ENDPOINTS.index(next_endpoint) - 1]
    result = Path(result_root).resolve()
    dataset_status, expected_active_datasets, blocked_datasets = _dataset_execution_status(result)
    normalized_rows = [_canonical_plan_row(row) for row in prior_rows]
    active_datasets = tuple(
        dataset
        for dataset in DATASET_NAMES
        if any(str(row.get("dataset")) == dataset for row in normalized_rows)
    )
    if active_datasets != expected_active_datasets:
        raise ValueError("continuation source rows do not match the authorized dataset subset")
    plan_identities = {str(row.get("plan_seal_identity", "")) for row in normalized_rows}
    if plan_identities != {str(prior_seal_identity)} or len(str(prior_seal_identity)) != 64:
        raise ValueError("continuation plan is not bound to one exact prior plan seal")
    if not active_datasets or set(selected_candidate_hashes) != set(active_datasets) or any(
        not set(selected_candidate_hashes[dataset]) for dataset in active_datasets
    ):
        raise ValueError("continuation selection must be non-empty for every dataset")
    for row in normalized_rows:
        if (
            row.get("stage") != "SEARCH"
            or int(row.get("endpoint_update", -1)) != prior_endpoint
            or int(row.get("model_seed", -1)) not in SEARCH_MODEL_SEEDS
        ):
            raise ValueError("continuation source rows are not the exact prior SEARCH boundary")
    for dataset in active_datasets:
        available = {
            str(row["candidate_set_hash"])
            for row in normalized_rows
            if str(row["dataset"]) == dataset
        }
        selected_hashes = set(map(str, selected_candidate_hashes[dataset]))
        if not selected_hashes.issubset(available):
            raise ValueError("continuation selection references an unknown prior candidate")
        for digest in selected_hashes:
            cohort = [
                row
                for row in normalized_rows
                if str(row["dataset"]) == dataset and str(row["candidate_set_hash"]) == digest
            ]
            if len(cohort) != 3 or {int(row["model_seed"]) for row in cohort} != set(SEARCH_MODEL_SEEDS):
                raise ValueError("continuation selection lacks the exact three search seeds")
    rows: list[dict[str, Any]] = []
    for row in normalized_rows:
        if str(row["candidate_set_hash"]) not in selected_candidate_hashes[str(row["dataset"])]:
            continue
        run_parent = result / "search_runs" / str(row["physical_run_id"])
        marker_name = f"M6F_M7F_SEARCH_{prior_endpoint:06d}_COMPLETE.json"
        markers = [
            attempt / marker_name
            for attempt in sorted(run_parent.glob("attempt_*"))
            if (attempt / marker_name).is_file()
        ]
        if len(markers) != 1:
            raise RuntimeError(f"continuation source boundary is not unique: {row['physical_run_id']}")
        attempt_index = int(markers[0].parent.name.removeprefix("attempt_"))
        marker = read_json(markers[0])
        basis = validate_run_marker(
            marker,
            row,
            stage="SEARCH",
            endpoint=prior_endpoint,
            expected_attempt_index=attempt_index,
        )
        run_root = markers[0].parent
        resolved_config_path = run_root / "RESOLVED_CONFIG.json"
        if not resolved_config_path.is_file() or resolved_config_path.is_symlink():
            raise RuntimeError("continuation source lacks its resolved training config")
        resolved_config = read_json(resolved_config_path)
        run_config_identity = canonical_json_hash(resolved_config)
        run_config_plan_seal_identity = str(resolved_config.get("plan_seal_hash", ""))
        if (
            len(run_config_plan_seal_identity) != 64
            or basis.get("run_identity") != run_config_identity
            or (
                "run_config_plan_seal_identity" in row
                and str(row["run_config_plan_seal_identity"]) != run_config_plan_seal_identity
            )
            or (
                "run_config_identity" in row
                and str(row["run_config_identity"]) != run_config_identity
            )
        ):
            raise ValueError("continuation source resolved config identity is invalid")
        checkpoint = result / Path(*str(basis["checkpoint_relative_path"]).split("/"))
        expected_checkpoint = run_root / "checkpoints" / f"update_{prior_endpoint:06d}.pt"
        if (
            checkpoint.is_symlink()
            or not checkpoint.is_file()
            or checkpoint.resolve(strict=True) != expected_checkpoint.resolve(strict=True)
            or sha256_file(checkpoint) != basis.get("checkpoint_sha256")
        ):
            raise ValueError("continuation source checkpoint does not match its completion marker")
        rows.append(
            {
                **row,
                "stage": "SEARCH",
                "attempt_index": attempt_index,
                "continuation_from_update": prior_endpoint,
                "endpoint_update": next_endpoint,
                "execution_action": f"STRICT_CONTINUATION_{prior_endpoint:04d}_TO_{next_endpoint:04d}",
                "resume_checkpoint_path": str(checkpoint.resolve()),
                "resume_checkpoint_sha256": sha256_file(checkpoint),
                "run_config_plan_seal_identity": run_config_plan_seal_identity,
                "run_config_identity": run_config_identity,
                "status": "AUTHORIZED",
                "plan_seal_identity": "",
            }
        )
    unsealed = [{key: value for key, value in row.items() if key != "plan_seal_identity"} for row in rows]
    seal = envelope(
        "metachorus.v53.m6f-m7f.search-continuation-plan-seal.v1",
        "SEALED_BEFORE_SEARCH_CONTINUATION",
        {
            "row_count": len(rows),
            "rows_identity": canonical_json_hash(unsealed),
            "prior_plan_seal_identity": prior_seal_identity,
            "continuation_from_update": prior_endpoint,
            "endpoint_update": next_endpoint,
            "active_datasets": list(active_datasets),
            "blocked_datasets": list(blocked_datasets),
            "dataset_execution_status_identity": dataset_status["identity"],
            "labels_mounted": False,
            "prelim_locked_report_access": False,
        },
    )
    return [{**row, "plan_seal_identity": seal["identity"]} for row in rows], seal


def seal_final_search_selection(
    *,
    result_root: str | Path,
    cache_resolutions: Mapping[str, Mapping[str, Any]],
    final_metric_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    result = Path(result_root).resolve()
    dataset_status, active_datasets, blocked_datasets = _dataset_execution_status(result)
    if {str(row.get("dataset")) for row in final_metric_rows} != set(active_datasets):
        raise ValueError("final search metrics do not match the authorized dataset subset")
    selected_rows: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    for dataset in active_datasets:
        cache = cache_resolutions[dataset]
        candidates = {
            str(row["candidate_set_hash"]): dict(row)
            for row in cache["all_sets"]["sets"]
        }
        dataset_metrics = [row for row in final_metric_rows if str(row["dataset"]) == dataset]
        if (
            not dataset_metrics
            or any(row.get("status") != "PASS" for row in dataset_metrics)
            or any(int(row.get("endpoint_update", -1)) != SEARCH_ENDPOINTS[-1] for row in dataset_metrics)
            or any(int(row.get("model_seed", -1)) not in SEARCH_MODEL_SEEDS for row in dataset_metrics)
        ):
            raise ValueError("final search metrics are not sealed update-3000 PASS rows")
        shortlist_hashes = {str(row["candidate_set_hash"]) for row in dataset_metrics}
        if not shortlist_hashes.issubset(candidates):
            raise ValueError("final search metrics reference an unknown frozen candidate")
        for digest in shortlist_hashes:
            rows_for_candidate = [row for row in dataset_metrics if str(row["candidate_set_hash"]) == digest]
            if (
                len(rows_for_candidate) != len(SEARCH_MODEL_SEEDS)
                or {int(row["model_seed"]) for row in rows_for_candidate} != set(SEARCH_MODEL_SEEDS)
            ):
                raise ValueError("final search candidate lacks the exact three search seeds")
            candidate = candidates[digest]
            expected_run_ids = {
                physical_run_id(dataset, int(candidate["candidate_set_ordinal"]), seed_ordinal)
                for seed_ordinal in range(len(SEARCH_MODEL_SEEDS))
            }
            observed_run_ids = {str(row.get("physical_run_id", "")) for row in rows_for_candidate}
            if (
                observed_run_ids != expected_run_ids
                or len({str(row.get("marker_identity", "")) for row in rows_for_candidate}) != 3
                or any(len(str(row.get("marker_identity", ""))) != 64 for row in rows_for_candidate)
                or any(
                    str(row.get("candidate_set_id")) != str(candidate["candidate_set_id"])
                    or int(row.get("candidate_set_ordinal", -1)) != int(candidate["candidate_set_ordinal"])
                    for row in rows_for_candidate
                )
            ):
                raise ValueError("final search metrics are not uniquely bound to frozen physical runs")
        decision = close_search_selection(
            dataset=dataset,
            metric_rows=dataset_metrics,
            candidate_rows=[candidates[digest] for digest in sorted(shortlist_hashes)],
            manual_set=cache["manual"],
            current_automatic=cache["current_automatic"],
        )
        selected = dict(decision["basis"]["selected_set"])
        selected_rows.append(
            {
                "dataset": dataset,
                "status": decision["status"],
                "candidate_set_id": selected["candidate_set_id"],
                "candidate_set_hash": selected["candidate_set_hash"],
                "candidate_set_ordinal": selected["candidate_set_ordinal"],
                "path_ids": list(selected["path_ids"]),
                "path_library_identity": cache["path_library_identity"],
                "all_sets_identity": cache["all_sets_identity"],
                "selection_decision_identity": decision["identity"],
                "no_second_choice": True,
                "algorithm_retuning": False,
            }
        )
        decisions.append(decision)
    registry_basis = {
        "records": selected_rows,
        "record_count": len(selected_rows),
        "datasets": list(active_datasets),
        "blocked_datasets": list(blocked_datasets),
        "dataset_execution_status_identity": dataset_status["identity"],
        "single_complete_set_per_active_dataset": True,
        "no_second_choice": True,
        "algorithm_retuning": False,
    }
    registry = envelope(
        "metachorus.v53.m6f-m7f.final-frozen-set-registry.v1",
        "PASS",
        registry_basis,
    )
    write_json_once(result / "FAST_FINAL_FROZEN_SET_REGISTRY.json", registry)
    write_csv_once(result / "FAST_FINAL_FROZEN_SET_REGISTRY.csv", selected_rows, fieldnames=list(selected_rows[0]))
    write_json_once(
        result / "FAST_SEARCH_SELECTED_SET.json",
        envelope(
            "metachorus.v53.m6f-m7f.search-selected-set.v1",
            "PASS",
            {"decisions": decisions, "records": selected_rows},
        ),
    )
    write_csv_once(result / "FAST_SEARCH_SELECTED_SET.csv", selected_rows, fieldnames=list(selected_rows[0]))
    decision_lines = ["# Fast search decision", ""]
    for row in selected_rows:
        decision_lines.append(
            f"- {row['dataset']}: {row['status']} / {row['candidate_set_id']} / {row['candidate_set_hash']}"
        )
    decision_lines.extend(["", "No second choice. No algorithm retuning.", ""])
    write_text_once(result / "FAST_SEARCH_DECISION.md", "\n".join(decision_lines))
    seal = envelope(
        "metachorus.v53.m6f-m7f.selection-seal.v1",
        "SEALED_AFTER_SEARCH_BEFORE_PRELIMINARY_EVALUATION",
        {
            "final_registry_identity": registry["identity"],
            "decision_identities": [row["identity"] for row in decisions],
            "search_checkpoint_reuse_for_preliminary": False,
            "evaluation_seeds_started": False,
            "active_datasets": list(active_datasets),
            "blocked_datasets": list(blocked_datasets),
            "dataset_execution_status_identity": dataset_status["identity"],
            "labels_mounted_during_training": False,
            "further_search_or_second_choice": False,
        },
    )
    write_json_once(result / "FAST_SELECTION_SEAL.json", seal)
    return selected_rows, seal


def publish_preliminary_plan(
    *,
    result_root: str | Path,
    cache_resolutions: Mapping[str, Mapping[str, Any]],
    source_commit: str,
    source_tree: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    result = Path(result_root).resolve()
    selection_seal = read_json(result / "FAST_SELECTION_SEAL.json")
    registry = read_json(result / "FAST_FINAL_FROZEN_SET_REGISTRY.json")
    if (
        selection_seal.get("status") != "SEALED_AFTER_SEARCH_BEFORE_PRELIMINARY_EVALUATION"
        or selection_seal.get("identity") != canonical_json_hash(selection_seal.get("basis", {}))
        or registry.get("identity") != selection_seal.get("basis", {}).get("final_registry_identity")
        or selection_seal.get("basis", {}).get("evaluation_seeds_started") is not False
    ):
        raise ValueError("selection is not sealed before preliminary planning")
    dataset_status, active_datasets, blocked_datasets = _dataset_execution_status(result)
    if (
        selection_seal.get("basis", {}).get("dataset_execution_status_identity") != dataset_status["identity"]
        or tuple(selection_seal.get("basis", {}).get("active_datasets", [])) != active_datasets
        or tuple(selection_seal.get("basis", {}).get("blocked_datasets", [])) != blocked_datasets
    ):
        raise ValueError("selection seal is not bound to the dataset budget status")
    by_dataset = {str(row["dataset"]): row for row in registry["basis"]["records"]}
    selected: dict[str, dict[str, Any]] = {}
    for dataset in active_datasets:
        cache = cache_resolutions[dataset]
        selected[dataset] = {
            "selected_set": by_dataset[dataset],
            "manual_set": cache["manual"],
            "current_automatic": cache["current_automatic"],
        }
    status_by_dataset = {
        str(row["dataset"]): dict(row)
        for row in dataset_status["basis"]["records"]
    }
    logical, physical, matrix_seal = build_preliminary_matrix(
        selected,
        dataset_statuses=status_by_dataset,
    )
    physical_rows = [
        {
            **row,
            "stage": "PRELIMINARY",
            "cache_identity": cache_resolutions[str(row["dataset"])]["cache_identity"],
            "path_library_identity": cache_resolutions[str(row["dataset"])]["path_library_identity"],
            "source_commit": str(source_commit),
            "source_tree": str(source_tree),
            "attempt_index": 0,
            "plan_seal_identity": "",
        }
        for row in physical
    ]
    unsealed = [{key: value for key, value in row.items() if key != "plan_seal_identity"} for row in physical_rows]
    plan_seal = envelope(
        "metachorus.v53.m6f-m7f.preliminary-run-plan-seal.v1",
        "SEALED_BEFORE_PRELIMINARY_GPU_TRAINING",
        {
            "row_count": len(physical_rows),
            "rows_identity": canonical_json_hash(unsealed),
            "logical_matrix_seal_identity": matrix_seal["identity"],
            "selection_seal_identity": selection_seal["identity"],
            "evaluation_model_seeds": list(EVAL_MODEL_SEEDS),
            "active_datasets": list(active_datasets),
            "blocked_datasets": list(blocked_datasets),
            "dataset_execution_status_identity": dataset_status["identity"],
            "fresh_from_zero": True,
            "search_state_reuse": False,
            "labels_mounted": False,
        },
    )
    sealed_physical = [{**row, "plan_seal_identity": plan_seal["identity"]} for row in physical_rows]
    write_csv_once(result / "FAST_PRELIMINARY_EXPERIMENT_MATRIX.csv", logical, fieldnames=list(logical[0]))
    role_map = [
        {
            "dataset": row["dataset"],
            "logical_role": row["logical_role"],
            "model_seed": row["model_seed"],
            "candidate_set_hash": row["candidate_set_hash"],
            "physical_run_id": next(
                (
                    physical_row["physical_run_id"]
                    for physical_row in sealed_physical
                    if physical_row["dataset"] == row["dataset"]
                    and physical_row["candidate_set_hash"] == row["candidate_set_hash"]
                    and int(physical_row["model_seed"]) == int(row["model_seed"])
                ),
                "",
            ),
            "availability": row["availability"],
            "status": row["status"],
        }
        for row in logical
    ]
    write_csv_once(result / "FAST_LOGICAL_PHYSICAL_RUN_MAP.csv", role_map, fieldnames=list(role_map[0]))
    write_csv_once(result / "FAST_PRELIMINARY_RUN_PLAN.csv", sealed_physical, fieldnames=list(sealed_physical[0]))
    write_json_once(result / "FAST_PRELIMINARY_RUN_PLAN_SEAL.json", plan_seal)
    return logical, sealed_physical, plan_seal


def evaluate_and_plan_next_search_stage(
    *,
    result_root: str | Path,
    cache_resolutions: Mapping[str, Mapping[str, Any]],
    role_manifest_path: str | Path,
    role_seal_path: str | Path,
    private_mount_audit_path: str | Path,
    private_search_mount_root: str | Path,
    plan_path: str | Path,
    seal_path: str | Path,
    expected_seal_status: str,
    endpoint: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None, dict[str, Any] | None]:
    """Close one search boundary and publish only its preregistered successor."""

    if endpoint not in SEARCH_ENDPOINTS:
        raise ValueError("search coordinator endpoint is outside the frozen schedule")
    result = Path(result_root).resolve()
    rows, seal = validate_plan(
        plan_path,
        seal_path,
        expected_status=expected_seal_status,
        expected_stage=("SEARCH", "R1_ALL_SETS_0000_TO_0300"),
    )
    if any(int(row.get("endpoint_update", -1)) != endpoint for row in rows):
        raise ValueError("search coordinator plan endpoint mismatch")
    dataset_status, active_datasets, blocked_datasets = _dataset_execution_status(result)
    if {str(row["dataset"]) for row in rows} != set(active_datasets):
        raise ValueError("search coordinator plan differs from the authorized dataset subset")
    metrics = evaluate_search_endpoint(
        result_root=result,
        role_manifest_path=role_manifest_path,
        role_seal_path=role_seal_path,
        private_mount_audit_path=private_mount_audit_path,
        private_search_mount_root=private_search_mount_root,
        plan_rows=rows,
        endpoint=endpoint,
    )
    metric_path = result / f"FAST_SEARCH_METRICS_{endpoint:06d}.csv"
    write_csv_once(metric_path, metrics, fieldnames=list(metrics[0]))
    metric_seal = envelope(
        "metachorus.v53.m6f-m7f.search-metrics-seal.v1",
        "PASS",
        {
            "endpoint_update": endpoint,
            "plan_seal_identity": seal["identity"],
            "row_count": len(metrics),
            "rows_identity": canonical_json_hash(metrics),
            "datasets": list(active_datasets),
            "blocked_datasets": list(blocked_datasets),
            "dataset_execution_status_identity": dataset_status["identity"],
            "model_seeds": list(SEARCH_MODEL_SEEDS),
            "metric_role": "GLOBAL_SEARCH_POOL_ONLY",
            "prelim_locked_report_access": False,
        },
    )
    write_json_once(result / f"FAST_SEARCH_METRICS_{endpoint:06d}_SEAL.json", metric_seal)
    if endpoint == SEARCH_ENDPOINTS[-1]:
        return metrics, None, None

    selected: dict[str, set[str]] = {}
    selection_rows: list[dict[str, Any]] = []
    if endpoint == SEARCH_ENDPOINTS[0]:
        selection_documents: dict[str, Any] = {}
        for dataset in active_datasets:
            cache = cache_resolutions[dataset]
            dataset_metrics = [row for row in metrics if row["dataset"] == dataset]
            decision = derive_r1_high_recall_pool(
                dataset_metrics,
                cache["all_sets"]["sets"],
                manual_candidate_set_hash=str(cache["manual"]["candidate_set_hash"]),
                current_automatic_candidate_set_hash=(
                    str(cache["current_automatic"]["candidate_set_hash"])
                    if cache["current_automatic"].get("status") == "EXACT_IDENTITY_RESOLVED"
                    else None
                ),
            )
            selection_documents[dataset] = decision
            selected[dataset] = {str(row["candidate_set_hash"]) for row in decision["selected"]}
            selection_rows.extend(
                {
                    "dataset": dataset,
                    "candidate_set_id": row["candidate_set_id"],
                    "candidate_set_hash": row["candidate_set_hash"],
                    "candidate_set_ordinal": int(row["candidate_set_ordinal"]),
                    "components": list(row["components"]),
                    "status": "SELECTED_FOR_R2_R3",
                }
                for row in decision["selected"]
            )
        write_json_once(
            result / "FAST_R1_HIGH_RECALL_POOL.json",
            envelope(
                "metachorus.v53.m6f-m7f.r1-high-recall-pool.v1",
                "PASS",
                {"per_dataset": selection_documents, "records": selection_rows},
            ),
        )
        write_csv_once(result / "FAST_R1_HIGH_RECALL_POOL.csv", selection_rows, fieldnames=list(selection_rows[0]))
    elif endpoint == SEARCH_ENDPOINTS[1]:
        for dataset in active_datasets:
            selected[dataset] = {
                str(row["candidate_set_hash"])
                for row in rows
                if row["dataset"] == dataset
            }
        selection_rows = [
            {
                "dataset": dataset,
                "candidate_set_hash": digest,
                "status": "R2_NO_PRUNING_PRESERVED_FOR_R3",
            }
            for dataset in active_datasets
            for digest in sorted(selected[dataset])
        ]
        write_csv_once(result / "FAST_R2_NO_PRUNING_REGISTRY.csv", selection_rows, fieldnames=list(selection_rows[0]))
    else:
        shortlist_documents: dict[str, Any] = {}
        for dataset in active_datasets:
            cache = cache_resolutions[dataset]
            candidate_by_hash = {
                str(row["candidate_set_hash"]): dict(row)
                for row in cache["all_sets"]["sets"]
            }
            present_hashes = {
                str(row["candidate_set_hash"])
                for row in rows
                if row["dataset"] == dataset
            }
            decision = derive_robust_shortlist(
                [row for row in metrics if row["dataset"] == dataset],
                [candidate_by_hash[digest] for digest in sorted(present_hashes)],
                manual_candidate_set_hash=str(cache["manual"]["candidate_set_hash"]),
                current_automatic_candidate_set_hash=(
                    str(cache["current_automatic"]["candidate_set_hash"])
                    if cache["current_automatic"].get("status") == "EXACT_IDENTITY_RESOLVED"
                    else None
                ),
            )
            shortlist_documents[dataset] = decision
            selected[dataset] = {str(row["candidate_set_hash"]) for row in decision["shortlist"]}
            selection_rows.extend(
                {
                    "dataset": dataset,
                    "candidate_set_id": row["candidate_set_id"],
                    "candidate_set_hash": row["candidate_set_hash"],
                    "candidate_set_ordinal": int(row["candidate_set_ordinal"]),
                    "components": list(row["components"]),
                    "status": "SELECTED_FOR_R4",
                }
                for row in decision["shortlist"]
            )
        write_json_once(
            result / "FAST_R3_ROBUST_SHORTLIST.json",
            envelope(
                "metachorus.v53.m6f-m7f.r3-robust-shortlist.v1",
                "PASS",
                {"per_dataset": shortlist_documents, "records": selection_rows},
            ),
        )
        write_csv_once(result / "FAST_R3_ROBUST_SHORTLIST.csv", selection_rows, fieldnames=list(selection_rows[0]))

    next_endpoint = SEARCH_ENDPOINTS[SEARCH_ENDPOINTS.index(endpoint) + 1]
    next_rows, next_seal = build_continuation_plan(
        prior_rows=rows,
        selected_candidate_hashes=selected,
        result_root=result,
        next_endpoint=next_endpoint,
        prior_seal_identity=seal["identity"],
    )
    write_csv_once(
        result / f"FAST_SEARCH_RUN_PLAN_{next_endpoint:06d}.csv",
        next_rows,
        fieldnames=list(next_rows[0]),
    )
    write_json_once(result / f"FAST_SEARCH_RUN_PLAN_{next_endpoint:06d}_SEAL.json", next_seal)
    if next_endpoint == SEARCH_ENDPOINTS[-1]:
        write_csv_once(
            result / "FAST_R4_RUN_PLAN.csv",
            next_rows,
            fieldnames=list(next_rows[0]),
        )
        write_json_once(result / "FAST_R4_RUN_PLAN_SEAL.json", next_seal)
    return metrics, next_rows, next_seal


def close_preliminary_training_registry(result_root: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Seal the exact physical preliminary matrix before any locked-label access."""

    result = Path(result_root).resolve()
    rows, plan_seal = validate_plan(
        result / "FAST_PRELIMINARY_RUN_PLAN.csv",
        result / "FAST_PRELIMINARY_RUN_PLAN_SEAL.json",
        expected_status="SEALED_BEFORE_PRELIMINARY_GPU_TRAINING",
        expected_stage="PRELIMINARY",
    )
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int]] = set()
    for row in rows:
        run_parent = result / "preliminary_runs" / str(row["physical_run_id"])
        markers = [
            attempt / "M6F_M7F_PRELIMINARY_003000_COMPLETE.json"
            for attempt in sorted(run_parent.glob("attempt_*"))
            if (attempt / "M6F_M7F_PRELIMINARY_003000_COMPLETE.json").is_file()
        ]
        if len(markers) != 1:
            raise RuntimeError(f"preliminary completion is not unique: {row['physical_run_id']}")
        attempt_index = int(markers[0].parent.name.removeprefix("attempt_"))
        marker = read_json(markers[0])
        basis = validate_run_marker(
            marker,
            row,
            stage="PRELIMINARY",
            endpoint=SEARCH_ENDPOINTS[-1],
            expected_attempt_index=attempt_index,
        )
        key = (str(row["dataset"]), str(row["candidate_set_hash"]), int(row["model_seed"]))
        if key in seen:
            raise ValueError("preliminary physical identity is duplicated")
        seen.add(key)
        records.append(
            {
                "dataset": row["dataset"],
                "physical_run_id": row["physical_run_id"],
                "logical_roles": list(row.get("logical_roles", [])),
                "candidate_set_id": row["candidate_set_id"],
                "candidate_set_hash": row["candidate_set_hash"],
                "candidate_set_ordinal": int(row["candidate_set_ordinal"]),
                "model_seed": int(row["model_seed"]),
                "attempt_index": attempt_index,
                "endpoint_update": int(basis["endpoint_update"]),
                "optimizer_steps": int(basis["optimizer_steps"]),
                "checkpoint_relative_path": basis["checkpoint_relative_path"],
                "checkpoint_sha256": basis["checkpoint_sha256"],
                "representation_relative_path": basis["representation_relative_path"],
                "representation_sha256": basis["representation_sha256"],
                "marker_identity": marker["identity"],
                "labels_mounted": False,
                "status": "PASS",
            }
        )
    expected = {
        (str(row["dataset"]), str(row["candidate_set_hash"]), int(row["model_seed"]))
        for row in rows
    }
    if seen != expected or not records:
        raise RuntimeError("preliminary physical registry differs from its sealed plan")
    active_datasets = tuple(
        dataset
        for dataset in DATASET_NAMES
        if any(str(row["dataset"]) == dataset for row in records)
    )
    dataset_status, expected_active, blocked_datasets = _dataset_execution_status(result)
    if active_datasets != expected_active:
        raise ValueError("preliminary registry differs from the authorized dataset subset")
    write_csv_once(
        result / "FAST_PRELIMINARY_RUN_REGISTRY.csv",
        records,
        fieldnames=list(records[0]),
    )
    seal = envelope(
        "metachorus.v53.m6f-m7f.preliminary-training-registry-seal.v1",
        "SEALED_BEFORE_PRELIMINARY_EVALUATION",
        {
            "plan_seal_identity": plan_seal["identity"],
            "record_count": len(records),
            "records_identity": canonical_json_hash(records),
            "all_status_pass": True,
            "all_endpoint_updates": [SEARCH_ENDPOINTS[-1]],
            "all_evaluation_model_seeds": sorted({int(row["model_seed"]) for row in records}),
            "active_datasets": list(active_datasets),
            "blocked_datasets": list(blocked_datasets),
            "dataset_execution_status_identity": dataset_status["identity"],
            "labels_mounted": False,
            "preliminary_evaluation_started": False,
        },
    )
    write_json_once(result / "FAST_PRELIMINARY_TRAINING_REGISTRY_SEAL.json", seal)
    return records, seal


def run_preliminary_evaluation(
    *,
    result_root: str | Path,
    role_manifest_path: str | Path,
    private_search_mount_root: str | Path,
    label_sources: Mapping[str, str | Path],
    label_source_roots: Mapping[str, str | Path],
) -> dict[str, Any]:
    """Perform the first locked-label access after the exact registry seal."""

    result = Path(result_root).resolve()
    registry_rows, training_seal = close_preliminary_training_registry(result)
    dataset_status, active_datasets, blocked_datasets = _dataset_execution_status(result)
    logical_rows = [_canonical_plan_row(row) for row in read_csv(result / "FAST_PRELIMINARY_EXPERIMENT_MATRIX.csv")]
    role_map_rows = [_canonical_plan_row(row) for row in read_csv(result / "FAST_LOGICAL_PHYSICAL_RUN_MAP.csv")]
    if len(logical_rows) != len(DATASET_NAMES) * 3 * len(EVAL_MODEL_SEEDS) or len(role_map_rows) != len(logical_rows):
        raise ValueError("preliminary logical matrix is not the exact 27-row declaration")
    session = PreliminaryLabelSession(
        role_manifest_path=role_manifest_path,
        private_search_mount_root=private_search_mount_root,
        label_sources=label_sources,
        label_source_roots=label_source_roots,
        training_registry_seal=training_seal,
    )
    by_run = {str(row["physical_run_id"]): row for row in registry_rows}
    physical_metrics: dict[str, dict[str, Any]] = {}
    for run_id, registry_row in sorted(by_run.items()):
        marker = {
            "basis": {
                "representation_relative_path": registry_row["representation_relative_path"],
                "representation_sha256": registry_row["representation_sha256"],
            }
        }
        evaluation = evaluate_preliminary_representation(
            _representation(result, marker),
            dataset=str(registry_row["dataset"]),
            roles=session.open_dataset(str(registry_row["dataset"])),
        )
        physical_metrics[run_id] = evaluation
    metrics_rows: list[dict[str, Any]] = []
    for logical in logical_rows:
        role = str(logical["logical_role"])
        if logical.get("availability") != "AVAILABLE":
            metrics_rows.append(
                {
                    "dataset": logical["dataset"],
                    "logical_role": role,
                    "availability": logical["availability"],
                    "physical_run_id": "",
                    "candidate_set_id": "",
                    "candidate_set_hash": "",
                    "model_seed": int(logical["model_seed"]),
                    "endpoint_update": "",
                    **{metric: "" for metric in METRICS},
                    "status": logical["availability"],
                }
            )
            continue
        matches = [
            row
            for row in role_map_rows
            if row["dataset"] == logical["dataset"]
            and row["logical_role"] == role
            and int(row["model_seed"]) == int(logical["model_seed"])
        ]
        if len(matches) != 1 or not str(matches[0].get("physical_run_id", "")):
            raise ValueError("logical role does not resolve to one physical preliminary run")
        run_id = str(matches[0]["physical_run_id"])
        registry_row = by_run[run_id]
        evaluation = physical_metrics[run_id]
        metrics_rows.append(
            {
                "dataset": logical["dataset"],
                "logical_role": role,
                "availability": "AVAILABLE",
                "physical_run_id": run_id,
                "candidate_set_id": registry_row["candidate_set_id"],
                "candidate_set_hash": registry_row["candidate_set_hash"],
                "model_seed": int(logical["model_seed"]),
                "endpoint_update": SEARCH_ENDPOINTS[-1],
                **evaluation["metrics"],
                "assignment_stability_mean": evaluation["assignment_stability_mean"],
                "classification_fit_role": evaluation["classification_fit_role"],
                "classification_metric_role": evaluation["classification_metric_role"],
                "clustering_fit_role": evaluation["clustering_fit_role"],
                "clustering_metric_role": evaluation["clustering_metric_role"],
                "confirmatory_claim_allowed": False,
                "status": "PASS",
            }
        )
    expected_access = {name: int(name in active_datasets) for name in DATASET_NAMES}
    if session.access_count != len(active_datasets) or session.per_dataset_access_count != expected_access:
        raise RuntimeError("preliminary label access count differs from the active dataset subset")
    write_csv_once(
        result / "FAST_PRELIMINARY_METRICS_BY_SEED.csv",
        metrics_rows,
        fieldnames=list(metrics_rows[0]),
    )
    summary_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    for dataset in DATASET_NAMES:
        available = [row for row in metrics_rows if row["dataset"] == dataset and row["status"] == "PASS"]
        if dataset in blocked_datasets:
            for role in ("FINAL_FROZEN_METHOD", "MANUAL"):
                summary_rows.append(
                    {
                        "dataset": dataset,
                        "logical_role": role,
                        "candidate_set_hash": "",
                        "model_seed_count": 0,
                        **{f"{metric}_mean": "" for metric in METRICS},
                        **{f"{metric}_sample_sd": "" for metric in METRICS},
                        "evidence_level": "FAST_SEARCH_BUDGET_BLOCKED",
                        "status": "FAST_SEARCH_BUDGET_BLOCKED",
                    }
                )
            comparison_rows.append(
                {
                    "dataset": dataset,
                    "comparison": "FINAL_FROZEN_METHOD_VS_MANUAL",
                    "model_seed": "",
                    "candidate_set_hash": "",
                    "manual_set_hash": "",
                    **{f"delta_{metric}": "" for metric in METRICS},
                    "decision_status": "FAST_SEARCH_BUDGET_BLOCKED",
                    "confirmatory_claim_allowed": False,
                    "status": "FAST_SEARCH_BUDGET_BLOCKED",
                }
            )
            decisions.append({"dataset": dataset, "status": "FAST_SEARCH_BUDGET_BLOCKED"})
            continue
        for role in ("FINAL_FROZEN_METHOD", "MANUAL"):
            arm = [row for row in available if row["logical_role"] == role]
            if len(arm) != len(EVAL_MODEL_SEEDS):
                raise ValueError("preliminary metrics lack one exact three-seed arm")
            summary_rows.append(
                {
                    "dataset": dataset,
                    "logical_role": role,
                    "candidate_set_hash": arm[0]["candidate_set_hash"],
                    "model_seed_count": len(arm),
                    **{
                        f"{metric}_mean": float(np.mean([float(row[metric]) for row in arm]))
                        for metric in METRICS
                    },
                    **{
                        f"{metric}_sample_sd": float(np.std([float(row[metric]) for row in arm], ddof=1))
                        for metric in METRICS
                    },
                    "evidence_level": "3_SEED_PRELIMINARY_NONCONFIRMATORY",
                    "status": "PASS",
                }
            )
        final_arm = [row for row in available if row["logical_role"] == "FINAL_FROZEN_METHOD"]
        manual_arm = [row for row in available if row["logical_role"] == "MANUAL"]
        decision = preliminary_decision(final_arm, manual_arm)
        decisions.append({"dataset": dataset, **decision})
        for seed in EVAL_MODEL_SEEDS:
            candidate = next(row for row in final_arm if int(row["model_seed"]) == seed)
            manual = next(row for row in manual_arm if int(row["model_seed"]) == seed)
            comparison_rows.append(
                {
                    "dataset": dataset,
                    "comparison": "FINAL_FROZEN_METHOD_VS_MANUAL",
                    "model_seed": int(seed),
                    "candidate_set_hash": candidate["candidate_set_hash"],
                    "manual_set_hash": manual["candidate_set_hash"],
                    **{f"delta_{metric}": float(candidate[metric]) - float(manual[metric]) for metric in METRICS},
                    "decision_status": decision["status"],
                    "confirmatory_claim_allowed": False,
                    "status": "PASS",
                }
            )
        comparison_rows.append(
            {
                "dataset": dataset,
                "comparison": "CURRENT_AUTOMATIC_VS_MANUAL",
                "model_seed": "",
                "candidate_set_hash": "",
                "manual_set_hash": manual_arm[0]["candidate_set_hash"],
                **{f"delta_{metric}": "" for metric in METRICS},
                "decision_status": "NOT_AVAILABLE_EXACT_IDENTITY",
                "confirmatory_claim_allowed": False,
                "status": "NOT_AVAILABLE_EXACT_IDENTITY",
            }
        )
    write_csv_once(result / "FAST_PRELIMINARY_SUMMARY.csv", summary_rows, fieldnames=list(summary_rows[0]))
    write_csv_once(
        result / "FAST_MANUAL_AUTOMATIC_COMPARISON.csv",
        comparison_rows,
        fieldnames=list(comparison_rows[0]),
    )
    access = envelope(
        "metachorus.v53.m6f-m7f.preliminary-access-audit.v1",
        "PASS",
        {
            "training_registry_seal_identity": training_seal["identity"],
            "label_role_manifest_identity": session.manifest["identity"],
            "dataset_execution_status_identity": dataset_status["identity"],
            "active_datasets": list(active_datasets),
            "blocked_datasets": list(blocked_datasets),
            "prelim_locked_report_access_count": session.access_count,
            "per_dataset_access_count": session.per_dataset_access_count,
            "raw_labels_persisted": False,
            "confirmatory_claim_allowed": False,
            "metrics_rows_identity": canonical_json_hash(metrics_rows),
            "summary_rows_identity": canonical_json_hash(summary_rows),
            "comparison_rows_identity": canonical_json_hash(comparison_rows),
            "decisions": decisions,
        },
    )
    write_json_once(result / "FAST_PRELIMINARY_ACCESS_AUDIT.json", access)
    return access


__all__ = [
    "validate_plan",
    "authorize_search_after_canary",
    "run_plan_shard",
    "close_canaries",
    "authorize_r1_budget",
    "evaluate_search_endpoint",
    "build_continuation_plan",
    "seal_final_search_selection",
    "publish_preliminary_plan",
    "evaluate_and_plan_next_search_stage",
    "close_preliminary_training_registry",
    "run_preliminary_evaluation",
]
