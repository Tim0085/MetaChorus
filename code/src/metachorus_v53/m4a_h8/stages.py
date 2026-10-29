"""H8 execution stages, retry fuse, canary gate, and training registry closure."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from metachorus_v53.audit.common import canonical_json_hash, read_json, sha256_file

from .common import (
    append_jsonl_once,
    envelope,
    read_csv,
    validate_envelope,
    write_csv_once,
    write_json_once,
)
from .constants import (
    COMMON_ROOT_CAUSE_PAUSE_THRESHOLD,
    ENGINEERING_RETRIES_PER_IDENTITY,
    EXPECTED_CANARY_RUNS,
    EXPECTED_SCIENCE_RUNS,
    SCIENCE_ENDPOINT,
)
from .planning import (
    PLAN_FIELDS,
    canonical_plan_row,
    seal_science_plan,
    validate_canary_plan_rows,
    validate_plan_seal,
    validate_science_plan_rows,
)
from .training import run_plan_row, validate_stage_marker


RUN_REGISTRY_FIELDS = (
    "physical_run_id",
    "dataset",
    "role",
    "candidate_set_id",
    "candidate_set_hash",
    "candidate_set_ordinal",
    "model_seed",
    "endpoint_update",
    "successful_attempt_index",
    "optimizer_steps",
    "model_total_seconds",
    "peak_gpu_allocated_bytes",
    "peak_gpu_reserved_bytes",
    "gpu_uuid",
    "runtime_domain_identity",
    "labels_mounted",
    "h5_locked_report_access",
    "status",
)

ARTIFACT_REGISTRY_FIELDS = (
    "physical_run_id",
    "dataset",
    "candidate_set_hash",
    "model_seed",
    "endpoint_update",
    "successful_attempt_index",
    "artifact_kind",
    "relative_path",
    "sha256",
    "size_bytes",
    "strict_state_complete",
    "labels_mounted",
    "status",
)


def _stage_files(stage: str) -> tuple[str, str, str, int]:
    if stage == "CANARY":
        return (
            "M4A_H8_CANARY_RUN_PLAN.csv",
            "M4A_H8_CANARY_PLAN_SEAL.json",
            "H8_CANARY_000020_COMPLETE.json",
            20,
        )
    if stage == "S_EVAL":
        return (
            "TC_LPMSO_H8_RUN_PLAN.csv",
            "TC_LPMSO_H8_RUN_PLAN_SEAL.json",
            "H8_STAGE_003000_COMPLETE.json",
            SCIENCE_ENDPOINT,
        )
    raise ValueError(stage)


def _root_cause_signature(error: BaseException) -> tuple[str, str, str]:
    kind = type(error).__name__
    message = str(error).casefold()
    message = re.sub(r"[a-z]:[\\/][^\s]+|/(?:[^\s/]+/)+[^\s]+", "<path>", message)
    message = re.sub(r"m4a-h8-[a-z0-9-]+", "<run>", message)
    message = re.sub(r"\b[0-9a-f]{32,64}\b", "<hash>", message)
    message = re.sub(r"\b\d+\b", "<n>", message)
    signature = canonical_json_hash({"failure_kind": kind, "sanitized": message})
    root = canonical_json_hash({"failure_kind": kind, "sanitized_error_signature": signature})
    return kind, signature, root


def _failure_path(result_root: Path, stage: str, run_id: str, attempt: int) -> Path:
    return (
        result_root
        / "failed_attempts"
        / stage.casefold()
        / f"{run_id}__attempt_{attempt:02d}.json"
    )


def _validated_failure(path: Path, *, stage: str, run_id: str, attempt: int) -> dict[str, Any]:
    value = validate_envelope(
        read_json(path),
        schema_version="metachorus.v53.m4a-h8.failed-attempt.v1",
        status="FAILED_PRESERVED",
    )
    basis = value["basis"]
    if (
        basis.get("stage") != stage
        or basis.get("physical_run_id") != run_id
        or int(basis.get("attempt_index", -1)) != attempt
        or basis.get("scientific_metrics_observed") is not False
        or int(basis.get("optimizer_steps_after_boundary", -1)) != 0
        or basis.get("labels_mounted") is not False
        or basis.get("h5_locked_report_access") is not False
    ):
        raise ValueError("invalid H8 failure receipt")
    return value


def _common_pause(result_root: Path, stage: str) -> dict[str, Any] | None:
    path = result_root / "M4A_H8_COMMON_ROOT_CAUSE_PAUSE.json"
    if not path.is_file():
        return None
    value = validate_envelope(
        read_json(path),
        schema_version="metachorus.v53.m4a-h8.common-root-cause-pause.v1",
        status="IMMEDIATE_COMMON_ROOT_CAUSE_PAUSE",
    )
    if value["basis"].get("stage") != stage or value["basis"].get("fanout_must_stop") is not True:
        raise ValueError("invalid H8 common-root-cause pause")
    return value


def run_plan_shard(
    *,
    repo_root: Path,
    cache_root: Path,
    result_root: Path,
    plan_path: Path,
    stage: str,
    shard_index: int,
    shard_count: int = 2,
    retry_failed: bool = False,
    enforce_target_gpu: bool = True,
) -> dict[str, Any]:
    if shard_count != 2 or shard_index not in {0, 1}:
        raise ValueError("H8 uses exactly two independent single-GPU shards")
    if _common_pause(result_root, stage) is not None:
        raise RuntimeError("H8 is paused by the common-root-cause fuse")
    expected_plan, expected_seal, _, _ = _stage_files(stage)
    if plan_path.name != expected_plan:
        raise ValueError("H8 plan filename/stage mismatch")
    rows, _ = validate_plan_seal(
        plan_path,
        result_root / expected_seal,
        stage=stage,
    )
    selected = [row for index, row in enumerate(rows) if index % shard_count == shard_index]
    if retry_failed and not any(
        _failure_path(result_root, stage, str(row["physical_run_id"]), 0).is_file()
        for row in selected
    ):
        raise RuntimeError("H8 retry requested but this shard has no preserved attempt_00 failure")
    failures: list[str] = []
    completions: list[str] = []
    for planned in selected:
        row = dict(planned)
        run_id = str(row["physical_run_id"])
        existing = [
            path
            for attempt in (0, 1)
            if (path := _failure_path(result_root, stage, run_id, attempt)).is_file()
        ]
        if existing and not retry_failed:
            failures.append(run_id)
            continue
        if retry_failed:
            if len(existing) != 1:
                if existing:
                    raise RuntimeError("H8 engineering retry limit reached")
                continue
            _validated_failure(existing[0], stage=stage, run_id=run_id, attempt=0)
            row["attempt_index"] = 1
        else:
            row["attempt_index"] = 0
        try:
            marker = run_plan_row(
                plan_row=row,
                repo_root=repo_root,
                cache_root=cache_root,
                result_root=result_root,
                enforce_target_gpu=enforce_target_gpu,
            )
            completions.append(run_id)
            if int(row["attempt_index"]) == 1:
                failed_path = existing[0]
                failed = _validated_failure(failed_path, stage=stage, run_id=run_id, attempt=0)
                resolution_basis = {
                    "physical_run_id": run_id,
                    "stage": stage,
                    "failed_attempt_index": 0,
                    "successful_attempt_index": 1,
                    "engineering_retry_count": 1,
                    "failed_attempt_receipt_sha256": sha256_file(failed_path),
                    "failed_attempt_identity": failed["identity"],
                    "successful_stage_marker_identity": marker["identity"],
                    "scientific_metrics_observed_before_retry": False,
                    "labels_mounted": False,
                }
                write_json_once(
                    result_root
                    / "failed_attempts"
                    / stage.casefold()
                    / f"{run_id}__retry_resolution.json",
                    envelope(
                        "metachorus.v53.m4a-h8.retry-resolution.v1",
                        "ENGINEERING_RETRY_RECOVERED",
                        resolution_basis,
                    ),
                )
        except Exception as error:
            attempt = int(row["attempt_index"])
            kind, signature, root_cause = _root_cause_signature(error)
            failure_basis = {
                "physical_run_id": run_id,
                "stage": stage,
                "attempt_index": attempt,
                "failure_kind": kind,
                "sanitized_error_signature": signature,
                "root_cause_identity": root_cause,
                "scientific_metrics_observed": False,
                "optimizer_steps_after_boundary": 0,
                "engineering_retry_limit": ENGINEERING_RETRIES_PER_IDENTITY,
                "labels_mounted": False,
                "h5_locked_report_access": False,
            }
            failure = envelope(
                "metachorus.v53.m4a-h8.failed-attempt.v1",
                "FAILED_PRESERVED",
                failure_basis,
            )
            path = _failure_path(result_root, stage, run_id, attempt)
            write_json_once(path, failure)
            failures.append(run_id)
            if stage == "S_EVAL":
                receipts = [
                    read_json(member)
                    for member in sorted(
                        (result_root / "failed_attempts" / "s_eval").glob("*__attempt_*.json")
                    )
                ]
                affected = sorted(
                    {
                        str(item.get("basis", {}).get("physical_run_id"))
                        for item in receipts
                        if item.get("basis", {}).get("root_cause_identity") == root_cause
                    }
                )
                if len(affected) >= COMMON_ROOT_CAUSE_PAUSE_THRESHOLD:
                    pause_basis = {
                        "stage": stage,
                        "root_cause_identity": root_cause,
                        "affected_physical_run_ids": affected,
                        "affected_scientific_identity_count": len(affected),
                        "threshold": COMMON_ROOT_CAUSE_PAUSE_THRESHOLD,
                        "fanout_must_stop": True,
                        "completed_runs_preserved": True,
                        "labels_mounted": False,
                        "h5_locked_report_access": False,
                    }
                    write_json_once(
                        result_root / "M4A_H8_COMMON_ROOT_CAUSE_PAUSE.json",
                        envelope(
                            "metachorus.v53.m4a-h8.common-root-cause-pause.v1",
                            "IMMEDIATE_COMMON_ROOT_CAUSE_PAUSE",
                            pause_basis,
                        ),
                    )
                    break
    summary = {
        "stage": stage,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "planned_count": len(selected),
        "completed_count": len(completions),
        "failed_count": len(failures),
        "completed_physical_run_ids": completions,
        "failed_physical_run_ids": failures,
        "retry_failed": retry_failed,
        "labels_mounted": False,
        "h5_locked_report_access": False,
    }
    if failures:
        raise RuntimeError(f"H8 shard preserved failures: {failures}")
    return summary


def _completed_attempts(
    result_root: Path, *, stage: str, row: Mapping[str, Any]
) -> tuple[Path, dict[str, Any], int]:
    _, _, marker_name, _ = _stage_files(stage)
    base = "canary_runs" if stage == "CANARY" else "runs"
    run_parent = result_root / base / str(row["physical_run_id"])
    attempts = sorted(run_parent.glob("attempt_*")) if run_parent.is_dir() else []
    if any(path.is_symlink() or path.name not in {"attempt_00", "attempt_01"} for path in attempts):
        raise ValueError("H8 attempt directory set is non-canonical")
    completed = [path for path in attempts if (path / marker_name).is_file()]
    if len(completed) != 1:
        raise RuntimeError(f"H8 requires exactly one completed attempt: {row['physical_run_id']}")
    attempt = completed[0]
    marker = validate_stage_marker(attempt / marker_name, plan_row=row, run_root=attempt)
    return attempt, marker, int(attempt.name.rsplit("_", 1)[-1])


def close_canary(result_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    rows, seal = validate_plan_seal(
        result_root / "M4A_H8_CANARY_RUN_PLAN.csv",
        result_root / "M4A_H8_CANARY_PLAN_SEAL.json",
        stage="CANARY",
    )
    validate_canary_plan_rows(rows)
    records: list[dict[str, Any]] = []
    for row in rows:
        attempt, marker, attempt_index = _completed_attempts(
            result_root, stage="CANARY", row=row
        )
        basis = marker["basis"]
        audit = basis.get("canary_engineering_audit", {})
        if (
            basis.get("eligible_for_scientific_result") is not False
            or basis.get("reusable_for_science") is not False
            or basis.get("h5_locked_report_access") is not False
            or basis.get("labels_mounted") is not False
            or int(basis.get("optimizer_steps", -1)) != 20
            or not isinstance(audit, Mapping)
            or audit.get("status") != "PASS"
            or audit.get("checkpoint_write_pass") is not True
            or audit.get("checkpoint_read_pass") is not True
            or audit.get("checkpoint_state_roundtrip") is not True
            or audit.get("checkpoint_restored_in_fresh_session") is not True
            or audit.get("checkpoint_strict_state_complete") is not True
            or audit.get("checkpoint_rng_roundtrip") is not True
            or audit.get("path_resolution_pass") is not True
            or audit.get("optimizer_step_pass") is not True
            or int(audit.get("optimizer_state_entry_count", 0)) <= 0
            or list(audit.get("optimizer_step_values", [])) != [20]
        ):
            raise ValueError("H8 canary violated its engineering-only boundary")
        records.append(
            {
                "physical_run_id": str(row["physical_run_id"]),
                "dataset": str(row["dataset"]),
                "candidate_set_hash": str(row["candidate_set_hash"]),
                "model_seed": int(row["model_seed"]),
                "successful_attempt_index": attempt_index,
                "stage_marker_identity": marker["identity"],
                "checkpoint_sha256": basis["checkpoint_sha256"],
                "representation_sha256": basis["representation_sha256"],
                "resource_trace_sha256": basis["resource_trace_sha256"],
                "checkpoint_write_pass": True,
                "checkpoint_read_pass": True,
                "checkpoint_state_roundtrip": True,
                "checkpoint_restored_in_fresh_session": True,
                "checkpoint_strict_state_complete": True,
                "checkpoint_rng_roundtrip": True,
                "path_resolution_pass": True,
                "optimizer_step_pass": True,
                "status": "PASS",
            }
        )
    result_basis = {
        "plan_sha256": sha256_file(result_root / "M4A_H8_CANARY_RUN_PLAN.csv"),
        "plan_seal_identity": seal["identity"],
        "run_count": len(records),
        "records": records,
        "records_identity": canonical_json_hash(records),
        "acm_pass": sum(row["dataset"] == "acm" for row in records) == 1,
        "freebase_pass": sum(row["dataset"] == "freebase" for row in records) == 1,
        "all_checkpoint_writes_pass": all(row["checkpoint_write_pass"] for row in records),
        "all_checkpoint_reads_pass": all(row["checkpoint_read_pass"] for row in records),
        "all_path_resolutions_pass": all(row["path_resolution_pass"] for row in records),
        "all_optimizer_steps_pass": all(row["optimizer_step_pass"] for row in records),
        "scientific_metrics_observed": False,
        "formal_training_started": False,
        "fanout_started_before_canary_pass": False,
        "labels_mounted": False,
        "h5_locked_report_access": False,
    }
    if len(records) != EXPECTED_CANARY_RUNS:
        raise RuntimeError("H8 canary completion count mismatch")
    results = envelope(
        "metachorus.v53.m4a-h8.canary-results.v1", "PASS", result_basis
    )
    write_json_once(result_root / "M4A_H8_CANARY_RESULTS.json", results)
    gate_basis = {
        "canary_results_identity": results["identity"],
        "acm_pass": True,
        "freebase_pass": True,
        "all_checkpoint_writes_pass": True,
        "all_checkpoint_reads_pass": True,
        "all_path_resolutions_pass": True,
        "all_optimizer_steps_pass": True,
        "fanout_authorized": True,
        "fanout_started_before_decision": False,
        "scientific_metrics_observed": False,
        "labels_mounted": False,
        "h5_locked_report_access": False,
    }
    gate = envelope(
        "metachorus.v53.m4a-h8.canary-gate-decision.v1",
        "H8_CANARY_PASS",
        gate_basis,
    )
    write_json_once(result_root / "M4A_H8_CANARY_GATE_DECISION.json", gate)
    append_jsonl_once(
        result_root / "M4A_H8_EXECUTION_LOG.jsonl",
        {
            "event_id": "H8_CANARY_GATE_PASS",
            "status": "H8_CANARY_PASS",
            "canary_results_identity": results["identity"],
            "canary_gate_identity": gate["identity"],
            "fanout_authorized": True,
            "labels_mounted": False,
            "h5_locked_report_access": False,
        },
    )
    return results, gate


def authorize_science(result_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _, gate = close_canary(result_root)
    draft_path = result_root / "TC_LPMSO_H8_RUN_PLAN_DRAFT.csv"
    draft = read_csv(draft_path)
    validate_science_plan_rows(draft, require_seal=False)
    draft_seal = validate_envelope(
        read_json(result_root / "TC_LPMSO_H8_RUN_PLAN_DRAFT_SEAL.json"),
        schema_version="metachorus.v53.m4a-h8.science-draft-seal.v1",
        status="PENDING_CANARY",
    )
    canonical = [canonical_plan_row(row) for row in draft]
    if (
        int(draft_seal["basis"].get("row_count", -1)) != EXPECTED_SCIENCE_RUNS
        or draft_seal["basis"].get("rows_identity") != canonical_json_hash(canonical)
        or draft_seal["basis"].get("formal_fanout_authorized") is not False
    ):
        raise ValueError("H8 science draft seal mismatch")
    sealed, seal = seal_science_plan(canonical, canary_gate_identity=gate["identity"])
    write_csv_once(
        result_root / "TC_LPMSO_H8_RUN_PLAN.csv",
        sealed,
        fieldnames=PLAN_FIELDS,
    )
    write_json_once(result_root / "TC_LPMSO_H8_RUN_PLAN_SEAL.json", seal)
    append_jsonl_once(
        result_root / "M4A_H8_EXECUTION_LOG.jsonl",
        {
            "event_id": "H8_S_EVAL_PLAN_SEALED_AFTER_CANARY_PASS",
            "status": "SEALED_BEFORE_GPU_TRAINING",
            "canary_gate_identity": gate["identity"],
            "science_plan_identity": seal["identity"],
            "science_run_count": EXPECTED_SCIENCE_RUNS,
            "labels_mounted": False,
            "h5_locked_report_access": False,
        },
    )
    return sealed, seal


def close_science_registry(result_root: Path) -> dict[str, Any]:
    rows, plan_seal = validate_plan_seal(
        result_root / "TC_LPMSO_H8_RUN_PLAN.csv",
        result_root / "TC_LPMSO_H8_RUN_PLAN_SEAL.json",
        stage="S_EVAL",
    )
    validate_science_plan_rows(rows)
    run_rows: list[dict[str, Any]] = []
    artifact_rows: list[dict[str, Any]] = []
    for row in rows:
        _, marker, attempt_index = _completed_attempts(
            result_root, stage="S_EVAL", row=row
        )
        basis = marker["basis"]
        if (
            basis.get("eligible_for_scientific_result") is not True
            or basis.get("eligible_for_selector_development") is not False
            or basis.get("reusable_for_science") is not True
            or basis.get("labels_mounted") is not False
            or basis.get("h5_locked_report_access") is not False
            or int(basis.get("optimizer_steps", -1)) != SCIENCE_ENDPOINT
            or basis.get("retention", {}).get("intermediate_checkpoint_count") != 1
        ):
            raise ValueError("H8 scientific run boundary mismatch")
        run_rows.append(
            {
                "physical_run_id": str(row["physical_run_id"]),
                "dataset": str(row["dataset"]),
                "role": str(row["role"]),
                "candidate_set_id": str(row["candidate_set_id"]),
                "candidate_set_hash": str(row["candidate_set_hash"]),
                "candidate_set_ordinal": int(row["candidate_set_ordinal"]),
                "model_seed": int(row["model_seed"]),
                "endpoint_update": SCIENCE_ENDPOINT,
                "successful_attempt_index": attempt_index,
                "optimizer_steps": int(basis["optimizer_steps"]),
                "model_total_seconds": float(basis["model_total_seconds"]),
                "peak_gpu_allocated_bytes": int(basis["peak_gpu_allocated_bytes"]),
                "peak_gpu_reserved_bytes": int(basis["peak_gpu_reserved_bytes"]),
                "gpu_uuid": str(basis["gpu_uuid"]),
                "runtime_domain_identity": str(basis["runtime_domain_identity"]),
                "labels_mounted": False,
                "h5_locked_report_access": False,
                "status": "PASS",
            }
        )
        for kind in ("checkpoint", "representation"):
            relative = str(basis[f"{kind}_relative_path"])
            path = result_root / Path(*relative.split("/"))
            if path.is_symlink() or not path.is_file() or sha256_file(path) != basis[f"{kind}_sha256"]:
                raise ValueError("H8 final artifact registry hash mismatch")
            if kind == "representation":
                array = np.load(path, allow_pickle=False)
                if array.ndim != 2 or array.size == 0 or not np.isfinite(array).all():
                    raise ValueError("H8 representation registry contains invalid data")
            artifact_rows.append(
                {
                    "physical_run_id": str(row["physical_run_id"]),
                    "dataset": str(row["dataset"]),
                    "candidate_set_hash": str(row["candidate_set_hash"]),
                    "model_seed": int(row["model_seed"]),
                    "endpoint_update": SCIENCE_ENDPOINT,
                    "successful_attempt_index": attempt_index,
                    "artifact_kind": kind.upper(),
                    "relative_path": relative,
                    "sha256": str(basis[f"{kind}_sha256"]),
                    "size_bytes": int(basis[f"{kind}_size_bytes"]),
                    "strict_state_complete": True,
                    "labels_mounted": False,
                    "status": "PASS",
                }
            )
    if len(run_rows) != EXPECTED_SCIENCE_RUNS or len(artifact_rows) != 2 * EXPECTED_SCIENCE_RUNS:
        raise RuntimeError("H8 science registry is incomplete")
    run_path = result_root / "TC_LPMSO_H8_RUN_REGISTRY.csv"
    artifact_path = result_root / "TC_LPMSO_H8_CHECKPOINT_REGISTRY.csv"
    write_csv_once(run_path, run_rows, fieldnames=RUN_REGISTRY_FIELDS)
    write_csv_once(artifact_path, artifact_rows, fieldnames=ARTIFACT_REGISTRY_FIELDS)
    seal_basis = {
        "plan_sha256": sha256_file(result_root / "TC_LPMSO_H8_RUN_PLAN.csv"),
        "plan_seal_identity": plan_seal["identity"],
        "run_registry_sha256": sha256_file(run_path),
        "artifact_registry_sha256": sha256_file(artifact_path),
        "run_count": len(run_rows),
        "checkpoint_count": sum(row["artifact_kind"] == "CHECKPOINT" for row in artifact_rows),
        "representation_count": sum(row["artifact_kind"] == "REPRESENTATION" for row in artifact_rows),
        "all_status_pass": True,
        "all_model_seeds_consumed": True,
        "labels_mounted": False,
        "h5_locked_report_access": False,
        "locked_evaluation_started": False,
    }
    seal = envelope(
        "metachorus.v53.m4a-h8.training-registry-seal.v1",
        "SEALED_BEFORE_LOCKED_EVALUATION",
        seal_basis,
    )
    write_json_once(result_root / "TC_LPMSO_H8_TRAINING_REGISTRY_SEAL.json", seal)
    domains: dict[tuple[str, int], set[str]] = {}
    for row in run_rows:
        key = (str(row["dataset"]), int(row["model_seed"]))
        domains.setdefault(key, set()).add(str(row["runtime_domain_identity"]))
    for (dataset, model_seed), observed in sorted(domains.items()):
        if len(observed) > 1:
            append_jsonl_once(
                result_root / "TC_LPMSO_H8_OPERATIONAL_DEVIATION_LOG.jsonl",
                {
                    "event_id": f"RUNTIME_DOMAIN_DEVIATION_{dataset.upper()}_{model_seed}",
                    "status": "RECORDED_WITHOUT_METRIC_RETRY",
                    "dataset": dataset,
                    "model_seed": model_seed,
                    "runtime_domain_identities": sorted(observed),
                    "scientific_identity_changed": False,
                    "successful_runs_reexecuted": False,
                },
            )
    append_jsonl_once(
        result_root / "M4A_H8_EXECUTION_LOG.jsonl",
        {
            "event_id": "H8_50_RUN_TRAINING_REGISTRY_SEALED",
            "status": "SEALED_BEFORE_LOCKED_EVALUATION",
            "run_count": len(run_rows),
            "registry_seal_identity": seal["identity"],
            "labels_mounted": False,
            "h5_locked_report_access": False,
        },
    )
    return seal
