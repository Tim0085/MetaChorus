"""H7 GPU sharding, engineering fuse, canary closure, and R4 registries."""

from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from metachorus_v53.audit.common import canonical_json_hash, read_json, sha256_file
from metachorus_v53.m4a_h4.continuation import checkpoint_state_audit

from .common import (
    append_jsonl_once,
    envelope,
    read_csv,
    safe_relative,
    write_csv_once,
    write_json_once,
    write_text_once,
)
from .constants import (
    CANARY_ENDPOINT,
    COMMON_ROOT_CAUSE_PAUSE_THRESHOLD,
    CONTROLS,
    ENGINEERING_RETRIES_PER_IDENTITY,
    EXPECTED_R4_RUNS,
    FUTURE_S_EVAL_MODEL_SEEDS,
    PARENT_UPDATE,
    R4_ENDPOINT,
)
from .continuation import run_plan_row, validate_stage_marker
from .evaluation import run_stage_evaluation
from .failure import write_failure
from .planning import (
    PLAN_FIELDS,
    build_r4_plan,
    canonical_plan_row,
    validate_canary_gate,
    validate_canary_plan_rows,
    validate_r4_plan_rows,
)
from .protocol import (
    aggregate_r4_objectives,
    manual_safety_audit,
    robust_r4_selection,
)


STAGE_PREFIX = {"R4": "TC_LPMSO_H7_R4"}


def _stable_shard(run_id: str, shard_count: int) -> int:
    digest = hashlib.sha256(run_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % int(shard_count)


def _plan_seal_path(result_root: Path, stage: str) -> Path:
    if stage == "CANARY":
        return result_root / "M4A_H7_CANARY_PLAN_SEAL.json"
    if stage == "R4":
        return result_root / "TC_LPMSO_H7_R4_PLAN_SEAL.json"
    raise ValueError(stage)


def validate_plan_seal(
    *, plan_path: Path, result_root: Path, stage: str, rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    if not rows or any(str(row["stage"]) != stage for row in rows):
        raise RuntimeError(f"invalid or empty H7 {stage} plan")
    seal = read_json(_plan_seal_path(result_root, stage))
    basis = seal.get("basis", {})
    canonical_rows = [canonical_plan_row(row) for row in rows]
    if (
        seal.get("schema_version") != "metachorus.v53.m4a-h7.execution-plan-seal.v1"
        or seal.get("status") != "SEALED_BEFORE_GPU_TRAINING"
        or seal.get("identity") != canonical_json_hash(basis)
        or basis.get("stage") != stage
        or int(basis.get("row_count", -1)) != len(rows)
        or basis.get("rows_identity") != canonical_json_hash(canonical_rows)
        or any(str(row.get("plan_seal_sha256")) != seal.get("identity") for row in rows)
        or basis.get("labels_mounted") is not False
        or basis.get("run_level_parallel_only") is not True
        or int(basis.get("maximum_gpu_count", -1)) != 2
        or basis.get("h5_locked_report_access") is not False
        or basis.get("s_eval_started") is not False
    ):
        raise RuntimeError(f"H7 {stage} plan seal mismatch")
    return seal


def _failure_payload(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    value = read_json(path)
    basis = value.get("basis", {})
    if (
        value.get("schema_version") != "metachorus.v53.m4a-h7.failed-attempt.v1"
        or value.get("status") != "FAILED_PRESERVED"
        or value.get("identity") != canonical_json_hash(basis)
    ):
        raise RuntimeError(f"invalid H7 failed-attempt receipt: {path.name}")
    return value, basis


def _root_cause_signature(exc: Exception) -> dict[str, str]:
    message = str(exc).replace("\\", "/")
    message = re.sub(r"[A-Za-z]:/[^\s,'\"]+", "<path>", message)
    message = re.sub(r"/(?:[^\s,'\"]+/)+[^\s,'\"]+", "<path>", message)
    message = re.sub(r"M4A-H7-[A-Z0-9-]+", "<run>", message)
    message = re.sub(r"\b[0-9a-fA-F]{32,64}\b", "<identity>", message)
    message = re.sub(r"\b\d+\b", "<n>", message)
    return {
        "failure_kind": type(exc).__name__,
        "sanitized_error_signature": canonical_json_hash(message),
    }


def _root_cause_counts(result_root: Path) -> dict[str, set[str]]:
    output: dict[str, set[str]] = {}
    root = result_root / "failed_attempts"
    if not root.is_dir():
        return output
    for path in root.rglob("*__attempt_*.json"):
        _, basis = _failure_payload(path)
        root_cause = str(basis["root_cause_identity"])
        output.setdefault(root_cause, set()).add(str(basis["physical_run_id"]))
    return output


def _parent_root(parent_source: str, *, h5_result_root: Path, h6_result_root: Path) -> Path:
    if parent_source == "H5_RESULT_ROOT":
        return h5_result_root
    if parent_source == "H6_RESULT_ROOT":
        return h6_result_root
    raise RuntimeError(f"unsupported H7 parent source: {parent_source}")


def run_plan_shard(
    *,
    plan_path: Path,
    stage: str,
    shard_index: int,
    shard_count: int,
    repo_root: Path,
    cache_root: Path,
    h5_result_root: Path,
    h6_result_root: Path,
    h7_result_root: Path,
    enforce_target_gpu: bool = True,
    retry_failed: bool = False,
) -> dict[str, Any]:
    if stage not in {"CANARY", "R4"}:
        raise ValueError(stage)
    if not (0 <= int(shard_index) < int(shard_count) <= 2):
        raise ValueError("H7 permits one or two run-level shards only")
    if (h7_result_root / "M4A_H7_COMMON_ROOT_CAUSE_PAUSE.json").exists():
        raise RuntimeError("H7 is paused by the common-root-cause fuse")
    all_rows = read_csv(plan_path)
    validate_plan_seal(
        plan_path=plan_path, result_root=h7_result_root, stage=stage, rows=all_rows
    )
    if stage == "CANARY":
        validate_canary_plan_rows(all_rows)
    else:
        validate_canary_gate(h7_result_root)
        validate_r4_plan_rows(all_rows)
    rows = [
        row
        for row in all_rows
        if _stable_shard(str(row["physical_run_id"]), shard_count) == int(shard_index)
    ]
    failed_root = h7_result_root / "failed_attempts" / stage.lower()
    failed_root.mkdir(parents=True, exist_ok=True)
    completed = 0
    failures: list[str] = []
    for row in rows:
        working = dict(row)
        run_id = str(row["physical_run_id"])
        planned_attempt = int(row.get("attempt_index", 0))
        receipt_path = failed_root / f"{run_id}__attempt_{planned_attempt:02d}.json"
        resolution_path = failed_root / f"{run_id}__retry_resolution.json"
        all_receipts = sorted(
            (h7_result_root / "failed_attempts").rglob(f"{run_id}__attempt_*.json")
        )
        if resolution_path.is_file():
            resolution = read_json(resolution_path)
            basis = resolution.get("basis", {})
            if (
                resolution.get("schema_version") != "metachorus.v53.m4a-h7.retry-resolution.v1"
                or resolution.get("status") != "ENGINEERING_RETRY_RECOVERED"
                or resolution.get("identity") != canonical_json_hash(basis)
                or basis.get("physical_run_id") != run_id
                or basis.get("stage") != stage
                or int(basis.get("failed_attempt_index", -1)) != planned_attempt
                or int(basis.get("successful_attempt_index", -1)) != planned_attempt + 1
                or int(basis.get("engineering_retry_count", -1))
                != ENGINEERING_RETRIES_PER_IDENTITY
            ):
                raise RuntimeError(f"invalid H7 retry resolution: {resolution_path.name}")
            working["attempt_index"] = int(basis["successful_attempt_index"])
        elif receipt_path.is_file():
            _failure_payload(receipt_path)
            if not retry_failed:
                raise RuntimeError(
                    f"{run_id} has a preserved failure; an explicit single engineering retry is required"
                )
            if len(all_receipts) != ENGINEERING_RETRIES_PER_IDENTITY:
                raise RuntimeError(f"H7 engineering retry limit reached for {run_id}")
            working["attempt_index"] = planned_attempt + 1
        elif retry_failed and all_receipts:
            raise RuntimeError(f"retry lineage mismatch for {run_id}")
        parent = _parent_root(
            str(working["parent_source"]),
            h5_result_root=h5_result_root,
            h6_result_root=h6_result_root,
        )
        relative = safe_relative(str(working["parent_attempt_relative_path"]))
        parent_attempt = parent.joinpath(*relative.parts)
        try:
            marker = run_plan_row(
                plan_row=working,
                repo_root=repo_root,
                cache_root=cache_root,
                result_root=h7_result_root,
                parent_attempt_root=parent_attempt,
                enforce_target_gpu=enforce_target_gpu,
            )
            if int(working["attempt_index"]) != planned_attempt:
                failed_value, _ = _failure_payload(receipt_path)
                resolution_basis = {
                    "physical_run_id": run_id,
                    "stage": stage,
                    "failed_attempt_index": planned_attempt,
                    "successful_attempt_index": int(working["attempt_index"]),
                    "failed_attempt_receipt_sha256": sha256_file(receipt_path),
                    "failed_attempt_identity": failed_value["identity"],
                    "successful_stage_marker_identity": marker["identity"],
                    "scientific_metrics_observed_before_retry": False,
                    "engineering_retry_count": ENGINEERING_RETRIES_PER_IDENTITY,
                }
                write_json_once(
                    resolution_path,
                    envelope(
                        "metachorus.v53.m4a-h7.retry-resolution.v1",
                        "ENGINEERING_RETRY_RECOVERED",
                        resolution_basis,
                    ),
                )
            completed += 1
        except Exception as exc:
            attempt_index = int(working.get("attempt_index", 0))
            failure_path = failed_root / f"{run_id}__attempt_{attempt_index:02d}.json"
            signature = _root_cause_signature(exc)
            failure_basis = {
                "physical_run_id": run_id,
                "stage": stage,
                "attempt_index": attempt_index,
                **signature,
                "root_cause_identity": canonical_json_hash(signature),
                "scientific_metrics_observed": False,
                "optimizer_steps_after_boundary": 0,
                "engineering_retry_limit": ENGINEERING_RETRIES_PER_IDENTITY,
                "labels_mounted": False,
                "h5_locked_report_access": False,
                "s_eval_started": False,
            }
            write_json_once(
                failure_path,
                envelope(
                    "metachorus.v53.m4a-h7.failed-attempt.v1",
                    "FAILED_PRESERVED",
                    failure_basis,
                ),
            )
            failures.append(run_id)
            root_causes = _root_cause_counts(h7_result_root)
            affected = root_causes[failure_basis["root_cause_identity"]]
            if stage == "R4" and len(affected) >= COMMON_ROOT_CAUSE_PAUSE_THRESHOLD:
                pause_basis = {
                    "root_cause_identity": failure_basis["root_cause_identity"],
                    "affected_physical_run_ids": sorted(affected),
                    "affected_scientific_identity_count": len(affected),
                    "threshold": COMMON_ROOT_CAUSE_PAUSE_THRESHOLD,
                    "fanout_must_stop": True,
                    "completed_runs_preserved": True,
                    "h5_locked_report_access": False,
                    "s_eval_started": False,
                }
                write_json_once(
                    h7_result_root / "M4A_H7_COMMON_ROOT_CAUSE_PAUSE.json",
                    envelope(
                        "metachorus.v53.m4a-h7.common-root-cause-pause.v1",
                        "IMMEDIATE_COMMON_ROOT_CAUSE_PAUSE",
                        pause_basis,
                    ),
                )
                raise RuntimeError("H7 immediate common-root-cause pause") from exc
    summary = {
        "stage": stage,
        "shard_index": int(shard_index),
        "shard_count": int(shard_count),
        "planned_on_shard": len(rows),
        "completed_training": completed,
        "failed": len(failures),
        "failed_run_ids": failures,
        "status": "PASS" if not failures else "FAIL",
    }
    if failures:
        raise RuntimeError(f"H7 {stage} shard failures: {failures[:3]}")
    return summary


def _canonical_attempts(run_parent: Path, marker_name: str) -> list[Path]:
    allowed = {"attempt_00", "attempt_01"}
    observed = [path for path in sorted(run_parent.glob("attempt_*")) if path.is_dir()]
    if any(path.name not in allowed for path in observed) or len(observed) > 2:
        raise RuntimeError(f"invalid H7 attempt directory set: {run_parent.name}")
    return [path for path in observed if (path / marker_name).is_file()]


def close_canary(*, result_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    plan_path = result_root / "M4A_H7_CANARY_RUN_PLAN.csv"
    rows = read_csv(plan_path)
    validate_plan_seal(plan_path=plan_path, result_root=result_root, stage="CANARY", rows=rows)
    validate_canary_plan_rows(rows)
    records: list[dict[str, Any]] = []
    marker_name = f"H7_CANARY_{CANARY_ENDPOINT:06d}_COMPLETE.json"
    for row in rows:
        run_parent = result_root / "canary_runs" / str(row["physical_run_id"])
        attempts = _canonical_attempts(run_parent, marker_name)
        if len(attempts) != 1:
            raise RuntimeError(f"H7 canary requires one completed attempt: {row['physical_run_id']}")
        attempt = attempts[0]
        marker = validate_stage_marker(
            attempt / marker_name,
            plan_row=row,
            endpoint_update=CANARY_ENDPOINT,
            run_root=attempt,
        )
        checkpoint = attempt / "checkpoints" / f"update_{CANARY_ENDPOINT:06d}.pt"
        representation = attempt / "representations" / f"update_{CANARY_ENDPOINT:06d}.npy"
        audit = checkpoint_state_audit(
            checkpoint,
            expected_update=CANARY_ENDPOINT,
            expected_sha256=str(marker["checkpoint_sha256"]),
        )
        array = np.load(representation, allow_pickle=False)
        with (attempt / "RESOURCE_TRACE.csv").open(
            "r", encoding="utf-8-sig", newline=""
        ) as stream:
            trace = list(csv.DictReader(stream))
        if (
            not audit["strict_state_complete"]
            or not trace
            or int(trace[-1]["update"]) != CANARY_ENDPOINT
            or array.ndim != 2
            or array.size == 0
            or not np.isfinite(array).all()
            or int(marker["optimizer_steps"]) != CANARY_ENDPOINT
            or int(marker["optimizer_steps_this_invocation"])
            != CANARY_ENDPOINT - PARENT_UPDATE
            or marker["eligible_for_scientific_result"] is not False
            or marker["eligible_for_selector_development"] is not False
            or marker["reusable_for_science"] is not False
            or marker["labels_mounted"] is not False
            or int(marker["downstream_evaluations"]) != 0
        ):
            raise RuntimeError(f"H7 canary contract failed: {row['physical_run_id']}")
        records.append(
            {
                "physical_run_id": str(row["physical_run_id"]),
                "dataset": str(row["dataset"]),
                "candidate_set_hash": str(row["candidate_set_hash"]),
                "model_seed": int(row["model_seed"]),
                "successful_attempt_index": int(attempt.name.rsplit("_", 1)[-1]),
                "optimizer_steps": CANARY_ENDPOINT,
                "optimizer_steps_this_invocation": CANARY_ENDPOINT - PARENT_UPDATE,
                "model_total_seconds": float(marker["model_total_seconds"]),
                "model_seconds_this_invocation": float(marker["model_seconds_this_invocation"]),
                "gpu_preference": int(row["gpu_preference"]),
                "checkpoint_sha256": str(marker["checkpoint_sha256"]),
                "representation_sha256": str(marker["representation_sha256"]),
                "resource_trace_sha256": sha256_file(attempt / "RESOURCE_TRACE.csv"),
                "stage_marker_identity": str(marker["identity"]),
                "checkpoint_roundtrip_state_complete": True,
                "representation_finite": True,
                "scientific_metrics_observed": False,
                "reusable_for_science": False,
                "labels_mounted": False,
                "status": "PASS",
            }
        )
    results_basis = {
        "run_count": 2,
        "records": sorted(records, key=lambda row: row["dataset"]),
        "all_optimizer_steps_exact": True,
        "all_checkpoint_roundtrips_pass": True,
        "scientific_metrics_observed": False,
        "fanout_started_before_canary_pass": False,
        "labels_mounted": False,
        "h5_locked_report_access": False,
        "s_eval_started": False,
    }
    results = envelope("metachorus.v53.m4a-h7.canary-results.v1", "PASS", results_basis)
    write_json_once(result_root / "M4A_H7_CANARY_RESULTS.json", results)
    decision_basis = {
        "canary_results_identity": results["identity"],
        "acm_pass": True,
        "freebase_pass": True,
        "fanout_authorized": True,
        "fanout_started_before_decision": False,
        "scientific_metrics_observed": False,
        "labels_mounted": False,
        "h5_locked_report_access": False,
        "s_eval_started": False,
    }
    decision = envelope(
        "metachorus.v53.m4a-h7.canary-gate-decision.v1",
        "H7_CANARY_PASS",
        decision_basis,
    )
    write_json_once(result_root / "M4A_H7_CANARY_GATE_DECISION.json", decision)
    append_jsonl_once(
        result_root / "M4A_H7_EXECUTION_LOG.jsonl",
        {
            "event_id": "H7_CANARY_GATE_PASS",
            "status": "H7_CANARY_PASS",
            "canary_results_identity": results["identity"],
            "fanout_authorized": True,
            "h5_locked_report_access": False,
            "s_eval_started": False,
        },
    )
    return results, decision


def authorize_r4(*, result_root: Path) -> dict[str, Any]:
    """Close the two engineering canaries, then publish the sealed 63-run R4 plan."""

    results, decision = close_canary(result_root=result_root)
    path_rows = read_csv(result_root / "M4A_H7_PATH_SET_RESOLUTION_AUDIT.csv")
    parent_rows = read_csv(
        result_root / "TC_LPMSO_H7_R4_PARENT_STATE_INVENTORY.csv"
    )
    rows, seal = build_r4_plan(
        result_root=result_root,
        path_audit_rows=path_rows,
        parent_inventory_rows=parent_rows,
    )
    plan_path = write_csv_once(
        result_root / "TC_LPMSO_H7_R4_RUN_PLAN.csv",
        rows,
        fieldnames=PLAN_FIELDS,
    )
    write_json_once(result_root / "TC_LPMSO_H7_R4_PLAN_SEAL.json", seal)
    append_jsonl_once(
        result_root / "M4A_H7_EXECUTION_LOG.jsonl",
        {
            "event_id": "H7_R4_PLAN_SEALED_AFTER_CANARY_PASS",
            "status": "SEALED_BEFORE_GPU_TRAINING",
            "canary_results_identity": results["identity"],
            "canary_gate_identity": decision["identity"],
            "r4_plan_sha256": sha256_file(plan_path),
            "r4_plan_seal_identity": seal["identity"],
            "r4_run_count": len(rows),
            "labels_mounted": False,
            "h5_locked_report_access": False,
            "s_eval_started": False,
        },
    )
    return {"rows": rows, "seal": seal, "canary": results, "decision": decision}


def close_stage_registry(
    *, stage: str, plan_path: Path, result_root: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if stage != "R4":
        raise ValueError(stage)
    rows = read_csv(plan_path)
    validate_plan_seal(plan_path=plan_path, result_root=result_root, stage=stage, rows=rows)
    validate_r4_plan_rows(rows)
    run_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    marker_name = f"H7_STAGE_{R4_ENDPOINT:06d}_COMPLETE.json"
    for row in rows:
        run_parent = result_root / "runs" / str(row["physical_run_id"])
        attempts = _canonical_attempts(run_parent, marker_name)
        if len(attempts) != 1:
            raise RuntimeError(
                f"H7 R4 requires exactly one completed attempt for "
                f"{row['physical_run_id']}, observed {len(attempts)}"
            )
        attempt = attempts[0]
        marker = validate_stage_marker(
            attempt / marker_name,
            plan_row=row,
            endpoint_update=R4_ENDPOINT,
            run_root=attempt,
        )
        checkpoint = attempt / "checkpoints" / f"update_{R4_ENDPOINT:06d}.pt"
        representation = attempt / "representations" / f"update_{R4_ENDPOINT:06d}.npy"
        strict = checkpoint_state_audit(
            checkpoint,
            expected_update=R4_ENDPOINT,
            expected_sha256=str(marker["checkpoint_sha256"]),
        )
        array = np.load(representation, allow_pickle=False)
        if (
            not strict["strict_state_complete"]
            or array.ndim != 2
            or array.size == 0
            or not np.isfinite(array).all()
        ):
            raise RuntimeError(f"H7 R4 endpoint state invalid: {row['physical_run_id']}")
        common = {
            "physical_run_id": str(row["physical_run_id"]),
            "dataset": str(row["dataset"]),
            "candidate_set_id": str(row["candidate_set_id"]),
            "candidate_set_hash": str(row["candidate_set_hash"]),
            "candidate_set_ordinal": int(row["candidate_set_ordinal"]),
            "model_seed": int(row["model_seed"]),
            "endpoint_update": R4_ENDPOINT,
            "successful_attempt_index": int(attempt.name.rsplit("_", 1)[-1]),
        }
        run_rows.append(
            {
                **common,
                "continuation_mode": str(marker["continuation_mode"]),
                "optimizer_steps": int(marker["optimizer_steps"]),
                "optimizer_steps_this_invocation": int(marker["optimizer_steps_this_invocation"]),
                "model_total_seconds": float(marker["model_total_seconds"]),
                "model_seconds_this_invocation": float(marker["model_seconds_this_invocation"]),
                "peak_gpu_allocated_bytes": int(marker["peak_gpu_allocated_bytes"]),
                "peak_gpu_reserved_bytes": int(marker["peak_gpu_reserved_bytes"]),
                "training_reexecuted": int(attempt.name.rsplit("_", 1)[-1]) > 0,
                "labels_mounted": False,
                "status": "PASS",
            }
        )
        checkpoint_rows.append(
            {
                **common,
                "source_root": "H7_RESULT_ROOT",
                "checkpoint_relative_path": checkpoint.relative_to(result_root).as_posix(),
                "checkpoint_sha256": sha256_file(checkpoint),
                "checkpoint_size_bytes": checkpoint.stat().st_size,
                "representation_relative_path": representation.relative_to(result_root).as_posix(),
                "representation_sha256": sha256_file(representation),
                "representation_size_bytes": representation.stat().st_size,
                "strict_state_complete": True,
                "status": "PASS",
            }
        )
    if len(run_rows) != EXPECTED_R4_RUNS:
        raise RuntimeError("H7 R4 registry count mismatch")
    prefix = STAGE_PREFIX[stage]
    run_path = result_root / f"{prefix}_RUN_REGISTRY.csv"
    checkpoint_path = result_root / f"{prefix}_CHECKPOINT_REGISTRY.csv"
    write_csv_once(run_path, run_rows, fieldnames=list(run_rows[0]))
    write_csv_once(checkpoint_path, checkpoint_rows, fieldnames=list(checkpoint_rows[0]))
    plan_seal_path = _plan_seal_path(result_root, stage)
    seal_basis = {
        "stage": stage,
        "endpoint_update": R4_ENDPOINT,
        "plan_sha256": sha256_file(plan_path),
        "plan_seal_sha256": sha256_file(plan_seal_path),
        "run_registry_sha256": sha256_file(run_path),
        "checkpoint_registry_sha256": sha256_file(checkpoint_path),
        "run_count": len(run_rows),
        "checkpoint_count": len(checkpoint_rows),
        "all_status_pass": True,
        "labels_mounted": False,
        "h5_locked_report_access": False,
        "s_eval_started": False,
    }
    write_json_once(
        result_root / f"{prefix}_REGISTRY_SEAL.json",
        envelope(
            "metachorus.v53.m4a-h7.stage-registry-seal.v1",
            "SEALED_BEFORE_STAGE_EVALUATION",
            seal_basis,
        ),
    )
    return run_rows, checkpoint_rows


def _resource_summary(
    *, result_root: Path, run_rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in run_rows:
        run_root = result_root / "runs" / str(row["physical_run_id"])
        size_bytes = sum(
            path.stat().st_size
            for path in run_root.rglob("*")
            if path.is_file() and not path.is_symlink()
        )
        output.append(
            {
                "physical_run_id": str(row["physical_run_id"]),
                "dataset": str(row["dataset"]),
                "candidate_set_hash": str(row["candidate_set_hash"]),
                "model_seed": int(row["model_seed"]),
                "model_seconds_this_invocation": float(row["model_seconds_this_invocation"]),
                "gpu_hours_this_invocation": float(row["model_seconds_this_invocation"]) / 3600.0,
                "storage_bytes": size_bytes,
                "failed_attempts": int(bool(row["training_reexecuted"])),
                "labels_mounted": False,
                "h5_locked_report_access": False,
                "s_eval_started": False,
                "status": "PASS",
            }
        )
    return output


def _manual_automatic_comparison(
    objectives: Sequence[Mapping[str, Any]],
    selection: Mapping[str, Any],
) -> list[dict[str, Any]]:
    by_identity = {
        (str(row["dataset"]), str(row["candidate_set_hash"])): dict(row)
        for row in objectives
    }
    selected_by_dataset = {
        str(row["dataset"]): dict(row) for row in selection.get("selected", [])
    }
    output: list[dict[str, Any]] = []
    for dataset in ("acm", "freebase"):
        selected = selected_by_dataset.get(dataset)
        if selected is None:
            continue
        for role in ("selected", "manual", "current_automatic"):
            if role == "selected":
                candidate_hash = str(selected["candidate_set_hash"])
            else:
                candidate_hash = CONTROLS[dataset][role][1]
            row = by_identity.get((dataset, candidate_hash))
            if row is None:
                raise RuntimeError(f"missing H7 R4 comparison identity: {dataset}/{role}")
            output.append(
                {
                    "dataset": dataset,
                    "role": role.upper(),
                    "candidate_set_id": str(row["candidate_set_id"]),
                    "candidate_set_hash": candidate_hash,
                    "Q_cls_R4": float(row["Q_cls_R4"]),
                    "Q_clu_R4": float(row["Q_clu_R4"]),
                    "W_cls_R4": float(row["W_cls_R4"]),
                    "W_clu_R4": float(row["W_clu_R4"]),
                    "is_final_selection": candidate_hash
                    == str(selected["candidate_set_hash"]),
                }
            )
    return output


def _write_s_eval_pending(
    *, result_root: Path, selection: Mapping[str, Any]
) -> None:
    if selection.get("status") != "TC_LPMSO_H7_NEW_SET_SELECTED":
        return
    selected_by_dataset = {
        str(row["dataset"]): dict(row) for row in selection.get("selected", [])
    }
    matrix: list[dict[str, Any]] = []
    for dataset in ("acm", "freebase"):
        selected = selected_by_dataset[dataset]
        identities = [
            ("H7_SELECTED", str(selected["candidate_set_id"]), str(selected["candidate_set_hash"])),
            ("MANUAL", *CONTROLS[dataset]["manual"]),
            ("CURRENT_AUTOMATIC", *CONTROLS[dataset]["current_automatic"]),
        ]
        deduplicated: dict[str, tuple[str, str]] = {}
        for role, candidate_id, candidate_hash in identities:
            previous = deduplicated.get(candidate_hash)
            if previous is None or role == "H7_SELECTED":
                deduplicated[candidate_hash] = (role, candidate_id)
        for candidate_hash, (role, candidate_id) in sorted(deduplicated.items()):
            for seed in FUTURE_S_EVAL_MODEL_SEEDS:
                matrix.append(
                    {
                        "dataset": dataset,
                        "role": role,
                        "candidate_set_id": candidate_id,
                        "candidate_set_hash": candidate_hash,
                        "model_seed": seed,
                        "continuation_from_update": 0,
                        "endpoint_update": 3000,
                        "trainer": "O0-S0",
                        "training_labels_visible": False,
                        "training_started": False,
                        "status": "PENDING_USER",
                    }
                )
    write_csv_once(
        result_root / "TC_LPMSO_H7_S_EVAL_CONFIRMATION_MATRIX_PENDING.csv",
        matrix,
        fieldnames=list(matrix[0]),
    )
    write_text_once(
        result_root / "TC_LPMSO_H7_S_EVAL_CONFIRMATION_PLAN_PENDING.md",
        "# H7 S_eval confirmation plan (PENDING_USER)\n\n"
        "This file is a preregistration only. No S_eval seed has been used and no S_eval run "
        "has started. Each deduplicated H7-selected, Manual, and Current Automatic complete "
        "path set is planned from step 0 to 3000 with O0-S0 and labels hidden from training.\n\n"
        "Classification will fit the frozen LogisticRegression protocol on GLOBAL_SEARCH_POOL "
        "and evaluate only on H5_LOCKED_REPORT after approval. Clustering will fit without "
        "labels and compute NMI/ARI on H5_LOCKED_REPORT after approval, using the frozen 20 "
        "KMeans seeds and n_init=10. Reports will include mean/SD, paired differences, paired "
        "t intervals, and a preregistered 10,000-resample paired bootstrap interval.\n",
    )
    budget_basis = {
        "status": "PENDING_USER",
        "matrix_row_count": len(matrix),
        "model_seed_count": len(FUTURE_S_EVAL_MODEL_SEEDS),
        "model_seeds": list(FUTURE_S_EVAL_MODEL_SEEDS),
        "endpoint_update": 3000,
        "training_started": False,
        "h5_locked_report_access": False,
        "s_eval_started": False,
    }
    write_json_once(
        result_root / "TC_LPMSO_H7_S_EVAL_CONFIRMATION_BUDGET_PENDING.json",
        envelope(
            "metachorus.v53.m4a-h7.s-eval-budget-pending.v1",
            "PENDING_USER",
            budget_basis,
        ),
    )
    approval_basis = {
        "selection_status": selection["status"],
        "selected_path_sets": [
            {
                "dataset": row["dataset"],
                "candidate_set_id": row["candidate_set_id"],
                "candidate_set_hash": row["candidate_set_hash"],
            }
            for row in selection["selected"]
        ],
        "future_seed_count": len(FUTURE_S_EVAL_MODEL_SEEDS),
        "final_evaluator_roles_registered": True,
        "training_started": False,
        "h5_locked_report_access": False,
        "s_eval_started": False,
    }
    write_json_once(
        result_root / "TC_LPMSO_H7_S_EVAL_APPROVAL_STATUS.json",
        envelope(
            "metachorus.v53.m4a-h7.s-eval-approval-status.v1",
            "PENDING_USER",
            approval_basis,
        ),
    )


def finalize_r4_metrics(
    *,
    result_root: Path,
    search_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    objectives = aggregate_r4_objectives(search_rows)
    safety = manual_safety_audit(objectives)
    selection = robust_r4_selection(objectives, safety)
    write_csv_once(
        result_root / "TC_LPMSO_H7_R4_MANUAL_SAFETY_AUDIT.csv",
        safety,
        fieldnames=list(safety[0]),
    )
    write_csv_once(
        result_root / "TC_LPMSO_H7_R4_ROBUST_PARETO_FRONTS.csv",
        selection["fronts"],
        fieldnames=list(selection["fronts"][0]) if selection["fronts"] else ["dataset"],
    )
    write_csv_once(
        result_root / "TC_LPMSO_H7_R4_ROBUST_RANKING.csv",
        selection["rankings"],
        fieldnames=list(selection["rankings"][0]) if selection["rankings"] else ["dataset"],
    )
    decision_basis = {
        "selected_path_sets": [
            {
                "dataset": row["dataset"],
                "candidate_set_id": row["candidate_set_id"],
                "candidate_set_hash": row["candidate_set_hash"],
                "candidate_set_ordinal": int(row["candidate_set_ordinal"]),
            }
            for row in selection["selected"]
        ],
        "selected_count": len(selection["selected"]),
        "unsafe_datasets": list(selection.get("unsafe_datasets", [])),
        "manual_safety_required": True,
        "four_objective_pareto_required": True,
        "single_complete_set_per_dataset": len(selection["selected"]) == 2,
        "taskwise_set_splicing": False,
        "h5_locked_report_access": False,
        "s_eval_started": False,
    }
    decision = envelope(
        "metachorus.v53.m4a-h7.r4-development-selection.v1",
        str(selection["status"]),
        decision_basis,
    )
    write_json_once(result_root / "TC_LPMSO_H7_R4_DEVELOPMENT_SELECTION.json", decision)
    selected_registry = envelope(
        "metachorus.v53.m4a-h7.selected-path-set-registry.v1",
        str(selection["status"]),
        {
            "records": decision_basis["selected_path_sets"],
            "record_count": len(decision_basis["selected_path_sets"]),
            "selection_identity": decision["identity"],
            "complete_sets_only": True,
            "taskwise_set_splicing": False,
            "h5_locked_report_access": False,
            "s_eval_started": False,
        },
    )
    write_json_once(result_root / "TC_LPMSO_H7_SELECTED_PATH_SET_REGISTRY.json", selected_registry)
    comparison = _manual_automatic_comparison(objectives, selection)
    write_csv_once(
        result_root / "TC_LPMSO_H7_MANUAL_AUTOMATIC_COMPARISON.csv",
        comparison,
        fieldnames=list(comparison[0]) if comparison else ["dataset"],
    )
    if selection["status"] == "TC_LPMSO_H7_NO_SAFE_SET":
        write_failure(
            result_root,
            failure_code="TC_LPMSO_H7_NO_SAFE_SET",
            evidence_artifact="TC_LPMSO_H7_R4_DEVELOPMENT_SELECTION.json",
            evidence_sha256=sha256_file(
                result_root / "TC_LPMSO_H7_R4_DEVELOPMENT_SELECTION.json"
            ),
            counts={"selected_count": 0, "r4_search_row_count": len(search_rows)},
            scientific_gate_failure=True,
            engineering_retry_permitted=False,
        )
    else:
        _write_s_eval_pending(result_root=result_root, selection=selection)
    return {"objectives": objectives, "safety": safety, "selection": selection}


def evaluate_r4_and_decide(
    *,
    result_root: Path,
    h5_result_root: Path,
    private_mount_root: Path,
    workers: int = 1,
) -> dict[str, Any]:
    plan_path = result_root / "TC_LPMSO_H7_R4_RUN_PLAN.csv"
    run_rows, _ = close_stage_registry(stage="R4", plan_path=plan_path, result_root=result_root)
    resource_rows = _resource_summary(result_root=result_root, run_rows=run_rows)
    write_csv_once(
        result_root / "TC_LPMSO_H7_R4_RESOURCE_SUMMARY.csv",
        resource_rows,
        fieldnames=list(resource_rows[0]),
    )
    metrics = run_stage_evaluation(
        plan_path=plan_path,
        result_root=result_root,
        h5_result_root=h5_result_root,
        private_mount_root=private_mount_root,
        workers=workers,
    )
    outcome = finalize_r4_metrics(result_root=result_root, search_rows=metrics["search"])
    append_jsonl_once(
        result_root / "M4A_H7_EXECUTION_LOG.jsonl",
        {
            "event_id": "H7_R4_EVALUATION_AND_SELECTION",
            "status": outcome["selection"]["status"],
            "r4_run_count": len(run_rows),
            "selected_count": len(outcome["selection"]["selected"]),
            "h5_locked_report_access": False,
            "s_eval_started": False,
        },
    )
    return {**outcome, "metrics": metrics, "resource_rows": resource_rows}
