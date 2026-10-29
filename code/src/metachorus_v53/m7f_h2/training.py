"""Four-worker dynamic queue over the frozen M6F/M7F O0 executor."""

from __future__ import annotations

import json
import os
import socket
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from metachorus_v53.audit.common import canonical_json_hash
from metachorus_v53.m6f_m7f_fast.training import run_plan_row as run_frozen_plan_row

from .common import envelope, read_json, sha256_file, write_json_once
from .constants import (
    COMMON_ROOT_CAUSE_PAUSE_THRESHOLD,
    EXPECTED_GPU_COUNT,
    MAX_ATTEMPTS_PER_IDENTITY,
)
from .planning import STAGE_STATUS, canonical_plan_row, validate_plan


ENGINE_STAGE = {
    "CANARY": "CANARY",
    "R1": "SEARCH",
    "R2": "SEARCH",
    "R3": "SEARCH",
    "R4": "SEARCH",
    "PRELIMINARY": "PRELIMINARY",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def h2_marker_name(logical_stage: str) -> str:
    if logical_stage not in ENGINE_STAGE:
        raise ValueError(f"unsupported H2 stage: {logical_stage}")
    return f"AMINER_H2_{logical_stage}_COMPLETE.json"


def _branch(logical_stage: str) -> str:
    if logical_stage == "CANARY":
        return "canary_runs"
    if logical_stage == "PRELIMINARY":
        return "preliminary_runs"
    return "search_runs"


def attempt_root(result_root: str | Path, row: Mapping[str, Any], logical_stage: str, attempt: int) -> Path:
    if int(attempt) not in range(MAX_ATTEMPTS_PER_IDENTITY):
        raise ValueError("H2 permits only attempt 0 and one engineering retry")
    return (
        Path(result_root).resolve()
        / _branch(logical_stage)
        / str(row["physical_run_id"])
        / f"attempt_{int(attempt):02d}"
    )


def completed_marker(
    result_root: str | Path, row: Mapping[str, Any], logical_stage: str
) -> tuple[Path, dict[str, Any]] | None:
    parent = Path(result_root).resolve() / _branch(logical_stage) / str(row["physical_run_id"])
    matches = [
        attempt / h2_marker_name(logical_stage)
        for attempt in sorted(parent.glob("attempt_*"))
        if (attempt / h2_marker_name(logical_stage)).is_file()
        and not (attempt / h2_marker_name(logical_stage)).is_symlink()
    ]
    if len(matches) > 1:
        raise RuntimeError(f"multiple H2 completion markers: {row['physical_run_id']} {logical_stage}")
    if not matches:
        return None
    marker = read_json(matches[0])
    basis = marker.get("basis", {})
    if (
        marker.get("schema_version") != "metachorus.v53.m7f-h2.run-complete.v1"
        or marker.get("status") != "PASS"
        or marker.get("identity") != canonical_json_hash(basis)
        or basis.get("logical_stage") != logical_stage
        or basis.get("physical_run_id") != row.get("physical_run_id")
        or basis.get("plan_seal_identity") != row.get("plan_seal_identity")
        or basis.get("labels_mounted_during_training") is not False
        or basis.get("one_process_per_gpu") is not True
        or basis.get("ddp_used") is not False
        or basis.get("data_parallel_used") is not False
        or basis.get("torchrun_used") is not False
    ):
        raise ValueError("H2 completion marker identity is invalid")
    return matches[0], marker


def _gpu_runtime(*, enforce_target_gpu: bool) -> dict[str, Any]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    uuid = os.environ.get("AMINER_H2_GPU_UUID", "")
    pci_bus_id = os.environ.get("AMINER_H2_GPU_PCI_BUS_ID", "")
    runtime_domain = os.environ.get("AMINER_H2_RUNTIME_DOMAIN", socket.gethostname())
    if enforce_target_gpu:
        if torch.cuda.device_count() != 1 or torch.cuda.current_device() != 0:
            raise RuntimeError("H2 worker must see exactly one local CUDA device 0")
        name = torch.cuda.get_device_name(0)
        if not uuid or not pci_bus_id or not visible:
            raise RuntimeError("H2 worker lacks frozen GPU UUID/PCI/visibility metadata")
    else:
        name = os.environ.get("AMINER_H2_GPU_NAME", "CPU_TEST_DOUBLE")
        uuid = uuid or "GPU-CPU-TEST-DOUBLE"
        pci_bus_id = pci_bus_id or "0000:00:00.0"
        visible = visible or "CPU_TEST_DOUBLE"
    return {
        "gpu_uuid": uuid,
        "gpu_pci_bus_id": pci_bus_id,
        "gpu_name": name,
        "cuda_visible_devices": visible,
        "runtime_domain": runtime_domain,
        "torch_cuda_device_count_inside_worker": int(torch.cuda.device_count()) if enforce_target_gpu else 1,
        "torch_current_local_device": int(torch.cuda.current_device()) if enforce_target_gpu else 0,
    }


def run_h2_row(
    *,
    plan_row: Mapping[str, Any],
    logical_stage: str,
    worker_index: int,
    repo_root: str | Path,
    cache_root: str | Path,
    result_root: str | Path,
    source_commit: str,
    source_tree: str,
    enforce_target_gpu: bool = True,
) -> dict[str, Any]:
    row = canonical_plan_row(plan_row)
    if logical_stage not in ENGINE_STAGE or row.get("logical_stage") != logical_stage:
        raise ValueError("H2 row/logical stage mismatch")
    attempt = int(row.get("attempt_index", 0))
    if attempt not in range(MAX_ATTEMPTS_PER_IDENTITY):
        raise ValueError("H2 permits at most one engineering retry per scientific identity")
    prior = completed_marker(result_root, row, logical_stage)
    if prior is not None:
        return prior[1]
    started_at = utc_now()
    gpu = _gpu_runtime(enforce_target_gpu=enforce_target_gpu)
    engine_marker = run_frozen_plan_row(
        plan_row=row,
        stage=ENGINE_STAGE[logical_stage],
        repo_root=repo_root,
        cache_root=cache_root,
        result_root=result_root,
        source_commit=source_commit,
        source_tree=source_tree,
        enforce_target_gpu=enforce_target_gpu,
    )
    root = attempt_root(result_root, row, logical_stage, attempt)
    engine_name = {
        "CANARY": f"M6F_M7F_CANARY_{int(row['endpoint_update']):06d}_COMPLETE.json",
        "SEARCH": f"M6F_M7F_SEARCH_{int(row['endpoint_update']):06d}_COMPLETE.json",
        "PRELIMINARY": f"M6F_M7F_PRELIMINARY_{int(row['endpoint_update']):06d}_COMPLETE.json",
    }[ENGINE_STAGE[logical_stage]]
    engine_path = root / engine_name
    if engine_path.is_symlink() or not engine_path.is_file() or engine_marker.get("status") != "PASS":
        raise RuntimeError("frozen engine did not publish its governed completion marker")
    basis = {
        "logical_stage": logical_stage,
        "engine_stage": ENGINE_STAGE[logical_stage],
        "physical_run_id": row["physical_run_id"],
        "dataset": row["dataset"],
        "candidate_set_id": row["candidate_set_id"],
        "candidate_set_hash": row["candidate_set_hash"],
        "candidate_set_ordinal": int(row["candidate_set_ordinal"]),
        "path_ids": list(row["path_ids"]),
        "model_seed": int(row["model_seed"]),
        "attempt_index": attempt,
        "continuation_from_update": int(row.get("continuation_from_update", 0)),
        "endpoint_update": int(row["endpoint_update"]),
        "plan_seal_identity": row["plan_seal_identity"],
        "engine_marker_identity": engine_marker["identity"],
        "engine_marker_relative_path": engine_path.resolve(strict=True).relative_to(
            Path(result_root).resolve()
        ).as_posix(),
        "engine_marker_sha256": sha256_file(engine_path),
        "worker_index": int(worker_index),
        **gpu,
        "started_at": started_at,
        "ended_at": utc_now(),
        "one_process_per_gpu": True,
        "ddp_used": False,
        "data_parallel_used": False,
        "torchrun_used": False,
        "labels_mounted_during_training": False,
        "downstream_evaluations_during_training": 0,
    }
    marker = envelope("metachorus.v53.m7f-h2.run-complete.v1", "PASS", basis)
    write_json_once(root / h2_marker_name(logical_stage), marker)
    return marker


def _atomic_claim(path: Path, payload: Mapping[str, Any]) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return False
    try:
        data = (json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        os.write(descriptor, data)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return True


def _failure_fingerprint(exc: BaseException) -> str:
    return canonical_json_hash(
        {
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "last_traceback_line": traceback.format_exc().strip().splitlines()[-1],
        }
    )


def _failures(queue_root: Path) -> list[dict[str, Any]]:
    values = []
    for path in sorted((queue_root / "failures").glob("*.json")):
        if path.is_symlink() or not path.is_file():
            raise RuntimeError("H2 failure registry contains a non-regular member")
        values.append(read_json(path))
    return values


def _validated_failure(
    queue_root: Path,
    row: Mapping[str, Any],
    *,
    logical_stage: str,
    plan_seal_identity: str,
    attempt: int,
) -> dict[str, Any] | None:
    path = queue_root / "failures" / f"{row['physical_run_id']}__attempt_{attempt:02d}.json"
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("H2 failure evidence is not a regular file")
    failure = read_json(path)
    basis = failure.get("basis", {})
    if (
        failure.get("schema_version") != "metachorus.v53.m7f-h2.run-failure.v1"
        or failure.get("status") != "FAILED_PRESERVED"
        or failure.get("identity") != canonical_json_hash(basis)
        or basis.get("logical_stage") != logical_stage
        or basis.get("physical_run_id") != row.get("physical_run_id")
        or int(basis.get("attempt_index", -1)) != int(attempt)
        or basis.get("plan_seal_identity") != plan_seal_identity
        or basis.get("scientific_metric_triggered_retry") is not False
        or basis.get("retry_allowed") != (attempt + 1 < MAX_ATTEMPTS_PER_IDENTITY)
    ):
        raise ValueError("H2 failure evidence identity is invalid")
    return failure


def _execution_row_for_retry(
    queue_root: Path,
    row: Mapping[str, Any],
    *,
    logical_stage: str,
    plan_seal_identity: str,
) -> dict[str, Any] | None:
    value = dict(row)
    attempt = int(value.get("attempt_index", 0))
    failure = _validated_failure(
        queue_root,
        value,
        logical_stage=logical_stage,
        plan_seal_identity=plan_seal_identity,
        attempt=attempt,
    )
    if failure is None:
        return value
    next_attempt = attempt + 1
    if next_attempt >= MAX_ATTEMPTS_PER_IDENTITY:
        return None
    if _validated_failure(
        queue_root,
        value,
        logical_stage=logical_stage,
        plan_seal_identity=plan_seal_identity,
        attempt=next_attempt,
    ) is not None:
        return None
    value["attempt_index"] = next_attempt
    value["engineering_retry_of_attempt"] = attempt
    return value


def _common_root_pause(queue_root: Path) -> dict[str, Any] | None:
    grouped: dict[str, set[str]] = {}
    for failure in _failures(queue_root):
        basis = failure.get("basis", {})
        grouped.setdefault(str(basis.get("error_fingerprint", "")), set()).add(
            str(basis.get("physical_run_id", ""))
        )
    offenders = sorted(
        fingerprint
        for fingerprint, identities in grouped.items()
        if fingerprint and len(identities) >= COMMON_ROOT_CAUSE_PAUSE_THRESHOLD
    )
    if not offenders:
        return None
    basis = {
        "status": "COMMON_ROOT_CAUSE_PAUSE",
        "error_fingerprints": offenders,
        "threshold": COMMON_ROOT_CAUSE_PAUSE_THRESHOLD,
        "new_claims_allowed": False,
    }
    pause = envelope("metachorus.v53.m7f-h2.common-root-pause.v1", "PAUSED", basis)
    write_json_once(queue_root / "AMINER_H2_COMMON_ROOT_CAUSE_PAUSE.json", pause)
    return pause


def run_dynamic_worker(
    *,
    rows: Sequence[Mapping[str, Any]],
    seal: Mapping[str, Any],
    logical_stage: str,
    worker_index: int,
    worker_count: int,
    repo_root: str | Path,
    cache_root: str | Path,
    result_root: str | Path,
    source_commit: str,
    source_tree: str,
    enforce_target_gpu: bool = True,
) -> dict[str, Any]:
    if worker_count != EXPECTED_GPU_COUNT or worker_index not in range(EXPECTED_GPU_COUNT):
        raise ValueError("H2 dynamic queue requires exactly four workers indexed 0..3")
    plan = validate_plan(rows, seal, logical_stage=logical_stage)
    result = Path(result_root).resolve()
    queue_root = result / "dynamic_queue" / logical_stage.casefold()
    worker_basis = {
        "logical_stage": logical_stage,
        "worker_index": int(worker_index),
        "worker_count": int(worker_count),
        "plan_seal_identity": seal["identity"],
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "launch_nonce": os.environ.get("AMINER_H2_LAUNCH_NONCE", "LOCAL_TEST_RUN"),
        "started_at": utc_now(),
        **_gpu_runtime(enforce_target_gpu=enforce_target_gpu),
    }
    worker_receipt_root = queue_root / "workers" / str(worker_basis["launch_nonce"])
    write_json_once(
        worker_receipt_root / f"worker_{worker_index}_STARTED.json",
        envelope("metachorus.v53.m7f-h2.worker-started.v1", "STARTED", worker_basis),
    )
    completed = 0
    claimed = 0
    failed = 0
    for row in sorted(plan, key=lambda value: (int(value.get("queue_ordinal", 0)), str(value["physical_run_id"]))):
        if (queue_root / "AMINER_H2_COMMON_ROOT_CAUSE_PAUSE.json").exists():
            break
        if completed_marker(result, row, logical_stage) is not None:
            continue
        execution_row = _execution_row_for_retry(
            queue_root,
            row,
            logical_stage=logical_stage,
            plan_seal_identity=str(seal["identity"]),
        )
        if execution_row is None:
            continue
        attempt = int(execution_row.get("attempt_index", 0))
        if attempt not in range(MAX_ATTEMPTS_PER_IDENTITY):
            raise ValueError("H2 retry attempt exceeds the frozen maximum")
        preferred = row.get("preferred_gpu_slot", "")
        if preferred not in (None, "") and int(preferred) != worker_index:
            continue
        claim_path = queue_root / "claims" / f"{row['physical_run_id']}__attempt_{attempt:02d}.claim"
        claim = {
            "physical_run_id": row["physical_run_id"],
            "logical_stage": logical_stage,
            "attempt_index": attempt,
            "worker_index": worker_index,
            "plan_seal_identity": seal["identity"],
            "claimed_at": utc_now(),
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
        }
        if not _atomic_claim(claim_path, claim):
            continue
        claimed += 1
        try:
            run_h2_row(
                plan_row=execution_row,
                logical_stage=logical_stage,
                worker_index=worker_index,
                repo_root=repo_root,
                cache_root=cache_root,
                result_root=result,
                source_commit=source_commit,
                source_tree=source_tree,
                enforce_target_gpu=enforce_target_gpu,
            )
            completed += 1
        except BaseException as exc:
            failed += 1
            fingerprint = _failure_fingerprint(exc)
            basis = {
                "logical_stage": logical_stage,
                "physical_run_id": row["physical_run_id"],
                "attempt_index": attempt,
                "worker_index": worker_index,
                "plan_seal_identity": seal["identity"],
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "error_fingerprint": fingerprint,
                "failed_at": utc_now(),
                "retry_allowed": attempt == 0,
                "scientific_metric_triggered_retry": False,
            }
            write_json_once(
                queue_root / "failures" / f"{row['physical_run_id']}__attempt_{attempt:02d}.json",
                envelope("metachorus.v53.m7f-h2.run-failure.v1", "FAILED_PRESERVED", basis),
            )
            _common_root_pause(queue_root)
            break
    final_basis = {
        **worker_basis,
        "ended_at": utc_now(),
        "claimed_run_count": claimed,
        "completed_run_count": completed,
        "failed_run_count": failed,
        "queue_paused": (queue_root / "AMINER_H2_COMMON_ROOT_CAUSE_PAUSE.json").is_file(),
    }
    receipt = envelope("metachorus.v53.m7f-h2.worker-complete.v1", "PASS" if failed == 0 else "FAILED_PRESERVED", final_basis)
    write_json_once(worker_receipt_root / f"worker_{worker_index}_COMPLETE.json", receipt)
    return receipt


__all__ = [
    "ENGINE_STAGE",
    "h2_marker_name",
    "attempt_root",
    "completed_marker",
    "run_h2_row",
    "run_dynamic_worker",
]
